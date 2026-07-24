import os
import signal
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from torchvision.datasets import folder as _tv_folder

_HAS_SIGALRM = hasattr(signal, 'SIGALRM')  # True on Linux/WSL2, False on Windows
_LOAD_TIMEOUT_S = 30

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


_CORRUPT_PLACEHOLDER: Image.Image | None = None

def _get_placeholder() -> Image.Image:
    global _CORRUPT_PLACEHOLDER
    if _CORRUPT_PLACEHOLDER is None:
        _CORRUPT_PLACEHOLDER = Image.new("RGB", (4, 4), color=(128, 128, 128))
    return _CORRUPT_PLACEHOLDER


def _load_image_inner(path: str) -> Image.Image:
    if not path.lower().endswith((".tif", ".tiff")):
        return Image.open(path).convert("RGB")
    img = Image.open(path)
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
        img = img.copy()
    if img.mode in ("I", "I;16", "I;16B", "F"):
        arr = np.array(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        img = Image.fromarray(arr)
        del arr
    return img.convert("RGB")


def _tif_safe_loader(path: str) -> Image.Image:
    """
    Memory-safe loader for microscopy TIF and JPG files.
    - Corrupt/unreadable files: returns a tiny gray placeholder.
    - Hung files (9P/NTFS stall over /mnt/c/): SIGALRM kills the open after 30s.
    - Multi-page TIFs: takes first frame only.
    - 16/32-bit images: normalises to 8-bit before RGB conversion.
    """
    if _HAS_SIGALRM:
        def _handler(signum, frame):
            raise TimeoutError(f"image load blocked >{_LOAD_TIMEOUT_S}s")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(_LOAD_TIMEOUT_S)
    try:
        return _load_image_inner(path)
    except TimeoutError as e:
        print(f"[TIMEOUT] {path}: {e}", flush=True)
        return _get_placeholder()
    except Exception as e:
        print(f"[CORRUPT] skipping bad image {path}: {e}", flush=True)
        return _get_placeholder()
    finally:
        if _HAS_SIGALRM:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def scan_dataset_for_corrupt(root_dir: str) -> list[str]:
    """
    Walk root_dir and return a list of image paths that PIL cannot open.
    Run this before training to find all corrupt files in one pass.
    """
    bad = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(_IMAGE_EXTS):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception as e:
                bad.append(fpath)
                print(f"[CORRUPT] {fpath}: {e}", flush=True)
    return bad

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
# Inception preprocess: (x/127.5) - 1  => normalize to [-1, 1]
_INCEPTION_MEAN = [0.5, 0.5, 0.5]
_INCEPTION_STD  = [0.5, 0.5, 0.5]


def _train_val_dirs(config):
    base_dir = config.base_dirs[config.training_type]
    train_base = getattr(config, "train_base_dir_override", None) or base_dir
    train_dir = os.path.join(train_base, "train")
    val_dir = os.path.join(base_dir, "val")
    print(f"[DATA] base_dir={base_dir}")
    print(f"[DATA] train_base_dir_override={getattr(config, 'train_base_dir_override', None)}")
    print(f"[DATA] train_dir={train_dir}")
    print(f"[DATA] val_dir={val_dir}")
    return train_dir, val_dir


def count_images_in_directory(root_dir: str) -> int:
    total = 0
    for _, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(_IMAGE_EXTS):
                total += 1
    return total


def compute_steps_per_epoch(config) -> int:
    train_dir, _ = _train_val_dirs(config)
    n_train = count_images_in_directory(train_dir)
    bs = int(config.batch_size)
    if bs <= 0:
        raise ValueError("config.batch_size must be > 0")
    if n_train <= 0:
        raise ValueError(f"No training images found under: {train_dir}")
    return max(1, n_train // bs)


def get_classes(config):
    if config.training_type == "microplastic":
        return config.microplastic_classes
    elif config.training_type == "whisky":
        return config.whisky_classes
    else:
        raise ValueError("Unknown training type")


def _get_model_io(model_name: str):
    """Returns (image_size, mean, std) for the given backbone."""
    model_name = (model_name or "inception").lower().strip()
    if model_name in {"inception", "inceptionv3"}:
        return (299, 299), _INCEPTION_MEAN, _INCEPTION_STD
    if model_name in {"efficient", "efficientnet", "efficientb0"}:
        return (224, 224), _IMAGENET_MEAN, _IMAGENET_STD
    if model_name in {"efficientb4", "efficientnetb4"}:
        return (380, 380), _IMAGENET_MEAN, _IMAGENET_STD
    if model_name in {"res", "resnet", "resnet50"}:
        return (224, 224), _IMAGENET_MEAN, _IMAGENET_STD
    raise ValueError(f"Unknown model_name={model_name!r}")


def _make_transforms(image_size, mean, std, augment: bool = False):
    h, w = image_size
    if augment:
        # Microplastic orientation is arbitrary, so flips/rotations add free
        # invariance. Mild colour jitter + scale crop mimic the LIF paper's
        # intensity/SNR diversity (5 mass fractions x 4 integration times).
        # Applied to the TRAIN loader only — val/test stay deterministic below.
        return transforms.Compose([
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop((h, w), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _filter_dataset(dataset: datasets.ImageFolder, class_filter: list[str]) -> datasets.ImageFolder:
    """
    Restrict an ImageFolder to only the classes in class_filter.
    Remaps labels to 0..N-1 so the model head size stays consistent.
    Classes missing from the folder are silently ignored.
    """
    valid_classes = [c for c in dataset.classes if c in class_filter]
    if not valid_classes:
        raise ValueError(
            f"None of the requested classes {class_filter} exist in {dataset.root}. "
            f"Found: {dataset.classes}"
        )
    old_to_new = {dataset.class_to_idx[c]: i for i, c in enumerate(valid_classes)}
    valid_old = set(old_to_new)
    dataset.samples = [(p, old_to_new[l]) for p, l in dataset.samples if l in valid_old]
    dataset.targets = [s[1] for s in dataset.samples]
    dataset.classes = valid_classes
    dataset.class_to_idx = {c: i for i, c in enumerate(valid_classes)}
    return dataset


def get_tfdata_datasets(
    config,
    model_name=None,
    balance_mode: str = "none",
    seed: int = 1337,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    mp_context=None,
    augment: bool = True,
):
    """
    Returns (train_loader, val_loader) as torch DataLoaders.

    augment:
      - True (default): train loader gets flips/rotation/jitter/scale-crop;
        val loader stays deterministic.
      - False: train loader matches val (no augmentation).

    balance_mode:
      - 'none': natural class distribution
      - 'balanced': WeightedRandomSampler for uniform class sampling

    If config.microplastic_classes is set, only those class folders are loaded
    (others in the directory are ignored). This lets you test a subset of classes
    without touching the folder structure.
    """
    model_name = (model_name or getattr(config, "model_name", "inception")).lower().strip()
    image_size, mean, std = _get_model_io(model_name)
    train_dir, val_dir = _train_val_dirs(config)
    bs = int(config.batch_size)

    class_filter = None
    if config.training_type == "microplastic" and getattr(config, "microplastic_classes", None):
        class_filter = [c.lower() for c in config.microplastic_classes]

    train_tf = _make_transforms(image_size, mean, std, augment=augment)
    val_tf = _make_transforms(image_size, mean, std, augment=False)
    print(f"[DATA] train augmentation={'ON' if augment else 'OFF'}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf, loader=_tif_safe_loader)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf, loader=_tif_safe_loader)

    if class_filter:
        train_dataset = _filter_dataset(train_dataset, class_filter)
        val_dataset   = _filter_dataset(val_dataset,   class_filter)
        print(f"[DATA] class_filter={class_filter} — loaded {len(train_dataset.classes)} class(es): {train_dataset.classes}")

    print(f"[DEBUG] TRAIN class_names: {train_dataset.classes}")
    print(f"[DEBUG] VAL   class_names: {val_dataset.classes}")
    print(f"[DEBUG] class_names match: {train_dataset.classes == val_dataset.classes}")
    print(f"Found {len(train_dataset)} files belonging to {len(train_dataset.classes)} classes.")

    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        multiprocessing_context=(mp_context if num_workers > 0 else None),
    )
    print(f"[DATA] loader kwargs: {_loader_kwargs}")

    if balance_mode == "balanced":
        targets = torch.tensor([s[1] for s in train_dataset.samples])
        class_counts = torch.bincount(targets)
        weights = 1.0 / class_counts.float()
        sample_weights = weights[targets]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights),
            replacement=True, generator=generator,
        )
        train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, **_loader_kwargs)
    elif balance_mode == "none":
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, **_loader_kwargs)
    else:
        raise ValueError("balance_mode must be 'none' or 'balanced'")

    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, **_loader_kwargs)

    try:
        xb, _ = next(iter(train_loader))
        print(f"[DEBUG] train batch stats: min={xb.min():.4f} max={xb.max():.4f} mean={xb.mean():.4f}")
        xvb, _ = next(iter(val_loader))
        print(f"[DEBUG] val batch stats  : min={xvb.min():.4f} max={xvb.max():.4f} mean={xvb.mean():.4f}")
    except Exception as e:
        print(f"[DEBUG] batch stats check failed: {e}")

    return train_loader, val_loader


def get_generators(config, model_name=None, augment_minority=True):
    """Alias for get_tfdata_datasets for backwards compatibility."""
    return get_tfdata_datasets(config, model_name=model_name, balance_mode="none")


def get_test_generator(config, model_name=None):
    model_name = (model_name or getattr(config, "model_name", "inception")).lower().strip()
    image_size, mean, std = _get_model_io(model_name)
    base_dir = config.base_dirs[config.training_type]
    test_dir = os.path.join(base_dir, "test")
    bs = int(config.batch_size)

    test_tf = _make_transforms(image_size, mean, std, augment=False)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf, loader=_tif_safe_loader)
    return DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=False,
    )
