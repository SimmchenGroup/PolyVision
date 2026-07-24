"""
Regenerate the train/val split files and `data.yaml` for a YOLO detection dataset.

Given a dataset root containing `images/`, `labels/`, and `data.yaml`, this rewrites
`train.txt` / `val.txt` with a fresh seeded split (optionally recursive, optionally
absolute image paths) and updates `data.yaml` to point at them. A maintenance utility
for the detection dataset — edit the paths in `main()` and run as a script.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(images_dir: Path, recursive: bool = True) -> list[Path]:
    """Yield image paths under `images_dir` (optionally recursing into subfolders)."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    it: Iterable[Path]
    if recursive:
        it = images_dir.rglob("*")
    else:
        it = images_dir.iterdir()

    imgs = [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    return imgs


def write_split_txt(paths: list[Path], out_path: Path, *, make_absolute: bool = True, dataset_root: Path | None = None) -> None:
    """Write one image path per line to a YOLO split-list file."""
    if make_absolute:
        lines = [str(p.resolve()) for p in paths]
    else:
        if dataset_root is None:
            raise ValueError("dataset_root must be provided when make_absolute=False")
        root = dataset_root.resolve()
        lines = [str(p.resolve().relative_to(root)) for p in paths]

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def make_train_val_split(
    images: list[Path],
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Shuffle and split image paths into (train, val) by the configured fraction."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1 (exclusive)")

    imgs = list(images)
    rng = random.Random(seed)
    rng.shuffle(imgs)

    split_idx = int(round(len(imgs) * train_fraction))
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]
    return train_imgs, val_imgs


def update_data_yaml(
    dataset_root: Path,
    data_yaml_path: Path,
    *,
    train_ref: str = "train.txt",
    val_ref: str = "val.txt",
) -> None:
    """
    For your structure:
      dataset_root/
        images/
        labels/
        data.yaml
        train.txt
        val.txt

    We write:
      path: <dataset_root>
      train: train.txt
      val: val.txt
    """
    dataset_root = dataset_root.resolve()
    data_yaml_path = data_yaml_path.resolve()

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data["path"] = str(dataset_root)
    data["train"] = train_ref
    data["val"] = val_ref
    data.pop("test", None)

    with data_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Updated {data_yaml_path}")
    print(f"  path : {data['path']}")
    print(f"  train: {data['train']}")
    print(f"  val  : {data['val']}")


def main() -> None:
    # -----------------------
    # EDIT THESE
    # -----------------------
    """Regenerate train.txt / val.txt and update data.yaml for the detection dataset (edit paths at the top)."""
    dataset_root = Path("data/datasets/detect")  # contains images/, labels/, data.yaml
    train_fraction = 0.8
    seed = 42
    recursive = True
    make_absolute = True
    # -----------------------

    images_dir = dataset_root / "images"
    data_yaml_path = dataset_root / "data.yaml"
    train_txt_path = dataset_root / "train.txt"
    val_txt_path = dataset_root / "val.txt"

    images = iter_images(images_dir, recursive=recursive)
    if not images:
        raise RuntimeError(f"No images found in: {images_dir}")

    train_imgs, val_imgs = make_train_val_split(images, train_fraction=train_fraction, seed=seed)

    write_split_txt(train_imgs, train_txt_path, make_absolute=make_absolute, dataset_root=dataset_root)
    write_split_txt(val_imgs, val_txt_path, make_absolute=make_absolute, dataset_root=dataset_root)

    print(f"Wrote {train_txt_path} ({len(train_imgs)} images)")
    print(f"Wrote {val_txt_path} ({len(val_imgs)} images)")

    update_data_yaml(dataset_root, data_yaml_path, train_ref="train.txt", val_ref="val.txt")


if __name__ == "__main__":
    main()