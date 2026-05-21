from pathlib import Path
import shutil
import random
import json
import cv2
from PIL import Image

CROP_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
WHOLE_IMAGE_EXTS = CROP_IMAGE_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_image(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != 'uint8':
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imwrite(str(p), img)

def convert_to_jpg(src: Path, dst_jpg: Path, quality: int = 95):
    """
    Read `src` (any supported format), normalize to 8-bit 3-channel BGR, and write as JPEG to `dst_jpg`.
    Does NOT modify the source file.
    """
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != "uint8":
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    ensure_dir(dst_jpg.parent)
    ok = cv2.imwrite(
        str(dst_jpg),
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    return bool(ok)

def append_class(
    local_class_folder: str | Path,
    global_class_folder: str | Path,
    out_local_root: str | Path,
    out_global_root: str | Path,
    split_ratios=(0.7, 0.2, 0.1),
    seed=42,
    include_sidecars=True,
    manifest_path="split_manifest.json",
    add_local: bool = True,
    add_global: bool = True,
):
    local_class_folder = Path(local_class_folder)
    global_class_folder = Path(global_class_folder)
    out_local_root = Path(out_local_root)
    out_global_root = Path(out_global_root)

    if not add_local and not add_global:
        raise ValueError("append_class: at least one of add_local/add_global must be True")

    random.seed(seed)

    manifest = {}

    cls_name = local_class_folder.name

    # List crop groups / whole images
    crop_groups = {p.name: p for p in local_class_folder.iterdir() if p.is_dir()} if add_local else {}
    whole_images = {p.stem: p for p in Path(global_class_folder).iterdir() if p.is_file()} if add_global else {}

    if add_local and add_global:
        paired_keys = sorted(set(crop_groups.keys()) & set(whole_images.keys()))
        if not paired_keys:
            print(f"No paired crops/whole images found for class '{cls_name}'.")
            return
    elif add_local:
        paired_keys = sorted(crop_groups.keys())
        if not paired_keys:
            print(f"No local crop groups found for class '{cls_name}'.")
            return
    else:
        paired_keys = sorted(whole_images.keys())
        if not paired_keys:
            print(f"No global whole images found for class '{cls_name}'.")
            return

    # Shuffle and split
    n = len(paired_keys)
    random.shuffle(paired_keys)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    n_test = n - n_train - n_val

    # Update manifest
    manifest.setdefault(cls_name, {})
    for k in paired_keys[:n_train]:
        manifest[cls_name][k] = "train"
    for k in paired_keys[n_train:n_train+n_val]:
        manifest[cls_name][k] = "val"
    for k in paired_keys[n_train+n_val:]:
        manifest[cls_name][k] = "test"

    # Copy files
    crops_copied = 0
    wholes_copied = 0
    for k in paired_keys:
        split = manifest[cls_name][k]

        # Local crops
        if add_local:
            src_group = crop_groups[k]
            dst_group = out_local_root / split / cls_name / k
            ensure_dir(dst_group)
            for f in src_group.rglob("*"):
                if f.is_file() and (f.suffix.lower() in CROP_IMAGE_EXTS or (include_sidecars and f.suffix.lower() in {".txt", ".json"})):
                    if f.suffix.lower() in CROP_IMAGE_EXTS:
                        dst_img = (dst_group / f.name).with_suffix(".jpg")
                        if convert_to_jpg(f, dst_img):
                            crops_copied += 1
                    else:
                        shutil.copy2(f, dst_group / f.name)
                        crops_copied += 1

        # Global whole image
        if add_global:
            src_img = whole_images[k]
            dst_img = (out_global_root / split / cls_name / src_img.name).with_suffix(".jpg")
            if convert_to_jpg(src_img, dst_img):
                wholes_copied += 1

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump({"manifest": manifest, "seed": seed, "ratios": split_ratios}, f, indent=2)

    print(f"✅ Class '{cls_name}' appended.")
    if add_local:
        print(f"Crops copied: {crops_copied}")
    if add_global:
        print(f"Whole images copied: {wholes_copied}")

def append_parent_folder(
    local_root: str | Path,
    global_root: str | Path,
    out_local_root: str | Path,
    out_global_root: str | Path,
    split_ratios=(0.7, 0.2, 0.1),
    seed=42,
    include_sidecars=True,
    manifest_path="split_manifest.json"
):
    local_root = Path(local_root)
    global_root = Path(global_root)

    # Get all class folders
    classes = [p for p in local_root.iterdir() if p.is_dir()]
    print(f"Found classes: {[c.name for c in classes]}")

    for cls in classes:
        local_class_folder = cls
        global_class_folder = global_root / cls.name
        append_class(
            local_class_folder=local_class_folder,
            global_class_folder=global_class_folder,
            out_local_root=out_local_root,
            out_global_root=out_global_root,
            split_ratios=split_ratios,
            seed=seed,
            include_sidecars=include_sidecars,
            manifest_path=manifest_path
        )

def reset_output_dirs(out_local_root: Path, out_global_root: Path, manifest_path: Path):
    if out_local_root.exists():
        shutil.rmtree(out_local_root)
    if out_global_root.exists():
        shutil.rmtree(out_global_root)
    if manifest_path.exists():
        manifest_path.unlink()

    print("🧹 Cleared existing dataset")

def build_dataset_from_parent(
    local_root: str | Path,
    global_root: str | Path,
    out_local_root: str | Path,
    out_global_root: str | Path,
    split_ratios=(0.7, 0.2, 0.1),
    seed=42,
    include_sidecars=True,
    manifest_path="split_manifest.json"
):
    local_root = Path(local_root)
    global_root = Path(global_root)
    out_local_root = Path(out_local_root)
    out_global_root = Path(out_global_root)
    manifest_path = Path(manifest_path)

    random.seed(seed)

    # 🔴 RESET EVERYTHING
    reset_output_dirs(out_local_root, out_global_root, manifest_path)

    # Get class folders
    classes = [p for p in local_root.iterdir() if p.is_dir()]
    print(f"Building dataset from classes: {[c.name for c in classes]}")

    manifest = {}

    for cls in classes:
        cls_name = cls.name
        local_class_folder = cls
        global_class_folder = global_root / cls_name / "whole_images"

        # List crop groups
        crop_groups = {p.name: p for p in local_class_folder.iterdir() if p.is_dir()}
        whole_images = {p.stem: p for p in global_class_folder.iterdir() if p.is_file()}

        paired_keys = sorted(set(crop_groups.keys()) & set(whole_images.keys()))
        if not paired_keys:
            print(f"⚠️ Skipping {cls_name} (no pairs)")
            continue

        random.shuffle(paired_keys)

        n = len(paired_keys)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        manifest[cls_name] = {}

        for k in paired_keys[:n_train]:
            manifest[cls_name][k] = "train"
        for k in paired_keys[n_train:n_train+n_val]:
            manifest[cls_name][k] = "val"
        for k in paired_keys[n_train+n_val:]:
            manifest[cls_name][k] = "test"

        # Copy
        for k in paired_keys:
            split = manifest[cls_name][k]

            # Crops
            src_group = crop_groups[k]
            dst_group = out_local_root / split / cls_name / k
            ensure_dir(dst_group)

            for f in src_group.rglob("*"):
                if f.is_file() and (f.suffix.lower() in CROP_IMAGE_EXTS or (include_sidecars and f.suffix.lower() in {".txt", ".json"})):
                    if f.suffix.lower() in CROP_IMAGE_EXTS:
                        dst_img = (dst_group / f.name).with_suffix(".jpg")
                        convert_to_jpg(f, dst_img)
                    else:
                        shutil.copy2(f, dst_group / f.name)

            # Whole image
            src_img = whole_images[k]
            dst_img = (out_global_root / split / cls_name / src_img.name).with_suffix(".jpg")
            convert_to_jpg(src_img, dst_img)

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump({"manifest": manifest, "seed": seed, "ratios": split_ratios}, f, indent=2)

    print("✅ Dataset built from scratch")


build_dataset_from_parent(
    local_root=r"C:\Users\joshk\OneDrive\Desktop\raw\complete",
    global_root=r"C:\Users\joshk\OneDrive\Desktop\raw\complete",
    out_local_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv6",
    out_global_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6",
    split_ratios=(0.7, 0.2, 0.1),
    seed=123,
    include_sidecars=True,
    manifest_path=r"C:\Users\joshk\OneDrive\Desktop\multiclass\split_manifest.json"
)

# append_class(
#     local_class_folder=r"C:\Users\joshk\OneDrive\Desktop\multiclass\testsetv4\pet",
#     global_class_folder=r"C:\Users\joshk\OneDrive\Desktop\multiclass\testsetv4\pet",
#     out_local_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv5",
#     out_global_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv5",
#     split_ratios=(0.7, 0.2, 0.1),
#     seed=123,
#     include_sidecars=True,
#     manifest_path=r"C:\Users\joshk\OneDrive\Desktop\multiclass\split_manifest.json",
#     add_local=False,
#     add_global=True,
# )

# append_parent_folder(
#     local_root=r"C:\Users\joshk\OneDrive\Desktop\raw\complete",
#     global_root=r"C:\Users\joshk\OneDrive\Desktop\raw\complete",
#     out_local_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv3",
#     out_global_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv3",
#     split_ratios=(0.7, 0.2, 0.1),
#     seed=123,
#     include_sidecars=True,
#     manifest_path=r"C:\Users\joshk\OneDrive\Desktop\multiclass\split_manifest.json"
# )