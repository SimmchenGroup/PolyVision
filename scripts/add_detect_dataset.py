from __future__ import annotations

from pathlib import Path
import random
import shutil
import yaml
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_WORKERS = 8


# -----------------------
# FAST PRECHECK
# -----------------------
def band_count_fast(p: Path) -> int | None:
    try:
        with Image.open(p) as im:
            return len(im.getbands())
    except Exception:
        return None


def dtype_fast(p: Path) -> np.dtype | None:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.dtype


# -----------------------
# NORMALIZATION
# -----------------------
def normalize_to_3ch_inplace(p: Path) -> tuple[Path, str]:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return p, "unreadable"

    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        out = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = img
    elif img.ndim == 3 and img.shape[2] == 4:
        out = img[:, :, :3]
    else:
        return p, f"unsupported(shape={img.shape})"

    if out.dtype != np.uint8:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ok = cv2.imwrite(str(p), out)
    return p, "fixed" if ok else "write_failed"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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

# -----------------------
# MAIN BUILDER (APPEND MODE)
# -----------------------
def build_or_append_yolo_dataset(
    parent_root: str | Path,
    out_root: str | Path,
    class_to_id: dict[str, int],
    val_split: float = 0.2,
    seed: int = 42,
    strict: bool = False,
    jpg_quality: int = 95,
):
    random.seed(seed)

    parent_root = Path(parent_root)
    out_root = Path(out_root)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # LOAD EXISTING DATA (if exists)
    # -----------------------
    train_file = out_root / "train.txt"
    val_file = out_root / "val.txt"

    existing_images = set()

    if train_file.exists():
        existing_images.update([Path(p).name for p in train_file.read_text().splitlines()])

    if val_file.exists():
        existing_images.update([Path(p).name for p in val_file.read_text().splitlines()])

    # -----------------------
    # CLASS VALIDATION
    # -----------------------
    id_to_class = {i: c for c, i in class_to_id.items()}
    names = [id_to_class[i] for i in range(len(id_to_class))]

    new_images: list[str] = []

    # -----------------------
    # PROCESS EACH CLASS
    # -----------------------
    for class_name, class_id in class_to_id.items():
        cls_path = parent_root / class_name
        whole_images = cls_path / "whole_images"

        if not whole_images.exists():
            if strict:
                raise FileNotFoundError(whole_images)
            continue

        for img_path in whole_images.iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            stem = img_path.stem
            label_in_path = cls_path / stem / f"{stem}.txt"

            if not label_in_path.exists():
                if strict:
                    raise FileNotFoundError(label_in_path)
                continue

            # -----------------------
            # FAST CHECK (normalize source only if needed)
            # -----------------------
            bands = band_count_fast(img_path)
            dtype = dtype_fast(img_path)
            if bands != 3 or dtype != np.uint8:
                normalize_to_3ch_inplace(img_path)

            # -----------------------
            # UNIQUE NAME (avoid overwrite) + force JPG output
            # -----------------------
            new_name = f"{class_name}_{stem}.jpg"
            if new_name in existing_images:
                continue  # already added

            out_img_path = images_dir / new_name
            ok = convert_to_jpg(img_path, out_img_path, quality=jpg_quality)
            if not ok:
                if strict:
                    raise RuntimeError(f"Could not read/convert image: {img_path}")
                continue

            # -----------------------
            # REMAP LABEL (OBB lines unchanged except class id)
            # -----------------------
            try:
                in_lines = label_in_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception as e:
                if strict:
                    raise
                continue

            out_lines: list[str] = []
            for line in in_lines:
                parts = line.strip().split()
                if not parts:
                    continue
                parts[0] = str(class_id)
                out_lines.append(" ".join(parts))

            (labels_dir / f"{Path(new_name).stem}.txt").write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""),
                encoding="utf-8",
            )

            new_images.append(new_name)

    # -----------------------
    # APPEND SPLIT
    # -----------------------
    random.shuffle(new_images)
    val_count = int(len(new_images) * val_split)

    val_new = new_images[:val_count]
    train_new = new_images[val_count:]

    with open(train_file, "a", encoding="utf-8") as f:
        for n in train_new:
            f.write(str((images_dir / n).resolve()) + "\n")

    with open(val_file, "a", encoding="utf-8") as f:
        for n in val_new:
            f.write(str((images_dir / n).resolve()) + "\n")

    # -----------------------
    # YAML (create if missing)
    # -----------------------
    yaml_path = out_root / "data.yaml"
    if not yaml_path.exists():
        data_yaml = {
            "path": str(out_root.resolve()),
            "train": "train.txt",
            "val": "val.txt",
            "nc": len(names),
            "names": names,
        }
        yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print(f"✅ Added {len(new_images)} new images")
if __name__ == "__main__":
    build_or_append_yolo_dataset(
        parent_root=r"C:\Users\joshk\OneDrive\Desktop\raw\complete",
        out_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\detectv6",
        class_to_id={
            "nylon": 0,
            "pe": 1,
            "pet": 2,
            "pla": 3,
            "pmma": 4,
            "pp": 5,
            "ps": 6,
            "pu": 7,
            "pvc": 8,
        },
        val_split=0.2,
        seed=42,
        strict=False,
    )