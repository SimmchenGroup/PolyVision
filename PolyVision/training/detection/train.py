from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ultralytics import YOLO
from PIL import Image

import cv2
import numpy as np


# -----------------------
# CONFIG
# -----------------------
DATA_YAML = Path(r"C:\Users\joshk\OneDrive\Desktop\multiclass\detectv7\data.yaml")
IMAGES_DIR = DATA_YAML.parent / "images"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_WORKERS = 8  # tune: 4-8 typical; if on slow disk/network, lower may be faster


# -----------------------
# FAST PRECHECK (metadata-only)
# -----------------------
def band_count_fast(p: Path) -> int | None:
    """Return number of bands (channels) using Pillow metadata; None if unreadable."""
    try:
        with Image.open(p) as im:
            return len(im.getbands())  # L->1, RGB->3, RGBA->4, etc.
    except Exception:
        return None

def dtype_fast(p: Path) -> np.dtype | None:
    """Return dtype via cv2 load; None if unreadable."""
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.dtype

# Scan recursively (train/val/test etc.)
files = [p for p in IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

needs_fix: list[Path] = []
unreadable: list[Path] = []

needs_fix_bands = 0
needs_fix_dtype = 0

for p in files:
    bands = band_count_fast(p)
    if bands is None:
        unreadable.append(p)
        continue

    dt = dtype_fast(p)
    if dt is None:
        unreadable.append(p)
        continue

    band_bad = (bands != 3)
    dtype_bad = (dt != np.uint8)

    if band_bad or dtype_bad:
        needs_fix.append(p)
        needs_fix_bands += int(band_bad)
        needs_fix_dtype += int(dtype_bad)

print(f"Images scanned: {len(files)}")
print(f"Needs fix (bands != 3): {needs_fix_bands}")
print(f"Needs fix (dtype != uint8): {needs_fix_dtype}")
print(f"Total needs_fix (union): {len(needs_fix)}")
print(f"Unreadable (Pillow/cv2): {len(unreadable)}")
if unreadable:
    print("First unreadable example:", unreadable[0])

# -----------------------
# CONVERT ONLY WHAT NEEDS IT (optionally parallel)
# -----------------------
def normalize_to_3ch_inplace(p: Path) -> tuple[Path, str]:
    """
    Ensures image on disk becomes 3-channel uint8 (H,W,3) so Ultralytics can batch.
    Returns (path, status): ok/fixed/unreadable/write_failed/unsupported
    """
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return p, "unreadable"

    # ---- channel normalize to 3ch ----
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        out = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = img
    elif img.ndim == 3 and img.shape[2] == 4:
        out = img[:, :, :3]  # drop alpha
    else:
        return p, f"unsupported(shape={getattr(img, 'shape', None)})"

    # ---- dtype normalize to uint8 ----
    if out.dtype != np.uint8:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ok = cv2.imwrite(str(p), out)
    return p, "fixed" if ok else "write_failed"


counts = {"ok": 0, "fixed": 0, "unreadable": 0, "write_failed": 0, "unsupported": 0}

if needs_fix:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(normalize_to_3ch_inplace, p) for p in needs_fix]
        for fut in as_completed(futs):
            p, status = fut.result()
            if status.startswith("unsupported"):
                counts["unsupported"] += 1
                print("Unsupported:", p, status)
            else:
                counts[status] = counts.get(status, 0) + 1

print("Normalize results:", counts)

bad = []
for p in IMAGES_DIR.iterdir():
    if p.suffix.lower() in IMG_EXTS:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint8:
            bad.append((p, None if img is None else img.dtype, None if img is None else img.shape))
print("Non-uint8:", len(bad))
print("Example:", bad[0] if bad else "none")
# -----------------------
# TRAIN
# -----------------------
# Create a dedicated project folder for this training run
project_dir = Path(r"detection/runs")
run_name = "train_multiclass_detect_v3"  # Give your run a meaningful name

try:
    model = YOLO("yolov8n.pt")  # or yolov8s.pt
    results = model.train(
        data=str(DATA_YAML),
        epochs=15,
        imgsz=800,
        batch=5,
        device="cpu",

        # ===== CHECKPOINT SETTINGS =====
        project=str(project_dir),  # Where to save runs
        name=run_name,  # Name of this specific run
        exist_ok=False,  # False = create new folder each time (train, train2, train3...)
        # True = overwrite existing folder

        save=True,  # Save checkpoints (enabled by default)
        save_period=1,  # Save checkpoint every N epochs (1 = every epoch)

        # Resume from checkpoint if training was interrupted
        # resume=True,                   # Uncomment to resume from best.pt in this run
    )

    print("✓ Training completed!")
    print(f"📁 Results: {project_dir}/{run_name}")
    print(f"🏆 Best model: {project_dir}/{run_name}/weights/best.pt")

except KeyboardInterrupt:
    print("\n⚠ Interrupted - checkpoint saved")
    print(f"💾 Resume with: model = YOLO('{project_dir}/{run_name}/weights/last.pt'); model.train(resume=True)")

except Exception as e:
    print(f"❌ Error: {e}")
    print(f"💾 Last checkpoint: {project_dir}/{run_name}/weights/last.pt")
    raise

# After training completes, checkpoints will be in:
# project_dir / run_name / weights /
#   ├── best.pt       <- Most recent checkpoint (auto-updated each epoch)
#   ├── best.pt       <- Best model based on validation metrics
#   ├── epoch1.pt     <- If save_period=1, saves every epoch
#   ├── epoch2.pt
#   └── ...