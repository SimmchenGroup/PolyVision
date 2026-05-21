from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np


IMAGE_EXTS = {".png", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".jpg", ".jpeg"}


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize arbitrary numeric image to uint8."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(arr, 0, 255).astype(np.uint8)


def convert_tree_to_jpg(
    src_root: str | Path,
    dst_root: str | Path,
    *,
    quality: int = 92,
    keep_existing_jpg: bool = True,
    delete_original: bool = False,
) -> dict:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    failed = 0

    for src_path in src_root.rglob("*"):
        if not src_path.is_file():
            continue

        ext = src_path.suffix.lower()
        if ext not in IMAGE_EXTS:
            continue

        if keep_existing_jpg and ext in {".jpg", ".jpeg"}:
            skipped += 1
            continue

        rel = src_path.relative_to(src_root)
        dst_path = (dst_root / rel).with_suffix(".jpg")
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src_path) as im:
                # Convert to RGB (JPEG doesn't support alpha)
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    im = Image.alpha_composite(bg, im).convert("RGB")
                elif im.mode != "RGB":
                    # For 16-bit / grayscale etc., go through numpy -> uint8 then RGB/ L
                    arr = np.array(im)
                    arr8 = _to_uint8(arr)

                    # If grayscale, keep grayscale; if multi-channel, ensure RGB
                    if arr8.ndim == 2:
                        im = Image.fromarray(arr8, mode="L").convert("RGB")
                    elif arr8.ndim == 3 and arr8.shape[2] >= 3:
                        im = Image.fromarray(arr8[:, :, :3], mode="RGB")
                    else:
                        # fallback
                        im = Image.fromarray(arr8).convert("RGB")

                im.save(dst_path, format="JPEG", quality=int(quality), optimize=True)

            converted += 1

            if delete_original:
                src_path.unlink(missing_ok=True)

        except Exception as e:
            failed += 1
            print(f"[FAIL] {src_path} -> {dst_path}: {e}")

    summary = {"converted": converted, "skipped": skipped, "failed": failed}
    print(f"[DONE] {summary}")
    return summary


if __name__ == "__main__":
    # Example:
    src = r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv3"
    dst = r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv3_jpg"
    src = input("Source root folder: ").strip().strip('"')
    dst = input("Destination root folder: ").strip().strip('"')

    convert_tree_to_jpg(
        src_root=src,
        dst_root=dst,
        quality=92,
        keep_existing_jpg=True,
        delete_original=False,  # set True only if you’re confident
    )