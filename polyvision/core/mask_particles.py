"""
Replace a micrograph's background with a flat colour while preserving the particle.

Used to standardise crop backgrounds: an Otsu mask separates particle from
background (inverting if Otsu picked the bright background as foreground), the mask is
cleaned morphologically and reduced to its largest connected component, and the
background pixels are painted a constant colour. Returns both the composited image and
the 0/255 particle mask.
"""
import cv2
import numpy as np
from pathlib import Path


def mask_background_flat_color(
    img_bgr: np.ndarray,
    flat_color_bgr=(128, 128, 128),
    invert_if_needed=True,
    keep_largest=True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      composited_bgr: original particle preserved, background replaced by flat color
      mask_uint8:     0 background, 255 particle
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu threshold -> mask candidate
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If particle is dark and background bright, Otsu may pick background as foreground.
    if invert_if_needed:
        if (mask > 0).mean() > 0.5:
            mask = cv2.bitwise_not(mask)

    # Clean mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Keep largest connected component
    if keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = 1 + int(np.argmax(areas))
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    flat = np.full_like(img_bgr, flat_color_bgr, dtype=np.uint8)
    composited = np.where(mask[..., None] == 255, img_bgr, flat)

    return composited, mask


def process_folder(input_dir, output_dir, flat_color=(127, 127, 127)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in extensions:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        out, mask = mask_background_flat_color(img, flat_color_bgr=flat_color)

        out_img_path = output_img_dir / f"{img_path.stem}_flatbg.png"
        out_mask_path = output_mask_dir / f"{img_path.stem}_mask.png"

        cv2.imwrite(str(out_img_path), out)
        cv2.imwrite(str(out_mask_path), mask)

        print(f"Processed: {img_path.name}")


if __name__ == "__main__":
    input_folder = r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv3\train\nylon\ny1_et_1_10X_001_tile_0_4"
    output_folder = r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv3\train\nylon\ny1_et_1_10X_001_tile_0_4\test"

    process_folder(input_folder, output_folder)