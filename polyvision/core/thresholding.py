"""
Classical (non-learned) particle segmentation used by the annotation GUI.

`threshold_image` produces a binary particle mask from a greyscale micrograph. The
default 'otsu' path first removes the uneven background with a morphological opening
(rolling-ball style), smooths, then applies Otsu's threshold; small-object removal,
opening/closing, and hole-filling clean the result. This gives the automatic bounding
boxes an annotator can then accept, correct, or replace by hand.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage import morphology
from skimage.morphology import disk, binary_opening
from scipy.ndimage import binary_fill_holes


def threshold_image(
    img: np.ndarray,
    method: str = "otsu",
    object_bright: bool = True,
    block_size: int = 21,
    C: int = 5,
    otsu_offset: int = 0,
    min_obj_size: int = 25,
    fallback_to_adaptive: bool = True,
    manual_thresh: int = 128,
    background_kernel_size: int = 51,
) -> tuple[np.ndarray, int]:
    """Segment particles from a greyscale image; returns (binary mask uint8, threshold used). See module docstring for the method options."""
    if img.dtype != np.uint8:
        img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_8 = img.copy()

    threshold_used = -1

    if method == "otsu":
        if background_kernel_size > 0:
            ksize = background_kernel_size | 1  # ensure odd
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            background = cv2.morphologyEx(img_8, cv2.MORPH_OPEN, kernel)
            img_proc = cv2.subtract(background, img_8) if not object_bright else cv2.subtract(img_8, background)
            img_proc = cv2.GaussianBlur(img_proc, (3, 3), 0)
            ret, _ = cv2.threshold(img_proc, 0, 255, cv2.THRESH_OTSU)
            otsu_thresh = int(ret) + otsu_offset
            otsu_thresh = int(np.clip(otsu_thresh, 0, 255))
            _, thr = cv2.threshold(img_proc, otsu_thresh, 255, cv2.THRESH_BINARY)
        else:
            ret, _ = cv2.threshold(img_8, 0, 255, cv2.THRESH_OTSU)
            otsu_thresh = int(ret) + otsu_offset
            otsu_thresh = int(np.clip(otsu_thresh, 0, 255))
            thr_type = cv2.THRESH_BINARY_INV if object_bright else cv2.THRESH_BINARY
            _, thr = cv2.threshold(img_8, otsu_thresh, 255, thr_type)
        threshold_used = otsu_thresh

        thr_bool = thr > 0
        thr_bool = binary_opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, min_size=min_obj_size)

        if fallback_to_adaptive and not thr_bool.any():
            method = "adaptive"

    if method == "manual":
        thr_type = cv2.THRESH_BINARY_INV if object_bright else cv2.THRESH_BINARY
        _, thr = cv2.threshold(img_8, int(manual_thresh), 255, thr_type)
        threshold_used = int(manual_thresh)

        thr_bool = thr > 0
        thr_bool = binary_opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, min_size=min_obj_size)

    if method == "adaptive":
        thr_type = cv2.THRESH_BINARY_INV if object_bright else cv2.THRESH_BINARY
        thr = cv2.adaptiveThreshold(
            img_8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thr_type,
            int(block_size),
            int(C),
        )
        threshold_used = -1

        thr_bool = thr > 0
        thr_bool = binary_opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, min_size=min_obj_size)

    thr_bool = binary_fill_holes(thr_bool)
    return thr_bool.astype(np.uint8), threshold_used


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill fully-enclosed holes in a binary mask."""
    filled = binary_fill_holes(binary.astype(bool))
    return filled.astype(np.uint8)