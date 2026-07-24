"""
Image loading helpers shared across the pipeline.

Micrographs are 16-bit single-channel TIFFs but may arrive as 3- or 4-channel files;
`load_as_gray` collapses any of these to a single greyscale plane at native bit depth,
and `ensure_8bit` min-max normalises to 8-bit only when a downstream consumer (e.g.
YOLO, JPEG export) requires it.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def load_as_gray(path: str | Path) -> np.ndarray:
    """Read an image at native bit depth and return a single greyscale channel."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
    return img


def ensure_8bit(img: np.ndarray) -> np.ndarray:
    """Min-max normalise to uint8 if needed; return a copy so callers can't mutate the source."""
    if img.dtype != np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img.copy()