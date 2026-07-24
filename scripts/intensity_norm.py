"""
Test->train intensity normalisation for inference.

`match_to_reference` linearly rescales an image so its luminance mean and std
match a fixed reference (computed once from the TRAINING data — see
compute_intensity_reference.py). This shifts each evaluation image's brightness
and contrast toward the training distribution, without retraining.

    out = (in - in_mean) / (in_std + eps) * ref_std + ref_mean   (clipped 0..255)

Honest use: the reference must come from training data and be applied uniformly
to every image — never tuned to the evaluation outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

EPS = 1e-6


def match_to_reference(img, ref_mean: float, ref_std: float):
    """Rescale `img` (a PIL RGB image OR a numpy grayscale/colour array) so its
    luminance mean/std match (ref_mean, ref_std). Returns the same type as input."""
    is_pil = isinstance(img, Image.Image)
    arr = np.asarray(img).astype(np.float32)

    lum = arr if arr.ndim == 2 else arr[..., :3].mean(axis=2)
    in_mean, in_std = float(lum.mean()), float(lum.std())

    a = ref_std / (in_std + EPS)
    b = ref_mean - a * in_mean
    out = np.clip(a * arr + b, 0, 255).astype(np.uint8)

    if is_pil:
        return Image.fromarray(out, mode=("L" if out.ndim == 2 else "RGB"))
    return out


def load_reference(path: str | Path) -> dict:
    """Load the per-model intensity reference JSON
    ({"local": {"mean":..,"std":..}, "global": {...}, "detection": {...}})."""
    return json.loads(Path(path).read_text())


def get_ref(reference: dict, key: str) -> tuple[float, float] | None:
    """Return (mean, std) for a model key, or None if absent."""
    entry = reference.get(key)
    if not entry:
        return None
    return float(entry["mean"]), float(entry["std"])
