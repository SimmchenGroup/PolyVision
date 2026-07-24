"""
Thin inference wrapper around the trained YOLOv8 particle detector.

`YoloDetector.detect()` runs the Ultralytics model on a single greyscale micrograph
and returns a list of `YoloDet` records — each an axis-aligned bounding box (in
(row_min, col_min, row_max, col_max) pixel order, matching the rest of the codebase),
a confidence, and a class id. These boxes drive both the crop extraction that feeds
the Local classifier and the detector's own vote in the fusion stage.

YOLO expects an 8-bit 3-channel image, so greyscale input is normalised to 8-bit and
tiled to BGR by `prepare_for_yolo()` before prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO

from polyvision.core.image_io import ensure_8bit


def prepare_for_yolo(gray_img: np.ndarray) -> np.ndarray:
    """Convert a greyscale (or single-channel) image to the 3-channel BGR YOLO expects."""
    if gray_img.ndim == 2:
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    if gray_img.ndim == 3 and gray_img.shape[2] == 1:
        return cv2.cvtColor(gray_img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if gray_img.ndim == 3 and gray_img.shape[2] == 3:
        return gray_img
    raise ValueError(f"Unexpected image shape for YOLO: {gray_img.shape}")


@dataclass(frozen=True)
class YoloDet:
    bbox_rc: tuple[int, int, int, int]
    conf: float
    cls_id: int


class YoloDetector:
    def __init__(self, model_path: str, device: str = "cpu", imgsz: int = 800):
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz

    def detect(self, img_gray: np.ndarray, conf: float = 0.25, classes: list[int] | None = None):
        img_8 = ensure_8bit(img_gray)
        img_input = prepare_for_yolo(img_8)

        results = self.model.predict(
            source=img_input,
            conf=conf,
            verbose=False,
            device=self.device,
            imgsz=self.imgsz,
            classes=classes,
        )[0]

        dets: list[YoloDet] = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            for b, c, k in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, b)
                dets.append(YoloDet(bbox_rc=(y1, x1, y2, x2), conf=float(c), cls_id=int(k)))

        return dets, results