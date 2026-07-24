# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 19:26:02 2026

@author: joshk
"""
import json
import sys
import math
import random
from pathlib import Path
import ctypes  # for screen size

import cv2
import numpy as np
from skimage import measure, morphology
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
from shutil import move
from ultralytics import YOLO

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QProgressBar, QSpinBox, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QCheckBox, QSlider, QComboBox,
    QDockWidget, QGraphicsTextItem, QGraphicsRectItem, QGridLayout, QSizePolicy, QListWidget,
    QListWidgetItem, QDialog, QGroupBox, QFrame, QScrollArea, QButtonGroup, QToolButton
)
from PyQt5.QtGui import (
    QPixmap, QImage, QKeySequence, QPen, QBrush, QColor, QPainter, QCursor, QFont,
    QPolygonF, QKeyEvent, QIcon
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF, QSize, QEvent
import pyqtgraph as pg

from polyvision.ml.fusion import ParticleFusionClassifier, fuse_predictions
from polyvision.ml.local_classifier import LocalParticleClassifier
from polyvision.ml.global_classifier import GlobalImageClassifier
from configs.load import load_json, load_microplastic_classes
from polyvision.ml.yolo_detector import YoloDetector
import polyvision.core.db as db
from polyvision.app.gui.dataset_reviewer import DatasetReviewer

def load_microplastic_classes(config: dict) -> list[tuple[int, str]]:
    """
    Returns list of (class_id, name) including any Auto class like (-1, "Auto (model)").
    Falls back to a small default if missing.
    """
    classes_block = config.get("classes", {})
    items = classes_block.get("items", [])
    if items:
        out: list[tuple[int, str]] = []
        for item in items:
            out.append((int(item["id"]), str(item["name"])))
        return out

    # Fallback (keeps app usable if config is old)
    return [
        (-1, "Auto (model)"),
        (0, "Nylon"),
        (1, "PE"),
        (2, "PMMA"),
        (3, "PP"),
        (4, "PS"),
        (5, "PU"),
        (6, "PVC"),
    ]

def bbox_coverage(blob, box) -> float:
    """Fraction of blob's bbox area covered by box. Boxes are (min_r, min_c, max_r, max_c)."""
    inter_h = max(0, min(blob[2], box[2]) - max(blob[0], box[0]))
    inter_w = max(0, min(blob[3], box[3]) - max(blob[1], box[1]))
    blob_area = max(1, (blob[2] - blob[0]) * (blob[3] - blob[1]))
    return (inter_h * inter_w) / blob_area


def triage_sort(img_paths: list) -> list:
    """
    Order images by pre-triage result: clean first, then flagged, then
    untriaged. Alphabetical within each group (input must be pre-sorted).
    """
    rank = {"clean": 0, "flagged": 1, None: 2}
    return sorted(img_paths, key=lambda p: rank.get(db.get_triage_by_path(p)[0], 2))


def yolo_detect(img_gray, conf=0.25, classes=None):
    """
    Returns:
      dets: list[dict] with keys:
        - bbox_rc: (min_r, min_c, max_r, max_c)
        - conf: float
        - cls_id: int
      results: ultralytics Results object
    """
    img_8 = ensure_8bit(img_gray)
    img_input = prepare_for_yolo(img_8)

    results = yolo_model.predict(
        source=img_input,
        conf=conf,
        verbose=False,
        device="cpu",
        imgsz=800,
        classes=classes,  # <-- allow class-filtered runs
    )[0]

    dets = []
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        for b, c, k in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = map(int, b)
            dets.append({
                "bbox_rc": (y1, x1, y2, x2),
                "conf": float(c),
                "cls_id": int(k),
            })

    return dets, results


def prepare_for_yolo(gray_img: np.ndarray) -> np.ndarray:
    """
    Convert grayscale -> 3-channel BGR for YOLO input.
    Output shape: (H, W, 3)
    """
    if gray_img.ndim == 2:
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    if gray_img.ndim == 3 and gray_img.shape[2] == 1:
        return cv2.cvtColor(gray_img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if gray_img.ndim == 3 and gray_img.shape[2] == 3:
        return gray_img
    raise ValueError(f"Unexpected image shape for YOLO: {gray_img.shape}")


def load_as_gray(path: str | Path) -> np.ndarray:
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
    if img.dtype != np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img.copy()


def threshold_image(
        img: np.ndarray,
        method: str = "otsu",
        object_bright: bool = True,
        block_size: int = 21,
        C: int = 5,
        otsu_offset: int = 0,
        min_obj_size: int = 25,
        fallback_to_adaptive: bool = True,
        manual_thresh: int = 128,  # NEW
) -> tuple[np.ndarray, int]:
    # Ensure 8-bit
    if img.dtype != np.uint8:
        img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_8 = img.copy()

    thr = np.zeros_like(img_8)
    threshold_used = -1

    if method == "otsu":
        ret, _ = cv2.threshold(img_8, 0, 255, cv2.THRESH_OTSU)
        otsu_thresh = int(ret) + otsu_offset
        otsu_thresh = np.clip(otsu_thresh, 0, 255)

        # Apply threshold
        thr_type = cv2.THRESH_BINARY_INV if object_bright else cv2.THRESH_BINARY
        _, thr = cv2.threshold(img_8, otsu_thresh, 255, thr_type)
        threshold_used = otsu_thresh

        # Convert to boolean for morphology
        thr_bool = thr > 0

        # Morphology
        thr_bool = morphology.opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, max_size=min_obj_size)

        # Check if any objects remain
        if fallback_to_adaptive and not thr_bool.any():
            # Fallback to adaptive
            print(f"[WARN] Otsu returned empty mask, switching to adaptive threshold")
            method = "adaptive"  # force adaptive fallback

    if method == "manual":
        thr_type = cv2.THRESH_BINARY_INV if object_bright else cv2.THRESH_BINARY
        _, thr = cv2.threshold(img_8, manual_thresh, 255, thr_type)
        threshold_used = manual_thresh

        thr_bool = thr > 0
        thr_bool = morphology.opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, max_size=min_obj_size)

    if method == "adaptive":
        thr_type = cv2.THRESH_BINARY_INV if object_bright else cv2.THRESH_BINARY
        thr = cv2.adaptiveThreshold(
            img_8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thr_type,
            block_size,
            C
        )
        threshold_used = -1

        # Morphology
        thr_bool = thr > 0
        thr_bool = morphology.opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, max_size=min_obj_size)

    # Fill holes
    thr_bool = binary_fill_holes(thr_bool)

    return thr_bool.astype(np.uint8), threshold_used


def fill_holes(binary: np.ndarray) -> np.ndarray:
    filled = binary_fill_holes(binary.astype(bool))
    return filled.astype(np.uint8)


def square_bbox(bbox, img_shape, margin: int = 0):
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r
    w = max_c - min_c
    side = max(h, w) + 2 * margin

    c_r = (min_r + max_r) / 2
    c_c = (min_c + max_c) / 2

    new_min_r = int(round(c_r - side / 2))
    new_max_r = int(round(c_r + side / 2))
    new_min_c = int(round(c_c - side / 2))
    new_max_c = int(round(c_c + side / 2))

    H, W = img_shape[:2]
    new_min_r = max(new_min_r, 0)
    new_min_c = max(new_min_c, 0)
    new_max_r = min(new_max_r, H)
    new_max_c = min(new_max_c, W)

    return new_min_r, new_min_c, new_max_r, new_max_c


def extract_particle_crops(
        img: np.ndarray,
        binary: np.ndarray,
        min_area: int = 50,
        max_area: int | None = None,
        margin: int = 0,
        out_dir: str | Path = ".",
        base_name: str = "image"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = img.shape[:2]
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)

    crops_meta = []
    idx = 1
    for r in regions:
        if r.area < min_area:
            continue
        if max_area is not None and r.area > max_area:
            continue

        min_r, min_c, max_r, max_c = square_bbox(r.bbox, img.shape, margin=margin)
        crop = img[min_r:max_r, min_c:max_c]

        crop_name = f"{base_name}_{idx:04d}.tif"
        crop_path = out_dir / crop_name
        cv2.imwrite(str(crop_path), crop.astype(np.uint16) if crop.dtype == np.uint16 else crop)

        box_w = max_c - min_c
        box_h = max_r - min_r
        x_center = (min_c + box_w / 2) / W
        y_center = (min_r + box_h / 2) / H
        norm_w = box_w / W
        norm_h = box_h / H

        crops_meta.append({
            "crop_path": str(crop_path),
            "bbox": (min_r, min_c, max_r, max_c),
            "area": r.area,
            "yolo": (x_center, y_center, norm_w, norm_h)
        })
        idx += 1

    return crops_meta


def extract_particle_crops_yolo(img, boxes, margin=0, out_dir=".", base_name="image"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = img.shape[:2]
    crops_meta = []
    idx = 1
    for min_r, min_c, max_r, max_c in boxes:
        min_r, min_c, max_r, max_c = square_bbox((min_r, min_c, max_r, max_c), img.shape, margin=margin)
        crop = img[min_r:max_r, min_c:max_c]
        crop_name = f"{base_name}_{idx:04d}.tif"
        crop_path = out_dir / crop_name
        cv2.imwrite(str(crop_path), crop.astype(np.uint16) if crop.dtype == np.uint16 else crop)

        box_w = max_c - min_c
        box_h = max_r - min_r
        x_center = (min_c + box_w / 2) / W
        y_center = (min_r + box_h / 2) / H
        norm_w = box_w / W
        norm_h = box_h / H

        crops_meta.append({
            "crop_path": str(crop_path),
            "bbox": (min_r, min_c, max_r, max_c),
            "area": box_w * box_h,
            "yolo": (x_center, y_center, norm_w, norm_h)
        })
        idx += 1

    return crops_meta


def process_one_image(
        img_path: Path,
        out_root: Path,
        raw_out: Path,
        object_class_id: int,
        min_area: int,
        margin: int,
        use_yolo: bool,
        yolo_detector: YoloDetector | None = None,
):
    """Non-interactive version: uses either YOLO or threshold with fixed params."""
    img = load_as_gray(str(img_path))

    if use_yolo:
        if yolo_detector is None:
            raise ValueError("use_yolo=True but no yolo_detector was provided")

        dets, results = yolo_detect(img, conf=0.25)

        crop_folder = out_root / img_path.stem
        crop_folder.mkdir(exist_ok=True, parents=True)

        # move raw image
        move(img_path, raw_out / img_path.name)

        boxes_rc = [d["bbox_rc"] for d in dets]
        crops_meta = extract_particle_crops_yolo(
            img,
            boxes_rc,
            margin=margin,
            out_dir=crop_folder,
            base_name=img_path.stem
        )
    else:
        # simple fixed threshold (you can plug in your interactive params logic here)
        binary_raw, _ = threshold_image(img, method="otsu", object_bright=True, otsu_offset=0)
        binary_filled = fill_holes(binary_raw)

        crop_folder = out_root / img_path.stem
        crop_folder.mkdir(exist_ok=True, parents=True)

        move(img_path, raw_out / img_path.name)

        crops_meta = extract_particle_crops(
            img,
            binary_filled,
            min_area=min_area,
            max_area=None,
            margin=margin,
            out_dir=crop_folder,
            base_name=img_path.stem
        )

    # YOLO label file
    # object_class_id semantics:
    #   - if object_class_id >= 0: force that class for ALL boxes
    #   - if object_class_id == -1 and use_yolo: use model-predicted class per box
    yolo_txt_path = crop_folder / f"{img_path.stem}.txt"
    with open(yolo_txt_path, "w") as f:
        for i, c in enumerate(crops_meta):
            x, y, w, h = c["yolo"]
            if use_yolo and object_class_id == -1:
                cls_id = dets[i]["cls_id"] if i < len(dets) else 0
            else:
                cls_id = object_class_id
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return {
        "image": str(img_path),
        "folder": str(crop_folder),
        "crops": crops_meta
    }


def draw_yolo_boxes(gray_img, results, conf_thresh=0.25):
    """Draw YOLO boxes (from your original code)"""
    vis = cv2.cvtColor(ensure_8bit(gray_img), cv2.COLOR_GRAY2BGR)
    if results.boxes is None:
        return vis

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), conf in zip(boxes, scores):
        if conf < conf_thresh:
            continue
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(vis, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


# ---------------------------------------------------------------------
# === BBOX HELPERS ===
# ---------------------------------------------------------------------
def bbox_iou_rc(a, b) -> float:
    """IoU for bboxes in (min_r, min_c, max_r, max_c)."""
    a_min_r, a_min_c, a_max_r, a_max_c = a
    b_min_r, b_min_c, b_max_r, b_max_c = b

    inter_min_r = max(a_min_r, b_min_r)
    inter_min_c = max(a_min_c, b_min_c)
    inter_max_r = min(a_max_r, b_max_r)
    inter_max_c = min(a_max_c, b_max_c)

    inter_h = max(0, inter_max_r - inter_min_r)
    inter_w = max(0, inter_max_c - inter_min_c)
    inter = inter_h * inter_w

    a_area = max(0, a_max_r - a_min_r) * max(0, a_max_c - a_min_c)
    b_area = max(0, b_max_r - b_min_r) * max(0, b_max_c - b_min_c)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms_dets_class_agnostic(dets, iou_thresh: float = 0.6):
    """
    Simple class-agnostic NMS. Keeps highest-conf boxes, suppresses overlaps.
    dets: list[dict] with keys bbox_rc, conf, cls_id
    """
    if not dets:
        return []

    dets_sorted = sorted(dets, key=lambda d: float(d["conf"]), reverse=True)
    kept = []

    for d in dets_sorted:
        b = d["bbox_rc"]
        if all(bbox_iou_rc(b, k["bbox_rc"]) < iou_thresh for k in kept):
            kept.append(d)

    return kept


# ---------------------------------------------------------------------
# === WORKER THREAD ===
# ---------------------------------------------------------------------

class ProcessThread(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(int)

    def __init__(self, in_dir, out_root, raw_out, object_class_id, min_area, margin, use_yolo, yolo_detector=None):
        super().__init__()
        self.in_dir = Path(in_dir)
        self.out_root = Path(out_root)
        self.raw_out = Path(raw_out)
        self.object_class_id = object_class_id
        self.min_area = min_area
        self.margin = margin
        self.use_yolo = use_yolo
        self.yolo_detector = yolo_detector

    def run(self):
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

        imgs = []
        for p in self.in_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                imgs.append(p)

        imgs = sorted(imgs)
        n = len(imgs)
        if n == 0:
            self.message.emit("No images found (.tif/.tiff/.png/.jpg/.jpeg/.bmp).")
            self.finished_ok.emit(0)
            return

        self.out_root.mkdir(exist_ok=True, parents=True)
        self.raw_out.mkdir(exist_ok=True, parents=True)

        count = 0
        for i, img_path in enumerate(imgs, start=1):
            try:
                meta = process_one_image(
                    img_path=img_path,
                    out_root=self.out_root,
                    raw_out=self.raw_out,
                    object_class_id=self.object_class_id,
                    min_area=self.min_area,
                    margin=self.margin,
                    use_yolo=self.use_yolo,
                    yolo_detector = self.yolo_detector,
                )
                self.message.emit(f"Processed {Path(meta['image']).name}, crops: {len(meta['crops'])}")
                count += 1
            except Exception as e:
                self.message.emit(f"Error on {img_path.name}: {e}")
            self.progress.emit(int(i * 100 / n))

        self.finished_ok.emit(count)


# ---------------------------------------------------------------------
# === DRAWABLE GRAPHICS VIEW (for manual bbox drawing) ===
# ---------------------------------------------------------------------

class DrawableGraphicsView(QGraphicsView):
    rectCreated = pyqtSignal(tuple)  # emits bbox as (min_r, min_c, max_r, max_c)
    bboxClicked = pyqtSignal(tuple)
    areaDeleted = pyqtSignal(tuple)  # emits a region (min_r, min_c, max_r, max_c) to clear
    zoomChanged = pyqtSignal(float)  # emits current view scale after a zoom/fit change

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draw_mode = False
        self._dragging = False
        self._start_scene = None
        self._rubber_item = None
        self._delete_click_mode = False
        self._draw_right_delete = False  # right-drag while in draw mode == delete-mode left-drag
        # No context menu — right-click is used for the draw-mode delete gesture
        self.setContextMenuPolicy(Qt.PreventContextMenu)

        # --- Zoom / pan (NEW) ---
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._min_scale = 0.2
        self._max_scale = 12.0
        self._fit_scale = 1.0  # scale at which the image exactly fits the view

    # ---- zoom helpers ----
    def current_scale(self) -> float:
        return float(self.transform().m11())

    def is_zoomed(self) -> bool:
        """True when the user has zoomed in beyond the fit scale."""
        return self.current_scale() > self._fit_scale * 1.05

    def _apply_scale(self, factor: float):
        scale = self.current_scale() * factor
        scale = max(self._min_scale, min(self._max_scale, scale))
        cur = self.current_scale()
        if cur <= 0:
            return
        factor = scale / cur
        if abs(factor - 1.0) < 1e-4:
            return
        self.scale(factor, factor)
        self.zoomChanged.emit(self.current_scale())

    def zoom_by(self, factor: float):
        self._apply_scale(factor)

    def fit_view(self):
        """Fit the scene into the view (today's default) and record the fit scale."""
        if self.scene() is None:
            return
        rect = self.scene().sceneRect()
        if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return
        self.resetTransform()
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._fit_scale = self.current_scale()
        self.zoomChanged.emit(self.current_scale())

    def wheelEvent(self, event):
        # Wheel zoom, anchored under the mouse. Works in every tool.
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._apply_scale(factor)
        event.accept()

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = bool(enabled)
        if not self._draw_mode:
            self._clear_rubber()

    def set_delete_click_mode(self, enabled: bool):
        """Enable/disable delete mode (click a box, or drag a region to clear)."""
        self._delete_click_mode = bool(enabled)
        if not self._delete_click_mode:
            self._clear_rubber()

    def _clear_rubber(self):
        self._dragging = False
        self._start_scene = None
        self._draw_right_delete = False
        if self._rubber_item is not None and self.scene() is not None:
            self.scene().removeItem(self._rubber_item)
        self._rubber_item = None

    def _finish_delete(self, event):
        """Shared delete-on-release: tiny drag -> click-delete a box, else region delete."""
        rect = self._rubber_item.rect().normalized() if self._rubber_item is not None else None
        self._clear_rubber()
        if rect is None:
            return
        if rect.width() < 5 and rect.height() < 5:
            # Treat as a single click: delete first bbox under the point
            items = self.scene().items(self.mapToScene(event.pos()))
            for item in items:
                if isinstance(item, QGraphicsRectItem) and item.data(0) is not None:
                    self.bboxClicked.emit(item.data(0))
                    break
        else:
            self.areaDeleted.emit((
                int(round(rect.top())), int(round(rect.left())),
                int(round(rect.bottom())), int(round(rect.right())),
            ))

    def _begin_rubber(self, event, color):
        self._dragging = True
        self._start_scene = self.mapToScene(event.pos())
        if self._rubber_item is None:
            pen = QPen(QColor(color), 2, Qt.DashLine)
            self._rubber_item = QGraphicsRectItem()
            self._rubber_item.setPen(pen)
            self._rubber_item.setBrush(QBrush(Qt.NoBrush))
            self._rubber_item.setZValue(10_000)
            if self.scene() is not None:
                self.scene().addItem(self._rubber_item)
        self._rubber_item.setRect(QRectF(self._start_scene, self._start_scene))

    def mousePressEvent(self, event):
        # Delete mode: start a selection rubber band (release decides click vs region)
        if self._delete_click_mode and event.button() == Qt.LeftButton:
            self._begin_rubber(event, "#ff3333")
            event.accept()
            return

        if self._draw_mode and event.button() == Qt.LeftButton:
            self._begin_rubber(event, "#ffffff")
            event.accept()
            return

        # Draw mode + RIGHT button: act like delete mode's left-drag (delete box/region)
        if self._draw_mode and event.button() == Qt.RightButton:
            self._begin_rubber(event, "#ff3333")
            self._draw_right_delete = True
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._draw_mode or self._delete_click_mode) and self._dragging \
                and self._rubber_item is not None and self._start_scene is not None:
            cur = self.mapToScene(event.pos())
            self._rubber_item.setRect(QRectF(self._start_scene, cur).normalized())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # Delete mode release: tiny drag -> single-click delete, else region delete
        if self._delete_click_mode and self._dragging and event.button() == Qt.LeftButton:
            self._finish_delete(event)
            event.accept()
            return

        # Draw mode + RIGHT button release: same delete behaviour as delete mode
        if self._draw_right_delete and self._dragging and event.button() == Qt.RightButton:
            self._finish_delete(event)
            event.accept()
            return

        if self._draw_mode and self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            if self._rubber_item is None:
                return

            rect = self._rubber_item.rect().normalized()
            # Remove rubber band item after creation
            if self.scene() is not None:
                self.scene().removeItem(self._rubber_item)
            self._rubber_item = None

            # Ignore tiny drags
            if rect.width() < 5 or rect.height() < 5:
                event.accept()
                return

            min_c = int(round(rect.left()))
            min_r = int(round(rect.top()))
            max_c = int(round(rect.right()))
            max_r = int(round(rect.bottom()))

            self.rectCreated.emit((min_r, min_c, max_r, max_c))
            event.accept()
            return

        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------
# === PRODUCTIVITY MODE: REWARD WHEEL ===
# ---------------------------------------------------------------------

class SpinningWheel(QWidget):
    """
    A circular prize wheel. Segments are (label, QColor, weight) tuples drawn
    as pie slices whose arc size is proportional to weight, with a fixed
    pointer at the top (12 o'clock). spin() animates an ease-out rotation that
    lands a weight-proportional random segment under the pointer and emits
    `landed(index)`.
    """
    landed = pyqtSignal(int)

    def __init__(self, segments, parent=None):
        super().__init__(parent)
        self.rotation = 0.0             # current rotation, degrees clockwise from top
        self.setMinimumSize(300, 300)
        self.set_segments(segments)

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick)
        self._start_rot = 0.0
        self._target_rot = 0.0
        self._elapsed = 0
        self._duration = 0
        self._winner = -1
        self._spinning = False

    def set_segments(self, segments):
        """segments: list of (label, QColor, weight)."""
        self.segments = [(str(l), c, max(1e-6, float(w))) for (l, c, w) in segments]
        self._total_w = sum(w for _, _, w in self.segments) or 1.0
        # cumulative (start_theta, span) per segment, in degrees clockwise from top
        self._spans = []
        acc = 0.0
        for (_, _, w) in self.segments:
            span = 360.0 * w / self._total_w
            self._spans.append((acc, span))
            acc += span

    def is_spinning(self) -> bool:
        return self._spinning

    def _pick_weighted(self) -> int:
        r = random.uniform(0.0, self._total_w)
        acc = 0.0
        for i, (_, _, w) in enumerate(self.segments):
            acc += w
            if r <= acc:
                return i
        return len(self.segments) - 1

    def spin(self, winner: int | None = None):
        """Spin the wheel; if winner is None, choose proportional to weight."""
        if self._spinning or not self.segments:
            return
        if winner is None:
            winner = self._pick_weighted()
        self._winner = winner

        start_theta, span = self._spans[winner]
        center = start_theta + span / 2.0
        # rotation placing the winner's centre under the top pointer (theta=0)
        base = (-center) % 360.0
        turns = random.randint(4, 6)
        current_mod = self.rotation % 360.0
        delta = (base - current_mod) % 360.0
        self._start_rot = self.rotation
        self._target_rot = self.rotation + turns * 360.0 + delta

        self._elapsed = 0
        self._duration = random.randint(3200, 4200)  # ms
        self._spinning = True
        self._anim.start(16)

    def _tick(self):
        self._elapsed += 16
        t = min(1.0, self._elapsed / self._duration)
        eased = 1.0 - (1.0 - t) ** 3  # ease-out cubic
        self.rotation = self._start_rot + (self._target_rot - self._start_rot) * eased
        self.update()
        if t >= 1.0:
            self._anim.stop()
            self.rotation = self._target_rot % 360.0
            self._spinning = False
            self.update()
            self.landed.emit(self._winner)

    def paintEvent(self, event):
        if not self.segments:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        side = min(self.width(), self.height()) - 20
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = side / 2.0
        rect = QRectF(cx - r, cy - r, side, side)
        rot = self.rotation

        for i, (label, color, _w) in enumerate(self.segments):
            start_theta, span = self._spans[i]
            # theta clockwise from top -> Qt angle (CCW from 3 o'clock)
            qt_start = 90.0 - (start_theta + span + rot)
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor("#222222"), 2))
            p.drawPie(rect, int(round(qt_start * 16)), int(round(span * 16)))

            # label along the segment centre (skip if the slice is very thin)
            if span >= 6.0:
                theta_c = math.radians(start_theta + span / 2.0 + rot)
                lx = cx + (r * 0.62) * math.sin(theta_c)
                ly = cy - (r * 0.62) * math.cos(theta_c)
                p.save()
                p.translate(lx, ly)
                p.setPen(QPen(QColor("#000000")))
                f = QFont()
                f.setBold(True)
                f.setPointSize(9)
                p.setFont(f)
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(label)
                p.drawText(QPointF(-tw / 2.0, fm.height() / 4.0), label)
                p.restore()

        # hub
        p.setBrush(QBrush(QColor("#1f1f1f")))
        p.setPen(QPen(QColor("#888888"), 2))
        hub = side * 0.08
        p.drawEllipse(QRectF(cx - hub, cy - hub, hub * 2, hub * 2))

        # fixed pointer at top
        p.setBrush(QBrush(QColor("#ff3333")))
        p.setPen(QPen(QColor("#000000"), 1))
        tip_y = cy - r - 2
        poly = QPolygonF([
            QPointF(cx - 14, tip_y),
            QPointF(cx + 14, tip_y),
            QPointF(cx, tip_y + 26),
        ])
        p.drawPolygon(poly)


class RewardDialog(QDialog):
    """
    Productivity reward popup. Auto-spins a break / no-break wheel as soon as it
    opens; the chance of landing on "Break" scales with how much progress each
    reward represents (`reward_percent`) — e.g. 1% per reward makes a break
    unlikely. On a break it auto-spins a second wheel that picks a Rubik's cube
    or daily game and starts an operator-set countdown timer.
    """
    # (label, weight) — cubes are weighted higher than daily games
    CUBES = [("3x3", 2.0), ("7x7", 2.0), ("Gigaminx", 2.0), ("Shard Cube", 2.0)]
    GAMES = [
        ("Enclose Horse", 1.0), ("TimeGuessr", 1.0), ("Flaggle", 1.0),
        ("Shikaku", 1.0), ("Movie Grid", 1.0), ("Movie to Movie", 1.0),
        ("FoodGuessr", 1.0), ("Travle", 1.0),
    ]

    def __init__(self, break_minutes: int, milestone_pct: int,
                 break_prob: float, parent=None):
        super().__init__(parent)
        self.break_minutes = max(1, int(break_minutes))
        self.setWindowTitle("Productivity Reward")
        self.setModal(True)
        self.setMinimumWidth(420)

        # Break likelihood is supplied by the caller (clamped so neither outcome
        # is ever impossible). `got_break` reports the result back to the caller.
        self.break_prob = min(0.95, max(0.05, float(break_prob)))
        self.got_break = None

        layout = QVBoxLayout(self)

        title = QLabel(f"🎉 {milestone_pct}% complete!")
        title.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        title.setFont(f)
        layout.addWidget(title)

        # --- main break / no-break wheel (weighted) ---
        break_color = QColor("#2ecc71")
        nobreak_color = QColor("#e74c3c")
        segments = []
        for i in range(8):  # alternating slices keep it looking like a wheel
            if i % 2 == 0:
                segments.append(("Break", break_color, self.break_prob))
            else:
                segments.append(("No Break", nobreak_color, 1.0 - self.break_prob))
        self.main_wheel = SpinningWheel(segments)
        self.main_wheel.landed.connect(self._on_main_landed)
        layout.addWidget(self.main_wheel)

        self.result_label = QLabel(
            f"Spinning… (break chance {self.break_prob * 100:.0f}%)"
        )
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        # --- break section (hidden until a break is won) ---
        self.break_box = QGroupBox("Break time!")
        self.break_box.setVisible(False)
        break_layout = QVBoxLayout(self.break_box)

        reward_items = self.CUBES + self.GAMES
        reward_segments = [
            (label, QColor.fromHsv(int(360 * i / len(reward_items)), 180, 230), weight)
            for i, (label, weight) in enumerate(reward_items)
        ]
        self.reward_wheel = SpinningWheel(reward_segments)
        self.reward_wheel.landed.connect(self._on_reward_landed)
        break_layout.addWidget(self.reward_wheel)

        self.timer_label = QLabel("")
        self.timer_label.setAlignment(Qt.AlignCenter)
        tf = QFont()
        tf.setBold(True)
        tf.setPointSize(20)
        self.timer_label.setFont(tf)
        break_layout.addWidget(self.timer_label)

        layout.addWidget(self.break_box)

        self.close_btn = QPushButton("Back to work")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

        # countdown state
        self._remaining = 0
        self._countdown = QTimer(self)
        self._countdown.timeout.connect(self._tick_countdown)

        # auto-spin shortly after the dialog is shown
        self._autospun = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._autospun:
            self._autospun = True
            QTimer.singleShot(400, self.main_wheel.spin)

    # --- main wheel ---
    def _on_main_landed(self, index: int):
        label = self.main_wheel.segments[index][0]
        if label == "Break":
            self.got_break = True
            self.result_label.setText("🟢 BREAK! Spinning for your reward…")
            self.break_box.setVisible(True)
            self._start_countdown()
            QTimer.singleShot(400, self.reward_wheel.spin)
        else:
            self.got_break = False
            self.result_label.setText("🔴 No break — back to work!")
            # Auto-return to work shortly after the result is shown.
            QTimer.singleShot(1500, self.accept)

    # --- break / reward wheel ---
    def _start_countdown(self):
        self._remaining = self.break_minutes * 60
        self._update_timer_label()
        self._countdown.start(1000)

    def _tick_countdown(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self._remaining = 0
            self._countdown.stop()
            self.timer_label.setText("⏰ Break over — back to work!")
            return
        self._update_timer_label()

    def _update_timer_label(self):
        m, s = divmod(max(0, self._remaining), 60)
        self.timer_label.setText(f"⏳ {m:02d}:{s:02d}")

    def _on_reward_landed(self, index: int):
        reward = self.reward_wheel.segments[index][0]
        self.break_box.setTitle(f"Break time! Reward: {reward}")


# ---------------------------------------------------------------------
# === THEME TOKENS + STYLESHEET (GUI redesign) ===
# ---------------------------------------------------------------------

# Two themes, keyed like gui_colors.json. Dark is the default.
THEMES = {
    "dark": {
        "window_bg": "#1d1e26",
        "panel_bg": "#262834",
        "panel_bg_alt": "#2e3040",
        "border": "#3a3d4d",
        "text_color": "#e8e9ee",
        "text_muted": "#9a9eb0",
        "highlight": "#61AFEF",
        "button_text": "#0e1116",
        "canvas": "#14151b",
        "good": "#7EE2A8",
        "warn": "#E5C07B",
        "danger": "#e06c75",
    },
    "light": {
        "window_bg": "#eef0f3",
        "panel_bg": "#ffffff",
        "panel_bg_alt": "#f3f4f7",
        "border": "#d8dbe2",
        "text_color": "#23242e",
        "text_muted": "#6b7080",
        "highlight": "#2f7fc4",
        "button_text": "#ffffff",
        "canvas": "#dfe2e7",
        "good": "#1f8a5b",
        "warn": "#a87616",
        "danger": "#c0392b",
    },
}

# Class overlay palette (index by class id, shared across both themes).
CLASS_OVERLAY_PALETTE = [
    "#00c853", "#00b3ff", "#ffcc00", "#ff0066", "#a66bff",
    "#ff6600", "#00bcd4", "#8d99ae", "#ff3333",
]

# Colours currently in use by custom-painted widgets (e.g. ToggleSwitch).
_ACTIVE = dict(THEMES["dark"])


def build_stylesheet(c: dict) -> str:
    """Build the full application Qt Style Sheet from a theme-token dict."""
    return f"""
    QWidget {{
        background-color: {c['window_bg']};
        color: {c['text_color']};
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 12px;
    }}
    QMainWindow, QDialog {{ background-color: {c['window_bg']}; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{
        background-color: {c['window_bg']};
        border: none;
    }}
    QToolTip {{
        background-color: {c['panel_bg']};
        color: {c['text_color']};
        border: 1px solid {c['border']};
    }}

    /* Cards */
    QFrame#card {{
        background-color: {c['panel_bg']};
        border: 1px solid {c['border']};
        border-radius: 10px;
    }}
    QLabel {{ background: transparent; color: {c['text_color']}; }}
    QLabel#cardTitle {{ font-weight: 600; font-size: 12px; }}
    QLabel#muted {{ color: {c['text_muted']}; font-size: 12px; }}
    QLabel#hint {{ color: {c['text_muted']}; font-size: 11px; }}
    QLabel#mono {{ font-family: 'Consolas', ui-monospace, monospace; color: {c['text_muted']}; font-size: 10px; }}
    QLabel#value {{ font-family: 'Consolas', ui-monospace, monospace; color: {c['text_color']}; font-size: 12px; }}
    QLabel#appTitle {{ font-weight: 700; font-size: 15px; }}

    /* Buttons */
    QPushButton {{
        background-color: {c['panel_bg_alt']};
        color: {c['text_color']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 10px;
    }}
    QPushButton:hover {{ border-color: {c['highlight']}; }}
    QPushButton#accent {{
        background-color: {c['highlight']};
        color: {c['button_text']};
        border: 1px solid {c['highlight']};
        font-weight: 600;
    }}
    QPushButton#accent:hover {{ background-color: {c['highlight']}; }}
    QPushButton#link {{
        background-color: {c['panel_bg']};
        color: {c['highlight']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        font-weight: 600;
        padding: 9px 0;
    }}
    QPushButton#danger {{
        background: transparent;
        color: {c['danger']};
        border: 1px solid {c['danger']};
    }}
    QPushButton#iconbtn {{
        background-color: {c['panel_bg_alt']};
        color: {c['text_color']};
        border: 1px solid {c['border']};
        border-radius: 7px;
        padding: 4px 8px;
        min-width: 22px;
    }}
    QPushButton#seg, QPushButton#tool {{
        background: transparent;
        border: none;
        border-radius: 5px;
        color: {c['text_muted']};
        padding: 5px 12px;
    }}
    QPushButton#seg:checked, QPushButton#tool:checked {{
        background-color: {c['panel_bg']};
        color: {c['text_color']};
        font-weight: 600;
    }}
    QWidget#segbar {{ background-color: {c['panel_bg_alt']}; border-radius: 7px; }}

    /* Inputs */
    QComboBox {{
        background-color: {c['panel_bg_alt']};
        color: {c['text_color']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 5px 8px;
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background-color: {c['panel_bg']};
        color: {c['text_color']};
        border: 1px solid {c['border']};
        selection-background-color: {c['panel_bg_alt']};
        outline: none;
    }}
    QLineEdit {{
        background-color: {c['panel_bg_alt']};
        color: {c['text_color']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 4px 6px;
    }}
    QSpinBox {{
        background-color: {c['panel_bg_alt']};
        color: {c['text_color']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 3px 4px;
    }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; border: none; }}

    /* Sliders */
    QSlider::groove:horizontal {{
        height: 4px;
        background-color: {c['panel_bg_alt']};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{ background-color: {c['highlight']}; border-radius: 2px; }}
    QSlider::add-page:horizontal {{ background-color: {c['panel_bg_alt']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        width: 14px; height: 14px;
        margin: -6px 0;
        border-radius: 7px;
        background-color: {c['highlight']};
    }}

    /* Lists */
    QListWidget {{ background: transparent; border: none; outline: none; }}
    QListWidget::item {{ padding: 4px 6px; border-radius: 6px; color: {c['text_color']}; }}
    QListWidget::item:selected {{ background-color: {c['panel_bg_alt']}; color: {c['text_color']}; }}

    /* Group box (productivity card) */
    QGroupBox {{
        background-color: {c['panel_bg']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        margin-top: 4px;
        padding: 10px 12px 12px 12px;
        font-weight: 600;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 3px; }}
    QGroupBox::indicator {{ width: 30px; height: 16px; }}

    /* Progress */
    QProgressBar {{ background-color: {c['panel_bg_alt']}; border: none; text-align: center; color: {c['text_color']}; }}
    QProgressBar::chunk {{ background-color: {c['highlight']}; }}
    QProgressBar#topstrip {{ min-height: 3px; max-height: 3px; border-radius: 0; }}

    /* Top / status bars */
    QWidget#topbar {{ background-color: {c['panel_bg']}; border-bottom: 1px solid {c['border']}; }}
    QWidget#canvasToolbar {{ background-color: {c['panel_bg']}; border-bottom: 1px solid {c['border']}; }}
    QStatusBar {{ background-color: {c['panel_bg']}; color: {c['text_muted']}; border-top: 1px solid {c['border']}; }}
    QStatusBar::item {{ border: none; }}

    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 20px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 8px; }}
    QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 4px; min-width: 20px; }}
    """


class ToggleSwitch(QCheckBox):
    """A QCheckBox skinned as a pill toggle (accent when on, border when off).

    It is still a QCheckBox — same object name, signals and slots — only the
    paint is custom, matching the prototype's 32×18 pill with a white knob.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setText("")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(44, 24)

    def sizeHint(self):
        return QSize(44, 24)

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, event):
        c = _ACTIVE
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(1, 3, -1, -3)
        on = self.isChecked()
        track = QColor(c["highlight"]) if on else QColor(c["border"])
        if not self.isEnabled():
            track.setAlpha(120)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        radius = r.height() / 2.0
        p.drawRoundedRect(r, radius, radius)
        d = r.height() - 6
        x = r.right() - d - 3 if on else r.left() + 3
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(int(x), int(r.top() + 3), int(d), int(d))
        p.end()


# ---------------------------------------------------------------------
# === INTERACTIVE PREVIEW WINDOW ===
# ---------------------------------------------------------------------

class ImagePreview(QMainWindow):
    def __init__(self, img_paths, output_root, raw_out, object_class_id=1,
                 use_yolo=False, min_area=50, parent=None,
                 enable_fusion=False, local_model_path=None, global_model_path=None,
                 fusion_weights=(0.3, 0.5, 0.2),
                 microplastic_classes=None,
                 yolo_detector: YoloDetector | None = None,
                 dataset_root=None):
        super().__init__(parent)
        self.object_class_id = object_class_id
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.setWindowTitle("Interactive Preview")
        self.setWindowState(Qt.WindowMaximized)

        # NEW: lightweight debug toggle (keep True while diagnosing)
        self.fusion_debug = True

        # Image state
        self.img_paths = img_paths
        self.output_root = Path(output_root)
        self.raw_out = Path(raw_out)
        self.current_idx = 0
        self.current_img_gray = None
        self.img_name = ""
        self.selected_bbox_idx = -1
        self.selected_bbox_key = None
        self.deleted_bboxes = set()
        self.bbox_manual_class = {}
        self.user_bboxes = set()
        self.suggested_bboxes = set()  # triage suggestions (subset of user_bboxes)
        self._delete_click_mode = False

        # Settings
        self.use_fusion = False
        self.use_fusion_active = False
        self.use_yolo = use_yolo
        self.yolo_conf = 0.20
        self.min_area_live = min_area
        self.otsu_offset = 0
        self.method = "otsu"
        self.object_bright = True
        self.adaptive_block_size = 21
        self.yolo_detector = yolo_detector
        self.num_classes = len([cid for cid, _name in (microplastic_classes or []) if int(cid) >= 0]) or 7

        # --- Productivity mode state ---
        self.productivity_enabled = False
        self.reward_percent = 10          # reward every N% of images completed
        self.break_minutes = 5            # break-timer duration when a break is won
        self.completed_count = 0          # images saved this session
        self._last_reward_pct = 0.0       # highest completion % already rewarded
        self._break_bonus = 0.0           # added break chance, grows per no-break
        self._break_bonus_step = 0.05   # extra break chance after each no-break

        # Cache
        self.cached_binaries = {}
        self.cached_yolo = {}
        self.cached_scaled_img = {}
        self.cached_yolo_by_class = {}
        # Labeled-region cache: (bbox, area) per connected component. Depends only
        # on the binary mask, so it survives min-area changes (invalidated with
        # cached_binaries). Avoids re-running skimage.label on every update.
        self.cached_regions = {}

        # Per-bbox manual labels (bbox tuple -> class_id)
        self.bbox_manual_class = {}

        self.model_nms_iou = 0.6

        # Color palette for classes (repeatable, distinct-ish)
        self.class_palette = [
            QColor("#00ff66"),  # green
            QColor("#00b3ff"),  # blue
            QColor("#ffcc00"),  # yellow
            QColor("#ff0066"),  # magenta
            QColor("#a66bff"),  # purple
            QColor("#ff6600"),  # orange
            QColor("#00ffff"),  # cyan
            QColor("#ffffff"),  # white
            QColor("#ff3333"),  # red
        ]

        self.bbox_items = []
        self.text_items = []

        # ===================== GUI REDESIGN LAYOUT =====================
        # Five-region shell: top action bar · left tool panel · center canvas
        # · right inspector · status bar. Every existing widget keeps its
        # object name, signals and slots — only its container and skin change.
        self.current_theme = "dark"
        self.current_tool = "select"
        self.binary_view = False
        self.working_folder = None
        self._needs_fit = True

        # ---- local layout helpers ----
        def _card(title, hint=None):
            frame = QFrame()
            frame.setObjectName("card")
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(12, 12, 12, 12)
            lay.setSpacing(11)
            if title is not None:
                header = QHBoxLayout()
                header.setContentsMargins(0, 0, 0, 0)
                t = QLabel(title)
                t.setObjectName("cardTitle")
                header.addWidget(t)
                header.addStretch()
                if hint:
                    h = QLabel(hint)
                    h.setObjectName("mono")
                    header.addWidget(h)
                lay.addLayout(header)
            return frame, lay

        def _slider_row(caption, hint, slider, value_label):
            box = QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(6)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            cap = QLabel(caption + (f"  ({hint})" if hint else ""))
            cap.setObjectName("muted")
            top.addWidget(cap)
            top.addStretch()
            value_label.setObjectName("value")
            top.addWidget(value_label)
            box.addLayout(top)
            box.addWidget(slider)
            return box

        def _switch_row(caption, switch, sub=None):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(caption))
            if sub:
                s = QLabel(sub)
                s.setObjectName("muted")
                row.addWidget(s)
            row.addStretch()
            row.addWidget(switch)
            return row

        # ---- hidden legacy widgets (kept alive so their slots still work) ----
        self._legacy_host = QWidget()
        self._legacy_host.setVisible(False)
        _legacy = QVBoxLayout(self._legacy_host)
        _legacy.setContentsMargins(0, 0, 0, 0)

        self.bbox_dropdown = QComboBox()
        self.bbox_dropdown.currentIndexChanged.connect(self.on_bbox_selected)
        _legacy.addWidget(self.bbox_dropdown)

        self.delete_bbox_btn = QPushButton("Delete Selected BBox")
        self.delete_bbox_btn.clicked.connect(self.on_delete_bbox)
        _legacy.addWidget(self.delete_bbox_btn)

        self.min_area_input = QLineEdit(str(self.min_area_live))
        self.min_area_input.setMaximumWidth(60)
        self.min_area_input.editingFinished.connect(self.on_min_area_input)
        _legacy.addWidget(self.min_area_input)

        self.otsu_input = QLineEdit(str(self.otsu_offset))
        self.otsu_input.setMaximumWidth(60)
        self.otsu_input.editingFinished.connect(self.on_otsu_input)
        _legacy.addWidget(self.otsu_input)

        self.progress_label = QLabel("Image 0 / 0 — File: None — Boxes: 0")
        _legacy.addWidget(self.progress_label)

        self.progress_bar_live = QProgressBar()
        self.progress_bar_live.setRange(0, 100)
        _legacy.addWidget(self.progress_bar_live)

        # Raw + binary preview views (still updated each frame; shown via the
        # canvas "Binary mask" toggle rather than as separate panes).
        self.left_raw_scene = QGraphicsScene()
        self.left_raw_view = QGraphicsView(self.left_raw_scene)
        self.left_raw_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_raw_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _legacy.addWidget(self.left_raw_view)

        self.left_bin_scene = QGraphicsScene()
        self.left_bin_view = QGraphicsView(self.left_bin_scene)
        self.left_bin_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_bin_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _legacy.addWidget(self.left_bin_view)

        # ---------------- LEFT PANEL: Segmentation card ----------------
        self.method_combo = QComboBox()
        self.method_combo.addItems(["otsu", "adaptive", "manual"])
        self.method_combo.setCurrentText(self.method)
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        _legacy.addWidget(self.method_combo)

        seg_tabs = QWidget()
        seg_tabs.setObjectName("segbar")
        seg_tabs_l = QHBoxLayout(seg_tabs)
        seg_tabs_l.setContentsMargins(3, 3, 3, 3)
        seg_tabs_l.setSpacing(3)
        self.method_btn_group = QButtonGroup(self)
        self.method_btn_group.setExclusive(True)
        self._method_buttons = {}
        for val, label in [("otsu", "Otsu"), ("adaptive", "Adaptive"), ("manual", "Manual")]:
            b = QPushButton(label)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, v=val: self.method_combo.setCurrentText(v))
            self.method_btn_group.addButton(b)
            seg_tabs_l.addWidget(b)
            self._method_buttons[val] = b
        self._method_buttons.get(self.method, self._method_buttons["otsu"]).setChecked(True)

        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(1, 1000)
        self.min_area_slider.setValue(self.min_area_live)
        self.min_area_slider.valueChanged.connect(self.on_min_area_changed)
        self.min_area_value_label = QLabel(f"{self.min_area_live} px²")

        self.otsu_slider = QSlider(Qt.Horizontal)
        self.manual_thresh = 128
        self.otsu_slider.setRange(0, 255)
        self.otsu_slider.setValue(self.otsu_offset)
        self.otsu_slider.valueChanged.connect(self.on_otsu_offset_changed)
        self.thresh_value_label = QLabel(str(self.otsu_offset))

        seg_card, seg_lay = _card("Segmentation", "A · + −")
        seg_lay.addWidget(seg_tabs)
        seg_lay.addLayout(_slider_row("Threshold offset", None, self.otsu_slider, self.thresh_value_label))
        seg_lay.addLayout(_slider_row("Min area", ", .", self.min_area_slider, self.min_area_value_label))

        # ---------------- LEFT PANEL: Detection card ----------------
        self.yolo_checkbox = ToggleSwitch("Use YOLO")
        self.yolo_checkbox.setChecked(self.use_yolo)
        self.yolo_checkbox.stateChanged.connect(self.on_yolo_toggled)

        self.yolo_conf_label = QLabel("0.20")
        self.yolo_conf_label.setMinimumWidth(32)

        self.yolo_conf_slider = QSlider(Qt.Horizontal)
        self.yolo_conf_slider.setRange(1, 80)
        self.yolo_conf_slider.setValue(20)
        self.yolo_conf_slider.valueChanged.connect(self.on_yolo_conf_changed)

        self.fusion_checkbox = ToggleSwitch("Use Fusion (YOLO + Local + Global)")
        self.fusion_checkbox.setChecked(False)
        self.fusion_checkbox.setEnabled(False)
        self.fusion_checkbox.stateChanged.connect(self.on_fusion_toggled)

        det_card, det_lay = _card("Detection", "M · [ ]")
        det_lay.addLayout(_switch_row("YOLO detector", self.yolo_checkbox))
        det_lay.addLayout(_slider_row("Confidence", None, self.yolo_conf_slider, self.yolo_conf_label))
        det_lay.addLayout(_switch_row("Fusion", self.fusion_checkbox, "(YOLO + Local + Global)"))

        # ---------------- LEFT PANEL: Object class card ----------------
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.SingleSelection)
        self.class_list.setMaximumHeight(220)

        self.class_definitions = microplastic_classes if microplastic_classes else [
            (-1, "Auto (model)"),
            (0, "Nylon"),
            (1, "PE"),
            (2, "PET"),
            (3, "PLA"),
            (2, "PMMA"),
            (3, "PP"),
            (4, "PS"),
            (5, "PU"),
            (6, "PVC"),
        ]

        for class_id, name in self.class_definitions:
            item = QListWidgetItem(f"{class_id} — {name}" if class_id >= 0 else name)
            item.setData(Qt.UserRole, class_id)
            if class_id >= 0:
                item.setIcon(self._dot_icon(CLASS_OVERLAY_PALETTE[class_id % len(CLASS_OVERLAY_PALETTE)]))
            self.class_list.addItem(item)

        self.class_list.setCurrentRow(0)
        self.object_class_id = -1
        self.class_list.currentItemChanged.connect(self.on_class_changed)

        self.assign_class_btn = QPushButton("Assign to box")
        self.assign_class_btn.clicked.connect(self.on_assign_class_to_selected_bbox)
        self.clear_class_btn = QPushButton("Back to Auto")
        self.clear_class_btn.clicked.connect(self.on_clear_class_for_selected_bbox)

        cls_card, cls_lay = _card("Object class")
        cls_lay.addWidget(self.class_list)
        cls_btn_row = QHBoxLayout()
        cls_btn_row.setContentsMargins(0, 0, 0, 0)
        cls_btn_row.setSpacing(6)
        cls_btn_row.addWidget(self.assign_class_btn)
        cls_btn_row.addWidget(self.clear_class_btn)
        cls_lay.addLayout(cls_btn_row)

        # ---------------- LEFT PANEL: Sample context card ----------------
        self.context_combo = QComboBox()
        self.context_combo.addItems([
            "None (no expectation)",
            "Pure sample (1 class expected)",
            "Contaminated (2+ classes expected)",
            "Environmental (class + background)"
        ])
        self.context_combo.setCurrentIndex(0)
        self.context_combo.currentIndexChanged.connect(self.on_context_changed)

        self.context_info_label = QLabel("")
        self.context_info_label.setObjectName("hint")
        self.context_info_label.setWordWrap(True)

        ctx_card, ctx_lay = _card("Sample context")
        ctx_lay.addWidget(self.context_combo)
        ctx_lay.addWidget(self.context_info_label)

        # ---------------- LEFT PANEL: Productivity card ----------------
        self.productivity_box = QGroupBox("Productivity mode")
        self.productivity_box.setCheckable(True)
        self.productivity_box.setChecked(self.productivity_enabled)
        self.productivity_box.toggled.connect(self.on_productivity_toggled)
        prod_layout = QVBoxLayout(self.productivity_box)
        prod_layout.setSpacing(8)

        steppers = QHBoxLayout()
        steppers.setSpacing(8)

        self.reward_percent_spin = QSpinBox()
        self.reward_percent_spin.setRange(1, 100)
        self.reward_percent_spin.setValue(self.reward_percent)
        self.reward_percent_spin.valueChanged.connect(self.on_reward_percent_changed)
        reward_col = QVBoxLayout()
        reward_col.setSpacing(4)
        reward_lbl = QLabel("Reward every (%)")
        reward_lbl.setObjectName("hint")
        reward_col.addWidget(reward_lbl)
        reward_col.addWidget(self.reward_percent_spin)

        self.break_minutes_spin = QSpinBox()
        self.break_minutes_spin.setRange(1, 120)
        self.break_minutes_spin.setValue(self.break_minutes)
        self.break_minutes_spin.valueChanged.connect(self.on_break_minutes_changed)
        break_col = QVBoxLayout()
        break_col.setSpacing(4)
        break_lbl = QLabel("Break (min)")
        break_lbl.setObjectName("hint")
        break_col.addWidget(break_lbl)
        break_col.addWidget(self.break_minutes_spin)

        steppers.addLayout(reward_col)
        steppers.addLayout(break_col)
        prod_layout.addLayout(steppers)

        # ---------------- LEFT PANEL: Review link ----------------
        self.review_btn = QPushButton("Review Dataset →")
        self.review_btn.setObjectName("link")
        self.review_btn.clicked.connect(self._open_dataset_reviewer)

        left_panel = QWidget()
        lp = QVBoxLayout(left_panel)
        lp.setContentsMargins(12, 12, 12, 12)
        lp.setSpacing(12)
        for w in (seg_card, det_card, cls_card, ctx_card, self.productivity_box, self.review_btn):
            lp.addWidget(w)
        lp.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setFixedWidth(280)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ---------------- CENTER CANVAS ----------------
        self.show_bbox_labels = True

        self.draw_mode_checkbox = QCheckBox("Draw boxes")
        self.draw_mode_checkbox.setChecked(False)
        self.draw_mode_checkbox.stateChanged.connect(self.on_draw_mode_toggled)
        _legacy.addWidget(self.draw_mode_checkbox)

        self.delete_click_mode_checkbox = QCheckBox("Delete on click")
        self.delete_click_mode_checkbox.setChecked(False)
        self.delete_click_mode_checkbox.stateChanged.connect(self.on_delete_click_mode_toggled)
        _legacy.addWidget(self.delete_click_mode_checkbox)

        self.bbox_label_checkbox = QCheckBox("Labels")
        self.bbox_label_checkbox.setChecked(True)
        self.bbox_label_checkbox.stateChanged.connect(self.on_bbox_label_toggle)

        self.binary_mask_checkbox = QCheckBox("Binary mask")
        self.binary_mask_checkbox.setChecked(False)
        self.binary_mask_checkbox.stateChanged.connect(self.on_binary_mask_toggled)

        # Right preview (main image view) — promoted to the center canvas.
        self.right_preview_scene = QGraphicsScene()
        self.right_preview_scene.selectionChanged.connect(self.on_scene_selection_changed)

        self.right_preview_view = DrawableGraphicsView(self.right_preview_scene)
        self.right_preview_view.rectCreated.connect(self.on_user_rect_created)
        self.right_preview_view.bboxClicked.connect(self.on_bbox_clicked_for_delete)
        self.right_preview_view.areaDeleted.connect(self.on_area_deleted)
        self.right_preview_view.zoomChanged.connect(self._on_zoom_changed)
        self.right_preview_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_preview_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_preview_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_preview_view.setFrameShape(QFrame.NoFrame)

        # Floating log chip, bottom-left of the canvas.
        self.log_label = QLabel("Awaiting: (no image loaded)", self.right_preview_view)
        self.log_label.setObjectName("logChip")
        self.log_label.move(10, 10)

        # Canvas toolbar
        canvas_toolbar = QWidget()
        canvas_toolbar.setObjectName("canvasToolbar")
        canvas_toolbar.setFixedHeight(38)
        ctl = QHBoxLayout(canvas_toolbar)
        ctl.setContentsMargins(10, 0, 10, 0)
        ctl.setSpacing(8)

        tool_tabs = QWidget()
        tool_tabs.setObjectName("segbar")
        tt_l = QHBoxLayout(tool_tabs)
        tt_l.setContentsMargins(3, 3, 3, 3)
        tt_l.setSpacing(3)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_select_btn = QPushButton("Select")
        self.tool_select_btn.setToolTip("Click boxes")
        self.tool_draw_btn = QPushButton("Draw")
        self.tool_draw_btn.setToolTip("D toggles draw/delete")
        self.tool_delete_btn = QPushButton("Delete")
        self.tool_delete_btn.setToolTip("D toggles draw/delete")
        for b, tool in [(self.tool_select_btn, "select"), (self.tool_draw_btn, "draw"), (self.tool_delete_btn, "delete")]:
            b.setObjectName("tool")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, t=tool: self.set_tool(t))
            self.tool_group.addButton(b)
            tt_l.addWidget(b)
        self.tool_select_btn.setChecked(True)

        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setObjectName("iconbtn")
        self.zoom_out_btn.setFixedSize(26, 26)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("mono")
        self.zoom_label.setMinimumWidth(44)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setObjectName("iconbtn")
        self.zoom_in_btn.setFixedSize(26, 26)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("iconbtn")
        self.fit_btn.setFixedHeight(26)
        self.fit_btn.clicked.connect(self.zoom_fit)

        ctl.addWidget(tool_tabs)
        ctl.addWidget(self.bbox_label_checkbox)
        ctl.addWidget(self.binary_mask_checkbox)
        ctl.addStretch()
        ctl.addWidget(self.zoom_out_btn)
        ctl.addWidget(self.zoom_label)
        ctl.addWidget(self.zoom_in_btn)
        ctl.addWidget(self.fit_btn)

        center = QWidget()
        center.setObjectName("centerCanvas")
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(canvas_toolbar)
        cl.addWidget(self.right_preview_view, 1)

        # ---------------- RIGHT INSPECTOR ----------------
        # Detections card (custom header: count + Clear suggested)
        det_info_card = QFrame()
        det_info_card.setObjectName("card")
        dic = QVBoxLayout(det_info_card)
        dic.setContentsMargins(12, 12, 12, 12)
        dic.setSpacing(8)
        dic_hdr = QHBoxLayout()
        dic_hdr.setContentsMargins(0, 0, 0, 0)
        self.detections_title = QLabel("Detections (0)")
        self.detections_title.setObjectName("cardTitle")
        self.clear_suggested_btn = QPushButton("Clear suggested (X)")
        self.clear_suggested_btn.setCursor(Qt.PointingHandCursor)
        self.clear_suggested_btn.clicked.connect(
            lambda: self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_X, Qt.NoModifier))
        )
        self.clear_suggested_btn.setVisible(False)
        dic_hdr.addWidget(self.detections_title)
        dic_hdr.addStretch()
        dic_hdr.addWidget(self.clear_suggested_btn)
        dic.addLayout(dic_hdr)

        self.bbox_info_list = QListWidget()
        self.bbox_info_list.setSelectionMode(QListWidget.SingleSelection)
        self.bbox_info_list.currentRowChanged.connect(self.on_bbox_info_row_changed)
        self.bbox_info_list.setMaximumHeight(230)
        dic.addWidget(self.bbox_info_list)

        # Particle preview card (square crop)
        self.placeholder_top = QLabel("Select a box to preview")
        self.placeholder_top.setAlignment(Qt.AlignCenter)
        self.placeholder_top.setMinimumHeight(240)
        self.placeholder_top.setScaledContents(False)
        self.crop_meta_label = QLabel("")
        self.crop_meta_label.setObjectName("mono")

        crop_card, crop_lay = _card("Particle preview")
        crop_lay.addWidget(self.placeholder_top)
        crop_lay.addWidget(self.crop_meta_label)

        # Particle size distribution card (histogram)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(THEMES[self.current_theme]["panel_bg"])
        self.plot_widget.setLabel("bottom", "Area (px²)")
        self.plot_widget.setLabel("left", "Count")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMaximumHeight(150)
        self.area_curve = self.plot_widget.plot(stepMode=True, fillLevel=0, brush=(97, 175, 239, 120))

        hist_card, hist_lay = _card("Particle size distribution")
        hist_lay.addWidget(self.plot_widget)

        right_panel = QWidget()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(12, 12, 12, 12)
        rp.setSpacing(12)
        for w in (det_info_card, crop_card, hist_card):
            rp.addWidget(w)
        rp.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_panel)
        right_scroll.setFixedWidth(300)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ---------------- TOP ACTION BAR ----------------
        topbar = QWidget()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(50)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(14, 0, 14, 0)
        tb.setSpacing(12)

        app_title = QLabel("PolyVision")
        app_title.setObjectName("appTitle")
        app_subtitle = QLabel("Interactive Preview")
        app_subtitle.setObjectName("muted")
        tb.addWidget(app_title)
        tb.addWidget(app_subtitle)

        self.folder_btn = QPushButton("Open folder  ▾")
        self.folder_btn.setObjectName("iconbtn")
        self.folder_btn.setCursor(Qt.PointingHandCursor)
        self.folder_btn.clicked.connect(self.on_pick_folder)
        tb.addWidget(self.folder_btn)

        tb.addStretch()

        self.prev_btn = QPushButton("‹")
        self.prev_btn.setObjectName("iconbtn")
        self.prev_btn.setToolTip("Left / P")
        self.prev_btn.setFixedSize(30, 30)
        self.prev_btn.clicked.connect(self.prev_image)

        nav_center = QWidget()
        nav_center.setMinimumWidth(170)
        nc = QVBoxLayout(nav_center)
        nc.setContentsMargins(0, 0, 0, 0)
        nc.setSpacing(0)
        self.filename_label = QLabel("—")
        self.filename_label.setObjectName("value")
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.imgpos_label = QLabel("Image 0 of 0")
        self.imgpos_label.setObjectName("mono")
        self.imgpos_label.setAlignment(Qt.AlignCenter)
        nc.addWidget(self.filename_label)
        nc.addWidget(self.imgpos_label)

        self.next_btn = QPushButton("›")
        self.next_btn.setObjectName("iconbtn")
        self.next_btn.setToolTip("Right / N")
        self.next_btn.setFixedSize(30, 30)
        self.next_btn.clicked.connect(self.next_image)

        self.triage_badge = QLabel("UNTRIAGED")
        self.triage_badge.setObjectName("badge")

        tb.addWidget(self.prev_btn)
        tb.addWidget(nav_center)
        tb.addWidget(self.next_btn)
        tb.addWidget(self.triage_badge)

        tb.addSpacing(6)
        self.theme_btn = QPushButton("◐")
        self.theme_btn.setObjectName("iconbtn")
        self.theme_btn.setToolTip("Toggle theme")
        self.theme_btn.setFixedSize(30, 30)
        self.theme_btn.clicked.connect(self.toggle_theme)

        self.skip_btn = QPushButton("Skip  Esc")
        self.skip_btn.setObjectName("iconbtn")
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.clicked.connect(self._skip)

        self.save_btn = QPushButton("Save & Next  ↵")
        self.save_btn.setObjectName("accent")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self._save_and_next)

        tb.addWidget(self.theme_btn)
        tb.addWidget(self.skip_btn)
        tb.addWidget(self.save_btn)

        # 3px progress strip directly under the bar
        self.top_progress = QProgressBar()
        self.top_progress.setObjectName("topstrip")
        self.top_progress.setRange(0, 100)
        self.top_progress.setValue(0)
        self.top_progress.setTextVisible(False)
        self.top_progress.setFixedHeight(3)

        # ---------------- ASSEMBLE ----------------
        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)
        main_row.addWidget(left_scroll)
        main_row.addWidget(center, 1)
        main_row.addWidget(right_scroll)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(topbar)
        root.addWidget(self.top_progress)
        root.addLayout(main_row, 1)
        root.addWidget(self._legacy_host)
        self.setCentralWidget(central)

        # Status bar: live message left, keybind cheat-sheet right (monospace)
        self.keybind_label = QLabel(
            "↵ save · Esc skip · ←→ nav · M yolo · A method · +− thresh · "
            ", . area · [ ] conf · D draw/delete · X suggested · Del box · F fit"
        )
        self.keybind_label.setObjectName("mono")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.keybind_label)
        self.statusBar().showMessage("Ready.")

        # Apply the (dark) theme now that every widget exists.
        self.apply_theme()

        # ------------------- set canvas background and remove frame -------------------
        canvas_color = QColor(THEMES[self.current_theme]["canvas"])

        for view in [self.left_raw_view, self.left_bin_view, self.right_preview_view]:
            view.setBackgroundBrush(canvas_color)
            view.setStyleSheet("border: 0px;")  # remove white frame
        # --------------------- PIXMAP ITEMS ---------------------
        self.left_raw_pix = QGraphicsPixmapItem()
        self.left_raw_scene.addItem(self.left_raw_pix)
        self.left_bin_pix = QGraphicsPixmapItem()
        self.left_bin_scene.addItem(self.left_bin_pix)
        self.right_pix = QGraphicsPixmapItem()
        self.right_preview_scene.addItem(self.right_pix)

        # Bounding boxes and text overlays
        self.bbox_items = []
        self.text_items = []

        if self.img_paths:
            self.load_current_image()

        self.use_fusion = False
        if enable_fusion and local_model_path and global_model_path:
            try:
                self.local_clf = LocalParticleClassifier(
                    model_path=local_model_path,  # ← Now passed as parameter
                    num_classes=self.num_classes,
                    device="cpu"
                )
                self.global_clf = GlobalImageClassifier(
                    model_path=global_model_path,  # ← Now passed as parameter
                    num_classes=self.num_classes,
                    device="cpu"
                )
                self.fusion_clf = ParticleFusionClassifier(
                    weights=fusion_weights,  # ← Now passed as parameter
                    num_classes=self.num_classes,
                    use_meta_model=False
                )
                self.use_fusion = True

                self.fusion_checkbox.setEnabled(True)
                self.use_fusion_active = True
                self.fusion_checkbox.setChecked(True)

                print("✓ Fusion classifiers loaded successfully")
                print(f"   Local model: {local_model_path}")
                print(f"   Global model: {global_model_path}")
                print(f"   Fusion weights: {fusion_weights}")
            except Exception as e:
                print(f"⚠ Fusion disabled: {e}")
                self.use_fusion = False
                self.use_fusion_active = False
                self.fusion_checkbox.setEnabled(False)
                self.fusion_checkbox.setChecked(False)
        elif enable_fusion:
            print("⚠ Fusion disabled: Missing model paths")
            self.use_fusion = False
            self.use_fusion_active = False
            self.fusion_checkbox.setEnabled(False)
            self.fusion_checkbox.setChecked(False)

        # Force an initial update after the event loop runs
        QTimer.singleShot(50, self.update_display)
        QTimer.singleShot(60, self._position_log_chip)

    # ==================================================================
    # === GUI REDESIGN: theme, tools, zoom/pan, folder picker ===
    # ==================================================================
    def _theme(self) -> dict:
        return THEMES[self.current_theme]

    def _dot_icon(self, color_hex: str, size: int = 12) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color_hex))
        p.drawEllipse(1, 1, size - 2, size - 2)
        p.end()
        return QIcon(pm)

    def apply_theme(self):
        """Rebuild the whole application stylesheet from the active theme dict."""
        global _ACTIVE
        c = THEMES[self.current_theme]
        _ACTIVE = c

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(c))

        for v in (self.left_raw_view, self.left_bin_view, self.right_preview_view):
            v.setBackgroundBrush(QColor(c["canvas"]))
        self.plot_widget.setBackground(c["panel_bg"])
        self.theme_btn.setText("◐" if self.current_theme == "dark" else "◑")

        # Dynamic styling for widgets QSS object-name rules can't fully reach.
        self.log_label.setStyleSheet(
            f"background-color:{c['panel_bg']}; border:1px solid {c['border']};"
            f"border-radius:6px; padding:4px 9px; color:{c['text_muted']}; font-size:11px;"
        )
        self.log_label.adjustSize()
        self.placeholder_top.setStyleSheet(
            f"background-color:{c['canvas']}; border:1px dashed {c['border']};"
            f"border-radius:8px; color:{c['text_muted']};"
        )
        self.clear_suggested_btn.setStyleSheet(
            f"background:transparent; border:1px solid {c['warn']}; color:{c['warn']};"
            f"border-radius:5px; padding:2px 7px; font-size:10px;"
        )
        self._refresh_triage_badge()
        self._position_log_chip()
        self.update()

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()
        # Overlay colours / list rows depend on theme; redraw them.
        self.update_display()

    def _refresh_triage_badge(self):
        if not hasattr(self, "triage_badge"):
            return
        c = self._theme()
        status = getattr(self, "_triage_status", None)
        n = len(getattr(self, "suggested_bboxes", []) or [])
        if status == "flagged":
            text, col = f"FLAGGED · {n}", c["warn"]
        elif status == "clean":
            text, col = "CLEAN", c["good"]
        else:
            text, col = "UNTRIAGED", c["text_muted"]
        self.triage_badge.setText(text)
        self.triage_badge.setStyleSheet(
            f"color:{col}; border:1px solid {col}; border-radius:20px;"
            f"padding:3px 9px; font-size:11px; font-weight:600;"
        )

    def _refresh_topbar(self):
        if not hasattr(self, "filename_label"):
            return
        total = len(self.img_paths)
        self.filename_label.setText(self.img_name or "—")
        self.imgpos_label.setText(f"Image {self.current_idx + 1} of {total}")
        if total > 0:
            self.top_progress.setValue(int((self.current_idx + 1) / total * 100))
        self._refresh_triage_badge()

    # ---- tool tabs (drive the existing draw/delete checkboxes) ----
    def set_tool(self, tool: str):
        if tool == "select":
            self.draw_mode_checkbox.setChecked(False)
            self.delete_click_mode_checkbox.setChecked(False)
        elif tool == "draw":
            self.delete_click_mode_checkbox.setChecked(False)
            self.draw_mode_checkbox.setChecked(True)
        else:  # delete
            self.draw_mode_checkbox.setChecked(False)
            self.delete_click_mode_checkbox.setChecked(True)
        self._sync_tool_tabs()

    def _sync_tool_tabs(self):
        if self.draw_mode_checkbox.isChecked():
            self.current_tool = "draw"
        elif self.delete_click_mode_checkbox.isChecked():
            self.current_tool = "delete"
        else:
            self.current_tool = "select"
        for b, t in ((self.tool_select_btn, "select"),
                     (self.tool_draw_btn, "draw"),
                     (self.tool_delete_btn, "delete")):
            b.blockSignals(True)
            b.setChecked(self.current_tool == t)
            b.blockSignals(False)
        self._update_drag_mode()

    def _update_drag_mode(self):
        """Hand-pan only in Select tool when zoomed in; off during draw/delete."""
        if self.current_tool == "select" and self.right_preview_view.is_zoomed():
            self.right_preview_view.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            self.right_preview_view.setDragMode(QGraphicsView.NoDrag)

    def _sync_method_buttons(self):
        for val, b in getattr(self, "_method_buttons", {}).items():
            b.blockSignals(True)
            b.setChecked(val == self.method)
            b.blockSignals(False)

    # ---- zoom / pan ----
    def zoom_in(self):
        self.right_preview_view.zoom_by(1.25)

    def zoom_out(self):
        self.right_preview_view.zoom_by(1.0 / 1.25)

    def zoom_fit(self):
        self._needs_fit = False
        self.right_preview_view.fit_view()

    def _on_zoom_changed(self, scale: float):
        self._update_zoom_label(scale)
        self._update_drag_mode()

    def _update_zoom_label(self, scale=None):
        if not hasattr(self, "zoom_label"):
            return
        if scale is None:
            scale = self.right_preview_view.current_scale()
        fit = getattr(self.right_preview_view, "_fit_scale", 1.0) or 1.0
        self.zoom_label.setText(f"{int(round(scale / fit * 100))}%")

    # ---- working-folder picker ----
    def on_pick_folder(self):
        start = str(self.working_folder) if self.working_folder else ""
        d = QFileDialog.getExistingDirectory(self, "Open working folder", start)
        if not d:
            return
        folder = Path(d)
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        imgs = triage_sort(sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        ))
        if not imgs:
            QMessageBox.warning(self, "No images", f"No images found in:\n{folder}")
            return
        self.working_folder = folder
        self.img_paths = imgs
        self.current_idx = 0
        self.cached_binaries.clear()
        self.cached_regions.clear()
        self.cached_yolo.clear()
        self.cached_scaled_img.clear()
        self.cached_yolo_by_class.clear()
        self.folder_btn.setText(f"{folder.name}  ▾")
        self._needs_fit = True
        self.load_current_image()
        self.statusBar().showMessage(f"Opened {folder} — {len(imgs)} images")

    # ---- canvas toolbar toggles ----
    def on_binary_mask_toggled(self, state):
        self.binary_view = (state == Qt.Checked)
        self.update_display()

    # ---- top-bar Save / Skip (reuse the keyboard workflow verbatim) ----
    def _save_and_next(self):
        self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))

    def _skip(self):
        self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))

    # ---- detections list per-row delete ----
    def _delete_bbox_index(self, i: int):
        if 0 <= i < len(getattr(self, "bbox_list", [])):
            bbox = self.bbox_list[i]
            self.deleted_bboxes.add(bbox)
            if self.selected_bbox_key == bbox:
                self.selected_bbox_idx = -1
                self.selected_bbox_key = None
            self.update_display()

    def _make_detection_row(self, i, text, conf_str, dot_color, is_suggested):
        c = self._theme()
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(6, 2, 4, 2)
        lay.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background:{dot_color}; border-radius:5px;")
        lay.addWidget(dot)
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{c['warn'] if is_suggested else c['text_color']}; background:transparent;"
        )
        lay.addWidget(lbl, 1)
        cf = QLabel(conf_str)
        cf.setObjectName("mono")
        lay.addWidget(cf)
        x = QToolButton()
        x.setText("✕")
        x.setAutoRaise(True)
        x.setCursor(Qt.PointingHandCursor)
        x.setToolTip("Delete (Del)")
        x.setStyleSheet(f"color:{c['text_muted']}; border:none; background:transparent;")
        x.clicked.connect(lambda _=False, idx=i: self._delete_bbox_index(idx))
        lay.addWidget(x)
        return w

    def _position_log_chip(self):
        if not hasattr(self, "log_label") or not hasattr(self, "right_preview_view"):
            return
        self.log_label.adjustSize()
        vp = self.right_preview_view.viewport()
        self.log_label.move(10, max(10, vp.height() - self.log_label.height() - 10))
        self.log_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "right_preview_view") and self.right_preview_view.scene() is not None:
            rect = self.right_preview_view.scene().sceneRect()
            if rect.width() > 0 and rect.height() > 0:
                self.right_preview_view.fit_view()
                self._needs_fit = False
        self._position_log_chip()

    def on_context_changed(self, idx):
        """Update UI based on selected sample context"""
        context_hints = {
            0: "",
            1: "Pure sample: Expect all particles to be the same class. Review boxes with different predicted classes.",
            2: "Contaminated: Expect 2+ distinct classes. Check for class diversity in detections.",
            3: "Environmental: Expect target particles plus background matrix. Consider filtering by size/morphology."
        }
        self.context_info_label.setText(context_hints.get(idx, ""))
        self.update_display()

    def get_context_type(self) -> str:
        """Returns the current context type"""
        idx = self.context_combo.currentIndex()
        return ["none", "pure", "contaminated", "environmental"][idx]

    def analyze_context_compliance(self):
        """
        Analyze current detections against expected context.
        Returns dict with warnings/suggestions.
        """
        context = self.get_context_type()
        if context == "none" or not self.use_yolo:
            return {}

        # Count unique predicted classes
        predicted_classes = set()
        for i, bbox in enumerate(self.bbox_list):
            manual_cls = self.bbox_manual_class.get(bbox, None)
            if manual_cls is not None:
                predicted_classes.add(int(manual_cls))
            else:
                predicted_classes.add(int(self.last_yolo_pred_cls[i]))

        num_classes = len(predicted_classes)
        num_boxes = len(self.bbox_list)

        analysis = {
            "num_classes": num_classes,
            "num_boxes": num_boxes,
            "warnings": [],
            "suggestions": []
        }

        if context == "pure":
            if num_classes > 1:
                analysis["warnings"].append(
                    f"⚠️ Pure sample expected, but {num_classes} different classes detected"
                )
                analysis["suggestions"].append(
                    "Review boxes with minority classes - they may be misclassifications"
                )
            elif num_classes == 1 and num_boxes > 0:
                analysis["suggestions"].append(
                    f"✓ All {num_boxes} particles classified as same class (expected for pure sample)"
                )

        elif context == "contaminated":
            if num_classes < 2:
                analysis["warnings"].append(
                    f"⚠️ Contaminated sample expected, but only {num_classes} class detected"
                )
                analysis["suggestions"].append(
                    "Check if contamination is present or adjust detection confidence threshold"
                )
            else:
                analysis["suggestions"].append(
                    f"✓ {num_classes} different classes detected (expected for contamination)"
                )

        elif context == "environmental":
            # For environmental, we expect one dominant class
            if num_boxes > 0:
                class_counts = {}
                for i, bbox in enumerate(self.bbox_list):
                    manual_cls = self.bbox_manual_class.get(bbox, None)
                    cls = int(manual_cls if manual_cls is not None else self.last_yolo_pred_cls[i])
                    class_counts[cls] = class_counts.get(cls, 0) + 1

                if class_counts:
                    dominant_cls = max(class_counts, key=class_counts.get)
                    dominant_pct = class_counts[dominant_cls] / num_boxes * 100

                    analysis["suggestions"].append(
                        f"Dominant class: {self._class_name(dominant_cls)} "
                        f"({class_counts[dominant_cls]}/{num_boxes} = {dominant_pct:.1f}%)"
                    )

                    if dominant_pct < 70:
                        analysis["warnings"].append(
                            "⚠️ No clearly dominant class - check for heavy background interference"
                        )

        return analysis

    def update_crop_preview(self):
        """
        Updates the placeholder panel with a zoomed crop of the selected bbox.
        """
        if self.current_img_gray is None:
            self.placeholder_top.setText("No image loaded")
            self.placeholder_top.setPixmap(QPixmap())
            if hasattr(self, "crop_meta_label"):
                self.crop_meta_label.setText("")
            return

        if self.selected_bbox_key is None:
            self.placeholder_top.setText("Select a box to preview")
            self.placeholder_top.setPixmap(QPixmap())
            if hasattr(self, "crop_meta_label"):
                self.crop_meta_label.setText("")
            return

        try:
            # Use square crop for consistent view; margin is optional (tweak if you want)
            min_r, min_c, max_r, max_c = square_bbox(self.selected_bbox_key, self.current_img_gray.shape, margin=10)
            crop = self.current_img_gray[min_r:max_r, min_c:max_c]

            crop_8 = ensure_8bit(crop)
            disp = cv2.cvtColor(crop_8, cv2.COLOR_GRAY2RGB)
            h, w = disp.shape[:2]
            qimg = QImage(disp.data, w, h, disp.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            # Fit crop into the placeholder label while keeping aspect ratio
            target = self.placeholder_top.size()
            if target.width() > 0 and target.height() > 0:
                pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.placeholder_top.setText("")  # clear text when showing an image
            self.placeholder_top.setPixmap(pix)

            if hasattr(self, "crop_meta_label"):
                sr, sc, er, ec = self.selected_bbox_key
                bw, bh = ec - sc, er - sr
                self.crop_meta_label.setText(f"{bw}×{bh} px · area ≈ {bw * bh} px²")
        except Exception as e:
            self.placeholder_top.setText(f"Preview error: {e}")
            self.placeholder_top.setPixmap(QPixmap())
            if hasattr(self, "crop_meta_label"):
                self.crop_meta_label.setText("")

    def _set_log(self, state: str, img_name: str | None = None):
        """
        state: 'Awaiting' | 'Processed' | 'Skipped'
        """
        name = img_name if img_name is not None else getattr(self, "img_name", "")
        if not name:
            self.log_label.setText(f"{state}: (no image)")
        else:
            self.log_label.setText(f"{state}: {name}")

    def on_bbox_info_row_changed(self, row: int):
        """
        Sidebar list selection -> selects the corresponding bbox.
        """
        if row < 0 or row >= len(getattr(self, "bbox_list", [])):
            return
        self.selected_bbox_idx = row
        self.selected_bbox_key = self.bbox_list[row]

        # Sync dropdown safely
        self.bbox_dropdown.blockSignals(True)
        self.bbox_dropdown.setCurrentIndex(row)
        self.bbox_dropdown.blockSignals(False)

        # Update rectangle highlight
        for i, item in enumerate(self.bbox_items):
            item.setPen(QPen(QColor("#FF0000") if i == self.selected_bbox_idx else QColor("#00FF00"), 2))

    def on_scene_selection_changed(self):
        selected_items = self.right_preview_scene.selectedItems()

        if not selected_items:
            self.selected_bbox_key = None
        else:
            bbox_key = selected_items[0].data(0)
            self.selected_bbox_key = bbox_key if bbox_key is not None else None

        if self.selected_bbox_key in getattr(self, "bbox_list", []):
            self.selected_bbox_idx = self.bbox_list.index(self.selected_bbox_key)
        else:
            self.selected_bbox_idx = -1

        self.bbox_dropdown.blockSignals(True)
        self.bbox_dropdown.setCurrentIndex(self.selected_bbox_idx if self.selected_bbox_idx >= 0 else -1)
        self.bbox_dropdown.blockSignals(False)

        self.bbox_info_list.blockSignals(True)
        self.bbox_info_list.setCurrentRow(self.selected_bbox_idx if self.selected_bbox_idx >= 0 else -1)
        self.bbox_info_list.blockSignals(False)

        for i, rect in enumerate(self.bbox_items):
            rect.setPen(QPen(QColor("#FF0000") if i == self.selected_bbox_idx else QColor("#00FF00"), 2))

        self.update_crop_preview()

    def _reset_per_image_state(self):
        self.user_bboxes.clear()
        self.suggested_bboxes.clear()
        self.deleted_bboxes.clear()
        self.bbox_manual_class.clear()

        # Clear YOLO-per-class cache too (since it’s image-specific)
        self.cached_yolo_by_class.clear()

        self.selected_bbox_idx = -1
        self.selected_bbox_key = None

    # -------------------- Image Loading --------------------
    def load_current_image(self):
        self.current_idx = max(0, min(self.current_idx, len(self.img_paths) - 1))
        self.img_name = self.img_paths[self.current_idx].name

        # Reset state for the newly loaded image
        self._reset_per_image_state()

        # Log: awaiting this image
        self._set_log("Awaiting", self.img_name)

        # Load grayscale image
        if self.img_paths[self.current_idx] not in self.cached_scaled_img:
            img_gray = load_as_gray(str(self.img_paths[self.current_idx]))
            self.current_img_gray = ensure_8bit(img_gray)
            self.cached_scaled_img[self.img_paths[self.current_idx]] = self.current_img_gray
        else:
            self.current_img_gray = self.cached_scaled_img[self.img_paths[self.current_idx]]

        # Register image in DB and store id for caching lookups
        h, w = self.current_img_gray.shape[:2]
        self.current_image_id = db.get_or_create_image(
            self.img_paths[self.current_idx],
            width=w, height=h,
        )

        # Pre-draw triage suggestions (blobs threshold found but YOLO missed)
        triage_status, missed_boxes = db.get_triage_by_path(self.img_paths[self.current_idx])

        # Drop suggestions already covered by the YOLO boxes shown at the
        # CURRENT confidence (triage may have run at a different conf)
        if self.use_yolo and missed_boxes and self.yolo_detector is not None:
            try:
                dets, _ = self.get_yolo_results()
                yolo_boxes = [tuple(d["bbox_rc"]) for d in dets]
            except Exception:
                yolo_boxes = []
            if yolo_boxes:
                missed_boxes = [
                    b for b in missed_boxes
                    if max(bbox_coverage(b, y) for y in yolo_boxes) < 0.5
                ]

        for box in missed_boxes:
            min_r, min_c, max_r, max_c = box
            min_r = int(np.clip(min_r, 0, h - 1))
            max_r = int(np.clip(max_r, 0, h))
            min_c = int(np.clip(min_c, 0, w - 1))
            max_c = int(np.clip(max_c, 0, w))
            if (max_r - min_r) < 5 or (max_c - min_c) < 5:
                continue
            key = (min_r, min_c, max_r, max_c)
            self.user_bboxes.add(key)
            self.suggested_bboxes.add(key)

        if triage_status == "flagged":
            triage_txt = f"[FLAGGED: {len(self.suggested_bboxes)} suggestions]"
        elif triage_status == "clean":
            triage_txt = "[CLEAN]"
        else:
            triage_txt = "[UNTRIAGED]"
        self.setWindowTitle(f"{self.img_name}  {triage_txt}")

        # Redesign: refresh the top action bar + triage badge, refit the canvas.
        self._triage_status = triage_status
        self._needs_fit = True
        self._refresh_topbar()

        self.update_display()

    # -------------------- Controls --------------------

    def on_min_area_changed(self, val):
        self.min_area_live = val
        if hasattr(self, "min_area_value_label"):
            self.min_area_value_label.setText(f"{val} px²")
        self.update_display()

    def on_otsu_offset_changed(self, val):
        self.otsu_offset = val
        self.otsu_input.setText(str(val))
        if hasattr(self, "thresh_value_label"):
            self.thresh_value_label.setText(str(val))
        # Invalidate cache
        key = self.img_paths[self.current_idx]
        if key in self.cached_binaries:
            del self.cached_binaries[key]
        self.cached_regions.pop(key, None)
        self.update_display()

    def on_otsu_input(self):
        try:
            val = int(self.otsu_input.text())
            self.otsu_offset = val
            self.otsu_slider.setValue(val)
            key = self.img_paths[self.current_idx]
            if key in self.cached_binaries:
                del self.cached_binaries[key]
            self.cached_regions.pop(key, None)
            self.update_display()
        except ValueError:
            self.otsu_input.setText(str(self.otsu_offset))

    def on_method_changed(self, text):
        self.method = text
        self._sync_method_buttons()
        # Invalidate cache for current image
        key = self.img_paths[self.current_idx]
        if key in self.cached_binaries:
            del self.cached_binaries[key]
        self.cached_regions.pop(key, None)
        self.update_display()

    def on_yolo_toggled(self, state):
        self.use_yolo = state == Qt.Checked
        key = self.img_paths[self.current_idx]
        if key in self.cached_binaries:
            del self.cached_binaries[key]
        self.cached_regions.pop(key, None)

        if not self.use_yolo and hasattr(self, "fusion_checkbox"):
            self.use_fusion_active = False
            self.fusion_checkbox.blockSignals(True)
            self.fusion_checkbox.setChecked(False)
            self.fusion_checkbox.blockSignals(False)

        self.update_display()

    def on_yolo_conf_changed(self, val):
        self.yolo_conf = round(val / 100.0, 2)
        self.yolo_conf_label.setText(f"{self.yolo_conf:.2f}")
        # Invalidate YOLO cache so results are re-run at the new threshold
        key = self.img_paths[self.current_idx]
        if key in self.cached_yolo:
            del self.cached_yolo[key]
        self.cached_yolo_by_class.clear()
        self.update_display()

    def on_fusion_toggled(self, state):
        """
        User toggle: apply fusion or use raw YOLO predictions.
        Only meaningful if YOLO is enabled and fusion models are loaded.
        """
        self.use_fusion_active = (state == Qt.Checked)

        # If user turns fusion ON but YOLO is OFF, keep it OFF (fusion depends on YOLO boxes here)
        if self.use_fusion_active and not self.use_yolo:
            self.use_fusion_active = False
            self.fusion_checkbox.blockSignals(True)
            self.fusion_checkbox.setChecked(False)
            self.fusion_checkbox.blockSignals(False)

        self.update_display()

    def on_min_area_input(self):
        try:
            val = int(self.min_area_input.text())
            self.min_area_live = max(1, val)
            self.min_area_slider.setValue(self.min_area_live)
            if hasattr(self, "min_area_value_label"):
                self.min_area_value_label.setText(f"{self.min_area_live} px²")
            self.update_display()
        except ValueError:
            self.min_area_input.setText(str(self.min_area_live))

    def on_bbox_selected(self, idx):
        if idx < 0 or idx >= len(getattr(self, "bbox_list", [])):
            self.selected_bbox_idx = -1
            self.selected_bbox_key = None
        else:
            self.selected_bbox_idx = idx
            self.selected_bbox_key = self.bbox_list[idx]

        for i, item in enumerate(self.bbox_items):
            item.setPen(QPen(QColor("#FF0000") if i == self.selected_bbox_idx else QColor("#00FF00"), 2))

        self.update_crop_preview()

    def on_delete_bbox(self):
        if self.selected_bbox_key is None:
            return  # nothing selected

        # Mark as deleted (stable across redraws)
        self.deleted_bboxes.add(self.selected_bbox_key)

        # Clear selection and redraw from source-of-truth
        self.selected_bbox_idx = -1
        self.selected_bbox_key = None
        self.update_display()

    def next_image(self):
        if self.current_idx + 1 < len(self.img_paths):
            self.current_idx += 1
            self.load_current_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()

    # -------------------- Productivity mode --------------------

    def on_productivity_toggled(self, enabled: bool):
        self.productivity_enabled = bool(enabled)

    def on_reward_percent_changed(self, val: int):
        self.reward_percent = int(val)

    def on_break_minutes_changed(self, val: int):
        self.break_minutes = int(val)

    def _register_completion(self):
        """
        Count a completed (saved) image and, if productivity mode is on, pop a
        reward each time another `reward_percent` of the images is finished.
        """
        self.completed_count += 1
        if not self.productivity_enabled:
            return

        total = len(self.img_paths)
        if total <= 0 or self.reward_percent <= 0:
            return

        # Track progress as a unit-independent completion percentage so that
        # changing reward_percent mid-session stays consistent. Reward at the
        # next multiple of the CURRENT reward_percent above what we last paid.
        done_pct = self.completed_count / total * 100.0
        next_threshold = (
            math.floor(self._last_reward_pct / self.reward_percent) + 1
        ) * self.reward_percent

        if done_pct + 1e-9 >= next_threshold:
            # Advance past every threshold crossed; show a single reward for the
            # highest one reached (avoids stacking dialogs).
            milestone = next_threshold
            while done_pct + 1e-9 >= milestone + self.reward_percent:
                milestone += self.reward_percent
            self._last_reward_pct = milestone
            self._show_reward(int(round(milestone)))

    def _show_reward(self, milestone_pct: int):
        # Base break chance scales with progress-per-reward, plus an accumulating
        # bonus from prior no-breaks (pity timer).
        base_prob = self.reward_percent / 100.0
        break_prob = min(0.95, max(0.05, base_prob + self._break_bonus))

        dlg = RewardDialog(self.break_minutes, milestone_pct, break_prob, parent=self)
        dlg.exec_()

        # Update the pity bonus from the outcome: reset on a break, grow on a
        # no-break so the next reward is more likely to be a break.
        if dlg.got_break:
            self._break_bonus = 0.0
        elif dlg.got_break is False:
            self._break_bonus += self._break_bonus_step

    def on_class_changed(self, current, previous):
        if current:
            self.object_class_id = int(current.data(Qt.UserRole))
            self.update_display()

    def on_assign_class_to_selected_bbox(self):
        """
        Assign currently selected sidebar class to the currently selected bbox.
        Only works when a real class is selected (>=0).
        """
        if self.selected_bbox_key is None:
            return
        if self.object_class_id < 0:
            return
        self.bbox_manual_class[self.selected_bbox_key] = int(self.object_class_id)
        self.update_display()

    def on_clear_class_for_selected_bbox(self):
        """Remove manual class assignment for selected bbox."""
        if self.selected_bbox_key is None:
            return
        if self.selected_bbox_key in self.bbox_manual_class:
            del self.bbox_manual_class[self.selected_bbox_key]
        self.update_display()

    def on_bbox_label_toggle(self, state):
        self.show_bbox_labels = state == Qt.Checked
        for text_item in self.text_items:
            text_item.setVisible(self.show_bbox_labels)

    def on_delete_click_mode_toggled(self, state):
        """
        When enabled, clicking on a bbox immediately deletes it.
        Mutually exclusive with draw mode.
        """
        enabled = state == Qt.Checked
        self._delete_click_mode = enabled
        self.right_preview_view.set_delete_click_mode(enabled)

        # Disable draw mode if delete mode is enabled
        if enabled and self.draw_mode_checkbox.isChecked():
            self.draw_mode_checkbox.setChecked(False)

        # Make bbox items non-selectable in delete mode
        for item in getattr(self, "bbox_items", []):
            item.setFlag(QGraphicsRectItem.ItemIsSelectable, not enabled)

        # Update cursor to indicate delete mode
        if enabled:
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            painter.setPen(QColor("red"))
            painter.drawLine(0, 16, 31, 16)
            painter.drawLine(16, 0, 16, 31)
            painter.end()

            cursor = QCursor(pixmap)
            self.right_preview_view.setCursor(cursor)
        else:
            self.right_preview_view.setCursor(Qt.ArrowCursor)

        self._sync_tool_tabs()

    def on_draw_mode_toggled(self, state):
        """
        When enabled, click-drag draws rectangles on the right preview.
        When disabled, normal selection works.
        """
        enabled = state == Qt.Checked
        self.right_preview_view.set_draw_mode(enabled)

        # Disable delete mode if draw mode is enabled
        if enabled and self.delete_click_mode_checkbox.isChecked():
            self.delete_click_mode_checkbox.setChecked(False)

        # Optional: make existing bbox items non-selectable while drawing
        for item in getattr(self, "bbox_items", []):
            item.setFlag(QGraphicsRectItem.ItemIsSelectable, not enabled)

        self._sync_tool_tabs()

    def on_bbox_clicked_for_delete(self, bbox_tuple):
        """
        NEW: Called when a bbox is clicked in delete-on-click mode.
        Immediately deletes the bbox.
        """
        # Allowed in delete mode, or via the right-click delete gesture in draw mode
        if not (self._delete_click_mode or self.draw_mode_checkbox.isChecked()):
            return

        if bbox_tuple in self.bbox_list:
            self.deleted_bboxes.add(bbox_tuple)

            # If this was the selected bbox, clear selection
            if self.selected_bbox_key == bbox_tuple:
                self.selected_bbox_idx = -1
                self.selected_bbox_key = None

            # Immediately update display to show deletion
            self.update_display()

    def on_area_deleted(self, region):
        """
        NEW: Called when the user drags a selection rectangle in delete mode.
        Deletes every bbox whose center falls inside the dragged region.
        """
        # Allowed in delete mode, or via the right-click delete gesture in draw mode
        if not (self._delete_click_mode or self.draw_mode_checkbox.isChecked()):
            return

        r_min, c_min, r_max, c_max = region
        deleted = 0
        for bbox in list(getattr(self, "bbox_list", [])):
            b_r0, b_c0, b_r1, b_c1 = bbox
            cy = (b_r0 + b_r1) / 2.0
            cx = (b_c0 + b_c1) / 2.0
            if r_min <= cy <= r_max and c_min <= cx <= c_max:
                self.deleted_bboxes.add(bbox)
                if self.selected_bbox_key == bbox:
                    self.selected_bbox_idx = -1
                    self.selected_bbox_key = None
                deleted += 1

        if deleted:
            self.update_display()
            self.statusBar().showMessage(f"Deleted {deleted} box(es)")

    def on_user_rect_created(self, bbox_rc):
        """
        bbox_rc: (min_r, min_c, max_r, max_c) in scene/image coordinates.
        Add to user_bboxes and refresh.
        """
        if self.current_img_gray is None:
            return

        min_r, min_c, max_r, max_c = bbox_rc
        H, W = self.current_img_gray.shape[:2]

        # Clip to image bounds
        min_r = int(np.clip(min_r, 0, H - 1))
        max_r = int(np.clip(max_r, 0, H))
        min_c = int(np.clip(min_c, 0, W - 1))
        max_c = int(np.clip(max_c, 0, W))

        # Ignore tiny boxes
        if (max_r - min_r) < 5 or (max_c - min_c) < 5:
            return

        bbox_key = (min_r, min_c, max_r, max_c)
        self.user_bboxes.add(bbox_key)

        # If a class is explicitly selected, assign it to this drawn box
        if int(getattr(self, "object_class_id", -1)) >= 0:
            self.bbox_manual_class[bbox_key] = int(self.object_class_id)

        # Select the newly created bbox
        self.selected_bbox_key = bbox_key
        self.selected_bbox_idx = -1

        self.update_display()

    # -------------------- Caching helpers --------------------
    def get_current_binary(self):
        key = self.img_paths[self.current_idx]
        if key in self.cached_binaries:
            return self.cached_binaries[key]

        if self.use_yolo:
            # YOLO doesn’t use threshold
            binary = np.zeros_like(self.current_img_gray)
        else:
            # Pass current interactive parameters
            if self.method == "adaptive":
                binary, _ = threshold_image(
                    self.current_img_gray,
                    method="adaptive",
                    object_bright=self.object_bright,
                    block_size=self.adaptive_block_size,
                    C=self.otsu_offset,  # adaptive C
                )
            else:  # Otsu
                binary, _ = threshold_image(
                    self.current_img_gray,
                    method="otsu",
                    object_bright=self.object_bright,
                    otsu_offset=self.otsu_offset,  # <-- make sure this is applied
                )
            binary = fill_holes(binary)

        self.cached_binaries[key] = binary
        return binary

    def get_yolo_results(self):
        key = self.img_paths[self.current_idx]
        if key in self.cached_yolo:
            return self.cached_yolo[key]

        if self.yolo_detector is None:
            raise ValueError("YOLO is enabled but no yolo_detector was passed to ImagePreview")

        image_id = getattr(self, "current_image_id", None)

        # Check persistent DB cache
        if image_id is not None:
            cached = db.get_cached_detections(image_id, "yolo", conf=self.yolo_conf)
            if cached is not None:
                self.cached_yolo[key] = (cached, None)
                return cached, None

        dets, results = yolo_detect(self.current_img_gray, conf=self.yolo_conf)

        # Persist to DB (store as list of dicts, drop the ultralytics Results object)
        if image_id is not None:
            serialisable = [
                {"bbox_rc": list(d["bbox_rc"]), "conf": float(d["conf"]), "cls_id": int(d["cls_id"])}
                for d in dets
            ]
            db.cache_detections(image_id, "yolo", serialisable, conf=self.yolo_conf)

        self.cached_yolo[key] = (dets, results)
        return dets, results

    def get_yolo_results_for_class(self, class_id: int):
        """
        Run YOLO filtered to a single class, cache results.
        Used to estimate "probability/confidence for selected class" per bbox.
        """
        img_key = self.img_paths[self.current_idx]
        cache_key = (img_key, int(class_id))
        if cache_key in self.cached_yolo_by_class:
            return self.cached_yolo_by_class[cache_key]

        if self.yolo_detector is None:
            raise ValueError("YOLO is enabled but no yolo_detector was passed to ImagePreview")

        dets, _ = yolo_detect(self.current_img_gray, conf=0.001, classes=[int(class_id)])
        self.cached_yolo_by_class[cache_key] = dets
        return dets

    def class_conf_for_bbox(self, bbox_rc, class_id: int, iou_match_thresh: float = 0.3) -> float:
        """
        Estimate confidence for (bbox, class_id) by matching against class-filtered detections.
        """
        dets_c = self.get_yolo_results_for_class(class_id)
        best = 0.0
        for d in dets_c:
            iou = bbox_iou_rc(bbox_rc, d["bbox_rc"])
            if iou >= iou_match_thresh:
                best = max(best, float(d["conf"]))
        return float(best)

    def _class_name(self, cls_id: int) -> str:
        for k, name in self.class_definitions:
            if k == cls_id:
                return name
        return f"class_{cls_id}"

    def _class_color(self, cls_id: int) -> QColor:
        if cls_id < 0:
            return QColor("#aaaaaa")
        return self.class_palette[cls_id % len(self.class_palette)]

    def get_current_regions(self):
        """
        Connected-component regions of the current binary mask as
        [(bbox, area), ...]. Cached per image and reused across updates so
        skimage.label runs once per binary (not once per parameter change).
        """
        key = self.img_paths[self.current_idx]
        cached = self.cached_regions.get(key)
        if cached is not None:
            return cached

        binary = self.get_current_binary()
        labeled = measure.label(binary, connectivity=2)
        regions = [
            (tuple(int(v) for v in r.bbox), int(r.area))
            for r in measure.regionprops(labeled)
        ]
        self.cached_regions[key] = regions
        return regions

    # -------------------- Particle areas --------------------
    def get_particle_areas(self, binary_raw):
        areas = [a for (_bbox, a) in self.get_current_regions() if a >= self.min_area_live]
        return np.asarray(areas)

    def _topk_str(self, probs: np.ndarray, k: int = 3) -> str:
        probs = np.asarray(probs).reshape(-1)
        if probs.size == 0:
            return "[]"
        kk = int(min(k, probs.size))
        idxs = np.argsort(probs)[::-1][:kk]
        parts = []
        for j in idxs:
            parts.append(f"{self._class_name(int(j))}:{float(probs[j]):.6f}")
        return "[" + ", ".join(parts) + "]"

    def _prob_stats(self, probs: np.ndarray) -> str:
        p = np.asarray(probs).reshape(-1)
        if p.size == 0:
            return "sum=nan max=nan second=nan"
        order = np.sort(p)
        second = float(order[-2]) if p.size >= 2 else float("nan")
        nonzero = p[p != 0]
        mnz = float(nonzero.min()) if nonzero.size else 0.0
        return f"sum={float(p.sum()):.6f} max={float(p.max()):.6f} second={second:.3e} min_nonzero={mnz:.3e}"

    # -------------------- DISPLAY UPDATE --------------------
    def update_display(self):
        if self.current_img_gray is None:
            return

        # ------------------- Get binary mask -------------------
        binary_raw = self.get_current_binary()
        areas = self.get_particle_areas(binary_raw)

        # ------------------- Update plot -------------------
        if len(areas) > 0:
            areas = np.array(areas)
            bins = np.linspace(areas.min(), areas.max(), 25)
            hist, edges = np.histogram(areas, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            self.area_curve.setData(centers, hist, stepMode=False, fillLevel=0)
        else:
            self.area_curve.setData([])

        # ------------------- Bounding boxes -------------------
        if self.use_yolo:
            dets, results = self.get_yolo_results()

            # NEW: remove overlaps from model detections (class-agnostic NMS)
            dets_nms = nms_dets_class_agnostic(dets, iou_thresh=self.model_nms_iou)

            # apply delete filter after NMS
            dets_nms = [d for d in dets_nms if tuple(d["bbox_rc"]) not in self.deleted_bboxes]

            model_boxes = [tuple(int(v) for v in d["bbox_rc"]) for d in dets_nms]
            model_scores = [float(d["conf"]) for d in dets_nms]
            model_cls = [int(d["cls_id"]) for d in dets_nms]
        else:
            all_boxes = [
                bbox for (bbox, area) in self.get_current_regions()
                if area >= self.min_area_live
            ]
            model_boxes = [box for box in all_boxes if box not in self.deleted_bboxes]
            model_scores = [0.0] * len(model_boxes)
            model_cls = [-1] * len(model_boxes)

            # Add user boxes (also respect deleted_bboxes)
        user_boxes = [b for b in sorted(self.user_bboxes) if b not in self.deleted_bboxes]

        self.bbox_list = model_boxes + user_boxes
        self.last_yolo_scores = model_scores + [0.0] * len(user_boxes)
        self.last_yolo_pred_cls = model_cls + [-1] * len(user_boxes)

        # ------------------- FUSION INFERENCE -------------------
        # Apply fusion only if:
        #  - models are loaded (self.use_fusion)
        #  - user checkbox is ON (self.use_fusion_active)
        #  - YOLO is enabled and we have boxes
        if self.use_fusion and self.use_fusion_active and self.use_yolo and len(self.bbox_list) > 0:
            global_output = self.global_clf.predict(self.current_img_gray, return_features=True)
            sample_context = self.get_context_type()

            # --- Pass 1 (local predictions for ALL bboxes) ---
            local_by_bbox: dict[tuple, dict] = {}
            local_confs = []
            local_probs_stack = []

            for bbox in self.bbox_list:
                if bbox in self.bbox_manual_class:
                    continue

                min_r, min_c, max_r, max_c = bbox
                crop = self.current_img_gray[min_r:max_r, min_c:max_c]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                try:
                    out = self.local_clf.predict(crop, return_features=True)
                except Exception as e:
                    print(f"Local classifier failed for bbox {bbox}: {e}")
                    continue

                local_by_bbox[bbox] = out
                local_probs = np.asarray(out.get("probs", np.array([])), dtype=np.float32).reshape(-1)
                if local_probs.size == self.num_classes:
                    local_probs_stack.append(local_probs)
                    local_confs.append(float(out.get("confidence", 0.0)))

            # Aggregate local distribution across all particles (confidence-weighted mean)
            if local_probs_stack:
                W = np.asarray(local_confs, dtype=np.float32)
                W = W / (float(W.sum()) + 1e-9)
                agg_local_probs = (W[:, None] * np.stack(local_probs_stack, axis=0)).sum(axis=0)
                agg_local_probs = agg_local_probs / (float(agg_local_probs.sum()) + 1e-9)
            else:
                agg_local_probs = None  # no usable local preds

            # --- Pass 2 (per-bbox fusion) ---
            dbg_i = self.selected_bbox_idx if self.selected_bbox_idx is not None and self.selected_bbox_idx >= 0 else 0
            dbg_payload = None

            for i, bbox in enumerate(self.bbox_list):
                if bbox in self.bbox_manual_class:
                    continue

                min_r, min_c, max_r, max_c = bbox
                crop = self.current_img_gray[min_r:max_r, min_c:max_c]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                yolo_probs = self._make_one_hot(self.last_yolo_pred_cls[i], self.num_classes)
                yolo_output = {
                    "probs": yolo_probs,
                    "conf": self.last_yolo_scores[i],
                    "class_id": self.last_yolo_pred_cls[i],
                }

                local_output = local_by_bbox.get(bbox, None)
                if local_output is None:
                    continue

                local_probs = np.asarray(local_output.get("probs", np.array([])), dtype=np.float32).reshape(-1)
                if agg_local_probs is not None and local_probs.size == self.num_classes:
                    alpha = 0.25
                    local_probs_eff = (1.0 - alpha) * local_probs + alpha * agg_local_probs
                    local_probs_eff = local_probs_eff / (float(local_probs_eff.sum()) + 1e-9)
                    local_output = dict(local_output)
                    local_output["probs"] = local_probs_eff

                bbox_coords = self._bbox_to_normalized(bbox, self.current_img_gray.shape)
                final_class, confidence, probs = self.fusion_clf.predict(
                    yolo_output,
                    local_output,
                    global_output,
                    bbox_coords=bbox_coords,
                    sample_context=sample_context
                )

                self.last_yolo_pred_cls[i] = final_class
                self.last_yolo_scores[i] = confidence

                if self.fusion_debug and i == dbg_i:
                    local_probs_dbg = np.asarray(local_output.get("probs", np.array([])))
                    global_probs = np.asarray(global_output.get("probs", np.array([])))

                    dbg_payload = {
                        "i": i,
                        "context": sample_context,
                        "yolo_cls": int(yolo_output["class_id"]),
                        "yolo_conf": float(yolo_output["conf"]),
                        "local_cls": int(np.argmax(local_probs_dbg)) if local_probs_dbg.size else -999,
                        "local_conf": float(np.max(local_probs_dbg)) if local_probs_dbg.size else float("nan"),
                        "local_stats": self._prob_stats(local_probs_dbg),
                        "local_top": self._topk_str(local_probs_dbg),
                        "global_cls": int(np.argmax(global_probs)) if global_probs.size else -999,
                        "global_stats": self._prob_stats(global_probs),
                        "global_top": self._topk_str(global_probs),
                        "fused_cls": int(final_class),
                        "fused_conf": float(confidence),
                        "fused_stats": self._prob_stats(probs),
                        "fused_top": self._topk_str(probs),
                    }

            if self.fusion_debug and dbg_payload is not None:
                dbg_txt = (
                    f" | FUSION dbg bbox#{dbg_payload['i'] + 1} ctx={dbg_payload['context']}"
                    f" | YOLO={self._class_name(dbg_payload['yolo_cls'])}:{dbg_payload['yolo_conf']:.2f}"
                    f" | Local={self._class_name(dbg_payload['local_cls'])}:{dbg_payload['local_conf']:.2f}"
                    f" ({dbg_payload['local_stats']}) {dbg_payload['local_top']}"
                    f" | Global={self._class_name(dbg_payload['global_cls'])}"
                    f" ({dbg_payload['global_stats']}) {dbg_payload['global_top']}"
                    f" | Fused={self._class_name(dbg_payload['fused_cls'])}:{dbg_payload['fused_conf']:.2f}"
                    f" ({dbg_payload['fused_stats']}) {dbg_payload['fused_top']}"
                )
                base = self.log_label.text().split(" | FUSION dbg", 1)[0]
                self.log_label.setText(base + dbg_txt)

        # Reconcile selection
        if self.selected_bbox_key in self.bbox_list:
            self.selected_bbox_idx = self.bbox_list.index(self.selected_bbox_key)
        else:
            self.selected_bbox_key = None
            self.selected_bbox_idx = -1

        # ------------------- Update BBox dropdown -------------------
        self.bbox_dropdown.blockSignals(True)
        self.bbox_dropdown.clear()
        for i, bbox in enumerate(self.bbox_list):
            self.bbox_dropdown.addItem(f"BBox {i + 1}")
        self.bbox_dropdown.setCurrentIndex(self.selected_bbox_idx if self.selected_bbox_idx >= 0 else -1)
        self.bbox_dropdown.blockSignals(False)

        # ------------------- Update inspector Detections list -------------------
        self.bbox_info_list.blockSignals(True)
        self.bbox_info_list.clear()

        selected_class = int(getattr(self, "object_class_id", -1))

        for i, bbox in enumerate(self.bbox_list):
            manual_cls = self.bbox_manual_class.get(bbox, None)
            is_suggested = bbox in self.suggested_bboxes

            if self.use_yolo:
                # Decide which class we want to display probability for:
                # - if bbox has manual class -> show that class conf
                # - else if a class is selected in sidebar (>=0) -> show selected class conf
                # - else (Auto) -> show predicted class conf
                if manual_cls is not None:
                    shown_cls = int(manual_cls)
                    shown_conf = self.class_conf_for_bbox(bbox, shown_cls)
                elif selected_class >= 0:
                    shown_cls = selected_class
                    shown_conf = self.class_conf_for_bbox(bbox, shown_cls)
                else:
                    shown_cls = int(self.last_yolo_pred_cls[i])
                    shown_conf = float(self.last_yolo_scores[i])
            else:
                shown_cls = -1
                shown_conf = 0.0

            if is_suggested:
                body = "suggested"
                dot_color = self._theme()["warn"]
                conf_str = "—"
            elif self.use_yolo:
                body = self._class_name(shown_cls) + (" ·manual" if manual_cls is not None else "")
                dot_color = self._class_color(shown_cls).name()
                conf_str = f"{shown_conf:.2f}"
            else:
                body = f"Box {i + 1}"
                dot_color = "#00c853"
                conf_str = "—"

            text = f"{i + 1} · {body}"

            item = QListWidgetItem()
            item.setData(Qt.UserRole, i)
            item.setSizeHint(QSize(0, 30))
            self.bbox_info_list.addItem(item)
            self.bbox_info_list.setItemWidget(
                item, self._make_detection_row(i, text, conf_str, dot_color, is_suggested)
            )

        self.bbox_info_list.setCurrentRow(self.selected_bbox_idx if self.selected_bbox_idx >= 0 else -1)
        self.bbox_info_list.blockSignals(False)

        # Detections card header + Clear-suggested button
        if hasattr(self, "detections_title"):
            self.detections_title.setText(f"Detections ({len(self.bbox_list)})")
        if hasattr(self, "clear_suggested_btn"):
            self.clear_suggested_btn.setVisible(bool(self.suggested_bboxes))

        # ------------------- Update progress bar & label -------------------
        total_images = len(self.img_paths)
        current_image_idx = self.current_idx + 1  # 1-based index
        current_file = self.img_name
        num_boxes = len(self.bbox_list)

        self.progress_label.setText(
            f"Image {current_image_idx} / {total_images} — File: {current_file} — Boxes: {num_boxes}"
        )
        pct_through = int(current_image_idx / total_images * 100) if total_images else 0
        self.progress_bar_live.setValue(pct_through)
        if hasattr(self, "top_progress"):
            self.top_progress.setValue(pct_through)
        if hasattr(self, "imgpos_label"):
            self.imgpos_label.setText(f"Image {current_image_idx} of {total_images}")

        # ------------------- Context Analysis -------------------
        if self.use_yolo and hasattr(self, 'context_combo'):
            analysis = self.analyze_context_compliance()

            # Build status message
            status_parts = []
            if analysis.get("warnings"):
                status_parts.extend(analysis["warnings"])
            if analysis.get("suggestions"):
                status_parts.extend(analysis["suggestions"])

            if status_parts:
                context_status = " | ".join(status_parts)
                # Update log label with context info
                current_log = self.log_label.text()
                self.log_label.setText(f"{current_log} | {context_status}")

        # (Legacy raw/binary side panes are no longer shown in the redesign —
        # the "Binary mask" canvas toggle swaps the main view instead, so we
        # skip rendering those hidden views entirely.)

        # ------------------- RIGHT PREVIEW -------------------
        # Canvas "Binary mask" toggle swaps the shown image between raw + binary.
        if getattr(self, "binary_view", False):
            src = (binary_raw > 0).astype(np.uint8) * 255
        else:
            src = self.current_img_gray
        vis = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        h, w = vis.shape[:2]
        qimg = QImage(vis.data, w, h, vis.strides[0], QImage.Format_RGB888)
        self.right_pix.setPixmap(QPixmap.fromImage(qimg))

        # Hide scrollbars
        self.right_preview_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_preview_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Scene rect = image size
        self.right_preview_scene.setSceneRect(0, 0, w, h)

        # Fit only when a new image loads / window resizes — otherwise preserve
        # the user's current zoom/pan across parameter edits (NEW).
        if getattr(self, "_needs_fit", True):
            self.right_preview_view.fit_view()
            self._needs_fit = False
        else:
            self._update_zoom_label()

        self.right_preview_view.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        # ------------------- Draw bounding boxes -------------------
        for rect in self.bbox_items + self.text_items:
            self.right_preview_scene.removeItem(rect)
        self.bbox_items.clear()
        self.text_items.clear()

        for i, bbox in enumerate(self.bbox_list):
            min_r, min_c, max_r, max_c = bbox

            manual_cls = self.bbox_manual_class.get(bbox, None)
            if manual_cls is not None:
                vis_cls = int(manual_cls)
            else:
                vis_cls = int(self.last_yolo_pred_cls[i]) if self.use_yolo else -1

            is_user_drawn = bbox in self.user_bboxes
            is_suggested = bbox in self.suggested_bboxes
            # Suggested = dashed warn; otherwise the class colour (or green in
            # threshold mode), per the design tokens.
            if is_suggested:
                base_color = QColor(self._theme()["warn"])
            else:
                base_color = self._class_color(vis_cls) if self.use_yolo else QColor("#00c853")

            is_selected = (i == self.selected_bbox_idx or bbox == self.selected_bbox_key)

            rect = QGraphicsRectItem(min_c, min_r, max_c - min_c, max_r - min_r)
            rect.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
            rect.setData(0, bbox)
            rect.setAcceptedMouseButtons(Qt.LeftButton)

            if is_selected:
                pen = QPen(QColor(self._theme()["highlight"]), 3)
            else:
                pen = QPen(base_color, 2)
            if is_suggested:
                pen.setStyle(Qt.DashLine)
            rect.setPen(pen)

            rect.setBrush(QBrush(Qt.NoBrush))
            self.right_preview_scene.addItem(rect)
            self.bbox_items.append(rect)

            if self.show_bbox_labels:
                if is_suggested:
                    label_txt = "suggested"
                elif self.use_yolo:
                    label_txt = self._class_name(vis_cls)
                else:
                    label_txt = f"{i + 1}"
                pill = base_color.name()
                text_item = QGraphicsTextItem()
                text_item.setHtml(
                    f'<div style="background:{pill};color:#ffffff;'
                    f'padding:0px 4px;border-radius:3px;font-weight:600;">{label_txt}</div>'
                )
                font = text_item.font()
                font.setPointSize(9)
                font.setBold(True)
                text_item.setFont(font)
                text_item.setPos(min_c - 2, max(0, min_r - 20))
                self.right_preview_scene.addItem(text_item)
                self.text_items.append(text_item)

        # Keep crop preview synced
        self.update_crop_preview()

    def _make_one_hot(self, class_id: int, num_classes: int) -> np.ndarray:
        """Convert class ID to one-hot probability distribution."""
        if class_id < 0 or class_id >= num_classes:
            # Unknown class: uniform distribution
            return np.ones(num_classes) / num_classes

        probs = np.zeros(num_classes)
        probs[class_id] = 1.0
        return probs

    def _bbox_to_normalized(self, bbox: tuple, img_shape: tuple) -> np.ndarray:
        """
        Convert bbox from (min_r, min_c, max_r, max_c) to normalized [x_center, y_center, w, h].

        Args:
            bbox: (min_r, min_c, max_r, max_c) in pixel coordinates
            img_shape: (H, W) or (H, W, C)

        Returns:
            np.array([x_center, y_center, width, height]) normalized to 0-1
        """
        min_r, min_c, max_r, max_c = bbox
        H, W = img_shape[:2]

        width = max_c - min_c
        height = max_r - min_r
        x_center = (min_c + width / 2) / W
        y_center = (min_r + height / 2) / H
        norm_w = width / W
        norm_h = height / H

        return np.array([x_center, y_center, norm_w, norm_h])

    # -------------------- Dataset Reviewer --------------------
    def _open_dataset_reviewer(self):
        dataset_root = getattr(self, "dataset_root", None)
        if dataset_root is None:
            # Try to resolve from config
            try:
                repo_root = find_repo_root(Path(__file__))
                cfg = load_json(repo_root / "configs" / "config.json")
                dr = cfg.get("paths", {}).get("dataset_root", "data/complete")
                dataset_root = resolve_repo_path(repo_root, dr)
            except Exception:
                dataset_root = self.output_root.parent
        if not dataset_root.exists():
            QMessageBox.warning(self, "Dataset not found",
                                f"Could not locate dataset root:\n{dataset_root}")
            return
        self._reviewer = DatasetReviewer(
            dataset_root=dataset_root,
            class_names=self.class_definitions,
            parent=None,
        )
        self._reviewer.show()

    # -------------------- Intervention tracking --------------------
    def _update_intervention_status(self):
        stats = db.intervention_stats()
        if stats["total"] == 0:
            return
        pct = round(stats["auto_rate"] * 100)
        self.statusBar().showMessage(
            f"YOLO auto-only: {stats['auto_only']}/{stats['total']} images ({pct}%)  |  "
            f"Needed manual boxes: {stats['had_manual']}"
        )

    # -------------------- Keyboard navigation --------------------
    def keyPressEvent(self, event):
        key = event.key()

        # -------------------- YOLO confidence adjustment --------------------
        # [ decreases by 0.05,  ] increases by 0.05
        if key == Qt.Key_BracketLeft:
            new_val = max(1, self.yolo_conf_slider.value() - 5)
            self.yolo_conf_slider.setValue(new_val)
            return
        elif key == Qt.Key_BracketRight:
            new_val = min(80, self.yolo_conf_slider.value() + 5)
            self.yolo_conf_slider.setValue(new_val)
            return

        # -------------------- Skip to next image --------------------
        if key == Qt.Key_Escape:
            # Log skip for current image
            self._set_log("Skipped", getattr(self, "img_name", ""))
            image_id = getattr(self, "current_image_id", None)
            if image_id is not None:
                db.mark_skipped(image_id)

            if self.current_idx + 1 < len(self.img_paths):
                self.next_image()
            else:
                self.statusBar().showMessage("Last image reached. Press Enter to save or close manually.")
            return

        # -------------------- Save & Next --------------------
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._set_log("Processed", getattr(self, "img_name", ""))
            img_path = self.img_paths[self.current_idx]
            img_gray = self.current_img_gray
            H, W = img_gray.shape[:2]

            crop_folder = Path(self.output_root) / img_path.stem
            crop_folder.mkdir(exist_ok=True, parents=True)

            # --- Save particle crops (SQUARE crops) ---
            squared_bboxes = []
            for bbox in self.bbox_list:
                min_r, min_c, max_r, max_c = square_bbox(bbox, img_gray.shape, margin=0)
                squared_bboxes.append((min_r, min_c, max_r, max_c))

            for idx, (min_r, min_c, max_r, max_c) in enumerate(squared_bboxes, start=1):
                crop = img_gray[min_r:max_r, min_c:max_c]
                crop_name = f"{img_path.stem}_{idx:04d}.tif"
                crop_path = crop_folder / crop_name
                cv2.imwrite(str(crop_path), crop.astype(np.uint16) if crop.dtype == np.uint16 else crop)

            # --- Save metadata with context info ---
            if hasattr(self, 'context_combo'):
                metadata = {
                    "image_name": img_path.name,
                    "context_type": self.get_context_type(),
                    "num_boxes": len(self.bbox_list),
                    "analysis": self.analyze_context_compliance()
                }

                metadata_path = crop_folder / f"{img_path.stem}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            # --- Save YOLO labels (per-bbox class support) ---
            yolo_txt_path = crop_folder / f"{img_path.stem}.txt"
            with open(yolo_txt_path, "w") as f:
                for i, ((min_r, min_c, max_r, max_c), orig_bbox_key) in enumerate(zip(squared_bboxes, self.bbox_list)):
                    box_w = max_c - min_c
                    box_h = max_r - min_r
                    x_center = (min_c + box_w / 2) / W
                    y_center = (min_r + box_h / 2) / H
                    norm_w = box_w / W
                    norm_h = box_h / H

                    # Determine class id for this bbox
                    manual_cls = self.bbox_manual_class.get(orig_bbox_key, None)
                    if manual_cls is not None:
                        cls_id = int(manual_cls)
                    else:
                        if self.use_yolo:
                            # Auto: predicted class
                            cls_id = int(self.last_yolo_pred_cls[i]) if i < len(
                                getattr(self, "last_yolo_pred_cls", [])) else 0
                        else:
                            # Threshold mode fallback
                            cls_id = 0 if self.object_class_id < 0 else int(self.object_class_id)

                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            # Record annotation in DB
            image_id = getattr(self, "current_image_id", None)
            if image_id is not None:
                class_name = next(
                    (name for cid, name in (self.class_definitions or [])
                     if int(cid) == self.object_class_id),
                    str(self.object_class_id),
                )
                db.mark_annotated(image_id, class_name)
                # Accepted triage suggestions don't count as manual intervention
                had_manual = bool((self.user_bboxes - self.suggested_bboxes) - self.deleted_bboxes)
                db.mark_had_manual_boxes(image_id, had_manual)
                self._update_intervention_status()
                for idx, ((min_r, min_c, max_r, max_c), orig_bbox_key) in enumerate(
                    zip(squared_bboxes, self.bbox_list), start=1
                ):
                    manual_cls = self.bbox_manual_class.get(orig_bbox_key, None)
                    cls_id = int(manual_cls) if manual_cls is not None else (
                        int(self.last_yolo_pred_cls[idx - 1])
                        if self.use_yolo and (idx - 1) < len(getattr(self, "last_yolo_pred_cls", []))
                        else (0 if self.object_class_id < 0 else int(self.object_class_id))
                    )
                    crop_path = crop_folder / f"{img_path.stem}_{idx:04d}.tif"
                    db.record_crop(
                        image_id, crop_path,
                        (min_r, min_c, max_r, max_c),
                        cls_id,
                        manual_override=manual_cls is not None,
                    )

            # Write YOLO label file alongside the raw image (original boxes, not squared)
            raw_labels_dir = Path(self.raw_out) / "labels"
            raw_labels_dir.mkdir(exist_ok=True, parents=True)
            label_path = raw_labels_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as lf:
                for i, (orig_bbox_key, orig_box) in enumerate(zip(self.bbox_list, self.bbox_list)):
                    min_r, min_c, max_r, max_c = orig_box
                    box_w = max_c - min_c
                    box_h = max_r - min_r
                    x_center = (min_c + box_w / 2) / W
                    y_center = (min_r + box_h / 2) / H
                    norm_w = box_w / W
                    norm_h = box_h / H
                    manual_cls = self.bbox_manual_class.get(orig_box, None)
                    if manual_cls is not None:
                        cls_id = int(manual_cls)
                    elif self.use_yolo and i < len(getattr(self, "last_yolo_pred_cls", [])):
                        cls_id = int(self.last_yolo_pred_cls[i])
                    else:
                        cls_id = 0 if self.object_class_id < 0 else int(self.object_class_id)
                    lf.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            # Move raw image to raw_out folder
            move(img_path, Path(self.raw_out) / img_path.name)

            # Count this image as completed (may trigger a productivity reward)
            self._register_completion()

            # Go to next image
            self.next_image()
            return

        # -------------------- Navigation --------------------
        elif key in (Qt.Key_Right, ord('N')):
            self.next_image()
        elif key in (Qt.Key_Left, ord('P')):
            self.prev_image()

        # -------------------- Toggle YOLO --------------------
        elif key == ord('M'):
            self.use_yolo = not self.use_yolo
            self.yolo_checkbox.setChecked(self.use_yolo)
            self.update_display()

        # -------------------- Adjust Otsu Offset --------------------
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.otsu_offset += 1
            self.otsu_slider.setValue(self.otsu_offset)
            self.update_display()

        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.otsu_offset -= 1
            self.otsu_slider.setValue(self.otsu_offset)
            self.update_display()

        elif key == ord('A'):
            self.method = "adaptive" if self.method == "otsu" else "otsu"
            self.method_combo.setCurrentText(self.method)
            self.update_display()

        elif key == ord(']') and self.method == "adaptive":
            self.adaptive_block_size += 2
            if self.adaptive_block_size < 3:
                self.adaptive_block_size = 3
            self.update_display()

        elif key == ord('[') and self.method == "adaptive":
            self.adaptive_block_size = max(3, self.adaptive_block_size - 2)
            self.update_display()

        # -------------------- Adjust Min Area --------------------
        elif key == ord('.'):
            self.min_area_live += 10
            self.min_area_slider.setValue(self.min_area_live)
            self.update_display()
        elif key == ord(','):
            self.min_area_live = max(1, self.min_area_live - 10)
            self.min_area_slider.setValue(self.min_area_live)
            self.update_display()

        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_bbox_key is not None:
                self.on_delete_bbox()
            return

        # NEW: 'D' flips between draw boxes and delete boxes modes
        elif key == ord('D'):
            if self.delete_click_mode_checkbox.isChecked():
                # Currently deleting -> switch to drawing
                self.delete_click_mode_checkbox.setChecked(False)
                self.draw_mode_checkbox.setChecked(True)
                self.statusBar().showMessage("Draw boxes mode")
            else:
                # Switch to deleting (also turns off draw via mutual exclusivity)
                self.draw_mode_checkbox.setChecked(False)
                self.delete_click_mode_checkbox.setChecked(True)
                self.statusBar().showMessage("Delete boxes mode (click or drag)")
            return

        # Remove all triage-suggested (orange) boxes on this image
        elif key == ord('X'):
            if self.suggested_bboxes:
                n = len(self.suggested_bboxes)
                for b in list(self.suggested_bboxes):
                    self.user_bboxes.discard(b)
                    self.bbox_manual_class.pop(b, None)
                    if self.selected_bbox_key == b:
                        self.selected_bbox_key = None
                        self.selected_bbox_idx = -1
                self.suggested_bboxes.clear()
                self.update_display()
                self.statusBar().showMessage(f"Removed {n} suggested box(es)")
            return

        # NEW: 'F' fits the image to the canvas
        elif key == ord('F'):
            self.zoom_fit()
            return

        # Toggle fullscreen/windowed (keeps both workflows)
        elif key == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
                self.setWindowState(Qt.WindowMaximized)
            else:
                self.showFullScreen()
            return


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setCentralWidget(self.view)
        self.setWindowTitle("Microplastics Preprocessor (Interactive PyQt5)")
        self.setMinimumSize(800, 400)

        # --------------------
        # Widgets
        # --------------------
        self.in_dir_edit = QLineEdit()
        self.in_dir_btn = QPushButton("Browse...")
        self.in_dir_btn.clicked.connect(self.browse_in_dir)

        self.out_dir_edit = QLineEdit()
        self.out_dir_btn = QPushButton("Browse...")
        self.out_dir_btn.clicked.connect(self.browse_out_dir)

        self.raw_dir_edit = QLineEdit()
        self.raw_dir_btn = QPushButton("Browse...")
        self.raw_dir_btn.clicked.connect(self.browse_raw_dir)

        self.use_yolo_checkbox = QCheckBox("Use YOLO")
        self.use_yolo_checkbox.setChecked(False)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100000)
        self.min_area_spin.setValue(30)

        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 1000)
        self.margin_spin.setValue(3)

        self.preview_btn = QPushButton("🔍 Preview First Image (Interactive)")
        self.run_btn = QPushButton("⚡ Process All (Non-interactive)")

        self.status_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        # --------------------
        # Layout
        # --------------------
        main_layout = QVBoxLayout()

        # Input folder
        h_in = QHBoxLayout()
        h_in.addWidget(self.in_dir_edit)
        h_in.addWidget(self.in_dir_btn)
        main_layout.addWidget(QLabel("Input folder:"))
        main_layout.addLayout(h_in)

        # Output folder
        h_out = QHBoxLayout()
        h_out.addWidget(self.out_dir_edit)
        h_out.addWidget(self.out_dir_btn)
        main_layout.addWidget(QLabel("Output folder:"))
        main_layout.addLayout(h_out)

        # Raw-out folder
        h_raw = QHBoxLayout()
        h_raw.addWidget(self.raw_dir_edit)
        h_raw.addWidget(self.raw_dir_btn)
        main_layout.addWidget(QLabel("Raw-out folder:"))
        main_layout.addLayout(h_raw)

        main_layout.addWidget(self.use_yolo_checkbox)

        # Parameters
        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Min area:"))
        h_params.addWidget(self.min_area_spin)
        h_params.addSpacing(20)
        h_params.addWidget(QLabel("Margin:"))
        h_params.addWidget(self.margin_spin)
        main_layout.addLayout(h_params)

        # Buttons
        h_buttons = QHBoxLayout()
        h_buttons.addWidget(self.preview_btn)
        h_buttons.addWidget(self.run_btn)
        main_layout.addLayout(h_buttons)

        # Progress & status
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        use_yolo = self.use_yolo_checkbox.isChecked()

        # --------------------
        # Connections
        # --------------------
        self.preview_btn.clicked.connect(self.preview_first_image)
        self.run_btn.clicked.connect(self.start_processing)

        # Internal state
        self.worker = None
        self.preview_settings = None

    def get_current_binary(self):
        """Get current binary mask based on threshold/YOLO settings"""
        if self.use_yolo:
            return None  # YOLO doesn’t use binary threshold
        binary, _ = threshold_image(
            self.current_img_gray,
            self.method,
            self.object_bright,
            otsu_offset=self.otsu_offset
        )
        return fill_holes(binary)

    def preview_first_image(self):
        """Open interactive preview on first image."""
        in_dir = Path(self.in_dir_edit.text().strip())

        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        imgs = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        imgs = triage_sort(sorted(imgs))

        if not imgs:
            QMessageBox.warning(self, "No images", "No images found (.tif/.tiff/.png/.jpg/.jpeg/.bmp).")
            return

        out_dir = Path(self.out_dir_edit.text().strip())
        raw_dir = Path(self.raw_dir_edit.text().strip())

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        if not raw_dir.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)

        self._previewer = ImagePreview(
            img_paths=imgs,
            output_root=out_dir,
            raw_out=raw_dir,
            use_yolo=self.use_yolo_checkbox.isChecked(),
            min_area=self.min_area_spin.value(),
            parent=self
        )
        self._previewer.show()

    def accept_preview_settings(self, settings):
        """Called when user presses Enter in preview"""
        self.preview_settings = settings
        self.status_label.setText("Preview accepted! Ready to process.")

    # [rest of methods same: browse folders, start_processing, etc.]

    def browse_in_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder")
        if d:
            self.in_dir_edit.setText(d)

    def browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_dir_edit.setText(d)

    def browse_raw_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select raw-out folder")
        if d:
            self.raw_dir_edit.setText(d)

    def start_processing(self):
        in_dir = self.in_dir_edit.text().strip()
        out_dir = self.out_dir_edit.text().strip()
        raw_dir = self.raw_dir_edit.text().strip()

        if not in_dir or not out_dir or not raw_dir:
            QMessageBox.warning(self, "Missing paths", "Please set all three folders.")
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running...")

        self.worker = ProcessThread(
            in_dir=in_dir,
            out_root=out_dir,
            raw_out=raw_dir,
            object_class_id=1,
            min_area=self.min_area_spin.value(),
            margin=self.margin_spin.value(),
            use_yolo=self.use_yolo_checkbox.isChecked()
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.message.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, n):
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Finished. Processed {n} images.")

def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]  # PolyVision/ (adjust if needed)

    config_path = repo_root / "configs" / "config.json"
    colors_path = repo_root / "configs" / "gui_colors.json"

    config = load_json(config_path)
    microplastic_classes = load_microplastic_classes(config)

    app = QApplication(sys.argv)

    # GUI redesign: theme-token driven stylesheet (dark by default).
    app.setStyleSheet(build_stylesheet(THEMES["dark"]))

    # Prefer config paths, not hard-coded absolute paths
    yolo = YoloDetector(
        model_path=str((repo_root / config["models"]["yolo_weights"]).resolve()),
        device=config.get("models", {}).get("device", "cpu"),
        imgsz=int(config.get("models", {}).get("imgsz", 800)),
    )

    # TODO: pass `yolo` into your ImagePreview instead of using global yolo_model
    # previewer = ImagePreview(..., yolo_detector=yolo, microplastic_classes=microplastic_classes)

    # ... existing code that creates/launches the preview window ...

    return app.exec_()

def find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find the repo root marker (configs/config.json).
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / "configs" / "config.json").exists():
            return p
    raise FileNotFoundError(f"Could not find repo root above: {start} (expected configs/config.json)")

def resolve_repo_path(repo_root: Path, p: str | Path) -> Path:
    """
    Resolve config path to absolute:
      - absolute paths kept as-is
      - repo-relative paths resolved against repo_root
      - legacy 'PolyVision/...' prefix stripped
    """
    p = Path(str(p))
    if p.is_absolute():
        return p
    if p.parts and p.parts[0].lower() == "polyvision":
        p = Path(*p.parts[1:])
    return (repo_root / p).resolve()

# ---------------------------------------------------------------------
# === RUN ===
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import sys
    from PyQt5.QtWidgets import QApplication

    # Load configuration first; model/data paths all come from configs/config.json.
    repo_root = find_repo_root(Path(__file__))

    # Detector used by the module-level yolo_detect() helper. The path is resolved
    # from config ("models.yolo"/"models.yolo_weights") relative to the repo root.
    with open(repo_root / "configs" / "config.json", "r", encoding="utf-8") as _cf:
        _models_cfg = json.load(_cf).get("models", {})
    YOLO_MODEL_PATH = str(resolve_repo_path(
        repo_root, _models_cfg.get("yolo") or _models_cfg.get("yolo_weights", "models/detect/best.pt")))
    yolo_model = YOLO(YOLO_MODEL_PATH)

    with open(repo_root / "configs" / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    microplastic_classes = load_microplastic_classes(config)

    models_cfg = config.get("models", {})

    yolo_weights_path = resolve_repo_path(repo_root, models_cfg["yolo_weights"])
    local_model_path = resolve_repo_path(repo_root, models_cfg.get("local_classifier", ""))
    global_model_path = resolve_repo_path(repo_root, models_cfg.get("global_classifier", ""))

    print(f"[paths] repo_root        = {repo_root}")
    print(f"[paths] yolo_weights     = {yolo_weights_path}")
    print(f"[paths] local_classifier = {local_model_path}")
    print(f"[paths] global_classifier= {global_model_path}")

    if not yolo_weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")

    # Optional but recommended: sanity checks before enabling fusion
    enable_fusion = bool(models_cfg.get("enable_fusion", False))
    if enable_fusion:
        if not local_model_path.exists():
            print(f"⚠ Fusion disabled: local model not found: {local_model_path}")
            enable_fusion = False
        elif local_model_path.suffix.lower() != ".keras":
            print(f"⚠ Fusion disabled: local model is not a .keras file: {local_model_path}")
            enable_fusion = False

        if not global_model_path.exists():
            print(f"⚠ Fusion disabled: global model not found: {global_model_path}")
            enable_fusion = False
        elif global_model_path.suffix.lower() != ".keras":
            print(f"⚠ Fusion disabled: global model is not a .keras file: {global_model_path}")
            enable_fusion = False

    yolo_detector = YoloDetector(
        model_path=str(yolo_weights_path),
        device=models_cfg.get("device", "cpu"),
        imgsz=int(models_cfg.get("imgsz", 800)),
    )

    # Input / output folders are taken from configs/config.json ("paths" block),
    # resolved relative to the repo root:
    #   input_dir  = raw micrographs to annotate      (default: data/raw)
    #   output_root = where crops + labels are written (default: data/complete/<class>)
    #   raw_out     = where the whole image is moved   (default: <output_root>/whole_images)
    paths_cfg = config.get("paths", {})
    input_dir = resolve_repo_path(repo_root, paths_cfg.get("input_dir", "data/raw"))
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    img_list = triage_sort(sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])) if input_dir.exists() else []
    if not img_list:
        print(f"No image files found in {input_dir} (expected .tif/.tiff/.png/.jpg/.jpeg/.bmp)")
        sys.exit(1)

    output_root = resolve_repo_path(repo_root, paths_cfg.get("output_root", "data/complete"))
    raw_out = resolve_repo_path(repo_root, paths_cfg.get("raw_out", str(output_root / "whole_images")))
    output_root.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)

    # 4️⃣ Start PyQt5 application
    app = QApplication(sys.argv)

    # GUI redesign: theme-token driven stylesheet (dark by default). The window
    # itself re-applies this via ImagePreview.apply_theme() and the theme toggle.
    app.setStyleSheet(build_stylesheet(THEMES["dark"]))

    # Resolve dataset root (data/complete) from config for the Dataset Reviewer
    dataset_root = resolve_repo_path(
        repo_root,
        config.get("paths", {}).get("dataset_root", "data/complete"),
    )

    # 5️⃣ Launch interactive preview directly
    previewer = ImagePreview(
        img_paths=img_list,
        output_root=output_root,
        raw_out=raw_out,
        use_yolo=config["processing"]["use_yolo"],
        min_area=config["processing"]["min_area"],
        enable_fusion=enable_fusion,
        local_model_path=str(local_model_path) if enable_fusion else None,
        global_model_path=str(global_model_path) if enable_fusion else None,
        fusion_weights=tuple(config["fusion"]["weights"]),
        microplastic_classes=microplastic_classes,
        yolo_detector=yolo_detector,
        dataset_root=dataset_root,
    )
    previewer.show()

    sys.exit(app.exec_())

# python -m polyvision.app.main