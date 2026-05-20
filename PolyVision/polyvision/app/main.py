# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 19:26:02 2026

@author: joshk
"""
import json
import sys
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
    QListWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QPen, QBrush, QColor, QPainter, QCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
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
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

        imgs = []
        for p in self.in_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                imgs.append(p)

        imgs = sorted(imgs)
        n = len(imgs)
        if n == 0:
            self.message.emit("No images found (.tif/.tiff/.png/.jpg/.jpeg).")
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draw_mode = False
        self._dragging = False
        self._start_scene = None
        self._rubber_item = None
        self._delete_click_mode = False

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = bool(enabled)
        if not self._draw_mode:
            self._dragging = False
            self._start_scene = None
            if self._rubber_item is not None and self.scene() is not None:
                self.scene().removeItem(self._rubber_item)
            self._rubber_item = None

    def set_delete_click_mode(self, enabled: bool):
        """NEW: Enable/disable delete-on-click mode"""
        self._delete_click_mode = bool(enabled)

    def mousePressEvent(self, event):
        # NEW: Handle delete-on-click mode
        if self._delete_click_mode and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            items = self.scene().items(scene_pos)

            # Find first bbox item at click position
            for item in items:
                if isinstance(item, QGraphicsRectItem) and item.data(0) is not None:
                    bbox = item.data(0)
                    self.bboxClicked.emit(bbox)
                    event.accept()
                    return

            event.accept()
            return

        if self._draw_mode and event.button() == Qt.LeftButton:
            self._dragging = True
            self._start_scene = self.mapToScene(event.pos())

            if self._rubber_item is None:
                pen = QPen(QColor("#ffffff"), 2, Qt.DashLine)
                self._rubber_item = QGraphicsRectItem()
                self._rubber_item.setPen(pen)
                self._rubber_item.setBrush(QBrush(Qt.NoBrush))
                self._rubber_item.setZValue(10_000)
                if self.scene() is not None:
                    self.scene().addItem(self._rubber_item)

            self._rubber_item.setRect(QRectF(self._start_scene, self._start_scene))
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._draw_mode and self._dragging and self._rubber_item is not None and self._start_scene is not None:
            cur = self.mapToScene(event.pos())
            self._rubber_item.setRect(QRectF(self._start_scene, cur).normalized())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
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
# === INTERACTIVE PREVIEW WINDOW ===
# ---------------------------------------------------------------------

class ImagePreview(QMainWindow):
    def __init__(self, img_paths, output_root, raw_out, object_class_id=1,
                 use_yolo=False, min_area=50, parent=None,
                 enable_fusion=False, local_model_path=None, global_model_path=None,
                 fusion_weights=(0.3, 0.5, 0.2),
                 microplastic_classes=None,
                 yolo_detector: YoloDetector | None = None):
        super().__init__(parent)
        self.object_class_id = object_class_id
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
        self._delete_click_mode = False

        # Settings
        self.use_fusion = False
        self.use_fusion_active = False
        self.use_yolo = use_yolo
        self.min_area_live = min_area
        self.otsu_offset = 0
        self.method = "otsu"
        self.object_bright = True
        self.adaptive_block_size = 21
        self.yolo_detector = yolo_detector
        self.num_classes = len([cid for cid, _name in (microplastic_classes or []) if int(cid) >= 0]) or 7

        # Cache
        self.cached_binaries = {}
        self.cached_yolo = {}
        self.cached_scaled_img = {}
        self.cached_yolo_by_class = {}

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

        # --------------------- MAIN LAYOUT ---------------------
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # remove outer padding
        main_layout.setSpacing(0)  # remove spacing between widgets

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # --------------------- LEFT VBOX (controls) ---------------------
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)

        # --- Bounding box dropdown ---
        self.bbox_dropdown = QComboBox()
        self.bbox_dropdown.currentIndexChanged.connect(self.on_bbox_selected)

        self.controls_layout.addWidget(QLabel("Select bounding box"))
        self.controls_layout.addWidget(self.bbox_dropdown)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(5)

        # --- Delete bbox button ---
        self.delete_bbox_btn = QPushButton("Delete Selected BBox")
        self.delete_bbox_btn.clicked.connect(self.on_delete_bbox)
        self.controls_layout.addWidget(self.delete_bbox_btn)

        # --- Min area slider ---
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(1, 1000)
        self.min_area_slider.setValue(self.min_area_live)
        self.min_area_slider.valueChanged.connect(self.on_min_area_changed)

        self.controls_layout.addWidget(QLabel("Min area"))
        self.controls_layout.addWidget(self.min_area_slider)

        # --- Min area input ---
        self.min_area_input = QLineEdit(str(self.min_area_live))
        self.min_area_input.setMaximumWidth(60)
        self.min_area_input.editingFinished.connect(self.on_min_area_input)

        self.controls_layout.addWidget(self.min_area_input)

        # --- Otsu offset slider ---
        self.otsu_slider = QSlider(Qt.Horizontal)
        self.manual_thresh = 128
        self.otsu_slider.setRange(0, 255)
        self.otsu_slider.setValue(self.manual_thresh)
        self.otsu_slider.setValue(self.otsu_offset)
        self.otsu_slider.valueChanged.connect(self.on_otsu_offset_changed)

        self.controls_layout.addWidget(QLabel("Threshold Value"))
        self.controls_layout.addWidget(self.otsu_slider)

        # --- Otsu input ---
        self.otsu_input = QLineEdit(str(self.otsu_offset))
        self.otsu_input.setMaximumWidth(60)
        self.otsu_input.editingFinished.connect(self.on_otsu_input)

        self.controls_layout.addWidget(self.otsu_input)

        # --- Threshold method ---
        self.method_combo = QComboBox()
        self.method_combo.addItems(["otsu", "adaptive", "manual"])
        self.method_combo.setCurrentText(self.method)
        self.method_combo.currentTextChanged.connect(self.on_method_changed)

        self.controls_layout.addWidget(QLabel("Threshold method"))
        self.controls_layout.addWidget(self.method_combo)

        # --- YOLO toggle ---
        self.yolo_checkbox = QCheckBox("Use YOLO")
        self.yolo_checkbox.setChecked(self.use_yolo)
        self.yolo_checkbox.stateChanged.connect(self.on_yolo_toggled)
        self.controls_layout.addWidget(self.yolo_checkbox)

        # --- NEW: Fusion toggle (applies only when YOLO is ON and fusion models are loaded) ---
        self.fusion_checkbox = QCheckBox("Use Fusion (YOLO + Local + Global)")
        self.fusion_checkbox.setChecked(False)
        self.fusion_checkbox.setEnabled(False)  # enabled after models load
        self.fusion_checkbox.stateChanged.connect(self.on_fusion_toggled)
        self.controls_layout.addWidget(self.fusion_checkbox)

        # --- Review Dataset button ---
        self.review_btn = QPushButton("Review Dataset")
        self.review_btn.clicked.connect(self._open_dataset_reviewer)
        self.controls_layout.addWidget(self.review_btn)

        self.controls_layout.addStretch()

        self.controls_widget.setFixedWidth(260)
        main_layout.addWidget(self.controls_widget, 0)

        # --- NEW: per-bbox label/probability list (sidebar) ---
        self.controls_layout.addWidget(QLabel("BBoxes (label / probability)"))

        self.bbox_info_list = QListWidget()
        self.bbox_info_list.setSelectionMode(QListWidget.SingleSelection)
        self.bbox_info_list.currentRowChanged.connect(self.on_bbox_info_row_changed)
        self.controls_layout.addWidget(self.bbox_info_list)

        # --- NEW: Draw mode toggle ---
        self.draw_mode_checkbox = QCheckBox("Draw boxes")
        self.draw_mode_checkbox.setChecked(False)
        self.draw_mode_checkbox.stateChanged.connect(self.on_draw_mode_toggled)
        self.controls_layout.addWidget(self.draw_mode_checkbox)

        # --- NEW: Delete on click mode ---
        self.delete_click_mode_checkbox = QCheckBox("Delete on click")
        self.delete_click_mode_checkbox.setChecked(False)
        self.delete_click_mode_checkbox.stateChanged.connect(self.on_delete_click_mode_toggled)
        self.controls_layout.addWidget(self.delete_click_mode_checkbox)

        self.controls_layout.addStretch()

        # --- Toggle bbox labels ---
        self.show_bbox_labels = True  # default ON

        self.bbox_label_checkbox = QCheckBox("Show BBox Labels")
        self.bbox_label_checkbox.setChecked(True)
        self.bbox_label_checkbox.stateChanged.connect(self.on_bbox_label_toggle)

        self.controls_layout.addWidget(self.bbox_label_checkbox)

        # --- Object Class Selection ---
        self.controls_layout.addWidget(QLabel("Object Class"))

        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.SingleSelection)

        # Define your classes here
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
            self.class_list.addItem(item)

        # Select default class
        self.class_list.setCurrentRow(0)
        self.object_class_id = -1

        self.class_list.currentItemChanged.connect(self.on_class_changed)
        self.controls_layout.addWidget(self.class_list)

        # Assign/clear buttons for per-bbox labels
        self.assign_class_btn = QPushButton("Assign class to selected BBox")
        self.assign_class_btn.clicked.connect(self.on_assign_class_to_selected_bbox)
        self.controls_layout.addWidget(self.assign_class_btn)

        self.clear_class_btn = QPushButton("Clear label for selected BBox (back to Auto)")
        self.clear_class_btn.clicked.connect(self.on_clear_class_for_selected_bbox)
        self.controls_layout.addWidget(self.clear_class_btn)

        # --- Sample Context Selection ---
        self.controls_layout.addWidget(QLabel("Sample Context"))

        self.context_combo = QComboBox()
        self.context_combo.addItems([
            "None (no expectation)",
            "Pure sample (1 class expected)",
            "Contaminated (2+ classes expected)",
            "Environmental (class + background)"
        ])
        self.context_combo.setCurrentIndex(0)
        self.context_combo.currentIndexChanged.connect(self.on_context_changed)

        self.controls_layout.addWidget(self.context_combo)

        # Context info label
        self.context_info_label = QLabel("")
        self.context_info_label.setWordWrap(True)
        self.context_info_label.setStyleSheet("color: #aaaaaa; font-size: 10px; padding: 5px;")
        self.controls_layout.addWidget(self.context_info_label)

        # --------------------- CENTER GRID (3x2) ---------------------
        self.center_widget = QWidget()
        center_layout = QVBoxLayout(self.center_widget)

        # --- NEW: log/status banner above images ---
        self.log_label = QLabel("Awaiting: (no image loaded)")
        self.log_label.setStyleSheet(
            "background-color: #1f1f1f;"
            "color: #dddddd;"
            "padding: 6px;"
            "border: 1px solid #444;"
        )
        self.log_label.setWordWrap(True)
        center_layout.addWidget(self.log_label)

        # --- Progress label & bar ---
        self.progress_label = QLabel("Image 0 / 0 — File: None — Boxes: 0")
        center_layout.addWidget(self.progress_label)

        self.progress_bar_live = QProgressBar()
        self.progress_bar_live.setRange(0, 100)
        center_layout.addWidget(self.progress_bar_live)

        # ---------------- LEFT RAW IMAGE ----------------
        self.left_raw_scene = QGraphicsScene()
        self.left_raw_view = QGraphicsView(self.left_raw_scene)
        self.left_raw_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_raw_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_raw_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ---------------- LEFT BINARY IMAGE ----------------
        self.left_bin_scene = QGraphicsScene()
        self.left_bin_view = QGraphicsView(self.left_bin_scene)
        self.left_bin_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_bin_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_bin_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # -------- Placeholder (top row) --------
        self.placeholder_top = QLabel("Select a BBox to preview")
        self.placeholder_top.setAlignment(Qt.AlignCenter)
        self.placeholder_top.setMinimumHeight(500)
        self.placeholder_top.setStyleSheet(
            "background-color: #2b2b2b; border: 1px dashed #555;"
        )
        self.placeholder_top.setScaledContents(False)

        # ---------------- PARTICLE AREA PLOT ----------------
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("k")
        self.plot_widget.setLabel("bottom", "Area (pixels)")
        self.plot_widget.setLabel("left", "Count")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.area_curve = self.plot_widget.plot(stepMode=True, fillLevel=0, brush=(0, 150, 255, 80))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        # raw + binary
        grid.addWidget(self.left_raw_view, 0, 0)
        grid.addWidget(self.left_bin_view, 0, 1)

        # placeholders
        grid.addWidget(self.placeholder_top, 1, 0, 1, 2)

        # plot
        grid.addWidget(self.plot_widget, 2, 0, 1, 2)

        center_layout.addLayout(grid)
        center_layout.addStretch()  # IMPORTANT
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        main_layout.addWidget(self.center_widget, 1)

        # ---------------- RIGHT PREVIEW (MAIN IMAGE) ----------------
        self.right_preview_scene = QGraphicsScene()
        self.right_preview_scene.selectionChanged.connect(self.on_scene_selection_changed)

        self.right_preview_view = DrawableGraphicsView(self.right_preview_scene)
        self.right_preview_view.rectCreated.connect(self.on_user_rect_created)
        self.right_preview_view.bboxClicked.connect(self.on_bbox_clicked_for_delete)  # NEW
        self.right_preview_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_preview_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_preview_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Pixmap item (lives inside the scene)
        self.right_pix = QGraphicsPixmapItem()
        self.right_preview_scene.addItem(self.right_pix)

        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)

        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.right_preview_view)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        main_layout.addWidget(self.right_widget, 2)

        # ------------------- set dark background and remove frame -------------------
        panel_color = QColor("#2b2b2b")  # matches your GUI panel background

        for view in [self.left_raw_view, self.left_bin_view, self.right_preview_view]:
            view.setBackgroundBrush(panel_color)
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
            return

        if self.selected_bbox_key is None:
            self.placeholder_top.setText("Select a BBox to preview")
            self.placeholder_top.setPixmap(QPixmap())
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
        except Exception as e:
            self.placeholder_top.setText(f"Preview error: {e}")
            self.placeholder_top.setPixmap(QPixmap())

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
            item.setPen(QPen(Qt.red if i == self.selected_bbox_idx else Qt.green, 2))

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
            rect.setPen(QPen(Qt.red if i == self.selected_bbox_idx else Qt.green, 2))

        self.update_crop_preview()

    def _reset_per_image_state(self):
        self.user_bboxes.clear()
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

        self.update_display()

    # -------------------- Controls --------------------

    def on_min_area_changed(self, val):
        self.min_area_live = val
        self.update_display()

    def on_otsu_offset_changed(self, val):
        self.otsu_offset = val
        self.otsu_input.setText(str(val))
        # Invalidate cache
        key = self.img_paths[self.current_idx]
        if key in self.cached_binaries:
            del self.cached_binaries[key]
        self.update_display()

    def on_otsu_input(self):
        try:
            val = int(self.otsu_input.text())
            self.otsu_offset = val
            self.otsu_slider.setValue(val)
            key = self.img_paths[self.current_idx]
            if key in self.cached_binaries:
                del self.cached_binaries[key]
            self.update_display()
        except ValueError:
            self.otsu_input.setText(str(self.otsu_offset))

    def on_method_changed(self, text):
        self.method = text
        # Invalidate cache for current image
        key = self.img_paths[self.current_idx]
        if key in self.cached_binaries:
            del self.cached_binaries[key]
        self.update_display()

    def on_yolo_toggled(self, state):
        self.use_yolo = state == Qt.Checked
        # Invalidate cache for current image
        key = self.img_paths[self.current_idx]
        if key in self.cached_binaries:
            del self.cached_binaries[key]

        if not self.use_yolo and hasattr(self, "fusion_checkbox"):
            self.use_fusion_active = False
            self.fusion_checkbox.blockSignals(True)
            self.fusion_checkbox.setChecked(False)
            self.fusion_checkbox.blockSignals(False)

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
            item.setPen(QPen(Qt.red if i == self.selected_bbox_idx else Qt.green, 2))

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

    def on_bbox_clicked_for_delete(self, bbox_tuple):
        """
        NEW: Called when a bbox is clicked in delete-on-click mode.
        Immediately deletes the bbox.
        """
        if not self._delete_click_mode:
            return

        if bbox_tuple in self.bbox_list:
            self.deleted_bboxes.add(bbox_tuple)

            # If this was the selected bbox, clear selection
            if self.selected_bbox_key == bbox_tuple:
                self.selected_bbox_idx = -1
                self.selected_bbox_key = None

            # Immediately update display to show deletion
            self.update_display()

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
            cached = db.get_cached_detections(image_id, "yolo", conf=0.25)
            if cached is not None:
                # Reconstruct lightweight det dicts from stored data
                self.cached_yolo[key] = (cached, None)
                return cached, None

        dets, results = yolo_detect(self.current_img_gray, conf=0.25)

        # Persist to DB (store as list of dicts, drop the ultralytics Results object)
        if image_id is not None:
            serialisable = [
                {"bbox_rc": list(d["bbox_rc"]), "conf": float(d["conf"]), "cls_id": int(d["cls_id"])}
                for d in dets
            ]
            db.cache_detections(image_id, "yolo", serialisable, conf=0.25)

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

    # -------------------- Particle areas --------------------
    def get_particle_areas(self, binary_raw):
        labeled = measure.label(binary_raw > 0, connectivity=2)
        counts = np.bincount(labeled.ravel())[1:]  # skip background
        return counts[counts >= self.min_area_live]

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
            labeled = measure.label(binary_raw, connectivity=2)
            all_boxes = [
                tuple(int(v) for v in r.bbox)
                for r in measure.regionprops(labeled)
                if r.area >= self.min_area_live
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

        # ------------------- Update sidebar BBox info list -------------------
        self.bbox_info_list.blockSignals(True)
        self.bbox_info_list.clear()

        selected_class = int(getattr(self, "object_class_id", -1))

        for i, bbox in enumerate(self.bbox_list):
            manual_cls = self.bbox_manual_class.get(bbox, None)

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

                label = f"{i + 1}. {self._class_name(shown_cls)} ({shown_conf:.2f})"
            else:
                label = f"{i + 1}."

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, i)
            if self.use_yolo:
                item.setForeground(self._class_color(shown_cls))
            self.bbox_info_list.addItem(item)

        self.bbox_info_list.setCurrentRow(self.selected_bbox_idx if self.selected_bbox_idx >= 0 else -1)
        self.bbox_info_list.blockSignals(False)

        # ------------------- Update progress bar & label -------------------
        total_images = len(self.img_paths)
        current_image_idx = self.current_idx + 1  # 1-based index
        current_file = self.img_name
        num_boxes = len(self.bbox_list)

        self.progress_label.setText(
            f"Image {current_image_idx} / {total_images} — File: {current_file} — Boxes: {num_boxes}"
        )
        self.progress_bar_live.setValue(int(current_image_idx / total_images * 100))

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

        # ------------------- LEFT IMAGES (raw + binary) -------------------
        for img, pix_item, view in zip(
                [self.current_img_gray, (binary_raw > 0).astype(np.uint8) * 255],
                [self.left_raw_pix, self.left_bin_pix],
                [self.left_raw_view, self.left_bin_view]
        ):
            disp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w = disp_img.shape[:2]
            qimg = QImage(disp_img.data, w, h, disp_img.strides[0], QImage.Format_RGB888)
            pix_item.setPixmap(QPixmap.fromImage(qimg))

            # Hide scrollbars
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            # Scene rect = image size
            view.scene().setSceneRect(0, 0, w, h)

            # Fit in view, keep aspect ratio
            view.fitInView(pix_item, Qt.KeepAspectRatio)

        # ------------------- RIGHT PREVIEW -------------------
        vis = cv2.cvtColor(self.current_img_gray, cv2.COLOR_GRAY2RGB)
        h, w = vis.shape[:2]
        qimg = QImage(vis.data, w, h, vis.strides[0], QImage.Format_RGB888)
        self.right_pix.setPixmap(QPixmap.fromImage(qimg))

        # Hide scrollbars
        self.right_preview_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_preview_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Scene rect = image size
        self.right_preview_scene.setSceneRect(0, 0, w, h)

        # Fit vertically, keep aspect ratio
        self.right_preview_view.fitInView(self.right_pix, Qt.KeepAspectRatio)

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

            base_color = self._class_color(vis_cls) if self.use_yolo else QColor("#00ff00")

            rect = QGraphicsRectItem(min_c, min_r, max_c - min_c, max_r - min_r)
            rect.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
            rect.setData(0, bbox)
            rect.setAcceptedMouseButtons(Qt.LeftButton)

            if i == self.selected_bbox_idx or bbox == self.selected_bbox_key:
                rect.setPen(QPen(Qt.red, 3))
            else:
                rect.setPen(QPen(base_color, 2))

            rect.setBrush(QBrush(Qt.NoBrush))
            self.right_preview_scene.addItem(rect)
            self.bbox_items.append(rect)

            if self.show_bbox_labels:
                text_item = QGraphicsTextItem(f"{i + 1}")
                font = text_item.font()
                font.setPointSize(14)
                font.setBold(True)
                text_item.setFont(font)
                text_item.setDefaultTextColor(Qt.white if i != self.selected_bbox_idx else Qt.red)
                text_item.setPos(min_c, max(0, min_r - 20))
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
            # Fall back to sibling of output_root (data/complete/ lives next to the class folder)
            dataset_root = self.output_root.parent
        if not dataset_root.exists():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Dataset not found",
                                f"Could not locate dataset root:\n{dataset_root}")
            return
        self._reviewer = DatasetReviewer(
            dataset_root=dataset_root,
            class_names=self.class_definitions,
            parent=None,
        )
        self._reviewer.show()

    # -------------------- Keyboard navigation --------------------
    def keyPressEvent(self, event):
        key = event.key()

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

            # Move raw image to raw_out folder
            move(img_path, Path(self.raw_out) / img_path.name)

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

        elif key == Qt.Key_Delete:
            if self.selected_bbox_key is not None:
                self.on_delete_bbox()
            return

        # NEW: Toggle delete-on-click mode with 'D' key
        elif key == ord('D'):
            current = self.delete_click_mode_checkbox.isChecked()
            self.delete_click_mode_checkbox.setChecked(not current)
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

        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        imgs = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        imgs = sorted(imgs)

        if not imgs:
            QMessageBox.warning(self, "No images", "No images found (.tif/.tiff/.png/.jpg/.jpeg).")
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

    colors = load_json(colors_path)
    style = f"""
    QMainWindow {{
        background-color: {colors['window_bg']};
        color: {colors['text_color']};
    }}
    QDockWidget {{
        background-color: {colors['panel_bg']};
    }}
    QLabel, QLineEdit, QSpinBox {{
        color: {colors['text_color']};
        background-color: {colors['panel_bg']};
    }}
    QPushButton {{
        background-color: {colors['button_bg']};
        color: {colors['button_text']};
        border: 1px solid {colors['highlight']};
        padding: 3px;
    }}
    QCheckBox {{
        color: {colors['text_color']};
    }}
    QComboBox {{
        background-color: {colors['dropdown_bg']};
        color: {colors['dropdown_text']};
    }}
    QSlider::groove:horizontal {{
        background: {colors['slider_bg']};
        height: 6px;
    }}
    QSlider::handle:horizontal {{
        background: {colors['highlight']};
        width: 12px;
    }}
    QProgressBar {{
        text-align: center;
        color: {colors['text_color']};
        border: 1px solid {colors['highlight']};
        background-color: {colors['panel_bg']};
    }}
    QProgressBar::chunk {{
        background-color: {colors['highlight']};
    }}
    """
    app.setStyleSheet(style)

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

    # # ---------------------------------------------------------------------
    # # === MODELS ===
    # # ---------------------------------------------------------------------
    YOLO_MODEL_PATH = r"C:/Users/joshk/OneDrive/Documents/GitHub_Strath/PolyVision/models/detect/YOLOv8.1/best.pt"
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ENABLE_FUSION = True  # Set to False to disable fusion
    LOCAL_MODEL_PATH = r"C:/Users/joshk/OneDrive/Documents/GitHub_Strath/PolyVision/models/local/EfficientNetB0/best_model.keras"
    GLOBAL_MODEL_PATH = r"C:/Users/joshk/OneDrive/Documents/GitHub_Strath/PolyVision/models/global/EfficientNetB0/best_model.keras"
    FUSION_WEIGHTS = (0.3, 0.5, 0.2)  # (YOLO, Local, Global)
    #
    # # ---------------------------------------------------------------------

    # Load configuration
    repo_root = find_repo_root(Path(__file__))

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

    # 1 Input folder containing TIFF images
    input_dir = Path(
        r"C:\Users\joshk\OneDrive\Desktop\raw\testset\ps1_et")
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    img_list = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not img_list:
        print(f"No image files found in {input_dir} (expected .tif/.tiff/.png/.jpg/.jpeg)")
        sys.exit(1)

    output_root = Path(r"C:\Users\joshk\OneDrive\Desktop\raw\testset\ps1_et\ps")
    raw_out = Path(r"C:\Users\joshk\OneDrive\Desktop\raw\testset\ps1_et\ps\whole_images")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)

    # 4️⃣ Start PyQt5 application
    app = QApplication(sys.argv)

    with open("C:/Users/joshk/OneDrive/Documents/GitHub_Strath/PolyVision/configs/gui_colors.json", "r") as f:
        colors = json.load(f)

    style = f"""
    QMainWindow {{
        background-color: {colors['window_bg']};
        color: {colors['text_color']};
    }}

    QDockWidget {{
        background-color: {colors['panel_bg']};
    }}

    QLabel, QLineEdit, QSpinBox {{
        color: {colors['text_color']};
        background-color: {colors['panel_bg']};
    }}

    QPushButton {{
        background-color: {colors['button_bg']};
        color: {colors['button_text']};
        border: 1px solid {colors['highlight']};
        padding: 3px;
    }}

    QCheckBox {{
        color: {colors['text_color']};
    }}

    QComboBox {{
        background-color: {colors['dropdown_bg']};
        color: {colors['dropdown_text']};
    }}

    QSlider::groove:horizontal {{
        background: {colors['slider_bg']};
        height: 6px;
    }}
    QSlider::handle:horizontal {{
        background: {colors['highlight']};
        width: 12px;
    }}

    QProgressBar {{
        text-align: center;
        color: {colors['text_color']};
        border: 1px solid {colors['highlight']};
        background-color: {colors['panel_bg']};
    }}
    QProgressBar::chunk {{
        background-color: {colors['highlight']};
    }}
    """

    app.setStyleSheet(style)

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
    )
    previewer.show()

    sys.exit(app.exec_())

# python -m polyvision.app.main