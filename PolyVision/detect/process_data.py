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
from skimage.morphology import disk, binary_opening
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
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import pyqtgraph as pg

# ---------------------------------------------------------------------
# === YOLO DETECTION MODEL ===
# ---------------------------------------------------------------------
YOLO_MODEL_PATH = r"C:\Users\joshk\OneDrive\Desktop\AI Microplastics Detection\runs\detect\train5\weights\best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)


# ---------------------------------------------------------------------

def yolo_detect(img_gray, conf=0.25):
    img_8 = ensure_8bit(img_gray)
    if img_8.ndim == 2:
        img_input = prepare_for_yolo(img_8)
    elif img_8.ndim == 3 and img_8.shape[2] == 1:
        img_input = img_8
    else:
        raise ValueError(f"YOLO expects 1-channel grayscale, got {img_8.shape}")

    results = yolo_model.predict(
        source=img_input,
        conf=conf,
        verbose=False,
        device="cpu",
        imgsz=1024
    )[0]

    boxes = []
    scores = []
    if results.boxes is not None:
        for b, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, b)
            boxes.append((y1, x1, y2, x2))
            scores.append(float(c))
    return boxes, scores, results


def prepare_for_yolo(gray_img):
    if gray_img.ndim == 2:
        return gray_img[:, :, None]
    if gray_img.ndim == 3 and gray_img.shape[2] == 1:
        return gray_img
    raise ValueError("YOLO expects grayscale image")


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
        thr_bool = binary_opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, min_size=min_obj_size)

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
        thr_bool = binary_opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, min_size=min_obj_size)

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
        thr_bool = binary_opening(thr_bool, disk(1))
        thr_bool = morphology.remove_small_objects(thr_bool, min_size=min_obj_size)

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
        use_yolo: bool
):
    """Non-interactive version: uses either YOLO or threshold with fixed params."""
    img = load_as_gray(str(img_path))

    if use_yolo:
        boxes, scores, results = yolo_detect(img, conf=0.25)
        crop_folder = out_root / img_path.stem
        crop_folder.mkdir(exist_ok=True, parents=True)

        # move raw image
        move(img_path, raw_out / img_path.name)

        crops_meta = extract_particle_crops_yolo(
            img,
            boxes,
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
    yolo_txt_path = crop_folder / f"{img_path.stem}.txt"
    with open(yolo_txt_path, "w") as f:
        for c in crops_meta:
            x, y, w, h = c["yolo"]
            f.write(f"{object_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

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
# === WORKER THREAD ===
# ---------------------------------------------------------------------

class ProcessThread(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(int)

    def __init__(self, in_dir, out_root, raw_out, object_class_id, min_area, margin, use_yolo):
        super().__init__()
        self.in_dir = Path(in_dir)
        self.out_root = Path(out_root)
        self.raw_out = Path(raw_out)
        self.object_class_id = object_class_id
        self.min_area = min_area
        self.margin = margin
        self.use_yolo = use_yolo

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
                    use_yolo=self.use_yolo
                )
                self.message.emit(f"Processed {Path(meta['image']).name}, crops: {len(meta['crops'])}")
                count += 1
            except Exception as e:
                self.message.emit(f"Error on {img_path.name}: {e}")
            self.progress.emit(int(i * 100 / n))

        self.finished_ok.emit(count)


# ---------------------------------------------------------------------
# === INTERACTIVE PREVIEW WINDOW ===
# ---------------------------------------------------------------------

class ImagePreview(QMainWindow):
    def __init__(self, img_paths, output_root, raw_out, object_class_id=1, use_yolo=False, min_area=50, parent=None):
        super().__init__(parent)
        self.object_class_id = object_class_id
        self.setWindowTitle("Interactive Preview")
        self.setWindowState(Qt.WindowFullScreen)

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

        # Settings
        self.use_yolo = use_yolo
        self.min_area_live = min_area
        self.otsu_offset = 0
        self.method = "otsu"
        self.object_bright = True
        self.adaptive_block_size = 21

        # Cache
        self.cached_binaries = {}
        self.cached_yolo = {}
        self.cached_scaled_img = {}

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

        # --- Progress label & bar ---
        self.progress_label = QLabel("Image 0 / 0 — File: None — Boxes: 0")
        self.controls_layout.addWidget(self.progress_label)

        self.progress_bar_live = QProgressBar()
        self.progress_bar_live.setRange(0, 100)
        self.controls_layout.addWidget(self.progress_bar_live)

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

        self.controls_layout.addStretch()

        self.controls_widget.setFixedWidth(260)
        main_layout.addWidget(self.controls_widget, 0)

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
        self.class_definitions = [
            (0, "Nylon"),
            (1, "PE"),
            (2, "PET"),
            (3, "PLA"),
            (4, "PMMA"),
            (5, "PP"),
            (6, "PS"),
            (7, "PU"),
            (8, "PVC"),
        ]

        for class_id, name in self.class_definitions:
            item = QListWidgetItem(f"{class_id} — {name}")
            item.setData(Qt.UserRole, class_id)
            self.class_list.addItem(item)

        # Select default class
        self.class_list.setCurrentRow(0)
        self.object_class_id = self.class_definitions[0][0]

        self.class_list.currentItemChanged.connect(self.on_class_changed)

        self.controls_layout.addWidget(self.class_list)

        # --------------------- CENTER GRID (3x2) ---------------------
        self.center_widget = QWidget()
        center_layout = QVBoxLayout(self.center_widget)

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
        self.placeholder_top = QLabel("Placeholder")
        self.placeholder_top.setAlignment(Qt.AlignCenter)
        self.placeholder_top.setMinimumHeight(500)
        self.placeholder_top.setStyleSheet(
            "background-color: #2b2b2b; border: 1px dashed #555;"
        )

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

        # --------------------- RIGHT BBOX IMAGE ---------------------
        # ---------------- RIGHT PREVIEW (MAIN IMAGE) ----------------
        self.right_preview_scene = QGraphicsScene()
        self.right_preview_scene.selectionChanged.connect(self.on_scene_selection_changed)

        self.right_preview_view = QGraphicsView(self.right_preview_scene)
        self.right_preview_view.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
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

        # Force an initial update after the event loop runs
        QTimer.singleShot(50, self.update_display)

    def on_scene_selection_changed(self):
        selected_items = self.right_preview_scene.selectedItems()

        if not selected_items:
            self.selected_bbox_key = None
        else:
            bbox_key = selected_items[0].data(0)
            self.selected_bbox_key = bbox_key if bbox_key is not None else None

        # Derive index from key (so dropdown + red highlight work)
        if self.selected_bbox_key in getattr(self, "bbox_list", []):
            self.selected_bbox_idx = self.bbox_list.index(self.selected_bbox_key)
        else:
            self.selected_bbox_idx = -1

        # Sync dropdown safely
        self.bbox_dropdown.blockSignals(True)
        self.bbox_dropdown.setCurrentIndex(self.selected_bbox_idx if self.selected_bbox_idx >= 0 else -1)
        self.bbox_dropdown.blockSignals(False)

        # Update rectangle colors
        for i, rect in enumerate(self.bbox_items):
            rect.setPen(QPen(Qt.red if i == self.selected_bbox_idx else Qt.green, 2))

    # -------------------- Image Loading --------------------
    def load_current_image(self):
        self.current_idx = max(0, min(self.current_idx, len(self.img_paths) - 1))
        self.img_name = self.img_paths[self.current_idx].name
        self.deleted_bboxes.clear()

        # Load grayscale image
        if self.img_paths[self.current_idx] not in self.cached_scaled_img:
            img_gray = load_as_gray(str(self.img_paths[self.current_idx]))
            self.current_img_gray = ensure_8bit(img_gray)
            self.cached_scaled_img[self.img_paths[self.current_idx]] = self.current_img_gray
        else:
            self.current_img_gray = self.cached_scaled_img[self.img_paths[self.current_idx]]

        self.selected_bbox_idx = -1
        self.selected_bbox_key = None
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
            self.object_class_id = current.data(Qt.UserRole)

    def on_bbox_label_toggle(self, state):
        self.show_bbox_labels = state == Qt.Checked
        for text_item in self.text_items:
            text_item.setVisible(self.show_bbox_labels)

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

        boxes, scores, results = yolo_detect(self.current_img_gray, conf=0.25)
        self.cached_yolo[key] = (boxes, scores, results)
        return boxes, scores, results

    # -------------------- Particle areas --------------------
    def get_particle_areas(self, binary_raw):
        labeled = measure.label(binary_raw > 0, connectivity=2)
        counts = np.bincount(labeled.ravel())[1:]  # skip background
        return counts[counts >= self.min_area_live]

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
            boxes, scores, results = self.get_yolo_results()
            all_boxes = [tuple(int(v) for v in box) for box in boxes]

            keep = [i for i, box in enumerate(all_boxes) if box not in self.deleted_bboxes]
            self.bbox_list = [all_boxes[i] for i in keep]
            self.last_yolo_scores = [scores[i] for i in keep]
        else:
            labeled = measure.label(binary_raw, connectivity=2)
            all_boxes = [
                tuple(int(v) for v in r.bbox)
                for r in measure.regionprops(labeled)
                if r.area >= self.min_area_live
            ]

            self.bbox_list = [box for box in all_boxes if box not in self.deleted_bboxes]
            self.last_yolo_scores = [0.0] * len(self.bbox_list)

        # Reconcile selection after rebuild
        if self.selected_bbox_key in self.bbox_list:
            self.selected_bbox_idx = self.bbox_list.index(self.selected_bbox_key)
        else:
            self.selected_bbox_key = None
            self.selected_bbox_idx = -1

        # ------------------- Update BBox dropdown -------------------
        self.bbox_dropdown.blockSignals(True)  # prevent triggering on_bbox_selected
        self.bbox_dropdown.clear()
        for i in range(len(self.bbox_list)):
            conf_text = f" ({self.last_yolo_scores[i]:.2f})" if self.use_yolo else ""
            self.bbox_dropdown.addItem(f"BBox {i + 1}{conf_text}")
        self.bbox_dropdown.setCurrentIndex(self.selected_bbox_idx if self.selected_bbox_idx >= 0 else -1)
        self.bbox_dropdown.blockSignals(False)

        # ------------------- Update progress bar & label -------------------
        total_images = len(self.img_paths)
        current_image_idx = self.current_idx + 1  # 1-based index
        current_file = self.img_name
        num_boxes = len(self.bbox_list)

        self.progress_label.setText(
            f"Image {current_image_idx} / {total_images} — File: {current_file} — Boxes: {num_boxes}"
        )
        self.progress_bar_live.setValue(int(current_image_idx / total_images * 100))

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
            rect = QGraphicsRectItem(min_c, min_r, max_c - min_c, max_r - min_r)
            rect.setFlag(QGraphicsRectItem.ItemIsSelectable, True)

            # Store bbox tuple as the selection/deletion key
            rect.setData(0, bbox)

            rect.setAcceptedMouseButtons(Qt.LeftButton)
            rect.setPen(QPen(Qt.red if i == self.selected_bbox_idx else Qt.green, 2))
            rect.setBrush(QBrush(Qt.NoBrush))
            self.right_preview_scene.addItem(rect)
            self.bbox_items.append(rect)

            # Text label
            if self.show_bbox_labels:
                conf_text = f" ({self.last_yolo_scores[i]:.2f})" if self.use_yolo else ""
                text_item = QGraphicsTextItem(f"{i + 1}{conf_text}")
                font = text_item.font()
                font.setPointSize(14)
                font.setBold(True)
                text_item.setFont(font)
                text_item.setDefaultTextColor(Qt.red)
                text_item.setPos(min_c, max(0, min_r - 20))
                self.right_preview_scene.addItem(text_item)
                self.text_items.append(text_item)

    # -------------------- Keyboard navigation --------------------
    def keyPressEvent(self, event):
        key = event.key()

        # -------------------- Skip to next image --------------------
        if key == Qt.Key_Escape:
            if self.current_idx + 1 < len(self.img_paths):
                self.next_image()
            else:
                # Last image — do nothing
                self.statusBar().showMessage("Last image reached. Press Enter to save or close manually.")
            return

        # -------------------- Save & Next --------------------
        elif key in (Qt.Key_Return, Qt.Key_Enter):
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

            # --- Save YOLO labels (match the SQUARE crops) ---
            yolo_txt_path = crop_folder / f"{img_path.stem}.txt"
            with open(yolo_txt_path, "w") as f:
                for (min_r, min_c, max_r, max_c) in squared_bboxes:
                    box_w = max_c - min_c
                    box_h = max_r - min_r
                    x_center = (min_c + box_w / 2) / W
                    y_center = (min_r + box_h / 2) / H
                    norm_w = box_w / W
                    norm_h = box_h / H
                    f.write(
                        f"{self.object_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
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

        previewer = ImagePreview(
            img_paths=imgs,
            output_root=out_dir,
            raw_out=raw_dir,
            use_yolo=self.use_yolo_checkbox.isChecked(),
            min_area=self.min_area_spin.value(),
            parent=self
        )
        previewer.show()

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


# ---------------------------------------------------------------------
# === RUN ===
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import sys
    from PyQt5.QtWidgets import QApplication

    # 1️⃣ Input folder containing TIFF images
    input_dir = Path(
        r"I:\Science\Chemistry\RDMS\Juliane Simmchen group\Josh\2. Microplastic AI Project\datasets\raw\pe\pe2_et")
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    img_list = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not img_list:
        print(f"No image files found in {input_dir} (expected .tif/.tiff/.png/.jpg/.jpeg)")
        sys.exit(1)

    # 2️⃣ Output folder for cropped particles
    output_root = Path(
        r"I:\Science\Chemistry\RDMS\Juliane Simmchen group\Josh\2. Microplastic AI Project\datasets\raw\pe\pe2_et")
    output_root.mkdir(parents=True, exist_ok=True)

    # 3️⃣ Folder to move raw images after processing
    raw_out = Path(
        r"I:\Science\Chemistry\RDMS\Juliane Simmchen group\Josh\2. Microplastic AI Project\datasets\raw\pe\pe2_et\whole_images")
    raw_out.mkdir(parents=True, exist_ok=True)

    # 4️⃣ Start PyQt5 application
    app = QApplication(sys.argv)

    with open("gui_colors.json", "r") as f:
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
        use_yolo=False,  # <-- explicitly pass
        min_area=50,
    )
    previewer.show()

    sys.exit(app.exec_())
