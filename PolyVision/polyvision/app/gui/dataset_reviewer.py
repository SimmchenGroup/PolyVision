"""
Dataset Reviewer — browse, edit, and delete annotated images.

Launch from within the annotation GUI via the "Review Dataset" button,
or standalone:
    python -m polyvision.app.gui.dataset_reviewer  <dataset_root>
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPen, QBrush, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QShortcut, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
    QGraphicsPixmapItem, QCheckBox,
)

from polyvision.core.geometry import square_bbox
from polyvision.core.image_io import ensure_8bit, load_as_gray

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── minimal drawable view (mirrors DrawableGraphicsView in main.py) ───────────

class _DrawableView(QGraphicsView):
    rectCreated = pyqtSignal(tuple)   # (min_r, min_c, max_r, max_c)
    bboxClicked = pyqtSignal(int)     # delete-mode click: index into self._boxes
    bboxSelected = pyqtSignal(int)    # normal-mode click: index into self._boxes
    zoomed = pyqtSignal()             # emitted on any wheel zoom

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._draw_mode = False
        self._delete_mode = False
        self._dragging = False
        self._start = None
        self._rubber = None

    def set_draw_mode(self, on: bool):
        self._draw_mode = on
        if not on:
            self._dragging = False
            if self._rubber and self.scene():
                self.scene().removeItem(self._rubber)
            self._rubber = None

    def set_delete_mode(self, on: bool):
        self._delete_mode = on

    def mousePressEvent(self, event):
        if self._delete_mode and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            # IntersectsItemBoundingRect makes the full box interior clickable.
            # The default IntersectsItemShape only hits the outline for unfilled rects.
            items = self.scene().items(pos, Qt.IntersectsItemBoundingRect)
            for item in items:
                if isinstance(item, QGraphicsRectItem) and item.data(0) is not None:
                    self.bboxClicked.emit(int(item.data(0)))
                    event.accept()
                    return
            event.accept()
            return

        if self._draw_mode and event.button() == Qt.LeftButton:
            self._dragging = True
            self._start = self.mapToScene(event.pos())
            pen = QPen(QColor("#ffffff"), 2, Qt.DashLine)
            self._rubber = QGraphicsRectItem()
            self._rubber.setPen(pen)
            self._rubber.setBrush(QBrush(Qt.NoBrush))
            self._rubber.setZValue(10_000)
            self.scene().addItem(self._rubber)
            self._rubber.setRect(QRectF(self._start, self._start))
            event.accept()
            return

        # Normal mode: select the clicked box
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            items = self.scene().items(pos, Qt.IntersectsItemBoundingRect)
            for item in items:
                if isinstance(item, QGraphicsRectItem) and item.data(0) is not None:
                    self.bboxSelected.emit(int(item.data(0)))
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._draw_mode and self._dragging and self._rubber and self._start:
            self._rubber.setRect(QRectF(self._start, self.mapToScene(event.pos())).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draw_mode and self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            if self._rubber is None:
                return
            rect = self._rubber.rect().normalized()
            self.scene().removeItem(self._rubber)
            self._rubber = None
            if rect.width() < 5 or rect.height() < 5:
                event.accept()
                return
            self.rectCreated.emit((
                int(round(rect.top())),
                int(round(rect.left())),
                int(round(rect.bottom())),
                int(round(rect.right())),
            ))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.zoomed.emit()
        event.accept()


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_labels(txt_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Parse YOLO label file → list of (cls_id, min_r, min_c, max_r, max_c)."""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        xc, yc, nw, nh = map(float, parts[1:])
        w = nw * img_w
        h = nh * img_h
        min_c = int(round(xc * img_w - w / 2))
        min_r = int(round(yc * img_h - h / 2))
        max_c = int(round(xc * img_w + w / 2))
        max_r = int(round(yc * img_h + h / 2))
        boxes.append((cls_id, min_r, min_c, max_r, max_c))
    return boxes


def _write_labels(txt_path: Path, boxes: list[tuple], img_w: int, img_h: int):
    """Write list of (cls_id, min_r, min_c, max_r, max_c) to YOLO label file."""
    lines = []
    for cls_id, min_r, min_c, max_r, max_c in boxes:
        bw = max_c - min_c
        bh = max_r - min_r
        xc = (min_c + bw / 2) / img_w
        yc = (min_r + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _extract_crops(img_gray: np.ndarray, boxes: list[tuple], crop_folder: Path, stem: str):
    """Delete old crops and re-extract from current box list."""
    for old in crop_folder.glob(f"{stem}_*.tif"):
        old.unlink()
    for idx, (cls_id, min_r, min_c, max_r, max_c) in enumerate(boxes, start=1):
        sr, sc, er, ec = square_bbox((min_r, min_c, max_r, max_c), img_gray.shape)
        crop = img_gray[sr:er, sc:ec]
        out = crop_folder / f"{stem}_{idx:04d}.tif"
        cv2.imwrite(str(out), crop.astype(np.uint16) if crop.dtype == np.uint16 else crop)


def _load_gray_robust(path: Path) -> np.ndarray:
    """
    Load a grayscale image, trying cv2 first then PIL as fallback.
    PIL handles more TIFF variants and works around OneDrive cloud-file
    issues that cause cv2.imread to return None.
    Raises FileNotFoundError if neither reader can open the file.
    """
    # cv2 attempt
    try:
        img = load_as_gray(str(path))
        return img
    except (FileNotFoundError, Exception):
        pass

    # PIL fallback
    if _PIL_AVAILABLE:
        try:
            pil_img = _PILImage.open(str(path)).convert("L")
            return np.array(pil_img)
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not read image: {path}\n"
        "If this file is stored in OneDrive cloud-only, right-click the "
        "folder and select 'Always keep on this device', then retry."
    )


def _gray_to_pixmap(img: np.ndarray) -> QPixmap:
    img8 = ensure_8bit(img)
    h, w = img8.shape
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


# ── main window ───────────────────────────────────────────────────────────────

_BOX_COLOURS = [
    "#00ff66", "#00b3ff", "#ff6666", "#ffcc00",
    "#ff00ff", "#00ffff", "#ff8800", "#aaaaff", "#ffffff",
]


class DatasetReviewer(QMainWindow):
    """
    Browse and edit the annotated dataset in data/complete/.

    dataset_root  — path to the data/complete/ directory
    class_names   — optional list of (id, name) tuples for label display;
                    if None, uses folder names as class names
    """

    def __init__(self, dataset_root: str | Path,
                 class_names: list[tuple[int, str]] | None = None,
                 parent=None):
        super().__init__(parent)
        self.dataset_root = Path(dataset_root)
        self.class_names = {int(cid): name for cid, name in (class_names or [])}
        self.setWindowTitle("Dataset Reviewer")
        self.setWindowState(Qt.WindowMaximized)

        # State
        self._current_class: str | None = None
        self._current_stem: str | None = None
        self._img_gray: np.ndarray | None = None
        self._boxes: list[tuple] = []        # [(cls_id, min_r, min_c, max_r, max_c)]
        self._selected_idx: int | None = None
        self._reviewed: dict[str, set[str]] = {}   # class -> set of reviewed stems

        self._build_ui()
        self._populate_classes()
        self._load_progress()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── shared control widgets (created once, reparented on layout switch) ──
        self.class_list = QListWidget()
        self.class_list.currentTextChanged.connect(self._on_class_selected)

        self.image_list = QListWidget()
        self.image_list.currentTextChanged.connect(self._on_image_selected)

        self.box_count_label = QLabel("")

        self.draw_check = QCheckBox("Draw Boxes")
        self.draw_check.toggled.connect(self._on_draw_toggled)

        self.delete_check = QCheckBox("Delete on Click")
        self.delete_check.toggled.connect(self._on_delete_toggled)

        self._save_btn = QPushButton("Save  (Ctrl+S)")
        self._save_btn.clicked.connect(self._save)

        self.delete_image_btn = QPushButton("Delete Entire Image")
        self.delete_image_btn.setStyleSheet("color: #ff4444;")
        self.delete_image_btn.clicked.connect(self._delete_entire_image)

        self.status_label = QLabel("Select a class and image.")
        self.status_label.setWordWrap(True)

        self.crop_preview = QLabel("Select a box\nto preview")
        self.crop_preview.setAlignment(Qt.AlignCenter)
        self.crop_preview.setMinimumSize(200, 200)
        self.crop_preview.setStyleSheet(
            "background-color: #2b2b2b; border: 1px dashed #555; color: #888;"
        )

        # ── main image view ──
        self.scene = QGraphicsScene()
        self.view = _DrawableView(self.scene)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundBrush(QColor("#2b2b2b"))
        self.view.setStyleSheet("border: 0px;")
        self.view.rectCreated.connect(self._on_rect_created)
        self.view.bboxClicked.connect(self._on_bbox_clicked)
        self.view.bboxSelected.connect(self._on_bbox_selected)
        self.view.zoomed.connect(lambda: setattr(self, '_user_zoomed', True))
        self._user_zoomed = False

        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)

        # ── hidden holder keeps shared widgets alive across layout switches ──
        self._widget_holder = QWidget()

        # ── splitter (holds view + controls panel, orientation set in _apply_layout) ──
        self._splitter = QSplitter()
        self._is_portrait: bool | None = None   # None forces initial layout

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._splitter)
        self.setCentralWidget(root)

        self._apply_layout(portrait=False)

        # keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save)
        QShortcut(QKeySequence("Delete"), self).activated.connect(self._delete_selected_box)
        QShortcut(QKeySequence("Backspace"), self).activated.connect(self._delete_selected_box)

    # ── layout switching ──────────────────────────────────────────────────────

    def _make_landscape_panel(self) -> QWidget:
        """Vertical controls strip for landscape (left side)."""
        panel = QWidget()
        panel.setFixedWidth(260)
        lo = QVBoxLayout(panel)
        lo.setContentsMargins(6, 6, 6, 6)
        lo.setSpacing(5)
        lo.addWidget(QLabel("Class"))
        lo.addWidget(self.class_list)
        lo.addWidget(QLabel("Image"))
        lo.addWidget(self.image_list)
        lo.addWidget(self.box_count_label)
        lo.addWidget(self.draw_check)
        lo.addWidget(self.delete_check)
        lo.addWidget(self._save_btn)
        lo.addStretch()
        lo.addWidget(self.delete_image_btn)
        lo.addWidget(self.status_label)
        return panel

    def _make_portrait_panel(self) -> QWidget:
        """Horizontal controls strip for portrait (bottom)."""
        panel = QWidget()
        panel.setMaximumHeight(220)
        lo = QHBoxLayout(panel)
        lo.setContentsMargins(6, 6, 6, 6)
        lo.setSpacing(10)

        class_col = QVBoxLayout()
        class_col.addWidget(QLabel("Class"))
        class_col.addWidget(self.class_list)
        lo.addLayout(class_col, stretch=1)

        img_col = QVBoxLayout()
        img_col.addWidget(QLabel("Image"))
        img_col.addWidget(self.image_list)
        lo.addLayout(img_col, stretch=2)

        btn_col = QVBoxLayout()
        btn_col.addWidget(self.box_count_label)
        btn_col.addWidget(self.draw_check)
        btn_col.addWidget(self.delete_check)
        btn_col.addWidget(self._save_btn)
        btn_col.addStretch()
        btn_col.addWidget(self.delete_image_btn)
        btn_col.addWidget(self.status_label)
        lo.addLayout(btn_col, stretch=1)
        return panel

    def _make_image_area(self) -> QSplitter:
        """Main image on top, box preview below — always a vertical split."""
        inner = QSplitter(Qt.Vertical)
        inner.addWidget(self.view)
        inner.addWidget(self.crop_preview)
        h = inner.height() or 900
        inner.setSizes([int(h * 0.65), int(h * 0.35)])
        return inner

    def _apply_layout(self, portrait: bool):
        if portrait == self._is_portrait:
            return
        self._is_portrait = portrait

        # Rescue all shared/movable widgets before Qt destroys the old containers
        for w in (self.class_list, self.image_list, self.box_count_label,
                  self.draw_check, self.delete_check, self._save_btn,
                  self.delete_image_btn, self.status_label,
                  self.crop_preview, self.view):
            w.setParent(self._widget_holder)

        # Tear down old containers
        while self._splitter.count():
            self._splitter.widget(0).setParent(None)

        if portrait:
            # Top: image + preview stacked vertically; bottom: controls strip
            self._splitter.setOrientation(Qt.Vertical)
            self._splitter.addWidget(self._make_image_area())
            self._splitter.addWidget(self._make_portrait_panel())
            h = self._splitter.height() or 900
            self._splitter.setSizes([int(h * 0.80), int(h * 0.20)])
        else:
            # Left: controls; right: image + preview stacked vertically
            self._splitter.setOrientation(Qt.Horizontal)
            self._splitter.addWidget(self._make_landscape_panel())
            self._splitter.addWidget(self._make_image_area())

    # ── population ────────────────────────────────────────────────────────────

    def _populate_classes(self):
        self.class_list.clear()
        for d in sorted(self.dataset_root.iterdir()):
            if d.is_dir() and d.name != "whole_images":
                self.class_list.addItem(d.name)

    def _populate_images(self, class_name: str):
        self.image_list.clear()
        class_dir = self.dataset_root / class_name
        all_stems = sorted(
            d.name for d in class_dir.iterdir()
            if d.is_dir() and d.name != "whole_images"
        )
        done = self._reviewed.get(class_name, set())
        pending = [s for s in all_stems if s not in done]
        for stem in pending:
            self.image_list.addItem(stem)
        reviewed_count = len(done)
        total = len(all_stems)
        self._set_status(
            f"{len(pending)} remaining, {reviewed_count}/{total} reviewed."
        )

    # ── selection handlers ────────────────────────────────────────────────────

    def _on_class_selected(self, class_name: str):
        if not class_name:
            return
        self._current_class = class_name
        self._current_stem = None
        self._populate_images(class_name)
        self._clear_view()

    def _on_image_selected(self, stem: str):
        if not stem or not self._current_class:
            return
        self._current_stem = stem
        self._load_image_and_labels()
        self._save_progress()

    # ── image loading ─────────────────────────────────────────────────────────

    def _whole_image_path(self) -> Path | None:
        if not self._current_class or not self._current_stem:
            return None
        p = self.dataset_root / self._current_class / "whole_images" / f"{self._current_stem}.tif"
        if not p.exists():
            # try other extensions
            for ext in (".tiff", ".png", ".jpg", ".jpeg"):
                q = p.with_suffix(ext)
                if q.exists():
                    return q
        return p if p.exists() else None

    def _crop_folder(self) -> Path | None:
        if not self._current_class or not self._current_stem:
            return None
        return self.dataset_root / self._current_class / self._current_stem

    def _label_path(self) -> Path | None:
        folder = self._crop_folder()
        if folder is None:
            return None
        return folder / f"{self._current_stem}.txt"

    def _load_image_and_labels(self):
        img_path = self._whole_image_path()
        if img_path is None:
            self._set_status(
                f"Whole image not found for '{self._current_stem}'.\n"
                f"Expected: {self.dataset_root / self._current_class / 'whole_images' / (self._current_stem + '.tif')}"
            )
            self._clear_view()
            return

        try:
            img = _load_gray_robust(img_path)
        except FileNotFoundError as e:
            self._set_status(str(e))
            self._clear_view()
            return

        self._img_gray = ensure_8bit(img)
        h, w = self._img_gray.shape

        lbl = self._label_path()
        self._boxes = _read_labels(lbl, w, h) if lbl else []
        self._selected_idx = None
        self._user_zoomed = False

        self._render()
        self._set_status(f"{self._current_stem}  —  {len(self._boxes)} box(es)")

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self):
        self.scene.clear()
        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)

        if self._img_gray is None:
            return

        self._pix_item.setPixmap(_gray_to_pixmap(self._img_gray))
        h, w = self._img_gray.shape
        self.scene.setSceneRect(0, 0, w, h)
        self.view.fitInView(self._pix_item, Qt.KeepAspectRatio)

        for i, (cls_id, min_r, min_c, max_r, max_c) in enumerate(self._boxes):
            colour = QColor(_BOX_COLOURS[cls_id % len(_BOX_COLOURS)])
            pen = QPen(colour, 2)
            rect_item = QGraphicsRectItem(min_c, min_r, max_c - min_c, max_r - min_r)
            rect_item.setPen(pen)
            # Transparent fill so the full interior registers for click hit-testing
            rect_item.setBrush(QBrush(QColor(255, 255, 255, 0)))
            rect_item.setData(0, i)           # store index for click identification
            rect_item.setZValue(100)
            if i == self._selected_idx:
                rect_item.setPen(QPen(QColor("#ffffff"), 3, Qt.DotLine))
            self.scene.addItem(rect_item)

            # class label
            cls_name = self.class_names.get(cls_id, str(cls_id))
            text = self.scene.addText(f"{i+1}: {cls_name}")
            text.setDefaultTextColor(colour)
            text.setPos(min_c, max(0, min_r - 18))
            text.setZValue(101)

        self.box_count_label.setText(f"{len(self._boxes)} box(es)")
        self.update_crop_preview()

    def _clear_view(self):
        self._img_gray = None
        self._boxes = []
        self._selected_idx = None
        self.scene.clear()
        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)
        self.box_count_label.setText("")
        self.update_crop_preview()

    # ── box editing ───────────────────────────────────────────────────────────

    def _on_bbox_selected(self, idx: int):
        """Normal-mode click: select box and show crop preview."""
        self._selected_idx = idx if 0 <= idx < len(self._boxes) else None
        self._render()

    def update_crop_preview(self):
        if self._img_gray is None or self._selected_idx is None:
            self.crop_preview.setText("Select a box\nto preview")
            self.crop_preview.setPixmap(QPixmap())
            return
        if not (0 <= self._selected_idx < len(self._boxes)):
            self.crop_preview.setText("Select a box\nto preview")
            self.crop_preview.setPixmap(QPixmap())
            return

        _, min_r, min_c, max_r, max_c = self._boxes[self._selected_idx]
        sr, sc, er, ec = square_bbox((min_r, min_c, max_r, max_c),
                                     self._img_gray.shape, margin=10)
        crop = ensure_8bit(self._img_gray[sr:er, sc:ec])
        h, w = crop.shape
        qimg = QImage(crop.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)

        target = self.crop_preview.size()
        if target.width() > 0 and target.height() > 0:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.crop_preview.setText("")
        self.crop_preview.setPixmap(pix)

    def _on_rect_created(self, bbox: tuple):
        """User drew a new box — add with class 0 (first class)."""
        min_r, min_c, max_r, max_c = bbox
        self._boxes.append((0, min_r, min_c, max_r, max_c))
        self._selected_idx = len(self._boxes) - 1
        self._render()
        self._set_status(f"Box added. {len(self._boxes)} box(es). Save with Ctrl+S.")

    def _on_bbox_clicked(self, idx):
        """Delete-on-click: idx stored in item.data(0) is the box index."""
        idx = int(idx)
        if 0 <= idx < len(self._boxes):
            self._boxes.pop(idx)
            self._selected_idx = None
            self._render()
            self._set_status(f"Box removed. {len(self._boxes)} box(es). Save with Ctrl+S.")

    def _delete_selected_box(self):
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._boxes):
            self._boxes.pop(self._selected_idx)
            self._selected_idx = None
            self._render()
            self._set_status(f"Box deleted. {len(self._boxes)} box(es). Save with Ctrl+S.")

    # ── save ──────────────────────────────────────────────────────────────────

    def _save(self):
        if self._img_gray is None or not self._current_stem:
            return
        h, w = self._img_gray.shape

        lbl = self._label_path()
        folder = self._crop_folder()
        if lbl is None or folder is None:
            return

        _write_labels(lbl, self._boxes, w, h)
        _extract_crops(self._img_gray, self._boxes, folder, self._current_stem)

        self._set_status(
            f"Saved. {len(self._boxes)} box(es), {len(self._boxes)} crop(s) written."
        )

    # ── delete entire image ───────────────────────────────────────────────────

    def _delete_entire_image(self):
        if not self._current_stem or not self._current_class:
            return
        reply = QMessageBox.question(
            self, "Delete entire image",
            f"Permanently delete all crops, labels, and the whole image for "
            f"'{self._current_stem}'?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        folder = self._crop_folder()
        if folder and folder.exists():
            shutil.rmtree(folder)

        img_path = self._whole_image_path()
        if img_path and img_path.exists():
            img_path.unlink()

        self._clear_view()
        self._set_status(f"Deleted '{self._current_stem}'.")

        # refresh image list
        current_row = self.image_list.currentRow()
        self._populate_images(self._current_class)
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(min(current_row, self.image_list.count() - 1))

    # ── draw / delete mode toggles ────────────────────────────────────────────

    def _on_draw_toggled(self, on: bool):
        self.view.set_draw_mode(on)
        if on:
            self.delete_check.setChecked(False)

    def _on_delete_toggled(self, on: bool):
        self.view.set_delete_mode(on)
        if on:
            self.draw_check.setChecked(False)

    # ── status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.status_label.setText(msg)

    # ── image navigation ──────────────────────────────────────────────────────

    def _next_image(self):
        row = self.image_list.currentRow()
        if row + 1 < self.image_list.count():
            self.image_list.setCurrentRow(row + 1)
        elif self._current_class:
            # advance to the next class
            class_row = self.class_list.currentRow()
            if class_row + 1 < self.class_list.count():
                self.class_list.setCurrentRow(class_row + 1)
                if self.image_list.count() > 0:
                    self.image_list.setCurrentRow(0)

    def _prev_image(self):
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)
        elif self._current_class:
            class_row = self.class_list.currentRow()
            if class_row > 0:
                self.class_list.setCurrentRow(class_row - 1)
                last = self.image_list.count() - 1
                if last >= 0:
                    self.image_list.setCurrentRow(last)

    # ── session progress persistence ──────────────────────────────────────────

    def _progress_path(self) -> Path:
        return self.dataset_root / ".review_progress.json"

    def _save_progress(self):
        if not self._current_class or not self._current_stem:
            return
        try:
            reviewed_serialisable = {
                cls: list(stems) for cls, stems in self._reviewed.items()
            }
            self._progress_path().write_text(
                json.dumps({
                    "class": self._current_class,
                    "stem": self._current_stem,
                    "reviewed": reviewed_serialisable,
                }),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load_progress(self):
        p = self._progress_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return

        # Restore reviewed sets before populating any image list
        for cls, stems in data.get("reviewed", {}).items():
            self._reviewed[cls] = set(stems)

        saved_class = data.get("class", "")
        saved_stem = data.get("stem", "")
        if not saved_class or not saved_stem:
            return

        # Select the saved class (triggers _populate_images with reviewed filter)
        for i in range(self.class_list.count()):
            if self.class_list.item(i).text() == saved_class:
                self.class_list.setCurrentRow(i)
                break
        else:
            return

        # Select the saved stem in the filtered image list
        for i in range(self.image_list.count()):
            if self.image_list.item(i).text() == saved_stem:
                self.image_list.setCurrentRow(i)
                return

    # ── mark as reviewed (Enter) ──────────────────────────────────────────────

    def _mark_reviewed(self):
        """Save labels, mark image as reviewed, remove from list, advance."""
        if not self._current_class or not self._current_stem:
            return

        self._save()

        cls = self._current_class
        stem = self._current_stem
        self._reviewed.setdefault(cls, set()).add(stem)

        # Remove the current item from the list widget
        row = self.image_list.currentRow()
        self.image_list.takeItem(row)

        # Update count in status
        done = len(self._reviewed.get(cls, set()))
        class_dir = self.dataset_root / cls
        total = sum(
            1 for d in class_dir.iterdir()
            if d.is_dir() and d.name != "whole_images"
        )
        remaining = self.image_list.count()

        if remaining > 0:
            new_row = min(row, remaining - 1)
            self.image_list.setCurrentRow(new_row)
            # Force load in case Qt didn't fire currentTextChanged (row unchanged)
            item = self.image_list.item(new_row)
            if item:
                self._on_image_selected(item.text())
            self._set_status(f"Reviewed. {remaining} remaining, {done}/{total} done.")
        else:
            self._clear_view()
            self._set_status(f"All {done}/{total} images reviewed for '{cls}'.")
            self._next_image()

        self._save_progress()

    # ── keyboard shortcuts (mirrors ImagePreview.keyPressEvent) ───────────────

    def keyPressEvent(self, event):
        key = event.key()

        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._mark_reviewed()

        elif key in (Qt.Key_Right, ord('N')):
            self._next_image()

        elif key in (Qt.Key_Left, ord('P')):
            self._prev_image()

        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected_box()

        elif key == ord('D'):
            self.delete_check.setChecked(not self.delete_check.isChecked())

        elif key == ord('F'):
            self._fit_view()

        elif key == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
                self.setWindowState(Qt.WindowMaximized)
            else:
                self.showFullScreen()

        else:
            super().keyPressEvent(event)

    # ── resize: re-fit image ──────────────────────────────────────────────────

    def _fit_view(self):
        """Reset zoom to fit the current image in the view."""
        self._user_zoomed = False
        if self._img_gray is not None and self._pix_item.pixmap():
            self.view.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sz = event.size()
        self._apply_layout(portrait=sz.height() > sz.width())
        if not self._user_zoomed and self._img_gray is not None and self._pix_item.pixmap():
            self.view.fitInView(self._pix_item, Qt.KeepAspectRatio)


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/complete")
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        sys.exit(1)

    win = DatasetReviewer(dataset_root)
    win.show()
    sys.exit(app.exec_())
