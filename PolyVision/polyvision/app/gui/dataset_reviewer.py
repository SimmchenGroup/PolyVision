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
from PyQt5.QtGui import QColor, QImage, QPen, QBrush, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QShortcut, QSizePolicy, QVBoxLayout, QWidget,
    QGraphicsPixmapItem, QCheckBox,
)
from PyQt5.QtGui import QKeySequence

from polyvision.core.geometry import square_bbox
from polyvision.core.image_io import ensure_8bit, load_as_gray


# ── minimal drawable view (mirrors DrawableGraphicsView in main.py) ───────────

class _DrawableView(QGraphicsView):
    rectCreated = pyqtSignal(tuple)   # (min_r, min_c, max_r, max_c)
    bboxClicked = pyqtSignal(int)     # index into self._boxes

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
            for item in self.scene().items(pos):
                if isinstance(item, QGraphicsRectItem) and item.data(0) is not None:
                    self.bboxClicked.emit(item.data(0))
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

        self._build_ui()
        self._populate_classes()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── left panel ──
        left = QWidget()
        left.setFixedWidth(240)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        left_layout.addWidget(QLabel("Class"))
        self.class_list = QListWidget()
        self.class_list.currentTextChanged.connect(self._on_class_selected)
        left_layout.addWidget(self.class_list)

        left_layout.addWidget(QLabel("Image"))
        self.image_list = QListWidget()
        self.image_list.currentTextChanged.connect(self._on_image_selected)
        left_layout.addWidget(self.image_list)

        self.delete_image_btn = QPushButton("Delete Entire Image")
        self.delete_image_btn.setStyleSheet("color: #ff4444;")
        self.delete_image_btn.clicked.connect(self._delete_entire_image)
        left_layout.addWidget(self.delete_image_btn)

        self.status_label = QLabel("Select a class and image.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        # ── right panel ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # toolbar
        toolbar = QHBoxLayout()
        self.draw_check = QCheckBox("Draw Boxes")
        self.draw_check.toggled.connect(self._on_draw_toggled)
        toolbar.addWidget(self.draw_check)

        self.delete_check = QCheckBox("Delete on Click")
        self.delete_check.toggled.connect(self._on_delete_toggled)
        toolbar.addWidget(self.delete_check)

        toolbar.addStretch()

        self.box_count_label = QLabel("")
        toolbar.addWidget(self.box_count_label)

        save_btn = QPushButton("Save  (Ctrl+S)")
        save_btn.clicked.connect(self._save)
        toolbar.addWidget(save_btn)

        right_layout.addLayout(toolbar)

        # graphics view
        self.scene = QGraphicsScene()
        self.view = _DrawableView(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.rectCreated.connect(self._on_rect_created)
        self.view.bboxClicked.connect(self._on_bbox_clicked)
        right_layout.addWidget(self.view)

        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)

        # ── root layout ──
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(left)
        root_layout.addWidget(right, stretch=1)
        self.setCentralWidget(root)

        # keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save)
        QShortcut(QKeySequence("Delete"), self).activated.connect(self._delete_selected_box)

    # ── population ────────────────────────────────────────────────────────────

    def _populate_classes(self):
        self.class_list.clear()
        for d in sorted(self.dataset_root.iterdir()):
            if d.is_dir() and d.name != "whole_images":
                self.class_list.addItem(d.name)

    def _populate_images(self, class_name: str):
        self.image_list.clear()
        class_dir = self.dataset_root / class_name
        stems = sorted(
            d.name for d in class_dir.iterdir()
            if d.is_dir() and d.name != "whole_images"
        )
        for stem in stems:
            self.image_list.addItem(stem)

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
            self._set_status(f"Whole image not found for '{self._current_stem}'.")
            self._clear_view()
            return

        img = load_as_gray(str(img_path))
        self._img_gray = ensure_8bit(img)
        h, w = self._img_gray.shape

        lbl = self._label_path()
        self._boxes = _read_labels(lbl, w, h) if lbl else []
        self._selected_idx = None

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
            rect_item.setBrush(QBrush(Qt.NoBrush))
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

    def _clear_view(self):
        self._img_gray = None
        self._boxes = []
        self._selected_idx = None
        self.scene.clear()
        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)
        self.box_count_label.setText("")

    # ── box editing ───────────────────────────────────────────────────────────

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

    # ── resize: re-fit image ──────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._img_gray is not None and self._pix_item.pixmap():
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
