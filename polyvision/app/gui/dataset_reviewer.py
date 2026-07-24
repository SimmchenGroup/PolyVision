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

from PyQt5.QtCore import Qt, QRectF, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPen, QBrush, QPixmap, QPainter, QIcon
from PyQt5.QtWidgets import (
    QApplication, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QShortcut, QSizePolicy, QVBoxLayout, QWidget,
    QGraphicsPixmapItem, QGraphicsTextItem, QCheckBox, QFrame, QScrollArea,
    QButtonGroup, QToolButton,
)

from polyvision.core.geometry import square_bbox
from polyvision.core.image_io import ensure_8bit, load_as_gray

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── Theme tokens + stylesheet (mirrors polyvision/app/main.py) ─────────────────

THEMES = {
    "dark": {
        "window_bg": "#1d1e26", "panel_bg": "#262834", "panel_bg_alt": "#2e3040",
        "border": "#3a3d4d", "text_color": "#e8e9ee", "text_muted": "#9a9eb0",
        "highlight": "#61AFEF", "button_text": "#0e1116", "canvas": "#14151b",
        "good": "#7EE2A8", "warn": "#E5C07B", "danger": "#e06c75",
    },
    "light": {
        "window_bg": "#eef0f3", "panel_bg": "#ffffff", "panel_bg_alt": "#f3f4f7",
        "border": "#d8dbe2", "text_color": "#23242e", "text_muted": "#6b7080",
        "highlight": "#2f7fc4", "button_text": "#ffffff", "canvas": "#dfe2e7",
        "good": "#1f8a5b", "warn": "#a87616", "danger": "#c0392b",
    },
}

# Class overlay palette (index by class id, shared across both themes).
CLASS_OVERLAY_PALETTE = [
    "#00c853", "#00b3ff", "#ffcc00", "#ff0066", "#a66bff",
    "#ff6600", "#00bcd4", "#8d99ae", "#ff3333",
]

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
        background-color: {c['window_bg']}; border: none;
    }}
    QToolTip {{ background-color: {c['panel_bg']}; color: {c['text_color']}; border: 1px solid {c['border']}; }}

    QFrame#card {{ background-color: {c['panel_bg']}; border: 1px solid {c['border']}; border-radius: 10px; }}
    QLabel {{ background: transparent; color: {c['text_color']}; }}
    QLabel#cardTitle {{ font-weight: 600; font-size: 12px; }}
    QLabel#muted {{ color: {c['text_muted']}; font-size: 12px; }}
    QLabel#hint {{ color: {c['text_muted']}; font-size: 11px; }}
    QLabel#mono {{ font-family: 'Consolas', ui-monospace, monospace; color: {c['text_muted']}; font-size: 10px; }}
    QLabel#value {{ font-family: 'Consolas', ui-monospace, monospace; color: {c['text_color']}; font-size: 12px; }}
    QLabel#appTitle {{ font-weight: 700; font-size: 15px; }}

    QPushButton {{
        background-color: {c['panel_bg_alt']}; color: {c['text_color']};
        border: 1px solid {c['border']}; border-radius: 6px; padding: 6px 10px;
    }}
    QPushButton:hover {{ border-color: {c['highlight']}; }}
    QPushButton#accent {{
        background-color: {c['highlight']}; color: {c['button_text']};
        border: 1px solid {c['highlight']}; font-weight: 600;
    }}
    QPushButton#danger {{ background: transparent; color: {c['danger']}; border: 1px solid {c['danger']}; border-radius: 10px; padding: 9px 0; }}
    QPushButton#iconbtn {{
        background-color: {c['panel_bg_alt']}; color: {c['text_color']};
        border: 1px solid {c['border']}; border-radius: 7px; padding: 4px 8px; min-width: 22px;
    }}
    QPushButton#tool {{
        background: transparent; border: none; border-radius: 5px;
        color: {c['text_muted']}; padding: 5px 12px;
    }}
    QPushButton#tool:checked {{ background-color: {c['panel_bg']}; color: {c['text_color']}; font-weight: 600; }}
    QWidget#segbar {{ background-color: {c['panel_bg_alt']}; border-radius: 7px; }}

    QListWidget {{ background: transparent; border: none; outline: none; }}
    QListWidget::item {{ padding: 4px 6px; border-radius: 6px; color: {c['text_color']}; }}
    QListWidget::item:selected {{ background-color: {c['panel_bg_alt']}; color: {c['text_color']}; }}

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


def _dot_icon(color_hex: str, size: int = 12) -> QIcon:
    """Return a small round colour-swatch QIcon of the given hex colour."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color_hex))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


# ── drawable view (mirrors DrawableGraphicsView gesture model in main.py) ──────

class _DrawableView(QGraphicsView):
    rectCreated = pyqtSignal(tuple)   # (min_r, min_c, max_r, max_c)
    bboxClicked = pyqtSignal(int)     # delete a single box: index into self._boxes
    bboxSelected = pyqtSignal(int)    # select a box: index into self._boxes
    areaDeleted = pyqtSignal(tuple)   # delete region (min_r, min_c, max_r, max_c)
    zoomed = pyqtSignal()             # emitted on any zoom / fit change

    def __init__(self, scene, parent=None):
        """Initialise the drawable graphics view with draw/delete modes off."""
        super().__init__(scene, parent)
        self._draw_mode = False
        self._delete_mode = False
        self._dragging = False
        self._start = None
        self._rubber = None
        self._draw_right_delete = False  # right-drag while drawing == delete
        # Right-click drives the draw-mode delete gesture, so no context menu.
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._min_scale = 0.2
        self._max_scale = 12.0
        self._fit_scale = 1.0

    def set_draw_mode(self, on: bool):
        """Enable or disable box-drawing mode."""
        self._draw_mode = bool(on)
        if not on:
            self._clear_rubber()

    def set_delete_mode(self, on: bool):
        """Enable or disable click/drag-to-delete-box mode."""
        self._delete_mode = bool(on)
        if not on:
            self._clear_rubber()

    # ---- rubber-band helpers ----
    def _clear_rubber(self):
        """Remove the current rubber-band selection rectangle."""
        self._dragging = False
        self._start = None
        self._draw_right_delete = False
        if self._rubber is not None and self.scene() is not None:
            self.scene().removeItem(self._rubber)
        self._rubber = None

    def _begin_rubber(self, event, colour):
        """Start a rubber-band rectangle at the click position."""
        self._dragging = True
        self._start = self.mapToScene(event.pos())
        pen = QPen(QColor(colour), 2, Qt.DashLine)
        self._rubber = QGraphicsRectItem()
        self._rubber.setPen(pen)
        self._rubber.setBrush(QBrush(Qt.NoBrush))
        self._rubber.setZValue(10_000)
        if self.scene() is not None:
            self.scene().addItem(self._rubber)
        self._rubber.setRect(QRectF(self._start, self._start))

    def _box_index_at(self, event):
        """Return the index of the box under the cursor, or None."""
        pos = self.mapToScene(event.pos())
        items = self.scene().items(pos, Qt.IntersectsItemBoundingRect)
        for item in items:
            if isinstance(item, QGraphicsRectItem) and item.data(0) is not None:
                return int(item.data(0))
        return None

    def _finish_delete(self, event):
        """Complete a delete drag: remove boxes inside the rubber-band rectangle."""
        rect = self._rubber.rect().normalized() if self._rubber is not None else None
        self._clear_rubber()
        if rect is None:
            return
        if rect.width() < 5 and rect.height() < 5:
            idx = self._box_index_at(event)
            if idx is not None:
                self.bboxClicked.emit(idx)
        else:
            self.areaDeleted.emit((
                int(round(rect.top())), int(round(rect.left())),
                int(round(rect.bottom())), int(round(rect.right())),
            ))

    def mousePressEvent(self, event):
        # Delete tool: left-drag rubber (release decides click vs region)
        """Mouse-press handler: start drawing/deleting, or fall back to panning."""
        if self._delete_mode and event.button() == Qt.LeftButton:
            self._begin_rubber(event, "#ff3333")
            event.accept()
            return

        # Draw tool: left-drag adds a box
        if self._draw_mode and event.button() == Qt.LeftButton:
            self._begin_rubber(event, "#ffffff")
            event.accept()
            return

        # Draw tool: right-click / right-drag deletes (box or region)
        if self._draw_mode and event.button() == Qt.RightButton:
            self._begin_rubber(event, "#ff3333")
            self._draw_right_delete = True
            event.accept()
            return

        # Select tool: click a box to select it; empty space falls through
        # to the base view (hand-pan when zoomed).
        if event.button() == Qt.LeftButton and not self._draw_mode and not self._delete_mode:
            idx = self._box_index_at(event)
            if idx is not None:
                self.bboxSelected.emit(idx)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Mouse-move handler: update the active rubber-band rectangle."""
        if (self._draw_mode or self._delete_mode) and self._dragging \
                and self._rubber is not None and self._start is not None:
            self._rubber.setRect(QRectF(self._start, self.mapToScene(event.pos())).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # Delete tool release
        """Mouse-release handler: finalise the drawn box or the delete action."""
        if self._delete_mode and self._dragging and event.button() == Qt.LeftButton:
            self._finish_delete(event)
            event.accept()
            return

        # Draw tool, right-button release == delete
        if self._draw_right_delete and self._dragging and event.button() == Qt.RightButton:
            self._finish_delete(event)
            event.accept()
            return

        # Draw tool, left-button release == create box
        if self._draw_mode and self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            if self._rubber is None:
                return
            rect = self._rubber.rect().normalized()
            if self.scene() is not None:
                self.scene().removeItem(self._rubber)
            self._rubber = None
            if rect.width() < 5 or rect.height() < 5:
                event.accept()
                return
            self.rectCreated.emit((
                int(round(rect.top())), int(round(rect.left())),
                int(round(rect.bottom())), int(round(rect.right())),
            ))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ---- zoom / pan ----
    def current_scale(self) -> float:
        """Current view zoom scale factor."""
        return float(self.transform().m11())

    def is_zoomed(self) -> bool:
        """True if the view is zoomed in beyond its fit scale."""
        return self.current_scale() > self._fit_scale * 1.05

    def _apply_scale(self, factor: float):
        """Scale the view by `factor`, clamped to sensible zoom bounds."""
        scale = self.current_scale() * factor
        scale = max(self._min_scale, min(self._max_scale, scale))
        cur = self.current_scale()
        if cur <= 0:
            return
        factor = scale / cur
        if abs(factor - 1.0) < 1e-4:
            return
        self.scale(factor, factor)
        self.zoomed.emit()

    def zoom_by(self, factor: float):
        """Zoom the view by a multiplicative factor."""
        self._apply_scale(factor)

    def fit_view(self):
        """Fit the whole scene within the view."""
        if self.scene() is None:
            return
        rect = self.scene().sceneRect()
        if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return
        self.resetTransform()
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._fit_scale = self.current_scale()
        self.zoomed.emit()

    def wheelEvent(self, event):
        """Zoom in/out on mouse-wheel scroll."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._apply_scale(factor)
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


def _count_labels(txt_path: Path) -> int:
    """Number of valid YOLO label lines in a file (cheap, no image needed)."""
    if not txt_path.exists():
        return 0
    n = 0
    for line in txt_path.read_text().strip().splitlines():
        if len(line.strip().split()) == 5:
            n += 1
    return n


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
    """Convert a greyscale image array to a QPixmap for display."""
    img8 = ensure_8bit(img)
    h, w = img8.shape
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


# ── main window ───────────────────────────────────────────────────────────────

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
        """Build the Dataset Reviewer window for the given dataset root and class list."""
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
        self._user_zoomed = False
        self._assign_class_id = 0
        self.current_theme = "dark"
        self.current_tool = "select"
        self._class_count_labels: dict[str, QLabel] = {}

        self._build_ui()
        self.apply_theme()
        self._populate_classes()
        self._populate_assign_classes()
        self._load_progress()

    # ── UI construction ───────────────────────────────────────────────────────

    def _card(self, title, hint=None, stretch_body=False):
        """Build a titled card frame used to group a panel's widgets."""
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        if title is not None:
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            t = QLabel(title)
            t.setObjectName("cardTitle")
            header.addWidget(t)
            header.addStretch()
            if hint is not None:
                self.__dict__[hint[0]] = QLabel(hint[1])
                self.__dict__[hint[0]].setObjectName("muted")
                header.addWidget(self.__dict__[hint[0]])
            lay.addLayout(header)
        return frame, lay

    def _build_ui(self):
        # ── shared control widgets (kept: names, signals, slots) ──
        """Construct the reviewer's widgets and lay out the window."""
        self.class_list = QListWidget()
        self.class_list.currentTextChanged.connect(self._on_class_selected)

        self.image_list = QListWidget()
        self.image_list.currentTextChanged.connect(self._on_image_selected)

        self.box_count_label = QLabel("")   # kept (updated in _render); hidden host

        # Draw / delete checkboxes stay as the mode source; the canvas tool tabs
        # drive them. Kept hidden so their toggled slots still run.
        self.draw_check = QCheckBox("Draw Boxes")
        self.draw_check.toggled.connect(self._on_draw_toggled)
        self.delete_check = QCheckBox("Delete on Click")
        self.delete_check.toggled.connect(self._on_delete_toggled)

        self._save_btn = QPushButton("Save  (Ctrl+S)")   # kept; top-bar Save mirrors it
        self._save_btn.clicked.connect(self._save)

        self.delete_image_btn = QPushButton("Delete entire image")
        self.delete_image_btn.setObjectName("danger")
        self.delete_image_btn.setCursor(Qt.PointingHandCursor)
        self.delete_image_btn.clicked.connect(self._delete_entire_image)

        self.status_label = QLabel("Select a class and image.")   # kept; status bar
        self.status_label.setWordWrap(True)

        self.crop_preview = QLabel("Select a box to preview")
        self.crop_preview.setAlignment(Qt.AlignCenter)
        self.crop_preview.setMinimumHeight(220)

        self.crop_meta_label = QLabel("")
        self.crop_meta_label.setObjectName("mono")

        # Assign-class list + button (right inspector)
        self.assign_class_list = QListWidget()
        self.assign_class_list.setMaximumHeight(200)
        self.assign_class_list.currentRowChanged.connect(self._on_assign_class_row)
        self.assign_btn = QPushButton("Assign to selected box")
        self.assign_btn.clicked.connect(self._assign_class_to_selected)

        # Boxes-on-image list (right inspector, per-row ✕)
        self.box_list = QListWidget()
        self.box_list.setMaximumHeight(160)

        # ── main image view ──
        self.scene = QGraphicsScene()
        self.view = _DrawableView(self.scene)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.rectCreated.connect(self._on_rect_created)
        self.view.bboxClicked.connect(self._on_bbox_clicked)
        self.view.bboxSelected.connect(self._on_bbox_selected)
        self.view.areaDeleted.connect(self._on_area_deleted)
        self.view.zoomed.connect(self._on_zoomed)

        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)

        # Hidden holder keeps the kept-but-unshown widgets alive.
        self._hidden_host = QWidget()
        self._hidden_host.setVisible(False)
        hh = QVBoxLayout(self._hidden_host)
        hh.setContentsMargins(0, 0, 0, 0)
        for w in (self.box_count_label, self.draw_check, self.delete_check, self._save_btn):
            hh.addWidget(w)

        # ================= TOP ACTION BAR =================
        topbar = QWidget()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(50)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(14, 0, 14, 0)
        tb.setSpacing(12)

        app_title = QLabel("PolyVision")
        app_title.setObjectName("appTitle")
        app_subtitle = QLabel("Dataset Reviewer")
        app_subtitle.setObjectName("muted")
        tb.addWidget(app_title)
        tb.addWidget(app_subtitle)

        tb.addStretch()

        self.prev_btn = QPushButton("‹")
        self.prev_btn.setObjectName("iconbtn")
        self.prev_btn.setToolTip("Left / P")
        self.prev_btn.setFixedSize(30, 30)
        self.prev_btn.clicked.connect(self._prev_image)

        nav_center = QWidget()
        nav_center.setMinimumWidth(180)
        nc = QVBoxLayout(nav_center)
        nc.setContentsMargins(0, 0, 0, 0)
        nc.setSpacing(0)
        self.filename_label = QLabel("—")
        self.filename_label.setObjectName("value")
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.review_progress_label = QLabel("")
        self.review_progress_label.setObjectName("mono")
        self.review_progress_label.setAlignment(Qt.AlignCenter)
        nc.addWidget(self.filename_label)
        nc.addWidget(self.review_progress_label)

        self.next_btn = QPushButton("›")
        self.next_btn.setObjectName("iconbtn")
        self.next_btn.setToolTip("Right / N")
        self.next_btn.setFixedSize(30, 30)
        self.next_btn.clicked.connect(self._next_image)

        tb.addWidget(self.prev_btn)
        tb.addWidget(nav_center)
        tb.addWidget(self.next_btn)

        tb.addSpacing(6)
        self.theme_btn = QPushButton("◐")
        self.theme_btn.setObjectName("iconbtn")
        self.theme_btn.setToolTip("Toggle theme")
        self.theme_btn.setFixedSize(30, 30)
        self.theme_btn.clicked.connect(self.toggle_theme)

        self.preview_link_btn = QPushButton("← Preview")
        self.preview_link_btn.setObjectName("iconbtn")
        self.preview_link_btn.setCursor(Qt.PointingHandCursor)
        self.preview_link_btn.clicked.connect(self.close)

        self.top_save_btn = QPushButton("Save  ⌃S")
        self.top_save_btn.setObjectName("accent")
        self.top_save_btn.setCursor(Qt.PointingHandCursor)
        self.top_save_btn.clicked.connect(self._save)

        tb.addWidget(self.theme_btn)
        tb.addWidget(self.preview_link_btn)
        tb.addWidget(self.top_save_btn)

        # ================= LEFT PANEL =================
        cls_card, cls_lay = self._card("Class folder")
        cls_lay.addWidget(self.class_list)

        img_card, img_lay = self._card("Pending images", hint=("pending_count_label", "(0)"))
        img_lay.addWidget(self.image_list, 1)

        left_panel = QWidget()
        lp = QVBoxLayout(left_panel)
        lp.setContentsMargins(12, 12, 12, 12)
        lp.setSpacing(12)
        lp.addWidget(cls_card)
        lp.addWidget(img_card, 1)
        lp.addWidget(self.delete_image_btn)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setFixedWidth(280)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ================= CENTER CANVAS =================
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
        for b, tool in [(self.tool_select_btn, "select"),
                        (self.tool_draw_btn, "draw"),
                        (self.tool_delete_btn, "delete")]:
            b.setObjectName("tool")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, t=tool: self.set_tool(t))
            self.tool_group.addButton(b)
            tt_l.addWidget(b)
        self.tool_select_btn.setChecked(True)

        gesture_hint = QLabel("Draw: L-drag add · R delete · Delete: L-click/drag remove")
        gesture_hint.setObjectName("hint")

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
        self.fit_btn = QPushButton("Fit (F)")
        self.fit_btn.setObjectName("iconbtn")
        self.fit_btn.setFixedHeight(26)
        self.fit_btn.clicked.connect(self._fit_view)

        ctl.addWidget(tool_tabs)
        ctl.addWidget(gesture_hint)
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
        cl.addWidget(self.view, 1)

        # ================= RIGHT INSPECTOR =================
        assign_card, assign_lay = self._card("Assign class to new boxes")
        assign_lay.addWidget(self.assign_class_list)
        assign_lay.addWidget(self.assign_btn)

        crop_card, crop_lay = self._card("Crop preview")
        crop_lay.addWidget(self.crop_preview)
        crop_lay.addWidget(self.crop_meta_label)

        boxes_card, boxes_lay = self._card("Boxes on this image", hint=("box_list_count_label", "(0)"))
        boxes_lay.addWidget(self.box_list)

        right_panel = QWidget()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(12, 12, 12, 12)
        rp.setSpacing(12)
        rp.addWidget(assign_card)
        rp.addWidget(crop_card)
        rp.addWidget(boxes_card)
        rp.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_panel)
        right_scroll.setFixedWidth(300)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ================= ASSEMBLE =================
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
        root.addLayout(main_row, 1)
        root.addWidget(self._hidden_host)
        self.setCentralWidget(central)

        # Status bar: live message left, keybind cheat-sheet right (monospace)
        self.keybind_label = QLabel(
            "↵ mark reviewed · ←→ nav · ⌃S save · D draw/delete · Del box · F fit"
        )
        self.keybind_label.setObjectName("mono")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.keybind_label)
        self.statusBar().showMessage("Select a class and image.")

        # keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save)
        QShortcut(QKeySequence("Delete"), self).activated.connect(self._delete_selected_box)
        QShortcut(QKeySequence("Backspace"), self).activated.connect(self._delete_selected_box)

    # ── theme ─────────────────────────────────────────────────────────────────

    def _theme(self) -> dict:
        """Return the active theme's colour tokens."""
        return THEMES[self.current_theme]

    def apply_theme(self):
        """Apply the current theme's stylesheet to the window."""
        global _ACTIVE
        c = THEMES[self.current_theme]
        _ACTIVE = c
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(c))
        self.view.setBackgroundBrush(QColor(c["canvas"]))
        self.theme_btn.setText("◐" if self.current_theme == "dark" else "◑")
        self.crop_preview.setStyleSheet(
            f"background-color:{c['canvas']}; border:1px dashed {c['border']};"
            f"border-radius:8px; color:{c['text_muted']};"
        )
        self.update()

    def toggle_theme(self):
        """Switch between the light and dark themes."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()
        self._render()

    # ── tools ─────────────────────────────────────────────────────────────────

    def set_tool(self, tool: str):
        """Switch the active tool (select / draw / delete)."""
        if tool == "select":
            self.draw_check.setChecked(False)
            self.delete_check.setChecked(False)
        elif tool == "draw":
            self.delete_check.setChecked(False)
            self.draw_check.setChecked(True)
        else:  # delete
            self.draw_check.setChecked(False)
            self.delete_check.setChecked(True)
        self._sync_tool_tabs()

    def _sync_tool_tabs(self):
        """Sync the tool toggle checkboxes with the active tool."""
        if self.draw_check.isChecked():
            self.current_tool = "draw"
        elif self.delete_check.isChecked():
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
        """Enable scroll-hand dragging only when zoomed in select mode."""
        if self.current_tool == "select" and self.view.is_zoomed():
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            self.view.setDragMode(QGraphicsView.NoDrag)

    # ── zoom ──────────────────────────────────────────────────────────────────

    def zoom_in(self):
        """Zoom the image view in."""
        self.view.zoom_by(1.25)

    def zoom_out(self):
        """Zoom the image view out."""
        self.view.zoom_by(1.0 / 1.25)

    def _on_zoomed(self):
        """Record that the user manually zoomed (disables auto-fit)."""
        self._user_zoomed = True
        self._update_zoom_label()
        self._update_drag_mode()

    def _update_zoom_label(self):
        """Update the zoom-percentage label."""
        fit = getattr(self.view, "_fit_scale", 1.0) or 1.0
        self.zoom_label.setText(f"{int(round(self.view.current_scale() / fit * 100))}%")

    # ── population ────────────────────────────────────────────────────────────

    def _class_dot(self, name: str, row: int) -> str:
        # Colour by class id if the folder name maps to a known class, else row.
        """Return the overlay colour swatch icon for a class row."""
        name_to_id = {n: i for i, n in self.class_names.items()}
        cid = name_to_id.get(name, row)
        return CLASS_OVERLAY_PALETTE[cid % len(CLASS_OVERLAY_PALETTE)]

    def _class_totals(self, class_name: str) -> tuple[int, int]:
        """Return (annotated, total) image counts for a class."""
        class_dir = self.dataset_root / class_name
        total = sum(
            1 for d in class_dir.iterdir()
            if d.is_dir() and d.name != "whole_images"
        ) if class_dir.exists() else 0
        done = len(self._reviewed.get(class_name, set()))
        return total - done, total

    def _make_class_row(self, name: str, dot_color: str, remaining: int, total: int):
        """Build a class-list row widget with its colour dot and progress counter."""
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(9, 9)
        dot.setStyleSheet(f"background:{dot_color}; border-radius:4px;")
        lay.addWidget(dot)
        lbl = QLabel(name)
        lay.addWidget(lbl, 1)
        cnt = QLabel(f"{remaining}/{total}")
        cnt.setObjectName("mono")
        lay.addWidget(cnt)
        self._class_count_labels[name] = cnt
        return w

    def _populate_classes(self):
        """Fill the class list from the dataset root."""
        self.class_list.blockSignals(True)
        self.class_list.clear()
        self._class_count_labels.clear()
        row = 0
        for d in sorted(self.dataset_root.iterdir()):
            if d.is_dir() and d.name != "whole_images":
                remaining, total = self._class_totals(d.name)
                item = QListWidgetItem()
                item.setText(d.name)            # kept for logic + currentTextChanged
                item.setData(Qt.UserRole, d.name)
                item.setSizeHint(QSize(0, 28))
                self.class_list.addItem(item)
                self.class_list.setItemWidget(
                    item, self._make_class_row(d.name, self._class_dot(d.name, row), remaining, total)
                )
                row += 1
        self.class_list.blockSignals(False)

    def _update_class_count(self, class_name: str):
        """Refresh the remaining/total counter shown on a class row."""
        if class_name in self._class_count_labels:
            remaining, total = self._class_totals(class_name)
            self._class_count_labels[class_name].setText(f"{remaining}/{total}")

    def _make_image_row(self, stem: str, box_count: int):
        """Build an image-list row widget showing the stem and its box count."""
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)
        lbl = QLabel(stem)
        lbl.setObjectName("value")
        lay.addWidget(lbl, 1)
        cnt = QLabel(f"{box_count} bx")
        cnt.setObjectName("mono")
        lay.addWidget(cnt)
        return w

    def _populate_images(self, class_name: str):
        """Fill the image list for the selected class."""
        self.image_list.blockSignals(True)
        self.image_list.clear()
        class_dir = self.dataset_root / class_name
        all_stems = sorted(
            d.name for d in class_dir.iterdir()
            if d.is_dir() and d.name != "whole_images"
        )
        done = self._reviewed.get(class_name, set())
        pending = [s for s in all_stems if s not in done]
        for stem in pending:
            n_boxes = _count_labels(class_dir / stem / f"{stem}.txt")
            item = QListWidgetItem()
            item.setText(stem)                  # kept for logic + currentTextChanged
            item.setData(Qt.UserRole, stem)
            item.setSizeHint(QSize(0, 28))
            self.image_list.addItem(item)
            self.image_list.setItemWidget(item, self._make_image_row(stem, n_boxes))
        self.image_list.blockSignals(False)

        reviewed_count = len(done)
        total = len(all_stems)
        if hasattr(self, "pending_count_label"):
            self.pending_count_label.setText(f"({len(pending)})")
        self._update_class_count(class_name)
        self._set_status(
            f"{len(pending)} remaining, {reviewed_count}/{total} reviewed."
        )

    def _populate_assign_classes(self):
        """Fill the class picker used to reassign a box's class."""
        self.assign_class_list.blockSignals(True)
        self.assign_class_list.clear()
        # Prefer explicit class_names; else derive from folder names.
        if self.class_names:
            entries = [(cid, name) for cid, name in sorted(self.class_names.items()) if cid >= 0]
        else:
            entries = [
                (i, d.name) for i, d in enumerate(
                    sorted(p for p in self.dataset_root.iterdir()
                           if p.is_dir() and p.name != "whole_images")
                )
            ]
        for cid, name in entries:
            item = QListWidgetItem(f"{name}")
            item.setData(Qt.UserRole, cid)
            item.setIcon(_dot_icon(CLASS_OVERLAY_PALETTE[cid % len(CLASS_OVERLAY_PALETTE)]))
            self.assign_class_list.addItem(item)
        if self.assign_class_list.count() > 0:
            self.assign_class_list.setCurrentRow(0)
            self._assign_class_id = int(self.assign_class_list.item(0).data(Qt.UserRole))
        self.assign_class_list.blockSignals(False)

    def _on_assign_class_row(self, row: int):
        """Handle selection in the assign-class list."""
        if 0 <= row < self.assign_class_list.count():
            self._assign_class_id = int(self.assign_class_list.item(row).data(Qt.UserRole))

    def _assign_class_to_selected(self):
        """Assign the chosen class to the currently selected box."""
        if self._selected_idx is None or not (0 <= self._selected_idx < len(self._boxes)):
            return
        _, min_r, min_c, max_r, max_c = self._boxes[self._selected_idx]
        self._boxes[self._selected_idx] = (int(self._assign_class_id), min_r, min_c, max_r, max_c)
        cls_name = self.class_names.get(self._assign_class_id, str(self._assign_class_id))
        self._render()
        self._set_status(
            f"Assigned {cls_name} to box {self._selected_idx + 1}. Save with Ctrl+S."
        )

    # ── selection handlers ────────────────────────────────────────────────────

    def _on_class_selected(self, class_name: str):
        """Handle selecting a class: load its images."""
        if not class_name:
            return
        self._current_class = class_name
        self._current_stem = None
        self._populate_images(class_name)
        self._clear_view()
        self._refresh_topbar()

    def _on_image_selected(self, stem: str):
        """Handle selecting an image: load it and its labels."""
        if not stem or not self._current_class:
            return
        self._current_stem = stem
        self._load_image_and_labels()
        self._save_progress()

    # ── image loading ─────────────────────────────────────────────────────────

    def _whole_image_path(self) -> Path | None:
        """Path to the current whole image."""
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
        """Path to the current image's crop folder."""
        if not self._current_class or not self._current_stem:
            return None
        return self.dataset_root / self._current_class / self._current_stem

    def _label_path(self) -> Path | None:
        """Path to the current image's YOLO label file."""
        folder = self._crop_folder()
        if folder is None:
            return None
        return folder / f"{self._current_stem}.txt"

    def _load_image_and_labels(self):
        """Load the current whole image and its bounding-box labels."""
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
        self._fit_view()
        self._refresh_topbar()
        self._set_status(f"{self._current_stem}  —  {len(self._boxes)} box(es)")

    # ── top bar ───────────────────────────────────────────────────────────────

    def _refresh_topbar(self):
        """Update the top-bar filename/info labels."""
        if not hasattr(self, "filename_label"):
            return
        self.filename_label.setText(self._current_stem or "—")
        if self._current_class:
            remaining, total = self._class_totals(self._current_class)
            done = total - remaining
            self.review_progress_label.setText(f"{self._current_class} · {done}/{total} reviewed")
        else:
            self.review_progress_label.setText("")

    # ── rendering ─────────────────────────────────────────────────────────────

    def _box_colour(self, cls_id: int) -> QColor:
        """Return the overlay colour for a class id."""
        return QColor(CLASS_OVERLAY_PALETTE[cls_id % len(CLASS_OVERLAY_PALETTE)])

    def _render(self):
        """Redraw the scene: the whole image plus all current bounding boxes."""
        self.scene.clear()
        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)

        if self._img_gray is None:
            self.box_count_label.setText("")
            self._populate_box_list()
            self.update_crop_preview()
            return

        self._pix_item.setPixmap(_gray_to_pixmap(self._img_gray))
        h, w = self._img_gray.shape
        self.scene.setSceneRect(0, 0, w, h)

        accent = self._theme()["highlight"]
        for i, (cls_id, min_r, min_c, max_r, max_c) in enumerate(self._boxes):
            colour = self._box_colour(cls_id)
            rect_item = QGraphicsRectItem(min_c, min_r, max_c - min_c, max_r - min_r)
            # Transparent fill so the full interior registers for click hit-testing
            rect_item.setBrush(QBrush(QColor(255, 255, 255, 0)))
            rect_item.setData(0, i)           # store index for click identification
            rect_item.setZValue(100)
            if i == self._selected_idx:
                rect_item.setPen(QPen(QColor(accent), 3))
            else:
                rect_item.setPen(QPen(colour, 2))
            self.scene.addItem(rect_item)

            # class label — filled pill in the box colour
            cls_name = self.class_names.get(cls_id, str(cls_id))
            text = QGraphicsTextItem()
            text.setHtml(
                f'<div style="background:{colour.name()};color:#ffffff;'
                f'padding:0px 4px;border-radius:3px;font-weight:600;">{i+1} · {cls_name}</div>'
            )
            f = text.font()
            f.setPointSize(9)
            f.setBold(True)
            text.setFont(f)
            text.setPos(min_c - 2, max(0, min_r - 20))
            text.setZValue(101)
            self.scene.addItem(text)

        self.box_count_label.setText(f"{len(self._boxes)} box(es)")
        self._populate_box_list()
        self.update_crop_preview()

    def _populate_box_list(self):
        """Fill the box list for the current image."""
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for i, (cls_id, *_rest) in enumerate(self._boxes):
            cls_name = self.class_names.get(cls_id, str(cls_id))
            item = QListWidgetItem()
            item.setData(Qt.UserRole, i)
            item.setSizeHint(QSize(0, 28))
            self.box_list.addItem(item)
            self.box_list.setItemWidget(
                item, self._make_box_row(i, f"{i + 1} · {cls_name}", self._box_colour(cls_id).name())
            )
        if self._selected_idx is not None and 0 <= self._selected_idx < self.box_list.count():
            self.box_list.setCurrentRow(self._selected_idx)
        self.box_list.blockSignals(False)
        if hasattr(self, "box_list_count_label"):
            self.box_list_count_label.setText(f"({len(self._boxes)})")

    def _make_box_row(self, i, text, dot_color):
        """Build a row widget for one bounding box in the box list."""
        c = self._theme()
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(6, 2, 4, 2)
        lay.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(9, 9)
        dot.setStyleSheet(f"background:{dot_color}; border-radius:2px;")
        lay.addWidget(dot)
        lbl = QLabel(text)
        lay.addWidget(lbl, 1)
        x = QToolButton()
        x.setText("✕")
        x.setAutoRaise(True)
        x.setCursor(Qt.PointingHandCursor)
        x.setToolTip("Delete (Del)")
        x.setStyleSheet(f"color:{c['text_muted']}; border:none; background:transparent;")
        x.clicked.connect(lambda _=False, idx=i: self._on_bbox_clicked(idx))
        lay.addWidget(x)
        # Row click selects the box (skip when the ✕ is what got clicked)
        w.mousePressEvent = lambda ev, idx=i: self._on_bbox_selected(idx)
        return w

    def _clear_view(self):
        """Clear the image view and reset the current selection state."""
        self._img_gray = None
        self._boxes = []
        self._selected_idx = None
        self.scene.clear()
        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)
        self.box_count_label.setText("")
        self._populate_box_list()
        self.update_crop_preview()

    # ── box editing ───────────────────────────────────────────────────────────

    def _on_bbox_selected(self, idx: int):
        """Select box and show crop preview."""
        self._selected_idx = idx if 0 <= idx < len(self._boxes) else None
        self._render()

    def update_crop_preview(self):
        """Update the crop-preview thumbnail for the selected box."""
        if self._img_gray is None or self._selected_idx is None \
                or not (0 <= self._selected_idx < len(self._boxes)):
            self.crop_preview.setText("Select a box to preview")
            self.crop_preview.setPixmap(QPixmap())
            self.crop_meta_label.setText("")
            return

        cls_id, min_r, min_c, max_r, max_c = self._boxes[self._selected_idx]
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
        cls_name = self.class_names.get(cls_id, str(cls_id))
        self.crop_meta_label.setText(f"{max_c - min_c}×{max_r - min_r} px · {cls_name}")

    def _on_rect_created(self, bbox: tuple):
        """User drew a new box — adopt the currently selected assign-class."""
        min_r, min_c, max_r, max_c = bbox
        cls_id = int(self._assign_class_id) if self._assign_class_id is not None else 0
        self._boxes.append((cls_id, min_r, min_c, max_r, max_c))
        self._selected_idx = len(self._boxes) - 1
        self._render()
        cls_name = self.class_names.get(cls_id, str(cls_id))
        self._set_status(f"Box added ({cls_name}). {len(self._boxes)} box(es). Save with Ctrl+S.")

    def _on_bbox_clicked(self, idx):
        """Delete a single box by index."""
        idx = int(idx)
        if 0 <= idx < len(self._boxes):
            self._boxes.pop(idx)
            self._selected_idx = None
            self._render()
            self._set_status(f"Box removed. {len(self._boxes)} box(es). Save with Ctrl+S.")

    def _on_area_deleted(self, region: tuple):
        """Delete every box whose centre falls inside the dragged region."""
        r_min, c_min, r_max, c_max = region
        kept = []
        removed = 0
        for box in self._boxes:
            _, b_r0, b_c0, b_r1, b_c1 = box
            cy = (b_r0 + b_r1) / 2.0
            cx = (b_c0 + b_c1) / 2.0
            if r_min <= cy <= r_max and c_min <= cx <= c_max:
                removed += 1
            else:
                kept.append(box)
        if removed:
            self._boxes = kept
            self._selected_idx = None
            self._render()
            self._set_status(f"Deleted {removed} box(es) in region. Save with Ctrl+S.")
        else:
            self._set_status("No boxes in region.")

    def _delete_selected_box(self):
        """Delete the currently selected bounding box."""
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._boxes):
            self._boxes.pop(self._selected_idx)
            self._selected_idx = None
            self._render()
            self._set_status(f"Box deleted. {len(self._boxes)} box(es). Save with Ctrl+S.")

    # ── save ──────────────────────────────────────────────────────────────────

    def _save(self):
        """Write the edited labels back to the .txt file and re-extract crops."""
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
        """Permanently delete the current image, its crops, and its label file."""
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
        """Handle the draw-boxes checkbox toggle."""
        self.view.set_draw_mode(on)
        if on:
            self.delete_check.setChecked(False)
        self._sync_tool_tabs()

    def _on_delete_toggled(self, on: bool):
        """Handle the delete-on-click checkbox toggle."""
        self.view.set_delete_mode(on)
        if on:
            self.draw_check.setChecked(False)
        self._sync_tool_tabs()

    # ── status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        """Show a message in the status bar."""
        self.status_label.setText(msg)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(msg)

    # ── image navigation ──────────────────────────────────────────────────────

    def _next_image(self):
        """Move to the next image in the list."""
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
        """Move to the previous image in the list."""
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
        """Path to the review-progress JSON file."""
        return self.dataset_root / ".review_progress.json"

    def _save_progress(self):
        """Persist the current review position to disk."""
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
        """Restore the last review position from disk."""
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
        if hasattr(self, "pending_count_label"):
            self.pending_count_label.setText(f"({remaining})")
        self._update_class_count(cls)

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

        self._refresh_topbar()
        self._save_progress()

    # ── keyboard shortcuts (mirrors ImagePreview.keyPressEvent) ───────────────

    def keyPressEvent(self, event):
        """Keyboard shortcuts: navigation, save, delete, and tool switching."""
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
            # Cycle draw ⇄ delete (matches the workbench D behaviour)
            if self.current_tool == "delete":
                self.set_tool("draw")
            else:
                self.set_tool("delete")

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
        if self._img_gray is not None:
            self.view.fit_view()
        self._update_zoom_label()

    def resizeEvent(self, event):
        """Re-fit the view when the window is resized."""
        super().resizeEvent(event)
        if not self._user_zoomed and self._img_gray is not None:
            self.view.fit_view()


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(build_stylesheet(THEMES["dark"]))

    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/complete")
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        sys.exit(1)

    win = DatasetReviewer(dataset_root)
    win.show()
    sys.exit(app.exec_())
