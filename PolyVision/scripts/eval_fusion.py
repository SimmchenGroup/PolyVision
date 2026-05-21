from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


from polyvision.ml.fusion import ParticleFusionClassifier


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class YoloLabel:
    cls: int
    x: float
    y: float
    w: float
    h: float


def read_yolo_labels(txt_path: Path) -> List[YoloLabel]:
    out: List[YoloLabel] = []
    if not txt_path.exists():
        return out
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = parts[:5]
        out.append(YoloLabel(int(cls), float(x), float(y), float(w), float(h)))
    return out


def index_label_files(labels_root: Path) -> Dict[str, Path]:
    """
    Index all *.txt under labels_root by filename stem.
    If stems are not unique across folders, you must change the keying strategy.
    """
    idx: Dict[str, Path] = {}
    for p in labels_root.rglob("*.txt"):
        idx[p.stem] = p
    return idx


def iter_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def sample_images_per_class(root: Path, n_per_class: int = 10, seed: int = 42) -> List[Path]:
    """
    Sample up to n_per_class images from each immediate class subfolder under `root`.
    Expected layout:
      root/
        classA/*.jpg
        classB/*.jpg
        ...

    Returns a balanced list (as balanced as available files allow).
    """
    rng = np.random.default_rng(int(seed))
    out: List[Path] = []

    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())

    for cls_dir in class_dirs:
        imgs = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not imgs:
            continue

        # stable shuffle then take N
        idx = np.arange(len(imgs))
        rng.shuffle(idx)
        take = int(min(len(imgs), int(n_per_class)))
        out.extend([imgs[i] for i in idx[:take]])

    return out

def infer_hw_from_keras(model) -> Tuple[int, int]:
    ishape = getattr(model, "input_shape", None)
    if not ishape or len(ishape) < 4:
        raise ValueError(f"Cannot infer keras model input_shape: {ishape}")
    h, w = ishape[1], ishape[2]
    if h is None or w is None:
        raise ValueError(f"Dynamic keras input shape not supported: {ishape}")
    return int(h), int(w)


def yolo_to_xyxy(lbl: YoloLabel, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    xc = lbl.x * img_w
    yc = lbl.y * img_h
    bw = lbl.w * img_w
    bh = lbl.h * img_h
    x1 = int(round(xc - bw / 2))
    y1 = int(round(yc - bh / 2))
    x2 = int(round(xc + bw / 2))
    y2 = int(round(yc + bh / 2))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def default_preprocess_rgb01(batch_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    batch_rgb_uint8: (N,H,W,3) uint8 RGB
    returns float32 in [0,1]
    """
    return batch_rgb_uint8.astype(np.float32) / 255.0


def make_batch_from_crops(
    img_bgr: np.ndarray,
    boxes_xyxy: List[Tuple[int, int, int, int]],
    out_hw: Tuple[int, int],
) -> Tuple[np.ndarray, List[int]]:
    """
    Returns:
      batch_rgb_uint8: (N,H,W,3) RGB uint8
      kept_indices: indices of boxes kept (skips invalid crops)
    """
    out_h, out_w = out_hw
    batch = []
    kept = []

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (out_w, out_h))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        batch.append(crop_rgb)
        kept.append(i)

    if not batch:
        return np.empty((0, out_h, out_w, 3), dtype=np.uint8), []

    return np.stack(batch, axis=0).astype(np.uint8), kept


def top_k_by_area(
    boxes_xyxy: List[Tuple[int, int, int, int]],
    k: Optional[int],
) -> List[Tuple[int, int, int, int]]:
    if k is None or k <= 0 or len(boxes_xyxy) <= k:
        return boxes_xyxy
    areas = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        areas.append(max(0, x2 - x1) * max(0, y2 - y1))
    order = np.argsort(areas)[::-1][:k]
    return [boxes_xyxy[i] for i in order]

def default_preprocess_rgb01(batch_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    batch_rgb_uint8: (N,H,W,3) uint8 RGB
    returns float32 in [0,1]
    """
    return batch_rgb_uint8.astype(np.float32) / 255.0

def preprocess_efficientnet(batch_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    EfficientNet expects its own preprocessing (same family used during training).
    Input: RGB uint8 in [0..255]
    Output: float32 tensor suitable for EfficientNet.
    """
    x = batch_rgb_uint8.astype(np.float32)
    return efficientnet_preprocess(x)

def _top1(probs: np.ndarray) -> Tuple[int, float]:
    i = int(np.argmax(probs))
    return i, float(probs[i])


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color=(255, 255, 255)) -> None:
    """
    Draw readable text with a black outline.
    img is BGR uint8.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _save_debug_image(
    out_path: Path,
    img_bgr: np.ndarray,
    boxes_xyxy: List[Tuple[int, int, int, int]],
    labels: List[YoloLabel],
    class_names: List[str],
    local_top1: List[Tuple[int, float]],
    fused_top1: List[Tuple[int, float]],
    global_top1: Tuple[int, float],
    kept_indices: List[int],
    max_draw: int = 50,
) -> None:
    """
    Saves a single annotated whole-image view.
    - boxes/labels correspond to the *aligned* list (after any top-k truncation).
    - kept_indices maps from batch rows -> box index (since invalid crops are skipped).
    """
    vis = img_bgr.copy()

    g_idx, g_conf = global_top1
    g_name = class_names[g_idx] if 0 <= g_idx < len(class_names) else str(g_idx)
    _draw_label(vis, f"GLOBAL: {g_name} ({g_conf:.2f})", 10, 20, color=(255, 255, 0))

    draw_n = min(len(kept_indices), max_draw)
    for row_i in range(draw_n):
        i = kept_indices[row_i]
        if i < 0 or i >= len(boxes_xyxy) or i >= len(labels):
            continue

        x1, y1, x2, y2 = boxes_xyxy[i]
        true_id = int(labels[i].cls)
        true_name = class_names[true_id] if 0 <= true_id < len(class_names) else str(true_id)

        l_id, l_conf = local_top1[row_i]
        f_id, f_conf = fused_top1[row_i]
        l_name = class_names[l_id] if 0 <= l_id < len(class_names) else str(l_id)
        f_name = class_names[f_id] if 0 <= f_id < len(class_names) else str(f_id)

        good = (f_id == true_id)
        color = (0, 200, 0) if good else (0, 0, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        text1 = f"GT:{true_name}"
        text2 = f"L:{l_name}({l_conf:.2f}) F:{f_name}({f_conf:.2f})"
        _draw_label(vis, text1, x1, max(15, y1 - 18), color=(255, 255, 255))
        _draw_label(vis, text2, x1, max(30, y1 - 4), color=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

def evaluate_fusion_per_particle(
    classification_test_root: Path,
    yolo_labels_root: Path,
    local_model_path: Path,
    global_model_path: Path,
    class_names: List[str],
    fusion_weights=(0.3, 0.5, 0.2),
    sample_context: Optional[str] = None,
    max_boxes_per_image: Optional[int] = None,
    preprocess_fn=preprocess_efficientnet,
    debug_out_dir: Optional[Path] = None,
    debug_max_images: int = 25,
    debug_only_mistakes: bool = True,
    limit_images_per_class: Optional[int] = None,
    limit_seed: int = 42,
) -> Dict[str, float]:
    local_model = load_model(str(local_model_path))
    global_model = load_model(str(global_model_path))

    local_hw = infer_hw_from_keras(local_model)
    global_hw = infer_hw_from_keras(global_model)

    fusion = ParticleFusionClassifier(weights=fusion_weights, num_classes=len(class_names), use_meta_model=False)

    label_index = index_label_files(yolo_labels_root)

    if limit_images_per_class is not None and int(limit_images_per_class) > 0:
        files = sample_images_per_class(classification_test_root, n_per_class=int(limit_images_per_class), seed=int(limit_seed))
    else:
        files = iter_images(classification_test_root)

    correct = 0
    total = 0
    missing_labels = 0
    empty_labels = 0
    bad_class_ids = 0

    debug_saved = 0
    if debug_out_dir is not None:
        debug_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        stem = img_path.stem
        lbl_path = label_index.get(stem)
        if lbl_path is None:
            missing_labels += 1
            continue

        labels = read_yolo_labels(lbl_path)
        if not labels:
            empty_labels += 1
            continue

        labels_ok = []
        for lbl in labels:
            if 0 <= lbl.cls < len(class_names):
                labels_ok.append(lbl)
            else:
                bad_class_ids += 1

        if not labels_ok:
            empty_labels += 1
            continue

        H, W = img.shape[:2]

        # Global
        g_h, g_w = global_hw
        g_bgr = cv2.resize(img, (g_w, g_h))
        g_rgb = cv2.cvtColor(g_bgr, cv2.COLOR_BGR2RGB)
        g_batch = preprocess_fn(np.expand_dims(g_rgb, axis=0).astype(np.uint8))
        g_probs = global_model.predict(g_batch, verbose=0)[0].astype(np.float32)
        g_top1 = _top1(g_probs)

        # Align (label, box) pairs + optional top-k by area
        pairs = []
        for lbl in labels_ok:
            box = yolo_to_xyxy(lbl, W, H)
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            pairs.append((area, lbl, box))
        pairs.sort(key=lambda t: t[0], reverse=True)
        if max_boxes_per_image and max_boxes_per_image > 0:
            pairs = pairs[:max_boxes_per_image]

        aligned_labels = [t[1] for t in pairs]
        aligned_boxes = [t[2] for t in pairs]

        # Local batch
        l_batch_rgb_u8, kept = make_batch_from_crops(img, aligned_boxes, out_hw=local_hw)
        if l_batch_rgb_u8.shape[0] == 0:
            continue

        l_batch = preprocess_fn(l_batch_rgb_u8)
        l_probs_batch = local_model.predict(l_batch, verbose=0).astype(np.float32)  # (N,C)

        local_top1 = []
        fused_top1 = []
        any_mistake_in_image = False

        for row_idx, kept_i in enumerate(kept):
            lbl = aligned_labels[kept_i]
            l_probs = l_probs_batch[row_idx]

            yolo_probs = np.zeros((len(class_names),), dtype=np.float32)
            yolo_probs[lbl.cls] = 1.0

            _, _, fused_probs = fusion.predict(
                yolo_output={"probs": yolo_probs, "conf": 1.0, "class_id": int(lbl.cls)},
                local_output={"probs": l_probs},
                global_output={"probs": g_probs},
                bbox_coords=np.array([lbl.x, lbl.y, lbl.w, lbl.h], dtype=np.float32),
                sample_context=sample_context,
            )

            l_t = _top1(l_probs)
            f_t = _top1(fused_probs)
            local_top1.append(l_t)
            fused_top1.append(f_t)

            y_pred = int(f_t[0])
            y_true = int(lbl.cls)

            if y_pred != y_true:
                any_mistake_in_image = True

            correct += int(y_pred == y_true)
            total += 1

        # Save debug visualization
        if debug_out_dir is not None and debug_saved < int(debug_max_images):
            should_save = (any_mistake_in_image if debug_only_mistakes else True)
            if should_save:
                out_png = debug_out_dir / f"{img_path.stem}_debug.png"
                _save_debug_image(
                    out_path=out_png,
                    img_bgr=img,
                    boxes_xyxy=aligned_boxes,
                    labels=aligned_labels,
                    class_names=class_names,
                    local_top1=local_top1,
                    fused_top1=fused_top1,
                    global_top1=g_top1,
                    kept_indices=kept,
                )
                debug_saved += 1

    acc = correct / total if total else 0.0
    return {
        "particle_accuracy": float(acc),
        "particles_evaluated": float(total),
        "images_total": float(len(files)),
        "images_missing_label_file": float(missing_labels),
        "images_with_empty_labels": float(empty_labels),
        "labels_with_bad_class_id": float(bad_class_ids),
        "debug_images_saved": float(debug_saved),
    }


if __name__ == "__main__":
    metrics = evaluate_fusion_per_particle(
        classification_test_root=Path(r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv3\test"),
        yolo_labels_root=Path(r"C:\Users\joshk\OneDrive\Desktop\multiclass\v3\labels"),
        local_model_path=Path("models/local/EfficientNetB0/best_model.keras"),
        global_model_path=Path("models/global/EfficientNetB0/best_model.keras"),
        class_names=["nylon", "pe", "pmma", "ps", "pp", "pu", "pvc"],
        fusion_weights=(0, 0.7, 0.3),
        sample_context="pure",
        max_boxes_per_image=20,
        preprocess_fn=preprocess_efficientnet,
        debug_out_dir=Path("results/classification/images"),
        debug_max_images=70,
        debug_only_mistakes=False,
        limit_images_per_class=10,
        limit_seed=42,
    )
    print(metrics)