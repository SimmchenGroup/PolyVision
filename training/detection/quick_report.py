"""
Quick evaluation report for a trained YOLOv8 detector.

Reads a dataset's `data.yaml`, resolves the requested split, lists its images and YOLO
labels, runs the detector, and summarises detection performance (per-class and overall)
without the full Ultralytics validation harness — a fast sanity check on a trained
detector against a chosen split.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class YoloBox:
    cls_id: int
    x_c: float
    y_c: float
    w: float
    h: float


@dataclass(frozen=True)
class Sample:
    split: str
    image_path: Path
    label_path: Path | None  # None if missing


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Quick dataset + training plots for YOLO datasets.")
    p.add_argument("--data-yaml", type=Path, required=True, help="Path to YOLO data.yaml")
    p.add_argument("--out-dir", type=Path, default=Path("report_figs"), help="Output folder for figures")
    p.add_argument("--results-csv", type=Path, default=None, help="Ultralytics results.csv (optional)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max-samples", type=int, default=50000, help="Cap number of images scanned per split")
    p.add_argument("--overlay-n", type=int, default=12, help="How many images to show in overlay grids per split")
    p.add_argument("--dup-hash-limit", type=int, default=8000, help="Only hash up to N images total for duplicates (speed)")
    p.add_argument("--imgsz", type=int, default=1024, help="Only used for display scaling; not model inference")
    return p.parse_args()


def read_data_yaml(path: Path) -> dict:
    """Load a YOLO data.yaml (split paths + class names)."""
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return d


def resolve_split_sources(data: dict, yaml_path: Path) -> dict[str, Path | str]:
    """
    Supports YOLO data.yaml fields:
      train/val/test: can be a dir, a filelist .txt, or a relative path.
      path: optional base path.
    """
    base = None
    if "path" in data and data["path"]:
        base = (yaml_path.parent / Path(str(data["path"]))).resolve()

    splits = {}
    for split in ("train", "val", "test"):
        if split in data and data[split]:
            raw = str(data[split])
            p = Path(raw)
            if not p.is_absolute():
                p = ((base if base else yaml_path.parent) / p).resolve()
            splits[split] = p
    return splits


def list_images_from_source(source: Path, max_n: int) -> list[Path]:
    """
    If source is:
      - directory: recursively find image files
      - .txt: treat as filelist of image paths (relative to txt file)
      - single image: return it
    """
    if source.is_file() and source.suffix.lower() == ".txt":
        out: list[Path] = []
        base = source.parent
        for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (base / p).resolve()
            if p.suffix.lower() in IMG_EXTS and p.exists():
                out.append(p)
            if len(out) >= max_n:
                break
        return out

    if source.is_dir():
        imgs = []
        for p in source.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                imgs.append(p.resolve())
                if len(imgs) >= max_n:
                    break
        return sorted(imgs)

    if source.is_file() and source.suffix.lower() in IMG_EXTS:
        return [source.resolve()]

    return []


def infer_label_path(img_path: Path) -> Path:
    """
    Common YOLO layout:
      .../images/<split>/abc.jpg -> .../labels/<split>/abc.txt
    Fallback:
      same folder as image -> abc.txt
    """
    parts = list(img_path.parts)
    if "images" in parts:
        i = parts.index("images")
        label_parts = parts[:]
        label_parts[i] = "labels"
        label_parts[-1] = img_path.with_suffix(".txt").name
        return Path(*label_parts)
    return img_path.with_suffix(".txt")


def parse_yolo_label_file(label_path: Path) -> tuple[list[YoloBox], list[str]]:
    """
    Returns (boxes, errors). Boxes are normalized xywh.
    """
    errors: list[str] = []
    boxes: list[YoloBox] = []
    try:
        text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as e:
        return [], [f"read_error: {e}"]

    if not text:
        return [], []  # empty label file is allowed but flagged elsewhere

    for ln, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 5:
            errors.append(f"line_{ln}: expected 5+ tokens, got {len(toks)}")
            continue
        try:
            cls_id = int(float(toks[0]))
            x_c, y_c, w, h = map(float, toks[1:5])
        except Exception as e:
            errors.append(f"line_{ln}: parse_error: {e}")
            continue

        # Basic sanity checks (still keep box, but report)
        if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            errors.append(f"line_{ln}: values_out_of_range")
        boxes.append(YoloBox(cls_id, x_c, y_c, w, h))

    return boxes, errors


def load_gray(img_path: Path) -> np.ndarray | None:
    """Load an image as greyscale."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return img


def yolo_to_xyxy_abs(box: YoloBox, w_img: int, h_img: int) -> tuple[int, int, int, int]:
    """Convert a normalised YOLO box to absolute (x1, y1, x2, y2) pixel coordinates."""
    x1 = int(round((box.x_c - box.w / 2) * w_img))
    y1 = int(round((box.y_c - box.h / 2) * h_img))
    x2 = int(round((box.x_c + box.w / 2) * w_img))
    y2 = int(round((box.y_c + box.h / 2) * h_img))
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img, x2))
    y2 = max(0, min(h_img, y2))
    return x1, y1, x2, y2


def sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-1 hash of a file, read in chunks."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_out_dir(out_dir: Path) -> None:
    """Create the output directory if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df_boxes: pd.DataFrame, class_names: list[str] | None, out_dir: Path) -> None:
    """Bar plot of bounding-box counts per class."""
    if df_boxes.empty:
        return

    counts = df_boxes["cls_id"].value_counts().sort_index()
    labels = []
    for k in counts.index.tolist():
        if class_names and 0 <= int(k) < len(class_names):
            labels.append(class_names[int(k)])
        else:
            labels.append(str(int(k)))

    total = counts.sum()
    pct = counts / max(1, total) * 100.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, counts.values)
    ax.set_title("Class distribution (objects)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    # percentage labels
    for i, (c, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, c, f"{p:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "dataset_class_distribution_objects.png", dpi=200)
    plt.close(fig)


def plot_split_sizes(df_imgs: pd.DataFrame, df_boxes: pd.DataFrame, out_dir: Path) -> None:
    """Bar plot of image and box counts per split."""
    splits = ["train", "val"]
    img_counts = df_imgs.groupby("split")["image_path"].nunique().reindex(splits).fillna(0).astype(int)
    obj_counts = df_boxes.groupby("split")["cls_id"].count().reindex(splits).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(splits))
    ax.bar(x, img_counts.values, label="Images")
    ax.bar(x, obj_counts.values, bottom=img_counts.values, label="Objects")
    ax.set_xticks(x, splits)
    ax.set_title("Dataset split sizes")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_split_sizes_images_objects_stacked.png", dpi=200)
    plt.close(fig)


def plot_bbox_geometry(df_boxes: pd.DataFrame, out_dir: Path) -> None:
    """Distributions of box width, height, and aspect ratio."""
    if df_boxes.empty:
        return

    # area in normalized coordinates; you can also multiply by image area later if you store H/W
    area = (df_boxes["w"] * df_boxes["h"]).to_numpy()
    ar = (df_boxes["w"] / np.clip(df_boxes["h"], 1e-9, None)).to_numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(area, bins=50)
    axs[0].set_title("BBox area (normalized)")
    axs[0].set_xlabel("w*h")
    axs[0].set_ylabel("Count")

    axs[1].hist(ar, bins=50)
    axs[1].set_title("Aspect ratio (w/h)")
    axs[1].set_xlabel("w/h")
    axs[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_dir / "dataset_bbox_area_aspect_ratio.png", dpi=200)
    plt.close(fig)


def plot_objects_per_image(df_boxes: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of the number of detections per image."""
    if df_boxes.empty:
        return
    per_img = df_boxes.groupby(["split", "image_path"]).size().reset_index(name="n_obj")

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, g in per_img.groupby("split"):
        ax.hist(g["n_obj"], bins=30, alpha=0.5, label=split)
    ax.set_title("Objects per image")
    ax.set_xlabel("# objects")
    ax.set_ylabel("Images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_objects_per_image_hist.png", dpi=200)
    plt.close(fig)


def draw_overlay_grid(samples: list[Sample], df_boxes: pd.DataFrame, class_names: list[str] | None, out_path: Path, n: int, seed: int) -> None:
    """Grid of sample images with their bounding boxes drawn."""
    rng = random.Random(seed)
    if not samples:
        return

    # choose from samples that are readable
    candidates = []
    for s in samples:
        img = load_gray(s.image_path)
        if img is not None:
            candidates.append(s)
    if not candidates:
        return

    chosen = rng.sample(candidates, k=min(n, len(candidates)))

    cols = 4
    rows = math.ceil(len(chosen) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = np.array(axs).reshape(rows, cols)

    for ax in axs.ravel():
        ax.axis("off")

    for i, s in enumerate(chosen):
        ax = axs[i // cols, i % cols]
        img = load_gray(s.image_path)
        if img is None:
            ax.set_title(f"{s.image_path.name}\n(corrupt)")
            continue

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h_img, w_img = vis.shape[:2]

        g = df_boxes[df_boxes["image_path"] == str(s.image_path)]
        for _, r in g.iterrows():
            box = YoloBox(int(r["cls_id"]), float(r["x_c"]), float(r["y_c"]), float(r["w"]), float(r["h"]))
            x1, y1, x2, y2 = yolo_to_xyxy_abs(box, w_img, h_img)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = str(box.cls_id)
            if class_names and 0 <= box.cls_id < len(class_names):
                name = class_names[box.cls_id]
            cv2.putText(vis, name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ax.imshow(vis[..., ::-1])
        ax.set_title(s.image_path.name, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def augment_preview(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Simple, report-friendly augmentations."""
    out: list[tuple[str, np.ndarray]] = []
    out.append(("original", img))

    out.append(("hflip", cv2.flip(img, 1)))
    out.append(("vflip", cv2.flip(img, 0)))

    # rotate 15 degrees
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out.append(("rot15", rot))

    # brightness/contrast
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=25)
    out.append(("bright+25", bright))

    # blur
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    out.append(("blur", blur))

    # gaussian noise
    noise = (np.random.default_rng(0).normal(0, 10, size=img.shape)).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out.append(("noise", noisy))

    return out


def plot_augmentation_panel(samples: list[Sample], out_dir: Path, seed: int) -> None:
    """Panel illustrating the training augmentations on sample images."""
    rng = random.Random(seed)
    readable = []
    for s in samples:
        img = load_gray(s.image_path)
        if img is not None:
            readable.append((s, img))
    if not readable:
        return

    s, img = rng.choice(readable)
    augs = augment_preview(img)

    cols = 4
    rows = math.ceil(len(augs) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = np.array(axs).reshape(rows, cols)
    for ax in axs.ravel():
        ax.axis("off")

    for i, (name, im) in enumerate(augs):
        ax = axs[i // cols, i % cols]
        ax.imshow(im, cmap="gray")
        ax.set_title(name)

    fig.suptitle(f"Augmentation preview (sample: {s.image_path.name})", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_augmentation_preview.png", dpi=200)
    plt.close(fig)


def plot_quality_checks(quality: dict[str, int], out_dir: Path) -> None:
    """Plot the results of the dataset quality checks."""
    keys = list(quality.keys())
    vals = [quality[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(keys, vals)
    ax.set_title("Data quality checks")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_quality_checks.png", dpi=200)
    plt.close(fig)


def plot_training_dynamics(results_csv: Path, out_dir: Path) -> None:
    """Plot detector training curves from an Ultralytics results.csv."""
    df = pd.read_csv(results_csv)
    if df.empty:
        return

    # Identify epoch column
    epoch_col = "epoch" if "epoch" in df.columns else None
    x = df[epoch_col].to_numpy() if epoch_col else np.arange(len(df))

    # Loss columns (common Ultralytics names)
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    metric_cols = [c for c in df.columns if any(k in c.lower() for k in ("precision", "recall", "map", "f1"))]

    def _plot_lines(cols: list[str], title: str, fname: str) -> None:
        """Helper: line-plot the given result columns onto one axis."""
        if not cols:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in cols:
            ax.plot(x, df[c].to_numpy(), label=c)
        ax.set_title(title)
        ax.set_xlabel("Epoch" if epoch_col else "Step")
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=2, fontsize=8)

        # best epoch marker (if there is a "best" metric like mAP)
        best_key = None
        for candidate in ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/mAP50-95", "metrics/mAP50"):
            if candidate in df.columns:
                best_key = candidate
                break
        if best_key:
            best_idx = int(np.nanargmax(df[best_key].to_numpy()))
            ax.axvline(x[best_idx], color="k", linestyle="--", linewidth=1)
            ax.text(x[best_idx], ax.get_ylim()[1], f" best@{best_key}", va="top", ha="left", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=200)
        plt.close(fig)

    _plot_lines(loss_cols, "Training dynamics: loss curves", "training_loss_curves.png")
    _plot_lines(metric_cols, "Training dynamics: metrics", "training_metric_curves.png")


def main() -> None:
    """CLI: generate the quick evaluation report for a trained detector."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_out_dir(args.out_dir)

    data = read_data_yaml(args.data_yaml)
    splits = resolve_split_sources(data, args.data_yaml)

    # class names
    class_names = None
    if "names" in data and isinstance(data["names"], (list, tuple)):
        class_names = [str(x) for x in data["names"]]

    # Collect samples + parse labels
    samples: list[Sample] = []
    rows_imgs: list[dict] = []
    rows_boxes: list[dict] = []

    quality = {
        "images_total": 0,
        "images_corrupt": 0,
        "labels_missing": 0,
        "labels_empty": 0,
        "labels_parse_error_lines": 0,
        "labels_read_error": 0,
    }

    for split, source in splits.items():
        img_paths = list_images_from_source(Path(source), max_n=args.max_samples)
        for img_path in img_paths:
            quality["images_total"] += 1
            label_path = infer_label_path(img_path)
            if not label_path.exists():
                quality["labels_missing"] += 1
                samples.append(Sample(split, img_path, None))
            else:
                samples.append(Sample(split, img_path, label_path))

            img = load_gray(img_path)
            if img is None:
                quality["images_corrupt"] += 1
                continue

            rows_imgs.append({"split": split, "image_path": str(img_path), "h": int(img.shape[0]), "w": int(img.shape[1])})

            if label_path.exists():
                boxes, errs = parse_yolo_label_file(label_path)
                if any(e.startswith("read_error") for e in errs):
                    quality["labels_read_error"] += 1
                quality["labels_parse_error_lines"] += sum(("parse_error" in e or "expected" in e) for e in errs)

                if len(boxes) == 0:
                    # distinguish empty file vs no objects after parse
                    try:
                        if label_path.read_text(encoding="utf-8", errors="ignore").strip() == "":
                            quality["labels_empty"] += 1
                    except Exception:
                        pass

                for b in boxes:
                    rows_boxes.append(
                        {
                            "split": split,
                            "image_path": str(img_path),
                            "cls_id": int(b.cls_id),
                            "x_c": float(b.x_c),
                            "y_c": float(b.y_c),
                            "w": float(b.w),
                            "h": float(b.h),
                        }
                    )

    df_imgs = pd.DataFrame(rows_imgs)
    df_boxes = pd.DataFrame(rows_boxes)

    # Duplicate detection (hashing can be slow, so cap)
    all_img_paths = [Path(s.image_path) for s in samples]
    all_img_paths = [p for p in all_img_paths if p.exists()]
    hash_paths = all_img_paths[: min(len(all_img_paths), args.dup_hash_limit)]
    hashes: dict[str, list[Path]] = {}
    for p in hash_paths:
        try:
            h = sha1_file(p)
        except Exception:
            continue
        hashes.setdefault(h, []).append(p)
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    quality["duplicate_groups_found"] = len(dup_groups)

    # ---- Plots ----
    plot_class_distribution(df_boxes, class_names, args.out_dir)
    plot_split_sizes(df_imgs, df_boxes, args.out_dir)
    plot_bbox_geometry(df_boxes, args.out_dir)
    plot_objects_per_image(df_boxes, args.out_dir)
    plot_quality_checks(quality, args.out_dir)

    # overlay grids per split
    for split in ("train", "val", "test"):
        ss = [s for s in samples if s.split == split]
        if ss:
            draw_overlay_grid(
                samples=ss,
                df_boxes=df_boxes,
                class_names=class_names,
                out_path=args.out_dir / f"overlay_gt_boxes_{split}.png",
                n=args.overlay_n,
                seed=args.seed,
            )

    # augmentation panel from train if available, else any split
    train_samples = [s for s in samples if s.split == "train"]
    plot_augmentation_panel(train_samples if train_samples else samples, args.out_dir, seed=args.seed)

    # training dynamics (optional)
    if args.results_csv and args.results_csv.exists():
        plot_training_dynamics(args.results_csv, args.out_dir)

    # Save a quick CSV summary for tables
    if not df_boxes.empty:
        df_boxes.to_csv(args.out_dir / "boxes_flat.csv", index=False)
    if not df_imgs.empty:
        df_imgs.to_csv(args.out_dir / "images_flat.csv", index=False)

    (args.out_dir / "README.txt").write_text(
        "Figures generated:\n"
        "- dataset_class_distribution_objects.png\n"
        "- dataset_split_sizes_images_objects_stacked.png\n"
        "- dataset_bbox_area_aspect_ratio.png\n"
        "- dataset_objects_per_image_hist.png\n"
        "- dataset_quality_checks.png\n"
        "- overlay_gt_boxes_{train,val,test}.png\n"
        "- dataset_augmentation_preview.png\n"
        "- training_loss_curves.png (if results.csv provided)\n"
        "- training_metric_curves.png (if results.csv provided)\n",
        encoding="utf-8",
    )

    print(f"[OK] Wrote figures to: {args.out_dir.resolve()}")
    if dup_groups:
        print(f"[WARN] Duplicate image groups found (hashed subset): {len(dup_groups)}")
        print("       Example group:")
        for p in dup_groups[0][:5]:
            print(f"       - {p}")


if __name__ == "__main__":
    main()

# python detect\quick_report.py --data-yaml "path\to\runs\v3\data.yaml" --out-dir detect\results\v3
# python detect\quick_report.py --data-yaml "path\to\runs\v2\data.yaml" --results-csv detect/runs/detect/train5/results.csv --out-dir detect\results\test