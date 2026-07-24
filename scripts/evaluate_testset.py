"""
Evaluate the three PolyVision models on a labeled testset, against folder labels.

Testset layout (label = top-level class folder):
    testset/<class>/<source_image>/<crops>.tif      <- LOCAL classifier input
    testset/<class>/whole_images/*.jpg              <- GLOBAL + DETECTION input

Each model runs on its appropriate input and is scored vs the class folder name:
  - LOCAL  (.pt) : classify every crop                 -> per-crop accuracy
  - GLOBAL (.pt) : classify every whole image          -> per-image accuracy
  - DETECT (.pt) : YOLO on every whole image           -> per-detection class acc

Uses the NEW PyTorch .pt classifiers (not the Keras polyvision/ml classes).
Detection/global silently skip if whole_images/ is empty (populate it later and
re-run — local is unaffected). All inference is eval/no_grad = stable GPU path.

Usage (from repo root):
    python -m scripts.evaluate_testset \
        --testset-root "path/to/data/testset" \
        --local-model  results/local/efficient/v1.0/best_model.pt \
        --global-model results/global/efficient/v1.0/best_model.pt \
        --yolo-model   models/detect/YOLOv8.8/best.pt \
        --out-dir      testset_eval
"""
# python -m scripts.evaluate_testset --testset-root "path\to\data\testset" --local-model "path\to\runs\ft_results\results\local\efficient\finetune_resume\finetune_resume\best_model.pt" --global-model "path\to\runs\ft_results\results\global\efficient\finetune_resume\finetune_resume\best_model.pt" --yolo-model "path\to\runs\detectv9\detect_colab_runs\detect\detect_colab_ft\weights\best.pt"
#

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.classification.data import _get_model_io, _make_transforms, _tif_safe_loader
from scripts.intensity_norm import match_to_reference, load_reference, get_ref

# Alphabetical = ImageFolder class_to_idx order (classifiers) AND the detect data.yaml
# names order. Index -> class name for all three models.
CLASSES = ["nylon", "pe", "pet", "pla", "pmma", "pp", "ps", "pu", "pvc"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_of(path: Path, root: Path) -> str | None:
    """Top-level folder under root = the ground-truth class."""
    try:
        return path.relative_to(root).parts[0].lower()
    except (ValueError, IndexError):
        return None


def write_confusion(rows, out_csv: Path, title: str):
    """rows: list of (true, pred). Writes a confusion CSV + prints accuracy."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    correct = total = 0
    for t, p in rows:
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
            total += 1
            correct += int(t == p)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *CLASSES])
        for c in CLASSES:
            w.writerow([c, *cm[idx[c]].tolist()])
    acc = correct / total if total else 0.0
    print(f"  [{title}] accuracy = {correct}/{total} = {acc:.4f}  -> {out_csv.name}")
    # per-class recall
    for c in CLASSES:
        n = cm[idx[c]].sum()
        if n:
            print(f"      {c:8s} recall {cm[idx[c], idx[c]]}/{n} = {cm[idx[c], idx[c]]/n:.3f}")
    return acc


# ----------------------------------------------------------------------------- classifier
def load_classifier(path: str, model_name: str):
    """Load a trained classifier checkpoint onto the eval device and return it in eval mode."""
    model = torch.load(path, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
    image_size, mean, std = _get_model_io(model_name)
    tf = _make_transforms(image_size, mean, std, augment=False)
    n_out = model.head[-1].out_features
    if n_out != len(CLASSES):
        print(f"  [WARN] model has {n_out} outputs, expected {len(CLASSES)} classes.")
    return model, tf


def classify_paths(model, tf, paths, root, out_csv: Path, title: str, batch_size=32,
                   norm_ref=None):
    """Classify a list of image paths in batches, writing per-image predictions to CSV; returns (true, predicted) pairs."""
    rows_cm, csv_rows = [], []
    buf_img, buf_path = [], []

    def flush():
        """Run the buffered batch through the model and record its predictions."""
        if not buf_img:
            return
        x = torch.stack(buf_img).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu()
        for p, pr in zip(buf_path, probs):
            i = int(pr.argmax())
            pred, conf, true = CLASSES[i], float(pr[i]), class_of(p, root)
            rows_cm.append((true, pred))
            csv_rows.append({"path": str(p), "true": true, "predicted": pred,
                             "confidence": round(conf, 4)})
        buf_img.clear(); buf_path.clear()

    for n, p in enumerate(paths, 1):
        try:
            img = _tif_safe_loader(str(p))
            if norm_ref is not None:                       # test->train intensity match
                img = match_to_reference(img, norm_ref[0], norm_ref[1])
            buf_img.append(tf(img)); buf_path.append(p)
        except Exception as e:
            print(f"    [skip] {p.name}: {e}")
        if len(buf_img) >= batch_size:
            flush()
        if n % 2000 == 0:
            print(f"    {title}: {n}/{len(paths)}")
    flush()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "true", "predicted", "confidence"])
        w.writeheader(); w.writerows(csv_rows)
    return rows_cm


# ----------------------------------------------------------------------------- main
def main():
    """CLI: evaluate the Local, Global, and Detection models on the labelled test set."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset-root", required=True)
    ap.add_argument("--local-model", default=None, help="local classifier .pt")
    ap.add_argument("--global-model", default=None, help="global classifier .pt")
    ap.add_argument("--yolo-model", default=None, help="detection .pt")
    ap.add_argument("--model", default="efficient", help="backbone (sets img size + norm)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--out-dir", default="testset_eval")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true",
                    help="Match each image's intensity toward the training distribution.")
    ap.add_argument("--reference", default=str(REPO_ROOT / "configs" / "intensity_reference.json"),
                    help="Per-model intensity reference JSON (used with --normalize).")
    args = ap.parse_args()

    root = Path(args.testset_root)
    # Tag the output dir so normalised / un-normalised runs don't overwrite.
    out = Path(args.out_dir + ("_norm" if args.normalize else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Device] {DEVICE}")

    ref = {}
    if args.normalize:
        ref = load_reference(args.reference)
        print(f"[Normalize] ON — reference: {args.reference}  keys={list(ref)}")
    ref_local = get_ref(ref, "local") if args.normalize else None
    ref_global = get_ref(ref, "global") if args.normalize else None
    ref_detect = get_ref(ref, "detection") if args.normalize else None

    # Partition files once: crops vs whole_images.
    all_imgs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    whole = [p for p in all_imgs if "whole_images" in {s.lower() for s in p.parts}]
    crops = [p for p in all_imgs if p not in set(whole)]
    print(f"[Scan] {len(crops)} crops, {len(whole)} whole images")

    # --- LOCAL ---
    if args.local_model and crops:
        print("\n=== LOCAL classifier (crops) ===")
        m, tf = load_classifier(args.local_model, args.model)
        rows = classify_paths(m, tf, crops, root, out / "local_predictions.csv", "local",
                              args.batch_size, norm_ref=ref_local)
        write_confusion(rows, out / "local_confusion.csv", "LOCAL")
    elif args.local_model:
        print("\n[LOCAL] no crop images found — skipped.")

    # --- GLOBAL ---
    if args.global_model and whole:
        print("\n=== GLOBAL classifier (whole images) ===")
        m, tf = load_classifier(args.global_model, args.model)
        rows = classify_paths(m, tf, whole, root, out / "global_predictions.csv", "global",
                              args.batch_size, norm_ref=ref_global)
        write_confusion(rows, out / "global_confusion.csv", "GLOBAL")
    elif args.global_model:
        print("\n[GLOBAL] whole_images/ is empty — skipped (populate .jpg and re-run).")

    # --- DETECTION ---
    if args.yolo_model and whole:
        print("\n=== DETECTION (YOLO on whole images) ===")
        from polyvision.ml.yolo_detector import YoloDetector
        from polyvision.core.image_io import load_as_gray
        det = YoloDetector(args.yolo_model, device=("0" if DEVICE.type == "cuda" else "cpu"))
        rows_cm, csv_rows = [], []
        per_img = []  # (image true, majority-vote pred)
        for n, p in enumerate(whole, 1):
            true = class_of(p, root)
            try:
                gray = load_as_gray(str(p))
                if ref_detect is not None:                 # test->train intensity match
                    gray = match_to_reference(gray, ref_detect[0], ref_detect[1])
                dets, _ = det.detect(gray, conf=args.conf)
            except Exception as e:
                print(f"    [skip] {p.name}: {e}"); continue
            cls_names = [CLASSES[d.cls_id] for d in dets if 0 <= d.cls_id < len(CLASSES)]
            for cn in cls_names:                 # per-detection accuracy
                rows_cm.append((true, cn))
            for d, cn in zip(dets, cls_names):
                csv_rows.append({"image": str(p), "true": true, "det_class": cn,
                                 "conf": round(d.conf, 4), "bbox_rc": d.bbox_rc})
            if cls_names:                        # per-image majority vote
                per_img.append((true, Counter(cls_names).most_common(1)[0][0]))
            if n % 200 == 0:
                print(f"    detect: {n}/{len(whole)}")
        with (out / "detection_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image", "true", "det_class", "conf", "bbox_rc"])
            w.writeheader(); w.writerows(csv_rows)
        write_confusion(rows_cm, out / "detection_perdet_confusion.csv", "DETECT/per-detection")
        write_confusion(per_img, out / "detection_perimage_confusion.csv", "DETECT/per-image-vote")
    elif args.yolo_model:
        print("\n[DETECT] whole_images/ is empty — skipped (populate .jpg and re-run).")

    print(f"\n[Done] all outputs in {out}/")


if __name__ == "__main__":
    main()