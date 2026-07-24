"""
Run a trained PolyVision classifier on a folder of UNLABELED images.

Outputs one CSV row per image: path, predicted class, confidence, and the full
per-class probability vector. Use this for new images the model has never seen
(not the labeled test set — for that use `--mode analyze`).

Preprocessing (resize + normalization) is pulled from training/classification/data.py
so it EXACTLY matches how the model was trained. Class index -> name mapping follows
ImageFolder's alphabetical order, which is how the model was trained.

Usage (run from repo root so the model class is importable for torch.load):
    python -m scripts.predict_folder \
        --model-path results/local/efficient/v1.0/best_model.pt \
        --model efficient \
        --in-dir "C:/path/to/unseen_images" \
        --out-csv predictions.csv

    # If your images sit in per-class subfolders and you DO know the truth,
    # add --labeled to also print an accuracy summary.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the EXACT preprocessing + robust loader the training pipeline uses.
from training.classification.data import _get_model_io, _make_transforms, _tif_safe_loader

# Alphabetical order = torchvision ImageFolder's class_to_idx ordering (v8, 9 classes).
DEFAULT_CLASSES = ["nylon", "pe", "pet", "pla", "pmma", "pp", "ps", "pu", "pvc"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def main():
    """CLI: run a trained model over every image in a folder and write predictions."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="Path to best_model.pt (full model object).")
    ap.add_argument("--model", default="efficient",
                    help="Backbone name used at training (efficient/inception/res) — sets img size + normalization.")
    ap.add_argument("--in-dir", required=True, help="Folder of images to classify (searched recursively).")
    ap.add_argument("--out-csv", default="predictions.csv")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Class names in ImageFolder (alphabetical) order. Must match training.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--labeled", action="store_true",
                    help="Treat each image's parent folder name as its true class and print accuracy.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Inference only (eval + no_grad) — the STABLE GPU path; won't BSOD on Blackwell.
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model = model.to(device).eval()

    n_out = model.head[-1].out_features
    classes = args.classes
    if len(classes) != n_out:
        print(f"[WARN] model has {n_out} outputs but {len(classes)} class names given — "
              f"using index labels for the extras. Check --classes ordering!")
        classes = (classes + [f"class_{i}" for i in range(n_out)])[:n_out]

    image_size, mean, std = _get_model_io(args.model)
    tf = _make_transforms(image_size, mean, std, augment=False)

    in_dir = Path(args.in_dir)
    paths = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not paths:
        print(f"No images found under {in_dir}")
        return
    print(f"[Predict] {len(paths)} images, {n_out} classes, batch={args.batch_size}")

    rows = []
    correct = total = 0
    batch_imgs, batch_paths = [], []

    def flush():
        """Run the buffered batch through the model and record predictions."""
        nonlocal correct, total
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu()
        for p, pr in zip(batch_paths, probs):
            idx = int(pr.argmax())
            pred = classes[idx]
            row = {"path": str(p), "predicted": pred, "confidence": round(float(pr[idx]), 4)}
            for c, v in zip(classes, pr.tolist()):
                row[f"p_{c}"] = round(float(v), 4)
            if args.labeled:
                truth = p.parent.name.lower()
                row["true"] = truth
                if truth in classes:
                    total += 1
                    correct += int(truth == pred)
            rows.append(row)
        batch_imgs.clear()
        batch_paths.clear()

    for i, p in enumerate(paths, 1):
        try:
            img = _tif_safe_loader(str(p))
            batch_imgs.append(tf(img))
            batch_paths.append(p)
        except Exception as e:
            print(f"[skip] {p}: {e}")
        if len(batch_imgs) >= args.batch_size:
            flush()
        if i % 500 == 0:
            print(f"  {i}/{len(paths)}")
    flush()

    fieldnames = ["path", "predicted", "confidence"] + \
                 (["true"] if args.labeled else []) + [f"p_{c}" for c in classes]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[Done] wrote {len(rows)} predictions -> {args.out_csv}")

    # Per-class prediction counts
    from collections import Counter
    counts = Counter(r["predicted"] for r in rows)
    print("[Summary] predicted class counts:")
    for c in classes:
        print(f"  {c:8s}: {counts.get(c, 0)}")
    if args.labeled and total:
        print(f"[Accuracy] {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    main()
