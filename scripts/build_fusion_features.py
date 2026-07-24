"""
Build the feature table for TRUE concatenation (stacking) of the three models.

For every source image it runs all three models and CONCATENATES their 9-class
probability vectors into one 27-d row:

    X = [ local_9 | global_9 | detection_9 ]      (per source image)
    y = true class  (from the complete/<class>/ folder name)
    split = 'val' or 'test'  (from the v9 dataset split — the model's own holdout)

This is the input a learned meta-classifier reads (scripts/train_fusion_meta.py),
as opposed to the parameter-free weighted average in scripts/fuse_testset.py.

Leakage control (critical): the base models were trained on the 'train' split, so we
ONLY build rows for images in the model's own 'val' and 'test' splits. The split of
each source image is read from the built dataset folders (e.g. globalv9/{val,test}/
<class>/<stem>.jpg) — the physical record of what the models held out — NOT from the
stale split_manifest.json (which predates pet/pla). We never touch 'train' images.

Images come from the nested `complete/` layout:
    complete/<class>/whole_images/<stem>.*     -> GLOBAL + DETECTION
    complete/<class>/<stem>/<crops>.tif        -> LOCAL   (the .txt labels are NOT used)

Reuses evaluate_testset (load_classifier, CLASSES, DEVICE) and
training.classification.data (_tif_safe_loader); the per-image classifier/detection
probability helpers are defined locally below.

Usage (from repo root):
    python -m scripts.build_fusion_features \
        --complete-root "path/to/data/complete" \
        --split-source  "path/to/runs/globalv9" \
        --local-model  ".../local/efficient/.../best_model.pt" \
        --global-model ".../global/efficient/.../best_model.pt" \
        --yolo-model   ".../detectv9/.../best.pt" \
        --out "fusion_features_v9.npz"
    # quick smoke test: add  --classes nylon --limit 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_testset import load_classifier, CLASSES, DEVICE
from scripts.intensity_norm import load_reference, get_ref, match_to_reference
from training.classification.data import _tif_safe_loader

K = len(CLASSES)


def _classifier_probs(model, tf, paths, norm_ref, batch_size=32):
    """(N,K) softmax probabilities for the given image paths (skips unreadable files).

    Optionally intensity-matches each image to a reference before inference.
    """
    out, buf = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(DEVICE)
        with torch.no_grad():
            out.append(F.softmax(model(x), dim=1).cpu().numpy())
        buf.clear()

    for p in paths:
        try:
            img = _tif_safe_loader(str(p))
            if norm_ref is not None:
                img = match_to_reference(img, norm_ref[0], norm_ref[1])
            buf.append(tf(img))
        except Exception as e:
            print(f"    [skip] {Path(p).name}: {e}")
        if len(buf) >= batch_size:
            flush()
    flush()
    return np.concatenate(out, axis=0) if out else np.zeros((0, K))


def _detect_vec(det, gray, conf):
    """Mean of per-detection class one-hots on one whole image, or None if no detections."""
    dets, _ = det.detect(gray, conf=conf)
    ids = [d.cls_id for d in dets if 0 <= d.cls_id < K]
    if not ids:
        return None
    v = np.zeros(K)
    for i in ids:
        v[i] += 1.0
    return v / v.sum()


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def build_split_map(split_source: Path, splits: list[str]) -> dict[tuple[str, str], str]:
    """{(class, stem): split} read from <split_source>/<split>/<class>/<stem>.* ."""
    smap: dict[tuple[str, str], str] = {}
    for split in splits:
        for cls in CLASSES:
            d = split_source / split / cls
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    smap[(cls, f.stem)] = split
    return smap


def _find_whole(complete_root: Path, cls: str, stem: str) -> Path | None:
    wdir = complete_root / cls / "whole_images"
    for ext in IMG_EXTS:
        p = wdir / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(wdir.glob(f"{stem}.*")) if wdir.is_dir() else []
    return hits[0] if hits else None


def _crops_for(complete_root: Path, cls: str, stem: str) -> list[Path]:
    cdir = complete_root / cls / stem
    if not cdir.is_dir():
        return []
    return [p for p in cdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete-root", required=True)
    ap.add_argument("--split-source", default=None, help="globalv9 root with train/val/test/ "
                    "(defines each image's split). Omit when using --all-as-test.")
    ap.add_argument("--all-as-test", action="store_true",
                    help="Ignore --split-source: enumerate EVERY source image in complete-root "
                         "and tag it split='test'. Use for an independent set like raw/testset, "
                         "where the whole folder is the held-out test.")
    ap.add_argument("--local-model", required=True)
    ap.add_argument("--global-model", required=True)
    ap.add_argument("--yolo-model", required=True)
    ap.add_argument("--model", default="efficient", help="classifier backbone")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--splits", default="val,test", help="comma list of splits to include")
    ap.add_argument("--classes", nargs="+", default=None, help="subset of classes (smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="max images per class (smoke test)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--reference", default=str(REPO_ROOT / "configs" / "intensity_reference.json"))
    ap.add_argument("--out", default="fusion_features.npz")
    args = ap.parse_args()

    complete = Path(args.complete_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    want_classes = {c.lower() for c in args.classes} if args.classes else set(CLASSES)

    if args.all_as_test:
        # Enumerate every source image directly from complete-root's whole_images/,
        # tagging all as 'test' (independent set — no train/val/test split to read).
        smap = {}
        for cls in CLASSES:
            wdir = complete / cls / "whole_images"
            if not wdir.is_dir():
                continue
            for f in wdir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    smap[(cls, f.stem)] = "test"
        print(f"[all-as-test] {len(smap)} source images from {complete}")
    else:
        if not args.split_source:
            ap.error("--split-source is required unless --all-as-test is set")
        smap = build_split_map(Path(args.split_source), splits)
        print(f"[split] {len(smap)} source images across splits {splits} from {args.split_source}")

    ref = load_reference(args.reference) if args.normalize else {}
    ref_local = get_ref(ref, "local") if args.normalize else None
    ref_global = get_ref(ref, "global") if args.normalize else None
    ref_detect = get_ref(ref, "detection") if args.normalize else None

    local_m = load_classifier(args.local_model, args.model)
    global_m = load_classifier(args.global_model, args.model)
    from polyvision.ml.yolo_detector import YoloDetector
    from polyvision.core.image_io import load_as_gray
    det = YoloDetector(args.yolo_model, device=("0" if DEVICE.type == "cuda" else "cpu"))

    X, y, split_arr, keys = [], [], [], []
    n_missing_whole = 0
    per_class_count = {c: 0 for c in CLASSES}

    # iterate deterministically: class then stem
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for (cls, stem), split in smap.items():
        by_class[cls].append((stem, split))

    for cls in CLASSES:
        if cls not in want_classes:
            continue
        items = sorted(by_class[cls])
        if args.limit:
            items = items[: args.limit]
        for stem, split in items:
            whole = _find_whole(complete, cls, stem)
            if whole is None:
                n_missing_whole += 1
                continue
            # local
            crops = _crops_for(complete, cls, stem)
            lp = _classifier_probs(*local_m, crops, ref_local, args.batch_size) if crops else np.zeros((0, K))
            l_vec = lp.mean(axis=0) if len(lp) else np.zeros(K)
            # global
            gp = _classifier_probs(*global_m, [whole], ref_global, args.batch_size)
            g_vec = gp[0] if len(gp) else np.zeros(K)
            # detection
            try:
                gray = load_as_gray(str(whole))
                if ref_detect is not None:
                    from scripts.intensity_norm import match_to_reference
                    gray = match_to_reference(gray, ref_detect[0], ref_detect[1])
                d_vec = _detect_vec(det, gray, args.conf)
            except Exception as e:
                print(f"    [skip det] {whole.name}: {e}")
                d_vec = None
            if d_vec is None:
                d_vec = np.zeros(K)

            X.append(np.concatenate([l_vec, g_vec, d_vec]).astype(np.float32))
            y.append(CLASSES.index(cls))
            split_arr.append(split)
            keys.append(f"{cls}/{stem}")
            per_class_count[cls] += 1
        print(f"[{cls}] rows={per_class_count[cls]}")

    if not X:
        print("No rows built — check paths/splits.")
        return

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    split_arr = np.array(split_arr)
    keys = np.array(keys)
    out = Path(args.out)
    np.savez_compressed(out, X=X, y=y, split=split_arr, keys=keys,
                        classes=np.array(CLASSES),
                        block_order=np.array(["local", "global", "detection"]))
    n_val = int((split_arr == "val").sum())
    n_test = int((split_arr == "test").sum())
    print(f"\n[done] {out}  X={X.shape}  val={n_val} test={n_test}  "
          f"missing_whole={n_missing_whole}")


if __name__ == "__main__":
    main()

# python -m scripts.build_fusion_features --complete-root "path/to/data/testset" --all-as-test --local-model  "path/to/runs/ft_results/results/local/efficient/finetune_resume/finetune_resume/best_model.pt" --global-model "path/to/runs/ft_results/results/global/efficient/finetune_resume/finetune_resume/best_model.pt" --yolo-model   "path/to/runs/detectv9/detect_colab_runs/detect/detect_colab_ft/weights/best.pt" --out "fusion_features_v9.npz"
#python -m scripts.train_fusion_meta --features "fusion_features_v9.npz" --test-features "testset_features.npz" --detect-train-list "path/to/runs/detectv9/train.txt"