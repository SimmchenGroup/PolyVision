"""
Fused %PET on the aged-PET application set — the concatenation/late-fusion of the
three models applied per source image (mirrors scripts/fuse_testset.py, but the
application samples have no per-image ground truth, so it reports %PET called,
not accuracy — like scripts/evaluate_application_set.py).

Per source image (crop-folder stem == whole-image stem):
    LOCAL   : mean softmax over that image's crops
    GLOBAL  : softmax of the whole image
    DETECT  : mean of per-detection class one-hots on the whole image
    fused   : normalize(w_local*local + w_global*global + w_det*detect), argmax
Weights renormalize per-image over whichever models produced output.

A folder's fused %PET = (# source images whose FUSED argmax == PET) / (# images),
which is directly comparable to the global (per-image) curve. Tracked by sample
colour + medium (SW/UV) over weeks 4/8/12/16.

Reuses: evaluate_application_set (FOLDER_RE, WEEKS, gather_inputs, plot_metric,
plot_uv_combined, SAMPLE_COLOURS), evaluate_testset (load_classifier, CLASSES, DEVICE).

Usage (from repo root):
    python -m scripts.fuse_application_set \
        --root "C:/Users/joshk/OneDrive/Desktop/raw/application_set" \
        --local-model  ".../local/efficient/.../best_model.pt" \
        --global-model ".../global/efficient/.../best_model.pt" \
        --yolo-model   ".../detect/.../best.pt" \
        [--w-local 0.4 --w-global 0.3 --w-det 0.3] [--normalize] [--samples blue brown] [--medium uv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.classification.data import _tif_safe_loader
from scripts.evaluate_testset import load_classifier, CLASSES, DEVICE
from scripts.evaluate_application_set import (
    FOLDER_RE, WEEKS, gather_inputs, plot_metric)
from scripts.intensity_norm import match_to_reference, load_reference, get_ref

K = len(CLASSES)


def _classifier_probs(model, tf, paths, norm_ref, batch_size=32):
    """(N,K) mean-normalized softmax for the given image paths (skips bad files)."""
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
    """Mean of per-detection class one-hots on one whole image, or None if no dets."""
    dets, _ = det.detect(gray, conf=conf)
    ids = [d.cls_id for d in dets if 0 <= d.cls_id < K]
    if not ids:
        return None
    v = np.zeros(K)
    for i in ids:
        v[i] += 1.0
    return v / v.sum()


def _fuse(vectors: dict, weights: dict) -> np.ndarray | None:
    """Weighted late fusion over whichever models are present; weights renormalized."""
    present = {k: v for k, v in vectors.items() if v is not None}
    wsum = sum(weights[k] for k in present)
    if wsum <= 0:
        return None
    p = np.zeros(K)
    for k, v in present.items():
        p += (weights[k] / wsum) * v
    s = p.sum()
    return p / s if s > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--local-model", required=True)
    ap.add_argument("--global-model", required=True)
    ap.add_argument("--yolo-model", required=True)
    ap.add_argument("--model", default="efficient", help="classifier backbone")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--pet-id", type=int, default=CLASSES.index("pet"))
    ap.add_argument("--w-local", type=float, default=0.4)
    ap.add_argument("--w-global", type=float, default=0.3)
    ap.add_argument("--w-det", type=float, default=0.3)
    ap.add_argument("--meta-model", default=None,
                    help="Path to meta_classifier.joblib (from train_fusion_meta.py). If set, "
                         "the LEARNED stacked model decides each image instead of the weighted "
                         "average — the [local|global|detection] 27-vector is fed to it.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--reference", default=str(REPO_ROOT / "configs" / "intensity_reference.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--samples", nargs="+", default=None)
    ap.add_argument("--medium", choices=["sw", "uv"], default=None)
    args = ap.parse_args()

    allowed = {s.lower() for s in args.samples} if args.samples else None
    allowed_medium = args.medium.lower() if args.medium else None
    weights = {"local": args.w_local, "global": args.w_global, "detection": args.w_det}

    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root / "evaluation"
    out = out / ("normalized" if args.normalize else "raw")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Device] {DEVICE}  [Normalize] {'ON' if args.normalize else 'off'}")
    print(f"[Fusion] weights local={args.w_local} global={args.w_global} det={args.w_det}")

    ref = load_reference(args.reference) if args.normalize else {}
    ref_local = get_ref(ref, "local") if args.normalize else None
    ref_global = get_ref(ref, "global") if args.normalize else None
    ref_detect = get_ref(ref, "detection") if args.normalize else None

    meta = None
    if args.meta_model:
        import joblib
        meta = joblib.load(args.meta_model)
        print(f"[Fusion] STACKED meta-model loaded from {args.meta_model} "
              f"(overrides weighted average)")

    local_m = load_classifier(args.local_model, args.model)
    global_m = load_classifier(args.global_model, args.model)
    from polyvision.ml.yolo_detector import YoloDetector
    from polyvision.core.image_io import load_as_gray
    det = YoloDetector(args.yolo_model, device=("0" if DEVICE.type == "cuda" else "cpu"))

    # results["fused"][(colour, medium)][week] = (pet_images, total_images)
    fused = defaultdict(dict)
    rows = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        m = FOLDER_RE.match(folder.name)
        if not m:
            continue
        colour, medium, week = m["colour"].lower(), m["medium"].lower(), int(m["week"])
        if allowed and colour not in allowed:
            continue
        if allowed_medium and medium != allowed_medium:
            continue

        whole, crops = gather_inputs(folder)
        if not whole:
            continue
        crops_by_key: dict[str, list] = defaultdict(list)
        for c in crops:                                   # crop parent folder == source stem
            crops_by_key[Path(c).parent.name].append(c)

        pet_imgs = tot_imgs = 0
        for wp in whole:
            key = Path(wp).stem
            # global (one image)
            g = _classifier_probs(*global_m, [wp], ref_global, args.batch_size)
            g_vec = g[0] if len(g) else None
            # local (this image's crops, if the folder was processed)
            l_vec = None
            if key in crops_by_key:
                lp = _classifier_probs(*local_m, crops_by_key[key], ref_local, args.batch_size)
                l_vec = lp.mean(axis=0) if len(lp) else None
            # detection (one image)
            d_vec = None
            try:
                gray = load_as_gray(str(wp))
                if ref_detect is not None:
                    gray = match_to_reference(gray, ref_detect[0], ref_detect[1])
                d_vec = _detect_vec(det, gray, args.conf)
            except Exception as e:
                print(f"    [skip det] {Path(wp).name}: {e}")

            if meta is not None:
                # true concatenation: [local|global|detection] 27-vector -> learned stack.
                # Missing model -> zeros, matching how the meta-model was trained.
                vec27 = np.concatenate([
                    l_vec if l_vec is not None else np.zeros(K),
                    g_vec if g_vec is not None else np.zeros(K),
                    d_vec if d_vec is not None else np.zeros(K),
                ]).reshape(1, -1).astype(np.float32)
                pred = int(meta.predict(vec27)[0])
            else:
                p = _fuse({"local": l_vec, "global": g_vec, "detection": d_vec}, weights)
                if p is None:
                    continue
                pred = int(p.argmax())
            tot_imgs += 1
            pet_imgs += int(pred == args.pet_id)

        if tot_imgs:
            fused[(colour, medium)][week] = (pet_imgs, tot_imgs)
            rows.append((colour, medium, week, pet_imgs, tot_imgs))
            print(f"[{folder.name}] fused PET {pet_imgs}/{tot_imgs} = {100*pet_imgs/tot_imgs:.1f}%")

    if not rows:
        print("\nNo results — check models and folder layout.")
        return

    tag = "stacked" if meta is not None else "weighted"
    model_label = "stacked" if meta is not None else "fused"
    csv_path = out / f"pet_summary_fused_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "sample", "medium", "week", "pet", "total", "pct_pet"])
        for colour, medium, week, pet, tot in sorted(rows):
            w.writerow([model_label, colour, medium, week, pet, tot,
                        round(100.0 * pet / tot, 3) if tot else 0.0])
    print(f"\n[csv] {csv_path}")

    ylabel = "Stacked PET (% of images)" if meta is not None else "Fused PET (% of images)"
    plot_metric(fused, out / f"pet_vs_aging_fused_{tag}.png", ylabel)


if __name__ == "__main__":
    main()

#python -m scripts.fuse_application_set --root "C:/Users/joshk/OneDrive/Desktop/raw/application_set" --local-model "C:/Users/joshk/OneDrive/Desktop/multiclass/ft_results/results/local/efficient/finetune_resume/finetune_resume/best_model.pt" --global-model "C:/Users/joshk/OneDrive/Desktop/multiclass/ft_results/results/global/efficient/finetune_resume/finetune_resume/best_model.pt" --yolo-model   "C:/Users/joshk/OneDrive/Desktop/multiclass/detectv9/detect_colab_runs/detect/detect_colab_ft/weights/best.pt --samples blue brown orange white --medium uv
