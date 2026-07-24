"""
Fuse the three PolyVision models into ONE combined per-image prediction.

The three models score different units, so fusion first aligns everything to the
SOURCE IMAGE (folder stem == whole-image stem, e.g. pet1_..._test_0):

    LOCAL   : many crops per source  -> averaged into one vector per image
    GLOBAL  : one whole image        -> one vector per image
    DETECT  : many detections/image  -> averaged into one vector per image

then combines them by weighted late fusion (soft vote):

    p(image) = normalize( w_local*local + w_global*global + w_det*detect )
    prediction = argmax p

Reads the CSVs written by scripts/evaluate_testset.py in <eval-dir>:
    local_predictions.csv, global_predictions.csv, detection_predictions.csv
Those store top-1 label + confidence (not full softmax), so each model's vector is
reconstructed as: conf on the predicted class, (1-conf)/(K-1) spread over the rest.
(If you later add full p_<class> columns to evaluate_testset.py, this script will
use them automatically — see PROB_COLS.)

Writes to <eval-dir>:
    fused_predictions.csv   (per-image: key, true, fused pred + each model's top-1)
    fused_confusion.csv     (9x9 counts, same format as the per-model confusions)

Usage (from repo root):
    python -m scripts.fuse_testset --eval-dir "C:/Users/joshk/OneDrive/Desktop/raw/testset_eval"
    # optional: --w-local 0.5 --w-global 0.2 --w-det 0.3
    # optional: --sweep      (grid-search weights on THIS testset — see caveat below)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_testset import CLASSES, write_confusion  # reuse class order + CM writer

K = len(CLASSES)
CIDX = {c: i for i, c in enumerate(CLASSES)}
PROB_COLS = [f"p_{c}" for c in CLASSES]  # used automatically if present in the CSVs


def _read(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[warn] {path.name} not found — that model is dropped from the fusion.")
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _vec_from_row(row: dict, label_key: str) -> np.ndarray | None:
    """One 9-vector for a row. Prefer full p_<class> columns; else reconstruct
    from top-1 label + confidence (conf on the class, rest spread uniformly)."""
    if all(c in row for c in PROB_COLS):
        v = np.array([float(row[c]) for c in PROB_COLS], dtype=np.float64)
        s = v.sum()
        return v / s if s > 0 else None
    lbl = (row.get(label_key) or "").lower()
    if lbl not in CIDX:
        return None
    try:
        conf = float(row.get("confidence") or row.get("conf") or 0.0)
    except ValueError:
        conf = 0.0
    conf = min(max(conf, 1.0 / K), 1.0)          # keep it a sane probability
    v = np.full(K, (1.0 - conf) / (K - 1), dtype=np.float64)
    v[CIDX[lbl]] = conf
    return v


def _aggregate(rows: list[dict], key_fn, label_key: str):
    """key -> (mean vector over that key's rows, true label, n_rows)."""
    acc: dict[str, list] = defaultdict(list)
    truth: dict[str, str] = {}
    for r in rows:
        key = key_fn(r)
        if not key:
            continue
        v = _vec_from_row(r, label_key)
        if v is None:
            continue
        acc[key].append(v)
        t = (r.get("true") or "").lower()
        if t in CIDX:
            truth.setdefault(key, t)
    out = {}
    for key, vecs in acc.items():
        out[key] = (np.mean(vecs, axis=0), truth.get(key), len(vecs))
    return out


def _local_key(r: dict) -> str:
    # crop path: .../testset/<class>/<source>/<crop>.tif  -> <source> folder name
    return Path(r["path"]).parent.name


def _whole_key(r: dict) -> str:
    # whole-image path: .../whole_images/<name>.jpg  -> <name> stem
    return Path(r.get("path") or r.get("image")).stem


def fuse(eval_dir: Path, weights: dict[str, float]):
    local = _aggregate(_read(eval_dir / "local_predictions.csv"), _local_key, "predicted")
    glob = _aggregate(_read(eval_dir / "global_predictions.csv"), _whole_key, "predicted")
    det = _aggregate(_read(eval_dir / "detection_predictions.csv"), _whole_key, "det_class")

    sources = {"local": local, "global": glob, "detection": det}
    sources = {k: v for k, v in sources.items() if v}  # drop empty models
    keys = set().union(*[set(s) for s in sources.values()]) if sources else set()

    fused_rows, cm_rows, out_records = [], [], []
    for key in sorted(keys):
        present = {name: s[key] for name, s in sources.items() if key in s}
        wsum = sum(weights[name] for name in present)
        if wsum <= 0:
            continue
        p = np.zeros(K)
        for name, (vec, _t, _n) in present.items():
            p += (weights[name] / wsum) * vec       # renormalize weights over available models
        true = next((present[n][1] for n in present if present[n][1]), None)
        pred = CLASSES[int(p.argmax())]
        if true is None:
            continue
        cm_rows.append((true, pred))
        top1 = {n: CLASSES[int(present[n][0].argmax())] if n in present else "" for n in sources}
        out_records.append({
            "key": key, "true": true, "fused_pred": pred,
            "fused_conf": round(float(p.max()), 4),
            "local_pred": top1.get("local", ""), "global_pred": top1.get("global", ""),
            "det_pred": top1.get("detection", ""),
            "n_crops": present.get("local", (None, None, 0))[2],
            "n_dets": present.get("detection", (None, None, 0))[2],
        })
    return cm_rows, out_records, list(sources)


def _accuracy(cm_rows) -> float:
    if not cm_rows:
        return 0.0
    return sum(t == p for t, p in cm_rows) / len(cm_rows)


def _macro_f1(cm_rows) -> float:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for t, p in cm_rows:
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1; fn[t] += 1
    f1s = []
    for c in CLASSES:
        if tp[c] + fn[c] == 0:
            continue                      # class absent from testset
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c])
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="dir with the 3 *_predictions.csv")
    ap.add_argument("--w-local", type=float, default=0.5)
    ap.add_argument("--w-global", type=float, default=0.2)
    ap.add_argument("--w-det", type=float, default=0.3)
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search weights on THIS testset (overfits — report as an "
                         "upper bound, or tune on a held-out split instead).")
    args = ap.parse_args()

    d = Path(args.eval_dir)
    weights = {"local": args.w_local, "global": args.w_global, "detection": args.w_det}

    if args.sweep:
        print("[sweep] grid over weight simplex (step 0.1)...")
        best = None
        grid = [i / 10 for i in range(11)]
        for wl in grid:
            for wg in grid:
                wd = round(1.0 - wl - wg, 4)
                if wd < 0:
                    continue
                cm_rows, _, _ = fuse(d, {"local": wl, "global": wg, "detection": wd})
                score = _macro_f1(cm_rows)
                if best is None or score > best[0]:
                    best = (score, _accuracy(cm_rows), (wl, wg, wd))
        print(f"[sweep] best macro-F1={best[0]:.4f}  acc={best[1]:.4f}  "
              f"weights(local,global,det)={best[2]}")
        weights = {"local": best[2][0], "global": best[2][1], "detection": best[2][2]}

    cm_rows, records, used = fuse(d, weights)
    print(f"\n[Fusion] models used: {used}")
    print(f"[Fusion] weights: local={weights['local']} global={weights['global']} "
          f"detection={weights['detection']}  (renormalized per-image over available models)")
    print(f"[Fusion] fused {len(records)} source images")

    # per-image prediction CSV
    with (d / "fused_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["key", "true", "fused_pred", "fused_conf",
                "local_pred", "global_pred", "det_pred", "n_crops", "n_dets"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(records)
    print(f"[wrote] {d / 'fused_predictions.csv'}")

    # confusion + accuracy + per-class recall (reuses evaluate_testset.write_confusion)
    write_confusion(cm_rows, d / "fused_confusion.csv", "FUSED (per-image)")
    print(f"[Fusion] macro-F1 = {_macro_f1(cm_rows):.4f}")


if __name__ == "__main__":
    main()
