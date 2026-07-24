"""
Train and evaluate the TRUE concatenation (stacking) meta-classifier.

Reads the 27-d feature table from scripts/build_fusion_features.py:
    X = [local_9 | global_9 | detection_9],  y = true class,  split in {val, test}

Trains a multinomial logistic-regression meta-classifier on the VAL rows (the base
models' own holdout — not their training data) and evaluates it on the TEST rows.
For an honest comparison it also scores, on the SAME test rows:
    - each single model (argmax of its 9-block)
    - the parameter-free weighted average (the scripts/fuse_testset.py recipe)

So you can see whether the LEARNED concatenation beats the fixed-weight fusion.

Usage (from repo root):
    python -m scripts.train_fusion_meta --features fusion_features_v9.npz
    # optional: --w-local 0.4 --w-global 0.3 --w-det 0.3   (weighted baseline weights)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from scripts.evaluate_testset import CLASSES

K = len(CLASSES)


def _blocks(X):
    """Split the 27-d rows back into (local, global, detection) 9-vectors."""
    return X[:, :K], X[:, K:2 * K], X[:, 2 * K:]


def _weighted(X, weights):
    """Parameter-free weighted average -> argmax, renormalized over present blocks."""
    l, g, d = _blocks(X)
    preds = np.zeros(len(X), dtype=np.int64)
    for i in range(len(X)):
        vecs = {"local": l[i], "global": g[i], "detection": d[i]}
        present = {k: v for k, v in vecs.items() if v.sum() > 0}
        wsum = sum(weights[k] for k in present)
        p = np.zeros(K)
        for k, v in present.items():
            p += (weights[k] / wsum) * v
        preds[i] = int(p.argmax())
    return preds


def _report(name, y_true, y_pred, labels=None):
    """macro-F1 is averaged over `labels` — pass the classes actually present in the
    test ground truth so absent classes (0 support) don't drag the average down."""
    if labels is None:
        labels = range(K)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    print(f"  {name:28s} acc={acc:.4f}  macro-F1={f1:.4f}")
    return acc, f1


def main():
    """CLI: train the stacking meta-classifier on validation features and evaluate it on the test set."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=None,
                    help="npz whose VAL rows train the meta-model. Optional if --meta-model "
                         "loads an already-trained model.")
    ap.add_argument("--meta-model", default=None,
                    help="Load an existing meta_classifier.joblib instead of retraining. Use "
                         "with --test-features to just evaluate a saved stack on a new set.")
    ap.add_argument("--w-local", type=float, default=0.4)
    ap.add_argument("--w-global", type=float, default=0.3)
    ap.add_argument("--w-det", type=float, default=0.3)
    ap.add_argument("--C", type=float, default=1.0, help="LogisticRegression inverse-reg strength")
    ap.add_argument("--detect-train-list", default=None,
                    help="detectv9/train.txt — rows whose image the DETECTOR trained on are "
                         "dropped (the detection feature would be leaked). Strongly recommended: "
                         "detectv9 has no test split, so ~79%% of complete-test is in detect-train.")
    ap.add_argument("--test-features", default=None,
                    help="Optional SECOND npz to evaluate on (e.g. raw/testset built with "
                         "--all-as-test). When set, the meta-model TRAINS on the val rows of "
                         "--features and is TESTED on ALL rows of this file — the honest "
                         "in-distribution-train / independent-test setup.")
    ap.add_argument("--test-keep-substr", default=None,
                    help="Keep only test rows whose key contains this substring (e.g. '_test_'). "
                         "Use to restrict raw/testset to its genuinely-independent images and "
                         "drop stray training-pool images mixed into the folder.")
    ap.add_argument("--out-dir", default=None, help="where to write meta model + report (default: alongside features)")
    args = ap.parse_args()

    det_train = None
    if args.detect_train_list:
        from pathlib import Path as _P
        det_train = {_P(l.strip()).stem for l in _P(args.detect_train_list).read_text().splitlines() if l.strip()}

    def _load(path, tag):
        """Load an NPZ feature table (X, y, keys) and report its size."""
        d = np.load(path, allow_pickle=True)
        X_, y_, split_, keys_ = d["X"], d["y"], d["split"], d["keys"]
        if det_train is not None:
            row_stem = np.array([k.split("/", 1)[1] for k in keys_])
            clean = np.array([s not in det_train for s in row_stem])
            n_drop = int((~clean).sum())
            print(f"[detect-clean:{tag}] dropped {n_drop}/{len(keys_)} detector-trained rows; "
                  f"{int(clean.sum())} remain.")
            X_, y_, split_, keys_ = X_[clean], y_[clean], split_[clean], keys_[clean]
        return X_, y_, split_, keys_

    # --- assemble the TEST rows (what we grade on) ---
    if args.test_features:
        Xte, yte, _, keys_te = _load(args.test_features, "test-src")
    elif args.features:
        X, y, split, keys = _load(args.features, "src")
        te = split == "test"
        Xte, yte, keys_te = X[te], y[te], keys[te]
    else:
        print("Provide --test-features and/or --features.")
        return

    if args.test_keep_substr:
        keep = np.array([args.test_keep_substr in k for k in keys_te])
        print(f"[test-keep] '{args.test_keep_substr}' -> kept {int(keep.sum())}/{len(keys_te)} "
              f"test rows (dropped {int((~keep).sum())} not matching).")
        Xte, yte = Xte[keep], yte[keep]

    # --- get the meta-model: load a saved one, or train on --features val rows ---
    if args.meta_model:
        import joblib
        meta = joblib.load(args.meta_model)
        print(f"[meta] loaded existing model from {args.meta_model} (no retraining)")
    else:
        if not args.features:
            print("--features is required to TRAIN a meta-model (or pass --meta-model).")
            return
        Xtr_all, ytr_all, split_tr, _ = _load(args.features, "train-src")
        tr = split_tr == "val"
        Xtr, ytr = Xtr_all[tr], ytr_all[tr]
        if len(ytr) == 0:
            print("No val rows in --features to train on.")
            return
        meta = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
        meta.fit(Xtr, ytr)
        print(f"[meta] trained on {len(ytr)} val rows of {args.features}")

    if len(yte) == 0:
        print("No test rows to evaluate.")
        return
    print(f"[data] test rows = {len(yte)}")

    weights = {"local": args.w_local, "global": args.w_global, "detection": args.w_det}

    present = sorted(set(int(v) for v in yte))   # classes with test data (>0 support)
    absent = [CLASSES[i] for i in range(K) if i not in present]
    print(f"\n=== TEST-set scores ({len(yte)} rows, macro-F1 over {len(present)} present "
          f"classes{'; absent: ' + ', '.join(absent) if absent else ''}) ===")
    l, g, d = _blocks(Xte)
    _report("local only", yte, l.argmax(1), present)
    _report("global only", yte, g.argmax(1), present)
    _report("detection only", yte, d.argmax(1), present)
    w_pred = _weighted(Xte, weights)
    _report(f"weighted avg {tuple(weights.values())}", yte, w_pred, present)

    m_pred = meta.predict(Xte)
    print("  " + "-" * 52)
    _report("STACKED (learned concat)", yte, m_pred, present)

    print("\n=== STACKED per-class report (test) ===")
    print(classification_report(yte, m_pred, labels=range(K),
                                target_names=[c.upper() for c in CLASSES], zero_division=0))

    ref_path = args.features or args.test_features or args.meta_model
    out_dir = Path(args.out_dir) if args.out_dir else Path(ref_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(yte, m_pred, labels=range(K))
    np.savetxt(out_dir / "stacked_confusion.csv", cm, fmt="%d", delimiter=",",
               header="true\\pred," + ",".join(CLASSES), comments="")
    if not args.meta_model:   # only re-save when we trained a fresh model
        try:
            import joblib
            joblib.dump(meta, out_dir / "meta_classifier.joblib")
            print(f"[saved] {out_dir / 'meta_classifier.joblib'}")
        except Exception as e:
            print(f"[warn] could not save meta model: {e}")
    print(f"[saved] {out_dir / 'stacked_confusion.csv'}")


if __name__ == "__main__":
    main()
