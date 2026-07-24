"""
finetune_local.py — finish the fine-tuning of a crashed classifier run.

The local EfficientNet-B0 run crashed partway through its fine-tune phase, so
`best_model.pt` holds the best weights reached so far (full pretrain + a few
fine-tune epochs). This script warm-starts from that checkpoint and runs the
fine-tune phase to completion — it reproduces exactly what pipeline.py Phase 2
does (unfreeze the top layers, SGD at a low LR, best-checkpoint on val_accuracy),
but starting from your saved weights instead of from a fresh model.

It does NOT retrain from scratch: epochs here are *additional* fine-tune epochs.

------------------------------------------------------------------------------
Run on Kaggle / Colab (recommended — GPU) or locally.
  - Add the LOCAL dataset (train/ val/ test/) as an input.
  - Upload the crashed best_model.pt (e.g. as a Kaggle dataset, or put it in Drive).
  - Edit the CONFIG block, then run.
------------------------------------------------------------------------------
"""
import glob
import os
import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------- CONFIG (edit)
MODEL_NAME   = "efficient"          # architecture the checkpoint was trained as
CKPT         = ""                   # path to crashed best_model.pt ("" = auto-find)
DATA_ROOT    = ""                   # dataset root w/ train/ val/ test/ ("" = auto-find)
FT_EPOCHS    = 15                   # additional fine-tune epochs (early-stops, patience=5)
FT_LR        = 1e-4                 # fine-tune LR (pipeline Phase 2 used 1e-4; lower if it looks unstable)
UNFREEZE_LAST_N = 20               # backbone tensors to unfreeze (pipeline default)
BATCH_SIZE   = 64
STEPS_PER_EPOCH = None              # cap train batches/epoch (None = full). Local set is big.
VAL_STEPS    = None
OUT_DIR      = ""                   # where to save ("" = <repo>/results/local/<model>/finetune_resume)
# ------------------------------------------------------------------------------

# Make the `training.classification` package importable no matter where this runs
HERE = Path(__file__).resolve()
REPO_ROOT = next((p for p in HERE.parents if (p / "training" / "classification").is_dir()), None)
if REPO_ROOT is None:  # Kaggle/Colab: search the input/working tree
    hits = glob.glob("/kaggle/**/training/classification/pipeline.py", recursive=True) \
        or glob.glob("/content/**/training/classification/pipeline.py", recursive=True)
    REPO_ROOT = Path(hits[0]).parents[2] if hits else HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.classification.config import ExperimentConfig
from training.classification.data import get_tfdata_datasets
from training.classification.fine_tune import fine_tune_model, make_sgd_optimizer
from training.classification.train import train_model


def _auto_find_dataset() -> str:
    """Find a dataset root containing train/ (prefer one with 'local' in the path)."""
    if DATA_ROOT:
        return DATA_ROOT
    cands = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train", recursive=True)
             if os.path.isdir(p) and "training/classification" not in p]
    cands += [os.path.dirname(p) for p in glob.glob("/content/**/train", recursive=True)
              if os.path.isdir(p)]
    assert cands, "No dataset with a train/ folder found — set DATA_ROOT."
    return next((c for c in cands if "local" in c.lower()), cands[0])


def _auto_find_ckpt(ckpt: str) -> str:
    if ckpt:
        return ckpt
    hits = (glob.glob("/kaggle/input/**/best_model.pt", recursive=True)
            or glob.glob("/content/**/best_model.pt", recursive=True))
    assert hits, "best_model.pt not found — pass --ckpt with the checkpoint path."
    return hits[0]


def _parse_args():
    """CLI args override the CONFIG defaults — so you DON'T have to edit this file
    in a Colab session (the prep cell's `unzip -o` would overwrite your edits)."""
    import argparse
    ap = argparse.ArgumentParser(description="Continue fine-tuning from a checkpoint.")
    ap.add_argument("--ckpt", default=CKPT, help="Path to best_model.pt to warm-start from.")
    ap.add_argument("--data-root", default=DATA_ROOT, help="Dataset root w/ train/ val/ test/.")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--ft-epochs", type=int, default=FT_EPOCHS)
    ap.add_argument("--ft-lr", type=float, default=FT_LR)
    ap.add_argument("--unfreeze", type=int, default=UNFREEZE_LAST_N)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--steps", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--val-steps", type=int, default=VAL_STEPS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    return ap.parse_args()


def main():
    args = _parse_args()
    data_root = _auto_find_dataset() if not args.data_root else args.data_root
    ckpt = _auto_find_ckpt(args.ckpt)
    out_dir = args.out_dir or str(REPO_ROOT / "results" / "local" / args.model / "finetune_resume")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = 0 if os.name == "nt" else 4   # parallel loading only off Windows

    print(f"[cfg] device={device}  data_root={data_root}")
    print(f"[cfg] checkpoint={ckpt}")
    print(f"[cfg] out_dir={out_dir}  model={args.model}  ft_epochs={args.ft_epochs}  ft_lr={args.ft_lr}")

    config = ExperimentConfig(
        training_type="microplastic",
        learning_rate=args.ft_lr,
        base_dirs={"microplastic": data_root, "whisky": ""},
        microplastic_classes=["nylon", "pe", "pet", "pla", "pmma", "pp", "ps", "pu", "pvc"],
        whisky_classes=[],
        weights_path="",
        save_path=out_dir,
        epochs=args.ft_epochs,
        batch_size=args.batch_size,
        model_name=args.model,
        steps_per_epoch=args.steps,
    )

    # 1. Data
    train_ds, val_ds = get_tfdata_datasets(
        config, model_name=args.model, balance_mode="none",
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    print(f"[data] train={len(train_ds.dataset)}  val={len(val_ds.dataset)}")

    # 2. Warm-start from the crashed checkpoint (full model object, not a state_dict)
    loaded = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        raise TypeError(f"{ckpt} is a state_dict, not a full model — re-save as a full model.")
    model = loaded.to(device)

    # 3. Fine-tune phase: unfreeze top layers + SGD at low LR (mirrors pipeline Phase 2)
    model = fine_tune_model(model, model_name=args.model, unfreeze_last_n=args.unfreeze)
    ft_optimizer = make_sgd_optimizer(model, learning_rate=args.ft_lr, momentum=0.9)

    # 4. Train — best_model.pt (best val_accuracy) is written into out_dir/<version>
    history = train_model(
        model, train_ds, val_ds, config,
        version="finetune_resume",
        steps_per_epoch=args.steps,
        val_steps=args.val_steps,
        history_filename="history_finetune_resume.csv",
        optimizer=ft_optimizer,
        burn_in_epochs=0,
    )
    print(f"\n[done] best weights -> {out_dir}/finetune_resume/best_model.pt")
    return history


if __name__ == "__main__":
    main()
