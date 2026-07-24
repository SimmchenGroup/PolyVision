"""
PolyVision — Classification training on Kaggle GPU (Inception / EfficientNetB0 / ResNet50)
==========================================================================================

Runs YOUR existing pipeline (training/classification/pipeline.run_full_experiment:
pretrain frozen -> fine-tune -> full post-analysis) on a Kaggle T4, where cuDNN is
enabled, DataLoader workers are safe, and the disk is fast — i.e. all three local
bottlenecks (the 0x119 Blackwell hacks, num_workers=0, the /mnt/c I/O stall) are gone.
Your WSL-only warmup self-disables here (it checks /proc/version for "microsoft").

------------------------------------------------------------------------------
UPLOAD (3 Kaggle datasets, made from local zips)
------------------------------------------------------------------------------
  polyvision-code   <- code_kaggle.zip      (training/ + configs/, a few MB)
  polyvision-global <- globalv8_kaggle.zip   (global dataset, 9 classes)
  polyvision-local  <- localv8_kaggle.zip    (local dataset, 9 classes, ~700k crops)

Each: Datasets -> New Dataset -> drag the zip -> Private -> Create.
Name them WITHOUT a version number — future dataset updates (v9, v10...) are then
just "New Version" on the same slug, so notebook inputs never need re-adding.

------------------------------------------------------------------------------
NOTEBOOK SETUP  (run GLOBAL and LOCAL in SEPARATE notebook sessions)
------------------------------------------------------------------------------
  Add Input: polyvision-code  AND  the ONE dataset for this session
             (polyvision-global for the global session, polyvision-local for local).
  For the LOCAL run, edit MODELS in Cell 2 to ONE architecture per notebook
  (e.g. ["efficient"]) — 700k crops x 3 architectures won't fit one 12hr commit,
  and a timed-out commit saves NO output.
  Settings -> Accelerator -> "GPU T4 x2"   (NOT P100 — sm_60 unsupported)
  Settings -> Internet -> ON   (torchvision downloads pretrained backbone weights)

Then set DATASET below to match the input you added, and Run All.
Download /kaggle/working/results/ from the Output tab when done.
------------------------------------------------------------------------------
"""

# ============================================================== CELL 1: locate code + GPU
import glob, os, sys
import torch

# Find the uploaded code (the folder that contains training/classification/pipeline.py)
hits = glob.glob("/kaggle/input/**/training/classification/pipeline.py", recursive=True)
assert hits, "code not found — did you add the polyvision-code dataset as an input?"
CODE_ROOT = hits[0].split("/training/")[0]
sys.path.insert(0, CODE_ROOT)
print("Code root:", CODE_ROOT)

print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert "P100" not in (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), \
    "You are on a P100 (sm_60) — switch Accelerator to GPU T4 x2 and restart."
# cuDNN ON + benchmark ON: fixed input sizes, so autotuned conv algos = free speedup.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ============================================================ CELL 2: pick dataset + config
DATASET = "global"          # <-- set to "global" or "local" to match the input you added
MODELS  = ["inception", "efficient", "res"]   # all 3 architectures

# Per-epoch caps — bound epoch time INDEPENDENT of dataset size (shuffled loader
# draws a fresh random subset each epoch, so coverage builds up over epochs).
#   global (29k imgs):  leave both None — full dataset fits a 12hr commit (~9h for all 3).
#   local  (~700k imgs): full set is ~24x global and will NOT fit 12hr -> CAP IT.
#       Measured efficient = ~310s/epoch on global (~1.2 batch/s incl. val), 30 epochs.
#       STEPS_PER_EPOCH=800, VAL_STEPS=150 -> ~13min/epoch -> ~7-8h total. MODELS=["efficient"].
#       (1500/300 would be ~12h on TRAINING ALONE — too close to the cap.)
STEPS_PER_EPOCH = None      # train batches/epoch (None = full)
VAL_STEPS       = None      # val batches/epoch  (None = full)

# Find the dataset root: the folder that has train/ val/ test/ subdirs.
cands = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train", recursive=True)
         if os.path.isdir(p) and "training/classification" not in p]
assert cands, "dataset not found — add the polyvision-globalv7 (or localv7) dataset as input."
# Prefer a root whose name matches DATASET (globalv7/localv7); else take the first.
DATA_ROOT = next((c for c in cands if DATASET in c.lower()), cands[0])
print(f"DATASET={DATASET}  DATA_ROOT={DATA_ROOT}")
print("  classes:", sorted(os.listdir(os.path.join(DATA_ROOT, "train"))))

from training.classification.config import ExperimentConfig

def make_config(model_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        training_type="microplastic",
        learning_rate=1e-5,                       # overridden per-backbone in build_model()
        base_dirs={"microplastic": DATA_ROOT, "whisky": ""},
        microplastic_classes=["nylon", "pe", "pet", "pla", "pmma", "pp", "ps", "pu", "pvc"],  # 9 classes (v8 added pla)
        whisky_classes=[],
        weights_path="",
        save_path=f"/kaggle/working/results/{DATASET}/{model_name}",
        epochs=15,
        batch_size=64,                            # drop to 32 if CUDA OOM (Inception @299px)
        model_name=model_name,
        steps_per_epoch=STEPS_PER_EPOCH,          # caps train batches/epoch (local)
    )


# ===================================================================== CELL 3: train all 3
from training.classification.pipeline import run_full_experiment

for model_name in MODELS:
    print("\n" + "=" * 70)
    print(f"  TRAINING  {DATASET} / {model_name}")
    print("=" * 70)
    try:
        cfg = make_config(model_name)
        run_full_experiment(
            cfg,
            balance_mode="balanced",  # WeightedRandomSampler — uniform class sampling
            num_workers=4,            # fast disk on Kaggle — parallel loading is safe here
            pin_memory=True,
            persistent_workers=True,
            val_steps=VAL_STEPS,      # caps val batches/epoch (local)
            augment=True,             # flips/rotation/jitter/scale-crop on train loader
            class_weighted_loss=True, # inverse-frequency class weights in the loss
            label_smoothing=0.1,      # softens PP/PET overconfidence
        )
    except Exception as e:
        # Don't let one architecture's failure kill the other two — log and continue.
        import traceback
        print(f"[FAILED] {DATASET}/{model_name}: {type(e).__name__}: {e}")
        traceback.print_exc()

print("\nAll done. Results in /kaggle/working/results/ — download from the Output tab.")
# Best weights per run: /kaggle/working/results/<DATASET>/<model>/<version>/best_model.pt


# ================================================== CELL 3-RESUME: continue from a checkpoint
# Use this INSTEAD of Cell 3 when a previous run was cut short (e.g. GPU quota / 12hr cap)
# and you have its best_model.pt. Warm-starts from those weights instead of training fresh.
#
# Setup: upload the saved best_model.pt as a Kaggle dataset (e.g. polyvision-ckpt) and add
# it as an input, OR point CKPT at a path from a prior committed run's output.
# RESUME_MODEL must match the architecture the checkpoint was trained as.
RUN_RESUME    = False                 # flip to True to use this cell
CKPT          = "/kaggle/input/polyvision-ckpt/best_model.pt"
RESUME_MODEL  = "efficient"
RESUME_EPOCHS = 10                    # continued-training epochs (not the full 30)

if RUN_RESUME:
    import glob
    from training.classification.data import get_tfdata_datasets
    from training.classification.retrain import load_and_retrain

    assert glob.glob(CKPT), f"checkpoint not found at {CKPT} — add it as an input dataset."
    cfg = make_config(RESUME_MODEL)
    train_ld, val_ld = get_tfdata_datasets(
        cfg, model_name=RESUME_MODEL, balance_mode="none",
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    out_dir = f"/kaggle/working/results/{DATASET}/{RESUME_MODEL}/resume"
    model, hist = load_and_retrain(
        CKPT, train_ld, val_ld,
        epochs=RESUME_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,   # same cap as the original local run
        fine_tune=True, fine_tune_epochs=5,
        checkpoint_dir=out_dir,
    )
    print(f"[Resume] done — new best at {out_dir}/retrain_best_model.pt")
    # NOTE: load_and_retrain restarts the optimizer/LR fresh (continued training, not an
    # exact optimizer-state resume). Fine for incremental improvement. Val loop here uses
    # the FULL val set (it does not honor VAL_STEPS), so budget extra time on local.
