"""
PolyVision — Classification training on Google Colab GPU (Inception / EfficientNetB0 / ResNet50)
================================================================================================

Colab twin of kaggle_train.py. Runs THE SAME pipeline
(training/classification/pipeline.run_full_experiment: pretrain frozen -> fine-tune ->
full post-analysis) with the same accuracy levers enabled (augmentation, balanced
sampling, class-weighted loss, label smoothing). Colab gives a T4 with cuDNN enabled,
safe DataLoader workers, and fast local disk — so none of the Blackwell/WSL hacks apply
(the WSL-only warmup self-disables: it checks /proc/version for "microsoft", absent here).

Colab differs from Kaggle in two ways this script handles:
  1. INPUT: there are no "Kaggle datasets". You put the zips on Google Drive and this
     notebook unzips them to /content (local SSD — far faster than reading off Drive FUSE).
  2. PERSISTENCE: /content is wiped when the session ends, and writing the per-batch
     crash log straight to Drive FUSE is slow/flaky. So we TRAIN to local /content/results
     and COPY the finished results to Drive after each model.

------------------------------------------------------------------------------
ONE-TIME SETUP ON GOOGLE DRIVE
------------------------------------------------------------------------------
Create a folder  MyDrive/PolyVision/  and upload into it:
  code_kaggle.zip      (build locally: python -m scripts.build_kaggle_code_zip)
  globalv8_kaggle.zip  (the global dataset: train/ val/ test/ of 9 class folders)
  localv8_kaggle.zip   (the local dataset — only needed for the local session)
Same zips as Kaggle — nothing new to build. Re-upload code_kaggle.zip after ANY code change.

------------------------------------------------------------------------------
NOTEBOOK SETUP
------------------------------------------------------------------------------
  Runtime -> Change runtime type -> Hardware accelerator: GPU (T4).
  Run GLOBAL and LOCAL in SEPARATE sessions. For the LOCAL run set MODELS=["efficient"]
  and keep the STEPS caps — 700k crops x 3 architectures won't fit a free-tier session.
  Paste each CELL below into its own Colab cell (or open as a notebook) and run in order.
------------------------------------------------------------------------------
"""

# ============================================================== CELL 1: mount Drive + unzip code/data
import glob, os, sys, zipfile, subprocess

from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = "/content/drive/MyDrive/PolyVision"     # <-- where you uploaded the zips
WORK       = "/content/polyvision"                    # local SSD workspace (fast, ephemeral)
os.makedirs(WORK, exist_ok=True)

def _unzip(zip_name, dest):
    """Extract a zip archive into `dest`."""
    src = os.path.join(DRIVE_ROOT, zip_name)
    assert os.path.exists(src), f"missing on Drive: {src}"
    print(f"[unzip] {src} -> {dest}")
    with zipfile.ZipFile(src) as z:
        z.extractall(dest)

# Code: unzip the same code_kaggle.zip you use on Kaggle (training/ + configs/).
_unzip("code_kaggle.zip", WORK)
hits = glob.glob(f"{WORK}/**/training/classification/pipeline.py", recursive=True)
assert hits, "code not found in code_kaggle.zip — rebuild with scripts.build_kaggle_code_zip"
CODE_ROOT = hits[0].split("/training/")[0]
sys.path.insert(0, CODE_ROOT)
print("Code root:", CODE_ROOT)

# Light deps — torch/torchvision/sklearn/numpy/matplotlib ship with Colab. Do NOT pip
# install torch here (it would replace Colab's CUDA-matched build). Just ensure the small ones.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil", "tqdm"], check=False)

import torch
print("CUDA:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert torch.cuda.is_available(), "No GPU — Runtime -> Change runtime type -> GPU (T4)."
# cuDNN ON + benchmark ON: fixed input sizes, so autotuned conv algos = free speedup.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ============================================================ CELL 2: pick dataset + config
DATASET = "global"          # <-- "global" or "local" to match the zip you unzip below
MODELS  = ["efficient"]     # start with ONE architecture to measure the new levers' delta
                            #   full sweep later: ["inception", "efficient", "res"]

# Per-epoch caps — bound epoch time INDEPENDENT of dataset size (shuffled/sampled loader
# draws a fresh subset each epoch, so coverage builds over epochs).
#   global (~29k imgs): leave both None — full dataset fits a session.
#   local  (~700k imgs): full set is ~24x global -> CAP IT (STEPS=800, VAL=150 ~ 13min/epoch).
STEPS_PER_EPOCH = None      # train batches/epoch (None = full)
VAL_STEPS       = None      # val batches/epoch  (None = full)

# Unzip the ONE dataset for this session to local SSD.
_unzip(f"{DATASET}v8_kaggle.zip", WORK)
cands = [os.path.dirname(p) for p in glob.glob(f"{WORK}/**/train", recursive=True)
         if os.path.isdir(p) and "training/classification" not in p]
assert cands, f"dataset not found — check {DATASET}v8_kaggle.zip contains train/ val/ test/"
DATA_ROOT = next((c for c in cands if DATASET in c.lower()), cands[0])
print(f"DATASET={DATASET}  DATA_ROOT={DATA_ROOT}")
print("  classes:", sorted(os.listdir(os.path.join(DATA_ROOT, "train"))))

# Checkpoints write STRAIGHT TO DRIVE every epoch (save_path below). best_model.pt is
# saved on every val-accuracy improvement, so a disconnect mid-run costs at most the
# current epoch — resume from it via Cell 3-RESUME. Everything else the run produces
# (epoch_log.csv, history CSVs, post-analysis figures, the per-batch crash_log.json
# heartbeat) also lands here on Drive, so nothing is lost when /content is wiped.
DRIVE_RESULTS = f"{DRIVE_ROOT}/results/{DATASET}"

from training.classification.config import ExperimentConfig

def make_config(model_name: str) -> ExperimentConfig:
    """Build the ExperimentConfig used for the Colab training runs (the paper's settings)."""
    return ExperimentConfig(
        training_type="microplastic",
        learning_rate=1e-5,                       # overridden per-backbone in build_model()
        base_dirs={"microplastic": DATA_ROOT, "whisky": ""},
        microplastic_classes=["nylon", "pe", "pet", "pla", "pmma", "pp", "ps", "pu", "pvc"],
        whisky_classes=[],
        weights_path="",
        save_path=f"{DRIVE_RESULTS}/{model_name}",  # -> Drive: every checkpoint persists
        epochs=15,
        batch_size=64,                            # drop to 32 if CUDA OOM (Inception @299px)
        model_name=model_name,
        steps_per_epoch=STEPS_PER_EPOCH,
    )


# ===================================================================== CELL 3: train
# save_path already points at Drive, so run_full_experiment writes every checkpoint there
# directly — no local->Drive copy needed, and progress survives a disconnect.
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
            num_workers=2,            # Colab free tier has ~2 vCPUs; 2 workers is the sweet spot
            pin_memory=True,
            persistent_workers=True,
            val_steps=VAL_STEPS,
            augment=True,             # flips/rotation/jitter/scale-crop on train loader
            class_weighted_loss=True, # inverse-frequency class weights in the loss
            label_smoothing=0.1,      # softens PP/PET overconfidence
        )
    except Exception as e:
        import traceback
        print(f"[FAILED] {DATASET}/{model_name}: {type(e).__name__}: {e}")
        traceback.print_exc()

print(f"\nAll done. Checkpoints + results already on Drive at {DRIVE_RESULTS}.")
# Best weights per run: {DRIVE_RESULTS}/<model>/<version>/best_model.pt


# ================================================== CELL 3-RESUME: continue from a checkpoint
# Use INSTEAD of Cell 3 when a previous session was cut short. Because Cell 3 checkpoints
# straight to Drive, its best_model.pt is already there — point CKPT at that prior run
# (e.g. .../results/<DATASET>/efficient/v1.0/best_model.pt). RESUME_MODEL must match the
# architecture the checkpoint was trained as. The resume also writes straight to Drive.
RUN_RESUME    = False
CKPT          = f"{DRIVE_RESULTS}/efficient/v1.0/best_model.pt"  # <- prior Drive checkpoint
RESUME_MODEL  = "efficient"
RESUME_EPOCHS = 10

if RUN_RESUME:
    from training.classification.data import get_tfdata_datasets
    from training.classification.retrain import load_and_retrain

    assert os.path.exists(CKPT), f"checkpoint not found at {CKPT} — check the path on Drive."
    cfg = make_config(RESUME_MODEL)
    train_ld, val_ld = get_tfdata_datasets(
        cfg, model_name=RESUME_MODEL, balance_mode="balanced",
        num_workers=2, pin_memory=True, persistent_workers=True, augment=True,
    )
    out_dir = f"{DRIVE_RESULTS}/{RESUME_MODEL}/resume"   # -> Drive: resume checkpoints persist
    model, hist = load_and_retrain(
        CKPT, train_ld, val_ld,
        epochs=RESUME_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        fine_tune=True, fine_tune_epochs=5,
        checkpoint_dir=out_dir,
    )
    print(f"[Resume] done — new best at {out_dir}/ on Drive")
