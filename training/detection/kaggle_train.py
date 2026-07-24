"""
PolyVision — YOLO detection training on Kaggle GPU
==================================================

Why this exists: the local RTX 5060 Ti (Blackwell sm_120a) BSODs (0x119
VIDEO_SCHEDULER_INTERNAL_ERROR) during YOLO *training* — see the
project-wsl-gpu-setup memory. Kaggle runs datacenter GPUs (P100/T4) on Linux,
which sidestep that bug entirely. NONE of the local workarounds (cuDNN-disable,
inplace-disable) are needed here.

------------------------------------------------------------------------------
STEP-BY-STEP (do this once per dataset version)
------------------------------------------------------------------------------
0. The zip `detectv8_kaggle.zip` was built locally (images/ + labels/ +
   train.rel.txt + val.rel.txt). The .rel.txt lists use portable './images/...'
   paths so they resolve wherever the dataset is mounted.

1. Kaggle account: https://www.kaggle.com  (free). Verify your phone number in
   Settings -> this is REQUIRED to enable GPU + Internet on notebooks.

2. Create the dataset:
   kaggle.com -> Datasets -> "New Dataset" -> drag in detectv8_kaggle.zip.
   Kaggle auto-extracts the zip. Give it a title, e.g. "polyvision-detectv8".
   Set visibility Private. Click Create. (Upload of ~12.5 GB takes a while.)

3. New notebook: kaggle.com -> Code -> "New Notebook".
   - Right panel -> "Add Input" -> add your polyvision-detectv8 dataset.
     It mounts read-only at /kaggle/input/<your-dataset-slug>/
   - Right panel -> Settings -> Accelerator -> "GPU T4 x2" (recommended).
     NOTE: do NOT pick "GPU P100" — it is sm_60 and Kaggle's current PyTorch
     dropped sm_60 support, so the P100 won't be usable. T4 is sm_75 (supported).
   - Right panel -> Settings -> Internet -> ON (needed to fetch yolov8s.pt).

4. Paste the CELLS below into the notebook (one cell each) and Run All.

5. When done, download the trained weights from the notebook's Output tab:
   /kaggle/working/runs/detect/train/weights/best.pt
   Put it in the repo at models/detect/<new-version>/best.pt
------------------------------------------------------------------------------
"""

# ============================================================== CELL 1: setup
# Install ultralytics WITHOUT upgrading torch. Kaggle's pre-installed PyTorch is
# matched to their GPUs; a `pip install -U` pulls a newer torch that DROPS older
# GPU support (e.g. P100 sm_60) and breaks CUDA. So no -U, and keep torch frozen.
# (Requires Internet = ON in notebook settings.)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics", "--no-deps"], check=True)
# ultralytics' own deps (numpy/opencv/pyyaml/pillow/tqdm/matplotlib/pandas) are
# already present on Kaggle, so --no-deps avoids any torch churn entirely.

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")


# ====================================================== CELL 2: locate dataset
# Find the mounted dataset by looking for the train list we packed into the zip.
import glob, os, yaml

cands = glob.glob("/kaggle/input/*/train.rel.txt") + glob.glob("/kaggle/input/**/train.rel.txt", recursive=True)
assert cands, "train.rel.txt not found under /kaggle/input — did you add the dataset as an input?"
DATA_ROOT = os.path.dirname(cands[0])
print("Dataset root:", DATA_ROOT)
print("Contents:", os.listdir(DATA_ROOT)[:10])

# Write a data.yaml into the writable working dir, pointing at the read-only input.
# train/val are resolved relative to `path`; the './images/...' lines inside the
# .rel.txt files then resolve relative to that same folder.
data_yaml = {
    "path": DATA_ROOT,
    "train": "train.rel.txt",
    "val": "val.rel.txt",
    "nc": 9,
    "names": ["nylon", "pe", "pet", "pla", "pmma", "pp", "ps", "pu", "pvc"],
}
DATA_YAML_PATH = "/kaggle/working/data.yaml"
with open(DATA_YAML_PATH, "w") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)
print("Wrote", DATA_YAML_PATH)
print(open(DATA_YAML_PATH).read())


# =============================================================== CELL 3: train
from ultralytics import YOLO

model = YOLO("yolov8s.pt")          # auto-downloads (needs Internet ON)
results = model.train(
    data=DATA_YAML_PATH,
    epochs=15,
    imgsz=800,                      # your original detection resolution
    batch=16,                       # fits P100/T4 16GB at imgsz=800; lower if OOM
    device=0,
    project="/kaggle/working/runs/detect",
    name="train",
    save=True,
    save_period=1,                  # checkpoint every epoch (Kaggle can time out)
    patience=20,
)
print("Best weights:", "/kaggle/working/runs/detect/train/weights/best.pt")


# ====================================================== CELL 4: quick sanity val
metrics = model.val(data=DATA_YAML_PATH, imgsz=800, device=0)
print("mAP50-95:", metrics.box.map)
print("mAP50   :", metrics.box.map50)
# Download best.pt from the Output tab on the right, then copy into the repo:
#   models/detect/<new-version>/best.pt
