# PolyVision

**Identification of microplastic polymers from optical-microscopy images by a
three-model deep-learning ensemble.**

PolyVision classifies microplastic particles into nine polymer types — nylon, PE,
PET, PLA, PMMA, PP, PS, PU, and PVC — directly from bright-field optical micrographs.
Rather than relying on a single network, it combines three complementary models by
weighted late fusion:

| Model | Backbone | Input | Role |
|-------|----------|-------|------|
| **Local**     | EfficientNet-B0 | single-particle crops | fine morphology, one vote per crop |
| **Global**    | EfficientNet-B0 | whole micrograph      | field-level context |
| **Detection** | YOLOv8s         | whole micrograph      | localises particles *and* votes per box |

The detector also produces the crops the Local model consumes, so one whole image
flows through all three branches. Each branch outputs a probability vector over the
nine classes; these are combined as a weighted average and arg-maxed for the final
prediction. Fusion weights are **selected on a validation set** and frozen before
being applied to the independent test set.

> This is an *image-only* method. Two polymers that are chemically distinct but
> morphologically similar (e.g. PP vs PET) can be inherently ambiguous under
> microscopy; see the paper's discussion of this modality limit.

---

## Repository structure

```
PolyVision/
├── polyvision/                 # Annotation application + inference library
│   ├── app/                    #   PyQt GUI: annotate, review, and edit the dataset
│   │   ├── main.py             #     entry point (python -m polyvision.app.main)
│   │   └── gui/                #     windows, graphics items, background workers
│   ├── core/                   #   image I/O, thresholding, particle masking,
│   │                           #   bounding-box geometry, SQLite catalogue
│   └── ml/                     #   thin wrappers around the trained models + fusion
│       ├── local_classifier.py
│       ├── global_classifier.py
│       ├── yolo_detector.py
│       └── fusion.py           #   weighted late fusion + optional stacking
├── training/
│   ├── classification/         # EfficientNet-B0 training (Local & Global share this)
│   │   ├── model.py            #   backbone + custom head
│   │   ├── data.py             #   datasets, transforms, class balancing
│   │   ├── pipeline.py         #   two-phase (head-only → fine-tune) training
│   │   ├── colab_train.py      #   Colab entry point (used for the paper runs)
│   │   └── kaggle_train.py     #   Kaggle entry point
│   └── detection/              # YOLOv8 detector training (Ultralytics)
├── scripts/                    # Dataset building, evaluation, and fusion
│   ├── build_detect_dataset.py         # assemble a YOLO dataset from data/complete
│   ├── add_classification_dataset.py   # assemble ImageFolder crop/whole datasets
│   ├── evaluate_testset.py             # score each model on the labelled test set
│   ├── build_fusion_features.py        # cache per-image [local|global|detection] vectors
│   ├── fuse_testset.py                 # weighted late fusion + weight sweep
│   └── train_fusion_meta.py            # stacking meta-classifier (validation-trained)
├── configs/config.json         # class list, model paths, fusion weights, processing
├── DATA_MANAGEMENT.md          # dataset layout, annotation workflow, DVC backup
└── README.md
```

Model weights and the image dataset are **not** stored in git (see *Data & weights*
below).

---

## Installation

Trained on Google Colab with **Python 3.12** and **Ultralytics YOLOv8 8.4.82**
(PyTorch installed unpinned; any recent PyTorch 2.x works).

```bash
git clone https://github.com/SimmchenGroup/PolyVision.git
cd PolyVision
conda env create -f environment.yml && conda activate PhD   # Python 3.13 + requirements.txt
```

Or without conda: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.
`requirements.txt` pins CUDA 12.8 builds of PyTorch; on a CPU-only machine install
`torch`/`torchvision` from <https://pytorch.org/get-started/locally/> first.

The nine classes and their fixed **alphabetical** index order (nylon=0 … pvc=8) are
defined in `configs/config.json` and used consistently everywhere.

---

## Running the annotation app

Only the GUI, `configs/`, and your own model weights + images are needed — no
training code or DVC setup.

1. **Models** — place the weights where `configs/config.json` → `models` points
   (or edit those paths):
   ```
   models/detect/best.pt                          # YOLO detector
   models/local/EfficientNetB0/best_model.keras   # optional, fusion only
   models/global/EfficientNetB0/best_model.keras  # optional, fusion only
   ```
   If `best.pt` is missing the app still starts; YOLO proposals are disabled and
   Otsu/adaptive thresholding + manual boxes are used instead. Fusion is disabled
   automatically when the classifier weights are absent.
2. **Images** — drop `.tif` micrographs into `paths.input_dir` (default `data/raw`).
   Annotations are written under `paths.output_root`.
3. **Launch** from the repo root:
   ```bash
   python -m polyvision.app.main
   ```

---

## Data

Datasets are versioned with **DVC** (backed up to the group's iDrive), not committed
to git. The annotation app, the on-disk layout, and the DVC workflow are documented
in **[DATA_MANAGEMENT.md](DATA_MANAGEMENT.md)**. In brief, each class folder holds:

```
data/complete/<class>/
├── whole_images/<stem>.tif        # original micrograph  → Global + Detection
└── <stem>/                        # one folder per micrograph
    ├── <stem>_0001.tif  …         # particle crops       → Local
    └── <stem>.txt                 # YOLO labels (class cx cy w h, normalised)
```

Particles are localised in the GUI by Otsu thresholding, by the trained detector, or
by hand (drag a box); each box is squared about its centre and clamped to the image
before the crop is written. Crops and whole images are linked by a shared filename
stem so the three models stay aligned per particle.

---

## Reproducing the method

### 1 — Build the training datasets from `data/complete`

These two modules expose dataset-assembly *functions* (call them from a short driver
script or notebook, passing your dataset root and the class→id map):

- `scripts.build_detect_dataset.build_yolo_dataset_with_class_map(...)` — assembles a
  YOLO dataset (whole images converted to JPEG, labels remapped to the global class
  index, randomised train/val split, `data.yaml`).
- `scripts.add_classification_dataset` — assembles the ImageFolder datasets: particle
  crops for the Local model, whole images for the Global model, paired by stem.

### 2 — Train the three models

```bash
# Local (crops) and Global (whole images): same EfficientNet-B0, different input.
# The paper runs used the Colab entry point; run_experiment.py is the local runner.
python -m scripts.run_experiment --mode train --model efficient   # backbone: efficient|inception|res
#   (the Local vs Global distinction is which dataset root you point it at)

# Detection: YOLOv8s via Ultralytics
python training/detection/train.py
```

Classifier training is two-phase (`training/classification/pipeline.py`): a frozen
backbone with only the head trained, then the top layers unfrozen for fine-tuning
(SGD, class-weighted loss, label smoothing).

### 3 — Evaluate the single models

```bash
python -m scripts.evaluate_testset \
    --local-model   <local_best.pt> \
    --global-model  <global_best.pt> \
    --yolo-model    <detect_best.pt> \
    --testset-root  <labelled test set> \
    --out-dir       <eval dir>
```

### 4 — Fuse

```bash
# Cache the per-image [local|global|detection] probability vectors once
python -m scripts.build_fusion_features --split-source <globalv9 splits> \
    --local-model … --global-model … --yolo-model … --out fusion_features.npz

# Weighted late fusion, and a full sweep over the weight simplex
python -m scripts.fuse_testset --eval-dir <eval dir> --sweep

# Optional: stacking meta-classifier (trained on the VALIDATION split only)
python -m scripts.train_fusion_meta --features fusion_features.npz
```

Weights are chosen on validation and frozen for the test set. The operating point
reported in the paper is `(w_local, w_global, w_detection) = (0.44, 0.24, 0.32)`.

Each script prints its full options with `-h`.

---

## Data & weights

- **Image dataset** — versioned with DVC on the Strathclyde iDrive (see
  DATA_MANAGEMENT.md). Available on request.
- **Trained model weights** — distributed via the repository's GitHub Releases
  (they are large and kept out of the git tree).

---

## Citation

If you use PolyVision, please cite the accompanying paper (details to follow).
Simmchen group, University of Strathclyde.
