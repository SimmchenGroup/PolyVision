# PolyVision — Data Management Guide

Dataset versioning is handled by **DVC**, backed up to the Strathclyde iDrive.  
The iDrive is accessible on campus or via the university VPN.

---

## Folder structure

```
PolyVision/
├── data/
│   ├── raw/                         ← drop new unannotated .tif files here
│   └── complete/                    ← curated dataset (DVC-tracked, git-ignored)
│       ├── nylon/
│       │   ├── image001/
│       │   │   ├── image001_0001.tif    crops
│       │   │   ├── image001_0002.tif
│       │   │   └── image001.txt         YOLO labels
│       │   └── whole_images/
│       │       └── image001.tif         original image
│       ├── PE/
│       └── ...
├── configs/
│   └── config.json                  ← model settings, class list, processing defaults
├── models/                          ← model weights (not in git, not in DVC)
└── scripts/
    └── dataset_sync.bat             ← one command to version + back up
```

**iDrive path (DVC remote — internal blob store, not for browsing):**  
`J:\Science\Chemistry\RDMS\Juliane Simmchen group\Josh\2. Microplastic AI Project\working_dataset\dvc_cache`

---

## One-time setup (first time only)

### 1. Migrate existing dataset off OneDrive

Run from the repo root in a terminal:

```bat
xcopy /E /I "path\to\your\existing\complete" "data\complete"
```

### 2. Take the first DVC snapshot and push to iDrive

Make sure the J: drive is connected (on campus or VPN), then:

```bat
scripts\dataset_sync.bat
```

### 3. Verify

```bat
dvc status        ← should say "Data and pipelines are up to date"
```

After this you can archive or delete the old `Desktop\raw\complete` folder — iDrive is now the source of truth.

---

## Daily annotation workflow

### Step 1 — Add raw images

Copy the new unannotated `.tif` files into:
```
data\raw\
```

### Step 2 — Launch the app

```bat
python -m polyvision.app.main
```

A **Session Setup dialog** appears. Select:
- **Class to annotate** — e.g. Nylon, PE, PP
- **Raw images folder** — defaults to `data\raw\`, browse if needed

Click **Start Session**.

### Step 3 — Annotate

- Press **Enter** to accept the current image and move to the next.
- Press **Escape** to skip.
- Use `,` / `.` to decrease / increase the minimum particle size.
- Use `-` / `+` to adjust the Otsu threshold.

Accepted images are saved automatically:
```
data\complete\{class}\{image_stem}\       ← crops + YOLO label file
data\complete\{class}\whole_images\       ← original image moved here
```

### Step 4 — Back up after each session

Connect to the university VPN (if off-campus), then run:

```bat
scripts\dataset_sync.bat
```

This:
1. Stages any changes to `data\complete` under DVC
2. Commits the updated DVC pointer to git
3. Pushes data blobs to iDrive

---

## Reviewing and editing the dataset

Click **Review Dataset** in the annotation window (or launch separately).

- Select a class in the left panel → images listed below
- Select an image → whole image shown with bounding box overlays
- **Draw boxes** checkbox — click-drag to add a new box
- **Delete on click** checkbox — click a box to remove it
- **Delete key** — removes the selected box
- **Save (Ctrl+S)** — rewrites the `.txt` label file and re-extracts crops
- **Delete Entire Image** — removes the crop folder and whole image permanently

After editing, run `scripts\dataset_sync.bat` to push the updated version.

---

## Working on another machine

```bat
git clone <repo-url>
cd PolyVision
dvc pull          ← downloads data\complete from iDrive (needs VPN if off-campus)
```

---

## Common DVC commands

| Command | What it does |
|---|---|
| `dvc status` | Show whether local data matches the last committed version |
| `dvc pull` | Download `data\complete` from iDrive |
| `dvc push` | Upload local changes to iDrive |
| `dvc diff` | Show what changed since the last commit |
| `git log data/complete.dvc` | History of dataset versions |
| `git checkout <hash> -- data/complete.dvc && dvc checkout` | Restore a previous dataset version |

---

## Changing the class list

Edit `configs/config.json` → `"classes"` → `"items"`.  
The startup dialog and annotation window both read from this list automatically.

---

## Troubleshooting

**`dvc push` / `dvc pull` fails**  
→ Check the J: drive is mounted. Connect to Strathclyde VPN if off-campus.  
→ Verify: `dir "J:\Science\Chemistry\RDMS\Juliane Simmchen group\Josh\2. Microplastic AI Project\working_dataset"`

**"No images found" on startup**  
→ Make sure `.tif` files are in `data\raw\` before launching, or use **Review Dataset** to inspect existing data without annotating.

**Dataset looks wrong after a `dvc pull`**  
→ Run `dvc status` to check for conflicts. To restore the last committed version:  
```bat
dvc checkout
```
