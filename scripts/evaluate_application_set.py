"""
Evaluate the three PolyVision models on the artificially-aged PET application set
and track how much PET they call, per sample type, over aging weeks.

Structurally this mirrors scripts/evaluate_testset.py (LOCAL on crops, GLOBAL on
whole-images, DETECTION on whole-images), but the application samples are all
assumed PET with no per-image ground truth, so instead of accuracy it reports the
**%PET the models assign**:
    - detection %PET = PET detections / total detections   (per folder)
    - global    %PET = whole-images predicted PET / total  (per folder)
    - local     %PET = crops predicted PET / total crops   (per folder)
aggregated by sample colour + medium (SW/UV) over weeks 4/8/12/16, and plotted as
%PET vs week (one figure per model, SW | UV panels, a line per sample colour).

Input layout per sample folder <colour>_pet_<medium>_<N>weeks (what the app
produces; the blue folders already have it):
    <folder>/whole_images/*            -> GLOBAL + DETECTION
    <folder>/<image_stem>/<crops>.tif  -> LOCAL
Folders that aren't processed yet (raw whole-images at the folder root, no crops)
still get detection/global; LOCAL is skipped for them.

Usage (from repo root):
    python -m scripts.evaluate_application_set \
        --root "C:/Users/joshk/OneDrive/Desktop/raw/application_set" \
        --local-model results/local/efficient/v1.0/best_model.pt \
        --global-model results/global/efficient/v1.0/best_model.pt \
        --yolo-model models/detect/YOLOv8.8/best.pt \
        [--normalize]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.classification.data import _tif_safe_loader
from scripts.evaluate_testset import load_classifier, CLASSES, DEVICE, IMG_EXTS
from scripts.intensity_norm import match_to_reference, load_reference, get_ref

WEEKS = [4, 8, 12, 16]
FOLDER_RE = re.compile(r"^(?P<colour>[a-z]+)_pet_(?P<medium>sw|uv)_(?P<week>\d+)weeks$",
                       re.IGNORECASE)
SAMPLE_COLOURS = {
    "blue": "#1f4e9c", "brown": "#7a4a1e", "orange": "#e8852b",
    "white": "#9e9e9e", "pristine": "#2ca25f",
}
MEDIUM_LABEL = {"sw": "Seawater (SW)", "uv": "UV"}
# very light tints of the local/global/detection theme colours for panel backgrounds
MODEL_BG = {"local": "#f8f5fb", "global": "#fdf6f3", "detection": "#fdf9f0"}
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
})


def gather_inputs(folder: Path):
    """Return (whole_images, crops) for one sample folder.
    whole = whole_images/* if present else the folder's depth-1 images;
    crops = images inside per-image subfolders (skipped for raw folders)."""
    imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    whole_sub = [p for p in imgs if "whole_images" in {s.lower() for s in p.parts}]
    depth1 = [p for p in imgs if p.parent == folder]
    crops = [p for p in imgs if p not in set(whole_sub) and p.parent != folder]
    whole = whole_sub if whole_sub else depth1
    return whole, crops


def classify_count(model, tf, paths, pet_id, norm_ref, batch_size=32):
    """Return (pet, total) over the predicted classes of the given image paths."""
    pet = tot = 0
    buf = []

    def flush():
        nonlocal pet, tot
        if not buf:
            return
        x = torch.stack(buf).to(DEVICE)
        with torch.no_grad():
            preds = model(x).argmax(1).cpu().tolist()
        for i in preds:
            tot += 1
            pet += int(i == pet_id)
        buf.clear()

    for p in paths:
        try:
            img = _tif_safe_loader(str(p))
            if norm_ref is not None:
                img = match_to_reference(img, norm_ref[0], norm_ref[1])
            buf.append(tf(img))
        except Exception as e:
            print(f"    [skip] {p.name}: {e}")
        if len(buf) >= batch_size:
            flush()
    flush()
    return pet, tot


def detect_count(det, paths, conf, pet_id, norm_ref, load_as_gray):
    """Return (pet_detections, total_detections) over the given whole-images."""
    pet = tot = 0
    for p in paths:
        try:
            gray = load_as_gray(str(p))
            if norm_ref is not None:
                gray = match_to_reference(gray, norm_ref[0], norm_ref[1])
            dets, _ = det.detect(gray, conf=conf)
        except Exception as e:
            print(f"    [skip] {p.name}: {e}")
            continue
        for d in dets:
            tot += 1
            pet += int(d.cls_id == pet_id)
    return pet, tot


def detect_records(det, paths, base_conf, norm_ref, load_as_gray):
    """One inference pass at a low base_conf. Returns a list of (cls_id, conf) for
    every detection, so %PET can be re-thresholded in post without re-running YOLO."""
    recs = []
    for p in paths:
        try:
            gray = load_as_gray(str(p))
            if norm_ref is not None:
                gray = match_to_reference(gray, norm_ref[0], norm_ref[1])
            dets, _ = det.detect(gray, conf=base_conf)
        except Exception as e:
            print(f"    [skip] {p.name}: {e}")
            continue
        recs.extend((int(d.cls_id), float(d.conf)) for d in dets)
    return recs


def counts_at_conf(recs, thr, pet_id):
    """(pet, total) among records whose confidence >= thr."""
    pet = tot = 0
    for cls, cf in recs:
        if cf >= thr:
            tot += 1
            pet += int(cls == pet_id)
    return pet, tot


def plot_metric(per_group: dict, out_png: Path, ylabel: str,
                show_title: bool = True, square: bool = False):
    """per_group[(colour, medium)][week] = (pet, total). Plot %PET vs week.

    show_title: draw the per-panel medium title (SW/UV). Set False to omit it.
    square:     force each plot area to a square box (ax.set_box_aspect(1)).
    """
    media = [m for m in ("sw", "uv") if any(k[1] == m for k in per_group)]
    if not media:
        return
    fig, axes = plt.subplots(1, len(media), figsize=(6.2 * len(media), 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, med in zip(axes, media):
        for (colour, m), per_week in sorted(per_group.items()):
            if m != med:
                continue
            xs = [w for w in WEEKS if w in per_week and per_week[w][1]]
            ys = [100.0 * per_week[w][0] / per_week[w][1] for w in xs]
            if xs:
                ax.plot(xs, ys, marker="o", lw=2.4, ms=7,
                        color=SAMPLE_COLOURS.get(colour), markeredgecolor="white",
                        markeredgewidth=0.6, label=colour.capitalize())
        if show_title:
            ax.set_title(MEDIUM_LABEL[med])
        ax.set_xlabel("Aging time (weeks)")
        ax.set_xticks(WEEKS); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
        ax.legend(title="Sample", framealpha=0.9)
        if square:
            ax.set_box_aspect(1)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out_png}")


def plot_uv_combined(results: dict, order: list, titles: dict, ylabel: str,
                     out_png: Path, medium: str = "uv"):
    """Stack the three models' %PET curves in one vertical (3x1) figure:
    one shared x-axis label, one shared y-axis label, one shared legend
    (at the bottom), a per-model title."""
    panels = [m for m in order if any(k[1] == medium for k in results.get(m, {}))]
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, 3.1 * len(panels)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    handles_by_colour = {}
    for ax, model in zip(axes, panels):
        ax.set_facecolor(MODEL_BG.get(model, "white"))
        for (colour, m), per_week in sorted(results[model].items()):
            if m != medium:
                continue
            xs = [w for w in WEEKS if w in per_week and per_week[w][1]]
            ys = [100.0 * per_week[w][0] / per_week[w][1] for w in xs]
            if xs:
                (line,) = ax.plot(xs, ys, marker="o", lw=2.4, ms=7,
                                  color=SAMPLE_COLOURS.get(colour), markeredgecolor="white",
                                  markeredgewidth=0.6, label=colour.capitalize())
                handles_by_colour.setdefault(colour.capitalize(), line)
        ax.set_title(titles[model])
        ax.set_xticks(WEEKS); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Aging time (weeks)")
    fig.supylabel(ylabel)
    handles = list(handles_by_colour.values())
    fig.legend(handles, [h.get_label() for h in handles], title="Sample",
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               framealpha=0.9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out_png}")


def plot_conf_sweep_by_sample(sweep: dict, confs: list, out_dir: Path):
    """For each sample colour, plot %PET vs week with one line per YOLO confidence
    (SW | UV panels), lines coloured light->dark by increasing confidence."""
    cmap = plt.colormaps["magma"]
    lo, hi = 0.15, 0.82
    conf_colour = {c: cmap(lo + (hi - lo) * (i / max(1, len(confs) - 1)))
                   for i, c in enumerate(confs)}
    samples = sorted({colour for c in confs for (colour, _m) in sweep[c]})
    for sample in samples:
        media = [m for m in ("sw", "uv") if any((sample, m) in sweep[c] for c in confs)]
        if not media:
            continue
        fig, axes = plt.subplots(1, len(media), figsize=(6.2 * len(media), 5.2),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, med in zip(axes, media):
            for c in confs:
                per_week = sweep[c].get((sample, med), {})
                xs = [w for w in WEEKS if w in per_week and per_week[w][1]]
                ys = [100.0 * per_week[w][0] / per_week[w][1] for w in xs]
                if xs:
                    ax.plot(xs, ys, marker="o", lw=2.2, ms=6, color=conf_colour[c],
                            markeredgecolor="white", markeredgewidth=0.5, label=f"{c:g}")
            ax.set_title(MEDIUM_LABEL[med]); ax.set_xlabel("Aging time (weeks)")
            ax.set_xticks(WEEKS); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
            ax.legend(title="YOLO conf", framealpha=0.9)
        axes[0].set_ylabel("PET detections (% of total)")
        fig.suptitle(f"{sample.capitalize()} — PET vs aging by confidence")
        fig.tight_layout()
        out_png = out_dir / f"pet_vs_aging_confsweep_{sample}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"[fig] {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="application_set root")
    ap.add_argument("--local-model", default=None)
    ap.add_argument("--global-model", default=None)
    ap.add_argument("--yolo-model", default=None)
    ap.add_argument("--model", default="efficient", help="classifier backbone")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--pet-id", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true",
                    help="Match each image's intensity toward the training distribution.")
    ap.add_argument("--reference", default=str(REPO_ROOT / "configs" / "intensity_reference.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--samples", nargs="+", default=None,
                    help="Only these sample colours (e.g. --samples blue brown). Default: all.")
    ap.add_argument("--medium", choices=["sw", "uv"], default=None,
                    help="Only this medium (sw or uv). Default: both. Use --medium uv to "
                         "ignore seawater folders (incl. any not yet fully processed).")
    ap.add_argument("--conf-sweep", nargs="+", type=float, default=None,
                    help="Detection-only: test several YOLO confidences in one pass, "
                         "e.g. --conf-sweep 0.1 0.25 0.4 0.6. One figure per conf.")
    args = ap.parse_args()
    allowed = {s.lower() for s in args.samples} if args.samples else None
    allowed_medium = args.medium.lower() if args.medium else None

    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root / "evaluation"
    out = out / ("normalized" if args.normalize else "raw")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Device] {DEVICE}  [Normalize] {'ON' if args.normalize else 'off'}")

    ref = load_reference(args.reference) if args.normalize else {}
    ref_local = get_ref(ref, "local") if args.normalize else None
    ref_global = get_ref(ref, "global") if args.normalize else None
    ref_detect = get_ref(ref, "detection") if args.normalize else None

    # Load models once
    local_m = load_classifier(args.local_model, args.model) if args.local_model else None
    global_m = load_classifier(args.global_model, args.model) if args.global_model else None
    det = None
    load_as_gray = None
    if args.yolo_model:
        from polyvision.ml.yolo_detector import YoloDetector
        from polyvision.core.image_io import load_as_gray as _lag
        load_as_gray = _lag
        det = YoloDetector(args.yolo_model, device=("0" if DEVICE.type == "cuda" else "cpu"))

    # ── Detection confidence sweep: one inference pass, many thresholds ──────────
    if args.conf_sweep:
        if det is None:
            print("--conf-sweep needs --yolo-model."); return
        confs = sorted(args.conf_sweep)
        base = min(confs)
        sweep = {c: defaultdict(dict) for c in confs}   # sweep[conf][(colour,medium)][week]=(pet,tot)
        rows = []
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            mm = FOLDER_RE.match(folder.name)
            if not mm:
                continue
            colour, medium, week = mm["colour"].lower(), mm["medium"].lower(), int(mm["week"])
            if allowed and colour not in allowed:
                continue
            if allowed_medium and medium != allowed_medium:
                continue
            whole, _ = gather_inputs(folder)
            if not whole:
                continue
            recs = detect_records(det, whole, base, ref_detect, load_as_gray)
            line = [f"{folder.name}:"]
            for c in confs:
                pet, tot = counts_at_conf(recs, c, args.pet_id)
                sweep[c][(colour, medium)][week] = (pet, tot)
                rows.append((c, colour, medium, week, pet, tot))
                line.append(f"c{c}={100.0*pet/tot:.0f}%" if tot else f"c{c}=—")
            print("  ".join(line))

        csv_path = out / "pet_summary_confsweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["conf", "sample", "medium", "week", "pet", "total", "pct_pet"])
            for c, colour, medium, week, pet, tot in sorted(rows):
                w.writerow([c, colour, medium, week, pet, tot,
                            round(100.0 * pet / tot, 3) if tot else 0.0])
        print(f"\n[csv] {csv_path}")
        for c in confs:
            plot_metric(sweep[c], out / f"pet_vs_aging_detection_conf{c}.png",
                        f"PET detections (% of total) — conf {c}")
        # combined: one figure per sample, lines coloured by confidence
        plot_conf_sweep_by_sample(sweep, confs, out)
        return

    # results[model][(colour, medium)][week] = (pet, total)
    results = {"local": defaultdict(dict), "global": defaultdict(dict),
               "detection": defaultdict(dict)}
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
        print(f"\n[{folder.name}] whole={len(whole)} crops={len(crops)}")

        if local_m and crops:
            pet, tot = classify_count(*local_m, crops, args.pet_id, ref_local, args.batch_size)
            results["local"][(colour, medium)][week] = (pet, tot)
            rows.append(("local", colour, medium, week, pet, tot))
            print(f"   local: {pet}/{tot} PET")
        if global_m and whole:
            pet, tot = classify_count(*global_m, whole, args.pet_id, ref_global, args.batch_size)
            results["global"][(colour, medium)][week] = (pet, tot)
            rows.append(("global", colour, medium, week, pet, tot))
            print(f"   global: {pet}/{tot} PET")
        if det and whole:
            pet, tot = detect_count(det, whole, args.conf, args.pet_id, ref_detect, load_as_gray)
            results["detection"][(colour, medium)][week] = (pet, tot)
            rows.append(("detection", colour, medium, week, pet, tot))
            print(f"   detection: {pet}/{tot} PET")

    if not rows:
        print("\nNo results — check models and folder layout.")
        return

    # CSV
    csv_path = out / "pet_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "sample", "medium", "week", "pet", "total", "pct_pet"])
        for model, colour, medium, week, pet, tot in sorted(rows):
            w.writerow([model, colour, medium, week, pet, tot,
                        round(100.0 * pet / tot, 3) if tot else 0.0])
    print(f"\n[csv] {csv_path}")

    # One figure per model
    labels = {"local": "PET crops (% of total)",
              "global": "PET images (% of total)",
              "detection": "PET detections (% of total)"}
    for model, per_group in results.items():
        if per_group:
            plot_metric(per_group, out / f"pet_vs_aging_{model}.png", labels[model])

    # Combined vertical (3x1) UV figure: one x-label, one y-label, one legend, per-model titles
    uv_titles = {"local": "Local Classifier",
                 "global": "Global Classifier",
                 "detection": "Detection"}
    plot_uv_combined(results, ["local", "global", "detection"], uv_titles,
                     "PET (% of total)", out / "pet_vs_aging_uv_combined.png")


if __name__ == "__main__":
    main()

# python scripts/compute_intensity_reference.py \
#     --local-train  "<localv9>/train" \
#     --global-train "<globalv9>/train" \
#     --detect-train "<detectv9>/images" \
#     --sample-n 200

# # baseline
#   python -m scripts.evaluate_application_set --root "C:\Users\joshk\OneDrive\Desktop\raw\application_set" --local-model "C:\Users\joshk\OneDrive\Desktop\multiclass\ft_results\results\local\efficient\finetune_resume\finetune_resume\best_model.pt" --global-model "C:\Users\joshk\OneDrive\Desktop\multiclass\ft_results\results\global\efficient\finetune_resume\finetune_resume\best_model.pt" --yolo-model "C:\Users\joshk\OneDrive\Desktop\multiclass\detectv9\detect_colab_runs\detect\detect_colab_ft\weights\best.pt" --samples blue brown
#   # normalized
#   python -m scripts.evaluate_application_set --root ".../application_set" \
#     --local-model ... --global-model ... --yolo-model ... --normalize