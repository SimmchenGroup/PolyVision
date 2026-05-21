r"""
dataset_class_balance.py

Create bar charts of class balance for:
1) Whole images dataset (e.g. globalv3/{train,val,test}/<class>/*)
2) Particle crops dataset (e.g. localv3/{train,val,test}/<class>/<group>/*)

Outputs:
- class_balance_whole.png
- class_balance_crops.png
- class_balance_combined.png

Usage examples:
  python utils\dataset_class_balance.py ^
    --whole-root "C:\Users\joshk\OneDrive\Desktop\multiclass\globalv3" ^
    --crops-root "C:\Users\joshk\OneDrive\Desktop\multiclass\localv3" ^
    --split test ^
    --out-dir "results/classification/balance_charts"

  python scripts\dataset_class_balance.py --whole-root "C:\Users\joshk\OneDrive\Desktop\multiclass\globalv3" --crops-root "C:\Users\joshk\OneDrive\Desktop\multiclass\localv3" --split all --out-dir "results\classification\comparison_figs"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _iter_split_class_dirs(root: Path, split: str) -> List[Path]:
    """
    Returns class directories for a given split.

    Expected layouts:
      root/<split>/<class>/...
    """
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def _count_images_in_dir(dir_path: Path) -> int:
    return sum(1 for p in dir_path.rglob("*") if _is_image(p))


def count_by_class(root: Path, split: str) -> Dict[str, int]:
    """
    Count images per class for a given split.
    If split == "all", sums across train/val/test if present.
    """
    root = root.resolve()
    splits = ["train", "val", "test"] if split == "all" else [split]

    counts: Dict[str, int] = {}
    for sp in splits:
        for class_dir in _iter_split_class_dirs(root, sp):
            cls = class_dir.name
            counts[cls] = counts.get(cls, 0) + _count_images_in_dir(class_dir)
    return counts


def _sorted_class_names(*dicts: Dict[str, int]) -> List[str]:
    names: Set[str] = set()
    for d in dicts:
        names.update(d.keys())
    return sorted(names, key=lambda s: s.lower())


def _values_in_order(d: Dict[str, int], names: List[str]) -> List[int]:
    return [int(d.get(n, 0)) for n in names]


def plot_bar(
    title: str,
    class_names: List[str],
    values: List[int],
    out_path: Path,
    color: str = "red",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)

    # 🔒 Cap height to avoid insane image sizes
    height = min(12, max(6, 0.4 * n))

    plt.figure(figsize=(10, height))
    bars = plt.barh(class_names, values, color=color, edgecolor="black", linewidth=0.5)

    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Class")

    for b, v in zip(bars, values):
        plt.text(
            b.get_width(),
            b.get_y() + b.get_height() / 2,
            str(v),
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_grouped_bars_dual_axis_horizontal(
    title: str,
    class_names: list[str],
    whole_values: list[int],
    crop_values: list[int],
    out_path: Path,
) -> None:
    import numpy as np

    n = len(class_names)

    # 🔒 Cap height
    height = min(20, max(8, 0.7 * n))

    spacing = 1.5  # 👈 increase this
    y = np.arange(n) * spacing
    bar_height = 0.6

    fig, ax1 = plt.subplots(figsize=(5, height))

    # 🔴 Whole images (shift UP)
    bars1 = ax1.barh(
        y - bar_height / 2,
        whole_values,
        height=bar_height,
        color="#ea5861ff",
        edgecolor="black",
        label="Whole images",
    )
    ax1.set_xlabel("Whole images")
    ax1.tick_params(axis="x")

    # 🟡 Crops (shift DOWN)
    ax2 = ax1.twiny()
    bars2 = ax2.barh(
        y + bar_height / 2,
        crop_values,
        height=bar_height,
        color="#feca8eff",
        edgecolor="black",
        label="Particle crops",
    )
    ax2.set_xlabel("Particle crops")
    ax2.tick_params(axis="x")

    # Shared Y axis
    ax1.set_yticks(y)
    ax1.set_yticklabels(class_names)
    ax1.set_ylabel("Class")

    plt.title(title)

    # annotate whole
    for b in bars1:
        ax1.text(
            b.get_width(),
            b.get_y() + b.get_height() / 2,
            f"{int(b.get_width())}",
            va="center",
            ha="left",
            fontsize=8,
            color="red",
        )

    # annotate crops
    for b in bars2:
        ax2.text(
            b.get_width(),
            b.get_y() + b.get_height() / 2,
            f"{int(b.get_width())}",
            va="center",
            ha="left",
            fontsize=8,
            color="goldenrod",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# def main() -> None:
#     ap = argparse.ArgumentParser(description="Plot class balance for whole images and particle crops datasets.")
#     ap.add_argument("--whole-root", required=True, help="Root for whole images dataset (e.g. globalv3)")
#     ap.add_argument("--crops-root", required=True, help="Root for particle crops dataset (e.g. localv3)")
#     ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to count")
#     ap.add_argument("--out-dir", default=".", help="Output folder for charts")
#     args = ap.parse_args()
#
#     whole_root = Path(args.whole_root)
#     crops_root = Path(args.crops_root)
#     out_dir = Path(args.out_dir)
#
#     whole_counts = count_by_class(whole_root, split=args.split)
#     crop_counts = count_by_class(crops_root, split=args.split)
#
#     class_names = _sorted_class_names(whole_counts, crop_counts)
#     whole_vals = _values_in_order(whole_counts, class_names)
#     crop_vals = _values_in_order(crop_counts, class_names)
#
#     plot_bar(
#         title=f"Whole images per class ({args.split})",
#         class_names=class_names,
#         values=whole_vals,
#         out_path=out_dir / f"class_balance_whole_{args.split}.png",
#         color="#52528c",
#     )
#     plot_bar(
#         title=f"Particle crops per class ({args.split})",
#         class_names=class_names,
#         values=crop_vals,
#         out_path=out_dir / f"class_balance_crops_{args.split}.png",
#         color="#7c9eb2",
#     )
#     plot_grouped_bars_scaled(
#         title=f"Whole vs Crops per class ({args.split})",
#         class_names=class_names,
#         whole_values=whole_vals,
#         crop_values=crop_vals,
#         out_path=out_dir / f"class_balance_combined_{args.split}.png",
#     )
#
#     # Totals for train/val (always reported)
#     whole_train_total = sum(count_by_class(whole_root, split="train").values())
#     whole_val_total = sum(count_by_class(whole_root, split="val").values())
#     crops_train_total = sum(count_by_class(crops_root, split="train").values())
#     crops_val_total = sum(count_by_class(crops_root, split="val").values())
#
#     print("\nTotals:")
#     print(f"  Whole images  - train: {whole_train_total:,} | val: {whole_val_total:,}")
#     print(f"  Cropped images - train: {crops_train_total:,} | val: {crops_val_total:,}")
#
#     print("\nWrote:")
#     print(" -", (out_dir / f"class_balance_whole_{args.split}.png").resolve())
#     print(" -", (out_dir / f"class_balance_crops_{args.split}.png").resolve())
#     print(" -", (out_dir / f"class_balance_combined_{args.split}.png").resolve())

def plot_vertical_bars_magma(
    labels: list[str],
    values: list[float],
    *,
    title: str = "Bar Chart",
    ylabel: str = "Value",
    out_path: Path | str = "bar_magma.png",
    dpi: int = 200,
    cbar_label: str = "Activation",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values, dtype=float)
    x = np.arange(len(labels))

    norm = mpl.colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = mpl.colormaps["magma"]
    colors = cmap(norm(values))

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(labels)), 6))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{int(v)}" if float(v).is_integer() else f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot class balance for whole images and particle crops datasets.")
    ap.add_argument("--whole-root", required=True, help="Root for whole images dataset (e.g. globalv3)")
    ap.add_argument("--crops-root", required=True, help="Root for particle crops dataset (e.g. localv3)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to count")
    ap.add_argument("--out-dir", default=".", help="Output folder for charts")
    args = ap.parse_args()

    whole_root = Path(args.whole_root)
    crops_root = Path(args.crops_root)
    out_dir = Path(args.out_dir)

    whole_counts = count_by_class(whole_root, split=args.split)
    crop_counts = count_by_class(crops_root, split=args.split)

    class_names = _sorted_class_names(whole_counts, crop_counts)
    whole_vals = _values_in_order(whole_counts, class_names)
    crop_vals = _values_in_order(crop_counts, class_names)

    plot_bar(
        title=f"Whole images per class ({args.split})",
        class_names=class_names,
        values=whole_vals,
        out_path=out_dir / f"class_balance_whole_{args.split}.png",
        color="red",  # 🔴 whole images
    )

    plot_bar(
        title=f"Particle crops per class ({args.split})",
        class_names=class_names,
        values=crop_vals,
        out_path=out_dir / f"class_balance_crops_{args.split}.png",
        color="yellow",  # 🟡 crops
    )

    # NEW: magma-colored versions
    plot_vertical_bars_magma(
        class_names,
        whole_vals,
        title=f"Whole images per class ({args.split})",
        ylabel="Count",
        cbar_label="Count",
        out_path=out_dir / f"class_balance_whole_{args.split}__magma.png",
    )
    plot_vertical_bars_magma(
        class_names,
        crop_vals,
        title=f"Particle crops per class ({args.split})",
        ylabel="Count",
        cbar_label="Count",
        out_path=out_dir / f"class_balance_crops_{args.split}__magma.png",
    )

    plot_grouped_bars_dual_axis_horizontal(
        title=f"Whole vs Crops per class ({args.split})",
        class_names=class_names,
        whole_values=whole_vals,
        crop_values=crop_vals,
        out_path=out_dir / f"class_balance_combined_{args.split}.png",
    )

    # Totals for train/val (always reported)
    whole_train_total = sum(count_by_class(whole_root, split="train").values())
    whole_val_total = sum(count_by_class(whole_root, split="val").values())
    crops_train_total = sum(count_by_class(crops_root, split="train").values())
    crops_val_total = sum(count_by_class(crops_root, split="val").values())

    print("\nTotals:")
    print(f"  Whole images  - train: {whole_train_total:,} | val: {whole_val_total:,}")
    print(f"  Cropped images - train: {crops_train_total:,} | val: {crops_val_total:,}")

    print("\nWrote:")
    print(" -", (out_dir / f"class_balance_whole_{args.split}.png").resolve())
    print(" -", (out_dir / f"class_balance_crops_{args.split}.png").resolve())
    print(" -", (out_dir / f"class_balance_combined_{args.split}.png").resolve())
    print(" -", (out_dir / f"class_balance_whole_{args.split}__magma.png").resolve())
    print(" -", (out_dir / f"class_balance_crops_{args.split}__magma.png").resolve())


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     main()

# python scripts\dataset_class_balance.py --whole-root "C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6" --crops-root "C:\Users\joshk\OneDrive\Desktop\multiclass\localv6" --split all --out-dir "results\classification\comparison_figsv6"
