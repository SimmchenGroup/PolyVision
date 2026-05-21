from __future__ import annotations

from pathlib import Path
import random
import shutil

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMAGE_EXTS


def _list_class_dirs(split_dir: Path) -> list[Path]:
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def _gather_images_any_depth(class_dir: Path) -> list[Path]:
    # Works for both:
    #   global: class_dir/*.jpg
    #   local : class_dir/<whole_image_folder>/*.jpg
    return sorted([p for p in class_dir.rglob("*") if _is_image(p)])


def undersample_split_images_to_minority(
    src_split_dir: str | Path,
    dst_split_dir: str | Path,
    *,
    seed: int = 1337,
    overwrite: bool = False,
    preserve_structure: bool = True,
) -> int:
    """
    Undersample each class down to the minority class count, using IMAGE FILES as samples.

    If preserve_structure=True:
      Copies to dst/<class>/<relative_path_from_class_dir>
      (so local folders like ny1_et_... are kept)

    If preserve_structure=False:
      Copies to dst/<class>/<filename> (flat). Collisions are handled by prefixing.
    """
    src_split_dir = Path(src_split_dir)
    dst_split_dir = Path(dst_split_dir)

    if not src_split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {src_split_dir}")

    class_dirs = _list_class_dirs(src_split_dir)
    if not class_dirs:
        raise ValueError(f"No class directories found in: {src_split_dir}")

    rng = random.Random(seed)

    per_class_images: dict[str, list[Path]] = {}
    for cdir in class_dirs:
        imgs = _gather_images_any_depth(cdir)
        if not imgs:
            raise ValueError(f"No images found for class={cdir.name} in {cdir}")
        per_class_images[cdir.name] = imgs

    n_min = min(len(v) for v in per_class_images.values())

    if dst_split_dir.exists():
        if overwrite:
            shutil.rmtree(dst_split_dir)
        else:
            raise FileExistsError(f"Destination exists: {dst_split_dir} (set overwrite=True)")

    dst_split_dir.mkdir(parents=True, exist_ok=True)

    for cdir in class_dirs:
        class_name = cdir.name
        imgs = per_class_images[class_name].copy()
        rng.shuffle(imgs)
        chosen = imgs[:n_min]

        out_class_dir = dst_split_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for p in chosen:
            if preserve_structure:
                rel = p.relative_to(cdir)  # includes local folder names if present
                out_path = out_class_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out_path)
            else:
                out_path = out_class_dir / p.name
                if out_path.exists():
                    # prefix with parent folder to reduce collisions
                    out_path = out_class_dir / f"{p.parent.name}__{p.name}"
                shutil.copy2(p, out_path)

    return n_min


if __name__ == "__main__":
    global_root = Path(r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6")
    local_root  = Path(r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv6")

    # Example: balance TRAIN splits (repeat similarly for val/test if you want)
    n_g = undersample_split_images_to_minority(
        src_split_dir=global_root / "train",
        dst_split_dir=Path(r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6_bal") / "train",
        overwrite=True,
        preserve_structure=False,  # global is usually already flat
    )
    print(f"[global] train balanced to {n_g} images/class")

    n_l = undersample_split_images_to_minority(
        src_split_dir=local_root / "train",
        dst_split_dir=Path(r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv6_bal") / "train",
        overwrite=True,
        preserve_structure=True,   # recommended for your local folder-per-whole-image layout
    )
    print(f"[local] train balanced to {n_l} images/class")