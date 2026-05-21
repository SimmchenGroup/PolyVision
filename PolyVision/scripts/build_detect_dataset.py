from __future__ import annotations

from pathlib import Path
import random
import shutil
import yaml
import cv2


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_to_jpg(src: Path, dst_jpg: Path, quality: int = 95):
    """
    Read `src` (any supported format), normalize to 8-bit 3-channel BGR, and write as JPEG to `dst_jpg`.
    Does NOT modify the source file.
    """
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != "uint8":
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    ensure_dir(dst_jpg.parent)
    ok = cv2.imwrite(
        str(dst_jpg),
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    return bool(ok)


def build_yolo_dataset_with_class_map(
    parent_root: str | Path,
    out_root: str | Path,
    class_to_id: dict[str, int],
    val_split: float = 0.2,
    seed: int = 42,
    image_glob: str = "*.*",
    strict: bool = True,
    jpg_quality: int = 95,
):
    """
    Expected input structure:
      parent_root/
        nylon/
          whole_images/
            IMG_001.tif
          IMG_001/
            IMG_001.txt
        pmma/
          whole_images/...
          ...

    Output (YOLOv8-style):
      out_root/
        images/
          IMG_001.jpg
        labels/
          IMG_001.txt   (class id remapped using class_to_id)
        train.txt
        val.txt
        data.yaml
    """
    random.seed(seed)

    parent_root = Path(parent_root)
    out_root = Path(out_root)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Validate mapping
    if not class_to_id:
        raise ValueError("class_to_id is empty")

    ids = list(class_to_id.values())
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate class IDs in class_to_id: {class_to_id}")

    # names must be ordered by ID for YOLO
    id_to_class = {i: c for c, i in class_to_id.items()}
    max_id = max(id_to_class)
    missing = [i for i in range(max_id + 1) if i not in id_to_class]
    if missing:
        raise ValueError(f"Missing class IDs in mapping: {missing}. IDs should be contiguous 0..N-1")

    names = [id_to_class[i] for i in range(max_id + 1)]

    all_images_out_names: list[str] = []
    skipped: list[str] = []

    for class_name, class_id in class_to_id.items():
        cls_path = parent_root / class_name
        whole_images = cls_path / "whole_images"
        if not whole_images.exists():
            msg = f"Missing folder: {whole_images}"
            if strict:
                raise FileNotFoundError(msg)
            skipped.append(msg)
            continue

        for img_path in whole_images.glob(image_glob):
            stem = img_path.stem
            label_in_path = cls_path / stem / f"{stem}.txt"

            if not label_in_path.exists():
                msg = f"Missing label: {label_in_path} (for image {img_path.name})"
                if strict:
                    raise FileNotFoundError(msg)
                skipped.append(msg)
                continue

            # Convert/copy image as JPG
            out_img_path = images_dir / f"{stem}.jpg"
            ok = convert_to_jpg(img_path, out_img_path, quality=jpg_quality)
            if not ok:
                msg = f"Could not read/convert image: {img_path}"
                if strict:
                    raise RuntimeError(msg)
                skipped.append(msg)
                continue

            # Read + remap label file (replace first token in each non-empty line)
            in_lines = label_in_path.read_text(encoding="utf-8").splitlines()
            out_lines: list[str] = []
            for line in in_lines:
                parts = line.strip().split()
                if not parts:
                    continue
                parts[0] = str(class_id)
                out_lines.append(" ".join(parts))

            # Write label next to output image name (same stem)
            out_label_path = labels_dir / f"{stem}.txt"
            out_label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

            all_images_out_names.append(out_img_path.name)

    # Shuffle + split
    random.shuffle(all_images_out_names)
    val_count = int(len(all_images_out_names) * val_split)
    val_names = all_images_out_names[:val_count]
    train_names = all_images_out_names[val_count:]

    (out_root / "train.txt").write_text(
        "\n".join(str((images_dir / n).resolve()) for n in train_names),
        encoding="utf-8",
    )
    (out_root / "val.txt").write_text(
        "\n".join(str((images_dir / n).resolve()) for n in val_names),
        encoding="utf-8",
    )

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(names),
        "names": names,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print(f"Built YOLO dataset at: {out_root}")
    print(f"Classes (id->name): {dict(enumerate(names))}")
    print(f"Images: train={len(train_names)} val={len(val_names)} total={len(all_images_out_names)}")
    if skipped:
        print(f"Skipped items ({len(skipped)}):")
        for s in skipped[:20]:
            print("  -", s)
        if len(skipped) > 20:
            print("  ...")


# Example usage
build_yolo_dataset_with_class_map(
    parent_root=r"C:\Users\joshk\OneDrive\Desktop\raw\complete",
    out_root=r"C:\Users\joshk\OneDrive\Desktop\multiclass\detectv4",
    class_to_id={
        "nylon": 0,
        "pe": 1,
        "pet": 2,
        "pla": 3,
        "pmma": 4,
        "pp": 5,
        "ps": 6,
        "pu": 7,
        "pvc": 8,
    },
    val_split=0.2,
    seed=42,
    strict=False,  # set True if you want it to error on any missing file
    jpg_quality=95,
)