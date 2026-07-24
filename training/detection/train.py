"""
Train the YOLOv8 particle detector (Ultralytics).

Entry point for detection training: sets up the GPU, runs a dataset precheck
(band/dtype sanity on the images), launches Ultralytics training on the assembled
detection dataset, and classifies common crash causes for easier debugging. Produces
the detector weights used by both the crop-extraction and detection-vote stages.
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image

import cv2
import numpy as np


# -----------------------
# CONFIG (defaults — override via CLI)
# -----------------------
DATA_YAML = Path(r"path\to\runs\detectv7\data.yaml")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_WORKERS = 8  # tune: 4-8 typical; if on slow disk/network, lower may be faster


# =============================================================================
# Blackwell sm_120a stability guards
# =============================================================================
# The classification models crashed on this GPU for two reasons (see
# project-wsl-gpu-setup memory):
#   1. inplace relu_/silu_ PTX kernels miscompile -> cudaErrorLaunchFailure
#   2. cuDNN training-mode conv algorithms fail   -> CUDNN_STATUS_EXECUTION_FAILED
#
# YOLO trains ALL weights (requires_grad=True), so unlike the frozen
# classification backbones it WILL request cuDNN training-mode algorithms.
# That is precisely cause #2 — so cuDNN must stay DISABLED here. The inplace
# pass guards against cause #1 (Ultralytics' SiLU defaults to non-inplace, but
# we flip anything that exposes the attribute to be safe).
def setup_gpu(disable_cudnn: bool = True):
    if not torch.cuda.is_available():
        print("[GPU] No CUDA device found — running on CPU")
        return
    print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = False
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[GPU] cuDNN DISABLED (training-mode conv algorithms crash on sm_120a).")
    else:
        print("[GPU] cuDNN left ENABLED (--keep-cudnn) — expect possible "
              "CUDNN_STATUS_EXECUTION_FAILED on Blackwell.")


def disable_inplace(torch_model: nn.Module) -> int:
    """Flip inplace=False on every module that exposes it (SiLU/ReLU/etc).

    Guards against the sm_120a inplace-activation PTX bug. Returns the count of
    modules changed. Must run on the underlying torch model (YOLO().model)
    before .train() builds the autograd graph.
    """
    n = 0
    for m in torch_model.modules():
        if getattr(m, "inplace", False):
            m.inplace = False
            n += 1
    print(f"[Model] Disabled inplace on {n} modules.")
    return n


def classify_crash(exc: Exception) -> str:
    """Map a CUDA crash onto one of the two known sm_120a root causes so a
    failed run tells us immediately which guard (if any) was insufficient."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "cudnn_status_execution_failed" in msg or "cudnn" in msg:
        return ("DIAGNOSIS: cuDNN training-mode crash (cause #2). cuDNN is "
                "already disabled here — if this fired, the cuBLAS/PTX fallback "
                "is also failing on sm_120a (matches the classification note). "
                "Next step: try imgsz lower or fall back to CPU for the final fit.")
    if ("launchfailure" in msg or "illegal memory access" in msg
            or "device-side assert" in msg):
        return ("DIAGNOSIS: kernel launch failure (cause #1 — inplace activation "
                "PTX bug). The inplace-disable pass should have prevented this; "
                "check that disable_inplace() ran on model.model and that "
                "Ultralytics didn't rebuild/fuse the model before training.")
    if "cudaerrorunknown" in msg or "unknown" in msg:
        return ("DIAGNOSIS: cudaErrorUnknown — usually a masked error. Re-run with "
                "CUDA_LAUNCH_BLOCKING=1 to surface the true location, as was done "
                "to diagnose the classification crashes.")
    if "out of memory" in msg or "outofmemory" in msg:
        return ("DIAGNOSIS: CUDA OOM — lower --batch or --imgsz.")
    return "DIAGNOSIS: not one of the known sm_120a signatures — see traceback above."


# -----------------------
# FAST PRECHECK (metadata-only) — normalise images to 3-channel uint8
# -----------------------
def band_count_fast(p: Path) -> int | None:
    try:
        with Image.open(p) as im:
            return len(im.getbands())
    except Exception:
        return None


def dtype_fast(p: Path) -> np.dtype | None:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.dtype


def normalize_to_3ch_inplace(p: Path) -> tuple[Path, str]:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return p, "unreadable"

    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        out = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = img
    elif img.ndim == 3 and img.shape[2] == 4:
        out = img[:, :, :3]
    else:
        return p, f"unsupported(shape={getattr(img, 'shape', None)})"

    if out.dtype != np.uint8:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ok = cv2.imwrite(str(p), out)
    return p, "fixed" if ok else "write_failed"


def run_precheck(images_dir: Path):
    files = [p for p in images_dir.rglob("*")
             if p.is_file() and p.suffix.lower() in IMG_EXTS]

    needs_fix: list[Path] = []
    unreadable: list[Path] = []
    needs_fix_bands = 0
    needs_fix_dtype = 0

    for p in files:
        bands = band_count_fast(p)
        if bands is None:
            unreadable.append(p)
            continue
        dt = dtype_fast(p)
        if dt is None:
            unreadable.append(p)
            continue
        band_bad = (bands != 3)
        dtype_bad = (dt != np.uint8)
        if band_bad or dtype_bad:
            needs_fix.append(p)
            needs_fix_bands += int(band_bad)
            needs_fix_dtype += int(dtype_bad)

    print(f"Images scanned: {len(files)}")
    print(f"Needs fix (bands != 3): {needs_fix_bands}")
    print(f"Needs fix (dtype != uint8): {needs_fix_dtype}")
    print(f"Total needs_fix (union): {len(needs_fix)}")
    print(f"Unreadable (Pillow/cv2): {len(unreadable)}")
    if unreadable:
        print("First unreadable example:", unreadable[0])

    counts = {"ok": 0, "fixed": 0, "unreadable": 0, "write_failed": 0, "unsupported": 0}
    if needs_fix:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(normalize_to_3ch_inplace, p) for p in needs_fix]
            for fut in as_completed(futs):
                p, status = fut.result()
                if status.startswith("unsupported"):
                    counts["unsupported"] += 1
                    print("Unsupported:", p, status)
                else:
                    counts[status] = counts.get(status, 0) + 1
    print("Normalize results:", counts)


# -----------------------
# TRAIN
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Train YOLO detection on GPU with sm_120a stability guards.")
    ap.add_argument("--model", default="yolov8s.pt", help="Base weights (e.g. yolov8s.pt, yolov8n.pt).")
    ap.add_argument("--data", default=str(DATA_YAML), help="Path to data.yaml.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8, help="Batch size. Lower if CUDA OOM at imgsz=800.")
    ap.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' for CPU fallback.")
    ap.add_argument("--amp", action="store_true", default=False,
                    help="Enable Ultralytics AMP (float16). Off by default — the AMP "
                         "self-check runs a GPU inference that may itself crash on sm_120a.")
    ap.add_argument("--keep-cudnn", action="store_true", default=False,
                    help="Do NOT disable cuDNN (debug only — expect crashes on Blackwell).")
    ap.add_argument("--keep-inplace", action="store_true", default=False,
                    help="Do NOT disable inplace activations (debug only).")
    ap.add_argument("--skip-precheck", action="store_true", default=False,
                    help="Skip the 3-channel/uint8 image normalisation scan.")
    ap.add_argument("--project", default="detection/runs")
    ap.add_argument("--name", default="train_multiclass_detect_gpu")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    images_dir = data_yaml.parent / "images"
    on_gpu = args.device != "cpu"

    if on_gpu:
        setup_gpu(disable_cudnn=not args.keep_cudnn)

    if not args.skip_precheck:
        run_precheck(images_dir)
    else:
        print("[Precheck] skipped (--skip-precheck).")

    model = YOLO(args.model)
    if on_gpu and not args.keep_inplace:
        disable_inplace(model.model)

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            amp=args.amp,

            project=args.project,
            name=args.name,
            exist_ok=False,
            save=True,
            save_period=1,
        )
        print("\n[OK] Training completed!")
        print(f"  Results:    {args.project}/{args.name}")
        print(f"  Best model: {args.project}/{args.name}/weights/best.pt")

    except KeyboardInterrupt:
        print("\n[Interrupted] checkpoint saved.")
        print(f"  Resume with: YOLO('{args.project}/{args.name}/weights/last.pt').train(resume=True)")

    except Exception as exc:
        print("\n" + "=" * 70)
        print(f"[CRASH] {type(exc).__name__}: {exc}")
        print("-" * 70)
        traceback.print_exc()
        print("-" * 70)
        print(classify_crash(exc))
        print(f"  Last checkpoint: {args.project}/{args.name}/weights/last.pt")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
