import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_uint8_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.min())
    x = x / (float(x.max()) + 1e-8)
    return (x * 255.0).clip(0, 255).astype(np.uint8)

def save_featuremap_mean(
    feature: torch.Tensor,
    out_path: Path,
    *,
    colormap: int = cv2.COLORMAP_VIRIDIS,  # try also COLORMAP_INFERNO / COLORMAP_PLASMA / COLORMAP_TURBO
) -> None:
    """
    feature: (B, C, H, W) torch tensor
    Saves a single color heatmap using channel-mean.
    """
    fm = feature[0].detach().float().cpu().numpy()  # (C,H,W)
    heat = fm.mean(axis=0)                          # (H,W)

    heat_u8 = _to_uint8_01(heat)                    # (H,W) uint8
    heat_color_bgr = cv2.applyColorMap(heat_u8, colormap)  # (H,W,3) BGR

    cv2.imwrite(str(out_path), heat_color_bgr)


def save_featuremap_channels(feature: torch.Tensor, out_dir: Path, max_channels: int = 32) -> None:
    """
    feature: (B, C, H, W)
    Saves each channel as a separate PNG (grayscale).
    """
    fm = feature[0].detach().float().cpu().numpy()  # (C,H,W)
    c = fm.shape[0]
    n = min(int(max_channels), int(c))

    for i in range(n):
        ch = fm[i]
        ch_u8 = _to_uint8_01(ch)
        cv2.imwrite(str(out_dir / f"ch{i:03d}.png"), ch_u8)


def visualise_yolo_convs(
    weights_path: str,
    image_path: str,
    out_dir: str,
    *,
    device: str = "cpu",
    layer_name_contains: str | None = None,  # e.g. "backbone" / "neck" / "Detect"
    max_layers: int = 20,
    max_channels_per_layer: int = 0,         # 0 = don't export per-channel, only mean heatmap
):
    out_dir = _ensure_dir(out_dir)

    yolo = YOLO(weights_path)
    model = yolo.model
    model.to(device)
    model.eval()

    captured: list[tuple[str, torch.Tensor]] = []

    def hook_fn(name: str):
        def _hook(module, inp, out):
            # Only keep 4D conv-like outputs
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                captured.append((name, out))
        return _hook

    # Register hooks on Conv2d modules (common choice)
    hooks = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if layer_name_contains and layer_name_contains not in name:
            continue
        hooks.append(m.register_forward_hook(hook_fn(name)))

    # Load image as BGR -> RGB -> torch BCHW float (0..1)
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(device).float() / 255.0  # HWC
    x = x.permute(2, 0, 1).unsqueeze(0)                   # BCHW

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Limit how many layers you dump (YOLO has many convs)
    captured = captured[: int(max_layers)]

    means_dir = out_dir / "means"
    means_dir.mkdir(parents=True, exist_ok=True)

    for idx, (lname, feat) in enumerate(captured, start=1):
        safe = lname.replace(".", "_").replace("/", "_")
        mean_path = means_dir / f"{idx:03d}__{safe}__mean.png"
        save_featuremap_mean(feat, mean_path, colormap=cv2.COLORMAP_VIRIDIS)

    print(f"Saved {len(captured)} mean heatmaps to: {means_dir}")

    # Save outputs
    for idx, (lname, feat) in enumerate(captured, start=1):
        safe = lname.replace(".", "_").replace("/", "_")
        layer_dir = _ensure_dir(out_dir / f"{idx:03d}__{safe}")

        # 1) mean heatmap (easy overview)
        save_featuremap_mean(feat, layer_dir / "mean.png")

        # 2) optional per-channel exports (like classification)
        if int(max_channels_per_layer) > 0:
            ch_dir = _ensure_dir(layer_dir / "channels")
            save_featuremap_channels(feat, ch_dir, max_channels=int(max_channels_per_layer))

    print(f"Saved {len(captured)} layer activations to: {out_dir}")




if __name__ == "__main__":
    visualise_yolo_convs(
        weights_path=r"training/detection/runs/detect/detection/runs/train_multiclass_detect_v6/weights/best.pt",
        image_path=r"C:\Users\joshk\OneDrive\Desktop\raw\complete\pe\whole_images\pe2_et_10X_DFK_1_1349.tiff",
        out_dir=r"C:\Users\joshk\OneDrive\Desktop\feature_maps_out\yolo_featuremaps_out",
        device="cpu",                 # or "cuda:0"
        layer_name_contains=None,      # or "Detect" to focus head
        max_layers=30,
        max_channels_per_layer=32,     # set 0 to only save mean heatmaps
    )