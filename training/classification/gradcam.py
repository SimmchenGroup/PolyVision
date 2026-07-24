from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model input size / preprocessing helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INCEPTION_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_INCEPTION_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _get_preprocess_and_size(model: nn.Module, model_name: str | None = None):
    name = (model_name or "").strip().lower()
    if name in {"inception", "inceptionv3"}:
        return (299, 299), _INCEPTION_MEAN, _INCEPTION_STD
    if name in {"res", "resnet", "resnet50"}:
        return (224, 224), _IMAGENET_MEAN, _IMAGENET_STD
    if name in {"efficientb4", "efficientnetb4", "effb4"}:
        return (380, 380), _IMAGENET_MEAN, _IMAGENET_STD
    # efficient / unknown — infer size from model if possible
    if name in {"efficient", "efficientnet", "efficientnetb0", "effb0"}:
        return (224, 224), _IMAGENET_MEAN, _IMAGENET_STD
    return (224, 224), _IMAGENET_MEAN, _IMAGENET_STD


def find_last_conv_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """Return (name, module) of the last Conv2d in the model."""
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            last_name, last_mod = name, mod
    if last_mod is None:
        raise ValueError("No Conv2d found in model.")
    return last_name, last_mod


def _load_and_preprocess(img_path, target_size, mean, std, device, debug=False):
    """Load image, resize, normalise, return (img_rgb_np, tensor_1CHW)."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = target_size
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    if debug:
        print(f"[GradCAM] preprocess stats: min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f}")
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return img_rgb, tensor


# ---------------------------------------------------------------------------
# Grad-CAM via PyTorch hooks
# ---------------------------------------------------------------------------

class _GradCAMHooks:
    """Registers forward + backward hooks on a target layer."""

    def __init__(self, layer: nn.Module):
        self.activations = None
        self.gradients = None
        self._fwd_hook = layer.register_forward_hook(self._fwd)
        self._bwd_hook = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, input, output):
        self.activations = output.detach()

    def _bwd(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _compute_gradcam(model: nn.Module, tensor: torch.Tensor, layer: nn.Module, class_index: int | None):
    hooks = _GradCAMHooks(layer)
    model.eval()

    output = model(tensor)
    if isinstance(output, tuple):
        output = output[0]

    if class_index is None:
        class_index = int(output.argmax(dim=1).item())

    model.zero_grad()
    score = output[0, class_index]
    score.backward()

    hooks.remove()

    acts = hooks.activations[0]   # (C, H, W)
    grads = hooks.gradients[0]    # (C, H, W)

    # Global average pool the gradients
    weights = grads.mean(dim=(1, 2))  # (C,)

    cam = (weights[:, None, None] * acts).sum(dim=0)  # (H, W)
    pos = torch.clamp(cam, min=0)
    neg = torch.clamp(-cam, min=0)

    def _norm(t):
        mx = t.max()
        return t / (mx + 1e-8) if mx > 1e-8 else torch.zeros_like(t)

    return _norm(pos).cpu().numpy(), _norm(neg).cpu().numpy(), class_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _apply_cmap(hm01: np.ndarray, cmap: str = "magma") -> np.ndarray:
    name = (cmap or "magma").strip().lower()
    cv2_map = {
        "jet": getattr(cv2, "COLORMAP_JET", None),
        "turbo": getattr(cv2, "COLORMAP_TURBO", None),
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", None),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", None),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", None),
        "magma": getattr(cv2, "COLORMAP_MAGMA", None),
    }.get(name)
    hm_u8 = np.uint8(np.clip(hm01 * 255.0, 0, 255))
    if cv2_map is not None:
        bgr = cv2.applyColorMap(hm_u8, int(cv2_map))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    m = plt.get_cmap(name if name in plt.colormaps() else "magma")
    rgba = m(hm01)
    return (rgba[..., :3] * 255.0).round().clip(0, 255).astype(np.uint8)


def _overlay_core(img_rgb, hm01, w, h, alpha, cmap, gamma, per_pixel_alpha, blur):
    """Composite one normalised CAM (hm01, any size) onto img_rgb (h×w). Smooth
    cubic upsample + light blur removes the coarse 7×7 blockiness; per-pixel alpha
    keeps cold regions transparent so the original image shows through."""
    hm_rs = cv2.resize(hm01, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    hm_rs = np.clip(hm_rs, 0.0, 1.0)
    if blur and blur > 1:
        k = int(blur) | 1
        hm_rs = np.clip(cv2.GaussianBlur(hm_rs, (k, k), 0), 0.0, 1.0)
    if gamma != 1.0:
        hm_rs = np.power(hm_rs, float(gamma))
    hm_rgb = _apply_cmap(hm_rs, cmap).astype(np.float32)
    amap = (float(alpha) * hm_rs)[..., None] if per_pixel_alpha else float(alpha)
    blend = img_rgb.astype(np.float32) * (1.0 - amap) + hm_rgb * amap
    return np.clip(blend, 0, 255).astype(np.uint8)


def _resolve_layer(model, last_conv_layer_name):
    if last_conv_layer_name is not None:
        for name, mod in model.named_modules():
            if name == last_conv_layer_name:
                return mod
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")
    return find_last_conv_layer(model)[1]


def gradcam_arrays(
    model: nn.Module,
    img_path: str | Path,
    *,
    model_name: str | None = None,
    last_conv_layer_name: str | None = None,
    class_index: int | None = None,
    alpha: float = 0.4,
    cmap: str = "magma",
    gamma: float = 1.0,
    per_pixel_alpha: bool = True,
    blur: int = 11,
):
    """Compute a Grad-CAM overlay and return it as an array (no file written).

    Returns (overlay_rgb_uint8, used_class_index, confidence_of_that_class, img_rgb).
    If class_index is None the model's predicted class is used.
    """
    device = _get_device()
    model = model.to(device)
    target_size, mean, std = _get_preprocess_and_size(model, model_name)
    img_rgb, tensor = _load_and_preprocess(img_path, target_size, mean, std, device)
    h, w = target_size
    layer = _resolve_layer(model, last_conv_layer_name)
    pos, _neg, used_ci = _compute_gradcam(model, tensor, layer, class_index)
    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, tuple):
            out = out[0]
        conf = float(torch.softmax(out, dim=1)[0, used_ci].item())
    overlay = _overlay_core(img_rgb, pos, w, h, alpha, cmap, gamma, per_pixel_alpha, blur)
    return overlay, int(used_ci), conf, img_rgb


def render_gradcam_for_path(
    model: nn.Module,
    img_path: str | Path,
    *,
    model_name: str | None = None,
    last_conv_layer_name: str | None = None,
    class_index: int | None = None,
    alpha: float = 0.4,
    show: bool = False,
    save_path: str | Path | None = None,
    debug: bool = False,
    show_negative: bool = False,
    cmap: str = "magma",
    gamma: float = 1.0,
    per_pixel_alpha: bool = True,
    blur: int = 11,
) -> int:
    device = _get_device()
    model = model.to(device)

    target_size, mean, std = _get_preprocess_and_size(model, model_name)
    img_rgb, tensor = _load_and_preprocess(img_path, target_size, mean, std, device, debug=debug)
    h, w = target_size

    # Find target conv layer
    if last_conv_layer_name is not None:
        layer = None
        for name, mod in model.named_modules():
            if name == last_conv_layer_name:
                layer = mod
                break
        if layer is None:
            raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")
    else:
        _, layer = find_last_conv_layer(model)

    pos, neg, class_index = _compute_gradcam(model, tensor, layer, class_index)

    def _overlay(hm01):
        return _overlay_core(img_rgb, hm01, w, h, alpha, cmap, gamma, per_pixel_alpha, blur)

    super_pos = _overlay(pos)
    super_neg = _overlay(neg)

    if show or save_path:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(super_pos)
        ax.axis("off")

        plt.tight_layout(pad=0.5)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    return int(class_index)


def _predict_on_filepaths(
    model: nn.Module,
    filepaths: list[str],
    *,
    model_name: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    device = _get_device()
    model = model.to(device)
    model.eval()

    target_size, mean, std = _get_preprocess_and_size(model, model_name)
    probs_out = []
    batch_tensors = []

    for p in filepaths:
        try:
            _, tensor = _load_and_preprocess(p, target_size, mean, std, device)
            batch_tensors.append(tensor)
        except Exception:
            continue

        if len(batch_tensors) >= int(batch_size):
            batch = torch.cat(batch_tensors, dim=0)
            with torch.no_grad():
                logits = model(batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs_out.append(torch.softmax(logits, dim=1).cpu().numpy())
            batch_tensors.clear()

    if batch_tensors:
        batch = torch.cat(batch_tensors, dim=0)
        with torch.no_grad():
            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs_out.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(probs_out, axis=0) if probs_out else np.empty((0,))


def generate_gradcam_top_bottom_confidence(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    top_k: int = 5,
    bottom_k: int = 5,
    alpha: float = 0.4,
    dataset_root: str | Path | None = None,
    batch_size: int = 64,
    cmap: str = "magma",
    gamma: float = 1.0,
):
    device = _get_device()
    model = torch.load(str(model_path), map_location=device, weights_only=False)

    output_dir = Path(output_dir) / "gradcams"
    output_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    target_size, mean, std = _get_preprocess_and_size(model, model_name)
    h, w = target_size
    tf = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(),
                              transforms.Normalize(mean=mean.tolist(), std=std.tolist())])

    base_dir = Path(dataset_root) if dataset_root else None
    val_dir = base_dir / "val" if base_dir else None
    if val_dir is None or not val_dir.exists():
        raise RuntimeError(f"val/ directory not found at: {val_dir}")

    val_dataset = datasets.ImageFolder(str(val_dir), transform=tf)
    filepaths = [s[0] for s in val_dataset.samples]
    class_names = val_dataset.classes

    print(f"[GradCAM] Predicting on {len(filepaths)} validation images...")
    probs = _predict_on_filepaths(model, filepaths, model_name=model_name, batch_size=batch_size)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    least = probs.argmin(axis=1)

    top_idx = np.argsort(-conf)[:top_k]
    bottom_idx = np.argsort(conf)[:bottom_k]

    _, last_layer = find_last_conv_layer(model)

    def _save_two_cams(out_dir, rank, p, c, pred_k, least_k):
        base = Path(p).stem
        render_gradcam_for_path(model=model, img_path=p, model_name=model_name,
                                class_index=pred_k, alpha=alpha, show=False, cmap=cmap, gamma=gamma,
                                save_path=out_dir / f"{rank:02d}_{base}__pred{pred_k}__conf{c:.3f}.png")
        render_gradcam_for_path(model=model, img_path=p, model_name=model_name,
                                class_index=least_k, alpha=alpha, show=False, cmap=cmap, gamma=gamma,
                                save_path=out_dir / f"{rank:02d}_{base}__least{least_k}__conf{c:.3f}.png")

    def _save_set(indices, tag):
        out_dir = output_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(indices, start=1):
            _save_two_cams(out_dir, rank, filepaths[int(i)], float(conf[int(i)]),
                           int(pred[int(i)]), int(least[int(i)]))

    _save_set(top_idx, "top_confidence")
    _save_set(bottom_idx, "bottom_confidence")

    per_class_root = output_dir / "per_class_top"
    per_class_root.mkdir(parents=True, exist_ok=True)
    for k, cname in enumerate(class_names):
        mask = (pred == k)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            continue
        idxs_sorted = idxs[np.argsort(-conf[idxs])]
        chosen = idxs_sorted[:top_k]
        out_dir = per_class_root / cname
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(chosen, start=1):
            _save_two_cams(out_dir, rank, filepaths[int(i)], float(conf[int(i)]),
                           int(pred[int(i)]), int(least[int(i)]))

    print(f"[GradCAM] Done. Outputs in: {output_dir}")


def generate_gradcam(model_path, output_dir, num_samples=5):
    return generate_gradcam_top_bottom_confidence(
        model_path=model_path, output_dir=output_dir,
        model_name=None, top_k=int(num_samples), bottom_k=int(num_samples), alpha=0.4,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bottom-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    generate_gradcam_top_bottom_confidence(
        model_path=args.model_path, output_dir=args.out_dir,
        model_name=args.model_name, top_k=args.top_k, bottom_k=args.bottom_k,
        alpha=args.alpha, dataset_root=args.dataset_root, batch_size=args.batch_size,
    )
