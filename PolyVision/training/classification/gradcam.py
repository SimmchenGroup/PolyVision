from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def _build_config_for_dataset_root(dataset_root: str | Path, *, model_name: str | None = None, batch_size: int = 64):
    """
    Small local config builder so this module does not depend on scripts.generate_plots (avoids circular imports).
    Only fields needed by _train_val_dirs / generators are required.
    """
    from training.classification.config import ExperimentConfig

    return ExperimentConfig(
        training_type="microplastic",
        learning_rate=1e-4,
        base_dirs={"microplastic": str(dataset_root)},
        microplastic_classes=[],
        whisky_classes=[],
        weights_path="",
        save_path="",
        epochs=1,
        batch_size=int(batch_size),
        model_name=(model_name or "inception").strip().lower(),
    )

def _make_val_generator_from_config(config, *, model_name: str | None):
    """
    Build only a validation DirectoryIterator (does not touch train/).
    This avoids crashing when train/ doesn't exist for a given dataset root.
    """
    from training.classification.data import _train_val_dirs  # local import to avoid cycles
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

    name = (model_name or getattr(config, "model_name", "inception")).lower().strip()

    if name in {"inception", "inceptionv3"}:
        target_size = (299, 299)
        preprocess_fn = inception_preprocess
    elif name in {"res", "resnet", "resnet50"}:
        target_size = (224, 224)
        preprocess_fn = resnet_preprocess
    elif name in {"efficientb4", "efficientnetb4", "effb4"}:
        target_size = (380, 380)
        preprocess_fn = efficientnet_preprocess
    else:
        target_size = (224, 224)
        preprocess_fn = efficientnet_preprocess

    _train_dir, val_dir = _train_val_dirs(config)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=int(getattr(config, "batch_size", 32)),
        class_mode="categorical",
        shuffle=False,
    )
    return val_gen

def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Pick the last Conv2D/DepthwiseConv2D-like layer with 4D output."""
    for layer in reversed(model.layers):
        cls = layer.__class__.__name__.lower()
        if ("conv2d" in cls or "depthwiseconv2d" in cls) and hasattr(layer, "output"):
            try:
                if len(layer.output.shape) == 4:
                    return layer.name
            except Exception:
                continue
    raise ValueError("No suitable conv layer found for Grad-CAM.")


def _model_has_builtin_preprocessing(model: tf.keras.Model) -> bool:
    """
    Heuristic: many Keras application models may include preprocessing layers
    such as Rescaling/Normalization. If present, DON'T apply preprocess_input again.
    """
    for layer in model.layers[:10]:
        cls = layer.__class__.__name__.lower()
        if "rescaling" in cls or "normalization" in cls:
            return True
    return False


def _identity_preprocess(x: np.ndarray) -> np.ndarray:
    return x

def _get_preprocess_and_size(
    model: tf.keras.Model,
    model_name: str | None = None,
) -> tuple[tuple[int, int], callable]:
    """
    Returns (target_size, preprocess_fn) matching the backbone.

    IMPORTANT:
      - For EfficientNet-like models, we avoid double-preprocessing:
        if the loaded model already has preprocessing layers, we skip preprocess_input.
    """
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

    name = (model_name or "").strip().lower()

    if name in {"inception", "inceptionv3"}:
        return (299, 299), inception_preprocess

    if name in {"res", "resnet", "resnet50"}:
        return (224, 224), resnet_preprocess

    if name in {"efficient", "efficientnet", "efficientnetb0", "effb0"}:
        preprocess_fn = _identity_preprocess if _model_has_builtin_preprocessing(model) else efficientnet_preprocess
        return (224, 224), preprocess_fn

    if name in {"efficientb4", "efficientnetb4", "effb4"}:
        preprocess_fn = _identity_preprocess if _model_has_builtin_preprocessing(model) else efficientnet_preprocess
        return (380, 380), preprocess_fn

    # Infer size from model input
    ishape = getattr(model, "input_shape", None)
    h = ishape[1] if ishape and len(ishape) >= 3 else None
    w = ishape[2] if ishape and len(ishape) >= 3 else None

    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        # Default: for unknown models, do NOT apply any special preprocess unless you know it.
        return (h, w), _identity_preprocess

    return (224, 224), _identity_preprocess


def _load_and_preprocess_image(
    img_path: str | Path,
    target_size: tuple[int, int],
    preprocess_fn,
    *,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, target_size), cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32)

    x = preprocess_fn(x)

    if debug:
        mn = float(np.min(x))
        mx = float(np.max(x))
        mean = float(np.mean(x))
        print(f"[GradCAM] preprocess stats: min={mn:.4f} max={mx:.4f} mean={mean:.4f} shape={x.shape}")

    x = np.expand_dims(x, axis=0)
    return img_rgb, x


def render_gradcam_for_path(
    model: tf.keras.Model,
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
):
    target_size, preprocess_fn = _get_preprocess_and_size(model, model_name=model_name)

    img_rgb, img_array = _load_and_preprocess_image(
        img_path, target_size, preprocess_fn, debug=debug
    )

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Pre-softmax "logits" assumption: model.layers[-1] is softmax Dense
    logits_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.layers[-1].input],
    )

    with tf.GradientTape() as tape:
        conv_out, logits = logits_model(img_array, training=False)

        if debug:
            lg = logits[0].numpy()
            print(f"[GradCAM] logits stats: min={lg.min():.4f} max={lg.max():.4f}")

        if class_index is None:
            class_index = int(tf.argmax(logits[0]).numpy())
        class_channel = logits[:, class_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    cam = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    pos = tf.maximum(cam, 0)
    neg = tf.maximum(-cam, 0)  # negative evidence

    def _normalize(hm: tf.Tensor) -> tf.Tensor:
        hmax = tf.reduce_max(hm)
        if debug:
            print(f"[GradCAM] heatmap max(before norm)={float(hmax.numpy()):.8f}")
        return tf.cond(
            hmax > 1e-8,
            lambda: hm / (hmax + tf.keras.backend.epsilon()),
            lambda: tf.zeros_like(hm),
        )

    pos = _normalize(pos)
    neg = _normalize(neg)

    def _apply_cmap(hm01: np.ndarray) -> np.ndarray:
        """
        hm01: float array in [0,1], shape (H,W)
        returns RGB uint8 image (H,W,3)
        """
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

        # Fallback to matplotlib colormaps (always available here)
        m = plt.get_cmap(name if name in plt.colormaps() else "magma")
        rgba = m(hm01)  # (H,W,4) float 0..1
        rgb = (rgba[..., :3] * 255.0).round().clip(0, 255).astype(np.uint8)
        return rgb

    def _overlay(hm01: tf.Tensor) -> np.ndarray:
        hm01_rs = cv2.resize(hm01.numpy(), target_size).astype(np.float32)
        hm01_rs = np.clip(hm01_rs, 0.0, 1.0)
        hm_rgb = _apply_cmap(hm01_rs)
        return np.uint8(hm_rgb * float(alpha) + img_rgb)

    super_pos = _overlay(pos)
    super_neg = _overlay(neg)

    if show or save_path:
        if show_negative:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.title("Original")
            plt.imshow(img_rgb)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title(f"Grad-CAM + (class={class_index})")
            plt.imshow(super_pos)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Grad-CAM - (class={class_index})")
            plt.imshow(super_neg)
            plt.axis("off")
        else:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(img_rgb)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title(f"Grad-CAM (class={class_index})")
            plt.imshow(super_pos)
            plt.axis("off")

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=200)
        if show:
            plt.show()
        plt.close()

    return int(class_index)

def _predict_on_filepaths(
    model: tf.keras.Model,
    filepaths: list[str],
    *,
    model_name: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Predict probabilities on a list of image paths using the same preprocessing as Grad-CAM.
    Returns probs: (N, C)
    """
    target_size, preprocess_fn = _get_preprocess_and_size(model, model_name=model_name)

    xs = []
    probs_out = []

    for p in filepaths:
        _img_rgb, x = _load_and_preprocess_image(p, target_size, preprocess_fn)
        xs.append(x[0])  # strip batch dim -> HWC

        if len(xs) >= int(batch_size):
            batch = np.stack(xs, axis=0)
            probs_out.append(model.predict(batch, verbose=0))
            xs.clear()

    if xs:
        batch = np.stack(xs, axis=0)
        probs_out.append(model.predict(batch, verbose=0))

    return np.concatenate(probs_out, axis=0)


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
):
    """
    Saves Grad-CAM for:
      - top_k most confident predictions (max softmax prob highest)
      - bottom_k least confident predictions (max softmax prob lowest)

    For each selected image, saves two maps:
      - predicted class (argmax)
      - least likely class (argmin)
    """
    from training.classification.data import get_generators
    # from scripts.generate_plots import setup_config

    output_dir = Path(output_dir) / "gradcams"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config_for_dataset_root(dataset_root, model_name=model_name, batch_size=int(batch_size))
    model = tf.keras.models.load_model(str(model_path))

    val_gen = _make_val_generator_from_config(config, model_name=model_name)

    if not getattr(val_gen, "filepaths", None):
        raise RuntimeError("No validation images found (val_gen.filepaths is empty).")

    filepaths = list(val_gen.filepaths)

    print(f"[GradCAM] Predicting on {len(filepaths)} validation images...")
    probs = _predict_on_filepaths(
        model,
        filepaths,
        model_name=model_name,
        batch_size=int(getattr(config, "batch_size", 32)),
    )

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    least = probs.argmin(axis=1)

    top_idx = np.argsort(-conf)[: int(top_k)]
    bottom_idx = np.argsort(conf)[: int(bottom_k)]

    # Build class name list in index order
    class_names = None
    if hasattr(val_gen, "class_indices") and isinstance(val_gen.class_indices, dict) and val_gen.class_indices:
        inv = {idx: name for name, idx in val_gen.class_indices.items()}
        class_names = [inv[i] for i in range(len(inv))]

    if class_names is None:
        num_classes = int(probs.shape[1])
        class_names = [f"class_{i}" for i in range(num_classes)]

    last_conv = find_last_conv_layer(model)
    print(f"[GradCAM] Using last conv layer: {last_conv}")
    print(f"[GradCAM] Writing outputs under: {output_dir}")

    def _save_two_cams(out_dir: Path, rank: int, p: str, c: float, pred_k: int, least_k: int):
        base = Path(p).stem

        # Predicted class CAM
        render_gradcam_for_path(
            model=model,
            img_path=p,
            model_name=model_name,
            last_conv_layer_name=last_conv,
            class_index=pred_k,
            alpha=alpha,
            show=False,
            save_path=out_dir / f"{rank:02d}_{base}__pred{pred_k}__conf{c:.3f}.png",
        )

        # Least-likely class CAM
        render_gradcam_for_path(
            model=model,
            img_path=p,
            model_name=model_name,
            last_conv_layer_name=last_conv,
            class_index=least_k,
            alpha=alpha,
            show=False,
            save_path=out_dir / f"{rank:02d}_{base}__least{least_k}__conf{c:.3f}.png",
        )

    def _save_set(indices: np.ndarray, tag: str):
        out_dir = output_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for rank, i in enumerate(indices, start=1):
            p = filepaths[int(i)]
            c = float(conf[int(i)])
            pred_k = int(pred[int(i)])
            least_k = int(least[int(i)])

            print(f"[GradCAM] {tag} {rank}: conf={c:.4f} pred={pred_k} least={least_k} path={p}")
            _save_two_cams(out_dir, rank, p, c, pred_k, least_k)

    # 1) Top/bottom confidence overall
    _save_set(top_idx, "top_confidence")
    _save_set(bottom_idx, "bottom_confidence")

    # 2) Top K per class (by confidence among samples predicted as that class)
    per_class_root = output_dir / "per_class_top"
    per_class_root.mkdir(parents=True, exist_ok=True)

    for k, cname in enumerate(class_names):
        mask = (pred == int(k))
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            print(f"[GradCAM] per_class_top/{cname}: no samples predicted as this class; skipping.")
            continue

        # rank within class by confidence
        idxs_sorted = idxs[np.argsort(-conf[idxs])]
        chosen = idxs_sorted[: int(top_k)]

        out_dir = per_class_root / cname
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GradCAM] per_class_top/{cname}: saving {len(chosen)} samples")
        for rank, i in enumerate(chosen, start=1):
            p = filepaths[int(i)]
            c = float(conf[int(i)])
            pred_k = int(pred[int(i)])
            least_k = int(least[int(i)])
            _save_two_cams(out_dir, rank, p, c, pred_k, least_k)

    print(f"[GradCAM] Done. Outputs in: {output_dir}")


# Backwards-compatible wrapper (kept name, new behavior)
def generate_gradcam(model_path, output_dir, num_samples=5):
    # Old behavior: first N val images; new behavior: keep it simple, do top/bottom with k=num_samples
    return generate_gradcam_top_bottom_confidence(
        model_path=model_path,
        output_dir=output_dir,
        model_name=None,
        top_k=int(num_samples),
        bottom_k=int(num_samples),
        alpha=0.4,
    )

# ... existing code above ...

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM exporter (top/bottom confidence + per-class top-k).")
    parser.add_argument("--model-path", required=True, help="Path to a saved .keras model.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing val/ (and optionally train/test).")
    parser.add_argument("--out-dir", required=True, help="Output directory (gradcams/ will be created inside).")
    parser.add_argument("--model-name", default=None, choices=[None, "inception", "efficient", "efficientb4", "res"], help="Backbone name.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bottom-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.4)

    args = parser.parse_args()

    generate_gradcam_top_bottom_confidence(
        model_path=args.model_path,
        output_dir=args.out_dir,
        model_name=args.model_name,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        alpha=args.alpha,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
    )

# python -m training.classification.gradcam ^
#   --model-path "C:\Users\joshk\OneDrive\Documents\GitHub_Strath\PolyVision\results\classification\v1.14\best_model.keras" ^
#   --dataset-root "C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6" ^
#   --out-dir "C:\Users\joshk\OneDrive\Documents\GitHub_Strath\PolyVision\results\classification\v1.14" ^
#   --model-name efficient ^
#   --top-k 5 ^
#   --bottom-k 5
#
# generate_gradcam_top_bottom_confidence(
#     model_path="results/classification/v1.12/best_model.keras",
#     output_dir="results/classification/v1.12/gradcam_top_bottom",
#     model_name="inception",
#     top_k=5,
#     bottom_k=5,
# )

