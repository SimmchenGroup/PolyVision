import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


def convert_to_jpg(input_path: str, output_path: str | None = None, quality: int = 95) -> str:
    """
    Converts input image (e.g. TIFF/PNG/WEBP) to JPEG on disk and returns the JPEG path.
    """
    input_path = str(input_path)

    if output_path is None:
        root, _ext = os.path.splitext(input_path)
        output_path = root + ".jpg"
    output_path = str(output_path)

    img = tf.keras.utils.load_img(input_path)  # uses PIL under the hood
    if img.mode != "RGB":
        img = img.convert("RGB")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path, format="JPEG", quality=int(quality), optimize=True)
    return output_path


def get_model_io(model_name: str):
    """
    Matches common training pipelines:
      - inception/inceptionv3 -> (299,299) + inception_preprocess
      - efficient/efficientnet/efficientb0 -> (224,224) + efficientnet_preprocess
      - efficientb4/efficientnetb4 -> (380,380) + efficientnet_preprocess
      - res/resnet/resnet50 -> (224,224) + efficientnet_preprocess
    """
    model_name = (model_name or "inception").lower().strip()

    if model_name in {"inception", "inceptionv3"}:
        return (299, 299), inception_preprocess
    if model_name in {"efficient", "efficientnet", "efficientb0"}:
        return (224, 224), efficientnet_preprocess
    if model_name in {"efficientb4", "efficientnetb4"}:
        return (380, 380), efficientnet_preprocess
    if model_name in {"res", "resnet", "resnet50"}:
        return (224, 224), efficientnet_preprocess

    raise ValueError(f"Unknown model_name={model_name!r}")


def load_image_for_model(image_path: str, model_name: str = "efficient"):
    target_size, preprocess_fn = get_model_io(model_name)

    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    x = tf.keras.utils.img_to_array(img)   # (H, W, 3), float32 in 0..255
    x = np.expand_dims(x, axis=0)          # (1, H, W, 3)
    x = preprocess_fn(x)                   # same style preprocessing as training
    return img, x


def build_activation_model(model: tf.keras.Model, layer_names=None):
    if layer_names is None:
        layer_names = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    outputs = [model.get_layer(n).output for n in layer_names]
    return tf.keras.Model(inputs=model.input, outputs=outputs), layer_names


def plot_feature_maps(feature_map, max_channels=16, title=None):
    fm = feature_map[0]
    c = fm.shape[-1]
    n = min(c, max_channels)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        x = fm[..., i]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        ax.imshow(x, cmap="magma")
        ax.set_axis_off()
        ax.set_title(f"ch {i}", fontsize=10)
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_feature_map_channels_as_jpg(
    feature_map: np.ndarray,
    output_dir: str,
    prefix: str,
    max_channels: int | None = None,
    quality: int = 95,
    cmap_name: str = "magma",
):
    """
    Saves each channel of a feature map as a separate JPG, using the same kind of colormap
    you see in matplotlib imshow(..., cmap="viridis") by default.

    feature_map: shape (1, H, W, C) or (H, W, C)
    Output files: <output_dir>/<prefix>_ch000.jpg, ...
    """
    os.makedirs(output_dir, exist_ok=True)

    fm = feature_map[0] if feature_map.ndim == 4 else feature_map
    if fm.ndim != 3:
        raise ValueError(f"Expected feature_map with 3 dims (H,W,C) (or 4 incl batch); got shape {fm.shape}")

    _h, _w, c = fm.shape
    n = c if max_channels is None else min(int(max_channels), int(c))

    cmap = plt.get_cmap(cmap_name)

    for i in range(n):
        x = fm[..., i].astype(np.float32)

        # Normalize per-channel to 0..1 (like your imshow normalization)
        x = x - float(np.min(x))
        x = x / (float(np.max(x)) + 1e-8)

        # Apply colormap -> RGBA in 0..1, then convert to RGB uint8
        rgba = cmap(x)                      # (H, W, 4), float64 in 0..1
        rgb_u8 = (rgba[..., :3] * 255.0).round().clip(0, 255).astype(np.uint8)  # (H, W, 3)

        jpg_bytes = tf.image.encode_jpeg(rgb_u8, quality=int(quality))
        out_path = os.path.join(output_dir, f"{prefix}_ch{i:03d}.jpg")
        tf.io.write_file(out_path, jpg_bytes)

# ---- Example usage ----
if __name__ == "__main__":
    input_path = r"C:\Users\joshk\OneDrive\Desktop\raw\complete\pe\whole_images\pe2_et_10X_DFK_1_1349.tiff"
    output_dir = r"C:\Users\joshk\OneDrive\Desktop\feature_maps_out\whole_magma"

    jpg_path = convert_to_jpg(input_path)  # writes alongside input by default
    model_name = "efficient"              # change to: inception / efficientb4 / resnet50 etc.

    target_size, _ = get_model_io(model_name)
    h, w = target_size

    model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(h, w, 3),
    )

    _pil_img, x = load_image_for_model(jpg_path, model_name=model_name)

    activation_model, conv_layer_names = build_activation_model(model)
    activations = activation_model.predict(x, verbose=0)

    # Choose which layer to export (0 = first Conv2D layer)
    layer_idx = 0
    layer_name = conv_layer_names[layer_idx]
    fm = activations[layer_idx]

    # Save each channel as a JPG
    save_feature_map_channels_as_jpg(
        fm,
        output_dir=output_dir,
        prefix=f"{layer_name}",
        max_channels=None,   # set None to save all channels
        quality=95,
        cmap_name="magma",
    )

    # Optional: quick grid preview
    plot_feature_maps(fm, max_channels=16, title=f"Feature maps: {layer_name}")