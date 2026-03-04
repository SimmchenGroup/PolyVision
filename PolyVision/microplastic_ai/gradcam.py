import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def _compute_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Returns a float heatmap in [0, 1] with shape (H, W) for img_array (1, H, W, 3).
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)  # ReLU
    denom = tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = heatmap / denom
    return heatmap.numpy()

def render_gradcam_for_path(model, img_path, last_conv_layer_name="mixed7", image_size=(150, 150),
                            alpha=0.4, save_path=None, show=False):
    """
    Loads an image from disk, computes Grad-CAM, and saves (or shows) a side-by-side figure.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(str(img_path))

    img_bgr = cv2.resize(img_bgr, image_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_array = np.expand_dims(img_rgb / 255.0, axis=0)

    heatmap = _compute_gradcam_heatmap(img_array, model, last_conv_layer_name)

    heatmap_u8 = np.uint8(255 * cv2.resize(heatmap, image_size))
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(heatmap_rgb * alpha + img_rgb)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(superimposed)
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=200)
    if show:
        plt.show()
    plt.close()