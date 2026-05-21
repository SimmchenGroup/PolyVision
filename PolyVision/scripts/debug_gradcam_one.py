from pathlib import Path
import tensorflow as tf

from training.classification.gradcam import render_gradcam_for_path

MODEL_PATH = r"C:\Users\joshk\OneDrive\Documents\GitHub_Strath\PolyVision\results\classification\EfficientNetB0\globalv6\best_model.keras"
IMG_PATH = r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6\val\pe\pe2_et_10X_DFK_1_1349.jpg"  # change this

model = tf.keras.models.load_model(MODEL_PATH)

conv_layers = [
    l.name for l in model.layers
    if "conv" in l.__class__.__name__.lower() or "depthwise" in l.__class__.__name__.lower()
]
print("num conv-ish layers:", len(conv_layers))
print("last 30 conv-ish layers:")
for name in conv_layers[-30:]:
    print("  ", name)

render_gradcam_for_path(
    model=model,
    img_path=IMG_PATH,
    model_name="efficient",
    last_conv_layer_name="block7a_project_conv",
    class_index=1,   # pred1
    debug=True,
    show=True,
    show_negative=True,
    cmap="jet",
)