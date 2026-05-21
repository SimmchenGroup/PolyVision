"""
Generate layer analysis JSON (required for network visualizations)
This only needs to be run once per model.
"""

import os
import json
import numpy as np
from pathlib import Path
from training.classification.config import ExperimentConfig
from training.classification.data import get_generators
from training.classification.layer_analysis import analyze_all_layers, summarize_layer_usage
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = r"classification\results\v1.3\best_model.keras"
OUTPUT_JSON = r"classification\results\v1.3\all_layers_usage.json"
MAX_BATCHES = 10  # Increase for more accurate analysis (takes longer)

BASE_DIRS = {
    "microplastic": r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv3",
}

config = ExperimentConfig(
    training_type="microplastic",
    learning_rate=0.0001,
    base_dirs=BASE_DIRS,
    microplastic_classes=["nylon", "pe", "pmma", "pp", "ps", "pu", "pvc"],
    whisky_classes=[],
    weights_path=r"classification\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    save_path=r"results\classification",
    epochs=15,
    batch_size=64,
    image_size=(150, 150),
    model_name="inception"
)

if __name__ == "__main__":
    print(" Generating layer analysis...")
    print(f" Model: {MODEL_PATH}")
    print(f" Output: {OUTPUT_JSON}")
    print(f" Max batches: {MAX_BATCHES} (this may take 2-5 minutes)\n")

    # Load model and data
    model = load_model(MODEL_PATH)
    train_gen, val_gen = get_generators(config)

    # Analyze all layers
    print("[1/2] Analyzing all layers...")
    usage_summary = analyze_all_layers(
        model,
        train_gen,
        max_batches=MAX_BATCHES,
        visualize_conv=False,
        verbose=True,
        print_every=20
    )

    print(f"\n[2/2] Saving results to JSON...")

    # Convert to JSON-serializable format
    output = {}
    for layer_name, usage in usage_summary.items():
        output[layer_name] = {
            "mean_activation": np.asarray(usage["mean_activation"]).tolist(),
            "nonzero_fraction": np.asarray(usage["nonzero_fraction"]).tolist(),
            "top_indices": np.asarray(usage["top_indices"]).tolist(),
            "summary": summarize_layer_usage(usage),
        }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Layer analysis complete!")
    print(f" Analyzed {len(output)} layers")
    print(f" Saved to: {OUTPUT_JSON}")
    print(f" File size: {os.path.getsize(OUTPUT_JSON) / 1024:.1f} KB")
    print("\n Now you can generate network visualizations!")