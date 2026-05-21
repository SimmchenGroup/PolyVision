"""
Test network visualization generation
"""

from pathlib import Path
from training.classification.config import ExperimentConfig
from training.classification.pipeline import analyze_only

MODEL_PATH = r"classification\results\v1.3\best_model.keras"
OUTPUT_VERSION = "test_viz"

BASE_DIRS = {
    "microplastic": r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv3",
}

config = ExperimentConfig(
    training_type="microplastic",
    learning_rate=0.0001,
    base_dirs=BASE_DIRS,
    microplastic_classes=[
        "Nylon", "PE", "PET", "PLA", "PMMA", "PP", "PS", "PU", "PVC"
    ],
    whisky_classes=[],
    weights_path=r"training\classification\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    save_path=r"results\classification",
    epochs=15,
    batch_size=64,
    image_size=(150, 150),
    model_name="inception"
)

if __name__ == "__main__":
    print(" Testing network visualization...")

    # Check if all_layers_usage.json already exists
    existing_json = Path(r"results\classification\v1.3\all_layers_usage.json")
    if existing_json.exists():
        print(f"✅ Found existing layer analysis: {existing_json}")
        print("  Will generate visualizations from this file\n")
    else:
        print("⚠ No existing layer analysis found")
        print("  Will analyze layers first (this may take a few minutes)\n")

    model, run_dir = analyze_only(config, MODEL_PATH, version=OUTPUT_VERSION)

    print(f"\n✅ Complete! Check: {run_dir}")
    print("\nExpected files:")
    print("  - network_heatmap.png")
    print("  - layer_strength_ranking.png")
    print("  - neuron_importance_mixed7.png")
    print("  - network_interactive.html (if plotly installed)")
    print("  - network_flow_sankey.html (if plotly installed)")