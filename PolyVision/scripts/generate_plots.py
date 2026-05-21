"""
Generate individual plots on demand without running full analysis.
Usage: python generate_plots.py --plot [plot_type] --model [path] --output [dir]
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import json

# Add matplotlib backend fix
import matplotlib as plt

plt.use('Agg')

from training.classification.config import ExperimentConfig
from training.classification.data import get_generators, get_test_generator
from tensorflow.keras.models import load_model
from training.classification.gradcam import generate_gradcam


def setup_config():
    """Setup default config"""
    return ExperimentConfig(
        training_type="microplastic",
        learning_rate=0.000001,
        base_dirs={"microplastic": r"C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6"},
        microplastic_classes=["nylon", "pe", "pet", "pla", "pmma", "pp", "ps", "pu", "pvc"],
        whisky_classes=[],
        weights_path=r"results\classification\EfficientNetB0\globalv6\best_model.keras",
        save_path=r"results\classification\EfficientNetB0\globalv6\extra",
        epochs=15,
        batch_size=64,
        model_name="efficient"
    )


def generate_confusion_matrix(model_path, output_dir, split='test'):
    """Generate only confusion matrix"""
    from training.classification.pipeline import _predict_generator, _class_names_from_directory_iterator, _save_confusion_matrix
    from sklearn.metrics import confusion_matrix

    config = setup_config()
    model = load_model(model_path)

    if split == 'test':
        gen = get_test_generator(config)
    else:
        _, gen = get_generators(config)

    print(f"[CM] Generating confusion matrix on {split}...")
    preds, _ = _predict_generator(model, gen, max_batches=None)

    y_true = np.asarray(gen.classes[:len(preds)], dtype=int)
    y_pred = np.argmax(preds, axis=1).astype(int)

    labels = np.arange(len(gen.class_indices))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = _class_names_from_directory_iterator(gen)

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"confusion_matrix_{split}.png")
    _save_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix ({split})")

    np.savetxt(os.path.join(output_dir, f"confusion_matrix_{split}.csv"), cm.astype(int), delimiter=",", fmt="%d")
    print(f"✅ Saved: {cm_path}")

def generate_size_confidence_analysis(model_path, output_dir, split='test'):
    from training.classification.pipeline import _predict_generator
    from training.classification.analysis import compute_particle_area, compute_confidence

    config = setup_config()
    model = load_model(model_path)

    # 🔥 FIX: infer correct input size from model
    input_shape = model.input_shape[1:3]
    print(f"[Size-Conf] Model expects input size: {input_shape}")

    # Override config image size
    config.image_size = input_shape

    if split == 'test':
        gen = get_test_generator(config)
    else:
        _, gen = get_generators(config)

    print(f"[Size-Conf] Running analysis on {split}...")

    preds, filepaths = _predict_generator(model, gen, max_batches=None)

    # Compute metrics
    conf = compute_confidence(preds)
    areas = compute_particle_area(filepaths, config.image_size)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Scatter plot
    # -------------------------
    try:
        plt.figure(figsize=(6, 5))
        plt.scatter(areas, conf, alpha=0.5)

        # Optional: trend line
        if len(areas) > 1:
            z = np.polyfit(areas, conf, 1)
            p = np.poly1d(z)
            plt.plot(areas, p(areas))

        plt.xlabel("Particle Area")
        plt.ylabel("Prediction Confidence")
        plt.title("Particle Size vs Confidence")

        # Optional log scale (uncomment if needed)
        # plt.xscale("log")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "size_vs_conf_scatter.png"), dpi=250)
        plt.close()

        print("  ✅ Saved scatter plot")
    except Exception as e:
        print(f"  ⚠ Scatter failed: {e}")

    # -------------------------
    # Correlation matrix
    # -------------------------
    try:
        corr_matrix = np.corrcoef(areas, conf)
        corr_value = float(corr_matrix[0, 1])

        np.savetxt(os.path.join(output_dir, "size_conf_corr.csv"),
                   corr_matrix, delimiter=",")

        with open(os.path.join(output_dir, "size_conf_corr_value.txt"), "w") as f:
            f.write(f"{corr_value}\n")

        # Heatmap
        plt.figure(figsize=(5, 4))
        im = plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(im)

        labels = ["Particle Area", "Confidence"]
        plt.xticks([0, 1], labels, rotation=45, ha="right")
        plt.yticks([0, 1], labels)

        # Values in cells
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{corr_matrix[i, j]:.3f}",
                         ha="center", va="center", color="black")

        plt.title(f"Correlation (r = {corr_value:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "size_conf_corr_heatmap.png"), dpi=250)
        plt.close()

        print(f"  ✅ Saved correlation (r = {corr_value:.4f})")

    except Exception as e:
        print(f"  ⚠ Correlation failed: {e}")

def generate_roc_curves(model_path, output_dir, split='test'):
    """Generate only ROC curves"""
    from training.classification.pipeline import _predict_generator, _class_names_from_directory_iterator
    from training.classification.visualization import plot_roc_curves

    config = setup_config()
    model = load_model(model_path)

    if split == 'test':
        gen = get_test_generator(config)
    else:
        _, gen = get_generators(config)

    print(f"[ROC] Generating ROC curves on {split}...")
    preds, _ = _predict_generator(model, gen, max_batches=None)

    y_true = np.asarray(gen.classes[:len(preds)], dtype=int)
    y_pred_probs = preds
    class_names = _class_names_from_directory_iterator(gen)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"roc_curves_{split}.png")
    plot_roc_curves(y_true, y_pred_probs, class_names, save_path=save_path)


def generate_pr_curves(model_path, output_dir, split='test'):
    """Generate only Precision-Recall curves"""
    from training.classification.pipeline import _predict_generator, _class_names_from_directory_iterator
    from training.classification.visualization import plot_precision_recall_curves

    config = setup_config()
    model = load_model(model_path)

    if split == 'test':
        gen = get_test_generator(config)
    else:
        _, gen = get_generators(config)

    print(f"[PR] Generating PR curves on {split}...")
    preds, _ = _predict_generator(model, gen, max_batches=None)

    y_true = np.asarray(gen.classes[:len(preds)], dtype=int)
    y_pred_probs = preds
    class_names = _class_names_from_directory_iterator(gen)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"pr_curves_{split}.png")
    plot_precision_recall_curves(y_true, y_pred_probs, class_names, save_path=save_path)


def generate_map(model_path, output_dir, split='test'):
    """Generate only mAP plot"""
    from training.classification.pipeline import _predict_generator, _class_names_from_directory_iterator
    from training.classification.visualization import plot_precision_recall_map

    config = setup_config()
    model = load_model(model_path)

    if split == 'test':
        gen = get_test_generator(config)
    else:
        _, gen = get_generators(config)

    print(f"[mAP] Generating mAP @ thresholds on {split}...")
    preds, _ = _predict_generator(model, gen, max_batches=None)

    y_true = np.asarray(gen.classes[:len(preds)], dtype=int)
    y_pred_probs = preds
    class_names = _class_names_from_directory_iterator(gen)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"map_confidence_thresholds_{split}.png")
    map_values = plot_precision_recall_map(y_true, y_pred_probs, class_names, save_path=save_path)

    # Save JSON
    with open(os.path.join(output_dir, f"map_values_{split}.json"), 'w') as f:
        json.dump(map_values, f, indent=2)


def generate_calibration(model_path, output_dir, split='test'):
    """Generate only calibration curves"""
    from training.classification.pipeline import _predict_generator, _class_names_from_directory_iterator
    from training.classification.visualization import plot_calibration_curves, plot_combined_calibration

    config = setup_config()
    model = load_model(model_path)

    if split == 'test':
        gen = get_test_generator(config)
    else:
        _, gen = get_generators(config)

    print(f"[Calibration] Generating calibration curves on {split}...")
    preds, _ = _predict_generator(model, gen, max_batches=None)

    y_true = np.asarray(gen.classes[:len(preds)], dtype=int)
    y_pred_probs = preds
    class_names = _class_names_from_directory_iterator(gen)

    os.makedirs(output_dir, exist_ok=True)

    # Individual
    plot_calibration_curves(
        y_true, y_pred_probs, class_names,
        save_path=os.path.join(output_dir, f"calibration_individual_{split}.png")
    )

    # Combined
    plot_combined_calibration(
        y_true, y_pred_probs, class_names,
        save_path=os.path.join(output_dir, f"calibration_combined_{split}.png")
    )


def generate_confidence_hist(model_path, output_dir, split='test'):
    """Generate only confidence histogram"""
    from training.classification.pipeline import _predict_generator, _class_names_from_directory_iterator
    from training.classification.visualization import plot_confidence_histogram

    config = setup_config()
    model = load_model(model_path)

    if split == 'test':
        gen = get_test_generator(config)
    else:
        _, gen = get_generators(config)

    print(f"[Confidence] Generating confidence histogram on {split}...")
    preds, _ = _predict_generator(model, gen, max_batches=None)

    y_pred_probs = preds
    class_names = _class_names_from_directory_iterator(gen)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"confidence_histogram_{split}.png")
    plot_confidence_histogram(y_pred_probs, class_names, save_path=save_path)


def generate_network_viz(model_path, output_dir, layer_analysis_json=None):
    """Generate network visualization from existing layer analysis"""
    from training.classification.visualization import (
        plot_layer_activations_heatmap,
        plot_network_summary_bars,
        plot_neuron_importance_per_layer
    )

    if layer_analysis_json is None:
        # Try to find it next to the model
        model_dir = os.path.dirname(model_path)
        layer_analysis_json = os.path.join(model_dir, "all_layers_usage.json")

    if not os.path.exists(layer_analysis_json):
        print(f"❌ Layer analysis not found: {layer_analysis_json}")
        print("   Run full analysis first to generate all_layers_usage.json")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("[NetworkViz] Generating network visualizations...")

    # Heatmap
    plot_layer_activations_heatmap(
        layer_analysis_json,
        save_path=os.path.join(output_dir, "network_heatmap.png"),
        top_n_layers=30
    )

    # Bar chart
    plot_network_summary_bars(
        layer_analysis_json,
        save_path=os.path.join(output_dir, "layer_strength_ranking.png")
    )

    # Neuron importance
    try:
        plot_neuron_importance_per_layer(
            layer_analysis_json,
            layer_name=None,
            save_path=os.path.join(output_dir, "neuron_importance_mixed7.png")
        )
    except Exception as e:
        print(f"  ⚠ Neuron importance plot failed: {e}")


def generate_gradcam(model_path, output_dir, num_samples=5):
    """Generate only Grad-CAM visualizations"""
    from training.classification.gradcam import render_gradcam_for_path

    config = setup_config()
    model = load_model(model_path)
    _, val_gen = get_generators(config)

    print(f"[GradCAM] Generating {num_samples} Grad-CAM samples...")

    if not hasattr(val_gen, 'filepaths') or not val_gen.filepaths:
        print("❌ No validation images found")
        return

    os.makedirs(output_dir, exist_ok=True)
    sample_paths = val_gen.filepaths[:num_samples]

    for i, img_path in enumerate(sample_paths, start=1):
        out_path = os.path.join(output_dir, f"gradcam_sample_{i:02d}.png")
        render_gradcam_for_path(
            model=model,
            img_path=img_path,
            last_conv_layer_name=None,
            save_path=out_path,
            show=False
        )

def main():
    parser = argparse.ArgumentParser(description="Generate individual plots")
    parser.add_argument('--plot', required=True, choices=[
        'confusion', 'roc', 'pr', 'map', 'calibration', 'confidence',
        'network', 'gradcam', 'size_conf', 'all'
    ], help='Type of plot to generate')
    parser.add_argument('--model', default=r"results\classification\v1.0\best_model.keras",
                        help='Path to model')
    parser.add_argument('--output', default=r"results\classification\v1.0\plots",
                        help='Output directory')
    parser.add_argument('--split', default='test', choices=['test', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--layer-json', default=None,
                        help='Path to all_layers_usage.json (for network viz)')


    args = parser.parse_args()

    print(f" Generating {args.plot} plot(s)...")
    print(f" Model: {args.model}")
    print(f" Output: {args.output}\n")

    if args.plot == 'confusion' or args.plot == 'all':
        generate_confusion_matrix(args.model, args.output, args.split)

    if args.plot == 'roc' or args.plot == 'all':
        generate_roc_curves(args.model, args.output, args.split)

    if args.plot == 'pr' or args.plot == 'all':
        generate_pr_curves(args.model, args.output, args.split)

    if args.plot == 'map' or args.plot == 'all':
        generate_map(args.model, args.output, args.split)

    if args.plot == 'calibration' or args.plot == 'all':
        generate_calibration(args.model, args.output, args.split)

    if args.plot == 'confidence' or args.plot == 'all':
        generate_confidence_hist(args.model, args.output, args.split)

    if args.plot == 'network' or args.plot == 'all':
        generate_network_viz(args.model, args.output, args.layer_json)

    if args.plot == 'gradcam' or args.plot == 'all':
        generate_gradcam(args.model, args.output, num_samples=5)

    if args.plot == 'size_conf' or args.plot == 'all':
        generate_size_confidence_analysis(args.model, args.output, args.split)

    print(f"\n✅ Done! Check: {args.output}")


if __name__ == "__main__":
    main()

# cd C:\Users\joshk\OneDrive\Documents\GitHub_Strath\PolyVision
#
# :: Generate only confusion matrix
# python generate_plots.py --plot confusion
#
# :: Generate only ROC curves
# python generate_plots.py --plot roc
#
# :: Generate only Precision-Recall curves
# python generate_plots.py --plot pr
#
# :: Generate only mAP plot
# python generate_plots.py --plot map
#
# :: Generate only calibration curves
# python generate_plots.py --plot calibration
#
# :: Generate only confidence histogram
# python generate_plots.py --plot confidence
#
# :: Generate only network visualization (requires all_layers_usage.json)
# python generate_plots.py --plot network
#
# :: Generate only Grad-CAM samples
# python -m scripts.generate_plots --plot gradcam

# python -m scripts.generate_plots --plot size_conf
#
# :: Generate ALL plots
# python generate_plots.py --plot all
#
# :: Custom output directory
# python generate_plots.py --plot roc --output "classification\results\v1.3\custom_plots"
#
# :: Use validation split instead of test
# python generate_plots.py --plot confusion --split val
#
# :: Specify different model
# python generate_plots.py --plot map --model "classification\results\v1.2\best_model.keras"

# python generate_plots.py --plot network --layer-json "classification\results\v1.3\all_layers_usage.json" --output "classification\results\v1.3\network_viz"

# python -m scripts.generate_plots --plot all --model "C:\Users\joshk\OneDrive\Documents\GitHub_Strath\PolyVision\results\classification\EfficientNetB0\globalv6\best_model.keras" --layer-json "results/classification/EfficientNetB0/globalv6/all_layers_usage.json" --output "results/classification/EfficientNetB0/globalv6/extra"