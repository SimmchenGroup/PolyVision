import matplotlib
matplotlib.use('Agg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from .data import get_classes, get_generators
from .model import build_model
from .train import train_model
from .fine_tune import fine_tune_model
from .analysis import compute_particle_area, compute_confidence, compute_correlation
import re
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from .config import ExperimentConfig
from .visualization import plot_training_history, plot_training_history_csv
from .gradcam import render_gradcam_for_path
from .layer_analysis import analyze_layer_usage, summarize_layer_usage, analyze_all_layers
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .data import get_tfdata_datasets, compute_steps_per_epoch, get_test_generator
from .metadata import write_run_metadata


def _predict_generator(model, gen, max_batches=None):
    """
    Predict on a DirectoryIterator safely. Returns (predictions, used_filepaths).
    If max_batches is None -> uses full generator once.
    """
    gen.reset()

    if max_batches is None:
        steps = math.ceil(gen.samples / gen.batch_size)
    else:
        steps = int(max_batches)

    preds = model.predict(gen, steps=steps, verbose=0)

    used = min(steps * gen.batch_size, len(gen.filepaths))
    used_filepaths = gen.filepaths[:used]
    print(f"[Predict] Done. predictions_shape={getattr(preds, 'shape', None)}, used_filepaths={len(used_filepaths)}")
    return preds, used_filepaths

def _class_names_from_directory_iterator(gen):
    """
    Return class names in index order for a Keras DirectoryIterator.
    """
    if not hasattr(gen, "class_indices") or not gen.class_indices:
        return None
    inv = {idx: name for name, idx in gen.class_indices.items()}
    return [inv[i] for i in range(len(inv))]

def _save_confusion_matrix(cm, class_names, out_png_path, title="Confusion Matrix"):
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    disp.ax_.grid(False)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close()

def _pick_last_conv_layer_name(model, preferred: str | None = None) -> str:
    """
    Pick a reasonable conv layer for Grad-CAM/layer analysis across different backbones.
    - If `preferred` exists, use it.
    - Else, return the last Conv2D/DepthwiseConv2D layer name.
    """
    if preferred:
        try:
            model.get_layer(preferred)
            return preferred
        except Exception:
            pass

    conv_names = []
    for layer in model.layers:
        cls = layer.__class__.__name__.lower()
        if "conv2d" in cls or "depthwiseconv2d" in cls:
            conv_names.append(layer.name)

    if not conv_names:
        # Fallback: last layer with 4D output if any
        for layer in reversed(model.layers):
            try:
                shape = getattr(layer.output, "shape", None)
                if shape is not None and len(shape) == 4:
                    return layer.name
            except Exception:
                continue
        raise ValueError("Could not find a suitable conv layer for Grad-CAM.")

    return conv_names[-1]

def run_post_analysis(
        config: ExperimentConfig,
        model,
        train_gen,
        val_gen,
        run_dir: str,
        history1=None,
        history2=None,
        model_path_for_history: str | None = None,
        gradcam_samples: int = 5,
        layer_name: str = "mixed7",
        layer_batches: int = 10,
        all_layers_batches: int = 10,
        pred_max_batches=None,
        cm_gen=None,
):
    os.makedirs(run_dir, exist_ok=True)

    # Choose a layer that actually exists for this backbone
    layer_name = _pick_last_conv_layer_name(model, preferred=layer_name)
    print(f"[Analysis] Using layer for Grad-CAM/layer-usage: {layer_name}")

    # -------------------------
    # Training curves
    # -------------------------
    try:
        # ... existing code ...
        pass
    except Exception as e:
        print(f"[WARN] Training history plots failed: {e}")

    # -------------------------
    # Grad-CAM
    # -------------------------
    try:
        if hasattr(val_gen, "filepaths") and val_gen.filepaths:
            sample_paths = val_gen.filepaths[:max(0, int(gradcam_samples))]
            for i, p in enumerate(sample_paths, start=1):
                out_path = os.path.join(run_dir, f"gradcam_val_{i:02d}.png")
                render_gradcam_for_path(
                    model=model,
                    img_path=p,
                    last_conv_layer_name=layer_name,
                    save_path=out_path,
                    show=False,
                )
    except Exception as e:
        print(f"[WARN] Grad-CAM generation failed: {e}")

    # -------------------------
    # Single-layer usage summary
    # -------------------------
    try:
        usage = analyze_layer_usage(model, val_gen, layer_name=layer_name, max_batches=int(layer_batches))
        summary = summarize_layer_usage(usage)
        with open(os.path.join(run_dir, f"layer_usage_{layer_name}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"[WARN] Layer analysis failed: {e}")

        # -------------------------
        # All-layers usage summary (new)
        # -------------------------
    try:
        print(f"[Analysis] Starting analyze_all_layers (max_batches={all_layers_batches})...")
        usage_summary = analyze_all_layers(model, train_gen, max_batches=int(all_layers_batches), visualize_conv=False)
        print(f"[Analysis] analyze_all_layers complete. layers_analyzed={len(usage_summary)}")

        # Save compact JSON (arrays converted to lists)
        out = {}
        for lname, u in usage_summary.items():
            out[lname] = {
                "mean_activation": np.asarray(u["mean_activation"]).tolist(),
                "nonzero_fraction": np.asarray(u["nonzero_fraction"]).tolist(),
                "top_indices": np.asarray(u["top_indices"]).tolist(),
                "summary": summarize_layer_usage(u),
            }

        out_path = os.path.join(run_dir, "all_layers_usage.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[Analysis] Wrote {out_path}")
    except Exception as e:
        print(f"[WARN] All-layers analysis failed: {e}")

        # -------------------------
        # Network Visualizations (Static only)
        # -------------------------
        try:
            from .visualization import (
                plot_layer_activations_heatmap,
                plot_network_summary_bars,
                plot_neuron_importance_per_layer,
            )

            usage_json = os.path.join(run_dir, "all_layers_usage.json")

            # Check if all_layers_usage.json exists
            if os.path.exists(usage_json):
                print("[Viz] Generating network visualizations...")

                # 1. Heatmap of all layers
                print("  [1/3] Creating activation heatmap...")
                plot_layer_activations_heatmap(
                    usage_json,
                    save_path=os.path.join(run_dir, "network_heatmap.png"),
                    top_n_layers=30
                )

                # 2. Bar chart summary
                print("  [2/3] Creating layer strength ranking...")
                plot_network_summary_bars(
                    usage_json,
                    save_path=os.path.join(run_dir, "layer_strength_ranking.png")
                )

                # 3. Individual layer analysis (if layer exists in analysis)
                print("  [3/3] Creating neuron importance plot...")
                try:
                    plot_neuron_importance_per_layer(
                        usage_json,
                        layer_name=layer_name,  # This should be "mixed7" or whatever layer was analyzed
                        save_path=os.path.join(run_dir, f"neuron_importance_{layer_name}.png"),
                        top_n=50
                    )
                except Exception as e:
                    print(f"    ⚠ Skipped neuron importance plot: {e}")

                print("[Viz] Network visualizations complete!")
            else:
                print(f"[Viz] SKIPPED: {usage_json} not found")
                print("     Run with all_layers_batches > 0 to generate layer analysis first")
        except Exception as e:
            import traceback
            print(f"[WARN] Network visualization failed: {e}")
            traceback.print_exc()

    # -------------------------
    # Size vs prediction-confidence correlation matrix (new)
    # -------------------------
    try:
        mb = "ALL" if pred_max_batches is None else str(pred_max_batches)
        print(f"[Corr] Starting size vs confidence correlation (pred_max_batches={mb})...")

        preds, used_filepaths = _predict_generator(model, val_gen, max_batches=pred_max_batches)
        print("[Corr] Computing confidence...")
        conf = compute_confidence(preds)
        print(f"[Corr] Confidence computed. n={len(conf)}")

        print("[Corr] Computing particle areas (simple threshold)...")
        areas = compute_particle_area(used_filepaths, config.image_size)
        print(f"[Corr] Areas computed. n={len(areas)}")

        corr_matrix = np.corrcoef(areas, conf)
        corr_value = float(corr_matrix[0, 1])
        print(f"[Corr] Correlation computed: {corr_value}")

        np.savetxt(os.path.join(run_dir, "size_conf_corr.csv"), corr_matrix, delimiter=",")

        with open(os.path.join(run_dir, "size_conf_corr_value.txt"), "w", encoding="utf-8") as f:
            f.write(f"{corr_value}\n")

        plt.figure(figsize=(4, 4))
        plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(label="Correlation")
        plt.xticks([0, 1], ["particle_area", "confidence"], rotation=45, ha="right")
        plt.yticks([0, 1], ["particle_area", "confidence"])
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "size_conf_corr_heatmap.png"), dpi=250)
        plt.close()

        print(f"[Corr] Wrote size_conf_corr.csv / size_conf_corr_heatmap.png in {run_dir}")
    except Exception as e:
        print(f"[WARN] Size/conf correlation failed: {e}")

    # -------------------------
    # Confusion matrix (now uses TEST generator if provided)
    # -------------------------
    try:
        mb = "ALL" if pred_max_batches is None else str(pred_max_batches)

        if cm_gen is None:
            cm_gen = val_gen
            cm_split = "val"
        else:
            cm_split = "test"

        print(f"[CM] Generating confusion matrix on {cm_split} (pred_max_batches={mb})...")

        preds, used_filepaths = _predict_generator(model, cm_gen, max_batches=pred_max_batches)
        used = len(used_filepaths)

        if not hasattr(cm_gen, "classes"):
            raise RuntimeError("cm_gen has no `.classes` attribute (expected a DirectoryIterator).")

        y_true = np.asarray(cm_gen.classes[:used], dtype=int)
        y_pred = np.argmax(preds[:used], axis=1).astype(int)

        if not hasattr(cm_gen, "class_indices") or not cm_gen.class_indices:
            raise RuntimeError("cm_gen has no `.class_indices` mapping (needed for display labels).")

        # labels ensure stable ordering even if some classes are missing in the split
        labels = np.arange(len(cm_gen.class_indices))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        class_names = _class_names_from_directory_iterator(cm_gen)

        cm_png = os.path.join(run_dir, f"confusion_matrix_{cm_split}.png")
        _save_confusion_matrix(cm, class_names, cm_png, title=f"Confusion Matrix ({cm_split})")
        np.savetxt(os.path.join(run_dir, f"confusion_matrix_{cm_split}.csv"), cm.astype(int), delimiter=",",
                   fmt="%d")

        print(f"[CM] Wrote {cm_png} and confusion_matrix_{cm_split}.csv in {run_dir}")
    except Exception as e:
        print(f"[WARN] Confusion matrix failed: {e}")

    # -------------------------
    # Scatter: particle size vs confidence (NEW)
    # -------------------------
    try:
        plt.figure(figsize=(6, 5))

        plt.scatter(areas, conf, alpha=0.5)
        plt.xscale("log")
        plt.xlabel("Particle Area")
        plt.ylabel("Prediction Confidence")
        plt.title("Particle Size vs Prediction Confidence")

        # Optional: log scale if areas vary a lot
        # plt.xscale("log")

        # Optional: trend line
        if len(areas) > 1:
            z = np.polyfit(areas, conf, 1)
            p = np.poly1d(z)
            plt.plot(areas, p(areas))

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "size_vs_conf_scatter.png"), dpi=250)
        plt.close()

        print(f"[Corr] Wrote size_vs_conf_scatter.png in {run_dir}")
    except Exception as e:
        print(f"[WARN] Scatter plot failed: {e}")

    # -------------------------
    # Advanced Evaluation Metrics (NEW)
    # -------------------------
    try:
        from .visualization import (
            plot_precision_recall_map,
            plot_roc_curves,
            plot_precision_recall_curves,
            plot_calibration_curves,
            plot_combined_calibration,
            plot_confidence_histogram,
            generate_classification_report_txt
        )

        mb = "ALL" if pred_max_batches is None else str(pred_max_batches)
        print(f"[Metrics] Computing advanced metrics (pred_max_batches={mb})...")

        # Get predictions
        if cm_gen is None:
            cm_gen = val_gen
            split_name = "val"
        else:
            split_name = "test"

        preds, used_filepaths = _predict_generator(model, cm_gen, max_batches=pred_max_batches)
        used = len(used_filepaths)

        y_true = np.asarray(cm_gen.classes[:used], dtype=int)
        y_pred = np.argmax(preds[:used], axis=1).astype(int)
        y_pred_probs = preds[:used]

        class_names = _class_names_from_directory_iterator(cm_gen)

        # 1. mAP at different confidence thresholds (like IoU 0.5, 0.75, 0.95)
        print("[Metrics] Computing mAP @ confidence thresholds...")
        map_values = plot_precision_recall_map(
            y_true,
            y_pred_probs,
            class_names,
            save_path=os.path.join(run_dir, f"map_confidence_thresholds_{split_name}.png"),
            iou_thresholds=[0.5, 0.75, 0.95]
        )

        # Save mAP values to JSON
        with open(os.path.join(run_dir, f"map_values_{split_name}.json"), 'w') as f:
            json.dump(map_values, f, indent=2)

        # 2. ROC Curves
        print("[Metrics] Generating ROC curves...")
        aucs = plot_roc_curves(
            y_true,
            y_pred_probs,
            class_names,
            save_path=os.path.join(run_dir, f"roc_curves_{split_name}.png")
        )

        # 3. Precision-Recall Curves
        print("[Metrics] Generating PR curves...")
        aps = plot_precision_recall_curves(
            y_true,
            y_pred_probs,
            class_names,
            save_path=os.path.join(run_dir, f"pr_curves_{split_name}.png")
        )

        # 4. Calibration Curves (individual)
        print("[Metrics] Generating calibration curves...")
        plot_calibration_curves(
            y_true,
            y_pred_probs,
            class_names,
            save_path=os.path.join(run_dir, f"calibration_individual_{split_name}.png"),
            n_bins=10
        )

        # 5. Calibration Curves (combined)
        eces = plot_combined_calibration(
            y_true,
            y_pred_probs,
            class_names,
            save_path=os.path.join(run_dir, f"calibration_combined_{split_name}.png"),
            n_bins=10
        )

        # 6. Confidence histogram
        print("[Metrics] Generating confidence distribution...")
        plot_confidence_histogram(
            y_pred_probs,
            class_names,
            save_path=os.path.join(run_dir, f"confidence_histogram_{split_name}.png")
        )

        # 7. Detailed text report
        print("[Metrics] Generating classification report...")
        generate_classification_report_txt(
            y_true,
            y_pred,
            class_names,
            save_path=os.path.join(run_dir, f"classification_report_{split_name}.txt")
        )

        # Save summary metrics to JSON
        summary = {
            "split": split_name,
            "num_samples": int(used),
            "map_values": {f"conf_{k}": float(v) for k, v in map_values.items()},
            "roc_auc_per_class": {class_names[i]: float(aucs[i]) for i in range(len(class_names))},
            "roc_auc_macro": float(np.mean(aucs)),
            "pr_ap_per_class": {class_names[i]: float(aps[i]) for i in range(len(class_names))},
            "pr_map": float(np.mean(aps)),
            "ece_per_class": {class_names[i]: float(eces[i]) if not np.isnan(eces[i]) else None for i in
                              range(len(class_names))},
            "ece_mean": float(np.nanmean(eces))
        }

        with open(os.path.join(run_dir, f"metrics_summary_{split_name}.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[Metrics] Advanced metrics complete! Summary:")
        print(f"  mAP@0.5  = {map_values[0.5]:.4f}")
        print(f"  mAP@0.75 = {map_values[0.75]:.4f}")
        print(f"  mAP@0.95 = {map_values[0.95]:.4f}")
        print(f"  ROC AUC (macro) = {np.mean(aucs):.4f}")
        print(f"  PR mAP   = {np.mean(aps):.4f}")
        print(f"  ECE (mean) = {np.nanmean(eces):.4f}")

    except Exception as e:
        import traceback

        print(f"[WARN] Advanced metrics generation failed: {e}")
        traceback.print_exc()

def run_full_experiment(config: ExperimentConfig, balance_mode):
    classes = get_classes(config)

    balance_mode = (balance_mode or "none").strip().lower()
    if balance_mode not in {"none", "balanced"}:
        raise ValueError("balance_mode must be 'none' or 'balanced'")

    train_ds, val_ds = get_tfdata_datasets(
            config,
            model_name=getattr(config, "model_name", "inception"),
            balance_mode=balance_mode,
        )

    # Build analysis generators (DirectoryIterators) for Grad-CAM/CM/plots
    train_gen, val_gen = get_generators(config, model_name=getattr(config, "model_name", "inception"))

    model_name = getattr(config, "model_name", "inception")
    model, base_model = build_model(config, len(classes), model_name=model_name)

    version = get_next_version(config.save_path)
    print(f"Running experiment {version}")

    run_dir = os.path.join(config.save_path, version)
    os.makedirs(run_dir, exist_ok=True)

    # IMPORTANT: if balance_mode="balanced", train_ds is infinite -> you MUST pass steps_per_epoch
    steps_per_epoch = getattr(config, "steps_per_epoch", None)
    if balance_mode == "balanced":
        if steps_per_epoch is None:
            steps_per_epoch = compute_steps_per_epoch(config)
        print(f"[Train] balance_mode=balanced steps_per_epoch={steps_per_epoch}")
    else:
        steps_per_epoch = None
        print(f"[Train] balance_mode=none steps_per_epoch=None")

        # Save run metadata BEFORE training starts (pretrain stage)
        write_run_metadata(
            run_dir,
            config=config,
            model=model,
            stage="pretrain_init",
            extra={
                "version": version,
                "balance_mode": balance_mode,
                "steps_per_epoch": steps_per_epoch,
                "num_classes": len(classes),
                "class_names": list(classes),
                "train_base_dir_override": getattr(config, "train_base_dir_override", None),
                "train_class_indices": getattr(train_gen, "class_indices", None),
                "val_class_indices": getattr(val_gen, "class_indices", None),
            },
        )

    history1 = train_model(
            model,
            train_ds,
            val_ds,
            config,
            version=version,
            steps_per_epoch=steps_per_epoch,
            history_filename="history_pretrain.csv",
        )

    # Save run metadata AFTER switching to fine-tune (finetune stage)
    write_run_metadata(
        run_dir,
        config=config,
        model=model,
        stage="finetune_init",
        extra={
            "version": version,
            "balance_mode": balance_mode,
            "steps_per_epoch": steps_per_epoch,
            "num_classes": len(classes),
            "class_names": list(classes),
            "train_base_dir_override": getattr(config, "train_base_dir_override", None),
            "train_class_indices": getattr(train_gen, "class_indices", None),
            "val_class_indices": getattr(val_gen, "class_indices", None),
        },
    )

    print(f"[{version}] Starting fine-tuning...")
    model = fine_tune_model(model, base_model, model_name=model_name)
    history2 = train_model(
        model,
        train_ds,
        val_ds,
        config,
        version=version,
        steps_per_epoch=steps_per_epoch,
        history_filename="history_finetune.csv",
    )

    # Keep your existing test generator for now (ok), OR switch later
    test_gen = get_test_generator(config, model_name=model_name)

    run_post_analysis(
        config=config,
        model=model,
        train_gen=train_gen,   # or update run_post_analysis to accept train_ds
        val_gen=val_gen,     # or update run_post_analysis to accept val_ds
        run_dir=run_dir,
        history1=history1,
        history2=history2,
        model_path_for_history=None,
        cm_gen=test_gen,
    )

    print(f"Experiment {version} completed!")
    return model, history1, history2

def analyze_only(config: ExperimentConfig, model_path: str, version: str | None = None):
    train_gen, val_gen = get_generators(config, model_name=getattr(config, "model_name", "inception"))
    model = load_model(model_path)

    model_dir = os.path.dirname(model_path)

    # If caller provides a version/name, treat it as a subfolder inside the model directory.
    # (So you can do: --version analysis_run2)
    if version:
        run_dir = os.path.join(model_dir, version)
    else:
        run_dir = model_dir

    print(f"Running analysis-only in {run_dir}")

    test_gen = get_test_generator(config)

    run_post_analysis(
        config=config,
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        run_dir=run_dir,
        history1=None,
        history2=None,
        model_path_for_history=model_path,
        cm_gen=test_gen,
    )
    return model, run_dir

def get_next_version(save_path, prefix="v"):
    if not os.path.exists(save_path):
        return f"{prefix}1.0"

    versions = []
    for name in os.listdir(save_path):
        match = re.match(rf"{prefix}(\d+)\.(\d+)", name)
        if match:
            major, minor = map(int, match.groups())
            versions.append((major, minor))

    if not versions:
        return f"{prefix}1.0"

    major, minor = max(versions)
    minor += 1
    return f"{prefix}{major}.{minor}"