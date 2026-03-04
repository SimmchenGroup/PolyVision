from .data import get_classes, get_generators
from .model import build_model
from .train import train_model
from .fine_tune import fine_tune_model
from .analysis import compute_particle_area, compute_confidence, compute_correlation
import os
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

    preds = model.predict(gen, steps=steps, verbose=1)

    used = min(steps * gen.batch_size, len(gen.filepaths))
    used_filepaths = gen.filepaths[:used]
    print(f"[Predict] Done. predictions_shape={getattr(preds, 'shape', None)}, used_filepaths={len(used_filepaths)}")
    return preds, used_filepaths

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
):
    os.makedirs(run_dir, exist_ok=True)

    # -------------------------
    # Training curves
    # -------------------------
    try:
        if history1 is not None:
            plot_training_history(history1, save_path=os.path.join(run_dir, "training_phase1.png"), show=False)
        if history2 is not None:
            plot_training_history(history2, save_path=os.path.join(run_dir, "training_phase2_finetune.png"), show=False)

        # analysis-only fallback: try to plot from history.csv near the model
        if history1 is None and history2 is None and model_path_for_history:
            model_dir = os.path.dirname(model_path_for_history)
            history_csv = os.path.join(model_dir, "history.csv")
            if os.path.exists(history_csv):
                plot_training_history_csv(history_csv, save_path=os.path.join(run_dir, "training_history_from_csv.png"), show=False)
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
                    image_size=config.image_size,
                    save_path=out_path,
                    show=False,
                )
    except Exception as e:
        print(f"[WARN] Grad-CAM generation failed: {e}")

    # -------------------------
    # Single-layer usage summary (existing)
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

        with open(os.path.join(run_dir, "all_layers_usage.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[Analysis] Wrote {out_path}")
    except Exception as e:
        print(f"[WARN] All-layers analysis failed: {e}")

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


def run_full_experiment(config: ExperimentConfig):
    classes = get_classes(config)
    train_gen, val_gen = get_generators(config)

    model, base_model = build_model(config, len(classes))

    version = get_next_version(config.save_path)
    print(f"Running experiment {version}")

    run_dir = os.path.join(config.save_path, version)
    os.makedirs(run_dir, exist_ok=True)

    history1 = train_model(model, train_gen, val_gen, config, version=version)

    print(f"[{version}] Starting fine-tuning (unfreezing from 'mixed6' and training for {config.epochs} more epochs)...")
    model = fine_tune_model(model, base_model, unfreeze_from="mixed6")
    history2 = train_model(model, train_gen, val_gen, config, version=version)

    run_post_analysis(
        config=config,
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        run_dir=run_dir,
        history1=history1,
        history2=history2,
        model_path_for_history=None,
    )

    print(f"Experiment {version} completed!")
    return model, history1, history2

def analyze_only(config: ExperimentConfig, model_path: str, version: str | None = None):
    train_gen, val_gen = get_generators(config)
    model = load_model(model_path)

    model_dir = os.path.dirname(model_path)

    # If caller provides a version/name, treat it as a subfolder inside the model directory.
    # (So you can do: --version analysis_run2)
    if version:
        run_dir = os.path.join(model_dir, version)
    else:
        run_dir = model_dir

    print(f"Running analysis-only in {run_dir}")

    run_post_analysis(
        config=config,
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        run_dir=run_dir,
        history1=None,
        history2=None,
        model_path_for_history=model_path,
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