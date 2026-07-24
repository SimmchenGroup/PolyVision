"""
End-to-end classification training pipeline.

Orchestrates a full run: load config, build datasets/loaders, construct the
pretrained model, then train in two phases — (1) head-only with the backbone frozen,
(2) fine-tune with the top backbone layers unfrozen (SGD, class-weighted loss, label
smoothing). Handles versioned output directories, checkpointing the best model,
early stopping, resuming from a crash, and writing training history + metadata so a
run is fully reproducible. Used by both the Local and Global classifier runs.
"""
import matplotlib
matplotlib.use('Agg')

import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .data import get_classes, get_tfdata_datasets, compute_steps_per_epoch, get_test_generator
from .model import build_model, get_model_input_size
from .train import train_model, CrashLogger
from .fine_tune import fine_tune_model, make_sgd_optimizer
from .config import ExperimentConfig
from .gradcam import render_gradcam_for_path, find_last_conv_layer


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _attach_sync_hooks(model: nn.Module) -> list:
    """
    Register a torch.cuda.synchronize() forward hook on every Conv2d layer.

    PTX JIT compilation fires mid-forward-pass, per kernel. The TDR watchdog
    cannot be reset by inter-iteration synchronize() calls — it fires during
    the forward pass itself if a single kernel takes longer than TdrDelay.
    Attaching a sync hook after every Conv2d breaks the forward pass into
    many short segments, each guaranteed to finish well within TdrDelay.

    Returns hook handles. Remove all after warmup — these must NOT be active
    during real training (synchronizing after every Conv2d is prohibitively slow).
    """
    handles = []
    def _sync(module, inp, out):
        torch.cuda.synchronize()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_sync))
    return handles


def _attach_sync_hooks_bwd(model: nn.Module) -> list:
    """
    Register a torch.cuda.synchronize() backward hook on every Conv2d layer.

    Mirrors _attach_sync_hooks for the backward pass so that backward-pass
    PTX kernels are also compiled in short TDR-safe segments during warmup.
    Without this, the backward half of each warmup iteration runs as one
    uninterrupted GPU submission — a single slow backward kernel can trigger
    TDR before the per-iteration synchronize() is reached.
    """
    handles = []
    def _sync_bwd(module, grad_in, grad_out):
        torch.cuda.synchronize()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_full_backward_hook(_sync_bwd))
    return handles


def _warmup_gpu(model: nn.Module, device, image_h: int, image_w: int, batch_size: int = 8):
    """
    Force PTX JIT compilation for all kernels before the real training loop starts.

    On Blackwell GPUs (RTX 5060 Ti, sm_120a), PyTorch has no precompiled wheels —
    every unique CUDA kernel JIT-compiles from PTX on first use. InceptionV3 and
    EfficientNet have many unique kernel types; JIT can take 5–30 min per model on
    first run. If this compile happens inside the training loop, the Windows TDR
    watchdog (TdrDelay) kills the CUDA context mid-epoch.

    batch_size must match the real training batch size — PTX kernels are compiled per
    unique CUDA grid/block configuration. Using a different size here than in training
    means the first real batch still triggers JIT mid-epoch.

    Per-Conv2d sync hooks break each forward pass into short segments so TDR resets
    after every layer, not just between iterations. Hooks are removed after warmup.
    Compiled kernels are cached in ~/.nv/ComputeCache and subsequent runs are fast.
    """
    if device.type != "cuda":
        return
    print(
        f"[Warmup] Pre-compiling CUDA/PTX kernels for {type(model.backbone).__name__} "
        f"({image_h}×{image_w}, batch={batch_size}) — first run only, may take 5–30 min on Blackwell..."
    )
    model = model.to(device)
    model.train()
    if hasattr(model, "backbone"):
        for m in model.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

    dummy_x = torch.zeros(batch_size, 3, image_h, image_w, device=device)
    dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )

    # Disable inplace activations before attaching backward hooks.
    # register_full_backward_hook wraps tensors in a view; subsequent inplace ops
    # (e.g. EfficientNet's silu_) on those views raise RuntimeError during forward.
    _inplace_off = [m for m in model.modules() if getattr(m, 'inplace', False)]
    for m in _inplace_off:
        m.inplace = False

    fwd_handles = _attach_sync_hooks(model)
    bwd_handles = _attach_sync_hooks_bwd(model)
    n_hooks = len(fwd_handles) + len(bwd_handles)
    print(f"[Warmup] Attached TDR-reset hooks to {len(fwd_handles)} Conv2d layers "
          f"({len(fwd_handles)} forward + {len(bwd_handles)} backward = {n_hooks} total).")

    WARMUP_ITERS = 3
    for i in range(WARMUP_ITERS):
        opt.zero_grad()
        logits = model(dummy_x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, dummy_y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    for handle in fwd_handles + bwd_handles:
        handle.remove()
    for m in _inplace_off:
        m.inplace = True

    torch.cuda.empty_cache()
    print("[Warmup] CUDA kernel cache populated — training will now run at full speed.")


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_loader(model: nn.Module, loader: DataLoader, device=None):
    """Run model over a DataLoader. Returns (probs_np, y_true_np, filepaths)."""
    if device is None:
        device = _get_device()
    model = model.to(device)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    filepaths = [s[0] for s in loader.dataset.samples]
    return probs_np, labels_np, filepaths


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Post-analysis
# ---------------------------------------------------------------------------

def run_post_analysis(
        config: ExperimentConfig,
        model: nn.Module,
        train_gen,       # DataLoader
        val_gen,         # DataLoader
        run_dir: str,
        history1=None,
        history2=None,
        model_path_for_history: str | None = None,
        gradcam_samples: int = 5,
        layer_name: str | None = None,
        layer_batches: int = 10,
        all_layers_batches: int = 10,
        pred_max_batches=None,
        cm_gen=None,     # DataLoader (test) or None → uses val_gen
):
    os.makedirs(run_dir, exist_ok=True)
    device = _get_device()

    # Find last conv layer for Grad-CAM
    try:
        _, conv_module = find_last_conv_layer(model)
        conv_layer_name = layer_name  # used for naming files only
        print(f"[Analysis] Found conv layer for Grad-CAM.")
    except Exception as e:
        conv_module = None
        print(f"[WARN] Could not find conv layer: {e}")

    # --- Grad-CAM ---
    try:
        val_filepaths = [s[0] for s in val_gen.dataset.samples]
        sample_paths = val_filepaths[:max(0, int(gradcam_samples))]
        for i, p in enumerate(sample_paths, start=1):
            out_path = os.path.join(run_dir, f"gradcam_val_{i:02d}.png")
            render_gradcam_for_path(
                model=model,
                img_path=p,
                model_name=getattr(config, "model_name", None),
                show=False,
                save_path=out_path,
            )
    except Exception as e:
        print(f"[WARN] Grad-CAM generation failed: {e}")

    # --- Confusion matrix ---
    try:
        eval_loader = cm_gen if cm_gen is not None else val_gen
        cm_split = "test" if cm_gen is not None else "val"
        print(f"[CM] Generating confusion matrix on {cm_split}...")

        preds_np, y_true, _ = _predict_loader(model, eval_loader, device)
        y_pred = preds_np.argmax(axis=1)
        class_names = eval_loader.dataset.classes

        labels = np.arange(len(class_names))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        cm_png = os.path.join(run_dir, f"confusion_matrix_{cm_split}.png")
        _save_confusion_matrix(cm, class_names, cm_png, title=f"Confusion Matrix ({cm_split})")
        np.savetxt(os.path.join(run_dir, f"confusion_matrix_{cm_split}.csv"), cm.astype(int),
                   delimiter=",", fmt="%d")
        print(f"[CM] Wrote {cm_png}")
    except Exception as e:
        print(f"[WARN] Confusion matrix failed: {e}")
        import traceback; traceback.print_exc()

    # --- Advanced metrics ---
    try:
        from .visualization import (
            plot_precision_recall_map,
            plot_roc_curves,
            plot_precision_recall_curves,
            plot_calibration_curves,
            plot_combined_calibration,
            plot_confidence_histogram,
            generate_classification_report_txt,
        )

        eval_loader = cm_gen if cm_gen is not None else val_gen
        split_name = "test" if cm_gen is not None else "val"
        print(f"[Metrics] Computing advanced metrics on {split_name}...")

        preds_np, y_true, _ = _predict_loader(model, eval_loader, device)
        y_pred = preds_np.argmax(axis=1)
        class_names = eval_loader.dataset.classes

        map_values = plot_precision_recall_map(
            y_true, preds_np, class_names,
            save_path=os.path.join(run_dir, f"map_confidence_thresholds_{split_name}.png"),
            iou_thresholds=[0.5, 0.75, 0.95],
        )
        with open(os.path.join(run_dir, f"map_values_{split_name}.json"), "w") as f:
            json.dump(map_values, f, indent=2)

        aucs = plot_roc_curves(y_true, preds_np, class_names,
                               save_path=os.path.join(run_dir, f"roc_curves_{split_name}.png"))
        aps = plot_precision_recall_curves(y_true, preds_np, class_names,
                                           save_path=os.path.join(run_dir, f"pr_curves_{split_name}.png"))
        plot_calibration_curves(y_true, preds_np, class_names,
                                save_path=os.path.join(run_dir, f"calibration_individual_{split_name}.png"))
        eces = plot_combined_calibration(y_true, preds_np, class_names,
                                         save_path=os.path.join(run_dir, f"calibration_combined_{split_name}.png"))
        plot_confidence_histogram(preds_np, class_names,
                                  save_path=os.path.join(run_dir, f"confidence_histogram_{split_name}.png"))
        generate_classification_report_txt(y_true, y_pred, class_names,
                                           save_path=os.path.join(run_dir, f"classification_report_{split_name}.txt"))

        summary = {
            "split": split_name,
            "num_samples": int(len(y_true)),
            "map_values": {f"conf_{k}": float(v) for k, v in map_values.items()},
            "roc_auc_per_class": {class_names[i]: float(aucs[i]) for i in range(len(class_names))},
            "roc_auc_macro": float(np.mean(aucs)),
            "pr_ap_per_class": {class_names[i]: float(aps[i]) for i in range(len(class_names))},
            "pr_map": float(np.mean(aps)),
            "ece_per_class": {class_names[i]: (float(eces[i]) if not np.isnan(eces[i]) else None)
                              for i in range(len(class_names))},
            "ece_mean": float(np.nanmean(eces)),
        }
        with open(os.path.join(run_dir, f"metrics_summary_{split_name}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Metrics] mAP@0.5={map_values[0.5]:.4f}  ROC AUC={np.mean(aucs):.4f}  PR mAP={np.mean(aps):.4f}")
    except Exception as e:
        print(f"[WARN] Advanced metrics failed: {e}")
        import traceback; traceback.print_exc()

    # --- Size vs confidence correlation ---
    try:
        from .analysis import compute_particle_area, compute_confidence
        preds_np, _, val_fps = _predict_loader(model, val_gen, device)
        conf = compute_confidence(preds_np)
        areas = compute_particle_area(val_fps, config.image_size)
        corr_matrix = np.corrcoef(areas, conf)
        np.savetxt(os.path.join(run_dir, "size_conf_corr.csv"), corr_matrix, delimiter=",")
        with open(os.path.join(run_dir, "size_conf_corr_value.txt"), "w") as f:
            f.write(f"{float(corr_matrix[0, 1])}\n")
        print(f"[Corr] size-confidence correlation = {corr_matrix[0, 1]:.4f}")
    except Exception as e:
        print(f"[WARN] Size/conf correlation failed: {e}")

    # --- Run metadata ---
    try:
        from .metadata import write_run_metadata
        write_run_metadata(run_dir, config=config, model=model, stage="post_analysis")
    except Exception as e:
        print(f"[WARN] Metadata write failed: {e}")


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def get_next_version(save_path, prefix="v"):
    if not os.path.exists(save_path):
        return f"{prefix}1.0"
    versions = []
    for name in os.listdir(save_path):
        match = re.match(rf"{prefix}(\d+)\.(\d+)", name)
        if match:
            versions.append(tuple(map(int, match.groups())))
    if not versions:
        return f"{prefix}1.0"
    major, minor = max(versions)
    return f"{prefix}{major}.{minor + 1}"


def _compute_class_weights(train_loader, num_classes: int):
    """Inverse-frequency class weights (normalised to mean 1) from the train set.
    Returns a float tensor of length num_classes, or None if counts unavailable."""
    try:
        targets = torch.tensor([s[1] for s in train_loader.dataset.samples])
        counts = torch.bincount(targets, minlength=num_classes).float()
        counts = counts.clamp(min=1.0)  # avoid div-by-zero for empty classes
        weights = counts.sum() / (num_classes * counts)  # inverse freq, mean ~1
        print(f"[Loss] class counts={counts.int().tolist()}  weights={[round(w,3) for w in weights.tolist()]}")
        return weights
    except Exception as e:
        print(f"[Loss] class weight computation failed ({e}) — using uniform weights")
        return None


def run_full_experiment(
    config: ExperimentConfig,
    balance_mode: str = "none",
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    mp_context=None,
    val_steps: int | None = None,
    augment: bool = True,
    class_weighted_loss: bool = False,
    label_smoothing: float = 0.0,
):
    classes = get_classes(config)
    balance_mode = (balance_mode or "none").strip().lower()
    if balance_mode not in {"none", "balanced"}:
        raise ValueError("balance_mode must be 'none' or 'balanced'")

    model_name = getattr(config, "model_name", "inception")

    # Create the run directory early so the pipeline-level crash log lands there
    version = get_next_version(config.save_path)
    run_dir = os.path.join(config.save_path, version)
    os.makedirs(run_dir, exist_ok=True)
    pipeline_log = os.path.join(run_dir, "crash_log.json")
    clog = CrashLogger(pipeline_log, model_name,
                       config.base_dirs.get(config.training_type, "unknown"), config)
    clog._state["status"] = "data_loading"
    clog._write()
    print(f"[CrashLog] Pipeline heartbeat: {pipeline_log}")

    try:
        print(f"[DATA] Loading datasets (balance_mode={balance_mode})...")
        train_ds, val_ds = get_tfdata_datasets(
            config,
            model_name=model_name,
            balance_mode=balance_mode,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            mp_context=mp_context,
            augment=augment,
        )
        print(f"[DATA] Datasets loaded — train={len(train_ds.dataset)}  val={len(val_ds.dataset)}")

        class_weights = _compute_class_weights(train_ds, len(classes)) if class_weighted_loss else None
        print(f"[Train] augment={augment}  class_weighted_loss={class_weighted_loss}  "
              f"label_smoothing={label_smoothing}")

        clog._state["status"] = "building_model"
        clog._write()
        model, _ = build_model(config, len(classes), model_name=model_name)

        device = _get_device()
        h, w, _ = get_model_input_size(model_name)

        # Warmup: only run in WSL2. On native Windows the CUDA driver handles PTX JIT
        # quickly (seconds not minutes), TdrDelay registry fix covers any timeout, and
        # the 486 synchronize() calls in warmup have been causing VIDEO_SCHEDULER_INTERNAL_ERROR
        # (BSOD 0x00000119) by overloading the Windows GPU scheduler.
        _is_wsl = "microsoft" in open("/proc/version").read().lower() if __import__("os").path.exists("/proc/version") else False
        if _is_wsl:
            clog._state["status"] = "warmup"
            clog._write()
            _warmup_gpu(model, device, h, w, batch_size=int(config.batch_size))
        else:
            print("[Warmup] Skipped — native Windows detected (not needed, caused BSOD 0x00000119)")

        print(f"Running experiment {version}")

        steps_per_epoch = getattr(config, "steps_per_epoch", None)
        if balance_mode == "balanced":
            if steps_per_epoch is None:
                steps_per_epoch = compute_steps_per_epoch(config)
        print(f"[Train] balance_mode={balance_mode} steps_per_epoch={steps_per_epoch}")

        # --- Phase 1: pretrain (frozen backbone) ---
        clog._state["status"] = "pretrain"
        clog._write()
        history1 = train_model(
            model, train_ds, val_ds, config,
            version=version,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            history_filename="history_pretrain.csv",
            crash_log_path=pipeline_log,
            burn_in_epochs=0,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

        # --- Phase 2: fine-tune (unfreeze top layers) ---
        print(f"[{version}] Starting fine-tuning...")

        # Drain any pending async CUDA errors from pretrain before changing the model.
        # cudaErrorUnknown is often deferred — without this sync it surfaces on the first
        # CUDA call of fine-tune (xb.to(device)), making it look like a fine-tune crash.
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        model = fine_tune_model(model, model_name=model_name)
        ft_optimizer = make_sgd_optimizer(model, learning_rate=1e-4, momentum=0.9)

        if _is_wsl:
            clog._state["status"] = "finetune_warmup"
            clog._write()
            _warmup_gpu(model, device, h, w, batch_size=int(config.batch_size))

        clog._state["status"] = "finetune"
        clog._write()
        history2 = train_model(
            model, train_ds, val_ds, config,
            version=version,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            history_filename="history_finetune.csv",
            optimizer=ft_optimizer,
            crash_log_path=pipeline_log,
            burn_in_epochs=0,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

        clog._state["status"] = "post_analysis"
        clog._write()
        test_ds = get_test_generator(config, model_name=model_name)

        run_post_analysis(
            config=config,
            model=model,
            train_gen=train_ds,
            val_gen=val_ds,
            run_dir=run_dir,
            history1=history1,
            history2=history2,
            cm_gen=test_ds,
        )

    except Exception as exc:
        clog.crash(exc)
        raise

    clog.complete()
    print(f"Experiment {version} completed!")
    return model, history1, history2


def analyze_only(config: ExperimentConfig, model_path: str, version: str | None = None):
    device = _get_device()
    model = torch.load(model_path, map_location=device, weights_only=False)

    model_name = getattr(config, "model_name", "inception")
    _, val_ds = get_tfdata_datasets(config, model_name=model_name, balance_mode="none")

    model_dir = os.path.dirname(model_path)
    run_dir = os.path.join(model_dir, version) if version else model_dir

    print(f"Running analysis-only in {run_dir}")

    test_ds = get_test_generator(config, model_name=model_name)

    run_post_analysis(
        config=config,
        model=model,
        train_gen=val_ds,
        val_gen=val_ds,
        run_dir=run_dir,
        cm_gen=test_ds,
    )
    return model, run_dir
