# retrain_model.py
from pathlib import Path
import argparse

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # avoids oneDNN/MKL conv path on CPU

from training.classification.config import ExperimentConfig
from training.classification.data import get_generators
from training.classification.retrain import load_and_retrain
from training.classification.pipeline import run_full_experiment, analyze_only
import tensorflow as tf



def build_config(model_name: str = "inception", weights_path: str | None = None) -> ExperimentConfig:
    # IMPORTANT: use the same training_type/classes/image_size as the model was trained with
    return ExperimentConfig(
        training_type="microplastic",
        learning_rate=0.00001,  # used by training pipeline
        base_dirs={
            "microplastic": r"C:\Users\joshk\OneDrive\Desktop\multiclass\localv6",
            "whisky": r"C:\path\to\other_dataset",
        },
        microplastic_classes=["nylon", "pe", "pet", "pla", "pmma", "ps", "pp", "pu", "pvc"],  # etc
        whisky_classes=[],
        weights_path=weights_path or r"training/classification/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
        save_path="results\classification",
        epochs=15,
        batch_size=16,
        model_name=(model_name or "inception").strip().lower(),
        steps_per_epoch=None,
    )


def main():
    parser = argparse.ArgumentParser(description="Run / retrain / analyze microplastic experiments.")
    parser.add_argument(
        "--mode",
        choices=["train", "retrain", "analyze"],
        default="retrain",
        help="train: run full pipeline from scratch; retrain: load saved .keras and continue; analyze: post-analysis only.",
    )
    parser.add_argument(
        "--model",
        choices=["inception", "efficient", "res"],
        default="inception",
        help="Which backbone to use for TRAIN mode (inception/efficient/res).",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional path to backbone weights (used by build_model).",
    )
    parser.add_argument(
        "--model-path",
        default=r"classification/results\v1.2\best_model.keras",
        help="Path to an existing saved .keras model (used by retrain/analyze).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Retrain epochs (retrain mode).")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Retrain learning rate (retrain mode).")
    parser.add_argument("--momentum", type=float, default=0.9, help="Retrain momentum (retrain mode).")
    parser.add_argument(
        "--out-path",
        default=str(Path("classification/results") / "retrained_model.keras"),
        help="Where to save the retrained model (retrain mode).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional analysis subfolder name (analyze mode). Example: --version analysis_run_01",
    )
    # NEW: tf.data balancing controls
    parser.add_argument(
        "--balance-mode",
        choices=["none", "balanced"],
        default="none",
        help="tf.data sampling: 'none' keeps natural imbalance; 'balanced' oversamples classes uniformly (requires steps_per_epoch).",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optional override. If omitted and --balance-mode balanced, it will be computed from train/ and batch_size.",
    )
    parser.add_argument(
        "--train-base-dir",
        default=None,
        help="If set, training reads from <train-base-dir>/train while val/test still read from base_dirs.",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="If set, run a second phase after retrain that unfreezes deeper layers and fine-tunes.",
    )
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Fine-tune epochs (retrain mode).")
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-7,
        help="Fine-tune learning rate (retrain mode). Usually smaller than retrain LR.",
    )
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=30,
        help="How many layers from the end to unfreeze during fine-tune (BatchNorm stays frozen).",
    )

    args = parser.parse_args()

    config = build_config(model_name=args.model, weights_path=args.weights_path)
    config.steps_per_epoch = args.steps_per_epoch

    # Apply training-root override (balanced dataset root)
    config.train_base_dir_override = args.train_base_dir
    if config.train_base_dir_override:
        print(f"[DATA] train_base_dir_override={config.train_base_dir_override}")

    if args.mode == "train":
        print(f"[MODE=train] model={config.model_name} balance_mode={args.balance_mode}")
        run_full_experiment(config, balance_mode=args.balance_mode)
        return

    if args.mode == "train":
        print(f"[MODE=train] model={config.model_name} balance_mode={args.balance_mode}")
        run_full_experiment(config, balance_mode=args.balance_mode)
        return

    if args.mode == "analyze":
        print(f"[MODE=analyze] model_path={args.model_path}")
        analyze_only(config=config, model_path=args.model_path, version=args.version)
        return

    # default: retrain
    from training.classification.data import get_tfdata_datasets, compute_steps_per_epoch
    if args.balance_mode == "balanced" and config.steps_per_epoch is None:
        config.steps_per_epoch = compute_steps_per_epoch(config)
        print(f"[retrain] computed steps_per_epoch={config.steps_per_epoch}")

    # Auto-detect expected input size from the saved model to avoid 224/299 mismatches
    try:
        _tmp_model = tf.keras.models.load_model(str(args.model_path))
        ishape = getattr(_tmp_model, "input_shape", None)
        h = ishape[1] if ishape and len(ishape) >= 3 else None
        w = ishape[2] if ishape and len(ishape) >= 3 else None

        detected = None
        if (h, w) == (299, 299):
            detected = "inception"
        elif (h, w) == (380, 380):
            detected = "efficientb4"
        elif (h, w) == (224, 224):
            # could be efficient OR res; both use 224 in your loader.
            # pick one consistent default for preprocessing/size:
            detected = "efficient"

        if detected:
            config.model_name = detected
            print(f"[retrain] detected model input={(h, w)} -> using config.model_name={config.model_name}")
        else:
            print(
                f"[retrain] could not detect model_name from input_shape={ishape}; using config.model_name={config.model_name}")
    except Exception as e:
        print(
            f"[retrain] model_name auto-detect failed: {type(e).__name__}: {e}. Using config.model_name={config.model_name}")

    if args.balance_mode == "balanced" and config.steps_per_epoch is None:
        config.steps_per_epoch = compute_steps_per_epoch(config)
        print(f"[retrain] computed steps_per_epoch={config.steps_per_epoch}")

    train_ds, val_ds = get_tfdata_datasets(
        config,
        model_name=getattr(config, "model_name", "inception"),
        balance_mode=args.balance_mode,
    )

    model, history = load_and_retrain(
        model_path=args.model_path,
        train_generator=train_ds,
        val_generator=val_ds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        steps_per_epoch=config.steps_per_epoch if args.balance_mode == "balanced" else None,
        fine_tune=bool(args.fine_tune),
        fine_tune_epochs=int(args.fine_tune_epochs),
        fine_tune_learning_rate=float(args.fine_tune_learning_rate),
        unfreeze_last_n=int(args.unfreeze_last_n),
        checkpoint_dir=str(Path(args.out_path).parent / "retrain_checkpoints"),
        monitor="val_accuracy",
        mode="max",
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved retrained model to: {out_path}")

if __name__ == "__main__":
    main()


# ------COMMANDS------
# FULL TRAIN PIPELINE (build backbone based on --model)
# python run_experiment.py --mode train --model inception
# python -m scripts.run_experiment --mode train --model efficient --balance-mode none # or balanced --steps-per-epoch 300
# python -m scripts.run_experiment --mode train --model efficient --balance-mode none --train-base-dir "C:\Users\joshk\OneDrive\Desktop\multiclass\globalv6_bal"
# python run_experiment.py --mode train --model efficientb4
# python run_experiment.py --mode train --model res
# python run_experiment.py --mode train --model res --weights-path C:\path\to\resnet_weights.h5

# RETRAINING (continues from an existing saved .keras)
# python -m scripts.run_experiment --mode retrain --model-path "classification/results/v1.25/best_model.keras"

# POST-ANALYSIS ONLY
# python run_experiment.py --mode analyze --model-path "classification/results/ResNet50/globalv6/best_model.keras"
# python run_experiment.py --mode analyze --model-path "classification/results/v1.25/best_model.keras" --version "analysis_run_01"
# python -m classification.pipeline --mode analyze --model-path "classification/results/v1.3/best_model.keras" --config microplastic --version metrics_full
# python run_experiment.py --mode analyze --model res --model-path "training/classification/results/ResNet50/globalv3/best_model.keras"
# python -m scripts.run_experiment --mode analyze --model efficient --model-path "results/classification/v1.39/best_model.keras"

# python -m scripts.run_experiment --mode retrain --model efficient --model-path "results/classification/v1.39/best_model.keras" --epochs 10 --learning-rate 1e-5 --fine-tune --fine-tune-epochs 10 --fine-tune-learning-rate 1e-6 --unfreeze-last-n 30
