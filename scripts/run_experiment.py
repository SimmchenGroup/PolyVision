from pathlib import Path
import argparse

import torch

from training.classification.config import ExperimentConfig
from training.classification.retrain import load_and_retrain
from training.classification.pipeline import run_full_experiment, analyze_only
from training.classification.data import get_tfdata_datasets, compute_steps_per_epoch


def _setup_gpu():
    if torch.cuda.is_available():
        print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = False
        # cuDNN re-enabled: backbone is now fully frozen (all requires_grad=False) so
        # cuDNN consistently picks inference-mode algorithms. Training-mode algorithm
        # crashes (CUDNN_STATUS_EXECUTION_FAILED_CUDART) only trigger when weights have
        # requires_grad=True. cuBLAS fallback (cudnn=False) also crashes on sm_120a.
    else:
        print("[GPU] No CUDA device found — running on CPU")


def build_config(model_name: str = "inception", weights_path: str | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        training_type="microplastic",
        learning_rate=0.00001,
        base_dirs={
            "microplastic": "data/datasets/global",  # point at your crop/whole-image dataset root
            "whisky": r"path/to/other_dataset",
        },
        microplastic_classes=["nylon", "pe", "pet", "pmma", "ps", "pp", "pu", "pvc"], # , "pet", "pmma", "ps", "pp", "pu", "pvc"
        whisky_classes=[],
        weights_path=weights_path or "",  # not used in PyTorch path (torchvision pretrained)
        save_path="results/classification",
        epochs=15,
        batch_size=16,
        model_name=(model_name or "inception").strip().lower(),
        steps_per_epoch=None,
    )


def main():
    _setup_gpu()

    parser = argparse.ArgumentParser(description="Run / retrain / analyze microplastic experiments.")
    parser.add_argument("--mode", choices=["train", "retrain", "analyze"], default="retrain")
    parser.add_argument("--model", choices=["inception", "efficient", "res"], default="inception",
                        help="Backbone for train mode.")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--model-path", default=r"classification/results/v1.2/best_model.pt",
                        help="Path to a saved .pt model (retrain/analyze).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--out-path", default=str(Path("classification/results") / "retrained_model.pt"))
    parser.add_argument("--version", default=None,
                        help="Analysis subfolder name (analyze mode).")
    parser.add_argument("--balance-mode", choices=["none", "balanced"], default="none")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--train-base-dir", default=None)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-7)
    parser.add_argument("--unfreeze-last-n", type=int, default=30)

    # Speed-up flags — all default to the known-stable baseline (workers=0,
    # no pin_memory, no persistent_workers). Enable one at a time to isolate
    # which are safe on Blackwell/WSL2.
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes. 0=safe baseline. Test: 2")
    parser.add_argument("--pin-memory", action="store_true", default=False,
                        help="Pin CPU tensors to page-locked memory. Test after workers stable.")
    parser.add_argument("--persistent-workers", action="store_true", default=False,
                        help="Keep worker processes alive between epochs. Requires --num-workers > 0.")
    parser.add_argument("--mp-context", default=None, choices=[None, "fork", "spawn", "forkserver"],
                        help="Multiprocessing context for DataLoader workers. "
                             "spawn avoids CUDA fork crash on WSL2. Default: OS default (fork on Linux).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config batch size. Default: 8 (stable baseline). Test: 16, 32.")
    parser.add_argument("--dataset-path", default=None,
                        help="Override dataset root (must contain train/ val/ test/ subfolders). "
                             "e.g. data/datasets/local")
    parser.add_argument("--val-steps", type=int, default=None,
                        help="Cap validation batches per epoch. Useful for large datasets where "
                             "full val takes longer than training. e.g. 500")

    args = parser.parse_args()

    config = build_config(model_name=args.model, weights_path=args.weights_path)
    config.steps_per_epoch = args.steps_per_epoch
    config.train_base_dir_override = args.train_base_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"[CONFIG] batch_size overridden to {config.batch_size}")
    if args.dataset_path is not None:
        config.base_dirs[config.training_type] = args.dataset_path
        print(f"[CONFIG] dataset_path overridden to {args.dataset_path}")

    if config.train_base_dir_override:
        print(f"[DATA] train_base_dir_override={config.train_base_dir_override}")

    if args.mode == "train":
        print(f"[MODE=train] model={config.model_name} balance_mode={args.balance_mode}")
        run_full_experiment(
            config,
            balance_mode=args.balance_mode,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            mp_context=args.mp_context,
            val_steps=args.val_steps,
        )
        return

    if args.mode == "analyze":
        print(f"[MODE=analyze] model_path={args.model_path}")
        analyze_only(config=config, model_path=args.model_path, version=args.version)
        return

    # --- retrain ---
    if args.balance_mode == "balanced" and config.steps_per_epoch is None:
        config.steps_per_epoch = compute_steps_per_epoch(config)
        print(f"[retrain] computed steps_per_epoch={config.steps_per_epoch}")

    # Auto-detect model_name from saved model input size
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tmp_model = torch.load(args.model_path, map_location=device, weights_only=False)
        # Detect from backbone type name or a dummy forward pass shape
        backbone_cls = type(getattr(tmp_model, "backbone", tmp_model)).__name__.lower()
        if "inception" in backbone_cls:
            config.model_name = "inception"
        elif "efficientnet_b4" in backbone_cls or "efficientnetb4" in backbone_cls:
            config.model_name = "efficientb4"
        elif "efficientnet" in backbone_cls:
            config.model_name = "efficient"
        elif "resnet" in backbone_cls:
            config.model_name = "res"
        print(f"[retrain] detected model: {config.model_name}")
        del tmp_model
    except Exception as e:
        print(f"[retrain] model_name auto-detect failed: {e}. Using config.model_name={config.model_name}")

    print(f"[DATA] Loading datasets (balance_mode={args.balance_mode})...")
    train_ds, val_ds = get_tfdata_datasets(
        config,
        model_name=config.model_name,
        balance_mode=args.balance_mode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        mp_context=args.mp_context,
    )
    print(f"[DATA] Datasets loaded — train={len(train_ds.dataset)}  val={len(val_ds.dataset)}")

    print(f"[Retrain] Starting — epochs={args.epochs}  lr={args.learning_rate}  fine_tune={args.fine_tune}")
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
    torch.save(model, out_path)
    print(f"Saved retrained model to: {out_path}")


if __name__ == "__main__":
    main()


# ------COMMANDS------
# cd path/to/PolyVision
# python -m scripts.run_experiment --mode train --model inception --balance-mode none
# python -m scripts.run_experiment --mode train --model efficient --balance-mode none
# python -m scripts.run_experiment --mode train --model res --balance-mode none
# python -m scripts.run_experiment --mode retrain --model-path "results/classification/v1.x/best_model.pt"
# python -m scripts.run_experiment --mode analyze --model inception --model-path "results/classification/v1.x/best_model.pt"# When it crashes and reboots, the log will be at: results / classification / v1.XX / crash_log.json
#
# After reboot, read it with:
#     cat results / classification / v1. * / crash_log.json
# nvidia-smi --query-gpu=driver_version --format=csv,noheader

# The thread's main fix — forces the link to 2.5 GT/s which is far more stable over
#   longer/lower-quality cables. You'd set this in NVIDIA Control Panel → Manage 3D Settings → PCIe
#   Express Maximum Performance → Gen 1. It will reduce bandwidth but GPU compute kernels don't need
#   much PCIe bandwidth — only data transfers do