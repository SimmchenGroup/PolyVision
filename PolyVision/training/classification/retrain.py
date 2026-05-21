import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

def _write_history_csv(history: tf.keras.callbacks.History, out_csv_path: str | Path) -> None:
    """
    Save per-epoch Keras History to CSV without extra dependencies.
    """
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    h = getattr(history, "history", None) or {}
    keys = list(h.keys())

    # Determine number of epochs recorded
    n = 0
    for v in h.values():
        try:
            n = max(n, len(v))
        except Exception:
            pass

    import csv
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", *keys])
        for i in range(n):
            row = [i + 1]
            for k in keys:
                vals = h.get(k, [])
                row.append(vals[i] if i < len(vals) else "")
            w.writerow(row)

def _set_trainable_last_n_layers(model: tf.keras.Model, unfreeze_last_n: int, *, keep_batchnorm_frozen: bool = True):
    """
    Unfreeze the last N layers of the loaded model. Optionally keep BatchNorm frozen
    (common best practice for fine-tuning).
    """
    n = max(0, int(unfreeze_last_n))
    if n <= 0:
        return

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-n:]:
        if keep_batchnorm_frozen and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


class BestEpochTracker(Callback):
    """
    Tracks the best epoch according to `monitor` and writes it to best_epoch.json.
    """
    def __init__(self, out_json_path: str | Path, monitor: str = "val_accuracy", mode: str = "max"):
        super().__init__()
        self.out_json_path = Path(out_json_path)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.best_epoch = None
        self.best_value = None

        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor, None)
        if val is None:
            return

        val = float(val)
        improved = False

        if self.best_value is None:
            improved = True
        elif self.mode == "max" and val > self.best_value:
            improved = True
        elif self.mode == "min" and val < self.best_value:
            improved = True

        if improved:
            self.best_value = val
            self.best_epoch = int(epoch)  # 0-based

            self.out_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_json_path.write_text(
                json.dumps(
                    {
                        "monitor": self.monitor,
                        "mode": self.mode,
                        "best_epoch_0_based": self.best_epoch,
                        "best_epoch_1_based": self.best_epoch + 1,
                        "best_value": self.best_value,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


def load_and_retrain(
    model_path,
    train_generator,
    val_generator,
    epochs=25,
    learning_rate=1e-6,
    momentum=0.9,
    steps_per_epoch=None,
    *,
    fine_tune: bool = False,
    fine_tune_epochs: int = 5,
    fine_tune_learning_rate: float = 1e-7,
    unfreeze_last_n: int = 30,
    checkpoint_dir: str | Path | None = None,
    monitor: str = "val_accuracy",
    mode: str = "max",
):
    model = load_model(model_path)

    callbacks = []
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_dir / "best_model.keras"),
                save_best_only=True,
                monitor=monitor,
                mode=mode,
                verbose=1,
            )
        )

        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_dir / "epoch_{epoch:04d}.keras"),
                save_best_only=False,
                verbose=0,
            )
        )

        callbacks.append(
            BestEpochTracker(
                out_json_path=checkpoint_dir / "best_epoch.json",
                monitor=monitor,
                mode=mode,
            )
        )

    # Phase 1: retrain
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    if checkpoint_dir:
        _write_history_csv(history1, checkpoint_dir / "retrain_history.csv")

    if not fine_tune:
        return model, history1

    # Phase 2: fine-tune
    _set_trainable_last_n_layers(model, unfreeze_last_n, keep_batchnorm_frozen=True)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=SGD(learning_rate=float(fine_tune_learning_rate), momentum=float(momentum)),
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_generator,
        epochs=int(fine_tune_epochs),
        validation_data=val_generator,
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    if checkpoint_dir:
        _write_history_csv(history2, checkpoint_dir / "finetune_history.csv")

    return model, {"retrain": history1, "fine_tune": history2}