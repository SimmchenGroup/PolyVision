import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model(model, train_gen, val_gen, config, version=None, steps_per_epoch=None, history_filename: str = "history.csv"):
    """
    Train model with automatic checkpoint and early stopping.
    version: optional string like "v1.19" to save outputs in structured folders

    NOTE:
      - If train_gen is a balanced tf.data pipeline that repeats forever,
        you MUST pass steps_per_epoch.
    """
    # Structured save path
    save_path = config.save_path
    if version:
        save_path = os.path.join(save_path, version)
    os.makedirs(save_path, exist_ok=True)

    # --- Compute class weights for imbalanced data ---
    # (Keep disabled if you want to remove class weights)
    # class_weights = None
    # if hasattr(train_gen, 'classes') and train_gen.classes is not None:
    #     unique_classes = np.unique(train_gen.classes)
    #     weights = compute_class_weight(
    #         class_weight='balanced',
    #         classes=unique_classes,
    #         y=train_gen.classes
    #     )
    #     class_weights = dict(zip(unique_classes, weights))
    #     print(f"[Train] Using class weights: {class_weights}")

    # --- Callbacks ---
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_path, "best_model.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=5,   # stop if no improvement after 5 epochs
        restore_best_weights=True,
        verbose=1
    )

    # --- Train ---
    history = model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=[checkpoint_cb, earlystop_cb],
        steps_per_epoch=steps_per_epoch,
        # class_weight=class_weights
    )

    # --- Save training history (per-epoch) ---
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))

    out_csv = os.path.join(save_path, history_filename or "history.csv")
    hist_df.to_csv(out_csv, index=False)
    print(f"[Train] Wrote training history CSV: {out_csv}")

    return history