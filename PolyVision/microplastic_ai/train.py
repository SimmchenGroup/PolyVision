import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, train_gen, val_gen, config, version=None):
    """
    Train model with automatic checkpoint and early stopping.
    version: optional string like "v1.19" to save outputs in structured folders
    """
    # Structured save path
    save_path = config.save_path
    if version:
        save_path = os.path.join(save_path, version)
    os.makedirs(save_path, exist_ok=True)

    # --- Callbacks ---
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_path, "best_model.keras"),
        save_best_only=True,
        monitor='val_acc',
        mode='max',
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor='val_acc',
        patience=5,   # stop if no improvement after 5 epochs
        restore_best_weights=True,
        verbose=1
    )

    # --- Train ---
    history = model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        verbose=2,
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    # --- Save training history ---
    pd.DataFrame(history.history).to_csv(
        os.path.join(save_path, "history.csv"),
        index=False
    )

    return history