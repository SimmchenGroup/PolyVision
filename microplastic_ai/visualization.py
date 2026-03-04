import matplotlib.pyplot as plt
import pandas as pd

def plot_training_history(history, save_path=None, show=False):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel("Epoch")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close()

def plot_training_history_csv(csv_path, save_path=None, show=False):
    """
    Plot training curves from a history.csv saved by train.py
    """
    df = pd.read_csv(csv_path)

    epochs = range(1, len(df) + 1)
    acc = df.get("acc")
    val_acc = df.get("val_acc")
    loss = df.get("loss")
    val_loss = df.get("val_loss")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if acc is not None:
        plt.plot(epochs, acc, label="Training accuracy")
    if val_acc is not None:
        plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    if loss is not None:
        plt.plot(epochs, loss, label="Training loss")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Validation loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close()