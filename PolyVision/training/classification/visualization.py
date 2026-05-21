import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import seaborn as sns

def plot_training_history(history, save_path=None, show=False):
    h = history.history

    acc = h.get("accuracy", h.get("acc"))
    val_acc = h.get("val_accuracy", h.get("val_acc"))
    loss = h.get("loss")
    val_loss = h.get("val_loss")

    epochs = range(1, len(loss) + 1) if loss is not None else range(1, len(acc) + 1)

    plt.figure(figsize=(12, 12))

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

def plot_training_history_csv(csv_path, save_path=None, show=False):
    df = pd.read_csv(csv_path)

    epochs = range(1, len(df) + 1)
    acc = df.get("accuracy", df.get("acc"))
    val_acc = df.get("val_accuracy", df.get("val_acc"))
    loss = df.get("loss")
    val_loss = df.get("val_loss")

    plt.figure(figsize=(12, 12))

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


def plot_layer_activations_heatmap(all_layers_usage_json, save_path=None, top_n_layers=30):
    """
    Visualize mean activation strength across layers as a heatmap.
    Highlights strong (hot) vs weak (cold) neurons per layer.

    Args:
        all_layers_usage_json: Path to all_layers_usage.json from analyze_all_layers
        save_path: Where to save the plot
        top_n_layers: Number of layers to visualize (too many = unreadable)
    """
    import json

    with open(all_layers_usage_json, 'r') as f:
        data = json.load(f)

    # Extract mean activations per layer
    layer_names = []
    activations_matrix = []

    for layer_name, layer_data in list(data.items())[:top_n_layers]:
        layer_names.append(layer_name)
        mean_acts = np.array(layer_data['mean_activation'])

        # Normalize to 0-1 per layer (for visualization)
        if mean_acts.max() > 0:
            mean_acts = mean_acts / mean_acts.max()

        activations_matrix.append(mean_acts)

    # Pad to same length (layers may have different neuron counts)
    max_len = max(len(row) for row in activations_matrix)
    padded = np.zeros((len(activations_matrix), max_len))

    for i, row in enumerate(activations_matrix):
        padded[i, :len(row)] = row

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(
        padded,
        yticklabels=layer_names,
        cmap='RdYlGn',  # Red (weak) → Yellow → Green (strong)
        cbar_kws={'label': 'Normalized Activation Strength'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Layer Name')
    ax.set_title('Layer Activation Strength (Strong vs Weak Nodes)', fontsize=14, weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved activation heatmap: {save_path}")

    plt.close()


def plot_network_summary_bars(all_layers_usage_json, save_path=None):
    """
    Bar chart showing average activation per layer.
    Identifies which entire layers are strong vs weak.
    """
    import json

    with open(all_layers_usage_json, 'r') as f:
        data = json.load(f)

    layer_names = []
    mean_activations = []
    median_activations = []

    for layer_name, layer_data in data.items():
        summary = layer_data['summary']
        layer_names.append(layer_name)
        mean_activations.append(summary['median'])  # Use median (more robust)
        median_activations.append(summary['median'])

    # Sort by activation strength
    sorted_indices = np.argsort(mean_activations)[::-1]
    layer_names = [layer_names[i] for i in sorted_indices]
    mean_activations = [mean_activations[i] for i in sorted_indices]

    # Color code: green (strong), yellow (medium), red (weak)
    colors = []
    for val in mean_activations:
        if val > 0.5:
            colors.append('#2ecc71')  # Green
        elif val > 0.1:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    fig, ax = plt.subplots(figsize=(12, max(8, len(layer_names) * 0.3)))

    y_pos = np.arange(len(layer_names))
    ax.barh(y_pos, mean_activations, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel('Median Activation Strength', fontsize=12)
    ax.set_title('Layer Strength Ranking (Strong → Weak)', fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Strong (>0.5)'),
        Patch(facecolor='#f39c12', label='Medium (0.1-0.5)'),
        Patch(facecolor='#e74c3c', label='Weak (<0.1)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved layer summary: {save_path}")

    plt.close()

def plot_neuron_importance_per_layer(all_layers_usage_json, layer_name, save_path=None, top_n=50):
    """
    Plot individual neuron importance within a specific layer.
    Shows which neurons are strong vs weak.
    """
    import json

    with open(all_layers_usage_json, 'r') as f:
        data = json.load(f)

    if layer_name not in data:
        print(f"❌ Layer '{layer_name}' not found in analysis data")
        return

    layer_data = data[layer_name]
    mean_acts = np.array(layer_data['mean_activation'])
    top_indices = np.array(layer_data['top_indices'])

    # Sort neurons by activation
    sorted_indices = np.argsort(mean_acts)[::-1][:top_n]
    sorted_activations = mean_acts[sorted_indices]

    # Color code
    colors = plt.cm.RdYlGn(sorted_activations / sorted_activations.max())

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(range(len(sorted_activations)), sorted_activations, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Neuron Index (sorted by strength)', fontsize=12)
    ax.set_ylabel('Mean Activation', fontsize=12)
    ax.set_title(f'Neuron Importance in Layer: {layer_name} (Top {top_n})', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=sorted_activations.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Activation Strength', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved neuron importance plot: {save_path}")

    plt.close()


def plot_precision_recall_map(y_true, y_pred_probs, class_names, save_path=None, iou_thresholds=[0.5, 0.75, 0.95]):
    """
    Compute mAP (mean Average Precision) at different IoU thresholds.
    For classification, we use confidence thresholds instead.

    Args:
        y_true: True labels (shape: [n_samples])
        y_pred_probs: Prediction probabilities (shape: [n_samples, n_classes])
        class_names: List of class names
        save_path: Where to save plot
        iou_thresholds: Confidence thresholds to evaluate (0.5 = 50% confidence, etc.)
    """
    n_classes = len(class_names)

    # Compute AP per class at different confidence thresholds
    results = {thresh: [] for thresh in iou_thresholds}

    for class_idx in range(n_classes):
        # Binarize labels for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_scores = y_pred_probs[:, class_idx]

        for thresh in iou_thresholds:
            # Predictions above threshold
            y_pred_binary = (y_scores >= thresh).astype(int)

            # Compute precision and recall
            if y_pred_binary.sum() > 0:
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                # AP approximation (for true AP, use precision_recall_curve)
                ap = average_precision_score(y_true_binary, y_scores)
            else:
                precision, recall, ap = 0, 0, 0

            results[thresh].append({
                'class': class_names[class_idx],
                'precision': precision,
                'recall': recall,
                'ap': ap
            })

    # Create visualization
    fig, axes = plt.subplots(1, len(iou_thresholds), figsize=(6 * len(iou_thresholds), 8))
    if len(iou_thresholds) == 1:
        axes = [axes]

    for idx, thresh in enumerate(iou_thresholds):
        ax = axes[idx]

        df = pd.DataFrame(results[thresh])

        # Bar plot of AP per class
        y_pos = np.arange(len(class_names))
        aps = df['ap'].values

        colors = plt.cm.RdYlGn(aps / (aps.max() + 1e-6))

        bars = ax.barh(y_pos, aps, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Average Precision (AP)', fontsize=11)
        ax.set_title(f'AP @ Conf {thresh:.2f}\nmAP = {aps.mean():.3f}', fontsize=12, weight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)

        # Annotate bars with values
        for i, (bar, ap_val) in enumerate(zip(bars, aps)):
            ax.text(ap_val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{ap_val:.3f}',
                    va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved mAP plot: {save_path}")

    plt.close()

    # Return mAP values
    map_values = {thresh: np.mean([r['ap'] for r in results[thresh]]) for thresh in iou_thresholds}
    return map_values


def plot_roc_curves(y_true, y_pred_probs, class_names, save_path=None):
    """
    Plot ROC curves for each class (one-vs-rest).
    """
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute ROC curve and AUC for each class
    aucs = []

    for class_idx in range(n_classes):
        # Binarize labels
        y_true_binary = (y_true == class_idx).astype(int)
        y_scores = y_pred_probs[:, class_idx]

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Plot
        ax.plot(fpr, tpr, lw=2,
                label=f'{class_names[class_idx]} (AUC = {roc_auc:.3f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (Macro-Avg AUC = {np.mean(aucs):.3f})', fontsize=14, weight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curves: {save_path}")

    plt.close()

    return aucs


def plot_precision_recall_curves(y_true, y_pred_probs, class_names, save_path=None):
    """
    Plot Precision-Recall curves for each class.
    """
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute PR curve and AP for each class
    aps = []

    for class_idx in range(n_classes):
        # Binarize labels
        y_true_binary = (y_true == class_idx).astype(int)
        y_scores = y_pred_probs[:, class_idx]

        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        ap = average_precision_score(y_true_binary, y_scores)
        aps.append(ap)

        # Plot
        ax.plot(recall, precision, lw=2,
                label=f'{class_names[class_idx]} (AP = {ap:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curves (mAP = {np.mean(aps):.3f})', fontsize=14, weight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved PR curves: {save_path}")

    plt.close()

    return aps


def plot_calibration_curves(y_true, y_pred_probs, class_names, save_path=None, n_bins=10):
    """
    Plot calibration curves (reliability diagrams) for each class.
    Shows whether predicted probabilities match actual outcomes.
    """
    n_classes = len(class_names)

    # Create subplots grid
    n_cols = 3
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]

    for class_idx in range(n_classes):
        ax = axes[class_idx]

        # Binarize labels
        y_true_binary = (y_true == class_idx).astype(int)
        y_scores = y_pred_probs[:, class_idx]

        # Compute calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_true_binary, y_scores, n_bins=n_bins, strategy='uniform')

            # Plot calibration curve
            ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')

            # Plot perfect calibration
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

            # Compute Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))

            ax.set_xlabel('Mean Predicted Probability', fontsize=10)
            ax.set_ylabel('Fraction of Positives', fontsize=10)
            ax.set_title(f'{class_names[class_idx]}\nECE = {ece:.3f}', fontsize=11, weight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(class_names[class_idx])

    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved calibration curves: {save_path}")

    plt.close()


def plot_combined_calibration(y_true, y_pred_probs, class_names, save_path=None, n_bins=10):
    """
    Plot all calibration curves on a single plot for comparison.
    """
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    eces = []

    for class_idx in range(n_classes):
        # Binarize labels
        y_true_binary = (y_true == class_idx).astype(int)
        y_scores = y_pred_probs[:, class_idx]

        # Compute calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_true_binary, y_scores, n_bins=n_bins, strategy='uniform')

            # Compute ECE
            ece = np.mean(np.abs(prob_true - prob_pred))
            eces.append(ece)

            # Plot
            ax.plot(prob_pred, prob_true, marker='o', linewidth=2,
                    label=f'{class_names[class_idx]} (ECE={ece:.3f})')
        except:
            eces.append(np.nan)

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration Curves (Mean ECE = {np.nanmean(eces):.3f})', fontsize=14, weight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved combined calibration: {save_path}")

    plt.close()

    return eces


def plot_confidence_histogram(y_pred_probs, class_names, save_path=None):
    """
    Plot histogram of prediction confidences.
    Shows distribution of max probabilities across predictions.
    """
    max_probs = np.max(y_pred_probs, axis=1)
    pred_classes = np.argmax(y_pred_probs, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall confidence distribution
    ax = axes[0]
    ax.hist(max_probs, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {max_probs.mean():.3f}')
    ax.axvline(np.median(max_probs), color='orange', linestyle='--', linewidth=2,
               label=f'Median = {np.median(max_probs):.3f}')
    ax.set_xlabel('Max Predicted Probability (Confidence)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Confidence Distribution', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Per-class confidence
    ax = axes[1]
    for class_idx, class_name in enumerate(class_names):
        mask = pred_classes == class_idx
        if mask.sum() > 0:
            class_confidences = max_probs[mask]
            ax.hist(class_confidences, bins=30, alpha=0.5, label=f'{class_name} (n={mask.sum()})', edgecolor='black')

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution by Predicted Class', fontsize=13, weight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confidence histogram: {save_path}")

    plt.close()


def generate_classification_report_txt(y_true, y_pred, class_names, save_path=None):
    """
    Generate and save detailed classification report.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    if save_path:
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            f.write("\n\n")

            # Add confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            f.write("=" * 60 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("=" * 60 + "\n\n")

            # Header
            f.write("True\\Pred".ljust(12))
            for name in class_names:
                f.write(name[:8].ljust(10))
            f.write("\n" + "-" * 60 + "\n")

            # Rows
            for i, name in enumerate(class_names):
                f.write(name[:10].ljust(12))
                for j in range(len(class_names)):
                    f.write(str(cm[i, j]).ljust(10))
                f.write("\n")

        print(f"✅ Saved classification report: {save_path}")

    return report
