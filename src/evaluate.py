"""
Evaluation module for the DL-IDS project.
Computes classification metrics, generates confusion matrices,
and produces visualization plots.
"""

import os
import json
import time
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from src.utils import get_project_root, load_config, format_duration

logger = logging.getLogger("DL-IDS.evaluate")

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def evaluate_model(model, X_test, y_test, label_names=None, device=None, batch_size=256):
    """
    Evaluate a PyTorch model on test data.
    
    Args:
        model: Trained PyTorch model.
        X_test: Test features (numpy array).
        y_test: Test labels (numpy array of encoded integers).
        label_names: List of class names for reporting.
        device: torch device.
        batch_size: Batch size for inference.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    
    # Create dataloader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    inference_time = time.time() - start_time
    
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    
    # Overall metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "total_samples": int(len(y_true)),
        "inference_time_seconds": float(inference_time),
        "avg_inference_latency_ms": float(inference_time / len(y_true) * 1000),
    }
    
    # Per-class metrics
    if label_names is None:
        label_names = [str(i) for i in range(len(np.unique(y_true)))]
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report as dict
    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    
    # Per-class accuracy (TPR) and FPR from confusion matrix
    per_class = {}
    for i, name in enumerate(label_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        per_class[name] = {
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0,
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),  # Same as TPR
            "f1_score": float(report[name]["f1-score"]),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            "support": int(report[name]["support"]),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = cm.tolist()
    
    # Overall FPR (weighted average)
    total_fp = sum(pc["fp"] for pc in per_class.values())
    total_tn = sum(pc["tn"] for pc in per_class.values())
    metrics["fpr_overall"] = float(total_fp / (total_fp + total_tn)) if (total_fp + total_tn) > 0 else 0
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"  FPR (overall): {metrics['fpr_overall']:.4f}")
    logger.info(f"  Inference time: {format_duration(inference_time)} ({metrics['avg_inference_latency_ms']:.3f} ms/sample)")
    
    return metrics


def evaluate_sklearn_model(model, X_test, y_test, label_names=None):
    """
    Evaluate a scikit-learn model.
    
    Args:
        model: Trained sklearn model.
        X_test: Test features.
        y_test: Test labels.
        label_names: List of class names.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    if label_names is None:
        label_names = [str(i) for i in range(len(np.unique(y_test)))]
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "total_samples": int(len(y_test)),
        "inference_time_seconds": float(inference_time),
        "avg_inference_latency_ms": float(inference_time / len(y_test) * 1000),
        "confusion_matrix": cm.tolist(),
    }
    
    # Per-class
    per_class = {}
    for i, name in enumerate(label_names):
        if name in report:
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            per_class[name] = {
                "accuracy": float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0,
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1_score": float(report[name]["f1-score"]),
                "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                "support": int(report[name]["support"]),
            }
    metrics["per_class"] = per_class
    
    return metrics


def plot_confusion_matrix(cm, label_names, save_path=None, title="Confusion Matrix", normalize=False):
    """Plot confusion matrix heatmap."""
    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title += " (Normalized)"
    else:
        cm_plot = cm
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(max(10, len(label_names)), max(8, len(label_names) * 0.8)))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None, title="Training History"):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history["val_loss"], 'r-o', label='Val Loss', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history["train_acc"], 'b-o', label='Train Acc', markersize=4)
    axes[1].plot(epochs, history["val_acc"], 'r-o', label='Val Acc', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    if "val_f1" in history:
        axes[2].plot(epochs, history["val_f1"], 'g-o', label='Val F1 (weighted)', markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1 Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_per_class_metrics(metrics, save_path=None, title="Per-Class Performance"):
    """Plot per-class precision, recall, and F1-score."""
    per_class = metrics["per_class"]
    labels = list(per_class.keys())
    precision = [per_class[l]["precision"] for l in labels]
    recall = [per_class[l]["recall"] for l in labels]
    f1 = [per_class[l]["f1_score"] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800', alpha=0.8)
    
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to {save_path}")
    
    plt.close()


def plot_class_distribution(label_dist, save_path=None, title="Class Distribution"):
    """Plot class distribution bar chart."""
    labels = list(label_dist.keys())
    counts = list(label_dist.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Regular scale
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = axes[0].bar(labels, counts, color=colors, edgecolor='gray', alpha=0.8)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{title} (Linear Scale)')
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # Log scale
    bars = axes[1].bar(labels, counts, color=colors, edgecolor='gray', alpha=0.8)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count (log scale)')
    axes[1].set_title(f'{title} (Log Scale)')
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.close()


def plot_model_comparison(all_metrics, save_path=None):
    """Plot comparison of multiple models."""
    models = list(all_metrics.keys())
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    display_names = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "F1\n(weighted)"]
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i], alpha=0.85)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.close()


def generate_all_plots(metrics, history, label_names, label_dist, plots_dir=None):
    """Generate all evaluation plots."""
    if plots_dir is None:
        root = get_project_root()
        config = load_config()
        plots_dir = os.path.join(root, config["evaluation"]["plots_dir"])
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, label_names,
                         save_path=os.path.join(plots_dir, "confusion_matrix.png"),
                         title="DL-IDS Confusion Matrix")
    plot_confusion_matrix(cm, label_names,
                         save_path=os.path.join(plots_dir, "confusion_matrix_normalized.png"),
                         title="DL-IDS Confusion Matrix", normalize=True)
    
    # Training history
    if history:
        plot_training_history(history,
                            save_path=os.path.join(plots_dir, "training_history.png"),
                            title="DL-IDS Training History")
    
    # Per-class metrics
    plot_per_class_metrics(metrics,
                          save_path=os.path.join(plots_dir, "per_class_metrics.png"),
                          title="DL-IDS Per-Class Performance")
    
    # Class distribution
    if label_dist:
        plot_class_distribution(label_dist,
                              save_path=os.path.join(plots_dir, "class_distribution.png"),
                              title="Dataset Class Distribution")
    
    logger.info(f"All plots saved to {plots_dir}")


def save_metrics(metrics, filepath):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {filepath}")
