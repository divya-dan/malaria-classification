import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import SEED, DEVICE, FIG_DIR, AUGMENT



def set_seed(seed: int = SEED):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5):
    """
    Compute classification metrics for binary predictions.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_probs: Predicted probabilities (sigmoid outputs)
        threshold: Decision threshold for converting probs to class labels

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc
    """
    # Convert probabilities to binary predictions
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_probs)
    }
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, normalize: bool = True, save_fig: str = None):
    """
    Plot and optionally save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()

    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig)
    plt.close()


def plot_training_curves(history: dict, metrics: list = ['loss', 'accuracy'], save_dir: str = FIG_DIR):
    """
    Plot training and validation curves stored in history dict.

    history format expects:
    {
      'train_loss': [...], 'val_loss': [...],
      'train_accuracy': [...], 'val_accuracy': [...], etc.
    }
    """
    epochs = range(1, len(history['train_' + metrics[0]]) + 1)
    for metric in metrics:
        plt.figure()
        plt.plot(epochs, history[f'train_{metric}'], label=f'Train {metric.title()}')
        plt.plot(epochs, history[f'val_{metric}'], label=f'Val {metric.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.title(f'Train vs Val {metric.title()}')
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(save_dir, f'{metric}_curve{"_aug" if AUGMENT else ""}.png')
        plt.savefig(fig_path)
        plt.close()
