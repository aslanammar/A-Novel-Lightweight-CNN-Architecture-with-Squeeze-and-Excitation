"""
Visualization helpers for training curves, confusion matrix, and sample
predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and accuracy curves recorded during training.

    Args:
        history: Dictionary with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``, each mapping to a list of epoch values.
        save_path: If provided, saves the figure to this path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Val", marker="s", markersize=4)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", marker="o", markersize=4)
    ax2.plot(epochs, history["val_acc"], label="Val", marker="s", markersize=4)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("MaizeSENet – Training Curves", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Render a normalized confusion matrix as a heatmap.

    Args:
        cm: Square integer array of shape ``(num_classes, num_classes)``.
        class_names: List of class label strings.
        save_path: If provided, saves the figure to this path.
        title: Figure title.
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(max(6, num_classes * 1.5), max(5, num_classes * 1.3)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_predictions(
    images: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: List[str],
    mean: List[float] = None,
    std: List[float] = None,
    num_samples: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """Plot a grid of sample images with true and predicted labels.

    Args:
        images: Batch of normalised image tensors ``(B, C, H, W)``.
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: List of class name strings.
        mean: Per-channel mean for de-normalisation.
        std: Per-channel std for de-normalisation.
        num_samples: Number of samples to display (≤ batch size).
        save_path: If provided, saves the figure to this path.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    n = min(num_samples, images.size(0))
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for idx in range(n):
        img = images[idx].cpu() * std_t + mean_t
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        true_name = class_names[y_true[idx].item()]
        pred_name = class_names[y_pred[idx].item()]
        correct = y_true[idx].item() == y_pred[idx].item()

        axes[idx].imshow(img)
        axes[idx].set_title(
            f"T: {true_name}\nP: {pred_name}",
            fontsize=7,
            color="green" if correct else "red",
        )
        axes[idx].axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
