"""
Evaluation metrics for multi-class classification.

Provides:
  - ``compute_metrics``: accuracy, precision, recall, F1 (macro & per-class)
  - ``AverageMeter``: running mean tracker used during training loops
"""

from typing import Dict, List

import numpy as np
import torch


class AverageMeter:
    """Computes and stores a running (arithmetic) mean and current value."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: List[str] = None,
) -> Dict:
    """Compute classification metrics from predicted and ground-truth labels.

    Args:
        y_true: 1-D integer tensor of ground-truth class indices.
        y_pred: 1-D integer tensor of predicted class indices.
        class_names: Optional list of class name strings.

    Returns:
        Dictionary containing:
          - ``accuracy`` (float)
          - ``macro_precision`` (float)
          - ``macro_recall`` (float)
          - ``macro_f1`` (float)
          - ``per_class`` (dict): per-class precision, recall, F1
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    num_classes = int(y_true_np.max()) + 1

    accuracy = float((y_pred_np == y_true_np).mean())

    per_class: Dict[str, Dict[str, float]] = {}
    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = int(((y_pred_np == c) & (y_true_np == c)).sum())
        fp = int(((y_pred_np == c) & (y_true_np != c)).sum())
        fn = int(((y_pred_np != c) & (y_true_np == c)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        name = class_names[c] if class_names and c < len(class_names) else str(c)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
    }
