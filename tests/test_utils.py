"""
Tests for utility functions (metrics and visualization helpers).

Run with::

    pytest tests/test_utils.py -v
"""

import numpy as np
import pytest
import torch

from utils.metrics import AverageMeter, compute_metrics


# ---------------------------------------------------------------------------
# AverageMeter tests
# ---------------------------------------------------------------------------


class TestAverageMeter:
    def test_initial_state(self):
        m = AverageMeter("loss")
        assert m.avg == 0.0
        assert m.count == 0

    def test_single_update(self):
        m = AverageMeter()
        m.update(0.5)
        assert m.avg == pytest.approx(0.5)
        assert m.val == pytest.approx(0.5)

    def test_running_average(self):
        m = AverageMeter()
        m.update(1.0, n=1)
        m.update(3.0, n=1)
        assert m.avg == pytest.approx(2.0)

    def test_weighted_update(self):
        m = AverageMeter()
        m.update(2.0, n=4)   # contributes 8
        m.update(6.0, n=4)   # contributes 24  -> avg = 32/8 = 4.0
        assert m.avg == pytest.approx(4.0)

    def test_reset(self):
        m = AverageMeter()
        m.update(99.0, n=10)
        m.reset()
        assert m.avg == 0.0
        assert m.count == 0

    def test_repr_contains_name(self):
        m = AverageMeter("val_loss")
        m.update(0.123)
        assert "val_loss" in repr(m)


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _make_labels(self, num_classes=4, n=100):
        torch.manual_seed(7)
        y_true = torch.randint(0, num_classes, (n,))
        return y_true

    def test_perfect_predictions(self):
        y = torch.tensor([0, 1, 2, 3, 0, 1])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)

    def test_accuracy_range(self):
        y_true = self._make_labels()
        y_pred = torch.randint(0, 4, (100,))
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_zero_accuracy(self):
        """Prediction always wrong -> accuracy = 0."""
        y_true = torch.zeros(10, dtype=torch.long)
        y_pred = torch.ones(10, dtype=torch.long)
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_per_class_keys(self):
        y_true = torch.tensor([0, 1, 2, 3])
        y_pred = torch.tensor([0, 1, 2, 3])
        names = ["A", "B", "C", "D"]
        metrics = compute_metrics(y_true, y_pred, class_names=names)
        assert set(metrics["per_class"].keys()) == set(names)

    def test_per_class_without_names(self):
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 2])
        metrics = compute_metrics(y_true, y_pred)
        assert "0" in metrics["per_class"]

    def test_f1_harmonic_mean(self):
        """When precision == recall, F1 should equal precision."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        for cls_stats in metrics["per_class"].values():
            assert cls_stats["f1"] == pytest.approx(
                cls_stats["precision"], abs=1e-6
            )

    def test_macro_f1_equals_mean_per_class_f1(self):
        torch.manual_seed(42)
        y_true = torch.randint(0, 4, (200,))
        y_pred = torch.randint(0, 4, (200,))
        names = ["A", "B", "C", "D"]
        metrics = compute_metrics(y_true, y_pred, class_names=names)
        manual_mean = sum(v["f1"] for v in metrics["per_class"].values()) / 4
        assert metrics["macro_f1"] == pytest.approx(manual_mean, abs=1e-6)

    def test_precision_recall_f1_range(self):
        torch.manual_seed(1)
        y_true = torch.randint(0, 3, (150,))
        y_pred = torch.randint(0, 3, (150,))
        metrics = compute_metrics(y_true, y_pred)
        for key in ("macro_precision", "macro_recall", "macro_f1"):
            assert 0.0 <= metrics[key] <= 1.0
