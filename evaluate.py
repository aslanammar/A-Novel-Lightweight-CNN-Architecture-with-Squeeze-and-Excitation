"""
Evaluation script for MaizeSENet.

Loads a trained checkpoint and evaluates on the test split (or val split if
no test split is available), reporting accuracy, precision, recall, F1, and
a confusion matrix.

Usage::

    python evaluate.py --data_dir /path/to/dataset \\
                       --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import get_dataloaders, LABEL_NAMES
from model import build_model
from utils import compute_metrics, plot_confusion_matrix, plot_sample_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MaizeSENet on test/val data"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Set to 0 for deterministic evaluation")
    parser.add_argument("--se_reduction", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, device, class_names):
    model.eval()
    all_true, all_pred = [], []
    all_images = []

    collected = 0
    for images, labels in loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        preds = outputs.argmax(dim=1).cpu()
        all_true.append(labels)
        all_pred.append(preds)
        if collected < 8:
            take = min(images.size(0), 8 - collected)
            all_images.append(images[:take])
            collected += take

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    sample_images = torch.cat(all_images, dim=0)[:8]

    metrics = compute_metrics(all_true, all_pred, class_names)

    # Build confusion matrix
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_true.numpy(), all_pred.numpy()):
        cm[t, p] += 1

    return metrics, cm, sample_images, all_true, all_pred


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    class_names = [LABEL_NAMES[i] for i in range(args.num_classes)]

    # Data: prefer test split, fall back to val
    _, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampling=False,
        pin_memory=device.type == "cuda",
    )
    loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "val"
    print(f"Evaluating on '{split_name}' split …")

    # Model
    model = build_model(
        num_classes=args.num_classes,
        dropout=args.dropout,
        se_reduction=args.se_reduction,
        pretrained_path=args.checkpoint,
    ).to(device)

    metrics, cm, sample_images, all_true, all_pred = run_evaluation(
        model, loader, device, class_names
    )

    # Print
    print(f"\n{'='*50}")
    print(f"{'Split':20s}: {split_name}")
    print(f"{'Accuracy':20s}: {metrics['accuracy'] * 100:.2f}%")
    print(f"{'Macro Precision':20s}: {metrics['macro_precision'] * 100:.2f}%")
    print(f"{'Macro Recall':20s}: {metrics['macro_recall'] * 100:.2f}%")
    print(f"{'Macro F1':20s}: {metrics['macro_f1'] * 100:.2f}%")
    print(f"\nPer-class results:")
    for cls, vals in metrics["per_class"].items():
        print(
            f"  {cls:45s}  P={vals['precision']:.3f}  "
            f"R={vals['recall']:.3f}  F1={vals['f1']:.3f}"
        )
    print(f"{'='*50}\n")

    # Save metrics JSON
    with open(res_dir / f"metrics_{split_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=str(res_dir / f"confusion_matrix_{split_name}.png"),
        title=f"Confusion Matrix – {split_name} split",
    )

    # Sample predictions
    plot_sample_predictions(
        sample_images,
        all_true[:8],
        all_pred[:8],
        class_names,
        save_path=str(res_dir / f"sample_predictions_{split_name}.png"),
    )

    print(f"Results saved to {res_dir}/")


if __name__ == "__main__":
    main()
