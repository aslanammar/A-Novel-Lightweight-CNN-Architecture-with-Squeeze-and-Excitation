"""
Training script for MaizeSENet.

Usage::

    python train.py --data_dir /path/to/dataset --epochs 60 --batch_size 32

The dataset directory must contain ``train/`` and ``val/`` sub-directories,
each organised as one folder per class (ImageFolder layout).

Checkpoints are saved to ``checkpoints/`` and training curves to
``results/training_curves.png``.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_dataloaders
from model import build_model
from utils import AverageMeter, compute_metrics, plot_training_curves

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MaizeSENet for maize leaf disease detection"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory with train/ and val/ splits")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for Adam (default: 1e-4)")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (default: 4)")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout probability (default: 0.4)")
    parser.add_argument("--se_reduction", type=int, default=16,
                        help="SE reduction ratio (default: 16)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--no_balanced_sampling", action="store_true",
                        help="Disable class-balanced sampling")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints (default: checkpoints)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for saving results (default: results)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    model.train()
    loss_meter = AverageMeter("loss")
    all_true, all_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        all_true.append(labels.cpu())
        all_pred.append(preds.cpu())

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    acc = (all_true == all_pred).float().mean().item() * 100
    return loss_meter.avg, acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_true, all_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        all_true.append(labels.cpu())
        all_pred.append(preds.cpu())

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    acc = (all_true == all_pred).float().mean().item() * 100
    return loss_meter.avg, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dirs
    ckpt_dir = Path(args.checkpoint_dir)
    res_dir = Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampling=not args.no_balanced_sampling,
        pin_memory=device.type == "cuda",
    )

    # Model
    model = build_model(
        num_classes=args.num_classes,
        dropout=args.dropout,
        se_reduction=args.se_reduction,
    ).to(device)
    print(f"MaizeSENet – trainable parameters: {model.count_parameters():,}")

    # Loss, optimiser, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.2f}% "
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"({elapsed:.1f}s)"
        )

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": vl_acc,
                    "args": vars(args),
                },
                ckpt_dir / "best_model.pth",
            )

    # Save final checkpoint
    torch.save(
        {"epoch": args.epochs, "model_state_dict": model.state_dict()},
        ckpt_dir / "last_model.pth",
    )

    # Save history
    with open(res_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    plot_training_curves(history, save_path=str(res_dir / "training_curves.png"))
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints: {ckpt_dir}/")
    print(f"Results:     {res_dir}/")


if __name__ == "__main__":
    main()
