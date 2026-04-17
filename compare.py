"""
Comparative evaluation script.

Compares MaizeSENet against several well-known lightweight CNN baselines:
  - MobileNetV2
  - EfficientNet-B0
  - ShuffleNet v2 (×1.0)
  - SqueezeNet 1.1

All baseline models are loaded from torchvision with random initialisation
(no ImageNet weights) so the comparison is architecture-only on the same
training budget.

Usage::

    python compare.py --data_dir /path/to/dataset --epochs 40

Results are saved to ``results/comparison/``.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from data import get_dataloaders
from model import build_model
from utils import AverageMeter


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparative evaluation of MaizeSENet vs baselines"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_dir", type=str, default="results/comparison")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Baseline model factory
# ---------------------------------------------------------------------------


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final classification layer to match ``num_classes``."""
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            last = list(clf.children())[-1]
            if isinstance(last, nn.Linear):
                clf[-1] = nn.Linear(last.in_features, num_classes)
        elif isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, num_classes)
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_baseline_models(num_classes: int) -> Dict[str, nn.Module]:
    """Return a dict of {name: model} for baseline architectures."""
    mobilenet = _replace_classifier(
        models.mobilenet_v2(weights=None), num_classes
    )
    efficientnet = _replace_classifier(
        models.efficientnet_b0(weights=None), num_classes
    )
    shufflenet = models.shufflenet_v2_x1_0(weights=None)
    shufflenet.fc = nn.Linear(shufflenet.fc.in_features, num_classes)

    squeezenet = models.squeezenet1_1(weights=None)
    squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    squeezenet.num_classes = num_classes

    return {
        "MobileNetV2": mobilenet,
        "EfficientNet-B0": efficientnet,
        "ShuffleNetV2-x1.0": shufflenet,
        "SqueezeNet-1.1": squeezenet,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def train_and_eval(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[Dict, float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history: Dict = {"train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        all_t, all_p = [], []
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            criterion(out, lbls).backward()
            optimizer.step()
            all_t.append(lbls.cpu())
            all_p.append(out.argmax(1).cpu())
        tr_acc = (torch.cat(all_t) == torch.cat(all_p)).float().mean().item() * 100

        # val
        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                all_t.append(lbls.cpu())
                all_p.append(out.argmax(1).cpu())
        vl_acc = (torch.cat(all_t) == torch.cat(all_p)).float().mean().item() * 100

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  val_acc={vl_acc:.1f}%")

    return history, max(history["val_acc"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def plot_comparison_bar(summary: Dict, save_path: str) -> None:
    names = list(summary.keys())
    accs = [summary[n]["best_val_acc"] for n in names]
    params = [summary[n]["parameters"] / 1e6 for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#1565C0" if "MaizeSENet" in n else "#90CAF9" for n in names]
    bars = ax1.bar(range(len(names)), accs, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Best Validation Accuracy (%)")
    ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylim(0, 108)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", va="bottom", fontsize=8,
                 fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(range(len(names)), params, color=colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Trainable Parameters (M)")
    ax2.set_title("Model Size Comparison")
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{p:.2f}M", ha="center", va="bottom", fontsize=8,
                 fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Comparative Evaluation – MaizeSENet vs Baselines", fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curves(all_histories: Dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = ["-", "--", "-.", ":", "-", "--"]
    for (name, hist), style in zip(all_histories.items(), styles):
        color = "#1565C0" if "MaizeSENet" in name else None
        lw = 2.5 if "MaizeSENet" in name else 1.5
        ax.plot(hist["val_acc"], linestyle=style, label=name,
                color=color, linewidth=lw)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy – MaizeSENet vs Baselines")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nComparison results → {res_dir}\n")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # All models: MaizeSENet first, then baselines
    all_models = {"MaizeSENet (ours)": build_model(num_classes=args.num_classes)}
    all_models.update(get_baseline_models(args.num_classes))

    summary: Dict = {}
    all_histories: Dict = {}

    for name, model in all_models.items():
        model = model.to(device)
        params = count_parameters(model)
        print(f"\n── {name} ({params:,} params) ──")
        t0 = time.time()
        history, best_acc = train_and_eval(
            model, train_loader, val_loader, args.epochs, args.lr, device
        )
        elapsed = time.time() - t0
        print(f"   Best val acc: {best_acc:.2f}%  | time: {elapsed:.1f}s")
        summary[name] = {
            "parameters": params,
            "best_val_acc": best_acc,
            "train_time_s": elapsed,
        }
        all_histories[name] = history

    # Print table
    header = f"{'Model':<25} {'Params':>10} {'Best Val Acc':>13} {'Time':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, info in summary.items():
        marker = " ◄" if "MaizeSENet" in name else ""
        print(
            f"{name:<25} {info['parameters']:>10,} "
            f"{info['best_val_acc']:>12.2f}% "
            f"{info['train_time_s']:>9.1f}s{marker}"
        )
    print("=" * len(header))

    # Save
    with open(res_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_comparison_bar(summary, str(res_dir / "comparison_bar.png"))
    plot_accuracy_curves(all_histories, str(res_dir / "comparison_curves.png"))
    print(f"\nComparison plots saved to {res_dir}/")


if __name__ == "__main__":
    main()
