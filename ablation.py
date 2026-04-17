"""
Ablation study script for MaizeSENet.

Compares the full MaizeSENet against three ablated variants to quantify the
contribution of each design choice:

  1. **Baseline CNN** – plain depthwise-separable CNN, *no* SE blocks.
  2. **SE-Only CNN** – standard (non-depthwise) Conv with SE blocks.
  3. **DW-Only CNN** – depthwise-separable CNN, *no* SE blocks (same as Baseline).
  4. **MaizeSENet (full)** – depthwise-separable CNN + SE blocks.

Each variant is trained from scratch with identical hyper-parameters; results
are aggregated into a summary table and bar charts.

Usage::

    python ablation.py --data_dir /path/to/dataset --epochs 40

Results are saved to ``results/ablation/``.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_dataloaders
from model.se_block import SEBlock
from utils import AverageMeter, compute_metrics, plot_confusion_matrix

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MaizeSENet ablation study")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_dir", type=str, default="results/ablation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ablation model definitions
# ---------------------------------------------------------------------------


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DepthSep(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = _ConvBNReLU(in_ch, in_ch, stride=stride, groups=in_ch)
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


def _make_backbone(use_depthsep: bool, use_se: bool, num_classes: int) -> nn.Module:
    """Factory that builds an ablation variant.

    Args:
        use_depthsep: Use depthwise-separable convolutions.
        use_se: Add SE blocks after each stage.
        num_classes: Number of output classes.
    """
    channels = [32, 64, 128, 256, 512]
    strides = [2, 2, 2, 2, 2]
    layers = [nn.Sequential(
        nn.Conv2d(3, channels[0], 3, stride=strides[0], padding=1, bias=False),
        nn.BatchNorm2d(channels[0]),
        nn.ReLU(inplace=True),
    )]
    for i in range(1, len(channels)):
        in_ch, out_ch = channels[i - 1], channels[i]
        if use_depthsep:
            block = _DepthSep(in_ch, out_ch, stride=strides[i])
        else:
            block = _ConvBNReLU(in_ch, out_ch, stride=strides[i])
        if use_se:
            layers.append(nn.Sequential(block, SEBlock(out_ch)))
        else:
            layers.append(block)

    backbone = nn.Sequential(*layers)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.drop = nn.Dropout(0.4)
            self.fc = nn.Linear(channels[-1], num_classes)
            self._init()

        def _init(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            x = self.backbone(x)
            x = self.gap(x).flatten(1)
            x = self.drop(x)
            return self.fc(x)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return _Model()


ABLATION_VARIANTS: Dict[str, Tuple[bool, bool]] = {
    "Baseline (no SE, no DW-Sep)": (False, False),
    "SE-Only (SE, no DW-Sep)":     (False, True),
    "DW-Sep-Only (no SE)":         (True, False),
    "MaizeSENet (SE + DW-Sep)":    (True, True),
}


# ---------------------------------------------------------------------------
# Training / evaluation loops
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

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        lm = AverageMeter()
        all_t, all_p = [], []
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            preds = out.argmax(1)
            lm.update(loss.item(), imgs.size(0))
            all_t.append(lbls.cpu()); all_p.append(preds.cpu())
        tr_acc = (torch.cat(all_t) == torch.cat(all_p)).float().mean().item() * 100
        history["train_loss"].append(lm.avg)
        history["train_acc"].append(tr_acc)

        # --- val ---
        model.eval()
        vlm = AverageMeter()
        all_t, all_p = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                preds = out.argmax(1)
                vlm.update(loss.item(), imgs.size(0))
                all_t.append(lbls.cpu()); all_p.append(preds.cpu())
        vl_acc = (torch.cat(all_t) == torch.cat(all_p)).float().mean().item() * 100
        history["val_loss"].append(vlm.avg)
        history["val_acc"].append(vl_acc)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:3d}/{epochs}  "
                f"train_acc={tr_acc:.1f}%  val_acc={vl_acc:.1f}%"
            )

    best_val_acc = max(history["val_acc"])
    return history, best_val_acc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(summary: Dict) -> None:
    header = f"{'Variant':<40} {'Params':>10} {'Best Val Acc':>12} {'Train Time':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, info in summary.items():
        print(
            f"{name:<40} {info['parameters']:>10,} "
            f"{info['best_val_acc']:>11.2f}% "
            f"{info['train_time_s']:>11.1f}s"
        )
    print("=" * len(header) + "\n")


def plot_ablation_bar(summary: Dict, save_path: str) -> None:
    names = list(summary.keys())
    accs = [summary[n]["best_val_acc"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), accs, color=["#90CAF9", "#64B5F6", "#42A5F5", "#1565C0"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Ablation Study – MaizeSENet Component Contributions")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_curves(all_histories: Dict, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#E57373", "#64B5F6", "#81C784", "#FFB74D"]
    for (name, hist), color in zip(all_histories.items(), colors):
        epochs = range(1, len(hist["val_loss"]) + 1)
        axes[0].plot(epochs, hist["val_loss"], label=name, color=color)
        axes[1].plot(epochs, hist["val_acc"], label=name, color=color)
    for ax, ylabel, title in zip(
        axes,
        ["Validation Loss", "Validation Accuracy (%)"],
        ["Validation Loss Curves", "Validation Accuracy Curves"],
    ):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Ablation Study – MaizeSENet", fontsize=13)
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
    print(f"Device: {device}\nAblation results → {res_dir}\n")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    summary: Dict = {}
    all_histories: Dict = {}

    for variant_name, (use_dw, use_se) in ABLATION_VARIANTS.items():
        print(f"\n── Variant: {variant_name} ──")
        model = _make_backbone(use_dw, use_se, args.num_classes).to(device)
        print(f"   Parameters: {model.count_parameters():,}")
        t0 = time.time()
        history, best_acc = train_and_eval(
            model, train_loader, val_loader, args.epochs, args.lr, device
        )
        elapsed = time.time() - t0
        print(f"   Best val acc: {best_acc:.2f}%  | time: {elapsed:.1f}s")

        summary[variant_name] = {
            "parameters": model.count_parameters(),
            "best_val_acc": best_acc,
            "train_time_s": elapsed,
        }
        all_histories[variant_name] = history

    print_summary(summary)

    # Save
    with open(res_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(res_dir / "ablation_histories.json", "w") as f:
        json.dump(all_histories, f, indent=2)

    plot_ablation_bar(summary, str(res_dir / "ablation_bar.png"))
    plot_ablation_curves(all_histories, str(res_dir / "ablation_curves.png"))
    print(f"Ablation plots saved to {res_dir}/")


if __name__ == "__main__":
    main()
