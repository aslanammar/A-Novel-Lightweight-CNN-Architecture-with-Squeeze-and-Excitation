"""
MaizeSENet: A Novel Lightweight CNN Architecture with
Squeeze-and-Excitation Attention for Maize Leaf Disease Detection.

Architecture overview
---------------------
The network is built from three types of building blocks:

  ConvBNReLU     – standard Conv → BN → ReLU unit.
  DepthSepBlock  – depthwise-separable convolution (depthwise Conv +
                   pointwise 1×1 Conv), each followed by BN → ReLU.
                   This dramatically reduces parameters and FLOPs.
  SEConvBlock    – ConvBNReLU or DepthSepBlock followed by an SE block
                   for channel-wise recalibration.

Stage layout (default, input 224×224):
  Stage 0  (stem) : 3  -> 32,   stride 2, ConvBNReLU       [112×112]
  Stage 1         : 32 -> 64,   stride 2, DepthSepBlock     [ 56×56]
  Stage 2         : 64 -> 128,  stride 2, SEDepthSepBlock   [ 28×28]
  Stage 3         : 128-> 128,  stride 1, SEDepthSepBlock×2 [ 28×28]
  Stage 4         : 128-> 256,  stride 2, SEDepthSepBlock   [ 14×14]
  Stage 5         : 256-> 256,  stride 1, SEDepthSepBlock×2 [ 14×14]
  Stage 6         : 256-> 512,  stride 2, SEDepthSepBlock   [  7×7 ]
  GAP + Dropout + FC -> num_classes
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .se_block import SEBlock


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthSepBlock(nn.Module):
    """Depthwise-Separable Convolution block.

    Replaces a standard Conv2d with:
      1. Depthwise 3×3 Conv (groups = in_channels)
      2. Pointwise 1×1 Conv
    Each followed by BN + ReLU.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.dw = ConvBNReLU(
            in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels
        )
        self.pw = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SEDepthSepBlock(nn.Module):
    """Depthwise-Separable block followed by a Squeeze-and-Excitation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.ds = DepthSepBlock(in_channels, out_channels, stride=stride)
        self.se = SEBlock(out_channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.ds(x))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class MaizeSENet(nn.Module):
    """Lightweight CNN with SE attention for maize leaf disease classification.

    Args:
        num_classes (int): Number of output classes.
            PlantVillage maize subset has 4 classes by default.
        dropout (float): Dropout probability before the final FC layer.
        se_reduction (int): Reduction ratio used in every SE block.
    """

    # Channel progression for each stage
    _CHANNELS: List[int] = [32, 64, 128, 128, 256, 256, 512]

    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.4,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        ch = self._CHANNELS

        self.stem = ConvBNReLU(3, ch[0], kernel_size=3, stride=2)           # 112

        self.stage1 = DepthSepBlock(ch[0], ch[1], stride=2)                 # 56

        self.stage2 = SEDepthSepBlock(ch[1], ch[2], stride=2,               # 28
                                      se_reduction=se_reduction)

        self.stage3 = nn.Sequential(
            SEDepthSepBlock(ch[2], ch[3], stride=1, se_reduction=se_reduction),
            SEDepthSepBlock(ch[3], ch[3], stride=1, se_reduction=se_reduction),
        )                                                                     # 28

        self.stage4 = SEDepthSepBlock(ch[3], ch[4], stride=2,               # 14
                                      se_reduction=se_reduction)

        self.stage5 = nn.Sequential(
            SEDepthSepBlock(ch[4], ch[5], stride=1, se_reduction=se_reduction),
            SEDepthSepBlock(ch[5], ch[5], stride=1, se_reduction=se_reduction),
        )                                                                     # 14

        self.stage6 = SEDepthSepBlock(ch[5], ch[6], stride=2,               # 7
                                      se_reduction=se_reduction)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(ch[6], num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def feature_maps(self, x: torch.Tensor):
        """Return intermediate feature maps (used in ablation / visualization)."""
        maps = {}
        x = self.stem(x);    maps["stem"]   = x
        x = self.stage1(x);  maps["stage1"] = x
        x = self.stage2(x);  maps["stage2"] = x
        x = self.stage3(x);  maps["stage3"] = x
        x = self.stage4(x);  maps["stage4"] = x
        x = self.stage5(x);  maps["stage5"] = x
        x = self.stage6(x);  maps["stage6"] = x
        return maps


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(
    num_classes: int = 4,
    dropout: float = 0.4,
    se_reduction: int = 16,
    pretrained_path: Optional[str] = None,
) -> MaizeSENet:
    """Construct a MaizeSENet and optionally load saved weights.

    Args:
        num_classes: Number of disease classes (default: 4).
        dropout: Dropout probability (default: 0.4).
        se_reduction: SE reduction ratio (default: 16).
        pretrained_path: Path to a ``.pth`` checkpoint file, or ``None``.

    Returns:
        Initialised ``MaizeSENet`` instance.
    """
    model = MaizeSENet(
        num_classes=num_classes,
        dropout=dropout,
        se_reduction=se_reduction,
    )
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu")
        # Support both raw state-dicts and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    return model
