"""
Squeeze-and-Excitation (SE) Block.

Reference:
    Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
    In CVPR (pp. 7132-7141).
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention block.

    The SE block adaptively recalibrates channel-wise feature responses by
    explicitly modelling interdependencies between channels. It consists of
    two steps:
      1. Squeeze: global average pooling to aggregate spatial information.
      2. Excitation: two fully-connected layers that produce per-channel
         scaling weights in [0, 1].

    Args:
        channels (int): Number of input (and output) channels.
        reduction (int): Reduction ratio for the bottleneck FC layer.
            Defaults to ``16``.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: (B, C, H, W) -> (B, C)
        s = self.squeeze(x).view(b, c)
        # Excitation: (B, C) -> (B, C, 1, 1)
        e = self.excitation(s).view(b, c, 1, 1)
        # Scale
        return x * e
