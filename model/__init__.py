"""
MaizeSENet: A Novel Lightweight CNN Architecture with Squeeze-and-Excitation Attention
for Maize Leaf Disease Detection.
"""

from .se_block import SEBlock
from .lightweight_cnn import MaizeSENet, build_model

__all__ = ["SEBlock", "MaizeSENet", "build_model"]
