"""
MaizeDataset and DataLoader factory.

Expected directory layout (ImageFolder-compatible)::

    root/
        train/
            Corn___Cercospora_leaf_spot_Gray_leaf_spot/
            Corn___Common_rust/
            Corn___Northern_Leaf_Blight/
            Corn___healthy/
        val/
            ...
        test/  (optional)
            ...

The class names follow the PlantVillage naming convention.  If you use a
different dataset, simply ensure the sub-folder names correspond to your
class labels.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from .augmentation import get_train_transforms, get_val_transforms

# ---------------------------------------------------------------------------
# Class constants
# ---------------------------------------------------------------------------

DISEASE_CLASSES: List[str] = [
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
]

CLASS_LABELS: Dict[str, int] = {cls: idx for idx, cls in enumerate(DISEASE_CLASSES)}
LABEL_NAMES: Dict[int, str] = {
    0: "Cercospora / Gray Leaf Spot",
    1: "Common Rust",
    2: "Northern Leaf Blight",
    3: "Healthy",
}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class MaizeDataset(ImageFolder):
    """Thin wrapper around ``torchvision.datasets.ImageFolder``.

    All heavy lifting (transform, class indexing) is handled by the parent
    class.  This subclass simply documents expected class names and provides
    a convenience ``class_weights`` property for balanced sampling.

    Args:
        root (str | Path): Path to the dataset split directory
            (e.g. ``data/train``).
        transform: Callable applied to each PIL image before returning.
    """

    def __init__(self, root, transform=None):
        super().__init__(str(root), transform=transform)
        # Pre-compute class counts for efficient weight calculations
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        self._class_counts = counts

    @property
    def class_weights(self) -> torch.Tensor:
        """Per-class inverse-frequency weights for balanced sampling.

        Returns:
            1-D float tensor of length ``num_classes``.
        """
        weights = 1.0 / self._class_counts.clamp(min=1)
        return weights / weights.sum()

    def sample_weights(self) -> torch.Tensor:
        """Per-sample weight vector for use with ``WeightedRandomSampler``."""
        cw = self.class_weights
        return torch.tensor([cw[label].item() for _, label in self.samples])


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    balanced_sampling: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train, validation, and (optionally) test DataLoaders.

    Args:
        data_dir: Root directory containing ``train/``, ``val/``,
            and optionally ``test/`` sub-directories.
        batch_size: Mini-batch size for all splits.
        num_workers: Number of worker processes for data loading.
        balanced_sampling: If ``True``, use ``WeightedRandomSampler`` for the
            training split to handle class imbalance.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
        ``test_loader`` is ``None`` when no ``test/`` directory exists.
    """
    root = Path(data_dir)

    train_ds = MaizeDataset(root / "train", transform=get_train_transforms())
    val_ds = MaizeDataset(root / "val", transform=get_val_transforms())

    # Balanced sampler
    if balanced_sampling:
        sample_weights = train_ds.sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Optional test split
    test_path = root / "test"
    if test_path.is_dir():
        test_ds = MaizeDataset(test_path, transform=get_val_transforms())
        test_loader: Optional[DataLoader] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
