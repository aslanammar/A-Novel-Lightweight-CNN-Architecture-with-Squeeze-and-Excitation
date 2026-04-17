"""
Data augmentation transforms for maize leaf disease images.

Training augmentations are deliberately aggressive to improve generalisation
on field-collected images (varying lighting, angles, etc.).
Validation/test transforms apply only resizing and normalisation.
"""

from torchvision import transforms

# ImageNet statistics are used as a reasonable starting point; fine-tuning
# these on the PlantVillage dataset is also common in the literature.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Return augmentation pipeline for training."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Return deterministic transform pipeline for validation / testing."""
    return transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 256 / 224)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )
