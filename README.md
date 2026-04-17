# A Novel Lightweight CNN Architecture with Squeeze-and-Excitation Attention for Maize Leaf Disease Detection

> **Design, Ablation Analysis, and Comparative Evaluation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)

---

## Overview

**MaizeSENet** is a novel, lightweight convolutional neural network designed specifically for multi-class maize leaf disease classification. It combines two efficiency-focused design choices:

1. **Depthwise-Separable Convolutions** – replace expensive standard 3×3 convolutions to drastically reduce parameter count and FLOPs while preserving representational capacity.
2. **Squeeze-and-Excitation (SE) Attention** – adaptively recalibrate channel-wise feature responses, allowing the network to focus on the most informative feature channels.

The result is a model with **< 2 M trainable parameters** that achieves competitive accuracy on the PlantVillage maize subset (4 disease categories).

---

## Supported Disease Classes

| Index | Class | Common Name |
|-------|-------|-------------|
| 0 | `Corn___Cercospora_leaf_spot_Gray_leaf_spot` | Cercospora / Gray Leaf Spot |
| 1 | `Corn___Common_rust` | Common Rust |
| 2 | `Corn___Northern_Leaf_Blight` | Northern Leaf Blight |
| 3 | `Corn___healthy` | Healthy |

---

## Architecture

```
Input (3 × 224 × 224)
       │
  ┌────▼────┐
  │  Stem   │  ConvBNReLU  3→32  stride=2    → 32×112×112
  └────┬────┘
  ┌────▼────┐
  │ Stage 1 │  DepthSepBlock  32→64  stride=2  → 64×56×56
  └────┬────┘
  ┌────▼────┐
  │ Stage 2 │  SEDepthSepBlock  64→128  stride=2  → 128×28×28
  └────┬────┘
  ┌────▼────┐
  │ Stage 3 │  SEDepthSepBlock × 2  128→128  stride=1  → 128×28×28
  └────┬────┘
  ┌────▼────┐
  │ Stage 4 │  SEDepthSepBlock  128→256  stride=2  → 256×14×14
  └────┬────┘
  ┌────▼────┐
  │ Stage 5 │  SEDepthSepBlock × 2  256→256  stride=1  → 256×14×14
  └────┬────┘
  ┌────▼────┐
  │ Stage 6 │  SEDepthSepBlock  256→512  stride=2  → 512×7×7
  └────┬────┘
  Global Average Pooling → Dropout(0.4) → FC(512 → num_classes)
```

Each **SEDepthSepBlock** consists of:
- Depthwise 3×3 Conv → BN → ReLU
- Pointwise 1×1 Conv → BN → ReLU
- SE block (Global Avg Pool → FC → ReLU → FC → Sigmoid → Scale)

---

## Repository Structure

```
.
├── model/
│   ├── __init__.py
│   ├── se_block.py          # Squeeze-and-Excitation block
│   └── lightweight_cnn.py   # MaizeSENet full architecture + factory
├── data/
│   ├── __init__.py
│   ├── dataset.py           # MaizeDataset (ImageFolder wrapper) + DataLoader factory
│   └── augmentation.py      # Train / val transforms
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # AverageMeter, compute_metrics (acc/P/R/F1)
│   └── visualization.py     # Training curves, confusion matrix, sample predictions
├── tests/
│   ├── test_model.py        # Unit tests for SE block and MaizeSENet
│   └── test_utils.py        # Unit tests for metrics utilities
├── train.py                 # Full training script
├── evaluate.py              # Evaluation + confusion matrix + sample predictions
├── ablation.py              # Ablation study (4 variants)
├── compare.py               # Comparative evaluation vs MobileNetV2, EfficientNet-B0, etc.
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 1.13, torchvision ≥ 0.14, NumPy, Matplotlib, Pillow.

---

## Dataset Preparation

Download the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and arrange it as follows:

```
dataset/
├── train/
│   ├── Corn___Cercospora_leaf_spot_Gray_leaf_spot/
│   ├── Corn___Common_rust/
│   ├── Corn___Northern_Leaf_Blight/
│   └── Corn___healthy/
├── val/
│   └── ...
└── test/           # optional
    └── ...
```

A common 70/15/15 split by class is recommended.

---

## Usage

### Training

```bash
python train.py \
    --data_dir /path/to/dataset \
    --epochs 60 \
    --batch_size 32 \
    --lr 1e-3
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | (required) | Dataset root directory |
| `--epochs` | 60 | Number of training epochs |
| `--batch_size` | 32 | Mini-batch size |
| `--lr` | 1e-3 | Initial learning rate (cosine decay) |
| `--se_reduction` | 16 | SE block reduction ratio |
| `--dropout` | 0.4 | Classifier dropout probability |
| `--no_balanced_sampling` | False | Disable class-balanced sampler |
| `--checkpoint_dir` | `checkpoints/` | Where to save checkpoints |
| `--results_dir` | `results/` | Where to save plots/history |

Checkpoints are saved to `checkpoints/best_model.pth` (best val accuracy) and `checkpoints/last_model.pth`.

### Evaluation

```bash
python evaluate.py \
    --data_dir /path/to/dataset \
    --checkpoint checkpoints/best_model.pth
```

Outputs accuracy, macro P/R/F1, per-class breakdown, confusion matrix, and sample prediction grid to `results/`.

### Ablation Study

```bash
python ablation.py --data_dir /path/to/dataset --epochs 40
```

Trains four variants from scratch and produces a summary table + bar charts in `results/ablation/`:

| Variant | DW-Sep | SE |
|---------|--------|----|
| Baseline CNN | ✗ | ✗ |
| SE-Only CNN | ✗ | ✓ |
| DW-Sep-Only CNN | ✓ | ✗ |
| **MaizeSENet (full)** | ✓ | ✓ |

### Comparative Evaluation

```bash
python compare.py --data_dir /path/to/dataset --epochs 40
```

Trains MaizeSENet and four baselines (MobileNetV2, EfficientNet-B0, ShuffleNetV2, SqueezeNet-1.1) under identical conditions and produces comparison bar charts and accuracy curves in `results/comparison/`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Design Rationale

| Choice | Motivation |
|--------|-----------|
| Depthwise-separable convolutions | Reduces parameters by ~8–9× vs standard convolutions at similar accuracy |
| SE attention after stages 2–6 | Channel recalibration is most beneficial at deeper feature levels |
| Label smoothing (ε = 0.1) | Reduces overconfidence, improves calibration on small datasets |
| Cosine LR annealing | Smooth decay; outperforms step decay on small agricultural datasets |
| Balanced sampling | Handles the natural class imbalance in PlantVillage |
| Aggressive augmentation | Accounts for field-condition variability (lighting, angle, background) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{maizesenet2024,
  title  = {A Novel Lightweight CNN Architecture with Squeeze-and-Excitation
             Attention for Maize Leaf Disease Detection},
  author = {Ammar Aslan},
  year   = {2024},
  url    = {https://github.com/aslanammar/A-Novel-Lightweight-CNN-Architecture-with-Squeeze-and-Excitation}
}
```

---

## License

This project is released under the MIT License.
