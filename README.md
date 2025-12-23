# CIPS-Net: Text-Guided Multi-Class Histopathology Segmentation

> **Collaboration**: RGUKT, India & KU, UAE

---

## 1. Project Overview

- **Text-conditional semantic segmentation** of histopathology tissue images using CIPS-Net
- Enables **query-driven segmentation**: *"Segment Neoplastic and Inflammatory tissue"*
- Addresses the clinical need for **automated, fine-grained tissue classification** in digital pathology

> **Key Innovation**: Single model handles compositional text queries to segment multiple tissue types simultaneously.

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| **Type** | Medical image segmentation (histopathology) |
| **Format** | RGB images + single-channel class-index masks |
| **Resolution** | 224×224 (resized) |
| **Annotations** | Pixel-wise segmentation |

### Classes (5-Class Multi-Class Problem)

| Index | Class Name |
|-------|------------|
| 0 | Neoplastic |
| 1 | Inflammatory |
| 2 | Connective / Soft Tissue |
| 3 | Epithelial |
| 4 | Dead |

> ⚠️ **Not binary segmentation** — all 5 tissue types are predicted simultaneously.

---

## 3. Label Encoding & Ignore Regions

```
Class indices:    {0, 1, 2, 3, 4}
Ignore index:     255
```

- **IGNORE_INDEX = 255** marks pixels that are:
  - Not queried in the text instruction
  - Uncertain or unlabeled regions

> **Critical Design Choice**: Ignored pixels are excluded from loss computation and metric evaluation. This prevents the model from being penalized for background predictions.

---

## 4. Data Loading Pipeline

**`MultiClassSegmentationDataset`** — Custom PyTorch Dataset

- Returns **single-channel class-index masks** (not one-hot)
- Mask dtype: `torch.long` for `CrossEntropyLoss` compatibility
- Non-queried pixels automatically assigned `IGNORE_INDEX=255`
- Augmentations: Random flip, rotation, color jitter

---

## 5. Model Architecture

**CIPS-Net** (Cross-modal Instruction-guided Pixel-wise Segmentation)

| Component | Implementation |
|-----------|----------------|
| Image Encoder | ViT-B/16 (CLIP pretrained) |
| Text Encoder | DistilBERT |
| Decoder | UNet-style with skip connections |
| Output | `(B, 5, H, W)` raw logits |

- **No sigmoid** in final layer — softmax applied implicitly via `CrossEntropyLoss`
- Text-image fusion enables compositional query understanding

> **Why CIPS-Net?** Designed for instruction-following segmentation; naturally handles multi-class tissue queries.

---

## 6. Loss Function & Optimization

### Combined Loss
```python
Loss = CrossEntropyLoss(weight, ignore_index=255) + MultiClassDiceLoss
```

| Component | Purpose |
|-----------|---------|
| **CrossEntropyLoss** | Pixel classification with class weighting |
| **DiceLoss** | Region-level overlap optimization |
| **Class weights** | Inverse frequency weighting (excluding ignored pixels) |

> **Why weighted loss?** Prevents collapse to dominant classes; ensures rare classes (e.g., Dead) receive adequate gradients.

---

## 7. Training Strategy

- **Multi-class training** (5 classes, not binary)
- **Model selection**: Best **Macro Dice** (not loss)
- **Checkpoints saved**:
  - `best.pth` — highest Macro Dice
  - `last.pth` — final epoch
- **Logging**: Per-class Dice every epoch
- **Visualization**: Every N epochs (configurable)
- **Early stopping**: Patience-based on Macro Dice plateau

---

## 8. Evaluation & Metrics

| Metric | Description |
|--------|-------------|
| **Dice (per-class)** | Overlap for each tissue type |
| **Macro Dice** | Mean of per-class Dice scores |
| **Mean IoU** | Intersection-over-Union averaged |

**All metrics**:
- Computed on `argmax` predictions
- **Exclude IGNORE_INDEX pixels**

---

## 9. Key Results (Early Observations)

After ~3 epochs:
- ✅ **80%+ pixel accuracy** on valid regions
- ✅ **All 5 classes learned** — no single-class collapse
- ✅ **Rare class (Dead) detected** early in training
- ✅ Model responds correctly to compositional queries

---

## 10. Project Structure

```
.
├── Dataset/
│   ├── multi_images/          # Input images
│   ├── multi_masks/           # Per-class mask channels
│   └── *.csv                  # Annotations with text queries
├── checkpoints/
│   ├── best.pth               # Best Macro Dice model
│   ├── last.pth               # Last epoch model
│   └── viz_epoch_*.png        # Training visualizations
├── CIPS-Net/                  # Model implementation
├── train_cipsnet_binary.ipynb # Main training notebook
└── README.md
```

---

## 11. Notes & Design Decisions

- ✅ **Multi-class over binary**: Single model predicts all tissue types
- ✅ **IGNORE_INDEX=255**: Prevents loss on non-queried regions
- ✅ **Macro Dice selection**: Ensures balanced class performance
- ✅ **No sigmoid output**: Softmax via loss for multi-class compatibility
- ✅ **Class weighting**: Addresses severe class imbalance in histopathology

---

## 12. How to Run

```bash
# 1. Prepare dataset
#    Place images in Dataset/multi_images/
#    Place masks in Dataset/multi_masks/
#    Ensure CSV annotations are present

# 2. Run training
#    Open train_cipsnet_binary.ipynb
#    Run All cells

# 3. Evaluate
#    Best model: checkpoints/best.pth
#    Visualizations: checkpoints/viz_epoch_*.png
```

---

## Requirements

```
torch >= 2.0
transformers
albumentations
opencv-python
pandas
matplotlib
tqdm
```

---

## License

MIT License

---
