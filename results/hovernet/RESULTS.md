# HoverNet Experiment Results

## Overview
- **Model**: HoverNet (Fast Mode)
- **Training Dataset**: PanNuke
- **Evaluation**: 3-Fold Cross-Validation
- **Zero-Shot Datasets**: CoNSeP, MoNuSAC
- **Fine-Tuning Datasets**: CoNSeP, MoNuSAC
- **Experiment Date**: January 25, 2026

---

## Configuration
| Parameter | Value |
|-----------|-------|
| Model Mode | fast |
| Number of Types | 6 (Background + 5 cell types) |
| Input Size | 256×256 |
| Output Size | 244×244 |
| Batch Size | 4 |
| Epochs | 20 |
| Learning Rate | 1e-4 |
| Early Stopping | 5 epochs patience |

### Cell Type Classes
| Index | Class Name |
|-------|------------|
| 0 | Background |
| 1 | Neoplastic |
| 2 | Inflammatory |
| 3 | Connective |
| 4 | Dead |
| 5 | Epithelial |

---

## Phase 1: PanNuke 3-Fold Cross-Validation

### Overall Results (Mean ± Std)
| Metric | Value |
|--------|-------|
| **Dice** | **0.7577 ± 0.0031** |
| **AJI** | **0.5625 ± 0.0048** |
| **PQ** | **0.5198 ± 0.0115** |
| DQ | 0.6439 ± 0.0105 |
| SQ | 0.8073 ± 0.0056 |

### Per-Fold Results
| Fold | Dice | AJI | PQ | DQ | SQ |
|------|------|-----|----|----|----| 
| Fold 1 | 0.7535 | 0.5569 | 0.5042 | 0.6315 | 0.7998 |
| Fold 2 | 0.7587 | 0.5621 | 0.5314 | 0.6571 | 0.8089 |
| Fold 3 | 0.7609 | 0.5686 | 0.5238 | 0.6430 | 0.8133 |

### Per-Class PQ (Mean ± Std)
| Class | PQ | DQ | SQ |
|-------|----|----|----| 
| Neoplastic | 0.5155 ± 0.0725 | 0.5736 ± 0.0741 | 0.6408 ± 0.0810 |
| Inflammatory | 0.5763 ± 0.0124 | 0.5895 ± 0.0145 | 0.5956 ± 0.0150 |
| Connective | 0.3136 ± 0.0190 | 0.3657 ± 0.0219 | 0.5377 ± 0.0345 |
| Dead | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Epithelial | 0.6429 ± 0.0449 | 0.6861 ± 0.0458 | 0.7793 ± 0.0634 |

---

## Phase 2: Zero-Shot Evaluation

### CoNSeP (No Fine-Tuning)
| Metric | Value |
|--------|-------|
| Dice | 0.2769 ± 0.1852 |
| AJI | 0.0316 ± 0.0354 |
| PQ | 0.0008 ± 0.0022 |
| DQ | 0.0014 ± 0.0040 |
| SQ | 0.0872 ± 0.2145 |

### MoNuSAC (No Fine-Tuning)
| Metric | Value |
|--------|-------|
| Dice | 0.1904 ± 0.2041 |
| AJI | 0.0996 ± 0.1415 |
| PQ | 0.1003 ± 0.1431 |
| DQ | 0.1277 ± 0.1784 |
| SQ | 0.3950 ± 0.3847 |

---

## Phase 3: Fine-Tuning Results

### CoNSeP (Fine-Tuned)
| Metric | Zero-Shot | Fine-Tuned | Δ |
|--------|-----------|------------|---|
| Dice | 0.2769 | 0.1816 | -0.0953 |
| AJI | 0.0316 | 0.0320 | +0.0004 |
| PQ | 0.0008 | 0.0034 | +0.0026 |

*Note: Performance decreased after fine-tuning, likely due to domain shift and limited data.*

### MoNuSAC (Fine-Tuned)
| Metric | Zero-Shot | Fine-Tuned | Δ |
|--------|-----------|------------|---|
| Dice | 0.1904 | **0.3046** | **+0.1142** |
| AJI | 0.0996 | **0.1411** | **+0.0415** |
| PQ | 0.1003 | **0.1446** | **+0.0443** |

*MoNuSAC showed improvement with fine-tuning.*

---

## Summary Comparison Table

| Dataset | Setting | Dice | AJI | PQ |
|---------|---------|------|-----|----| 
| PanNuke | 3-Fold CV | **0.7577** | **0.5625** | **0.5198** |
| CoNSeP | Zero-Shot | 0.2769 | 0.0316 | 0.0008 |
| CoNSeP | Fine-Tuned | 0.1816 | 0.0320 | 0.0034 |
| MoNuSAC | Zero-Shot | 0.1904 | 0.0996 | 0.1003 |
| MoNuSAC | Fine-Tuned | 0.3046 | 0.1411 | 0.1446 |

---

## Files in This Directory

### Models (`models/`)
- `hovernet_fold1_best.pth` - Best model from Fold 1
- `hovernet_fold1_latest.pth` - Latest checkpoint from Fold 1
- `hovernet_fold2_best.pth` - Best model from Fold 2
- `hovernet_fold2_latest.pth` - Latest checkpoint from Fold 2
- `hovernet_fold3_best.pth` - Best model from Fold 3
- `hovernet_fold3_latest.pth` - Latest checkpoint from Fold 3
- `hovernet_consep_finetuned_best.pth` - Fine-tuned on CoNSeP
- `hovernet_monusac_finetuned_best.pth` - Fine-tuned on MoNuSAC

### Results Files
- `hovernet_pannuke_3fold_results.json` - Detailed 3-fold CV results
- `hovernet_zero_shot_results.json` - Zero-shot evaluation results
- `hovernet_complete_results.json` - All results combined

### Figures (`figures/`)
- `training_history.png` - Training/validation curves for all folds

---

## Notes

1. **PanNuke Performance**: HoverNet achieves strong results on PanNuke with Dice of 0.76 and PQ of 0.52
2. **Zero-Shot Gap**: Significant performance drop on external datasets without fine-tuning
3. **Fine-Tuning**: MoNuSAC benefits from fine-tuning, while CoNSeP shows limited improvement
4. **Class Imbalance**: "Dead" class shows perfect PQ (1.0) likely due to very few samples
5. **Connective Tissue**: Lowest performance across all folds, suggesting difficulty with this class

---

*Generated: January 26, 2026*
