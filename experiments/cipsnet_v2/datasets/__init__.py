"""
CIPS-Net V2 Datasets Package
=============================

PanNuke Dataset loaders for nucleus instance segmentation.

Supported modes:
1. BaselineDataset - No text, just images and masks
2. SingleTextDataset - One instruction per image (from annotations.csv)
3. MultiTextDataset - All possible instructions per image
4. ConditionalTextDataset - Class-conditional instructions

Memory-Optimized Version:
- PanNukeOptimizedDataset - Lazy loading with memory-mapped files
- create_optimized_fold_dataloaders - Default for experiment runner

Permutation-Based Version (for proper text-mask correspondence):
- PanNukePermutationDataset - Uses permutation CSV with PARTIAL masks
- create_permutation_fold_dataloaders - For text-guided segmentation experiments
"""

from .pannuke import (
    PanNukeDataset,
    PanNukeBaselineDataset,
    PanNukeSingleTextDataset,
    PanNukeMultiTextDataset,
    create_pannuke_dataloaders,
    create_fold_dataloaders
)

# Memory-optimized versions (recommended for limited RAM)
from .pannuke_optimized import (
    PanNukeOptimizedDataset,
    create_optimized_dataloaders,
    create_optimized_fold_dataloaders
)

# Permutation-based version (for text-mask correspondence)
from .pannuke_permutation import (
    PanNukePermutationDataset,
    create_permutation_fold_dataloaders
)

# Data augmentation for dataset expansion
from .augmentation import AugmentationConfig

__all__ = [
    # Original (high memory)
    'PanNukeDataset',
    'PanNukeBaselineDataset',
    'PanNukeSingleTextDataset',
    'PanNukeMultiTextDataset',
    'create_pannuke_dataloaders',
    'create_fold_dataloaders',
    # Optimized (low memory)
    'PanNukeOptimizedDataset',
    'create_optimized_dataloaders',
    'create_optimized_fold_dataloaders',
    # Permutation-based (text-mask correspondence)
    'PanNukePermutationDataset',
    'create_permutation_fold_dataloaders',
    # Augmentation
    'AugmentationConfig',
]
