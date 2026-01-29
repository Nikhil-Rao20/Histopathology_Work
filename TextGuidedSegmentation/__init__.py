"""
TextGuidedSegmentation
======================

A comprehensive library of text-guided semantic segmentation models 
adapted for histopathology cell segmentation tasks.

This package implements 14 state-of-the-art models from CVPR, ICCV, 
NeurIPS, ECCV 2022-2024 for comparison with CIPS-Net baseline.

Supported Models (14 total):
1. CLIPSeg (CVPR 2022) - CLIP + Transformer Decoder
2. LSeg (ICLR 2022) - CLIP + DPT Decoder
3. GroupViT (CVPR 2022) - Hierarchical ViT Grouping
4. SAN (CVPR 2023) - Side Adapter Network
5. ODISE (CVPR 2023) - Stable Diffusion + CLIP
6. X-Decoder (CVPR 2023) - Universal Decoder + CLIP
7. OpenSeeD (ICCV 2023) - Open-Vocab Seg+Detection
8. FC-CLIP (NeurIPS 2023) - Frozen Convolutional CLIP
9. OVSeg (CVPR 2023) - Mask-adapted CLIP
10. CAT-Seg (CVPR 2024) - Cost Aggregation
11. SED (CVPR 2024) - Simple Encoder-Decoder
12. MAFT+ (ECCV 2024) - Collaborative Vision-Text
13. TagAlign (arXiv 2023) - Multi-tag Classification
14. Semantic-SAM (ECCV 2024) - Multi-granularity Segmentation

Task:
-----
5-class cell segmentation on histopathology images:
- Class 0: Neoplastic cells
- Class 1: Inflammatory cells  
- Class 2: Connective tissue cells
- Class 3: Dead cells
- Class 4: Epithelial cells

Datasets:
---------
- PanNuke: Training with 3-fold cross-validation
- CoNSeP: Zero-shot / fine-tuning evaluation
- MoNuSAC: Zero-shot / fine-tuning evaluation
"""

__version__ = "1.0.0"
__author__ = "Nikhil"

# Import model utilities from base_model
from .utils.base_model import (
    TextGuidedSegmentationBase,
    MODEL_REGISTRY,
    register_model,
    get_model,
    list_models,
)

# Import model classes (this triggers @register_model decorators)
from .models import (
    CLIPSeg,
    LSeg,
    GroupViT,
    SAN,
    FCCLIP,
    OVSeg,
    CATSeg,
    SED,
    MAFTPlus,
    XDecoder,
    OpenSeeD,
    ODISE,
    TagAlign,
    SemanticSAM,
    get_model_info,
    print_model_summary,
    MODEL_INFO,
)

# Import datasets
from .utils.dataset import (
    PanNukeDataset,
    CoNSePDataset,
    MoNuSACDataset,
    get_pannuke_3fold_splits,
    get_clip_transform,
    PANNUKE_CLASS_NAMES,
    PANNUKE_CLASS_PROMPTS,
)

# Import losses
from .utils.losses import (
    DiceLoss,
    FocalLoss,
    CombinedSegmentationLoss,
    ContrastiveLoss,
    TextAlignmentLoss,
    MaskCLIPLoss,
)

# Import metrics
from .utils.metrics import (
    compute_iou,
    compute_dice,
    compute_f1,
    compute_pq,
    ConfusionMatrix,
    MetricTracker,
)

# Default text prompts for histopathology
DEFAULT_TEXT_PROMPTS = [
    "neoplastic cells",
    "inflammatory cells", 
    "connective tissue cells",
    "dead cells",
    "epithelial cells",
]

# Backward compatibility alias
def build_model(model_name: str, **kwargs):
    """Backward compatible model builder."""
    return get_model(model_name, **kwargs)

__all__ = [
    # Version
    '__version__',
    
    # Model utilities
    'TextGuidedSegmentationBase',
    'MODEL_REGISTRY',
    'register_model',
    'get_model',
    'list_models',
    'build_model',  # backward compat
    'get_model_info',
    'print_model_summary',
    'MODEL_INFO',
    
    # Model classes
    'CLIPSeg',
    'LSeg',
    'GroupViT',
    'SAN',
    'FCCLIP',
    'OVSeg',
    'CATSeg',
    'SED',
    'MAFTPlus',
    'XDecoder',
    'OpenSeeD',
    'ODISE',
    'TagAlign',
    'SemanticSAM',
    
    # Datasets
    'PanNukeDataset',
    'CoNSePDataset',
    'MoNuSACDataset',
    'get_pannuke_3fold_splits',
    'get_clip_transform',
    'PANNUKE_CLASS_NAMES',
    'PANNUKE_CLASS_PROMPTS',
    
    # Losses
    'DiceLoss',
    'FocalLoss',
    'CombinedSegmentationLoss',
    'ContrastiveLoss',
    'TextAlignmentLoss',
    'MaskCLIPLoss',
    
    # Metrics
    'compute_iou',
    'compute_dice',
    'compute_f1',
    'compute_pq',
    'ConfusionMatrix',
    'MetricTracker',
    
    # Constants
    'DEFAULT_TEXT_PROMPTS',
]
