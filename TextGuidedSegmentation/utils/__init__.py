"""
TextGuidedSegmentation Utilities
"""

from .dataset import (
    PANNUKE_CLASSES,
    PANNUKE_CLASS_NAMES,
    PanNukeDataset,
    CoNSePDataset,
    MoNuSACDataset,
    get_dataloader,
    generate_text_prompts,
    get_prompt_templates,
)

from .losses import (
    DiceLoss,
    FocalLoss,
    CombinedSegmentationLoss,
    ContrastiveLoss,
    TextAlignmentLoss,
    get_loss_function,
)

from .metrics import (
    ConfusionMatrix,
    MetricTracker,
    compute_iou,
    compute_dice,
    compute_f1,
    pixel_accuracy,
    class_wise_metrics,
)

from .base_model import (
    TextGuidedSegmentationBase,
    ConvBlock,
    DecoderBlock,
    FPN,
    MODEL_REGISTRY,
    register_model,
    build_model,
)

__all__ = [
    # Dataset
    'PANNUKE_CLASSES',
    'PANNUKE_CLASS_NAMES', 
    'PanNukeDataset',
    'CoNSePDataset',
    'MoNuSACDataset',
    'get_dataloader',
    'generate_text_prompts',
    'get_prompt_templates',
    # Losses
    'DiceLoss',
    'FocalLoss',
    'CombinedSegmentationLoss',
    'ContrastiveLoss',
    'TextAlignmentLoss',
    'get_loss_function',
    # Metrics
    'ConfusionMatrix',
    'MetricTracker',
    'compute_iou',
    'compute_dice',
    'compute_f1',
    'pixel_accuracy',
    'class_wise_metrics',
    # Base Model
    'TextGuidedSegmentationBase',
    'ConvBlock',
    'DecoderBlock',
    'FPN',
    'MODEL_REGISTRY',
    'register_model',
    'build_model',
]
