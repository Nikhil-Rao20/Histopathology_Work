"""
TextGuidedSegmentation Models
============================

This module contains implementations of 14 state-of-the-art text-guided 
semantic segmentation models adapted for histopathology cell segmentation.

All models inherit from TextGuidedSegmentationBase and are registered
in the MODEL_REGISTRY for easy instantiation via model name.

Available Models:
-----------------
1. CLIPSeg (CVPR 2022) - timojl/clipseg
2. LSeg (ICLR 2022) - isl-org/lang-seg
3. GroupViT (CVPR 2022) - NVlabs/GroupViT
4. SAN (CVPR 2023) - MendelXu/SAN
5. FC-CLIP (NeurIPS 2023) - bytedance/fc-clip
6. OVSeg (CVPR 2023) - facebookresearch/ov-seg
7. CAT-Seg (CVPR 2024) - cvlab-kaist/CAT-Seg
8. SED (CVPR 2024) - xb534/SED
9. MAFT+ (ECCV 2024 Oral) - jiaosiyu1999/MAFT-Plus
10. X-Decoder (CVPR 2023) - microsoft/X-Decoder
11. OpenSeeD (ICCV 2023) - IDEA-Research/OpenSeeD
12. ODISE (CVPR 2023) - NVlabs/ODISE
13. TagAlign (arXiv 2023) - Qinying-Liu/TagAlign
14. Semantic-SAM (ECCV 2024) - UX-Decoder/Semantic-SAM

Usage:
------
>>> from TextGuidedSegmentation.models import get_model, list_models
>>> 
>>> # List available models
>>> print(list_models())
>>> 
>>> # Create a model by name
>>> model = get_model("clipseg", num_classes=5, image_size=256)
>>> 
>>> # Forward pass
>>> outputs = model(images, text_prompts)
>>> logits = outputs['logits']  # (B, num_classes, H, W)
"""

# Import base components
from ..utils.base_model import (
    TextGuidedSegmentationBase,
    MODEL_REGISTRY,
    register_model,
    get_model,
    list_models,
    ConvBlock,
    DecoderBlock,
    FPN,
)

# Import all model implementations
# Each import triggers the @register_model decorator
from .clipseg import CLIPSeg
from .lseg import LSeg
from .groupvit import GroupViT
from .san import SAN
from .fc_clip import FCCLIP
from .ovseg import OVSeg
from .cat_seg import CATSeg
from .sed import SED
from .maft_plus import MAFTPlus
from .x_decoder import XDecoder
from .openseed import OpenSeeD
from .odise import ODISE
from .tagalign import TagAlign
from .semantic_sam import SemanticSAM

# Convenience aliases
CLIPSegModel = CLIPSeg
LSegModel = LSeg
GroupViTModel = GroupViT
SANModel = SAN
FCCLIPModel = FCCLIP
OVSegModel = OVSeg
CATSegModel = CATSeg
SEDModel = SED
MAFTModel = MAFTPlus
XDecoderModel = XDecoder
OpenSeeDModel = OpenSeeD
ODISEModel = ODISE
TagAlignModel = TagAlign
SemanticSAMModel = SemanticSAM

# Export all public symbols
__all__ = [
    # Base classes
    'TextGuidedSegmentationBase',
    'MODEL_REGISTRY',
    'register_model',
    'get_model',
    'list_models',
    
    # Building blocks
    'ConvBlock',
    'DecoderBlock',
    'FPN',
    
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
    
    # Aliases
    'CLIPSegModel',
    'LSegModel',
    'GroupViTModel',
    'SANModel',
    'FCCLIPModel',
    'OVSegModel',
    'CATSegModel',
    'SEDModel',
    'MAFTModel',
    'XDecoderModel',
    'OpenSeeDModel',
    'ODISEModel',
    'TagAlignModel',
    'SemanticSAMModel',
]

# Model metadata for documentation
MODEL_INFO = {
    'clipseg': {
        'name': 'CLIPSeg',
        'venue': 'CVPR 2022',
        'paper': 'Image Segmentation Using Text and Image Prompts',
        'github': 'https://github.com/timojl/clipseg',
        'description': 'Uses CLIP features with FiLM conditioning for zero-shot segmentation',
    },
    'lseg': {
        'name': 'LSeg',
        'venue': 'ICLR 2022',
        'paper': 'Language-driven Semantic Segmentation',
        'github': 'https://github.com/isl-org/lang-seg',
        'description': 'Dense Prediction Transformer with CLIP text encoder',
    },
    'groupvit': {
        'name': 'GroupViT',
        'venue': 'CVPR 2022',
        'paper': 'GroupViT: Semantic Segmentation Emerges from Text Supervision',
        'github': 'https://github.com/NVlabs/GroupViT',
        'description': 'Hierarchical grouping mechanism with contrastive learning',
    },
    'san': {
        'name': 'SAN',
        'venue': 'CVPR 2023',
        'paper': 'Side Adapter Network for Open-Vocabulary Semantic Segmentation',
        'github': 'https://github.com/MendelXu/SAN',
        'description': 'Side adapter network preserving CLIP capabilities',
    },
    'fc_clip': {
        'name': 'FC-CLIP',
        'venue': 'NeurIPS 2023',
        'paper': 'Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Conv CLIP',
        'github': 'https://github.com/bytedance/fc-clip',
        'description': 'Fully convolutional CLIP for dense prediction',
    },
    'ovseg': {
        'name': 'OVSeg',
        'venue': 'CVPR 2023',
        'paper': 'Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP',
        'github': 'https://github.com/facebookresearch/ov-seg',
        'description': 'Mask-adapted CLIP with region-level classification',
    },
    'cat_seg': {
        'name': 'CAT-Seg',
        'venue': 'CVPR 2024',
        'paper': 'CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation',
        'github': 'https://github.com/cvlab-kaist/CAT-Seg',
        'description': 'Cost aggregation with spatial semantic relationship modeling',
    },
    'sed': {
        'name': 'SED',
        'venue': 'CVPR 2024',
        'paper': 'SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation',
        'github': 'https://github.com/xb534/SED',
        'description': 'Simple encoder-decoder with category-guided decoding',
    },
    'maft_plus': {
        'name': 'MAFT+',
        'venue': 'ECCV 2024 Oral',
        'paper': 'MAFT+: Multi-modal Adapter Finetuning for Vision-Language Models',
        'github': 'https://github.com/jiaosiyu1999/MAFT-Plus',
        'description': 'Multi-modal adapters with cross-modal fusion',
    },
    'x_decoder': {
        'name': 'X-Decoder',
        'venue': 'CVPR 2023',
        'paper': 'Generalized Decoding for Pixel, Image, and Language',
        'github': 'https://github.com/microsoft/X-Decoder',
        'description': 'Unified decoder supporting multiple vision-language tasks',
    },
    'openseed': {
        'name': 'OpenSeeD',
        'venue': 'ICCV 2023',
        'paper': 'A Simple Framework for Open-Vocabulary Segmentation and Detection',
        'github': 'https://github.com/IDEA-Research/OpenSeeD',
        'description': 'Unified framework with decoupled semantic and visual queries',
    },
    'odise': {
        'name': 'ODISE',
        'venue': 'CVPR 2023',
        'paper': 'Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models',
        'github': 'https://github.com/NVlabs/ODISE',
        'description': 'Leverages diffusion model features for semantic understanding',
    },
    'tagalign': {
        'name': 'TagAlign',
        'venue': 'arXiv 2023',
        'paper': 'TagAlign: Improving Vision-Language Alignment with Multi-Tag Classification',
        'github': 'https://github.com/Qinying-Liu/TagAlign',
        'description': 'Tag-guided alignment with multi-granularity matching',
    },
    'semantic_sam': {
        'name': 'Semantic-SAM',
        'venue': 'ECCV 2024',
        'paper': 'Semantic-SAM: Segment and Recognize Anything at Any Granularity',
        'github': 'https://github.com/UX-Decoder/Semantic-SAM',
        'description': 'Multi-granularity segmentation with semantic understanding',
    },
}

def get_model_info(model_name: str) -> dict:
    """Get metadata about a specific model."""
    if model_name.lower() not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_INFO.keys())}")
    return MODEL_INFO[model_name.lower()]

def print_model_summary():
    """Print a summary table of all available models."""
    print("\n" + "="*80)
    print("Text-Guided Segmentation Models for Histopathology")
    print("="*80)
    print(f"\n{'#':<3} {'Model':<15} {'Venue':<20} {'Description':<40}")
    print("-"*80)
    
    for i, (name, info) in enumerate(MODEL_INFO.items(), 1):
        desc = info['description'][:38] + '..' if len(info['description']) > 40 else info['description']
        print(f"{i:<3} {info['name']:<15} {info['venue']:<20} {desc:<40}")
    
    print("-"*80)
    print(f"Total: {len(MODEL_INFO)} models")
    print("="*80 + "\n")
