"""
CIPS-Net V2: Text-Guided Nucleus Instance Segmentation
=======================================================

A modular framework supporting ablation studies with the following variants:

1. Baseline: ViT encoder + HoVer-Net style decoder (NP, HV, Type heads)
2. + TextEncoder: Add text encoding with simple fusion
3. + CGR: Add Compositional Graph Reasoning module
4. + TextConditionedTypeHead: Text-weighted classification
5. Full CIPS-Net V2: All components combined

Additional Text-Guided Architectures:
- LAVT: Language-Aware Vision Transformer (early fusion)
- CRIS: CLIP-Driven Referring Image Segmentation (contrastive)
- LViT: Language-guided Vision Transformer (U-Net style)
- GroundingDINO: Detection-based with text-guided queries

Usage:
    from experiments.cipsnet_v2.models import create_model, ModelVariant
    
    # Create full model
    model = create_model('FULL', pretrained=True)
    
    # Or create specific variant
    model = create_model('BASELINE', pretrained=True)
    
    # Forward pass
    outputs = model(images, instructions)
    # outputs: {'np': [B,2,H,W], 'hv': [B,2,H,W], 'type': [B,6,H,W]}

Author: Nikhil
"""

from .encoders import ImageEncoder, TextEncoder
from .cgr_module import CompositionalGraphReasoning, GraphAttentionLayer, VisualTextCrossAttention
from .decoder import (
    HoVerNetDecoder, 
    TextConditionedTypeHead, 
    NPHead, 
    HVHead, 
    TypeHead,
    SharedDecoder
)
from .cipsnet_v2 import CIPSNetV2, ModelVariant, create_model
from .lavt_nuclei import LAVTNucleiSegmenter, create_lavt_model
from .cris_nuclei import CRISNucleiSegmenter, create_cris_model
from .lvit_nuclei import LViTNucleiSegmenter, create_lvit_model
from .lvit2_nuclei import LViT2NucleiSegmenter, create_lvit2_model
from .lvit3_nuclei import LViT3NucleiSegmenter, create_lvit3_model
from .lvit4_nuclei import LViT4NucleiSegmenter, create_lvit4_model
from .lvit5_nuclei import LViT5NucleiSegmenter, create_lvit5_model
from .grounding_dino_nuclei import GroundingDINONucleiSegmenter, create_grounding_dino_model
from .lvit_instance_embed import LViTInstanceEmbedSegmenter, create_lvit_ie_model
from .dinov2_encoder import (
    HierarchicalDINOv2Encoder,
    DINOv2ForPretraining,
    DINOv2ClassificationHead,
    create_dinov2_encoder,
    DINOV2_CONFIGS,
)
from .swin_encoder import (
    HierarchicalSwinEncoder,
    create_swin_encoder,
    SWIN_CONFIGS,
)

__all__ = [
    # Encoders
    'ImageEncoder',
    'TextEncoder',
    
    # CGR Module
    'CompositionalGraphReasoning',
    'GraphAttentionLayer',
    'VisualTextCrossAttention',
    
    # Decoder Components
    'HoVerNetDecoder',
    'SharedDecoder',
    'TextConditionedTypeHead',
    'NPHead',
    'HVHead',
    'TypeHead',
    
    # Main Models
    'CIPSNetV2',
    'ModelVariant',
    'create_model',
    
    # LAVT Model (Early Fusion)
    'LAVTNucleiSegmenter',
    'create_lavt_model',
    
    # CRIS Model (Contrastive)
    'CRISNucleiSegmenter',
    'create_cris_model',
    
    # LViT Model (U-Net style)
    'LViTNucleiSegmenter',
    'create_lvit_model',
    
    # LViT2 Model (Enhanced with Deep Supervision + Aux Classification)
    'LViT2NucleiSegmenter',
    'create_lvit2_model',
    
    # LViT3 Model (Instance Normalization + Contrastive Loss Support)
    'LViT3NucleiSegmenter',
    'create_lvit3_model',
    
    # LViT4 Model (Phase 2: Multi-stage Fusion + PWAM)
    'LViT4NucleiSegmenter',
    'create_lvit4_model',
    
    # LViT5 Model (Phase 3: Ultimate - Cross-Modal Decoder + Grounding)
    'LViT5NucleiSegmenter',
    'create_lvit5_model',
    
    # Grounding DINO Model (Detection-based)
    'GroundingDINONucleiSegmenter',
    'create_grounding_dino_model',
    
    # LViT-IE Model (Instance Embedding decoder — novel)
    'LViTInstanceEmbedSegmenter',
    'create_lvit_ie_model',
    
    # DINOv2 Encoder (Self-supervised backbone — Phase 4)
    'HierarchicalDINOv2Encoder',
    'DINOv2ForPretraining',
    'DINOv2ClassificationHead',
    'create_dinov2_encoder',
    'DINOV2_CONFIGS',
    # Swin Transformer Encoder (Phase 4 — hierarchical backbone)
    'HierarchicalSwinEncoder',
    'create_swin_encoder',
    'SWIN_CONFIGS',
]

__version__ = '2.0.0'
