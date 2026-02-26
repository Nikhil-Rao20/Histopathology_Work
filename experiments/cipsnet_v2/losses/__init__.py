"""
Loss Functions for CIPS-Net V2

This module provides all loss functions needed for training CIPS-Net V2:

1. HoVer-Net Style Losses:
   - NPLoss: Nuclei Presence (binary segmentation)
   - HVLoss: Horizontal-Vertical distance maps (regression)
   - TypeLoss: Nucleus type classification

2. Class Imbalance Handling:
   - LDAMLoss: Label-Distribution-Aware Margin Loss
   - FocalLoss: Focal Loss for hard example mining
   - WeightedFocalCELoss: Weighted Focal CE (RECOMMENDED for segmentation)
   - DRWScheduler: Deferred Re-Weighting schedule

3. Combined Losses:
   - HoVerNetLoss: Combined loss for all HoVer-Net outputs
   - CIPSNetV2Loss: Full loss for CIPS-Net V2 training

Loss Types (for ablation studies):
   - 'ce': Standard Cross-Entropy
   - 'weighted_ce': Weighted Cross-Entropy (class weights)
   - 'focal': Focal Loss (unweighted)
   - 'weighted_focal': Weighted Focal CE (RECOMMENDED)
   - 'ldam': LDAM Loss

Usage:
    from experiments.cipsnet_v2.losses import (
        CIPSNetV2Loss,
        DRWScheduler,
        get_class_frequencies_pannuke,
        get_pannuke_class_weights
    )
    
    # Create loss function with weighted focal (RECOMMENDED)
    loss_fn = CIPSNetV2Loss(
        num_classes=6,
        loss_type='weighted_focal',  # Options: 'ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam'
        focal_gamma=2.0,
        type_weight=2.0,  # Higher weight for type classification
    )
    
    # Create DRW scheduler (starts at 30% of training)
    drw = DRWScheduler(
        cls_num_list=get_class_frequencies_pannuke(),
        total_epochs=50,
        drw_start_ratio=0.3  # Earlier rebalancing
    )
    
    # Training loop
    for epoch in range(50):
        # Update weights based on DRW schedule
        loss_fn.update_type_weights(drw.get_weights(epoch))
        
        # Compute loss
        loss, loss_dict = loss_fn(outputs, targets)
"""

from .losses import (
    # Individual losses
    DiceLoss,
    BinaryDiceLoss,
    NPLoss,
    HVLoss,
    TypeLoss,
    
    # Class imbalance
    LDAMLoss,
    FocalLoss,
    WeightedFocalCELoss,  # NEW
    DRWScheduler,
    
    # Contrastive losses (Phase 1)
    TextPixelContrastiveLoss,
    TextPixelContrastiveLossV2,
    
    # Combined losses
    HoVerNetLoss,
    CIPSNetV2Loss,
    
    # Factory functions
    create_hovernet_loss,
    create_drw_scheduler,
    get_class_frequencies_pannuke,
    get_pannuke_class_weights,  # NEW
    
    # Testing
    test_losses,
)

# LViT3 Loss (separate file)
from .lvit3_loss import LViT3Loss, create_lvit3_loss

# LViT4 Loss (Phase 2)
from .lvit4_loss import LViT4Loss, MultiScaleContrastiveLoss, create_lvit4_loss

# LViT-IE Loss (Instance Embedding decoder — novel)
from .instance_embed_loss import (
    PullPushEmbeddingLoss,
    DistanceTransformLoss,
    InstancePooledTypeLoss,
    LViTIELoss,
    PCGrad,
    create_lvit_ie_loss,
)

__all__ = [
    # Individual losses
    'DiceLoss',
    'BinaryDiceLoss',
    'NPLoss',
    'HVLoss',
    'TypeLoss',
    
    # Class imbalance
    'LDAMLoss',
    'FocalLoss',
    'WeightedFocalCELoss',  # NEW
    'DRWScheduler',
    
    # Contrastive losses (Phase 1)
    'TextPixelContrastiveLoss',
    'TextPixelContrastiveLossV2',
    
    # Combined losses
    'HoVerNetLoss',
    'CIPSNetV2Loss',
    
    # LViT3 Loss
    'LViT3Loss',
    'create_lvit3_loss',
    
    # LViT4 Loss (Phase 2)
    'LViT4Loss',
    'MultiScaleContrastiveLoss',
    'create_lvit4_loss',
    
    # LViT-IE Loss (Instance Embedding decoder)
    'PullPushEmbeddingLoss',
    'DistanceTransformLoss',
    'InstancePooledTypeLoss',
    'LViTIELoss',
    'PCGrad',
    'create_lvit_ie_loss',
    
    # Factory functions
    'create_hovernet_loss',
    'create_drw_scheduler',
    'get_class_frequencies_pannuke',
    'get_pannuke_class_weights',  # NEW
    
    # Testing
    'test_losses',
]
