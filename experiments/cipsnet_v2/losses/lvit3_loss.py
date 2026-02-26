"""
Loss Functions for LViT3 Training
==================================

This module extends the base HoVerNetLoss with:
    1. Text-to-Pixel Contrastive Loss (from CRIS paper)
    2. Instance Normalization compatible loss computation

The contrastive loss improves text-visual grounding by:
- Aligning text embeddings with positive pixel features
- Pushing text embeddings away from negative pixels

Usage:
    loss_fn = LViT3Loss(
        num_classes=6,
        contrastive_weight=0.5,
        contrastive_temperature=0.07,
    )
    
    # Model must return contrastive features
    outputs = model(images, texts, return_contrastive_features=True)
    total_loss, loss_dict = loss_fn(outputs, targets)

Author: Enhanced for CIPS-Net V2 Phase 1 improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple

# Import base losses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.cipsnet_v2.losses.losses import (
    HoVerNetLoss,
    DRWScheduler,
    get_class_frequencies_pannuke,
    get_pannuke_class_weights,
    TextPixelContrastiveLoss,
)


class LViT3Loss(nn.Module):
    """
    Combined loss for LViT3 (Instance Norm + Contrastive Loss).
    
    Total Loss = L_hovernet + λ_contrastive * L_contrastive
    
    Where:
    - L_hovernet: Standard HoVer-Net loss (NP + HV + Type)
    - L_contrastive: Text-to-pixel contrastive loss for better grounding
    
    The contrastive loss aligns text embeddings with positive visual features,
    improving the model's ability to ground text descriptions to image regions.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        # HoVer-Net loss weights
        np_weight: float = 1.0,
        hv_weight: float = 2.0,
        type_weight: float = 2.0,
        # Loss type for type classification
        loss_type: str = 'weighted_focal',
        focal_gamma: float = 2.0,
        use_class_weights: bool = True,
        # Contrastive loss settings
        contrastive_weight: float = 0.5,  # Weight for contrastive loss
        contrastive_temperature: float = 0.07,  # Temperature for similarity scaling
        # DRW settings
        use_drw: bool = True,
        cls_num_list: Optional[List[int]] = None,
    ):
        """
        Args:
            num_classes: Number of nucleus types (including background)
            np_weight, hv_weight, type_weight: HoVer-Net loss weights
            loss_type: Type of classification loss
            focal_gamma: Gamma for focal loss
            use_class_weights: Whether to use class weights
            contrastive_weight: Weight for contrastive loss term
            contrastive_temperature: Temperature for contrastive loss
            use_drw: Whether to use DRW schedule
            cls_num_list: Class frequencies for LDAM/DRW
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.use_drw = use_drw
        self.loss_type = loss_type
        
        # Get class weights
        class_weights = get_pannuke_class_weights() if use_class_weights else None
        
        # Get class frequencies
        if cls_num_list is None:
            cls_num_list = get_class_frequencies_pannuke()
        
        # Base HoVer-Net loss
        self.hover_loss = HoVerNetLoss(
            num_classes=num_classes,
            np_weight=np_weight,
            hv_weight=hv_weight,
            type_weight=type_weight,
            type_class_weights=class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            cls_num_list=cls_num_list,
        )
        
        # Contrastive loss for text-visual alignment
        self.contrastive_loss = TextPixelContrastiveLoss(
            temperature=contrastive_temperature,
            reduction='mean'
        )
        
        print(f"[LViT3Loss] Initialized:")
        print(f"  - HoVer-Net: NP={np_weight}, HV={hv_weight}, Type={type_weight}")
        print(f"  - Loss Type: {loss_type}, Focal γ={focal_gamma}")
        print(f"  - Contrastive: weight={contrastive_weight}, τ={contrastive_temperature}")
    
    def update_type_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for DRW schedule."""
        self.hover_loss.update_type_weights(weights)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute LViT3 loss.
        
        Args:
            outputs: Model outputs dict with keys:
                - 'np': [B, 2, H, W]
                - 'hv': [B, 2, H, W]
                - 'type': [B, num_classes, H, W]
                - 'contrastive_visual': [B, D] (optional, for contrastive loss)
                - 'contrastive_text': [B, D] (optional, for contrastive loss)
            targets: Target dict with keys:
                - 'np': [B, H, W]
                - 'hv': [B, 2, H, W]
                - 'type': [B, H, W]
                
        Returns:
            Total loss and loss dict
        """
        # Get focus mask (nuclei regions)
        np_target = targets['np']
        if np_target.dim() == 4:
            np_target = np_target.squeeze(1)
        focus_mask = (np_target > 0).float()
        
        # Compute HoVer-Net loss
        hover_loss, hover_dict = self.hover_loss(
            pred_np=outputs['np'],
            pred_hv=outputs['hv'],
            pred_type=outputs['type'],
            target_np=targets['np'],
            target_hv=targets['hv'],
            target_type=targets['type'],
            focus_mask=focus_mask
        )
        
        # Compute contrastive loss if features are provided
        contrastive_loss_val = torch.tensor(0.0, device=hover_loss.device)
        if 'contrastive_visual' in outputs and 'contrastive_text' in outputs:
            contrastive_loss_val = self.contrastive_loss(
                visual_embed=outputs['contrastive_visual'],
                text_embed=outputs['contrastive_text'],
            )
        
        # Combined loss
        total_loss = hover_loss + self.contrastive_weight * contrastive_loss_val
        
        # Build loss dict
        loss_dict = {
            **hover_dict,
            'contrastive': contrastive_loss_val.item(),
            'contrastive_weight': self.contrastive_weight,
            'total': total_loss.item(),
            'loss_type': self.loss_type,
        }
        
        return total_loss, loss_dict


def create_lvit3_loss(
    num_classes: int = 6,
    np_weight: float = 1.0,
    hv_weight: float = 2.0,
    type_weight: float = 2.0,
    loss_type: str = 'weighted_focal',
    focal_gamma: float = 2.0,
    contrastive_weight: float = 0.5,
    contrastive_temperature: float = 0.07,
    use_drw: bool = True,
    device: str = 'cuda',
) -> LViT3Loss:
    """
    Factory function to create LViT3Loss.
    
    Args:
        num_classes: Number of classes
        np_weight, hv_weight, type_weight: Loss weights
        loss_type: Type of classification loss
        focal_gamma: Gamma for focal loss
        contrastive_weight: Weight for contrastive loss
        contrastive_temperature: Temperature for contrastive loss
        use_drw: Whether to enable DRW schedule
        device: Target device
        
    Returns:
        LViT3Loss instance
    """
    loss_fn = LViT3Loss(
        num_classes=num_classes,
        np_weight=np_weight,
        hv_weight=hv_weight,
        type_weight=type_weight,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        contrastive_weight=contrastive_weight,
        contrastive_temperature=contrastive_temperature,
        use_drw=use_drw,
    )
    
    return loss_fn.to(device)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("LViT3 Loss Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 2, 256, 256
    num_classes = 6
    embed_dim = 512
    
    # Create loss
    loss_fn = create_lvit3_loss(
        num_classes=num_classes,
        contrastive_weight=0.5,
        device=str(device),
    )
    
    # Create dummy data
    outputs = {
        'np': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'hv': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'type': torch.randn(B, num_classes, H, W, device=device, requires_grad=True),
        'contrastive_visual': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
        'contrastive_text': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
    }
    
    targets = {
        'np': torch.randint(0, 2, (B, H, W), device=device),
        'hv': torch.randn(B, 2, H, W, device=device) * 0.5,
        'type': torch.randint(0, num_classes, (B, H, W), device=device),
    }
    
    # Compute loss
    total_loss, loss_dict = loss_fn(outputs, targets)
    
    print(f"\nLoss Components:")
    for k, v in loss_dict.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    
    # Test gradient flow
    total_loss.backward()
    print("\n✓ Gradients computed successfully")
    
    # Test without contrastive features
    print("\n--- Testing without contrastive features ---")
    outputs_no_contrastive = {
        'np': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'hv': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'type': torch.randn(B, num_classes, H, W, device=device, requires_grad=True),
    }
    
    total_loss2, loss_dict2 = loss_fn(outputs_no_contrastive, targets)
    print(f"Contrastive loss (should be 0.0): {loss_dict2['contrastive']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ LViT3 Loss test passed!")
    print("=" * 70)
