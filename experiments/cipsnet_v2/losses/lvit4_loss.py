"""
Loss Functions for LViT4 Training
==================================

Phase 2 loss improvements:
    1. Multi-scale Contrastive Loss - from multiple decoder levels
    2. Learnable Temperature - model provides temperature
    3. All Phase 1 losses (HoVer-Net + basic contrastive)

Usage:
    loss_fn = LViT4Loss(
        num_classes=6,
        contrastive_weight=0.5,
        multi_scale_weights=[0.5, 0.3, 0.2],  # deep, mid, out
    )
    
    outputs = model(images, texts, return_contrastive_features=True)
    total_loss, loss_dict = loss_fn(outputs, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.cipsnet_v2.losses.losses import (
    HoVerNetLoss,
    DRWScheduler,
    get_class_frequencies_pannuke,
    get_pannuke_class_weights,
)


class MultiScaleContrastiveLoss(nn.Module):
    """
    Multi-scale Contrastive Loss.
    
    Computes contrastive loss at multiple decoder levels:
    - Deep (bottleneck): coarse alignment
    - Mid: intermediate alignment
    - Out: fine-grained alignment
    
    Uses model-provided temperature (learnable).
    """
    
    def __init__(
        self,
        scale_weights: List[float] = [0.5, 0.3, 0.2],  # deep, mid, out
        default_temperature: float = 0.07,
        reduction: str = 'mean'
    ):
        """
        Args:
            scale_weights: Weights for [deep, mid, out] scales
            default_temperature: Fallback temperature if not provided
            reduction: Loss reduction method
        """
        super().__init__()
        self.scale_weights = scale_weights
        self.default_temperature = default_temperature
        self.reduction = reduction
    
    def _contrastive_loss(
        self,
        visual_embed: torch.Tensor,
        text_embed: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """Compute single-scale contrastive loss."""
        B = visual_embed.shape[0]
        
        # Ensure normalized
        visual_embed = F.normalize(visual_embed, p=2, dim=-1)
        text_embed = F.normalize(text_embed, p=2, dim=-1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(text_embed, visual_embed.t()) / temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=visual_embed.device)
        
        # Symmetric loss
        loss_t2v = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)
        loss_v2t = F.cross_entropy(sim_matrix.t(), labels, reduction=self.reduction)
        
        return (loss_t2v + loss_v2t) / 2
    
    def forward(
        self,
        visual_deep: torch.Tensor,
        visual_mid: torch.Tensor,
        visual_out: torch.Tensor,
        text_embed: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-scale contrastive loss.
        
        Args:
            visual_deep: [B, D] deep visual features
            visual_mid: [B, D] mid visual features
            visual_out: [B, D] output visual features
            text_embed: [B, D] text embedding
            temperature: Learnable temperature from model
            
        Returns:
            Total loss and dict of component losses
        """
        if temperature is None:
            temperature = torch.tensor(self.default_temperature, device=visual_deep.device)
        
        # Compute loss at each scale
        loss_deep = self._contrastive_loss(visual_deep, text_embed, temperature)
        loss_mid = self._contrastive_loss(visual_mid, text_embed, temperature)
        loss_out = self._contrastive_loss(visual_out, text_embed, temperature)
        
        # Weighted sum
        total_loss = (
            self.scale_weights[0] * loss_deep +
            self.scale_weights[1] * loss_mid +
            self.scale_weights[2] * loss_out
        )
        
        loss_dict = {
            'contrastive_deep': loss_deep.item(),
            'contrastive_mid': loss_mid.item(),
            'contrastive_out': loss_out.item(),
            'contrastive_total': total_loss.item(),
            'temperature': temperature.item(),
        }
        
        return total_loss, loss_dict


class LViT4Loss(nn.Module):
    """
    Combined loss for LViT4 (Phase 2).
    
    Total Loss = L_hovernet + λ_contrastive * L_multi_scale_contrastive
    
    Features:
    - HoVer-Net loss (NP + HV + Type)
    - Multi-scale contrastive loss (deep, mid, out)
    - Learnable temperature support
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        # HoVer-Net weights
        np_weight: float = 1.0,
        hv_weight: float = 2.0,
        type_weight: float = 2.0,
        # Loss type
        loss_type: str = 'weighted_focal',
        focal_gamma: float = 2.0,
        use_class_weights: bool = True,
        # Contrastive loss
        contrastive_weight: float = 0.5,
        contrastive_scale_weights: List[float] = [0.5, 0.3, 0.2],
        default_temperature: float = 0.07,
        # DRW
        use_drw: bool = True,
        cls_num_list: Optional[List[int]] = None,
    ):
        """
        Args:
            num_classes: Number of classes
            np_weight, hv_weight, type_weight: HoVer-Net loss weights
            loss_type: Classification loss type
            focal_gamma: Focal loss gamma
            use_class_weights: Whether to use class weights
            contrastive_weight: Weight for contrastive loss
            contrastive_scale_weights: Weights for [deep, mid, out]
            default_temperature: Fallback temperature
            use_drw: Whether to use DRW
            cls_num_list: Class frequencies
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.loss_type = loss_type
        
        # Get class weights
        class_weights = get_pannuke_class_weights() if use_class_weights else None
        
        # Get class frequencies
        if cls_num_list is None:
            cls_num_list = get_class_frequencies_pannuke()
        
        # HoVer-Net loss
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
        
        # Multi-scale contrastive loss
        self.contrastive_loss = MultiScaleContrastiveLoss(
            scale_weights=contrastive_scale_weights,
            default_temperature=default_temperature,
        )
        
        print(f"[LViT4Loss] Initialized:")
        print(f"  - HoVer-Net: NP={np_weight}, HV={hv_weight}, Type={type_weight}")
        print(f"  - Loss Type: {loss_type}, Focal γ={focal_gamma}")
        print(f"  - Multi-scale Contrastive: weight={contrastive_weight}")
        print(f"  - Scale weights: deep={contrastive_scale_weights[0]}, "
              f"mid={contrastive_scale_weights[1]}, out={contrastive_scale_weights[2]}")
    
    def update_type_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for DRW."""
        self.hover_loss.update_type_weights(weights)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute LViT4 loss.
        
        Args:
            outputs: Model outputs with:
                - np, hv, type: segmentation outputs
                - contrastive_visual_deep, _mid, _out: multi-scale features
                - contrastive_text: text embedding
                - temperature: learnable temperature
            targets: Target dict
            
        Returns:
            Total loss and loss dict
        """
        # Focus mask
        np_target = targets['np']
        if np_target.dim() == 4:
            np_target = np_target.squeeze(1)
        focus_mask = (np_target > 0).float()
        
        # HoVer-Net loss
        hover_loss, hover_dict = self.hover_loss(
            pred_np=outputs['np'],
            pred_hv=outputs['hv'],
            pred_type=outputs['type'],
            target_np=targets['np'],
            target_hv=targets['hv'],
            target_type=targets['type'],
            focus_mask=focus_mask
        )
        
        # Multi-scale contrastive loss
        contrastive_loss_val = torch.tensor(0.0, device=hover_loss.device)
        contrastive_dict = {}
        
        if 'contrastive_visual_deep' in outputs:
            temperature = outputs.get('temperature', None)
            
            contrastive_loss_val, contrastive_dict = self.contrastive_loss(
                visual_deep=outputs['contrastive_visual_deep'],
                visual_mid=outputs['contrastive_visual_mid'],
                visual_out=outputs['contrastive_visual_out'],
                text_embed=outputs['contrastive_text'],
                temperature=temperature,
            )
        
        # Combined loss
        total_loss = hover_loss + self.contrastive_weight * contrastive_loss_val
        
        # Build loss dict
        loss_dict = {
            **hover_dict,
            **contrastive_dict,
            'contrastive_weight': self.contrastive_weight,
            'total': total_loss.item(),
            'loss_type': self.loss_type,
        }
        
        return total_loss, loss_dict


def create_lvit4_loss(
    num_classes: int = 6,
    np_weight: float = 1.0,
    hv_weight: float = 2.0,
    type_weight: float = 2.0,
    loss_type: str = 'weighted_focal',
    focal_gamma: float = 2.0,
    contrastive_weight: float = 0.5,
    contrastive_scale_weights: List[float] = [0.5, 0.3, 0.2],
    use_drw: bool = True,
    device: str = 'cuda',
) -> LViT4Loss:
    """Factory function for LViT4Loss."""
    loss_fn = LViT4Loss(
        num_classes=num_classes,
        np_weight=np_weight,
        hv_weight=hv_weight,
        type_weight=type_weight,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        contrastive_weight=contrastive_weight,
        contrastive_scale_weights=contrastive_scale_weights,
        use_drw=use_drw,
    )
    return loss_fn.to(device)


# ============================================================
# Testing
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("LViT4 Loss Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 2, 256, 256
    num_classes = 6
    embed_dim = 512
    
    # Create loss
    loss_fn = create_lvit4_loss(
        num_classes=num_classes,
        contrastive_weight=0.5,
        device=str(device),
    )
    
    # Create dummy data (simulating LViT4 outputs)
    outputs = {
        'np': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'hv': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'type': torch.randn(B, num_classes, H, W, device=device, requires_grad=True),
        'contrastive_visual_deep': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
        'contrastive_visual_mid': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
        'contrastive_visual_out': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
        'contrastive_text': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
        'temperature': torch.tensor(0.07, device=device),
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
    
    print("\n" + "=" * 70)
    print("✅ LViT4 Loss test passed!")
    print("=" * 70)
