"""
Enhanced Loss Functions for LViT2 Training
==========================================

This module extends the base HoVerNetLoss with:
    1. Dice + CE Combined Loss (for better per-class optimization)
    2. Label Smoothing (reduces overconfidence)
    3. Deep Supervision Loss (auxiliary losses at decoder levels)
    4. Auxiliary Classification Loss (global class prediction)

All original loss components are PRESERVED - these are ADDITIONS.

Usage:
    loss_fn = LViT2Loss(
        num_classes=6,
        deep_supervision=True,
        aux_classification=True,
        label_smoothing=0.1,
        dice_weight=0.5,
    )
    
    total_loss, loss_dict = loss_fn(predictions, targets)

Author: Enhanced for CIPS-Net V2 ablation study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple


# =============================================================================
# Label Smoothing Cross Entropy
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    
    Instead of hard labels (0,1), uses soft labels (ε/(C-1), 1-ε)
    This prevents overconfident predictions and improves generalization.
    
    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = recommended)
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N, C] logits
            target: [N] class indices
            
        Returns:
            Smoothed cross entropy loss
        """
        n_classes = pred.size(-1)
        
        # Convert to one-hot and smooth
        with torch.no_grad():
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Compute loss
        log_probs = F.log_softmax(pred, dim=-1)
        
        if self.weight is not None:
            # Apply class weights
            weight = self.weight.to(pred.device)
            loss = -torch.sum(one_hot * log_probs * weight.unsqueeze(0), dim=-1)
        else:
            loss = -torch.sum(one_hot * log_probs, dim=-1)
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


# =============================================================================
# Soft Dice Loss for Multi-class
# =============================================================================

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    
    Directly optimizes the Dice coefficient per class, which is beneficial
    for imbalanced datasets as it treats all classes equally.
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] softmax probabilities
            target: [B, H, W] class indices
            mask: Optional [B, H, W] valid pixel mask
            
        Returns:
            Dice loss
        """
        B, C, H, W = pred.shape
        
        # Convert target to one-hot
        target_onehot = F.one_hot(target.long(), num_classes=C)  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).float()  # [B, 1, H, W]
            pred = pred * mask
            target_onehot = target_onehot * mask
        
        # Flatten spatial dimensions
        pred_flat = pred.view(B, C, -1)  # [B, C, H*W]
        target_flat = target_onehot.view(B, C, -1)  # [B, C, H*W]
        
        # Compute Dice per class
        intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
        cardinality = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # [B, C]
        
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # [B, C]
        dice_loss = 1.0 - dice_per_class  # [B, C]
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            dice_loss = dice_loss * weights.unsqueeze(0)
        
        if self.reduction == 'none':
            return dice_loss
        elif self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()


# =============================================================================
# Enhanced Type Loss with Dice + CE + Label Smoothing
# =============================================================================

class EnhancedTypeLoss(nn.Module):
    """
    Enhanced Type Classification Loss.
    
    Combines:
        1. Focal CE (or Label Smoothing CE) - pixel-wise classification
        2. Soft Dice Loss - per-class overlap optimization
        
    The combination helps balance:
        - CE provides good gradients for learning
        - Dice directly optimizes the metric we care about
        - Label smoothing prevents overconfidence
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,  # NEW: Dice loss contribution
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,  # NEW: Label smoothing
        focal_gamma: float = 2.0,
        use_focal: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        
        # Default class weights for PanNuke
        if class_weights is None:
            class_weights = self._get_pannuke_weights()
        
        self.register_buffer('class_weights', class_weights)
        
        # Label smoothing CE
        self.smooth_ce = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            reduction='none',
            weight=class_weights
        )
        
        # Soft Dice
        self.dice_loss = SoftDiceLoss(
            smooth=1e-6,
            reduction='mean',
            class_weights=class_weights
        )
    
    def _get_pannuke_weights(self) -> torch.Tensor:
        """Get PanNuke class weights (inverse sqrt frequency)."""
        # Approximate frequencies from PanNuke
        frequencies = torch.tensor([
            0.70,   # Background
            0.10,   # Neoplastic
            0.06,   # Inflammatory
            0.05,   # Connective
            0.01,   # Dead (RARE!)
            0.08,   # Epithelial
        ])
        
        # Inverse sqrt frequency, normalized
        weights = 1.0 / torch.sqrt(frequencies + 1e-6)
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        focus_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: [B, C, H, W] type logits
            target: [B, H, W] class indices
            focus_mask: Optional [B, H, W] nuclei region mask
            
        Returns:
            Total loss and component dict
        """
        B, C, H, W = pred.shape
        
        # Flatten for CE
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        target_flat = target.reshape(-1).long()  # [B*H*W]
        
        # Apply focus mask if provided
        if focus_mask is not None:
            mask_flat = focus_mask.reshape(-1).bool()
            if mask_flat.sum() > 0:
                pred_flat_masked = pred_flat[mask_flat]
                target_flat_masked = target_flat[mask_flat]
            else:
                zero = torch.tensor(0.0, device=pred.device, requires_grad=True)
                return zero, {'type_ce': 0.0, 'type_dice': 0.0, 'type_total': 0.0}
        else:
            pred_flat_masked = pred_flat
            target_flat_masked = target_flat
        
        # CE Loss with Label Smoothing
        if self.use_focal:
            # Focal + Label Smoothing
            ce_loss_raw = self.smooth_ce(pred_flat_masked, target_flat_masked)  # [N]
            
            # Apply focal weighting
            probs = F.softmax(pred_flat_masked, dim=-1)
            pt = probs.gather(1, target_flat_masked.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** self.focal_gamma
            ce_loss = (focal_weight * ce_loss_raw).mean()
        else:
            ce_loss = self.smooth_ce(pred_flat_masked, target_flat_masked).mean()
        
        # Dice Loss (on full spatial resolution)
        pred_softmax = F.softmax(pred, dim=1)
        dice_loss = self.dice_loss(pred_softmax, target, focus_mask)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'type_ce': ce_loss.item(),
            'type_dice': dice_loss.item(),
            'type_total': total_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def update_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights (for DRW schedule)."""
        if weights is not None:
            self.class_weights = weights.to(self.class_weights.device)
            self.smooth_ce.weight = weights


# =============================================================================
# Deep Supervision Loss
# =============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Loss for deep supervision auxiliary outputs.
    
    Computes type classification loss on intermediate decoder predictions.
    The loss is weighted to be lower than the main loss.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        weights: List[float] = [0.4, 0.3, 0.2],  # ds1, ds2, ds3 weights
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.weights = weights
        self.num_classes = num_classes
        
        # Use CE + Dice for deep supervision
        self.loss_fn = EnhancedTypeLoss(
            num_classes=num_classes,
            ce_weight=1.0,
            dice_weight=0.3,
            class_weights=class_weights,
            label_smoothing=0.05,  # Less smoothing for auxiliary
            use_focal=True,
            focal_gamma=2.0
        )
    
    def forward(
        self,
        ds_outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        focus_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            ds_outputs: Dict with 'ds1', 'ds2', 'ds3' predictions
            target: [B, H, W] class indices
            focus_mask: Optional nuclei mask
            
        Returns:
            Total deep supervision loss and component dict
        """
        total_loss = torch.tensor(0.0, device=target.device)
        loss_dict = {}
        
        ds_keys = ['ds1', 'ds2', 'ds3']
        
        for i, (key, weight) in enumerate(zip(ds_keys, self.weights)):
            if key in ds_outputs:
                pred = ds_outputs[key]
                loss, _ = self.loss_fn(pred, target, focus_mask)
                weighted_loss = weight * loss
                total_loss = total_loss + weighted_loss
                loss_dict[f'{key}_loss'] = loss.item()
        
        loss_dict['ds_total'] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Auxiliary Classification Loss (Multi-label BCE)
# =============================================================================

class AuxiliaryClassificationLoss(nn.Module):
    """
    Loss for auxiliary classification branch.
    
    Multi-label classification: predict which classes are present in the image.
    Uses BCE with class weights.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        weight: float = 0.5,  # Weight relative to main loss
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.weight = weight
        self.num_classes = num_classes
        
        # Class weights for imbalanced classes
        if class_weights is None:
            # Higher weight for rare classes
            class_weights = torch.tensor([0.5, 1.0, 1.2, 1.3, 3.0, 1.0])  # Dead has 3x weight
        
        self.register_buffer('class_weights', class_weights)
    
    def forward(
        self,
        pred: torch.Tensor,
        target_type: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: [B, num_classes] class presence logits
            target_type: [B, H, W] type map (used to derive class presence)
            
        Returns:
            Auxiliary classification loss
        """
        B = pred.shape[0]
        device = pred.device
        
        # Derive ground truth class presence from type map
        target_presence = torch.zeros(B, self.num_classes, device=device)
        
        for b in range(B):
            unique_classes = torch.unique(target_type[b])
            for c in unique_classes:
                if c < self.num_classes:
                    target_presence[b, c] = 1.0
        
        # BCE loss with class weights
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target_presence,
            weight=self.class_weights.unsqueeze(0).expand(B, -1),
            reduction='mean'
        )
        
        weighted_loss = self.weight * bce_loss
        
        # Compute accuracy for logging
        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            accuracy = (pred_binary == target_presence).float().mean().item()
        
        loss_dict = {
            'aux_class_loss': bce_loss.item(),
            'aux_class_acc': accuracy,
        }
        
        return weighted_loss, loss_dict


# =============================================================================
# Complete LViT2 Loss (HoVerNet + Deep Supervision + Aux Classification)
# =============================================================================

class LViT2Loss(nn.Module):
    """
    Complete loss function for LViT2 training.
    
    Combines:
        1. NP Loss (BCE + Dice) - nuclei presence
        2. HV Loss (MSE + MSGE) - horizontal/vertical maps
        3. Enhanced Type Loss (Focal + Dice + Label Smoothing)
        4. Deep Supervision Loss (auxiliary type predictions)
        5. Auxiliary Classification Loss (global class prediction)
    
    This is an EXTENSION of HoVerNetLoss - all original components preserved!
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        # Main loss weights
        np_weight: float = 1.0,
        hv_weight: float = 2.0,
        type_weight: float = 2.0,
        # Enhanced type loss params
        type_dice_weight: float = 0.5,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        # Deep supervision params
        deep_supervision: bool = True,
        ds_weight: float = 0.5,  # Weight for total deep supervision loss
        # Auxiliary classification params
        aux_classification: bool = True,
        aux_weight: float = 0.3,
        # Class weights
        class_weights: Optional[torch.Tensor] = None,
        # DRW
        use_drw: bool = True,
    ):
        super().__init__()
        
        self.np_weight = np_weight
        self.hv_weight = hv_weight
        self.type_weight = type_weight
        self.ds_weight = ds_weight
        self.aux_weight = aux_weight
        self.use_deep_supervision = deep_supervision
        self.use_aux_classification = aux_classification
        self.use_drw = use_drw
        
        # Import base losses
        from .losses import NPLoss, HVLoss
        
        # NP Loss (unchanged)
        self.np_loss = NPLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            use_focal=True,
            focal_gamma=focal_gamma
        )
        
        # HV Loss (unchanged)
        self.hv_loss = HVLoss(
            mse_weight=1.0,
            msge_weight=1.0
        )
        
        # Enhanced Type Loss (NEW: Dice + Label Smoothing)
        self.type_loss = EnhancedTypeLoss(
            num_classes=num_classes,
            ce_weight=1.0,
            dice_weight=type_dice_weight,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            use_focal=True,
            focal_gamma=focal_gamma
        )
        
        # Deep Supervision Loss (NEW)
        if deep_supervision:
            self.ds_loss = DeepSupervisionLoss(
                num_classes=num_classes,
                weights=[0.4, 0.3, 0.2],
                class_weights=class_weights
            )
        
        # Auxiliary Classification Loss (NEW)
        if aux_classification:
            self.aux_loss = AuxiliaryClassificationLoss(
                num_classes=num_classes,
                weight=aux_weight,
                class_weights=class_weights
            )
    
    def update_type_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for DRW schedule."""
        self.type_loss.update_weights(weights)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        focus_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete LViT2 loss.
        
        Args:
            predictions: Dict with 'np', 'hv', 'type', optionally 'ds1', 'ds2', 'ds3', 'aux_class'
            targets: Dict with 'np', 'hv', 'type'
            focus_mask: Optional nuclei region mask
            
        Returns:
            Total loss and comprehensive loss dict
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=predictions['np'].device)
        
        # 1. NP Loss
        np_loss, np_dict = self.np_loss(predictions['np'], targets['np'])
        total_loss = total_loss + self.np_weight * np_loss
        loss_dict.update({f'np_{k}': v for k, v in np_dict.items()})
        
        # 2. HV Loss
        hv_loss, hv_dict = self.hv_loss(predictions['hv'], targets['hv'], focus_mask)
        total_loss = total_loss + self.hv_weight * hv_loss
        loss_dict.update({f'hv_{k}': v for k, v in hv_dict.items()})
        
        # 3. Enhanced Type Loss (with Dice + Label Smoothing)
        type_loss, type_dict = self.type_loss(predictions['type'], targets['type'], focus_mask)
        total_loss = total_loss + self.type_weight * type_loss
        loss_dict.update(type_dict)
        
        # 4. Deep Supervision Loss (NEW)
        if self.use_deep_supervision and 'ds1' in predictions:
            ds_outputs = {k: v for k, v in predictions.items() if k.startswith('ds')}
            ds_loss, ds_dict = self.ds_loss(ds_outputs, targets['type'], focus_mask)
            total_loss = total_loss + self.ds_weight * ds_loss
            loss_dict.update(ds_dict)
        
        # 5. Auxiliary Classification Loss (NEW)
        if self.use_aux_classification and 'aux_class' in predictions:
            aux_loss, aux_dict = self.aux_loss(predictions['aux_class'], targets['type'])
            total_loss = total_loss + aux_loss  # Already weighted inside
            loss_dict.update(aux_dict)
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Factory Function
# =============================================================================

def create_lvit2_loss(
    num_classes: int = 6,
    deep_supervision: bool = True,
    aux_classification: bool = True,
    label_smoothing: float = 0.1,
    type_dice_weight: float = 0.5,
    focal_gamma: float = 2.0,
    **kwargs
) -> LViT2Loss:
    """Create LViT2 loss with recommended settings."""
    return LViT2Loss(
        num_classes=num_classes,
        np_weight=1.0,
        hv_weight=2.0,
        type_weight=2.0,
        type_dice_weight=type_dice_weight,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        deep_supervision=deep_supervision,
        ds_weight=0.5,
        aux_classification=aux_classification,
        aux_weight=0.3,
        use_drw=True,
    )
