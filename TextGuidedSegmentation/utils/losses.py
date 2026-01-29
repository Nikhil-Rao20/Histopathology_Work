"""
Loss functions for text-guided semantic segmentation.
Includes CE, Dice, Focal, and combined losses with text-image alignment objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# =============================================================================
# BASIC SEGMENTATION LOSSES
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    Supports both binary and multi-class segmentation.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        num_classes = pred.shape[1]
        
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.clamp(0, num_classes-1), num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index).unsqueeze(1).expand_as(pred)
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        # Compute dice per class
        dims = (0, 2, 3)  # Sum over batch and spatial dims
        intersection = (pred * target_one_hot).sum(dims)
        cardinality = (pred + target_one_hot).sum(dims)
        
        dice_score = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        num_classes = pred.shape[1]
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            pred, target, 
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Get probabilities
        p = F.softmax(pred, dim=1)
        
        # Gather the probabilities for the true class
        target_clamped = target.clamp(0, num_classes-1)
        p_t = p.gather(1, target_clamped.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Handle ignore index
        valid_mask = (target != self.ignore_index)
        focal_loss = focal_loss * valid_mask
        
        if self.reduction == 'mean':
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss: CE + Dice + optional Focal
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(
            alpha=class_weights, 
            gamma=gamma, 
            ignore_index=ignore_index
        ) if focal_weight > 0 else None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce_loss(pred, target)
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.focal_weight > 0 and self.focal_loss is not None:
            loss += self.focal_weight * self.focal_loss(pred, target)
        
        return loss


# =============================================================================
# TEXT-IMAGE ALIGNMENT LOSSES
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    CLIP-style contrastive loss for text-image alignment.
    Used in models like CLIPSeg, GroupViT, etc.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (B, D) normalized image embeddings
            text_features: (N, D) normalized text embeddings  
            logit_scale: Optional learnable temperature
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        if logit_scale is not None:
            logit_scale = logit_scale.exp()
        else:
            logit_scale = 1.0 / self.temperature
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # Labels: diagonal elements
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Symmetric loss
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2


class TextAlignmentLoss(nn.Module):
    """
    Loss for aligning pixel-level features with text embeddings.
    Used in dense prediction tasks.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        pixel_features: torch.Tensor,  # (B, D, H, W)
        text_features: torch.Tensor,   # (C, D) 
        target: torch.Tensor,          # (B, H, W)
    ) -> torch.Tensor:
        """
        Compute pixel-text alignment loss.
        """
        B, D, H, W = pixel_features.shape
        C = text_features.shape[0]
        
        # Normalize
        pixel_features = F.normalize(pixel_features, dim=1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity: (B, C, H, W)
        pixel_flat = pixel_features.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
        logits = (pixel_flat @ text_features.t()) / self.temperature  # (B*H*W, C)
        logits = logits.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, target, ignore_index=-100)
        
        return loss


# =============================================================================
# MODEL-SPECIFIC LOSSES
# =============================================================================

class GroupViTLoss(nn.Module):
    """
    GroupViT-style loss combining contrastive and grouping losses.
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        grouping_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.grouping_weight = grouping_weight
        self.contrastive_loss = ContrastiveLoss(temperature)
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        group_tokens: Optional[torch.Tensor] = None,
        attn_maps: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        loss = self.contrastive_weight * self.contrastive_loss(image_features, text_features)
        
        # Add grouping regularization if available
        if self.grouping_weight > 0 and attn_maps is not None:
            for attn in attn_maps:
                # Encourage sparse attention
                entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
                loss += self.grouping_weight * entropy
        
        return loss


class MaskCLIPLoss(nn.Module):
    """
    Loss for mask-based CLIP models (CLIPSeg, OVSeg, etc.)
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        mask_weight: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.seg_loss = CombinedSegmentationLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
        )
        self.mask_weight = mask_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pred_masks: torch.Tensor,      # (B, C, H, W) class logits
        target: torch.Tensor,          # (B, H, W) class labels
        pred_binary: Optional[torch.Tensor] = None,  # (B, 1, H, W) optional binary mask
        target_binary: Optional[torch.Tensor] = None,  # (B, 1, H, W) optional binary target
    ) -> torch.Tensor:
        
        loss = self.seg_loss(pred_masks, target)
        
        if pred_binary is not None and target_binary is not None:
            loss += self.mask_weight * self.bce_loss(pred_binary, target_binary.float())
        
        return loss


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_loss_function(
    loss_type: str = 'combined',
    num_classes: int = 5,
    class_weights: Optional[List[float]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: 'ce', 'dice', 'focal', 'combined', 'groupvit', 'maskclip'
        num_classes: Number of classes
        class_weights: Optional class weights for handling imbalance
        **kwargs: Additional arguments for specific losses
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    losses = {
        'ce': lambda: nn.CrossEntropyLoss(weight=class_weights, ignore_index=kwargs.get('ignore_index', -100)),
        'dice': lambda: DiceLoss(ignore_index=kwargs.get('ignore_index', -100)),
        'focal': lambda: FocalLoss(alpha=class_weights, gamma=kwargs.get('gamma', 2.0), ignore_index=kwargs.get('ignore_index', -100)),
        'combined': lambda: CombinedSegmentationLoss(
            ce_weight=kwargs.get('ce_weight', 1.0),
            dice_weight=kwargs.get('dice_weight', 1.0),
            focal_weight=kwargs.get('focal_weight', 0.0),
            class_weights=class_weights,
            ignore_index=kwargs.get('ignore_index', -100),
        ),
        'groupvit': lambda: GroupViTLoss(
            contrastive_weight=kwargs.get('contrastive_weight', 1.0),
            grouping_weight=kwargs.get('grouping_weight', 0.1),
        ),
        'maskclip': lambda: MaskCLIPLoss(
            ce_weight=kwargs.get('ce_weight', 1.0),
            dice_weight=kwargs.get('dice_weight', 1.0),
            mask_weight=kwargs.get('mask_weight', 0.5),
            ignore_index=kwargs.get('ignore_index', -100),
        ),
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(losses.keys())}")
    
    return losses[loss_type]()
