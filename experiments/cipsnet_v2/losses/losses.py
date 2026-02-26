"""
Loss Functions for CIPS-Net V2 Training

Includes:
1. HoVer-Net style losses (NP, HV, Type)
2. LDAM Loss (Label-Distribution-Aware Margin) for class imbalance
3. DRW (Deferred Re-Weighting) schedule
4. Dice Loss for segmentation
5. Combined loss function for training

Reference:
- HoVer-Net: "Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"
- LDAM-DRW: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (Cao et al., NeurIPS 2019)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple


# =============================================================================
# Dice Loss for Segmentation
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions after softmax/sigmoid [B, C, H, W]
            target: One-hot encoded targets [B, C, H, W]
        
        Returns:
            Dice loss
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
        target_flat = target.view(target.size(0), target.size(1), -1)  # [B, C, H*W]
        
        # Compute dice per class
        intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
        cardinality = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # [B, C]
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # [B, C]
        dice_loss = 1.0 - dice_score
        
        if self.reduction == 'none':
            return dice_loss
        elif self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class BinaryDiceLoss(nn.Module):
    """
    Binary Dice Loss for single-channel segmentation.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, H, W] or [B, 1, H, W]
            target: Binary targets [B, H, W] or [B, 1, H, W]
        """
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        cardinality = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


# =============================================================================
# HoVer-Net Style Losses
# =============================================================================

class NPLoss(nn.Module):
    """
    Nuclei Presence (NP) Loss for binary nuclei segmentation.
    
    Combines:
    - Binary Cross Entropy (pixel-wise classification)
    - Dice Loss (region overlap)
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            use_focal: Whether to use Focal Loss instead of BCE
            focal_gamma: Gamma parameter for Focal Loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        
        self.bce = nn.CrossEntropyLoss(reduction='mean')
        self.dice = DiceLoss(reduction='mean')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: NP predictions [B, 2, H, W] (logits)
            target: NP targets [B, H, W] (class indices: 0=background, 1=nuclei)
            mask: Optional mask for valid regions [B, H, W]
        
        Returns:
            Total NP loss and dict of component losses
        """
        B, C, H, W = pred.shape
        assert C == 2, "NP prediction should have 2 channels (background, nuclei)"
        
        # Ensure target has correct shape [B, H, W]
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        # BCE Loss
        if self.use_focal:
            ce = F.cross_entropy(pred, target.long(), reduction='none')
            pt = torch.exp(-ce)
            bce_loss = ((1 - pt) ** self.focal_gamma * ce).mean()
        else:
            bce_loss = self.bce(pred, target.long())
        
        # Dice Loss
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), num_classes=2).permute(0, 3, 1, 2).float()
        dice_loss = self.dice(pred_softmax, target_onehot)
        
        # Combined loss
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'np_bce': bce_loss.item(),
            'np_dice': dice_loss.item(),
            'np_total': total_loss.item()
        }
        
        return total_loss, loss_dict


class HVLoss(nn.Module):
    """
    Horizontal-Vertical (HV) Map Loss for distance regression.
    
    Uses:
    - MSE or Smooth L1 for regression
    - Gradient loss for better edge preservation (optional)
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        msge_weight: float = 1.0,
        use_smooth_l1: bool = False,
    ):
        """
        Args:
            mse_weight: Weight for MSE loss
            msge_weight: Weight for Mean Squared Gradient Error
            use_smooth_l1: Use Smooth L1 instead of MSE
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.msge_weight = msge_weight
        self.use_smooth_l1 = use_smooth_l1
        
        if use_smooth_l1:
            self.regression_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.regression_loss = nn.MSELoss(reduction='none')
        
        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
    
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradients using Sobel filters."""
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Process each channel
        grads = []
        for c in range(C):
            xc = x[:, c:c+1, :, :]  # [B, 1, H, W]
            grad_x = F.conv2d(xc, self.sobel_x.to(x.device), padding=1)
            grad_y = F.conv2d(xc, self.sobel_y.to(x.device), padding=1)
            grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            grads.append(grad)
        
        return torch.cat(grads, dim=1)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        focus_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: HV predictions [B, 2, H, W] (H and V maps)
            target: HV targets [B, 2, H, W]
            focus_mask: Focus on nuclei regions [B, 1, H, W] or [B, H, W]
        
        Returns:
            Total HV loss and dict of component losses
        """
        B, C, H, W = pred.shape
        assert C == 2, "HV prediction should have 2 channels (H and V)"
        
        # Apply focus mask if provided
        if focus_mask is not None:
            if focus_mask.dim() == 3:
                focus_mask = focus_mask.unsqueeze(1)  # [B, 1, H, W]
            focus_mask = focus_mask.expand_as(pred).float()
            
            # MSE/L1 loss only on nuclei regions
            regression_loss = self.regression_loss(pred, target)
            mse_loss = (regression_loss * focus_mask).sum() / (focus_mask.sum() + 1e-8)
        else:
            mse_loss = self.regression_loss(pred, target).mean()
        
        # MSGE (Mean Squared Gradient Error) Loss
        if self.msge_weight > 0:
            pred_grad = self._compute_gradient(pred)
            target_grad = self._compute_gradient(target)
            
            if focus_mask is not None:
                # Erode mask slightly for gradient computation
                focus_mask_grad = focus_mask
                grad_diff = (pred_grad - target_grad) ** 2
                msge_loss = (grad_diff * focus_mask_grad).sum() / (focus_mask_grad.sum() + 1e-8)
            else:
                msge_loss = ((pred_grad - target_grad) ** 2).mean()
        else:
            msge_loss = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.msge_weight * msge_loss
        
        loss_dict = {
            'hv_mse': mse_loss.item(),
            'hv_msge': msge_loss.item(),
            'hv_total': total_loss.item()
        }
        
        return total_loss, loss_dict


class TypeLoss(nn.Module):
    """
    Nucleus Type Classification Loss.
    
    Combines:
    - Cross Entropy variant (pixel-wise classification)
    - Dice Loss (class overlap)
    
    Supports multiple loss types for ablation studies:
    - 'ce': Standard Cross-Entropy
    - 'weighted_ce': Weighted Cross-Entropy (class weights)
    - 'focal': Focal Loss
    - 'weighted_focal': Weighted Focal CE (RECOMMENDED)
    - 'ldam': Label-Distribution-Aware Margin Loss
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        # Loss type selection (for ablation studies)
        loss_type: str = 'weighted_focal',  # 'ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam'
        focal_gamma: float = 2.0,
        # LDAM parameters (legacy, use loss_type='ldam')
        use_ldam: bool = False,  # Deprecated: use loss_type='ldam'
        cls_num_list: Optional[List[int]] = None,
        ldam_max_m: float = 0.5,
        ldam_s: float = 30.0,
    ):
        """
        Args:
            num_classes: Number of nucleus types (including background)
            ce_weight: Weight for CE/Focal loss term
            dice_weight: Weight for Dice loss term
            class_weights: Per-class weights (for weighted variants)
            loss_type: Type of CE loss ('ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam')
            focal_gamma: Gamma for focal loss (only used if loss_type contains 'focal')
            use_ldam: Deprecated - use loss_type='ldam' instead
            cls_num_list: Class frequencies for LDAM
            ldam_max_m, ldam_s: LDAM parameters
        """
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        
        # Handle legacy use_ldam parameter
        if use_ldam and loss_type != 'ldam':
            loss_type = 'ldam'
            self.loss_type = 'ldam'
        
        # Default class weights for PanNuke if not provided
        if class_weights is None and loss_type in ['weighted_ce', 'weighted_focal']:
            class_weights = get_pannuke_class_weights()
        
        # Initialize the appropriate loss function
        if loss_type == 'ce':
            self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == 'weighted_ce':
            self.ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        elif loss_type == 'focal':
            self.ce_loss_fn = FocalLoss(gamma=focal_gamma, alpha=None, reduction='mean')
        elif loss_type == 'weighted_focal':
            self.ce_loss_fn = WeightedFocalCELoss(
                num_classes=num_classes,
                gamma=focal_gamma,
                class_weights=class_weights,
                reduction='mean'
            )
        elif loss_type == 'ldam':
            if cls_num_list is None:
                cls_num_list = get_class_frequencies_pannuke()
            self.ce_loss_fn = LDAMLoss(
                cls_num_list=cls_num_list,
                max_m=ldam_max_m,
                s=ldam_s,
                weight=class_weights
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of: 'ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam'")
        
        self.dice = DiceLoss(reduction='mean')
        
        # Store class weights for DRW updates
        self._class_weights = class_weights
    
    def update_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights (for DRW schedule)."""
        if weights is None:
            return
        
        if self.loss_type == 'weighted_ce':
            self.ce_loss_fn.weight = weights
        elif self.loss_type == 'weighted_focal':
            self.ce_loss_fn.update_weights(weights)
        elif self.loss_type == 'ldam':
            self.ce_loss_fn.weight = weights
        # 'ce' and 'focal' don't use class weights
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        focus_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: Type predictions [B, num_classes, H, W] (logits)
            target: Type targets [B, H, W] (class indices)
            focus_mask: Focus on nuclei regions [B, H, W]
        
        Returns:
            Total Type loss and dict of component losses
        """
        B, C, H, W = pred.shape
        assert C == self.num_classes, f"Type prediction should have {self.num_classes} channels"
        
        # Reshape for loss computation
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        target_flat = target.reshape(-1).long()  # [B*H*W]
        
        # Apply focus mask if provided (only compute loss on nuclei pixels)
        if focus_mask is not None:
            mask_flat = focus_mask.reshape(-1).bool()  # [B*H*W]
            if mask_flat.sum() > 0:
                pred_flat_masked = pred_flat[mask_flat]
                target_flat_masked = target_flat[mask_flat]
            else:
                # No valid pixels, return zero loss
                zero = torch.tensor(0.0, device=pred.device)
                return zero, {'type_ce': 0.0, 'type_dice': 0.0, 'type_total': 0.0, 'type_loss_type': self.loss_type}
        else:
            pred_flat_masked = pred_flat
            target_flat_masked = target_flat
        
        # CE/Focal/LDAM Loss
        ce_loss = self.ce_loss_fn(pred_flat_masked, target_flat_masked)
        
        # Dice Loss (on full spatial resolution)
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        if focus_mask is not None:
            # Ensure focus_mask is [B, H, W]
            while focus_mask.dim() > 3:
                focus_mask = focus_mask.squeeze(1)
            # Apply mask to dice computation
            focus_mask_expanded = focus_mask.unsqueeze(1).expand_as(pred_softmax).float()
            pred_masked = pred_softmax * focus_mask_expanded
            target_masked = target_onehot * focus_mask_expanded
            dice_loss = self.dice(pred_masked, target_masked)
        else:
            dice_loss = self.dice(pred_softmax, target_onehot)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'type_ce': ce_loss.item(),
            'type_dice': dice_loss.item(),
            'type_total': total_loss.item(),
            'type_loss_type': self.loss_type,
        }
        
        return total_loss, loss_dict


# =============================================================================
# LDAM Loss (Label-Distribution-Aware Margin Loss)
# =============================================================================

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss.
    
    From: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    (Cao et al., NeurIPS 2019)
    
    The key idea is to add class-dependent margins to enforce larger margins
    for minority classes. The margin for class j is: m_j = C / n_j^(1/4)
    where n_j is the number of samples in class j.
    """
    
    def __init__(
        self,
        cls_num_list: List[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            cls_num_list: List of number of samples per class
            max_m: Maximum margin value (margins are scaled to this)
            s: Scale factor for logits
            weight: Optional per-class weights for additional reweighting
        """
        super().__init__()
        
        # Compute margins: m_j = C / n_j^(1/4)
        cls_num_array = np.array(cls_num_list, dtype=np.float32)
        # Avoid division by zero for classes with no samples
        cls_num_array = np.maximum(cls_num_array, 1)
        
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_array))  # 1 / n_j^(1/4)
        m_list = m_list * (max_m / np.max(m_list))  # Scale to max_m
        
        self.register_buffer('m_list', torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.weight = weight
        self.num_classes = len(cls_num_list)
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Logits [N, C] where N is batch size, C is num classes
            target: Class labels [N]
        
        Returns:
            LDAM loss value
        """
        # Create one-hot encoding
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1).long(), True)
        
        # Get margin for each sample based on its target class
        index_float = index.float()
        # batch_m: [N, 1] - margin for each sample
        batch_m = torch.matmul(self.m_list.unsqueeze(0), index_float.t()).t()
        
        # Subtract margin from correct class logits
        x_m = x - batch_m
        
        # Apply margin only to correct class
        output = torch.where(index, x_m, x)
        
        # Scale and compute cross entropy
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    From: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
    
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma: Focusing parameter (0 = CE, higher = more focus on hard examples)
            alpha: Per-class weights
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits [N, C]
            target: Class labels [N]
        """
        ce = F.cross_entropy(pred, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma) * ce
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# =============================================================================
# Text-to-Pixel Contrastive Loss (from CRIS paper)
# =============================================================================

class TextPixelContrastiveLoss(nn.Module):
    """
    Text-to-Pixel Contrastive Loss for better text-visual grounding.
    
    From: "CLIP-Driven Referring Image Segmentation" (CRIS, CVPR 2022)
    
    This loss aligns text embeddings with positive pixel features and pushes
    away from negative pixels. It improves the model's ability to ground
    text descriptions to specific image regions.
    
    Formula:
        L_contrast = -log(exp(sim(t, v+)/τ) / Σ exp(sim(t, v)/τ))
    
    where:
        - t: text embedding
        - v+: positive visual feature (text-matching regions)
        - τ: temperature parameter
    
    For nuclei segmentation:
        - Positive: pixels belonging to the target nucleus type
        - Negative: pixels belonging to background or other nucleus types
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean'
    ):
        """
        Args:
            temperature: Temperature parameter τ for scaling similarities
                        Lower = sharper distribution, typical values: 0.05-0.1
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        visual_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute text-to-visual contrastive loss.
        
        This version computes cross-sample contrastive loss where:
        - Each text embedding should be close to its corresponding visual embedding
        - Each text embedding should be far from other visual embeddings in the batch
        
        Args:
            visual_embed: [B, D] L2-normalized visual embeddings (pooled from segmentation features)
            text_embed: [B, D] L2-normalized text embeddings
            
        Returns:
            Contrastive loss value
        """
        B, D = visual_embed.shape
        
        # Ensure inputs are normalized (should already be, but just in case)
        visual_embed = F.normalize(visual_embed, p=2, dim=-1)
        text_embed = F.normalize(text_embed, p=2, dim=-1)
        
        # Compute similarity matrix: [B, B]
        # sim[i, j] = cosine similarity between text[i] and visual[j]
        sim_matrix = torch.matmul(text_embed, visual_embed.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=visual_embed.device)
        
        # Compute cross-entropy loss (text-to-visual direction)
        loss_t2v = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)
        
        # Compute cross-entropy loss (visual-to-text direction)  
        loss_v2t = F.cross_entropy(sim_matrix.t(), labels, reduction=self.reduction)
        
        # Symmetric loss
        loss = (loss_t2v + loss_v2t) / 2
        
        return loss


class TextPixelContrastiveLossV2(nn.Module):
    """
    Enhanced Text-to-Pixel Contrastive Loss with pixel-level supervision.
    
    This version uses the segmentation mask to define positive/negative pixels:
    - Positive pixels: pixels belonging to the target nucleus type
    - Negative pixels: background or other nucleus types
    
    The loss encourages text embeddings to be close to positive pixel features
    and far from negative pixel features.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        num_samples: int = 256,  # Number of pixels to sample per image
        reduction: str = 'mean'
    ):
        """
        Args:
            temperature: Temperature parameter τ
            num_samples: Number of pixels to sample for efficiency
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.temperature = temperature
        self.num_samples = num_samples
        self.reduction = reduction
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_embed: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pixel-level contrastive loss.
        
        Args:
            visual_features: [B, D, H, W] visual feature map from decoder
            text_embed: [B, D] L2-normalized text embedding
            target_mask: [B, H, W] binary mask (1 = target class, 0 = non-target)
            
        Returns:
            Contrastive loss value
        """
        B, D, H, W = visual_features.shape
        device = visual_features.device
        
        # Normalize text embedding
        text_embed = F.normalize(text_embed, p=2, dim=-1)  # [B, D]
        
        # Reshape visual features: [B, D, H*W]
        visual_flat = visual_features.view(B, D, -1)
        
        # Normalize visual features along channel dimension
        visual_flat = F.normalize(visual_flat, p=2, dim=1)  # [B, D, H*W]
        
        # Reshape mask: [B, H*W]
        mask_flat = target_mask.view(B, -1).float()
        
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(B):
            # Get positive and negative indices
            pos_indices = (mask_flat[b] > 0).nonzero(as_tuple=True)[0]
            neg_indices = (mask_flat[b] == 0).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            # Sample pixels for efficiency
            n_pos = min(len(pos_indices), self.num_samples // 2)
            n_neg = min(len(neg_indices), self.num_samples // 2)
            
            pos_sample = pos_indices[torch.randperm(len(pos_indices), device=device)[:n_pos]]
            neg_sample = neg_indices[torch.randperm(len(neg_indices), device=device)[:n_neg]]
            
            # Get visual features for sampled pixels
            pos_features = visual_flat[b, :, pos_sample].t()  # [n_pos, D]
            neg_features = visual_flat[b, :, neg_sample].t()  # [n_neg, D]
            
            # Compute similarities with text embedding
            text_b = text_embed[b].unsqueeze(0)  # [1, D]
            
            pos_sim = torch.matmul(text_b, pos_features.t()) / self.temperature  # [1, n_pos]
            neg_sim = torch.matmul(text_b, neg_features.t()) / self.temperature  # [1, n_neg]
            
            # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # Average over positive samples
            for i in range(n_pos):
                pos_score = pos_sim[0, i]  # scalar
                all_scores = torch.cat([pos_score.unsqueeze(0), neg_sim[0]])  # [1 + n_neg]
                loss_i = -pos_score + torch.logsumexp(all_scores, dim=0)
                total_loss = total_loss + loss_i
                valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if self.reduction == 'mean':
            return total_loss / valid_samples
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss / valid_samples


# =============================================================================
# Weighted Focal Cross-Entropy Loss (RECOMMENDED FOR SEGMENTATION)
# =============================================================================

class WeightedFocalCELoss(nn.Module):
    """
    Weighted Focal Cross-Entropy Loss.
    
    Combines:
    - Class weights (inverse frequency) for imbalance
    - Focal term (1-p)^gamma for hard example mining
    
    Formula: L = -sum_c w_c * (1 - p_c)^gamma * log(p_c)
    
    This is the RECOMMENDED loss for dense segmentation tasks with class imbalance.
    More stable than LDAM for pixel-wise prediction.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        """
        Args:
            num_classes: Number of classes
            gamma: Focusing parameter (0 = weighted CE, 2.0 recommended)
            class_weights: Per-class weights [C]. If None, uses inverse sqrt frequency for PanNuke.
            reduction: 'none', 'mean', or 'sum'
            ignore_index: Label to ignore
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Default class weights for PanNuke (inverse sqrt frequency, normalized)
        # Background=0 (ignored), Neoplastic=1.0, Inflammatory=1.29, Connective=1.41, Dead=3.16, Epithelial=1.05
        if class_weights is None:
            class_weights = torch.tensor([
                0.0,    # Background (will be ignored in focus mask)
                1.0,    # Neoplastic (50K) - baseline
                1.29,   # Inflammatory (30K)
                1.41,   # Connective (25K)
                3.16,   # Dead (5K) - highest weight!
                1.05,   # Epithelial (45K)
            ], dtype=torch.float32)
        
        self.register_buffer('class_weights', class_weights)
    
    def update_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights (for DRW schedule)."""
        if weights is not None:
            self.class_weights = weights.to(self.class_weights.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits [N, C] or [B, C, H, W]
            target: Class labels [N] or [B, H, W]
        
        Returns:
            Weighted focal CE loss
        """
        # Handle spatial inputs
        if pred.dim() == 4:
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            target = target.reshape(-1)  # [B*H*W]
        
        # Filter valid targets
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Compute cross-entropy per sample
        ce = F.cross_entropy(pred, target, weight=self.class_weights, reduction='none')
        
        # Compute focal weight: (1 - p_t)^gamma
        log_pt = F.log_softmax(pred, dim=1)
        pt = torch.exp(log_pt.gather(1, target.unsqueeze(1)).squeeze(1))
        focal_weight = (1 - pt) ** self.gamma
        
        # Combine
        focal_loss = focal_weight * ce
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def get_pannuke_class_weights(normalize: bool = True) -> torch.Tensor:
    """
    Get class weights for PanNuke dataset based on inverse sqrt frequency.
    
    Approximate class frequencies:
    - Background: ~1M pixels (ignored)
    - Neoplastic: ~50K nuclei
    - Inflammatory: ~30K nuclei  
    - Connective: ~25K nuclei
    - Dead: ~5K nuclei (RARE!)
    - Epithelial: ~45K nuclei
    
    Returns:
        Class weights tensor [6]
    """
    # Approximate frequencies
    frequencies = torch.tensor([
        1000000.0,  # Background
        50000.0,    # Neoplastic
        30000.0,    # Inflammatory
        25000.0,    # Connective
        5000.0,     # Dead (rare!)
        45000.0,    # Epithelial
    ])
    
    # Inverse sqrt frequency
    weights = 1.0 / torch.sqrt(frequencies)
    
    # Normalize to Neoplastic = 1.0
    if normalize:
        weights = weights / weights[1]
    
    # Background gets 0 weight (computed on focus mask anyway)
    weights[0] = 0.0
    
    return weights


# =============================================================================
# DRW (Deferred Re-Weighting) Schedule
# =============================================================================

class DRWScheduler:
    """
    Deferred Re-Weighting (DRW) Schedule.
    
    Trains with uniform weights initially, then switches to class-balanced
    weights after a certain epoch. This allows the model to learn good
    representations first, then fine-tune for class balance.
    
    From: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    (Cao et al., NeurIPS 2019)
    """
    
    def __init__(
        self,
        cls_num_list: List[int],
        total_epochs: int,
        drw_start_epoch: Optional[int] = None,
        drw_start_ratio: float = 0.8,
        beta: float = 0.9999,
        device: str = 'cuda'
    ):
        """
        Args:
            cls_num_list: List of number of samples per class
            total_epochs: Total number of training epochs
            drw_start_epoch: Epoch to start DRW (if None, uses drw_start_ratio * total_epochs)
            drw_start_ratio: Ratio of epochs before DRW starts (default 80%)
            beta: Beta parameter for effective number computation
            device: Device for tensors
        """
        self.cls_num_list = np.array(cls_num_list, dtype=np.float32)
        self.total_epochs = total_epochs
        self.beta = beta
        self.device = device
        
        if drw_start_epoch is not None:
            self.drw_start_epoch = drw_start_epoch
        else:
            self.drw_start_epoch = int(drw_start_ratio * total_epochs)
        
        # Compute class-balanced weights using effective number
        # Effective number: (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - np.power(self.beta, self.cls_num_list)
        per_cls_weights = (1.0 - self.beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        
        self.uniform_weights = torch.ones(len(cls_num_list), dtype=torch.float32)
        self.balanced_weights = torch.tensor(per_cls_weights, dtype=torch.float32)
    
    def get_weights(self, epoch: int) -> torch.Tensor:
        """
        Get class weights for the current epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
        
        Returns:
            Class weights tensor
        """
        if epoch < self.drw_start_epoch:
            return self.uniform_weights.to(self.device)
        else:
            return self.balanced_weights.to(self.device)
    
    def is_drw_active(self, epoch: int) -> bool:
        """Check if DRW is active at the given epoch."""
        return epoch >= self.drw_start_epoch
    
    def __repr__(self) -> str:
        return (
            f"DRWScheduler(n_classes={len(self.cls_num_list)}, "
            f"drw_start_epoch={self.drw_start_epoch}, "
            f"total_epochs={self.total_epochs})"
        )


# =============================================================================
# Combined HoVer-Net Loss
# =============================================================================

class HoVerNetLoss(nn.Module):
    """
    Combined loss for HoVer-Net style training.
    
    Total Loss = w_np * L_np + w_hv * L_hv + w_type * L_type
    
    Where:
    - L_np: Nuclei Presence loss (BCE/Focal + Dice)
    - L_hv: HV map loss (MSE + MSGE)
    - L_type: Type classification loss (CE/Focal/LDAM + Dice)
    
    Supports multiple loss types for ablation:
    - 'ce': Standard Cross-Entropy
    - 'weighted_ce': Weighted Cross-Entropy
    - 'focal': Focal Loss
    - 'weighted_focal': Weighted Focal CE (RECOMMENDED)
    - 'ldam': LDAM Loss
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        np_weight: float = 1.0,
        hv_weight: float = 1.0,
        type_weight: float = 1.0,
        # NP loss params
        np_bce_weight: float = 1.0,
        np_dice_weight: float = 1.0,
        np_use_focal: bool = False,  # Use focal for NP loss
        np_focal_gamma: float = 2.0,
        # HV loss params
        hv_mse_weight: float = 1.0,
        hv_msge_weight: float = 1.0,
        # Type loss params
        type_ce_weight: float = 1.0,
        type_dice_weight: float = 1.0,
        type_class_weights: Optional[torch.Tensor] = None,
        # Loss type selection (NEW - for ablation studies)
        loss_type: str = 'weighted_focal',  # 'ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam'
        focal_gamma: float = 2.0,
        # LDAM params (legacy)
        use_ldam: bool = False,  # Deprecated: use loss_type='ldam'
        cls_num_list: Optional[List[int]] = None,
        ldam_max_m: float = 0.5,
        ldam_s: float = 30.0,
    ):
        """
        Args:
            num_classes: Number of nucleus types (including background)
            np_weight: Weight for NP loss
            hv_weight: Weight for HV loss
            type_weight: Weight for Type loss
            np_bce_weight, np_dice_weight: Weights within NP loss
            np_use_focal: Use focal loss for NP (binary segmentation)
            np_focal_gamma: Gamma for NP focal loss
            hv_mse_weight, hv_msge_weight: Weights within HV loss
            type_ce_weight, type_dice_weight: Weights within Type loss
            type_class_weights: Per-class weights for Type loss
            loss_type: Type of CE loss ('ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam')
            focal_gamma: Gamma for focal loss (Type)
            use_ldam: Deprecated - use loss_type='ldam'
            cls_num_list: Class frequencies for LDAM
            ldam_max_m, ldam_s: LDAM parameters
        """
        super().__init__()
        
        self.np_weight = np_weight
        self.hv_weight = hv_weight
        self.type_weight = type_weight
        self.loss_type = loss_type
        
        # Handle legacy use_ldam
        if use_ldam and loss_type != 'ldam':
            loss_type = 'ldam'
            self.loss_type = 'ldam'
        
        self.np_loss = NPLoss(
            bce_weight=np_bce_weight,
            dice_weight=np_dice_weight,
            use_focal=np_use_focal or ('focal' in loss_type),  # Enable focal if globally enabled
            focal_gamma=np_focal_gamma,
        )
        
        self.hv_loss = HVLoss(
            mse_weight=hv_mse_weight,
            msge_weight=hv_msge_weight
        )
        
        self.type_loss = TypeLoss(
            num_classes=num_classes,
            ce_weight=type_ce_weight,
            dice_weight=type_dice_weight,
            class_weights=type_class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            use_ldam=use_ldam,  # For backward compatibility
            cls_num_list=cls_num_list,
            ldam_max_m=ldam_max_m,
            ldam_s=ldam_s
        )
    
    def update_type_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for Type loss (used with DRW)."""
        self.type_loss.update_weights(weights)
    
    def forward(
        self,
        pred_np: torch.Tensor,
        pred_hv: torch.Tensor,
        pred_type: torch.Tensor,
        target_np: torch.Tensor,
        target_hv: torch.Tensor,
        target_type: torch.Tensor,
        focus_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined HoVer-Net loss.
        
        Args:
            pred_np: NP predictions [B, 2, H, W]
            pred_hv: HV predictions [B, 2, H, W]
            pred_type: Type predictions [B, num_classes, H, W]
            target_np: NP targets [B, H, W]
            target_hv: HV targets [B, 2, H, W]
            target_type: Type targets [B, H, W]
            focus_mask: Nuclei mask for focused loss computation [B, H, W]
        
        Returns:
            Total loss and dict of all component losses
        """
        # NP Loss
        np_loss, np_dict = self.np_loss(pred_np, target_np)
        
        # HV Loss (focus on nuclei regions)
        hv_loss, hv_dict = self.hv_loss(pred_hv, target_hv, focus_mask)
        
        # Type Loss (focus on nuclei regions)
        type_loss, type_dict = self.type_loss(pred_type, target_type, focus_mask)
        
        # Combined loss
        total_loss = (
            self.np_weight * np_loss +
            self.hv_weight * hv_loss +
            self.type_weight * type_loss
        )
        
        # Combine all loss dicts
        loss_dict = {
            **np_dict,
            **hv_dict,
            **type_dict,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


# =============================================================================
# Factory Functions
# =============================================================================

def create_hovernet_loss(
    num_classes: int = 6,
    cls_num_list: Optional[List[int]] = None,
    use_ldam: bool = True,
    np_weight: float = 1.0,
    hv_weight: float = 2.0,
    type_weight: float = 1.0,
) -> HoVerNetLoss:
    """
    Create HoVer-Net loss with recommended settings.
    
    Args:
        num_classes: Number of nucleus types
        cls_num_list: Class frequencies for LDAM
        use_ldam: Whether to use LDAM for type classification
        np_weight, hv_weight, type_weight: Loss weights
    
    Returns:
        HoVerNetLoss instance
    """
    return HoVerNetLoss(
        num_classes=num_classes,
        np_weight=np_weight,
        hv_weight=hv_weight,
        type_weight=type_weight,
        use_ldam=use_ldam and cls_num_list is not None,
        cls_num_list=cls_num_list,
    )


def create_drw_scheduler(
    cls_num_list: List[int],
    total_epochs: int = 50,
    drw_start_ratio: float = 0.6,
    device: str = 'cuda'
) -> DRWScheduler:
    """
    Create DRW scheduler with recommended settings.
    
    Default: 50 epochs total, DRW starts at epoch 30 (60%)
    - Epochs 0-29: Uniform weights (learn good representations)
    - Epochs 30-49: Class-balanced weights (fine-tune for balance)
    
    Args:
        cls_num_list: Class frequencies
        total_epochs: Total training epochs (default: 50)
        drw_start_ratio: When to start DRW (as fraction of total epochs)
        device: Device for tensors
    
    Returns:
        DRWScheduler instance
    """
    return DRWScheduler(
        cls_num_list=cls_num_list,
        total_epochs=total_epochs,
        drw_start_ratio=drw_start_ratio,
        device=device
    )


def get_class_frequencies_pannuke() -> List[int]:
    """
    Get approximate class frequencies for PanNuke dataset.
    
    Classes:
    0: Background (most frequent)
    1: Neoplastic
    2: Inflammatory
    3: Connective/Soft tissue
    4: Dead
    5: Epithelial
    
    Note: These are rough estimates. For exact values, compute from actual data.
    """
    # Approximate frequencies (adjust based on actual dataset analysis)
    # These are placeholder values - should be computed from actual data
    return [
        1000000,  # Background (very frequent)
        50000,    # Neoplastic
        30000,    # Inflammatory
        25000,    # Connective/Soft tissue
        5000,     # Dead (rare)
        45000,    # Epithelial
    ]


# =============================================================================
# Loss for Different Model Variants
# =============================================================================

class CIPSNetV2Loss(nn.Module):
    """
    Combined loss for CIPS-Net V2 training.
    
    Supports all model variants:
    - BASELINE: Standard HoVer-Net loss
    - WITH_TEXT: HoVer-Net loss + text-guided losses
    - WITH_CGR: HoVer-Net loss
    - WITH_TEXT_CONDITIONED_TYPE: HoVer-Net loss with text-conditioned type head
    - FULL: All losses combined
    
    Supports multiple loss types for ablation studies:
    - 'ce': Standard Cross-Entropy
    - 'weighted_ce': Weighted Cross-Entropy
    - 'focal': Focal Loss (unweighted)
    - 'weighted_focal': Weighted Focal CE (RECOMMENDED)
    - 'ldam': LDAM Loss
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        cls_num_list: Optional[List[int]] = None,
        # Loss weights
        np_weight: float = 1.0,
        hv_weight: float = 2.0,
        type_weight: float = 2.0,  # Increased for better type classification
        # Loss type selection (NEW - for ablation studies)
        loss_type: str = 'weighted_focal',  # 'ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam'
        focal_gamma: float = 2.0,
        # Class weights
        use_class_weights: bool = True,
        # Legacy LDAM parameter
        use_ldam: bool = False,  # Deprecated: use loss_type='ldam'
        ldam_max_m: float = 0.5,
        ldam_s: float = 30.0,
        # Optional text-guided loss
        text_consistency_weight: float = 0.0,
    ):
        """
        Args:
            num_classes: Number of nucleus types
            cls_num_list: Class frequencies for LDAM/DRW
            np_weight, hv_weight, type_weight: Main loss weights
            loss_type: Type of classification loss ('ce', 'weighted_ce', 'focal', 'weighted_focal', 'ldam')
            focal_gamma: Gamma for focal loss
            use_class_weights: Whether to use class weights (for weighted variants)
            use_ldam: Deprecated - use loss_type='ldam' instead
            ldam_max_m, ldam_s: LDAM parameters
            text_consistency_weight: Weight for text consistency loss
        """
        super().__init__()
        
        self.loss_type = loss_type
        
        # Handle legacy use_ldam parameter
        if use_ldam and loss_type not in ['ldam']:
            loss_type = 'ldam'
            self.loss_type = 'ldam'
        
        # Get class weights if needed
        class_weights = None
        if use_class_weights and loss_type in ['weighted_ce', 'weighted_focal']:
            class_weights = get_pannuke_class_weights()
        
        self.hover_loss = HoVerNetLoss(
            num_classes=num_classes,
            np_weight=np_weight,
            hv_weight=hv_weight,
            type_weight=type_weight,
            type_class_weights=class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            use_ldam=(loss_type == 'ldam'),
            cls_num_list=cls_num_list if cls_num_list else get_class_frequencies_pannuke(),
            ldam_max_m=ldam_max_m,
            ldam_s=ldam_s,
        )
        
        self.text_consistency_weight = text_consistency_weight
        
        # Log loss configuration
        print(f"  [Loss] Type: {loss_type}, Focal γ={focal_gamma}, "
              f"Weights: NP={np_weight}, HV={hv_weight}, Type={type_weight}")
    
    def update_type_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for DRW."""
        self.hover_loss.update_type_weights(weights)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for CIPS-Net V2.
        
        Args:
            outputs: Model outputs dict with keys:
                - 'np': [B, 2, H, W]
                - 'hv': [B, 2, H, W]
                - 'type': [B, num_classes, H, W]
            targets: Target dict with keys:
                - 'np': [B, H, W]
                - 'hv': [B, 2, H, W]
                - 'type': [B, H, W]
                - 'focus_mask': [B, H, W] (optional)
        
        Returns:
            Total loss and loss dict
        """
        # Get focus mask (nuclei regions)
        focus_mask = targets.get('focus_mask', None)
        if focus_mask is None:
            # Use NP target as focus mask (where nuclei exist)
            np_target = targets['np']
            if np_target.dim() == 4:
                np_target = np_target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            focus_mask = (np_target > 0).float()
        
        # Compute HoVer-Net loss
        total_loss, loss_dict = self.hover_loss(
            pred_np=outputs['np'],
            pred_hv=outputs['hv'],
            pred_type=outputs['type'],
            target_np=targets['np'],
            target_hv=targets['hv'],
            target_type=targets['type'],
            focus_mask=focus_mask
        )
        
        # Add loss type info
        loss_dict['loss_type'] = self.loss_type
        
        return total_loss, loss_dict


# =============================================================================
# Testing
# =============================================================================

def test_losses():
    """Test all loss functions."""
    print("Testing Loss Functions...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 2, 256, 256
    num_classes = 6
    
    # Create dummy predictions and targets (with requires_grad for gradient test)
    pred_np = torch.randn(B, 2, H, W, device=device, requires_grad=True)
    pred_hv = torch.randn(B, 2, H, W, device=device, requires_grad=True)
    pred_type = torch.randn(B, num_classes, H, W, device=device, requires_grad=True)
    
    target_np = torch.randint(0, 2, (B, H, W), device=device)
    target_hv = torch.randn(B, 2, H, W, device=device) * 0.5
    target_type = torch.randint(0, num_classes, (B, H, W), device=device)
    focus_mask = (target_np > 0).float()
    
    print(f"Device: {device}")
    print(f"Batch size: {B}, Spatial: {H}x{W}, Classes: {num_classes}")
    print()
    
    # Test NP Loss
    print("1. Testing NPLoss...")
    np_loss_fn = NPLoss().to(device)
    np_loss, np_dict = np_loss_fn(pred_np, target_np)
    print(f"   NP Loss: {np_loss.item():.4f}")
    print(f"   Components: {np_dict}")
    print()
    
    # Test HV Loss
    print("2. Testing HVLoss...")
    hv_loss_fn = HVLoss().to(device)
    hv_loss, hv_dict = hv_loss_fn(pred_hv, target_hv, focus_mask)
    print(f"   HV Loss: {hv_loss.item():.4f}")
    print(f"   Components: {hv_dict}")
    print()
    
    # Test Type Loss (standard)
    print("3. Testing TypeLoss (standard CE)...")
    type_loss_fn = TypeLoss(num_classes=num_classes).to(device)
    type_loss, type_dict = type_loss_fn(pred_type, target_type, focus_mask)
    print(f"   Type Loss: {type_loss.item():.4f}")
    print(f"   Components: {type_dict}")
    print()
    
    # Test Type Loss with LDAM
    print("4. Testing TypeLoss with LDAM...")
    cls_num_list = get_class_frequencies_pannuke()
    type_loss_ldam = TypeLoss(
        num_classes=num_classes,
        use_ldam=True,
        cls_num_list=cls_num_list
    ).to(device)
    ldam_loss, ldam_dict = type_loss_ldam(pred_type, target_type, focus_mask)
    print(f"   Type Loss (LDAM): {ldam_loss.item():.4f}")
    print(f"   Components: {ldam_dict}")
    print()
    
    # Test DRW Scheduler
    print("5. Testing DRW Scheduler (50 epochs, 30/20 split)...")
    drw = DRWScheduler(
        cls_num_list=cls_num_list,
        total_epochs=50,
        drw_start_ratio=0.6,
        device=str(device)
    )
    print(f"   {drw}")
    print(f"   Epoch 0: DRW active={drw.is_drw_active(0)}, weights sample={drw.get_weights(0)[:3]}")
    print(f"   Epoch 29: DRW active={drw.is_drw_active(29)}, weights sample={drw.get_weights(29)[:3]}")
    print(f"   Epoch 30: DRW active={drw.is_drw_active(30)}, weights sample={drw.get_weights(30)[:3]}")
    print()
    
    # Test Combined HoVer-Net Loss
    print("6. Testing HoVerNetLoss (combined)...")
    hover_loss = HoVerNetLoss(
        num_classes=num_classes,
        use_ldam=True,
        cls_num_list=cls_num_list
    ).to(device)
    total_loss, loss_dict = hover_loss(
        pred_np, pred_hv, pred_type,
        target_np, target_hv, target_type,
        focus_mask
    )
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   All components:")
    for k, v in loss_dict.items():
        print(f"      {k}: {v:.4f}")
    print()
    
    # Test gradient flow
    print("7. Testing gradient flow...")
    total_loss.backward()
    print("   ✓ Gradients computed successfully")
    print()
    
    # Test CIPSNetV2Loss
    print("8. Testing CIPSNetV2Loss...")
    cipsnet_loss = CIPSNetV2Loss(
        num_classes=num_classes,
        cls_num_list=cls_num_list,
        use_ldam=True
    ).to(device)
    
    outputs = {'np': pred_np, 'hv': pred_hv, 'type': pred_type}
    targets = {'np': target_np, 'hv': target_hv, 'type': target_type}
    
    total_loss2, loss_dict2 = cipsnet_loss(outputs, targets)
    print(f"   Total Loss: {total_loss2.item():.4f}")
    print()
    
    # Test weight update with DRW
    print("9. Testing DRW weight update...")
    cipsnet_loss.update_type_weights(drw.get_weights(80))
    total_loss3, _ = cipsnet_loss(outputs, targets)
    print(f"   Loss after DRW weights: {total_loss3.item():.4f}")
    print()
    
    print("=" * 60)
    print("All loss function tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_losses()
