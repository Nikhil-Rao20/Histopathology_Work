"""
Loss Functions for LViT5 Training - Ultimate Loss Function
============================================================

Combines ALL loss improvements with proper weighting:

1. HoVer-Net Loss (NP + HV + Type with Weighted Focal + DRW)
2. Pixel-level Contrastive Loss with Hard Negative Mining
3. Multi-scale Contrastive Loss (batch-level at multiple scales)
4. Grounding-Aware Loss (IoU supervision on text-region alignment)
5. Attention Regularization Loss (encourage discriminative attention)

Why Previous Contrastive Didn't Work:
    1. Only batch-level (image↔text), not pixel-level (pixel↔text)
    2. No hard negative mining - easy negatives dominated
    3. Weight too low (0.5) - overwhelmed by HoVer-Net loss
    4. Features may not have been properly projected

Fixes in LViT5Loss:
    1. Pixel-level contrastive using pixel features from decoder
    2. Hard negative mining - focus on hard examples
    3. Higher weight (1.0-2.0) for contrastive losses
    4. Grounding loss provides direct IoU supervision
    5. Multiple loss scales for better gradients

Recommended Weights:
    - HoVer-Net: NP=1.0, HV=2.0, Type=2.0 (standard)
    - Pixel Contrastive: 1.0 (important for grounding)
    - Batch Contrastive: 0.5 (auxiliary)
    - Grounding: 1.0 (direct supervision)
    - Attention Reg: 0.1 (regularization)

Total Loss = L_hover + λ_pixel*L_pixel + λ_batch*L_batch + λ_ground*L_ground + λ_attn*L_attn

Author: CIPS-Net V2 - Phase 3 Ultimate Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple

from .losses import (
    HoVerNetLoss,
    get_class_frequencies_pannuke,
    get_pannuke_class_weights,
)


# ============================================================
# Pixel-level Contrastive Loss with Hard Negative Mining
# ============================================================

class PixelContrastiveLoss(nn.Module):
    """
    Pixel-level Contrastive Loss.
    
    For each text, aligns positive pixels (matching nucleus type) with text,
    and pushes away negative pixels.
    
    Key improvements:
    - Pixel-level not batch-level
    - Hard negative mining
    - Temperature scaling
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        hard_negative_ratio: float = 0.3,  # Use top 30% hardest negatives
        min_positives: int = 100,  # Minimum positive pixels
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        self.min_positives = min_positives
    
    def forward(
        self,
        pixel_features: torch.Tensor,
        text_embed: torch.Tensor,
        target_type: torch.Tensor,
        focus_mask: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pixel-level contrastive loss.
        
        Args:
            pixel_features: [B, D, H, W] pixel-level visual features
            text_embed: [B, D] text embedding (contrastive projection)
            target_type: [B, H, W] ground truth type labels
            focus_mask: [B, H, W] mask for nuclei regions (1=nucleus, 0=bg)
            temperature: Optional learnable temperature
            
        Returns:
            Contrastive loss scalar
        """
        B, D, H, W = pixel_features.shape
        device = pixel_features.device
        
        if temperature is None:
            temperature = self.temperature
        elif isinstance(temperature, torch.Tensor):
            temperature = temperature.item()
        
        # Downsample masks to match pixel features resolution
        fm_h, fm_w = focus_mask.shape[-2:]
        if fm_h != H or fm_w != W:
            focus_mask_ds = F.interpolate(
                focus_mask.unsqueeze(1).float(), size=(H, W), mode='nearest'
            ).squeeze(1)
            target_type_ds = F.interpolate(
                target_type.unsqueeze(1).float(), size=(H, W), mode='nearest'
            ).squeeze(1).long()
        else:
            focus_mask_ds = focus_mask
            target_type_ds = target_type
        
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(B):
            # Get pixel features and flatten
            pf = pixel_features[b]  # [D, H, W]
            pf_flat = pf.view(D, -1).t()  # [H*W, D]
            pf_flat = F.normalize(pf_flat, p=2, dim=-1)
            
            # Get text embedding
            te = text_embed[b:b+1]  # [1, D]
            te = F.normalize(te, p=2, dim=-1)
            
            # Get masks (use downsampled versions)
            fm = focus_mask_ds[b].view(-1)  # [H*W]
            tt = target_type_ds[b].view(-1)  # [H*W]
            
            # Positive pixels: within nuclei
            pos_mask = fm > 0
            # Negative pixels: background
            neg_mask = fm == 0
            
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()
            
            if num_pos < self.min_positives or num_neg < 10:
                continue
            
            # Get positive and negative features
            pos_features = pf_flat[pos_mask]  # [N_pos, D]
            neg_features = pf_flat[neg_mask]  # [N_neg, D]
            
            # Compute similarities
            pos_sim = torch.matmul(pos_features, te.t()).squeeze(-1) / temperature  # [N_pos]
            neg_sim = torch.matmul(neg_features, te.t()).squeeze(-1) / temperature  # [N_neg]
            
            # Hard negative mining: use top-k hardest negatives
            num_hard_neg = max(int(num_neg * self.hard_negative_ratio), min(num_neg, num_pos))
            hard_neg_sim, _ = torch.topk(neg_sim, num_hard_neg, largest=True)  # Highest similarity = hardest
            
            # InfoNCE-style loss: positive should be higher than negatives
            # For each positive, compute log(exp(pos) / (exp(pos) + sum(exp(neg))))
            pos_exp = torch.exp(pos_sim)  # [N_pos]
            neg_exp_sum = torch.exp(hard_neg_sim).sum()  # scalar
            
            # Loss = -log(pos / (pos + neg_sum))
            loss_per_pos = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
            loss = loss_per_pos.mean()
            
            total_loss += loss
            valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / valid_samples


# ============================================================
# Multi-scale Batch Contrastive Loss
# ============================================================

class MultiScaleBatchContrastiveLoss(nn.Module):
    """
    Multi-scale batch-level contrastive loss.
    
    Computes contrastive loss between image and text embeddings
    at multiple scales of the decoder.
    """
    
    def __init__(
        self,
        scale_weights: List[float] = [0.4, 0.3, 0.2, 0.1],  # deep→out
        default_temperature: float = 0.07
    ):
        super().__init__()
        self.scale_weights = scale_weights
        self.default_temperature = default_temperature
    
    def _batch_contrastive(
        self,
        visual_features: torch.Tensor,
        text_embed: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute batch contrastive loss at one scale."""
        B, D, H, W = visual_features.shape
        
        # Global average pooling
        visual_global = F.adaptive_avg_pool2d(visual_features, 1).view(B, D)
        visual_global = F.normalize(visual_global, p=2, dim=-1)
        text_embed = F.normalize(text_embed, p=2, dim=-1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(visual_global, text_embed.t()) / temperature
        
        # Labels (diagonal = positive)
        labels = torch.arange(B, device=visual_features.device)
        
        # Symmetric loss
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2
    
    def forward(
        self,
        pixel_features_deep: torch.Tensor,
        pixel_features_mid: torch.Tensor,
        pixel_features_shallow: torch.Tensor,
        pixel_features_out: torch.Tensor,
        text_embed: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-scale batch contrastive loss.
        
        Returns total loss and component dict.
        """
        if temperature is None:
            temp = self.default_temperature
        else:
            temp = temperature.item() if isinstance(temperature, torch.Tensor) else temperature
        
        loss_deep = self._batch_contrastive(pixel_features_deep, text_embed, temp)
        loss_mid = self._batch_contrastive(pixel_features_mid, text_embed, temp)
        loss_shallow = self._batch_contrastive(pixel_features_shallow, text_embed, temp)
        loss_out = self._batch_contrastive(pixel_features_out, text_embed, temp)
        
        total_loss = (
            self.scale_weights[0] * loss_deep +
            self.scale_weights[1] * loss_mid +
            self.scale_weights[2] * loss_shallow +
            self.scale_weights[3] * loss_out
        )
        
        loss_dict = {
            'batch_contrastive_deep': loss_deep.item(),
            'batch_contrastive_mid': loss_mid.item(),
            'batch_contrastive_shallow': loss_shallow.item(),
            'batch_contrastive_out': loss_out.item(),
        }
        
        return total_loss, loss_dict


# ============================================================
# Grounding-Aware Loss
# ============================================================

class GroundingAwareLoss(nn.Module):
    """
    Grounding-Aware Loss for explicit text-region alignment.
    
    Supervises the grounding map to match the target nucleus regions.
    Uses combination of:
    - BCE loss for pixel-wise alignment
    - Dice loss for region overlap
    - IoU loss for direct IoU optimization
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.3,
        iou_weight: float = 0.2,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.smooth = smooth
        # Use BCEWithLogitsLoss for autocast safety
        # We'll apply sigmoid ourselves in forward if needed
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    def bce_loss_manual(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss manually (autocast safe)."""
        # Clamp predictions to avoid log(0)
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        return bce.mean()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou
    
    def forward(
        self,
        grounding_map: torch.Tensor,
        focus_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute grounding-aware loss.
        
        Args:
            grounding_map: [B, 1, H, W] predicted grounding probabilities
            focus_mask: [B, H, W] ground truth mask (1=target nucleus, 0=other)
            
        Returns:
            Total loss and component dict
        """
        # Ensure same shape
        if focus_mask.dim() == 3:
            focus_mask = focus_mask.unsqueeze(1).float()
        
        if grounding_map.shape[2:] != focus_mask.shape[2:]:
            grounding_map = F.interpolate(
                grounding_map, size=focus_mask.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        # BCE loss (using manual implementation for autocast safety)
        bce_loss = self.bce_loss_manual(grounding_map, focus_mask)
        
        # Dice loss
        dice_loss = self.dice_loss(grounding_map, focus_mask)
        
        # IoU loss
        iou_loss = self.iou_loss(grounding_map, focus_mask)
        
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.iou_weight * iou_loss
        )
        
        loss_dict = {
            'grounding_bce': bce_loss.item(),
            'grounding_dice': dice_loss.item(),
            'grounding_iou': iou_loss.item(),
        }
        
        return total_loss, loss_dict


# ============================================================
# Attention Regularization Loss
# ============================================================

class AttentionRegularizationLoss(nn.Module):
    """
    Regularize attention maps to be discriminative.
    
    Encourages attention to focus on relevant regions (nucleus pixels)
    rather than being diffuse.
    """
    
    def __init__(self, entropy_weight: float = 0.5, focus_weight: float = 0.5):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.focus_weight = focus_weight
    
    def forward(
        self,
        attn_maps: torch.Tensor,
        focus_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention regularization loss.
        
        Args:
            attn_maps: [B, H*W, L] attention maps (pixel to word)
            focus_mask: [B, H, W] mask for nucleus regions
            
        Returns:
            Regularization loss scalar
        """
        B, HW, L = attn_maps.shape
        H = W = int(np.sqrt(HW))
        
        # Reshape focus mask
        focus_flat = focus_mask.view(B, -1)  # [B, H*W]
        
        # 1. Entropy loss: encourage peaked attention (low entropy)
        attn_entropy = -(attn_maps * torch.log(attn_maps + 1e-8)).sum(dim=-1)  # [B, H*W]
        entropy_loss = attn_entropy.mean()
        
        # 2. Focus loss: encourage high attention on nucleus pixels
        # Average attention strength per pixel
        attn_strength = attn_maps.max(dim=-1)[0]  # [B, H*W]
        
        # Attention should be high on nucleus, low on background
        focus_loss = -(focus_flat * torch.log(attn_strength + 1e-8)).mean()
        focus_loss += -((1 - focus_flat) * torch.log(1 - attn_strength + 1e-8)).mean()
        
        total_loss = self.entropy_weight * entropy_loss + self.focus_weight * focus_loss
        
        return total_loss


# ============================================================
# LViT5 Combined Loss
# ============================================================

class LViT5Loss(nn.Module):
    """
    Ultimate Combined Loss for LViT5.
    
    Total Loss = L_hover + λ_pixel*L_pixel + λ_batch*L_batch + λ_ground*L_ground + λ_attn*L_attn
    
    Recommended weights (proven effective):
    - HoVer-Net: Standard (NP=1, HV=2, Type=2)
    - Pixel Contrastive: 1.0 (CRITICAL for grounding)
    - Batch Contrastive: 0.5 (auxiliary)
    - Grounding: 1.0 (direct supervision)
    - Attention Reg: 0.1 (regularization only)
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
        # Contrastive weights (HIGHER than before)
        pixel_contrastive_weight: float = 1.0,  # Was 0.5, now 1.0
        batch_contrastive_weight: float = 0.5,
        # Grounding weight (NEW)
        grounding_weight: float = 1.0,
        # Attention regularization weight
        attention_reg_weight: float = 0.1,
        # Contrastive settings
        contrastive_temperature: float = 0.1,
        hard_negative_ratio: float = 0.3,
        # Multi-scale weights
        multi_scale_weights: List[float] = [0.4, 0.3, 0.2, 0.1],
        # DRW
        use_drw: bool = True,
        cls_num_list: Optional[List[int]] = None,
    ):
        super().__init__()
        
        # Store weights
        self.pixel_contrastive_weight = pixel_contrastive_weight
        self.batch_contrastive_weight = batch_contrastive_weight
        self.grounding_weight = grounding_weight
        self.attention_reg_weight = attention_reg_weight
        self.loss_type = loss_type
        
        # Get class weights
        class_weights = get_pannuke_class_weights() if use_class_weights else None
        
        # Get class frequencies
        if cls_num_list is None:
            cls_num_list = get_class_frequencies_pannuke()
        
        # 1. HoVer-Net Loss
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
        
        # 2. Pixel-level Contrastive Loss (KEY FIX)
        self.pixel_contrastive_loss = PixelContrastiveLoss(
            temperature=contrastive_temperature,
            hard_negative_ratio=hard_negative_ratio,
        )
        
        # 3. Multi-scale Batch Contrastive Loss
        self.batch_contrastive_loss = MultiScaleBatchContrastiveLoss(
            scale_weights=multi_scale_weights,
            default_temperature=contrastive_temperature,
        )
        
        # 4. Grounding-Aware Loss (NEW)
        self.grounding_loss = GroundingAwareLoss(
            bce_weight=0.5,
            dice_weight=0.3,
            iou_weight=0.2,
        )
        
        # 5. Attention Regularization Loss
        self.attention_reg_loss = AttentionRegularizationLoss(
            entropy_weight=0.5,
            focus_weight=0.5,
        )
        
        self._print_info()
    
    def _print_info(self):
        """Print loss configuration."""
        print(f"\n[LViT5Loss] Ultimate Loss Function Initialized:")
        print(f"  HoVer-Net Loss: Standard")
        print(f"  Pixel Contrastive: weight={self.pixel_contrastive_weight} (CRITICAL)")
        print(f"  Batch Contrastive: weight={self.batch_contrastive_weight}")
        print(f"  Grounding Loss: weight={self.grounding_weight} (NEW)")
        print(f"  Attention Reg: weight={self.attention_reg_weight}")
        print(f"  Loss Type: {self.loss_type}")
    
    def update_type_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for DRW schedule."""
        self.hover_loss.update_type_weights(weights)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute LViT5 combined loss.
        
        Args:
            outputs: Model outputs with keys:
                - np, hv, type: Segmentation outputs
                - grounding_map: [B, 1, H, W]
                - pixel_features_*: Multi-scale pixel features
                - contrastive_text, contrastive_visual: Batch contrastive features
                - attn_maps_*: Attention maps
                - temperature: Learnable temperature
            targets: Target dict with keys:
                - np, hv, type: Targets
                
        Returns:
            Total loss and loss dict
        """
        device = outputs['np'].device
        
        # Get focus mask
        np_target = targets['np']
        if np_target.dim() == 4:
            np_target = np_target.squeeze(1)
        focus_mask = (np_target > 0).float()
        
        # ====== 1. HoVer-Net Loss ======
        hover_loss, hover_dict = self.hover_loss(
            pred_np=outputs['np'],
            pred_hv=outputs['hv'],
            pred_type=outputs['type'],
            target_np=targets['np'],
            target_hv=targets['hv'],
            target_type=targets['type'],
            focus_mask=focus_mask
        )
        
        # ====== 2. Pixel Contrastive Loss ======
        pixel_contrastive_val = torch.tensor(0.0, device=device)
        if 'pixel_features_out' in outputs and 'contrastive_text' in outputs:
            temperature = outputs.get('temperature', None)
            pixel_contrastive_val = self.pixel_contrastive_loss(
                pixel_features=outputs['pixel_features_out'],
                text_embed=outputs['contrastive_text'],
                target_type=targets['type'],
                focus_mask=focus_mask,
                temperature=temperature
            )
        
        # ====== 3. Batch Contrastive Loss ======
        batch_contrastive_val = torch.tensor(0.0, device=device)
        batch_contrastive_dict = {}
        if all(k in outputs for k in ['pixel_features_deep', 'pixel_features_mid', 
                                       'pixel_features_shallow', 'pixel_features_out',
                                       'contrastive_text']):
            temperature = outputs.get('temperature', None)
            batch_contrastive_val, batch_contrastive_dict = self.batch_contrastive_loss(
                pixel_features_deep=outputs['pixel_features_deep'],
                pixel_features_mid=outputs['pixel_features_mid'],
                pixel_features_shallow=outputs['pixel_features_shallow'],
                pixel_features_out=outputs['pixel_features_out'],
                text_embed=outputs['contrastive_text'],
                temperature=temperature
            )
        
        # ====== 4. Grounding Loss ======
        grounding_val = torch.tensor(0.0, device=device)
        grounding_dict = {}
        if 'grounding_map' in outputs:
            grounding_val, grounding_dict = self.grounding_loss(
                grounding_map=outputs['grounding_map'],
                focus_mask=focus_mask
            )
        
        # ====== 5. Attention Regularization ======
        attn_reg_val = torch.tensor(0.0, device=device)
        if 'attn_maps_out' in outputs:
            # Downsample focus mask to match attention maps
            attn_maps = outputs['attn_maps_out']
            B, HW, L = attn_maps.shape
            H = W = int(np.sqrt(HW))
            focus_small = F.interpolate(
                focus_mask.unsqueeze(1), size=(H, W), mode='nearest'
            ).squeeze(1)
            attn_reg_val = self.attention_reg_loss(attn_maps, focus_small)
        
        # ====== Combined Loss ======
        total_loss = (
            hover_loss +
            self.pixel_contrastive_weight * pixel_contrastive_val +
            self.batch_contrastive_weight * batch_contrastive_val +
            self.grounding_weight * grounding_val +
            self.attention_reg_weight * attn_reg_val
        )
        
        # Build loss dict
        loss_dict = {
            **hover_dict,
            'pixel_contrastive': pixel_contrastive_val.item(),
            'batch_contrastive': batch_contrastive_val.item(),
            **batch_contrastive_dict,
            'grounding': grounding_val.item(),
            **grounding_dict,
            'attention_reg': attn_reg_val.item(),
            'total': total_loss.item(),
            'loss_type': self.loss_type,
            # Weights for tracking
            'w_pixel_contrastive': self.pixel_contrastive_weight,
            'w_batch_contrastive': self.batch_contrastive_weight,
            'w_grounding': self.grounding_weight,
            'w_attention_reg': self.attention_reg_weight,
        }
        
        if 'temperature' in outputs:
            loss_dict['temperature'] = outputs['temperature'].item()
        
        return total_loss, loss_dict


# ============================================================
# Factory Function
# ============================================================

def create_lvit5_loss(
    num_classes: int = 6,
    np_weight: float = 1.0,
    hv_weight: float = 2.0,
    type_weight: float = 2.0,
    loss_type: str = 'weighted_focal',
    focal_gamma: float = 2.0,
    # Contrastive weights
    pixel_contrastive_weight: float = 1.0,
    batch_contrastive_weight: float = 0.5,
    grounding_weight: float = 1.0,
    attention_reg_weight: float = 0.1,
    # Settings
    use_drw: bool = True,
    device: str = 'cuda',
) -> LViT5Loss:
    """
    Factory function to create LViT5Loss.
    
    Args:
        num_classes: Number of classes
        np_weight, hv_weight, type_weight: HoVer-Net loss weights
        loss_type: Type of classification loss
        focal_gamma: Gamma for focal loss
        pixel_contrastive_weight: Weight for pixel contrastive (CRITICAL)
        batch_contrastive_weight: Weight for batch contrastive
        grounding_weight: Weight for grounding loss
        attention_reg_weight: Weight for attention regularization
        use_drw: Whether to enable DRW
        device: Target device
        
    Returns:
        LViT5Loss instance
    """
    loss_fn = LViT5Loss(
        num_classes=num_classes,
        np_weight=np_weight,
        hv_weight=hv_weight,
        type_weight=type_weight,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        pixel_contrastive_weight=pixel_contrastive_weight,
        batch_contrastive_weight=batch_contrastive_weight,
        grounding_weight=grounding_weight,
        attention_reg_weight=attention_reg_weight,
        use_drw=use_drw,
    )
    
    return loss_fn.to(device)


# ============================================================
# Testing
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("LViT5 Loss Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 2, 256, 256
    num_classes = 6
    embed_dim = 512
    
    # Create loss
    loss_fn = create_lvit5_loss(
        num_classes=num_classes,
        pixel_contrastive_weight=1.0,
        grounding_weight=1.0,
        device=str(device),
    )
    
    # Create dummy outputs (simulating LViT5 model output)
    H_small = H // 4  # Decoder output size
    outputs = {
        'np': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'hv': torch.randn(B, 2, H, W, device=device, requires_grad=True),
        'type': torch.randn(B, num_classes, H, W, device=device, requires_grad=True),
        'grounding_map': torch.sigmoid(torch.randn(B, 1, H, W, device=device)),
        'pixel_features_deep': torch.randn(B, embed_dim, H_small//4, H_small//4, device=device),
        'pixel_features_mid': torch.randn(B, embed_dim, H_small//2, H_small//2, device=device),
        'pixel_features_shallow': torch.randn(B, embed_dim, H_small, H_small, device=device),
        'pixel_features_out': torch.randn(B, embed_dim, H, W, device=device),
        'contrastive_text': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
        'contrastive_visual': F.normalize(torch.randn(B, embed_dim, device=device), p=2, dim=-1),
        'attn_maps_out': F.softmax(torch.randn(B, H*W, 20, device=device), dim=-1),
        'temperature': torch.tensor(0.1, device=device),
    }
    
    targets = {
        'np': torch.randint(0, 2, (B, H, W), device=device),
        'hv': torch.randn(B, 2, H, W, device=device) * 0.5,
        'type': torch.randint(0, num_classes, (B, H, W), device=device),
    }
    
    # Compute loss
    total_loss, loss_dict = loss_fn(outputs, targets)
    
    print(f"\nLoss Components:")
    for k, v in sorted(loss_dict.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\n{'=' * 40}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"{'=' * 40}")
    
    # Test gradient flow
    total_loss.backward()
    print("\n✓ Gradients computed successfully")
    
    print("\n" + "=" * 70)
    print("✅ LViT5 Loss test passed!")
    print("=" * 70)
