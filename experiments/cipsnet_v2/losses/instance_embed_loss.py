"""
Loss Functions for LViT-IE (Instance Embedding Decoder)
=========================================================

Novel loss functions for training the instance embedding decoder:

1. PullPushEmbeddingLoss — Discriminative loss for instance embeddings
   - Pull: same-instance pixel embeddings → cluster center
   - Push: different-instance cluster centers → apart
   (De Brabandere et al., 2017)

2. DistanceTransformLoss — Regression loss for normalized EDT
   - MSE on foreground pixels
   - Gradient loss for boundary sharpness

3. InstancePooledTypeLoss — Classification of whole nuclei
   - Cross-entropy on pooled instance features
   - Focal weighting for hard examples

4. LViTIELoss — Combined loss orchestrating all components

5. PCGrad — Projecting Conflicting Gradients for multi-task balancing
   (Yu et al., NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .losses import (
    NPLoss,
    TypeLoss,
    DiceLoss,
    WeightedFocalCELoss,
    get_pannuke_class_weights,
    get_class_frequencies_pannuke,
)


# =============================================================================
# Pull-Push Discriminative Embedding Loss
# =============================================================================

class PullPushEmbeddingLoss(nn.Module):
    """
    Discriminative loss for instance embeddings.
    
    From: De Brabandere et al., "Semantic Instance Segmentation with a
    Discriminative Loss Function" (2017)
    
    L = L_pull + L_push + L_reg
    
    L_pull = (1/K) Σ_k (1/|Ω_k|) Σ_{p ∈ Ω_k} max(0, ||e_p - μ_k|| - δ_v)²
    L_push = (1/K(K-1)) Σ_{k_A ≠ k_B} max(0, 2δ_d - ||μ_kA - μ_kB||)²
    L_reg  = (1/K) Σ_k ||μ_k||
    
    where:
        e_p = embedding at pixel p
        μ_k = mean embedding of instance k
        δ_v = pull margin (embeddings within δ_v of center are not penalized)
        δ_d = push margin (centers farther than 2δ_d apart are not penalized)
    """
    
    def __init__(
        self,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        pull_weight: float = 1.0,
        push_weight: float = 1.0,
        reg_weight: float = 0.001,
        max_instances: int = 300,
    ):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.reg_weight = reg_weight
        self.max_instances = max_instances
    
    def forward(
        self,
        embeddings: torch.Tensor,
        instance_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: [B, D, H, W] per-pixel embeddings
            instance_maps: [B, H, W] instance IDs (0 = background)
        
        Returns:
            loss, dict with pull/push/reg components
        """
        B, D, H, W = embeddings.shape
        device = embeddings.device
        
        total_pull = torch.tensor(0.0, device=device)
        total_push = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for b in range(B):
            embed_b = embeddings[b]  # [D, H, W]
            inst_b = instance_maps[b]  # [H, W]
            
            unique_ids = torch.unique(inst_b)
            unique_ids = unique_ids[unique_ids > 0]
            
            K = len(unique_ids)
            if K == 0:
                continue
            
            # Limit instances for memory
            if K > self.max_instances:
                perm = torch.randperm(K, device=device)[:self.max_instances]
                unique_ids = unique_ids[perm]
                K = self.max_instances
            
            # Compute cluster centers
            centers = []
            instance_masks = []
            
            embed_flat = embed_b.view(D, -1)  # [D, H*W]
            inst_flat = inst_b.view(-1)  # [H*W]
            
            for inst_id in unique_ids:
                mask = (inst_flat == inst_id)
                if mask.sum() < 3:
                    continue
                
                # Mean embedding for this instance
                inst_embeds = embed_flat[:, mask]  # [D, n_pixels]
                center = inst_embeds.mean(dim=1)  # [D]
                centers.append(center)
                instance_masks.append((mask, inst_embeds))
            
            K = len(centers)
            if K == 0:
                continue
            
            centers_tensor = torch.stack(centers, dim=0)  # [K, D]
            
            # ===== Pull Loss =====
            pull_loss = torch.tensor(0.0, device=device)
            for k, (mask, inst_embeds) in enumerate(instance_masks):
                # Distance of each pixel to its cluster center
                dists = torch.norm(inst_embeds - centers[k].unsqueeze(1), dim=0)  # [n_pixels]
                # Hinge: only penalize if beyond delta_v
                pull = F.relu(dists - self.delta_v) ** 2
                pull_loss = pull_loss + pull.mean()
            pull_loss = pull_loss / K
            
            # ===== Push Loss =====
            push_loss = torch.tensor(0.0, device=device)
            if K > 1:
                # Pairwise distances between cluster centers
                # [K, D] → [K, K]
                center_dists = torch.cdist(centers_tensor.unsqueeze(0), centers_tensor.unsqueeze(0)).squeeze(0)
                
                # Hinge: penalize if closer than 2 * delta_d
                push_matrix = F.relu(2 * self.delta_d - center_dists) ** 2
                
                # Exclude diagonal
                mask_diag = ~torch.eye(K, dtype=torch.bool, device=device)
                push_loss = push_matrix[mask_diag].mean()
            
            # ===== Regularization Loss =====
            reg_loss = centers_tensor.norm(dim=1).mean()
            
            total_pull = total_pull + pull_loss
            total_push = total_push + push_loss
            total_reg = total_reg + reg_loss
            valid_batches += 1
        
        if valid_batches == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {'embed_pull': 0.0, 'embed_push': 0.0, 'embed_reg': 0.0}
        
        total_pull = total_pull / valid_batches
        total_push = total_push / valid_batches
        total_reg = total_reg / valid_batches
        
        loss = (self.pull_weight * total_pull +
                self.push_weight * total_push +
                self.reg_weight * total_reg)
        
        return loss, {
            'embed_pull': total_pull.item(),
            'embed_push': total_push.item(),
            'embed_reg': total_reg.item(),
            'embed_total': loss.item(),
        }


# =============================================================================
# Distance Transform Regression Loss
# =============================================================================

class DistanceTransformLoss(nn.Module):
    """
    Loss for normalized distance transform prediction.
    
    L_dist = MSE(pred_dist, gt_dist) on foreground pixels
           + λ_grad * MSE(∇pred, ∇gt) for boundary sharpness
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        gradient_weight: float = 0.5,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight
        
        # Sobel kernels for gradient
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
    
    def forward(
        self,
        pred_dist: torch.Tensor,
        gt_dist: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_dist: [B, 1, H, W] predicted normalized EDT
            gt_dist: [B, 1, H, W] ground truth normalized EDT
            foreground_mask: [B, H, W] or [B, 1, H, W] binary mask
        
        Returns:
            loss, dict
        """
        if foreground_mask.dim() == 3:
            foreground_mask = foreground_mask.unsqueeze(1)
        foreground_mask = foreground_mask.float()
        
        # MSE loss on foreground pixels
        diff = (pred_dist - gt_dist) ** 2
        mse_loss = (diff * foreground_mask).sum() / (foreground_mask.sum() + 1e-8)
        
        # Gradient loss for boundary sharpness
        grad_loss = torch.tensor(0.0, device=pred_dist.device)
        if self.gradient_weight > 0:
            pred_gx = F.conv2d(pred_dist, self.sobel_x.to(pred_dist.device), padding=1)
            pred_gy = F.conv2d(pred_dist, self.sobel_y.to(pred_dist.device), padding=1)
            gt_gx = F.conv2d(gt_dist, self.sobel_x.to(gt_dist.device), padding=1)
            gt_gy = F.conv2d(gt_dist, self.sobel_y.to(gt_dist.device), padding=1)
            
            grad_diff = (pred_gx - gt_gx) ** 2 + (pred_gy - gt_gy) ** 2
            grad_loss = (grad_diff * foreground_mask).sum() / (foreground_mask.sum() + 1e-8)
        
        total = self.mse_weight * mse_loss + self.gradient_weight * grad_loss
        
        return total, {
            'dist_mse': mse_loss.item(),
            'dist_grad': grad_loss.item(),
            'dist_total': total.item(),
        }


# =============================================================================
# Instance-Pooled Type Classification Loss
# =============================================================================

class InstancePooledTypeLoss(nn.Module):
    """
    Loss for instance-pooled nucleus type classification.
    
    Uses Weighted Focal CE on per-instance logits + per-pixel CE+Dice
    as auxiliary loss for dense supervision.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        focal_gamma: float = 2.0,
        inst_weight: float = 1.0,
        pixel_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.inst_weight = inst_weight
        self.pixel_weight = pixel_weight
        
        # Per-instance: Weighted Focal CE
        class_weights = get_pannuke_class_weights()
        self.inst_loss_fn = WeightedFocalCELoss(
            num_classes=num_classes,
            gamma=focal_gamma,
            class_weights=class_weights,
            reduction='mean',
        )
        
        # Per-pixel auxiliary: same TypeLoss as HoVer-Net (for backward compat)
        self.pixel_loss_fn = TypeLoss(
            num_classes=num_classes,
            loss_type='weighted_focal',
            focal_gamma=focal_gamma,
        )
    
    def update_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for DRW."""
        if weights is not None:
            self.inst_loss_fn.update_weights(weights)
            self.pixel_loss_fn.update_weights(weights)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_type: torch.Tensor,
        instance_maps: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: model outputs with 'type' and optionally 'inst_type_logits'
            target_type: [B, H, W] pixel-level type labels
            instance_maps: [B, H, W] GT instance IDs
            foreground_mask: [B, H, W]
        
        Returns:
            loss, dict
        """
        device = target_type.device
        
        # Per-pixel auxiliary loss (always available)
        pixel_loss, pixel_dict = self.pixel_loss_fn(
            outputs['type'], target_type, foreground_mask
        )
        
        # Instance-pooled loss (training with GT)
        inst_loss = torch.tensor(0.0, device=device)
        if 'inst_type_logits' in outputs and outputs['inst_type_logits'] is not None:
            inst_logits = outputs['inst_type_logits']  # [N_inst, num_classes]
            
            # Compute GT labels per instance by majority vote
            inst_labels = self._get_instance_labels(
                instance_maps, target_type, len(inst_logits), device
            )
            
            if inst_labels is not None and len(inst_labels) == len(inst_logits):
                inst_loss = self.inst_loss_fn(inst_logits, inst_labels)
        
        total = self.pixel_weight * pixel_loss + self.inst_weight * inst_loss
        
        return total, {
            'type_pixel': pixel_loss.item(),
            'type_inst': inst_loss.item() if isinstance(inst_loss, torch.Tensor) else inst_loss,
            'type_total': total.item(),
        }
    
    def _get_instance_labels(
        self,
        instance_maps: torch.Tensor,
        type_maps: torch.Tensor,
        n_expected: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Get majority-vote type label per instance."""
        B = instance_maps.shape[0]
        labels = []
        
        for b in range(B):
            unique_ids = torch.unique(instance_maps[b])
            unique_ids = unique_ids[unique_ids > 0]
            
            for inst_id in unique_ids:
                mask = (instance_maps[b] == inst_id)
                if mask.sum() < 5:
                    continue
                
                types_in_inst = type_maps[b][mask].long()
                counts = torch.bincount(types_in_inst, minlength=self.num_classes)
                counts[0] = 0  # Ignore background
                label = counts.argmax()
                if label == 0:
                    label = torch.tensor(1, device=device)  # Default neoplastic
                labels.append(label)
        
        if len(labels) != n_expected:
            return None  # Mismatch, skip instance loss this batch
        
        return torch.stack(labels).to(device)


# =============================================================================
# PCGrad — Projecting Conflicting Gradients
# =============================================================================

class PCGrad:
    """
    PCGrad: Projecting Conflicting Gradients for Multi-Task Learning.
    
    From: Yu et al., "Gradient Surgery for Multi-Task Learning" NeurIPS 2020
    
    When two task gradients conflict (dot product < 0), project one
    onto the normal plane of the other:
    
        g_i^PC = g_i - (g_i · g_j / ||g_j||²) * g_j
    
    This removes the conflicting component while preserving non-conflicting ones.
    
    Usage:
        pcgrad = PCGrad(optimizer)
        losses = [loss_np, loss_dist, loss_embed, loss_type]
        pcgrad.step(losses, shared_params)
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
    
    @staticmethod
    def _project_conflicting(grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project conflicting gradient pairs."""
        n_tasks = len(grads)
        projected = [g.clone() for g in grads]
        
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                dot = torch.dot(projected[i].flatten(), grads[j].flatten())
                if dot < 0:
                    # Project: remove conflicting component
                    proj = dot / (torch.dot(grads[j].flatten(), grads[j].flatten()) + 1e-8)
                    projected[i] = projected[i] - proj * grads[j]
        
        return projected
    
    def step(
        self,
        losses: List[torch.Tensor],
        shared_parameters: List[nn.Parameter],
    ):
        """
        Compute PCGrad update.
        
        Args:
            losses: list of per-task scalar losses
            shared_parameters: parameters that all tasks share (encoder)
        """
        # Compute per-task gradients
        grads = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad = torch.cat([
                p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                for p in shared_parameters
            ])
            grads.append(grad)
        
        # Project conflicting gradients
        projected = self._project_conflicting(grads)
        
        # Sum projected gradients
        final_grad = sum(projected)
        
        # Apply to parameters
        self.optimizer.zero_grad()
        offset = 0
        for p in shared_parameters:
            numel = p.numel()
            p.grad = final_grad[offset:offset + numel].view_as(p)
            offset += numel
        
        self.optimizer.step()


# =============================================================================
# Combined Loss for LViT-IE
# =============================================================================

class LViTIELoss(nn.Module):
    """
    Combined loss for LViT-IE training.
    
    Total = w_np * L_np + w_dist * L_dist + w_embed * L_embed + w_type * L_type
    
    Components:
      - L_np:    Binary segmentation (Focal CE + Dice)
      - L_dist:  Distance transform regression (MSE + gradient)
      - L_embed: Instance embedding (pull-push discriminative)
      - L_type:  Instance-pooled + pixel auxiliary classification
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        # Head weights
        np_weight: float = 1.0,
        dist_weight: float = 2.0,
        embed_weight: float = 1.0,
        type_weight: float = 2.0,
        # Focal gamma
        focal_gamma: float = 2.0,
        # Embedding loss params
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        # Instance classification params
        inst_cls_weight: float = 1.0,
        pixel_cls_weight: float = 0.5,
        # Class weights
        use_class_weights: bool = True,
        # DRW
        use_drw: bool = True,
        cls_num_list: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.np_weight = np_weight
        self.dist_weight = dist_weight
        self.embed_weight = embed_weight
        self.type_weight = type_weight
        
        # NP Loss (same as HoVer-Net)
        self.np_loss = NPLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            use_focal=True,
            focal_gamma=focal_gamma,
        )
        
        # Distance Transform Loss (replaces HV loss)
        self.dist_loss = DistanceTransformLoss(
            mse_weight=1.0,
            gradient_weight=0.5,
        )
        
        # Instance Embedding Loss
        self.embed_loss = PullPushEmbeddingLoss(
            delta_v=delta_v,
            delta_d=delta_d,
        )
        
        # Type Classification Loss (instance-pooled + pixel auxiliary)
        self.type_loss = InstancePooledTypeLoss(
            num_classes=num_classes,
            focal_gamma=focal_gamma,
            inst_weight=inst_cls_weight,
            pixel_weight=pixel_cls_weight,
        )
        
        print(f"  [LViT-IE Loss] NP={np_weight}, Dist={dist_weight}, "
              f"Embed={embed_weight}, Type={type_weight}")
        print(f"  [LViT-IE Loss] Focal γ={focal_gamma}, "
              f"δ_v={delta_v}, δ_d={delta_d}")
    
    def update_type_weights(self, weights: Optional[torch.Tensor]):
        """Update class weights for DRW schedule."""
        self.type_loss.update_weights(weights)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: model outputs with keys:
                'np': [B, 2, H, W]
                'dist': [B, 1, H, W]
                'embed': [B, D, H, W]
                'type': [B, num_classes, H, W]
                'inst_type_logits': [N, num_classes] (optional)
            targets: with keys:
                'np': [B, H, W]
                'dist': [B, 1, H, W] normalized distance transform
                'type': [B, H, W]
                'instance': [B, H, W] instance map
        """
        # Foreground mask
        np_target = targets['np']
        if np_target.dim() == 4:
            np_target = np_target.squeeze(1)
        foreground_mask = (np_target > 0).float()
        
        # 1. NP Loss
        np_loss, np_dict = self.np_loss(outputs['np'], np_target)
        
        # 2. Distance Transform Loss
        dist_loss, dist_dict = self.dist_loss(
            outputs['dist'], targets['dist'], foreground_mask
        )
        
        # 3. Instance Embedding Loss
        embed_loss, embed_dict = self.embed_loss(
            outputs['embed'], targets['instance']
        )
        
        # 4. Type Classification Loss
        type_loss, type_dict = self.type_loss(
            outputs, targets['type'], targets['instance'], foreground_mask
        )
        
        # Total
        total = (self.np_weight * np_loss +
                 self.dist_weight * dist_loss +
                 self.embed_weight * embed_loss +
                 self.type_weight * type_loss)
        
        loss_dict = {
            **np_dict,
            **dist_dict,
            **embed_dict,
            **type_dict,
            'total': total.item(),
        }
        
        return total, loss_dict


# =============================================================================
# Factory
# =============================================================================

def create_lvit_ie_loss(
    num_classes: int = 6,
    focal_gamma: float = 2.0,
    type_weight: float = 2.0,
    use_class_weights: bool = True,
    **kwargs,
) -> LViTIELoss:
    """Create LViT-IE loss with recommended settings."""
    return LViTIELoss(
        num_classes=num_classes,
        np_weight=1.0,
        dist_weight=2.0,
        embed_weight=1.0,
        type_weight=type_weight,
        focal_gamma=focal_gamma,
        use_class_weights=use_class_weights,
        **kwargs,
    )
