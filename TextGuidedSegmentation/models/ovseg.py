"""
OVSeg: Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP
Paper: https://arxiv.org/abs/2210.04150 (CVPR 2023)
Official Implementation: https://github.com/facebookresearch/ov-seg

Architecture:
- Uses mask proposals from a class-agnostic mask generator
- Adapts CLIP for mask-level classification
- Fine-tunes CLIP with masked image regions
- Enables open-vocabulary segmentation via mask-text matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class MaskPooling(nn.Module):
    """
    Mask-based pooling for extracting region features.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        features: torch.Tensor,  # (B, D, H, W)
        masks: torch.Tensor,     # (B, M, H, W) binary masks
    ) -> torch.Tensor:
        """
        Pool features within each mask region.
        
        Returns:
            Region features (B, M, D)
        """
        B, D, H, W = features.shape
        M = masks.shape[1]
        
        # Normalize masks
        masks = masks.float()
        mask_sum = masks.sum(dim=(2, 3), keepdim=True).clamp(min=1)
        masks_normalized = masks / mask_sum
        
        # Pool: (B, M, H, W) @ (B, D, H, W) -> (B, M, D)
        features_flat = features.reshape(B, D, -1)  # (B, D, H*W)
        masks_flat = masks_normalized.reshape(B, M, -1)  # (B, M, H*W)
        
        pooled = torch.bmm(masks_flat, features_flat.transpose(1, 2))  # (B, M, D)
        
        return pooled


class SimpleMaskGenerator(nn.Module):
    """
    Simple mask proposal generator using learned queries.
    In practice, OVSeg uses Mask2Former or similar.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Learnable queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        
        # Cross attention to features
        self.cross_attn = nn.MultiheadAttention(
            in_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(in_dim)
        
        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Feature projection for mask dot product
        self.feat_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
    
    def forward(
        self,
        features: torch.Tensor,  # (B, D, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask proposals.
        
        Returns:
            Masks (B, Q, H, W) and query features (B, Q, D)
        """
        B, D, H, W = features.shape
        
        # Flatten features for attention
        features_flat = features.reshape(B, D, -1).permute(0, 2, 1)  # (B, N, D)
        
        # Query features
        queries = self.query_embed.expand(B, -1, -1)
        queries, _ = self.cross_attn(queries, features_flat, features_flat)
        queries = self.norm(queries)
        
        # Generate mask embeddings
        mask_embeds = self.mask_head(queries)  # (B, Q, hidden_dim)
        
        # Project features
        feat_proj = self.feat_proj(features)  # (B, hidden_dim, H, W)
        
        # Compute masks via dot product
        masks = torch.einsum('bqd,bdhw->bqhw', mask_embeds, feat_proj)
        
        return masks, queries


class MaskAdaptedCLIP(nn.Module):
    """
    CLIP adapted for mask-level classification.
    """
    
    def __init__(
        self,
        clip_model,
        visual_dim: int,
        embed_dim: int,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.freeze_clip = freeze_clip
        
        # Mask-adapted projection
        self.mask_proj = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
    
    def encode_masks(
        self,
        image_features: torch.Tensor,  # (B, D, H, W)
        masks: torch.Tensor,           # (B, M, H, W)
    ) -> torch.Tensor:
        """
        Encode masked regions.
        """
        B, D, H, W = image_features.shape
        M = masks.shape[1]
        
        # Apply masks to features
        masks = masks.sigmoid()  # Ensure [0, 1]
        masks_expanded = masks.unsqueeze(2)  # (B, M, 1, H, W)
        features_expanded = image_features.unsqueeze(1)  # (B, 1, D, H, W)
        
        # Masked average pooling
        masked_features = (features_expanded * masks_expanded).sum(dim=(3, 4))  # (B, M, D)
        mask_areas = masks_expanded.sum(dim=(3, 4)).clamp(min=1)  # (B, M, 1)
        masked_features = masked_features / mask_areas
        
        # Project
        masked_features = self.mask_proj(masked_features)  # (B, M, embed_dim)
        
        return masked_features
    
    def forward(
        self,
        mask_features: torch.Tensor,  # (B, M, D)
        text_features: torch.Tensor,  # (C, D)
    ) -> torch.Tensor:
        """
        Compute mask-text similarity.
        """
        # Normalize
        mask_features = F.normalize(mask_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Similarity
        sim = torch.einsum('bmd,cd->bmc', mask_features, text_features)
        sim = sim * self.logit_scale.exp()
        
        return sim  # (B, M, C)


@register_model("ovseg")
class OVSeg(TextGuidedSegmentationBase):
    """
    OVSeg: Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP
    
    Key features:
    - Generates class-agnostic mask proposals
    - Classifies masks using adapted CLIP
    - Fine-tuned for masked image-text matching
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_dim: int = 256,
        num_queries: int = 100,
        freeze_clip: bool = True,
        device: str = "cuda",
    ):
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            clip_model=clip_model,
            freeze_clip=freeze_clip,
            device=device,
        )
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Mask generator
        self.mask_generator = SimpleMaskGenerator(
            in_dim=self.visual_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
        )
        
        # Mask-adapted CLIP
        self.mask_clip = MaskAdaptedCLIP(
            self.clip_model,
            visual_dim=self.visual_dim,
            embed_dim=self.clip_embed_dim,
            freeze_clip=freeze_clip,
        )
        
        # Dense feature extractor
        self.dense_proj = nn.Sequential(
            nn.Conv2d(self.visual_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        
        self.to(device)
    
    def extract_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract dense features from CLIP."""
        visual = self.clip_model.visual
        B = image.shape[0]
        
        x = visual.conv1(image)
        patch_h, patch_w = x.shape[2], x.shape[3]
        
        x = x.reshape(B, self.visual_dim, -1).permute(0, 2, 1)
        
        cls_token = visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        # Interpolate positional embeddings if needed
        pos_embed = visual.positional_embedding
        if x.shape[1] != pos_embed.shape[0]:
            old_grid_size = int((pos_embed.shape[0] - 1) ** 0.5)
            new_grid_size = int((x.shape[1] - 1) ** 0.5)
            pos_embed = self.interpolate_pos_embed(pos_embed, new_grid_size, old_grid_size)
        x = x + pos_embed
        x = visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        
        x = x[:, 1:].permute(0, 2, 1).reshape(B, self.visual_dim, patch_h, patch_w)
        
        return x
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image and generate mask proposals.
        """
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            clip_features = self.extract_dense_features(image)
        
        # Generate masks
        masks, queries = self.mask_generator(clip_features)
        
        # Dense features for final prediction
        dense_features = self.dense_proj(clip_features)
        
        return clip_features, masks, dense_features
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text using CLIP."""
        return self.encode_text_clip(text_prompts)
    
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        B, _, H, W = image.shape
        
        # Encode image and generate masks
        clip_features, mask_logits, dense_features = self.encode_image(image)
        
        # Encode text
        text_features = self.encode_text(text_prompts)  # (C, D)
        
        # Get mask features
        masks = mask_logits.sigmoid()
        mask_features = self.mask_clip.encode_masks(clip_features, masks)
        
        # Mask-text similarity
        mask_text_sim = self.mask_clip(mask_features, text_features)  # (B, M, C)
        
        # Upsample masks to original size
        masks_upsampled = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        
        # Combine: weighted sum of masks by class probability
        # (B, M, H, W) weighted by (B, M, C) -> (B, C, H, W)
        class_probs = mask_text_sim.softmax(dim=-1)  # (B, M, C)
        
        # Weighted mask combination
        logits = torch.einsum('bmhw,bmc->bchw', masks_upsampled, class_probs)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'masks': masks_upsampled,
            'mask_logits': mask_logits,
            'mask_features': mask_features,
            'text_features': text_features,
            'mask_text_sim': mask_text_sim,
        }
