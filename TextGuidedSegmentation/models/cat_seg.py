"""
CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation
Paper: https://arxiv.org/abs/2303.11797 (CVPR 2024)
Official Implementation: https://github.com/cvlab-kaist/CAT-Seg

Architecture:
- Cost aggregation approach for dense prediction
- Aggregates pixel-text costs across spatial locations
- Uses CLIP features with cost volume construction
- Efficient aggregation network for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class CostVolumeEncoder(nn.Module):
    """
    Encodes the cost volume (pixel-text similarity) for aggregation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        ]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_volume: (B, C, H, W) where C is number of classes
            
        Returns:
            Encoded cost features
        """
        return self.encoder(cost_volume)


class SpatialAggregator(nn.Module):
    """
    Spatial aggregation module for cost volume refinement.
    """
    
    def __init__(
        self,
        hidden_channels: int = 64,
        num_heads: int = 4,
        window_size: int = 7,
    ):
        super().__init__()
        
        self.window_size = window_size
        self.num_heads = num_heads
        
        # Local attention for spatial aggregation
        self.attention = nn.MultiheadAttention(
            hidden_channels, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_channels)
        
        # Conv for local context
        self.local_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Cost features (B, D, H, W)
        """
        B, D, H, W = x.shape
        
        # Local conv path
        local_out = self.local_conv(x)
        
        # Attention path (on flattened spatial)
        x_flat = x.reshape(B, D, -1).permute(0, 2, 1)  # (B, H*W, D)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out + x_flat)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, D, H, W)
        
        return local_out + attn_out


class CostAggregationNetwork(nn.Module):
    """
    Cost aggregation network for refining pixel-text similarities.
    """
    
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 64,
        num_stages: int = 3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Cost encoder
        self.cost_encoder = CostVolumeEncoder(
            in_channels=num_classes,
            hidden_channels=hidden_channels,
        )
        
        # Aggregation stages
        self.aggregators = nn.ModuleList([
            SpatialAggregator(hidden_channels)
            for _ in range(num_stages)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )
    
    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_volume: Raw pixel-text similarities (B, C, H, W)
            
        Returns:
            Refined segmentation logits (B, C, H, W)
        """
        # Encode cost volume
        x = self.cost_encoder(cost_volume)
        
        # Aggregate
        for aggregator in self.aggregators:
            x = x + aggregator(x)
        
        # Output
        logits = self.output_head(x)
        
        # Residual connection with original costs
        logits = logits + cost_volume
        
        return logits


class FeatureGuidance(nn.Module):
    """
    Feature guidance module for incorporating spatial features.
    """
    
    def __init__(
        self,
        visual_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        features: torch.Tensor,  # (B, D, H, W)
        cost_volume: torch.Tensor,  # (B, C, H, W)
    ) -> torch.Tensor:
        """
        Guide cost volume with spatial features.
        """
        # Project features
        feat_proj = self.proj(features)
        
        # Compute spatial gate
        gate = self.gate(feat_proj)  # (B, 1, H, W)
        
        # Apply gate to cost volume
        return cost_volume * gate


@register_model("cat_seg")
class CATSeg(TextGuidedSegmentationBase):
    """
    CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation
    
    Key features:
    - Cost volume approach for pixel-text matching
    - Spatial aggregation for cost refinement
    - Feature-guided cost modulation
    - Strong performance on open-vocabulary benchmarks
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_channels: int = 64,
        num_agg_stages: int = 3,
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
        
        self.hidden_channels = hidden_channels
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Cost aggregation network
        self.cost_agg = CostAggregationNetwork(
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_stages=num_agg_stages,
        )
        
        # Feature guidance
        self.feat_guidance = FeatureGuidance(
            visual_dim=self.visual_dim,
            hidden_dim=hidden_channels,
        )
        
        # Feature upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(self.visual_dim, self.visual_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.visual_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.visual_dim // 2, self.visual_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.visual_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        # Project upsampled features for similarity
        self.feat_proj = nn.Conv2d(self.visual_dim // 4, self.clip_embed_dim, kernel_size=1)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        
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
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image.
        
        Returns:
            Dense features and upsampled features for similarity
        """
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            dense_features = self.extract_dense_features(image)
        
        # Upsample for higher resolution
        upsampled = self.upsampler(dense_features)
        upsampled = self.feat_proj(upsampled)
        
        return dense_features, upsampled
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text using CLIP."""
        return self.encode_text_clip(text_prompts)
    
    def compute_cost_volume(
        self,
        visual_features: torch.Tensor,  # (B, D, H, W)
        text_features: torch.Tensor,    # (C, D)
    ) -> torch.Tensor:
        """
        Compute cost volume (pixel-text similarities).
        """
        # Normalize
        visual_features = F.normalize(visual_features, dim=1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        cost_volume = self.compute_similarity(visual_features, text_features, normalize=False)
        cost_volume = cost_volume * self.logit_scale.exp()
        
        return cost_volume
    
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        B, _, H, W = image.shape
        
        # Encode image
        dense_features, upsampled_features = self.encode_image(image)
        
        # Encode text
        text_features = self.encode_text(text_prompts)
        
        # Compute initial cost volume
        cost_volume = self.compute_cost_volume(upsampled_features, text_features)
        
        # Feature guidance
        # Need to upsample dense features to match cost volume size
        dense_upsampled = F.interpolate(
            dense_features, 
            size=cost_volume.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        cost_volume = self.feat_guidance(dense_upsampled, cost_volume)
        
        # Cost aggregation
        logits = self.cost_agg(cost_volume)
        
        # Upsample to original size if needed
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'cost_volume': cost_volume,
            'visual_features': upsampled_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale,
        }
