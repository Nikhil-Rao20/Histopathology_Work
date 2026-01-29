"""
Semantic-SAM: Segment and Recognize Anything at Any Granularity
Paper: https://arxiv.org/abs/2307.04767 (ECCV 2024)
Official Implementation: https://github.com/UX-Decoder/Semantic-SAM

Architecture:
- Multi-granularity segmentation with semantic understanding
- Combines SAM-style architecture with semantic features
- Joint click, box, and text prompts
- Hierarchical mask decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder for extracting features at different granularities.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_scales: int = 4,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Multi-scale feature extraction
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_scales)
        ])
        
        # Scale-specific pooling
        self.pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2**i, stride=2**i) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []
        for conv, pool in zip(self.scale_convs, self.pools):
            pooled = pool(x)
            feat = conv(pooled)
            features.append(feat)
        return features


class SemanticDecoder(nn.Module):
    """
    Semantic decoder that incorporates text conditioning.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        text_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        
        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross attention for text conditioning
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        features: torch.Tensor,       # (B, N, D)
        text_features: torch.Tensor,  # (C, D_text)
    ) -> torch.Tensor:
        B = features.shape[0]
        
        # Project text
        text_proj = self.text_proj(text_features)  # (C, hidden)
        text_proj = text_proj.unsqueeze(0).expand(B, -1, -1)  # (B, C, hidden)
        
        # Cross attention with text
        x = self.norm1(features)
        x, _ = self.cross_attn(x, text_proj, text_proj)
        features = features + x
        
        # Self attention
        x = self.norm2(features)
        x, _ = self.self_attn(x, x, x)
        features = features + x
        
        # FFN
        features = features + self.ffn(self.norm3(features))
        
        return features


class HierarchicalMaskDecoder(nn.Module):
    """
    Hierarchical mask decoder for multi-granularity segmentation.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_granularities: int = 3,  # e.g., part, object, scene
    ):
        super().__init__()
        
        self.num_granularities = num_granularities
        
        # Granularity-specific decoders
        self.granularity_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            )
            for _ in range(num_granularities)
        ])
        
        # Mask heads
        self.mask_heads = nn.ModuleList([
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
            for _ in range(num_granularities)
        ])
    
    def forward(
        self,
        features: torch.Tensor,  # (B, D, H, W)
    ) -> List[torch.Tensor]:
        """
        Returns masks at different granularities.
        """
        masks = []
        for decoder, head in zip(self.granularity_decoders, self.mask_heads):
            feat = decoder(features)
            mask = head(feat)
            masks.append(mask)
        return masks


@register_model("semantic_sam")
class SemanticSAM(TextGuidedSegmentationBase):
    """
    Semantic-SAM: Segment and Recognize Anything at Any Granularity
    
    Key features:
    - Multi-granularity segmentation
    - Text-guided semantic understanding
    - Hierarchical mask decoding
    - Strong generalization capability
    
    Note: This is a simplified version focusing on text-guided segmentation.
    Full Semantic-SAM also supports point and box prompts.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_dim: int = 256,
        num_granularities: int = 1,  # Simplified to 1 for semantic segmentation
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
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Multi-scale encoder
        self.multi_scale = MultiScaleEncoder(
            in_dim=self.visual_dim,
            hidden_dim=hidden_dim,
            num_scales=4,
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Semantic decoder
        self.semantic_decoder = SemanticDecoder(
            hidden_dim=hidden_dim,
            text_dim=self.clip_embed_dim,
            num_heads=8,
        )
        
        # Dense feature upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, self.clip_embed_dim, kernel_size=1)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        
        self.to(device)
    
    def extract_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract visual features from CLIP."""
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
    
    def encode_image(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image with semantic guidance."""
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            clip_features = self.extract_visual_features(image)
        
        B, D, h, w = clip_features.shape
        
        # Multi-scale features
        scale_features = self.multi_scale(clip_features)
        
        # Upsample all to same size and concatenate
        target_size = scale_features[0].shape[2:]
        upsampled = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in scale_features
        ]
        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion(fused)  # (B, hidden, H, W)
        
        # Flatten for semantic decoder
        fused_flat = fused.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        
        # Semantic decoding with text guidance
        semantic_features = self.semantic_decoder(fused_flat, text_features)
        
        # Reshape back to spatial
        semantic_features = semantic_features.permute(0, 2, 1).reshape(B, self.hidden_dim, *target_size)
        
        # Upsample
        upsampled = self.upsampler(semantic_features)
        
        # Project to CLIP dimension
        output = self.output_proj(upsampled)
        
        return output
    
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
        
        # Encode text
        text_features = self.encode_text(text_prompts)
        
        # Encode image with semantic guidance
        visual_features = self.encode_image(image, text_features)
        
        # Normalize
        visual_features = F.normalize(visual_features, dim=1)
        text_features_norm = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        logits = self.compute_similarity(visual_features, text_features_norm, normalize=False)
        logits = logits * self.logit_scale.exp()
        
        # Ensure correct size
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'visual_features': visual_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale,
        }
