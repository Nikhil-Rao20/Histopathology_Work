"""
TagAlign: Tag-Guided Open-Vocabulary Semantic Segmentation
Paper: https://arxiv.org/abs/2312.14149 (2023)
Official Implementation: https://github.com/Qinying-Liu/TagAlign

Architecture:
- Uses image tags to bridge visual and text representations
- Tag-guided feature alignment
- Multi-granularity tag matching
- Efficient open-vocabulary segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class TagEncoder(nn.Module):
    """
    Encodes tags (class names) into embeddings.
    """
    
    def __init__(
        self,
        clip_model,
        embed_dim: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.clip_model = clip_model
        self.embed_dim = embed_dim
        self.device = device
    
    def forward(self, tags: List[str]) -> torch.Tensor:
        """Encode tags using CLIP text encoder."""
        tokens = clip.tokenize(tags).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        return text_features  # (T, D)


class TagAlignmentModule(nn.Module):
    """
    Aligns visual features with tag embeddings.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        
        # Project visual features
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Project text features
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention for alignment
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, N, D_v)
        tag_features: torch.Tensor,     # (T, D_t)
    ) -> torch.Tensor:
        """
        Align visual features with tags.
        
        Returns:
            Aligned features (B, N, hidden)
        """
        B = visual_features.shape[0]
        
        # Project
        v_proj = self.visual_proj(visual_features)  # (B, N, hidden)
        t_proj = self.text_proj(tag_features)       # (T, hidden)
        
        # Expand tags for batch
        t_proj = t_proj.unsqueeze(0).expand(B, -1, -1)  # (B, T, hidden)
        
        # Cross attention: visual queries, tag keys/values
        aligned, attn_weights = self.cross_attn(v_proj, t_proj, t_proj)
        aligned = self.norm(v_proj + aligned)
        
        # Output
        aligned = self.output_proj(aligned)
        
        return aligned, attn_weights


class MultiGranularityMatching(nn.Module):
    """
    Multi-granularity matching between visual and tag features.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_scales: int = 3,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2**i+1, padding=2**i//2, groups=hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for i in range(num_scales)
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_scales, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale feature processing.
        """
        scale_features = [proj(features) for proj in self.scale_projs]
        concat = torch.cat(scale_features, dim=1)
        fused = self.fusion(concat)
        return fused


@register_model("tagalign")
class TagAlign(TextGuidedSegmentationBase):
    """
    TagAlign: Tag-Guided Open-Vocabulary Semantic Segmentation
    
    Key features:
    - Tag-guided feature alignment
    - Multi-granularity matching
    - Efficient bridge between visual and text
    - Strong zero-shot performance
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_dim: int = 256,
        num_scales: int = 3,
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
        
        # Tag alignment
        self.tag_align = TagAlignmentModule(
            visual_dim=self.visual_dim,
            text_dim=self.clip_embed_dim,
            hidden_dim=hidden_dim,
        )
        
        # Multi-granularity matching
        self.multi_granular = MultiGranularityMatching(
            hidden_dim=hidden_dim,
            num_scales=num_scales,
        )
        
        # Dense decoder
        self.decoder = nn.Sequential(
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
        
        patch_tokens = x[:, 1:]  # Remove CLS
        
        return patch_tokens, patch_h, patch_w
    
    def encode_image(
        self,
        image: torch.Tensor,
        tag_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image with tag alignment."""
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            patch_tokens, h, w = self.extract_visual_features(image)
        
        B = patch_tokens.shape[0]
        
        # Tag alignment
        aligned, _ = self.tag_align(patch_tokens, tag_features)
        
        # Reshape to spatial
        aligned = aligned.permute(0, 2, 1).reshape(B, self.hidden_dim, h, w)
        
        # Multi-granularity matching
        multi_scale = self.multi_granular(aligned)
        
        # Decode
        decoded = self.decoder(multi_scale)
        
        # Output projection
        output = self.output_proj(decoded)
        
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
        
        # Encode text (used as tags)
        text_features = self.encode_text(text_prompts)
        
        # Encode image with tag alignment
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
