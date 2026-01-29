"""
MAFT+: Multi-modal Adapter for Open-Vocabulary Semantic Segmentation
Paper: https://arxiv.org/abs/2403.XXXXX (ECCV 2024 Oral)
Official Implementation: https://github.com/jiaosiyu1999/MAFT-Plus

Architecture:
- Multi-modal adapter design for efficient fine-tuning
- Parallel adapters for visual and text encoders
- Cross-modal interaction through adapter fusion
- Achieves SOTA with minimal trainable parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class VisualAdapter(nn.Module):
    """
    Visual adapter for efficient CLIP fine-tuning.
    Bottleneck architecture with residual connection.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.down = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, in_dim)
        
        # Initialize near identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter_out = self.up(self.dropout(self.act(self.down(x))))
        return x + self.scale * adapter_out


class TextAdapter(nn.Module):
    """
    Text adapter for text encoder fine-tuning.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )
        
        # Initialize near identity
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.adapter(x)


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion module for visual-text interaction.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        
        # Visual to text
        self.v2t_attn = nn.MultiheadAttention(
            text_dim, num_heads, batch_first=True
        )
        self.v2t_proj = nn.Linear(visual_dim, text_dim)
        self.v2t_norm = nn.LayerNorm(text_dim)
        
        # Text to visual
        self.t2v_attn = nn.MultiheadAttention(
            visual_dim, num_heads, batch_first=True
        )
        self.t2v_proj = nn.Linear(text_dim, visual_dim)
        self.t2v_norm = nn.LayerNorm(visual_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, N, D_v)
        text_features: torch.Tensor,    # (C, D_t)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal fusion.
        """
        B, N, D_v = visual_features.shape
        C, D_t = text_features.shape
        
        # Expand text for batch
        text_expanded = text_features.unsqueeze(0).expand(B, -1, -1)  # (B, C, D_t)
        
        # Visual to text
        v_proj = self.v2t_proj(visual_features)  # (B, N, D_t)
        text_enhanced, _ = self.v2t_attn(text_expanded, v_proj, v_proj)
        text_enhanced = self.v2t_norm(text_expanded + text_enhanced)
        
        # Text to visual
        t_proj = self.t2v_proj(text_expanded)  # (B, C, D_v)
        visual_enhanced, _ = self.t2v_attn(visual_features, t_proj, t_proj)
        visual_enhanced = self.t2v_norm(visual_features + visual_enhanced)
        
        return visual_enhanced, text_enhanced


class AdaptedTransformerBlock(nn.Module):
    """
    Transformer block with parallel adapter.
    """
    
    def __init__(
        self,
        original_block,
        adapter_dim: int = 64,
    ):
        super().__init__()
        self.original_block = original_block
        
        # Get hidden dimension from block
        if hasattr(original_block, 'attn'):
            hidden_dim = original_block.attn.embed_dim
        else:
            hidden_dim = 768  # Default
        
        self.adapter = VisualAdapter(hidden_dim, adapter_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original block forward
        out = self.original_block(x)
        
        # Add adapter (applied to output)
        out = self.adapter(out.permute(1, 0, 2)).permute(1, 0, 2)
        
        return out


class DenseHead(nn.Module):
    """
    Dense prediction head for segmentation.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
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
        
        self.output = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.upsample(x)
        x = self.output(x)
        return x


@register_model("maft_plus")
class MAFTPlus(TextGuidedSegmentationBase):
    """
    MAFT+: Multi-modal Adapter for Open-Vocabulary Semantic Segmentation
    
    Key features:
    - Efficient adapter-based fine-tuning
    - Cross-modal fusion for visual-text interaction
    - Parallel adapters on frozen CLIP
    - SOTA performance with few trainable parameters
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        adapter_dim: int = 64,
        hidden_dim: int = 256,
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
        
        self.adapter_dim = adapter_dim
        self.hidden_dim = hidden_dim
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Visual adapters (applied after each transformer block)
        self.visual_adapters = nn.ModuleList([
            VisualAdapter(self.visual_dim, adapter_dim)
            for _ in range(len(self.clip_model.visual.transformer.resblocks))
        ])
        
        # Text adapter
        self.text_adapter = TextAdapter(self.clip_embed_dim, adapter_dim)
        
        # Cross-modal fusion
        self.cross_modal = CrossModalFusion(
            visual_dim=self.visual_dim,
            text_dim=self.clip_embed_dim,
            hidden_dim=hidden_dim,
        )
        
        # Dense head
        self.dense_head = DenseHead(self.visual_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, self.clip_embed_dim, kernel_size=1)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        
        self.to(device)
    
    def extract_adapted_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features with visual adapters."""
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
        
        x = x.permute(1, 0, 2)  # (N+1, B, D)
        
        # Apply adapters after each block
        for block, adapter in zip(visual.transformer.resblocks, self.visual_adapters):
            with torch.no_grad() if self.freeze_clip else torch.enable_grad():
                x = block(x)
            # Apply adapter
            x_adapted = adapter(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = x + x_adapted - x.detach() + x.detach()  # Gradient flows only through adapter
        
        x = x.permute(1, 0, 2)
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            x = visual.ln_post(x)
        
        # Remove CLS and reshape
        x = x[:, 1:].permute(0, 2, 1).reshape(B, self.visual_dim, patch_h, patch_w)
        
        return x
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image with adapters."""
        features = self.extract_adapted_features(image)
        
        # Dense head
        features = self.dense_head(features)
        features = self.output_proj(features)
        
        return features
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text with adapter."""
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            text_features = self.encode_text_clip(text_prompts)
        
        # Apply text adapter
        text_features = self.text_adapter(text_features)
        
        return text_features
    
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        B, _, H, W = image.shape
        
        # Encode
        visual_features = self.encode_image(image)  # (B, D, H', W')
        text_features = self.encode_text(text_prompts)  # (C, D)
        
        # Cross-modal fusion
        visual_flat = visual_features.reshape(B, self.clip_embed_dim, -1).permute(0, 2, 1)
        visual_enhanced, text_enhanced = self.cross_modal(visual_flat, text_features)
        
        # Reshape back
        h, w = visual_features.shape[2:]
        visual_enhanced = visual_enhanced.permute(0, 2, 1).reshape(B, self.clip_embed_dim, h, w)
        text_enhanced = text_enhanced.mean(dim=0)  # (C, D) - average over batch
        
        # Normalize
        visual_enhanced = F.normalize(visual_enhanced, dim=1)
        text_enhanced = F.normalize(text_enhanced, dim=-1)
        
        # Compute similarity
        logits = self.compute_similarity(visual_enhanced, text_enhanced, normalize=False)
        logits = logits * self.logit_scale.exp()
        
        # Upsample to original size
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'visual_features': visual_enhanced,
            'text_features': text_enhanced,
            'logit_scale': self.logit_scale,
        }
