"""
OpenSeeD: A Simple Framework for Open-Vocabulary Segmentation and Detection
Paper: https://arxiv.org/abs/2303.08131 (ICCV 2023)
Official Implementation: https://github.com/IDEA-Research/OpenSeeD

Architecture:
- Unified framework for segmentation and detection
- Decoupled visual and semantic queries
- Interactive query design for open-vocabulary tasks
- Mask and box prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class SemanticQuery(nn.Module):
    """
    Semantic query module for text-conditioned queries.
    """
    
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        text_features: torch.Tensor,  # (C, D_text)
        learnable_queries: torch.Tensor,  # (Q, D)
    ) -> torch.Tensor:
        """
        Generate semantic queries from text.
        """
        # Project text
        text_proj = self.text_proj(text_features)  # (C, hidden)
        
        # Cross attention: queries attend to text
        Q = learnable_queries.shape[0]
        learnable_queries = learnable_queries.unsqueeze(0)  # (1, Q, D)
        text_proj = text_proj.unsqueeze(0)  # (1, C, D)
        
        queries, _ = self.attn(learnable_queries, text_proj, text_proj)
        queries = self.norm(learnable_queries + queries)
        
        queries = queries + self.ffn(self.norm2(queries))
        
        return queries.squeeze(0)  # (Q, D)


class VisualQuery(nn.Module):
    """
    Visual query module for learning from visual features.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        queries: torch.Tensor,        # (B, Q, D)
        visual_features: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        
        attended, _ = self.attn(queries, visual_features, visual_features)
        queries = self.norm(queries + attended)
        
        queries = queries + self.ffn(self.norm2(queries))
        
        return queries


class InteractiveDecoder(nn.Module):
    """
    Interactive decoder with semantic and visual queries.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_queries: int = 100,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Learnable queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        
        # Decoder layers
        self.visual_queries = nn.ModuleList([
            VisualQuery(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, N, D)
        semantic_queries: torch.Tensor,  # (Q, D)
    ) -> torch.Tensor:
        B = visual_features.shape[0]
        
        # Initialize with learnable queries + semantic queries
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        queries = queries + semantic_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Decode
        for visual_query, self_attn, norm in zip(
            self.visual_queries, self.self_attns, self.norms
        ):
            # Visual interaction
            queries = visual_query(queries, visual_features)
            
            # Self attention
            q = norm(queries)
            queries = queries + self_attn(q, q, q)[0]
        
        queries = self.output_norm(queries)
        
        return queries


class MaskHead(nn.Module):
    """Mask prediction head."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        queries: torch.Tensor,       # (B, Q, D)
        pixel_features: torch.Tensor,  # (B, D, H, W)
    ) -> torch.Tensor:
        
        mask_embeds = self.mask_mlp(queries)  # (B, Q, D)
        
        B, D, H, W = pixel_features.shape
        pixel_flat = pixel_features.reshape(B, D, -1)  # (B, D, H*W)
        
        masks = torch.bmm(mask_embeds, pixel_flat)  # (B, Q, H*W)
        masks = masks.reshape(B, -1, H, W)
        
        return masks


@register_model("openseed")
class OpenSeeD(TextGuidedSegmentationBase):
    """
    OpenSeeD: Open-Vocabulary Segmentation and Detection
    
    Key features:
    - Unified segmentation and detection
    - Decoupled semantic and visual queries
    - Interactive query design
    - Strong open-vocabulary performance
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
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
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Feature projections
        self.visual_proj = nn.Linear(self.visual_dim, hidden_dim)
        
        # Semantic query generator
        self.semantic_query = SemanticQuery(
            text_dim=self.clip_embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        
        # Learnable query basis
        self.query_basis = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        
        # Interactive decoder
        self.decoder = InteractiveDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            num_queries=num_queries,
        )
        
        # Pixel decoder
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(self.visual_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
        )
        
        # Mask head
        self.mask_head = MaskHead(hidden_dim)
        
        # Class head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.to(device)
    
    def extract_visual_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        patch_tokens = x[:, 1:]
        patch_spatial = patch_tokens.permute(0, 2, 1).reshape(B, self.visual_dim, patch_h, patch_w)
        
        return patch_tokens, patch_spatial
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image."""
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            patch_tokens, patch_spatial = self.extract_visual_features(image)
        
        # Project for decoder
        visual_feats = self.visual_proj(patch_tokens)
        
        # Pixel features
        pixel_feats = self.pixel_decoder(patch_spatial)
        
        return visual_feats, pixel_feats
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text."""
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
        
        # Encode
        visual_feats, pixel_feats = self.encode_image(image)
        text_feats = self.encode_text(text_prompts)
        
        # Generate semantic queries
        semantic_queries = self.semantic_query(text_feats, self.query_basis)
        
        # Decode
        queries = self.decoder(visual_feats, semantic_queries)
        
        # Predict masks
        masks = self.mask_head(queries, pixel_feats)
        
        # Class predictions via similarity
        class_embeds = self.class_head(queries)  # (B, Q, D)
        class_embeds = F.normalize(class_embeds, dim=-1)
        
        # Project text to hidden dim for similarity
        text_proj = nn.Linear(self.clip_embed_dim, self.hidden_dim).to(text_feats.device)
        text_hidden = text_proj(text_feats)
        text_hidden = F.normalize(text_hidden, dim=-1)
        
        class_logits = torch.einsum('bqd,cd->bqc', class_embeds, text_hidden)
        
        # Combine
        masks_up = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        class_probs = class_logits.softmax(dim=-1)
        
        logits = torch.einsum('bqhw,bqc->bchw', masks_up.sigmoid(), class_probs)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'masks': masks_up,
            'queries': queries,
            'class_logits': class_logits,
            'text_features': text_feats,
        }
