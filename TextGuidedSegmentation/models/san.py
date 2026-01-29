"""
SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation
Paper: https://arxiv.org/abs/2302.12242 (CVPR 2023)
Official Implementation: https://github.com/MendelXu/SAN

Architecture:
- Uses frozen CLIP for both visual and text encoding
- Side adapter network that processes CLIP features
- Attention-based fusion of multi-scale features
- Mask prediction through similarity computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D features."""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Adapter(nn.Module):
    """
    Adapter module for efficient fine-tuning.
    Bottleneck architecture: down-project -> nonlinearity -> up-project
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = None,
        out_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 4
        out_dim = out_dim or in_dim
        
        self.down = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, out_dim)
        
        # Initialize near-zero for gradual adaptation
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.act(self.down(x))))


class SideAdapterBlock(nn.Module):
    """
    Side adapter block for processing CLIP features.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(in_dim)
        self.adapter = Adapter(in_dim, hidden_dim, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(in_dim)
        self.cross_attn = nn.MultiheadAttention(
            in_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm3 = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 4, in_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        side_features: torch.Tensor,
        clip_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            side_features: Side network features (B, N, D)
            clip_features: CLIP features to attend to (B, M, D)
        """
        # Adapter path
        side_features = side_features + self.adapter(self.norm1(side_features))
        
        # Cross attention to CLIP features
        x = self.norm2(side_features)
        x, _ = self.cross_attn(x, clip_features, clip_features)
        side_features = side_features + x
        
        # MLP
        side_features = side_features + self.mlp(self.norm3(side_features))
        
        return side_features


class SideAdapterNetwork(nn.Module):
    """
    Side Adapter Network for processing CLIP features.
    """
    
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 8,
        num_queries: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Learnable queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        
        # Adapter blocks
        self.blocks = nn.ModuleList([
            SideAdapterBlock(in_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )
        
        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(
        self,
        clip_features: torch.Tensor,  # (B, N, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clip_features: CLIP visual features
            
        Returns:
            Query features and mask predictions
        """
        B = clip_features.shape[0]
        
        # Initialize queries
        queries = self.query_embed.expand(B, -1, -1)
        
        # Process through adapter blocks
        for block in self.blocks:
            queries = block(queries, clip_features)
        
        # Output projection
        queries = self.output_proj(queries)  # (B, Q, out_dim)
        
        # Mask features
        mask_features = self.mask_head(queries)  # (B, Q, out_dim)
        
        return queries, mask_features


class MaskDecoder(nn.Module):
    """
    Mask decoder for generating segmentation masks.
    """
    
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        
        # Upsampling path
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=2,
                    stride=2,
                ),
                LayerNorm2d(hidden_dim),
                nn.GELU(),
            ))
        
        # Final conv
        self.final_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, D, H, W)
            
        Returns:
            Upsampled features (B, D, H*8, W*8)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return x


@register_model("san")
class SAN(TextGuidedSegmentationBase):
    """
    SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation
    
    Key features:
    - Keeps CLIP frozen, only trains side adapter
    - Efficient adaptation with bottleneck adapters
    - Query-based mask prediction
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_dim: int = 256,
        num_adapter_blocks: int = 4,
        num_queries: int = 100,
        num_heads: int = 8,
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
        
        # Side adapter network
        self.side_adapter = SideAdapterNetwork(
            in_dim=self.visual_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_blocks=num_adapter_blocks,
            num_heads=num_heads,
            num_queries=num_queries,
        )
        
        # Text projection (to match query dimension)
        self.text_proj = nn.Linear(self.clip_embed_dim, hidden_dim)
        
        # Feature projection for dense prediction
        self.feat_proj = nn.Linear(self.visual_dim, hidden_dim)
        
        # Mask decoder
        self.mask_decoder = MaskDecoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
        )
        
        self.to(device)
    
    def extract_clip_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract all patch tokens from CLIP visual encoder."""
        visual = self.clip_model.visual
        
        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        cls_token = visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
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
        
        return x  # (B, N+1, D)
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image using CLIP + side adapter.
        """
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            clip_features = self.extract_clip_features(image)
        
        # Remove CLS for spatial features
        patch_features = clip_features[:, 1:]  # (B, N, D)
        
        # Side adapter processing
        queries, mask_features = self.side_adapter(clip_features)
        
        return queries, mask_features, patch_features
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode and project text."""
        text_features = self.encode_text_clip(text_prompts)
        text_features = self.text_proj(text_features)
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
        
        # Encode image
        queries, mask_features, patch_features = self.encode_image(image)
        
        # Encode text
        text_features = self.encode_text(text_prompts)  # (C, D)
        
        # Normalize for similarity computation
        mask_features_norm = F.normalize(mask_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        
        # Query-text similarity: (B, Q, C)
        query_text_sim = torch.einsum('bqd,cd->bqc', mask_features_norm, text_features_norm)
        
        # Project patch features and reshape to spatial
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        dense_features = self.feat_proj(patch_features)  # (B, N, D)
        dense_features = dense_features.permute(0, 2, 1).reshape(B, self.hidden_dim, patch_h, patch_w)
        
        # Upsample dense features
        dense_features = self.mask_decoder(dense_features)  # (B, D, H', W')
        
        # Compute masks through query-feature similarity
        # queries: (B, Q, D), dense_features: (B, D, H', W')
        queries_norm = F.normalize(queries, dim=-1)
        dense_norm = F.normalize(dense_features, dim=1)
        
        # (B, Q, H', W')
        query_masks = torch.einsum(
            'bqd,bdhw->bqhw',
            queries_norm,
            dense_norm
        )
        
        # Combine with text similarity to get class logits
        # (B, Q, C) @ (B, Q, H', W') summed over Q -> (B, C, H', W')
        logits = torch.einsum('bqc,bqhw->bchw', query_text_sim.softmax(dim=1), query_masks)
        
        # Upsample to original size
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'queries': queries,
            'mask_features': mask_features,
            'text_features': text_features,
        }
