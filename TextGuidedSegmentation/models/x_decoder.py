"""
X-Decoder: Generalized Decoding for Pixel, Image and Language Understanding
Paper: https://arxiv.org/abs/2212.11270 (CVPR 2023)
Official Implementation: https://github.com/microsoft/X-Decoder

Architecture:
- Unified decoder for multiple vision-language tasks
- Query-based architecture with text conditioning
- Supports segmentation, detection, and VQA
- End-to-end trainable with mixed task supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal positional embeddings."""
    
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device = x.device
        
        y_embed = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
        
        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # (D, H, W)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)
        
        return pos


class XDecoderLayer(nn.Module):
    """
    X-Decoder layer with self-attention and cross-attention.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross attention to visual features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross attention to text
        self.text_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        queries: torch.Tensor,       # (B, Q, D)
        visual_feats: torch.Tensor,  # (B, N, D)
        text_feats: torch.Tensor,    # (B, T, D)
        pos_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # Self attention
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        
        # Cross attention to visual
        q = self.norm2(queries)
        if pos_embed is not None:
            k = visual_feats + pos_embed
        else:
            k = visual_feats
        queries = queries + self.cross_attn(q, k, visual_feats)[0]
        
        # Cross attention to text
        q = self.norm3(queries)
        queries = queries + self.text_attn(q, text_feats, text_feats)[0]
        
        # FFN
        queries = queries + self.ffn(self.norm4(queries))
        
        return queries


class XDecoder(nn.Module):
    """
    X-Decoder: Unified decoder for pixel, image, and language.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        num_queries: int = 100,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Learnable queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            XDecoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        # Output norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        visual_feats: torch.Tensor,  # (B, N, D)
        text_feats: torch.Tensor,    # (B, T, D) or (T, D)
        pos_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        B = visual_feats.shape[0]
        
        # Handle text dimension
        if text_feats.dim() == 2:
            text_feats = text_feats.unsqueeze(0).expand(B, -1, -1)
        
        # Initialize queries
        queries = self.query_embed.expand(B, -1, -1)
        
        # Decode
        for layer in self.layers:
            queries = layer(queries, visual_feats, text_feats, pos_embed)
        
        queries = self.norm(queries)
        
        return queries


class MaskPredictor(nn.Module):
    """Predicts segmentation masks from queries."""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
    
    def forward(
        self,
        queries: torch.Tensor,       # (B, Q, D)
        pixel_feats: torch.Tensor,   # (B, D, H, W)
    ) -> torch.Tensor:
        
        mask_embeds = self.mask_embed(queries)  # (B, Q, D)
        
        # Dot product for mask prediction
        B, D, H, W = pixel_feats.shape
        pixel_flat = pixel_feats.reshape(B, D, -1)  # (B, D, H*W)
        
        masks = torch.bmm(mask_embeds, pixel_flat)  # (B, Q, H*W)
        masks = masks.reshape(B, -1, H, W)  # (B, Q, H, W)
        
        return masks


@register_model("x_decoder")
class XDecoderModel(TextGuidedSegmentationBase):
    """
    X-Decoder for open-vocabulary semantic segmentation.
    
    Key features:
    - Unified architecture for multiple tasks
    - Query-based decoding with text conditioning
    - Strong zero-shot transfer capability
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        d_model: int = 256,
        nhead: int = 8,
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
        
        self.d_model = d_model
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Project CLIP features to decoder dimension
        self.visual_proj = nn.Linear(self.visual_dim, d_model)
        self.text_proj = nn.Linear(self.clip_embed_dim, d_model)
        
        # Positional embedding
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        
        # X-Decoder
        self.decoder = XDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            num_queries=num_queries,
        )
        
        # Pixel decoder for dense features
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(self.visual_dim, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
        )
        
        # Mask predictor
        self.mask_predictor = MaskPredictor(d_model)
        
        # Class head
        self.class_head = nn.Linear(d_model, d_model)
        
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
        
        # Separate CLS and patches
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        
        # Reshape patches to spatial
        patch_spatial = patch_tokens.permute(0, 2, 1).reshape(B, self.visual_dim, patch_h, patch_w)
        
        return patch_tokens, patch_spatial
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image."""
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            patch_tokens, patch_spatial = self.extract_visual_features(image)
        
        # Project for decoder
        decoder_feats = self.visual_proj(patch_tokens)  # (B, N, d_model)
        
        # Pixel features for mask prediction
        pixel_feats = self.pixel_decoder(patch_spatial)  # (B, d_model, H, W)
        
        return decoder_feats, pixel_feats
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode and project text."""
        text_feats = self.encode_text_clip(text_prompts)
        text_feats = self.text_proj(text_feats)
        return text_feats
    
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
        decoder_feats, pixel_feats = self.encode_image(image)
        text_feats = self.encode_text(text_prompts)  # (C, d_model)
        
        # Positional embedding
        pos = self.pos_embed(pixel_feats)
        pos_flat = pos.reshape(B, self.d_model, -1).permute(0, 2, 1)
        
        # Decode
        queries = self.decoder(decoder_feats, text_feats, pos_flat)  # (B, Q, d_model)
        
        # Predict masks
        masks = self.mask_predictor(queries, pixel_feats)  # (B, Q, H', W')
        
        # Query-text similarity for class assignment
        query_embeds = self.class_head(queries)  # (B, Q, d_model)
        query_embeds = F.normalize(query_embeds, dim=-1)
        text_feats_norm = F.normalize(text_feats, dim=-1)
        
        class_logits = torch.einsum('bqd,cd->bqc', query_embeds, text_feats_norm)  # (B, Q, C)
        
        # Combine masks and class predictions
        # Weight masks by class probability
        class_probs = class_logits.softmax(dim=-1)  # (B, Q, C)
        
        # Upsample masks
        masks_up = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        
        # Weighted combination: (B, Q, H, W) * (B, Q, C) -> (B, C, H, W)
        logits = torch.einsum('bqhw,bqc->bchw', masks_up.sigmoid(), class_probs)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'masks': masks_up,
            'queries': queries,
            'class_logits': class_logits,
            'text_features': text_feats,
        }
