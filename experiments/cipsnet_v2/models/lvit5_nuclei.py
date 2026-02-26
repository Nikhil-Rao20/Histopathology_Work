"""
LViT5: Ultimate Language-guided Vision Transformer for Nuclei Instance Segmentation
====================================================================================

Combines ALL improvements from previous versions + new grounding-aware components:

From LViT3 (Phase 1):
    - Instance Normalization in all cross-modal modules
    
From LViT4 (Phase 2):
    - Multi-stage Early Fusion (text injection at ViT layers 3, 6, 9, 12)
    - Pixel-Word Attention Module (PWAM) for dense text-pixel alignment
    - Multi-scale Contrastive features

NEW in LViT5 (Phase 3):
    - Cross-Modal Decoder with text guidance at every stage
    - Pixel-level Contrastive Loss (not just batch-level)
    - Hard Negative Mining for contrastive learning
    - Grounding Head for explicit text-region alignment
    - Auxiliary Grounding Loss with direct IoU supervision

Key Fixes:
    - Pixel-level contrastive loss instead of batch-level
    - Hard negative mining for better training signal
    - Stronger loss weighting scheme

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Input                                     │
    │         Image [B, 3, H, W]    Text ["segment neoplastic..."]    │
    └─────────────────────────────────────────────────────────────────┘
                │                                │
                ▼                                ▼
    ┌───────────────────┐            ┌──────────────────────┐
    │ ViT-B/16 Encoder  │            │ BioClinicalBERT      │
    │ + Multi-stage     │◄───────────│ + Word Embeddings    │
    │   Text Fusion     │            │ + Sentence Embed     │
    │ (layers 3,6,9,12) │            │ + Contrastive Proj   │
    └───────────────────┘            └──────────────────────┘
                │                                │
                ▼                                │
    ┌───────────────────┐                       │
    │ Skip Connections  │                       │
    │ with InstanceNorm │                       │
    └───────────────────┘                       │
                │                               │
                ▼                               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                Cross-Modal Decoder (NEW)                         │
    │  Each stage has:                                                 │
    │   - Upsample + Skip connection                                   │
    │   - Language-guided Feature Enhancement (PWAM)                   │
    │   - Pixel-wise contrastive features                              │
    └─────────────────────────────────────────────────────────────────┘
                │
                ├──────────────────────────────────────────┐
                ▼                                          ▼
    ┌───────────────────┐                    ┌─────────────────────────┐
    │  Segmentation     │                    │  Grounding Head (NEW)   │
    │  Output Heads     │                    │  - Text-Region IoU      │
    │  (NP, HV, Type)   │                    │  - Pixel Grounding Map  │
    └───────────────────┘                    └─────────────────────────┘

Reference:
    - LAVT: "Language-Aware Vision Transformer" (CVPR 2022)
    - CRIS: "CLIP-Driven Referring Image Segmentation" (CVPR 2022)
    - PromptNucSeg: "Text-guided Nuclei Segmentation" (MICCAI 2024)

Author: CIPS-Net V2 - Phase 3 (Ultimate Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============================================================
# Text Encoder with Enhanced Projections
# ============================================================

class LViT5TextEncoder(nn.Module):
    """
    Text encoder with multiple projection heads for different losses.
    
    Returns:
        - sentence_embed: For global image-text alignment
        - word_embeds: For PWAM and dense alignment
        - contrastive_embed: L2-normalized for contrastive loss
        - grounding_embed: For grounding head supervision
    """
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        freeze: bool = True,
        max_length: int = 128
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for BERT
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Main projection
        self.text_proj = nn.Linear(self.hidden_size, embed_dim)
        
        # Contrastive projection head (for pixel-level contrastive loss)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Grounding projection head (for grounding-aware loss)
        self.grounding_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode text with multiple projection heads.
        
        Returns:
            Dict with:
                - sentence_embed: [B, embed_dim]
                - word_embeds: [B, L, embed_dim]  
                - attention_mask: [B, L]
                - contrastive_embed: [B, embed_dim] (L2 normalized)
                - grounding_embed: [B, embed_dim]
        """
        device = next(self.parameters()).device
        
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        
        outputs = self.encoder(**tokens)
        hidden_states = outputs.last_hidden_state  # [B, L, 768]
        
        # Project to common dimension
        word_embeds = self.text_proj(hidden_states)  # [B, L, embed_dim]
        
        # Sentence embedding (CLS token)
        sentence_embed = word_embeds[:, 0, :]  # [B, embed_dim]
        
        # Contrastive embedding (L2 normalized)
        contrastive_embed = self.contrastive_proj(sentence_embed)
        contrastive_embed = F.normalize(contrastive_embed, p=2, dim=-1)
        
        # Grounding embedding
        grounding_embed = self.grounding_proj(sentence_embed)
        
        return {
            'sentence_embed': sentence_embed,
            'word_embeds': word_embeds,
            'attention_mask': tokens['attention_mask'],
            'contrastive_embed': contrastive_embed,
            'grounding_embed': grounding_embed,
        }


# ============================================================
# Multi-stage Fusion ViT Encoder (from LVIT4)
# ============================================================

class LanguageAwareViTBlock(nn.Module):
    """
    ViT block with language injection (from LAVT).
    Uses Instance Normalization for better cross-modal fusion.
    """
    
    def __init__(
        self,
        vit_dim: int = 768,
        text_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Cross-attention: visual attends to text
        self.cross_attn = nn.MultiheadAttention(
            vit_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        # Text projection
        self.text_proj = nn.Linear(text_dim, vit_dim)
        
        # Instance Norm for better fusion (from LAVT paper)
        self.norm1 = nn.LayerNorm(vit_dim)
        self.norm2 = nn.LayerNorm(vit_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(vit_dim * 4, vit_dim),
            nn.Dropout(0.1)
        )
        
        # Learnable gate for text influence
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        visual: torch.Tensor,
        word_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inject language into visual features.
        
        Args:
            visual: [B, N+1, D] visual tokens
            word_embeds: [B, L, D_text] word embeddings
            attention_mask: [B, L] text attention mask
        """
        text_proj = self.text_proj(word_embeds)
        visual_norm = self.norm1(visual)
        
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        attn_out, _ = self.cross_attn(
            visual_norm, text_proj, text_proj,
            key_padding_mask=key_padding_mask
        )
        
        # Gated residual
        visual = visual + torch.sigmoid(self.gate) * attn_out
        visual = visual + self.ffn(self.norm2(visual))
        
        return visual


class MultiStageFusionViTEncoder(nn.Module):
    """
    ViT encoder with multi-stage language fusion (from LAVT).
    
    Injects text at layers 3, 6, 9, 12 for progressive alignment.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 256,
        text_dim: int = 512,
        fusion_layers: List[int] = [3, 6, 9, 12]
    ):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.hidden_dim = 768
        self.patch_size = 16
        self.img_size = img_size
        self.fusion_layers = fusion_layers
        
        self.target_h = self.target_w = img_size // self.patch_size
        self.vit.heads = nn.Identity()
        
        # Language-aware blocks at fusion layers
        self.fusion_blocks = nn.ModuleDict({
            str(layer): LanguageAwareViTBlock(
                vit_dim=self.hidden_dim,
                text_dim=text_dim,
                num_heads=8
            ) for layer in fusion_layers
        })
        
        # Skip connection projections with Instance Norm
        self.skip_proj_deep = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 512, 1),
            nn.InstanceNorm2d(512, affine=True),
        )
        self.skip_proj_mid = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, 1),
            nn.InstanceNorm2d(256, affine=True),
        )
        
        # Early feature extractors with Instance Norm
        self.early_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.early_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def _interpolate_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        """Interpolate position embeddings for different image sizes."""
        N = pos_embed.shape[1] - 1
        if N == self.target_h * self.target_w:
            return pos_embed
        
        cls_token = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        
        dim = pos_embed.shape[-1]
        h0 = w0 = int(math.sqrt(N))
        
        patch_pos = patch_pos.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(self.target_h, self.target_w), 
            mode='bicubic', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat([cls_token, patch_pos], dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        word_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-stage text fusion.
        
        Returns hierarchical features for decoder.
        """
        B = x.shape[0]
        
        # Early features for skip connections
        skip1 = self.early_conv1(x)  # [B, 64, H/2, W/2]
        skip2 = self.early_conv2(skip1)  # [B, 128, H/4, W/4]
        
        # ViT patch embedding
        x = self.vit.conv_proj(x)  # [B, 768, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, N, 768]
        
        # Add CLS token
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Interpolate position embeddings
        pos_embed = self._interpolate_pos_embed(self.vit.encoder.pos_embedding)
        x = x + pos_embed
        
        x = self.vit.encoder.dropout(x)
        
        # Process through ViT blocks with language fusion
        intermediate_features = {}
        
        for i, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            layer_num = i + 1
            
            # Apply language fusion at specified layers
            if layer_num in self.fusion_layers:
                x = self.fusion_blocks[str(layer_num)](
                    x, word_embeds, attention_mask
                )
                intermediate_features[f'layer_{layer_num}'] = x.clone()
        
        # Extract skip features from intermediate layers
        vit_tokens = x[:, 1:, :]  # Remove CLS
        H = W = self.target_h
        vit_spatial = vit_tokens.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        
        # Get mid and deep features from stored intermediates
        if 'layer_6' in intermediate_features:
            mid_tokens = intermediate_features['layer_6'][:, 1:, :]
            mid_spatial = mid_tokens.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
            skip3 = self.skip_proj_mid(mid_spatial)  # [B, 256, H/16, W/16]
        else:
            skip3 = self.skip_proj_mid(vit_spatial)
        
        skip4 = self.skip_proj_deep(vit_spatial)  # [B, 512, H/16, W/16]
        
        return {
            'vit_out': vit_spatial,  # [B, 768, H/16, W/16]
            'skip1': skip1,  # [B, 64, H/2, W/2]
            'skip2': skip2,  # [B, 128, H/4, W/4]
            'skip3': skip3,  # [B, 256, H/16, W/16]
            'skip4': skip4,  # [B, 512, H/16, W/16]
            'cls_token': x[:, 0, :],  # [B, 768]
        }


# ============================================================
# Pixel-Word Attention Module (PWAM) - Enhanced from LVIT4
# ============================================================

class PixelWordAttention(nn.Module):
    """
    Pixel-Word Attention Module with Instance Normalization.
    
    Each pixel attends to all words to get dense alignment.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Visual projection with Instance Norm
        self.visual_proj = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, 1),
            nn.InstanceNorm2d(hidden_dim, affine=True),
        )
        
        # Text projections
        self.text_key_proj = nn.Linear(text_dim, hidden_dim)
        self.text_value_proj = nn.Linear(text_dim, hidden_dim)
        
        # Output projection with Instance Norm
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, visual_dim, 1),
            nn.InstanceNorm2d(visual_dim, affine=True),
        )
        
        # Gating
        self.gate = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        visual: torch.Tensor,
        word_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply pixel-word attention.
        
        Returns:
            enhanced_visual: [B, C_v, H, W]
            attention_maps: [B, H*W, L] for grounding supervision
        """
        B, C_v, H, W = visual.shape
        L = word_embeds.shape[1]
        
        # Project visual features
        visual_proj = self.visual_proj(visual)
        queries = visual_proj.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]
        
        # Project text
        keys = self.text_key_proj(word_embeds)  # [B, L, hidden_dim]
        values = self.text_value_proj(word_embeds)
        
        # Multi-head attention
        queries = queries.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Store attention maps for grounding loss (average over heads)
        attn_maps = attn.mean(dim=1)  # [B, H*W, L]
        
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, values)
        out = out.transpose(1, 2).contiguous().view(B, H*W, self.hidden_dim)
        out = self.norm(out)
        
        # Reshape and project
        out = out.transpose(1, 2).view(B, self.hidden_dim, H, W)
        out = self.out_proj(out)
        
        # Gated residual
        gate = self.gate(torch.cat([visual, out], dim=1))
        enhanced = visual + gate * out
        
        return enhanced, attn_maps


# ============================================================
# Cross-Modal Decoder (NEW in LViT5)
# ============================================================

class CrossModalDecoderBlock(nn.Module):
    """
    Decoder block with cross-modal fusion at every stage.
    
    Each block:
    1. Upsamples features
    2. Fuses with skip connection
    3. Applies PWAM for text-guided refinement
    4. Outputs pixel-level features for contrastive loss
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        text_dim: int = 512,
        upsample: bool = True
    ):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        
        # Skip fusion with Instance Norm
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # PWAM for text guidance
        self.pwam = PixelWordAttention(
            visual_dim=out_channels,
            text_dim=text_dim,
            hidden_dim=min(256, out_channels),
            num_heads=8
        )
        
        # Refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Pixel-level feature projection for contrastive loss
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(out_channels, text_dim, 1),
            nn.InstanceNorm2d(text_dim, affine=True),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        word_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            decoded: [B, out_channels, H, W]
            pixel_features: [B, text_dim, H, W] for contrastive loss
            attn_maps: [B, H*W, L] for grounding loss
        """
        # Upsample
        x = self.upsample(x)
        
        # Fuse with skip (handle size mismatch)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.skip_fusion(x)
        
        # Text-guided refinement
        x, attn_maps = self.pwam(x, word_embeds, attention_mask)
        x = self.refine(x)
        
        # Pixel-level features for contrastive loss
        pixel_features = self.pixel_proj(x)
        
        return x, pixel_features, attn_maps


class CrossModalDecoder(nn.Module):
    """
    Full cross-modal decoder with text guidance at every stage.
    """
    
    def __init__(self, text_dim: int = 512):
        super().__init__()
        
        # Bottleneck processing
        self.bottleneck = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Decoder blocks (progressively upsample)
        self.block1 = CrossModalDecoderBlock(512, 512, 512, text_dim, upsample=True)   # 16->32
        self.block2 = CrossModalDecoderBlock(512, 256, 256, text_dim, upsample=True)   # 32->64
        self.block3 = CrossModalDecoderBlock(256, 128, 128, text_dim, upsample=True)   # 64->128
        self.block4 = CrossModalDecoderBlock(128, 64, 64, text_dim, upsample=True)     # 128->256
    
    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        word_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decode with cross-modal fusion at every stage.
        
        Returns dict with:
            - decoded: Final decoded features
            - pixel_features_X: Pixel features at each scale for contrastive
            - attn_maps_X: Attention maps at each scale for grounding
        """
        x = self.bottleneck(encoder_features['vit_out'])
        
        x, pf1, am1 = self.block1(x, encoder_features['skip4'], word_embeds, attention_mask)
        x, pf2, am2 = self.block2(x, encoder_features['skip3'], word_embeds, attention_mask)
        x, pf3, am3 = self.block3(x, encoder_features['skip2'], word_embeds, attention_mask)
        x, pf4, am4 = self.block4(x, encoder_features['skip1'], word_embeds, attention_mask)
        
        return {
            'decoded': x,
            'pixel_features_deep': pf1,
            'pixel_features_mid': pf2,
            'pixel_features_shallow': pf3,
            'pixel_features_out': pf4,
            'attn_maps_deep': am1,
            'attn_maps_mid': am2,
            'attn_maps_shallow': am3,
            'attn_maps_out': am4,
        }


# ============================================================
# Grounding Head (NEW in LViT5)
# ============================================================

class GroundingHead(nn.Module):
    """
    Grounding head for explicit text-region alignment supervision.
    
    Predicts a grounding map that indicates which pixels correspond
    to the described nucleus type.
    """
    
    def __init__(self, in_channels: int = 64, text_dim: int = 512, upsample_factor: int = 2):
        super().__init__()
        
        self.upsample_factor = upsample_factor
        
        # Feature projection
        self.visual_proj = nn.Sequential(
            nn.Conv2d(in_channels, text_dim, 1),
            nn.InstanceNorm2d(text_dim, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Grounding prediction
        self.grounding_conv = nn.Sequential(
            nn.Conv2d(text_dim, text_dim // 2, 3, padding=1),
            nn.InstanceNorm2d(text_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(text_dim // 2, 1, 1),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        grounding_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict grounding map.
        
        Args:
            visual_features: [B, C, H, W] decoder features
            grounding_embed: [B, text_dim] text grounding embedding
            
        Returns:
            grounding_map: [B, 1, H, W] probability map (upsampled to full resolution)
        """
        B, _, H, W = visual_features.shape
        
        # Project visual features
        visual_proj = self.visual_proj(visual_features)  # [B, text_dim, H, W]
        
        # Modulate with text embedding
        grounding_embed = grounding_embed.unsqueeze(-1).unsqueeze(-1)  # [B, text_dim, 1, 1]
        modulated = visual_proj * grounding_embed  # [B, text_dim, H, W]
        
        # Predict grounding map
        grounding_map = self.grounding_conv(modulated)  # [B, 1, H, W]
        grounding_map = torch.sigmoid(grounding_map)
        
        # Upsample to full resolution
        if self.upsample_factor > 1:
            grounding_map = F.interpolate(grounding_map, scale_factor=self.upsample_factor, 
                                         mode='bilinear', align_corners=False)
        
        return grounding_map


# ============================================================
# Output Heads
# ============================================================

class OutputHeads(nn.Module):
    """Segmentation output heads for NP, HV, and Type prediction."""
    
    def __init__(self, in_channels: int = 64, num_classes: int = 6, upsample_factor: int = 2):
        super().__init__()
        
        self.upsample_factor = upsample_factor
        
        self.np_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        
        self.hv_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        
        self.type_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        np_out = self.np_head(x)
        hv_out = self.hv_head(x)
        type_out = self.type_head(x)
        
        # Upsample to full resolution if needed
        if self.upsample_factor > 1:
            np_out = F.interpolate(np_out, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
            hv_out = F.interpolate(hv_out, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
            type_out = F.interpolate(type_out, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
        
        return {
            'np': np_out,
            'hv': hv_out,
            'type': type_out,
        }


# ============================================================
# LViT5 Main Model
# ============================================================

class LViT5NucleiSegmenter(nn.Module):
    """
    LViT5: Ultimate Language-guided Vision Transformer for Nuclei Segmentation.
    
    Combines:
    - Multi-stage Early Fusion ViT encoder (LAVT)
    - Cross-Modal Decoder with PWAM at every stage
    - Pixel-level contrastive features
    - Grounding head for explicit alignment
    
    This is the strongest version with all improvements.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        img_size: int = 256,
        embed_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Text encoder
        self.text_encoder = LViT5TextEncoder(
            embed_dim=embed_dim,
            freeze=True
        )
        
        # Multi-stage fusion ViT encoder
        self.image_encoder = MultiStageFusionViTEncoder(
            pretrained=pretrained,
            img_size=img_size,
            text_dim=embed_dim,
            fusion_layers=[3, 6, 9, 12]
        )
        
        # Cross-modal decoder
        self.decoder = CrossModalDecoder(text_dim=embed_dim)
        
        # Output heads
        self.output_heads = OutputHeads(in_channels=64, num_classes=num_classes)
        
        # Grounding head
        self.grounding_head = GroundingHead(in_channels=64, text_dim=embed_dim)
        
        # Learnable temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Global feature projection for batch contrastive
        self.global_visual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self._init_weights()
        self._print_info()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _print_info(self):
        """Print model info."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n[LViT5] Ultimate Language-guided Nuclei Segmenter")
        print(f"[LViT5] Total parameters: {total:,}")
        print(f"[LViT5] Trainable parameters: {trainable:,}")
        print(f"[LViT5] Features:")
        print(f"  - Multi-stage Early Fusion (layers 3, 6, 9, 12)")
        print(f"  - Cross-Modal Decoder with PWAM at every stage")
        print(f"  - Pixel-level Contrastive Features")
        print(f"  - Grounding Head for explicit alignment")
        print(f"  - Instance Normalization throughout")
        print(f"  - Learnable temperature: {self.temperature.item():.4f}")
    
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        return_contrastive_features: bool = True,
        return_grounding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W] input images
            texts: List of text instructions
            return_contrastive_features: Whether to return features for contrastive loss
            return_grounding: Whether to return grounding predictions
            
        Returns:
            Dict with:
                - np, hv, type: Segmentation outputs
                - grounding_map: [B, 1, H, W] grounding prediction
                - pixel_features_*: Multi-scale pixel features
                - attn_maps_*: Multi-scale attention maps
                - contrastive_text, contrastive_visual: Batch-level contrastive features
                - temperature: Learnable temperature
        """
        # Encode text
        text_out = self.text_encoder(texts)
        
        # Encode image with multi-stage text fusion
        encoder_features = self.image_encoder(
            images,
            text_out['word_embeds'],
            text_out['attention_mask']
        )
        
        # Decode with cross-modal fusion
        decoder_out = self.decoder(
            encoder_features,
            text_out['word_embeds'],
            text_out['attention_mask']
        )
        
        # Segmentation outputs
        outputs = self.output_heads(decoder_out['decoded'])
        
        # Add grounding prediction
        if return_grounding:
            grounding_map = self.grounding_head(
                decoder_out['decoded'],
                text_out['grounding_embed']
            )
            outputs['grounding_map'] = grounding_map
        
        # Add contrastive and grounding features
        if return_contrastive_features:
            outputs['contrastive_text'] = text_out['contrastive_embed']
            
            # Global visual feature for batch contrastive
            global_visual = self.global_visual_proj(decoder_out['decoded'])
            outputs['contrastive_visual'] = F.normalize(global_visual, p=2, dim=-1)
            
            # Pixel-level features for pixel contrastive
            outputs['pixel_features_deep'] = decoder_out['pixel_features_deep']
            outputs['pixel_features_mid'] = decoder_out['pixel_features_mid']
            outputs['pixel_features_shallow'] = decoder_out['pixel_features_shallow']
            outputs['pixel_features_out'] = decoder_out['pixel_features_out']
            
            # Attention maps for grounding loss
            outputs['attn_maps_deep'] = decoder_out['attn_maps_deep']
            outputs['attn_maps_mid'] = decoder_out['attn_maps_mid']
            outputs['attn_maps_shallow'] = decoder_out['attn_maps_shallow']
            outputs['attn_maps_out'] = decoder_out['attn_maps_out']
            
            # Temperature
            outputs['temperature'] = self.temperature
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_lvit5_model(
    num_classes: int = 6,
    img_size: int = 256,
    pretrained: bool = True,
    device: str = 'cuda'
) -> LViT5NucleiSegmenter:
    """
    Create LViT5 model.
    
    Args:
        num_classes: Number of nucleus types
        img_size: Input image size
        pretrained: Use pretrained ViT
        device: Target device
        
    Returns:
        LViT5NucleiSegmenter instance
    """
    model = LViT5NucleiSegmenter(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
    )
    
    return model.to(device)


# ============================================================
# Testing
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("LViT5 Model Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_lvit5_model(num_classes=6, device=str(device))
    
    # Test forward pass
    B = 2
    images = torch.randn(B, 3, 256, 256, device=device)
    texts = ["segment neoplastic cells", "segment inflammatory cells"]
    
    with torch.no_grad():
        outputs = model(images, texts)
    
    print(f"\nOutput shapes:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("✅ LViT5 Model test passed!")
    print("=" * 70)
