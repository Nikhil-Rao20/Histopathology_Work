"""
LViT4: Advanced Language-guided Vision Transformer for Nuclei Instance Segmentation
=====================================================================================

Phase 2 improvements building on LViT3:

Key Innovations:
    1. Multi-stage Early Fusion (from LAVT paper)
       - Inject text guidance at multiple ViT layers (3, 6, 9, 12)
       - Not just at bottleneck - enables progressive text-visual alignment
       
    2. Pixel-Word Attention Module (PWAM)
       - Dense attention between every pixel and every word token
       - Enables fine-grained grounding of text to specific image regions
       - Applied at decoder stages for multi-scale alignment
       
    3. Enhanced Contrastive Learning
       - Learnable temperature parameter
       - Multi-scale contrastive features from decoder
       
    4. All Phase 1 improvements (Instance Norm, Contrastive Loss)

Architecture:
    Encoder: ViT-B/16 with text injection at layers 3, 6, 9, 12
    Decoder: U-Net style with PWAM at each level
    Output: HoVer-Net style (NP, HV, Type) + contrastive features

Reference:
    - LAVT: "Language-Aware Vision Transformer for Referring Image Segmentation" (CVPR 2022)
    - CRIS: "CLIP-Driven Referring Image Segmentation" (CVPR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============================================================
# Text Encoder with Word-Level Features
# ============================================================

class LViT4TextEncoder(nn.Module):
    """
    Text encoder for LViT4.
    Returns sentence embedding, word-level features, and contrastive projection.
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
        
        # Project to common embedding dimension
        self.text_proj = nn.Linear(self.hidden_size, embed_dim)
        
        # Word-level projection for PWAM (projects to ViT dimension)
        self.word_proj = nn.Linear(self.hidden_size, 768)  # ViT hidden dim
        
        # Contrastive projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode text.
        
        Returns:
            Dict with:
                - sentence_embed: [B, embed_dim]
                - token_embeds: [B, L, embed_dim]
                - word_embeds: [B, L, 768] for PWAM
                - contrastive_embed: [B, embed_dim] normalized
                - attention_mask: [B, L] for masking padding
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
        token_embeds = self.text_proj(hidden_states)  # [B, L, embed_dim]
        
        # Word-level features for PWAM (project to ViT dim)
        word_embeds = self.word_proj(hidden_states)  # [B, L, 768]
        
        # Sentence embedding (CLS token)
        sentence_embed = token_embeds[:, 0, :]  # [B, embed_dim]
        
        # Contrastive embedding (L2 normalized)
        contrastive_embed = self.contrastive_proj(sentence_embed)
        contrastive_embed = F.normalize(contrastive_embed, p=2, dim=-1)
        
        return {
            'sentence_embed': sentence_embed,
            'token_embeds': token_embeds,
            'word_embeds': word_embeds,
            'contrastive_embed': contrastive_embed,
            'attention_mask': tokens.attention_mask,
        }


# ============================================================
# Pixel-Word Attention Module (PWAM)
# ============================================================

class PixelWordAttentionModule(nn.Module):
    """
    Pixel-Word Attention Module (PWAM).
    
    Computes dense attention between every pixel and every word token.
    This enables fine-grained grounding of text descriptions to image regions.
    
    For each pixel p and word w:
        attention(p, w) = softmax(proj_p(p) · proj_w(w)^T / sqrt(d))
        output(p) = sum_w(attention(p, w) * value(w))
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Visual projection (pixel queries)
        self.visual_proj = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, 1),
            nn.InstanceNorm2d(hidden_dim, affine=True),
        )
        
        # Text projection (word keys and values)
        self.text_key_proj = nn.Linear(text_dim, hidden_dim)
        self.text_value_proj = nn.Linear(text_dim, hidden_dim)
        
        # Output projection with Instance Norm
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, visual_dim, 1),
            nn.InstanceNorm2d(visual_dim, affine=True),
        )
        
        # Gating mechanism
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
    ) -> torch.Tensor:
        """
        Apply pixel-word attention.
        
        Args:
            visual: [B, C_v, H, W] visual features
            word_embeds: [B, L, C_t] word-level text embeddings
            attention_mask: [B, L] mask for padding tokens (1 = valid, 0 = pad)
            
        Returns:
            [B, C_v, H, W] enhanced visual features
        """
        B, C_v, H, W = visual.shape
        L = word_embeds.shape[1]
        
        # Project visual features to queries
        visual_proj = self.visual_proj(visual)  # [B, hidden_dim, H, W]
        queries = visual_proj.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]
        
        # Project text to keys and values
        keys = self.text_key_proj(word_embeds)  # [B, L, hidden_dim]
        values = self.text_value_proj(word_embeds)  # [B, L, hidden_dim]
        
        # Reshape for multi-head attention
        queries = queries.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, H*W, head_dim]
        keys = keys.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, L, head_dim]
        values = values.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, L, head_dim]
        
        # Compute attention scores
        attn = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale  # [B, heads, H*W, L]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask: [B, L] -> [B, 1, 1, L]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, values)  # [B, heads, H*W, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, H*W, self.hidden_dim)  # [B, H*W, hidden_dim]
        out = self.norm(out)
        
        # Reshape back to spatial
        out = out.transpose(1, 2).view(B, self.hidden_dim, H, W)  # [B, hidden_dim, H, W]
        out = self.out_proj(out)  # [B, C_v, H, W]
        
        # Gated residual connection
        gate = self.gate(torch.cat([visual, out], dim=1))
        output = visual + gate * out
        
        return output


# ============================================================
# Language-Aware ViT Block (for Multi-stage Fusion)
# ============================================================

class LanguageAwareViTBlock(nn.Module):
    """
    ViT block with language injection.
    
    Injects text information into the visual processing stream.
    Used at multiple layers (3, 6, 9, 12) for progressive alignment.
    """
    
    def __init__(
        self,
        vit_dim: int = 768,
        text_dim: int = 768,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Cross-attention: visual attends to text
        self.cross_attn = nn.MultiheadAttention(
            vit_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        # Text projection to match ViT dimension
        self.text_proj = nn.Linear(text_dim, vit_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(vit_dim)
        self.norm2 = nn.LayerNorm(vit_dim)
        
        # FFN after fusion
        self.ffn = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(vit_dim * 4, vit_dim),
            nn.Dropout(0.1)
        )
        
        # Learnable gate for controlling text influence
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
            visual: [B, N+1, D] visual tokens (with CLS)
            word_embeds: [B, L, D_text] word embeddings
            attention_mask: [B, L] text attention mask
            
        Returns:
            [B, N+1, D] language-enhanced visual tokens
        """
        # Project text to ViT dimension
        text_proj = self.text_proj(word_embeds)  # [B, L, vit_dim]
        
        # Cross-attention with text
        visual_norm = self.norm1(visual)
        
        # Create key padding mask for attention
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True = ignore
        
        attn_out, _ = self.cross_attn(
            visual_norm, text_proj, text_proj,
            key_padding_mask=key_padding_mask
        )
        
        # Gated residual
        visual = visual + torch.sigmoid(self.gate) * attn_out
        
        # FFN
        visual = visual + self.ffn(self.norm2(visual))
        
        return visual


# ============================================================
# Multi-stage Fusion ViT Encoder
# ============================================================

class MultiStageFusionViTEncoder(nn.Module):
    """
    ViT encoder with multi-stage language fusion.
    
    Injects text at layers 3, 6, 9, 12 for progressive alignment.
    Also extracts hierarchical features for U-Net decoder.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 256,
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
        
        # Store target grid size
        self.target_h = self.target_w = img_size // self.patch_size
        
        # Remove classification head
        self.vit.heads = nn.Identity()
        
        # Language-aware blocks for fusion layers
        self.lang_blocks = nn.ModuleDict({
            str(layer): LanguageAwareViTBlock(
                vit_dim=768,
                text_dim=768,  # word_embeds dimension
                num_heads=8
            ) for layer in fusion_layers
        })
        
        # CNN skip connections with Instance Norm
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def _interpolate_pos_embed(self, device):
        """Interpolate positional embeddings for new image size."""
        old_pos_embed = self.vit.encoder.pos_embedding.to(device)
        
        new_h, new_w = self.target_h, self.target_w
        num_patches = new_h * new_w
        
        cls_embed = old_pos_embed[:, 0:1, :]
        patch_embed = old_pos_embed[:, 1:, :]
        
        old_h = old_w = 14
        patch_embed = patch_embed.reshape(1, old_h, old_w, self.hidden_dim)
        patch_embed = patch_embed.permute(0, 3, 1, 2)
        
        patch_embed = F.interpolate(
            patch_embed, size=(new_h, new_w), mode='bicubic', align_corners=False
        )
        
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, num_patches, self.hidden_dim)
        new_pos_embed = torch.cat([cls_embed, patch_embed], dim=1)
        
        return new_pos_embed
    
    def forward(
        self,
        x: torch.Tensor,
        word_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features with multi-stage language fusion.
        
        Args:
            x: [B, 3, H, W] input image
            word_embeds: [B, L, 768] word-level embeddings
            attention_mask: [B, L] text attention mask
            
        Returns:
            Dictionary with features at different scales
        """
        B, C, H, W = x.shape
        device = x.device
        
        # CNN skip features
        f1 = self.downsample1(x)   # [B, 64, H/2, W/2]
        f2 = self.downsample2(f1)  # [B, 128, H/4, W/4]
        f3 = self.downsample3(f2)  # [B, 256, H/8, W/8]
        
        # ViT patch embedding
        vit_in = self.vit.conv_proj(x)  # [B, 768, H/16, W/16]
        vit_in = vit_in.flatten(2).transpose(1, 2)  # [B, N, 768]
        
        cls_token = self.vit.class_token.expand(B, -1, -1)
        vit_in = torch.cat([cls_token, vit_in], dim=1)
        
        # Add position embedding
        pos_embed = self._interpolate_pos_embed(device)
        vit_in = vit_in + pos_embed
        
        # Process through ViT layers with language fusion
        vit_feats = {}
        x_vit = vit_in
        
        for i, block in enumerate(self.vit.encoder.layers):
            x_vit = block(x_vit)
            
            layer_num = i + 1  # 1-indexed
            
            # Apply language fusion at specified layers
            if layer_num in self.fusion_layers:
                x_vit = self.lang_blocks[str(layer_num)](
                    x_vit, word_embeds, attention_mask
                )
            
            # Store features at key layers
            if layer_num == 6:
                vit_feats['mid'] = x_vit[:, 1:, :]  # Remove CLS
            elif layer_num == 12:
                vit_feats['deep'] = x_vit[:, 1:, :]
        
        # Reshape ViT features to spatial
        h, w = H // 16, W // 16
        vit_mid = vit_feats['mid'].transpose(1, 2).view(B, self.hidden_dim, h, w)
        vit_deep = vit_feats['deep'].transpose(1, 2).view(B, self.hidden_dim, h, w)
        
        return {
            'skip1': f1,        # [B, 64, H/2, W/2]
            'skip2': f2,        # [B, 128, H/4, W/4]
            'skip3': f3,        # [B, 256, H/8, W/8]
            'vit_mid': vit_mid, # [B, 768, H/16, W/16]
            'vit_deep': vit_deep,
        }


# ============================================================
# Decoder with PWAM at Each Level
# ============================================================

class LViT4Decoder(nn.Module):
    """
    U-Net style decoder with PWAM at each level.
    
    Features:
    - PWAM for dense pixel-word attention at each scale
    - Instance Normalization throughout
    - Multi-scale contrastive feature extraction
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [64, 128, 256, 768],
        decoder_dims: List[int] = [512, 256, 128, 64],
        text_dim: int = 512,
        word_dim: int = 768
    ):
        super().__init__()
        
        # Initial projection
        self.init_proj = nn.Sequential(
            nn.Conv2d(encoder_dims[-1], decoder_dims[0], 1),
            nn.InstanceNorm2d(decoder_dims[0], affine=True),
            nn.ReLU(inplace=True)
        )
        
        # PWAM at bottleneck
        self.pwam_bottleneck = PixelWordAttentionModule(
            visual_dim=decoder_dims[0],
            text_dim=word_dim,
            hidden_dim=256,
            num_heads=8
        )
        
        # Decoder level 1: /16 -> /8
        self.up1 = nn.ConvTranspose2d(decoder_dims[0], decoder_dims[1], 4, 2, 1)
        self.skip_fuse1 = nn.Sequential(
            nn.Conv2d(encoder_dims[2] + decoder_dims[1], decoder_dims[1], 3, 1, 1),
            nn.InstanceNorm2d(decoder_dims[1], affine=True),
            nn.ReLU(inplace=True)
        )
        self.pwam1 = PixelWordAttentionModule(
            visual_dim=decoder_dims[1],
            text_dim=word_dim,
            hidden_dim=256,
            num_heads=8
        )
        
        # Decoder level 2: /8 -> /4
        self.up2 = nn.ConvTranspose2d(decoder_dims[1], decoder_dims[2], 4, 2, 1)
        self.skip_fuse2 = nn.Sequential(
            nn.Conv2d(encoder_dims[1] + decoder_dims[2], decoder_dims[2], 3, 1, 1),
            nn.InstanceNorm2d(decoder_dims[2], affine=True),
            nn.ReLU(inplace=True)
        )
        self.pwam2 = PixelWordAttentionModule(
            visual_dim=decoder_dims[2],
            text_dim=word_dim,
            hidden_dim=128,
            num_heads=4
        )
        
        # Decoder level 3: /4 -> /2
        self.up3 = nn.ConvTranspose2d(decoder_dims[2], decoder_dims[3], 4, 2, 1)
        self.skip_fuse3 = nn.Sequential(
            nn.Conv2d(encoder_dims[0] + decoder_dims[3], decoder_dims[3], 3, 1, 1),
            nn.InstanceNorm2d(decoder_dims[3], affine=True),
            nn.ReLU(inplace=True)
        )
        self.pwam3 = PixelWordAttentionModule(
            visual_dim=decoder_dims[3],
            text_dim=word_dim,
            hidden_dim=64,
            num_heads=4
        )
        
        # Final upsample: /2 -> /1
        self.up4 = nn.ConvTranspose2d(decoder_dims[3], decoder_dims[3], 4, 2, 1)
        
        # Multi-scale contrastive projections
        self.contrastive_proj_deep = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(decoder_dims[0], 512),
        )
        self.contrastive_proj_mid = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(decoder_dims[1], 512),
        )
        self.contrastive_proj_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(decoder_dims[3], 512),
        )
    
    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        word_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_contrastive: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Decode with PWAM at each level.
        
        Args:
            encoder_features: Dict with skip1, skip2, skip3, vit_deep
            word_embeds: [B, L, 768] word-level embeddings
            attention_mask: [B, L] text attention mask
            return_contrastive: Whether to return contrastive features
            
        Returns:
            decoded: [B, 64, H, W]
            contrastive_feats: Dict of multi-scale contrastive features (optional)
        """
        contrastive_feats = {}
        
        # Initial projection + PWAM at bottleneck
        x = self.init_proj(encoder_features['vit_deep'])  # [B, 512, H/16, W/16]
        x = self.pwam_bottleneck(x, word_embeds, attention_mask)
        
        if return_contrastive:
            contrastive_feats['deep'] = F.normalize(self.contrastive_proj_deep(x), p=2, dim=-1)
        
        # Level 1: /16 -> /8
        x = self.up1(x)
        x = torch.cat([encoder_features['skip3'], x], dim=1)
        x = self.skip_fuse1(x)
        x = self.pwam1(x, word_embeds, attention_mask)
        
        if return_contrastive:
            contrastive_feats['mid'] = F.normalize(self.contrastive_proj_mid(x), p=2, dim=-1)
        
        # Level 2: /8 -> /4
        x = self.up2(x)
        x = torch.cat([encoder_features['skip2'], x], dim=1)
        x = self.skip_fuse2(x)
        x = self.pwam2(x, word_embeds, attention_mask)
        
        # Level 3: /4 -> /2
        x = self.up3(x)
        x = torch.cat([encoder_features['skip1'], x], dim=1)
        x = self.skip_fuse3(x)
        x = self.pwam3(x, word_embeds, attention_mask)
        
        # Final upsample
        x = self.up4(x)
        
        if return_contrastive:
            contrastive_feats['out'] = F.normalize(self.contrastive_proj_out(x), p=2, dim=-1)
        
        return x, contrastive_feats if return_contrastive else None


# ============================================================
# Output Heads
# ============================================================

class LViT4OutputHeads(nn.Module):
    """
    Output heads with Instance Normalization.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 6
    ):
        super().__init__()
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.np_head = nn.Conv2d(32, 2, 1)
        self.hv_head = nn.Conv2d(32, 2, 1)
        self.type_head = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.refine(x)
        return {
            'np': self.np_head(x),
            'hv': self.hv_head(x),
            'type': self.type_head(x)
        }


# ============================================================
# Main LViT4 Model
# ============================================================

class LViT4NucleiSegmenter(nn.Module):
    """
    LViT4: Advanced Language-guided Vision Transformer.
    
    Phase 2 improvements:
        1. Multi-stage Early Fusion - text injection at ViT layers 3, 6, 9, 12
        2. Pixel-Word Attention Module (PWAM) - dense pixel-to-word attention
        3. Multi-scale Contrastive Learning - features from multiple decoder levels
        4. Learnable Temperature - adaptive contrastive scaling
    
    Plus all Phase 1 improvements:
        - Instance Normalization throughout
        - Contrastive loss support
    """
    
    def __init__(
        self,
        text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        num_classes: int = 6,
        freeze_text_encoder: bool = True,
        img_size: int = 256,
        fusion_layers: List[int] = [3, 6, 9, 12],
        learnable_temperature: bool = True,
        init_temperature: float = 0.07
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.fusion_layers = fusion_layers
        
        # Learnable temperature for contrastive loss
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(init_temperature)))
        
        # Text encoder
        self.text_encoder = LViT4TextEncoder(
            model_name=text_encoder,
            embed_dim=embed_dim,
            freeze=freeze_text_encoder
        )
        
        # Multi-stage fusion ViT encoder
        print(f"[LViT4] Multi-stage fusion at layers: {fusion_layers}")
        self.image_encoder = MultiStageFusionViTEncoder(
            pretrained=True,
            img_size=img_size,
            fusion_layers=fusion_layers
        )
        
        # Decoder with PWAM
        self.decoder = LViT4Decoder(
            encoder_dims=[64, 128, 256, 768],
            decoder_dims=[512, 256, 128, 64],
            text_dim=embed_dim,
            word_dim=768
        )
        
        # Output heads
        self.output_heads = LViT4OutputHeads(
            in_channels=64,
            num_classes=num_classes
        )
        
        self._init_weights()
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return torch.exp(self.log_temperature)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        return_contrastive_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W]
            texts: List of text instructions
            return_contrastive_features: Whether to return contrastive features
            
        Returns:
            Dict with:
                - np, hv, type: segmentation outputs
                - contrastive_visual_*: multi-scale visual features (if requested)
                - contrastive_text: text features (if requested)
                - temperature: current temperature value (if requested)
        """
        # Encode text
        text_outputs = self.text_encoder(texts)
        
        # Encode image with multi-stage fusion
        encoder_features = self.image_encoder(
            images,
            text_outputs['word_embeds'],
            text_outputs['attention_mask']
        )
        
        # Decode with PWAM
        decoded, contrastive_feats = self.decoder(
            encoder_features,
            text_outputs['word_embeds'],
            text_outputs['attention_mask'],
            return_contrastive=return_contrastive_features
        )
        
        # Output predictions
        outputs = self.output_heads(decoded)
        
        # Add contrastive features
        if return_contrastive_features and contrastive_feats is not None:
            outputs['contrastive_visual_deep'] = contrastive_feats['deep']
            outputs['contrastive_visual_mid'] = contrastive_feats['mid']
            outputs['contrastive_visual_out'] = contrastive_feats['out']
            outputs['contrastive_text'] = text_outputs['contrastive_embed']
            outputs['temperature'] = self.temperature
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_lvit4_model(
    num_classes: int = 6,
    freeze_text_encoder: bool = True,
    img_size: int = 256,
    fusion_layers: List[int] = [3, 6, 9, 12],
    learnable_temperature: bool = True,
    init_temperature: float = 0.07,
    **kwargs
) -> LViT4NucleiSegmenter:
    """
    Create LViT4 model.
    
    Phase 2 improvements:
        1. Multi-stage Early Fusion at layers 3, 6, 9, 12
        2. PWAM at each decoder level
        3. Multi-scale contrastive features
        4. Learnable temperature
    
    Args:
        num_classes: Number of classes
        freeze_text_encoder: Whether to freeze text encoder
        img_size: Input image size
        fusion_layers: Which ViT layers to inject text
        learnable_temperature: Whether temperature is learnable
        init_temperature: Initial temperature value
        
    Returns:
        LViT4NucleiSegmenter model
    """
    model = LViT4NucleiSegmenter(
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=512,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder,
        img_size=img_size,
        fusion_layers=fusion_layers,
        learnable_temperature=learnable_temperature,
        init_temperature=init_temperature
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LViT4] Total parameters: {total_params:,}")
    print(f"[LViT4] Trainable parameters: {trainable_params:,}")
    print(f"[LViT4] Features: Multi-stage Fusion + PWAM + Learnable Temperature")
    
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LViT4 Nuclei Segmenter Test")
    print("=" * 70)
    
    model = create_lvit4_model(num_classes=6)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    texts = [
        "Segment neoplastic nuclei in breast tissue.",
        "Identify inflammatory cells in colon sample."
    ]
    
    # Test without contrastive
    with torch.no_grad():
        outputs = model(images, texts, return_contrastive_features=False)
    
    print(f"\nOutputs (without contrastive):")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
    
    # Test with contrastive
    with torch.no_grad():
        outputs = model(images, texts, return_contrastive_features=True)
    
    print(f"\nOutputs (with contrastive):")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {val}")
    
    print(f"\nTemperature: {model.temperature.item():.4f}")
    
    # Verify shapes
    assert outputs['np'].shape == (batch_size, 2, 256, 256)
    assert outputs['hv'].shape == (batch_size, 2, 256, 256)
    assert outputs['type'].shape == (batch_size, 6, 256, 256)
    assert outputs['contrastive_visual_deep'].shape == (batch_size, 512)
    assert outputs['contrastive_text'].shape == (batch_size, 512)
    
    print(f"\n✅ LViT4 test passed!")
