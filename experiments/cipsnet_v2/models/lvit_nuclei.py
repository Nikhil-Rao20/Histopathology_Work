"""
LViT: Language-guided Vision Transformer for Nuclei Instance Segmentation
==========================================================================

Based on: "LViT: Language meets Vision Transformer in Medical Image Segmentation"
Paper: https://arxiv.org/abs/2206.14718 (IEEE TMI 2023)

Key Innovation:
    - Language-guided multi-scale feature fusion
    - Token-level language guidance for fine-grained control
    - Skip-connection with language modulation
    - U-Net style architecture with ViT encoder

Architecture:
    1. Swin Transformer / ViT encoder for hierarchical features
    2. Text encoder (BioClinicalBERT for medical domain)
    3. Language-Guided Feature Enhancement (LFE) module
    4. Language-Modulated Skip Connections
    5. U-Net style decoder with HoVer-Net output heads

Adapted for nuclei instance segmentation on PanNuke dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============================================================
# Text Encoder (BioClinicalBERT for medical domain)
# ============================================================

class LViTTextEncoder(nn.Module):
    """
    Text encoder using BioClinicalBERT.
    Returns sentence embedding and token-level features.
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
        
        # Project to common embedding dimension
        self.text_proj = nn.Linear(self.hidden_size, embed_dim)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text.
        
        Args:
            texts: List of text strings
            
        Returns:
            sentence_embed: [B, embed_dim]
            token_embeds: [B, L, embed_dim]
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
        
        # Sentence embedding (CLS token)
        sentence_embed = token_embeds[:, 0, :]  # [B, embed_dim]
        
        return sentence_embed, token_embeds


# ============================================================
# ConvNeXt-Base Encoder with Hierarchical Features
# ============================================================

class HierarchicalConvNeXtEncoder(nn.Module):
    """
    ConvNeXt-Base encoder that extracts hierarchical (multi-scale) features.
    Native hierarchical architecture - perfect for U-Net style skip connections.
    
    ConvNeXt-Base:
        - Stage 1: 128 channels, /4
        - Stage 2: 256 channels, /8
        - Stage 3: 512 channels, /16
        - Stage 4: 1024 channels, /32
    
    We use stages 1-3 for skip connections and stage 4 as bottleneck.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 256
    ):
        super().__init__()
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        self.convnext = convnext_base(weights=weights)
        self.img_size = img_size
        
        # ConvNeXt-Base channel dimensions at each stage
        # Stage 0 (stem): /4, 128 channels
        # Stage 1: /4, 128 channels  
        # Stage 2: /8, 256 channels
        # Stage 3: /16, 512 channels
        # Stage 4: /32, 1024 channels
        
        # We need to provide features compatible with LViT decoder
        # LViT expects: skip1 (64, /2), skip2 (128, /4), skip3 (256, /8), vit_deep (768, /16)
        
        # Additional layers to match expected dimensions
        self.stem_upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # /4 -> /2
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Project stage 3 (512ch) to 256ch for skip3
        self.stage2_proj = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        self.stage3_proj = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        
        # Project stage 4 (1024ch) to 768ch to match ViT dimension
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(1024, 768, 1),
            nn.BatchNorm2d(768),
            nn.GELU()
        )
        
        # Upsample bottleneck from /32 to /16 to match ViT spatial size
        self.bottleneck_upsample = nn.Sequential(
            nn.ConvTranspose2d(768, 768, 4, 2, 1),  # /32 -> /16
            nn.BatchNorm2d(768),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features from ConvNeXt.
        
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            Dictionary with features at different scales matching LViT format
        """
        B, C, H, W = x.shape
        
        # ConvNeXt stem: /4
        x = self.convnext.features[0](x)  # Stem: [B, 128, H/4, W/4]
        stem_features = x
        
        # Stage 1: still /4
        x = self.convnext.features[1](x)  # [B, 128, H/4, W/4]
        stage1_features = x
        
        # Stage 2: /8
        x = self.convnext.features[2](x)  # Downsample
        x = self.convnext.features[3](x)  # [B, 256, H/8, W/8]
        stage2_features = x
        
        # Stage 3: /16
        x = self.convnext.features[4](x)  # Downsample
        x = self.convnext.features[5](x)  # [B, 512, H/16, W/16]
        stage3_features = x
        
        # Stage 4: /32 (bottleneck)
        x = self.convnext.features[6](x)  # Downsample
        x = self.convnext.features[7](x)  # [B, 1024, H/32, W/32]
        stage4_features = x
        
        # Map to LViT expected format
        # skip1: [B, 64, H/2, W/2]
        skip1 = self.stem_upsample(stem_features)  # /4 -> /2, 128 -> 64
        
        # skip2: [B, 128, H/4, W/4]
        skip2 = self.stage2_proj(stage2_features)  # Keep /8 but need /4
        skip2 = F.interpolate(skip2, scale_factor=2, mode='bilinear', align_corners=False)  # /8 -> /4
        
        # skip3: [B, 256, H/8, W/8]
        skip3 = self.stage3_proj(stage3_features)  # /16 -> project to 256ch
        skip3 = F.interpolate(skip3, scale_factor=2, mode='bilinear', align_corners=False)  # /16 -> /8
        
        # vit_deep: [B, 768, H/16, W/16]
        bottleneck = self.bottleneck_proj(stage4_features)  # 1024 -> 768
        vit_deep = self.bottleneck_upsample(bottleneck)  # /32 -> /16
        
        return {
            'skip1': skip1,        # [B, 64, H/2, W/2]
            'skip2': skip2,        # [B, 128, H/4, W/4]
            'skip3': skip3,        # [B, 256, H/8, W/8]
            'vit_mid': vit_deep,   # [B, 768, H/16, W/16] (same as deep for ConvNeXt)
            'vit_deep': vit_deep,  # [B, 768, H/16, W/16]
        }


# ============================================================
# ViT Encoder with Hierarchical Features
# ============================================================

class HierarchicalViTEncoder(nn.Module):
    """
    ViT encoder that extracts hierarchical (multi-scale) features.
    Simulates U-Net style encoder with skip connections.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 256
    ):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.hidden_dim = 768
        self.patch_size = 16
        self.img_size = img_size
        
        # Store target grid size for lazy pos embed interpolation
        self.target_h = self.target_w = img_size // self.patch_size
        
        # Remove classification head
        self.vit.heads = nn.Identity()
        
        # Additional downsampling for hierarchical features
        # These create "encoder levels" like U-Net
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),      # /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),    # /4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),   # /8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def _interpolate_pos_embed(self, device):
        """Interpolate positional embeddings for new image size (lazy, on same device)."""
        old_pos_embed = self.vit.encoder.pos_embedding.to(device)  # [1, 197, 768]
        
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
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features.
        
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            Dictionary with features at different scales
        """
        B, C, H, W = x.shape
        device = x.device
        
        # CNN-based hierarchical features (for skip connections)
        f1 = self.downsample1(x)   # [B, 64, H/2, W/2]
        f2 = self.downsample2(f1)  # [B, 128, H/4, W/4]
        f3 = self.downsample3(f2)  # [B, 256, H/8, W/8]
        
        # ViT features
        vit_in = self.vit.conv_proj(x)  # [B, 768, H/16, W/16]
        vit_in = vit_in.flatten(2).transpose(1, 2)  # [B, N, 768]
        
        cls_token = self.vit.class_token.expand(B, -1, -1)
        vit_in = torch.cat([cls_token, vit_in], dim=1)
        
        # Get interpolated pos embed on correct device
        pos_embed = self._interpolate_pos_embed(device)
        vit_in = vit_in + pos_embed
        
        # ViT transformer blocks - extract at different layers
        vit_feats = {}
        x_vit = vit_in
        for i, block in enumerate(self.vit.encoder.layers):
            x_vit = block(x_vit)
            if i == 5:  # Mid-depth feature
                vit_feats['mid'] = x_vit[:, 1:, :]
            elif i == 11:  # Deep feature
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
            'vit_deep': vit_deep,  # [B, 768, H/16, W/16]
        }


# ============================================================
# Language-Guided Feature Enhancement (LFE) Module
# ============================================================

class LanguageGuidedFeatureEnhancement(nn.Module):
    """
    Enhances visual features using language guidance.
    Key component of LViT for text-aware feature modulation.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Project to common dimension
        self.visual_proj = nn.Conv2d(visual_dim, hidden_dim, 1)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention: visual queries, text keys/values
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Gate for controlled fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Conv2d(hidden_dim, visual_dim, 1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        visual: torch.Tensor,
        text_tokens: torch.Tensor,
        text_sentence: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhance visual features with language guidance.
        
        Args:
            visual: [B, C_v, H, W] visual features
            text_tokens: [B, L, C_t] text token embeddings
            text_sentence: [B, C_t] sentence embedding
            
        Returns:
            [B, C_v, H, W] enhanced visual features
        """
        B, C_v, H, W = visual.shape
        
        # Project visual features
        visual_proj = self.visual_proj(visual)  # [B, hidden_dim, H, W]
        visual_flat = visual_proj.flatten(2).transpose(1, 2)  # [B, N, hidden_dim]
        visual_flat = self.norm1(visual_flat)
        
        # Project text features
        text_proj = self.text_proj(text_tokens)  # [B, L, hidden_dim]
        text_proj = self.norm2(text_proj)
        
        # Cross-attention
        attended, _ = self.cross_attn(visual_flat, text_proj, text_proj)
        
        # Gating mechanism with sentence-level context
        sentence_proj = self.text_proj(text_sentence)  # [B, hidden_dim]
        sentence_expanded = sentence_proj.unsqueeze(1).expand(-1, H*W, -1)
        
        gate_input = torch.cat([attended, sentence_expanded], dim=-1)
        gate = self.gate(gate_input)  # [B, N, hidden_dim]
        
        # Apply gate
        enhanced = visual_flat + gate * attended
        
        # Reshape and project back
        enhanced = enhanced.transpose(1, 2).view(B, -1, H, W)
        enhanced = self.out_proj(enhanced)
        
        # Residual connection
        output = visual + enhanced
        
        return output


# ============================================================
# Language-Modulated Skip Connection
# ============================================================

class LanguageModulatedSkip(nn.Module):
    """
    Skip connection modulated by language embedding.
    Allows text to control which skip features are emphasized.
    """
    
    def __init__(
        self,
        skip_dim: int,
        decoder_dim: int,
        text_dim: int
    ):
        super().__init__()
        
        # Spatial attention from text
        self.text_to_spatial = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        # Channel attention from text
        self.text_to_channel = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, skip_dim),
            nn.Sigmoid()
        )
        
        # Fusion conv
        self.fusion = nn.Sequential(
            nn.Conv2d(skip_dim + decoder_dim, decoder_dim, 3, 1, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        skip: torch.Tensor,
        decoder: torch.Tensor,
        text_sentence: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate skip connection with text.
        
        Args:
            skip: [B, C_s, H, W] skip features
            decoder: [B, C_d, H', W'] decoder features (upsampled)
            text_sentence: [B, C_t] sentence embedding
            
        Returns:
            [B, C_d, H, W] fused features
        """
        B, C_s, H, W = skip.shape
        
        # Channel attention
        channel_attn = self.text_to_channel(text_sentence)  # [B, C_s]
        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1)  # [B, C_s, 1, 1]
        
        # Modulate skip features
        skip_modulated = skip * channel_attn
        
        # Upsample decoder if needed
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        fused = torch.cat([skip_modulated, decoder], dim=1)
        output = self.fusion(fused)
        
        return output


# ============================================================
# U-Net Style Decoder with Language Guidance
# ============================================================

class LViTDecoder(nn.Module):
    """
    U-Net style decoder with language-modulated skip connections.
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [64, 128, 256, 768],
        decoder_dims: List[int] = [512, 256, 128, 64],
        text_dim: int = 512
    ):
        super().__init__()
        
        # Initial projection from ViT features
        self.init_proj = nn.Sequential(
            nn.Conv2d(encoder_dims[-1], decoder_dims[0], 1),
            nn.BatchNorm2d(decoder_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with language-modulated skips
        self.up1 = nn.ConvTranspose2d(decoder_dims[0], decoder_dims[1], 4, 2, 1)  # /16 -> /8
        self.skip1 = LanguageModulatedSkip(encoder_dims[2], decoder_dims[1], text_dim)
        
        self.up2 = nn.ConvTranspose2d(decoder_dims[1], decoder_dims[2], 4, 2, 1)  # /8 -> /4
        self.skip2 = LanguageModulatedSkip(encoder_dims[1], decoder_dims[2], text_dim)
        
        self.up3 = nn.ConvTranspose2d(decoder_dims[2], decoder_dims[3], 4, 2, 1)  # /4 -> /2
        self.skip3 = LanguageModulatedSkip(encoder_dims[0], decoder_dims[3], text_dim)
        
        self.up4 = nn.ConvTranspose2d(decoder_dims[3], decoder_dims[3], 4, 2, 1)  # /2 -> /1
    
    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        text_sentence: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode features with language guidance.
        
        Args:
            encoder_features: Dictionary with skip1, skip2, skip3, vit_deep
            text_sentence: [B, text_dim] sentence embedding
            
        Returns:
            [B, 64, H, W] decoded features
        """
        # Initial projection
        x = self.init_proj(encoder_features['vit_deep'])  # [B, 512, H/16, W/16]
        
        # Decoder with language-modulated skips
        x = self.up1(x)  # [B, 256, H/8, W/8]
        x = self.skip1(encoder_features['skip3'], x, text_sentence)
        
        x = self.up2(x)  # [B, 128, H/4, W/4]
        x = self.skip2(encoder_features['skip2'], x, text_sentence)
        
        x = self.up3(x)  # [B, 64, H/2, W/2]
        x = self.skip3(encoder_features['skip1'], x, text_sentence)
        
        x = self.up4(x)  # [B, 64, H, W]
        
        return x


# ============================================================
# HoVer-Net Style Output Heads
# ============================================================

class LViTOutputHeads(nn.Module):
    """
    Output heads for nuclei instance segmentation.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 6
    ):
        super().__init__()
        
        # Shared refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.np_head = nn.Conv2d(32, 2, 1)  # Binary
        self.hv_head = nn.Conv2d(32, 2, 1)  # Horizontal-Vertical
        self.type_head = nn.Conv2d(32, num_classes, 1)  # Types
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] decoder output
            
        Returns:
            Dictionary with np, hv, type predictions
        """
        x = self.refine(x)
        
        return {
            'np': self.np_head(x),
            'hv': self.hv_head(x),
            'type': self.type_head(x)
        }


# ============================================================
# Main LViT Model for Nuclei Segmentation
# ============================================================

class LViTNucleiSegmenter(nn.Module):
    """
    LViT: Language-guided Vision Transformer for Nuclei Instance Segmentation.
    
    Key features:
        - Hierarchical encoder (ViT or ConvNeXt-Base or DINOv2) with skip connections
        - Language-guided feature enhancement at bottleneck
        - Language-modulated skip connections
        - HoVer-Net style instance segmentation heads
    
    Backbone options:
        - 'vit': ViT-B/16 with CNN skip connections (default)
        - 'convnext_base': ConvNeXt-Base with native hierarchical features
        - 'dinov2_vit_b_14': DINOv2 ViT-B/14 (self-supervised, patch14)
        - 'dinov2_vit_l_14': DINOv2 ViT-L/14 (self-supervised, patch14)
        - 'dinov2_vit_s_14': DINOv2 ViT-S/14 (self-supervised, patch14)
        - 'dinov2_vit_g_14': DINOv2 ViT-g/14 (self-supervised, giant)
    """
    
    def __init__(
        self,
        text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        num_classes: int = 6,
        freeze_text_encoder: bool = True,
        img_size: int = 256,
        backbone: str = "vit",  # 'vit', 'convnext_base', 'dinov2_vit_b_14', etc.
        freeze_dinov2_backbone: bool = False,
        dinov2_pretrained_path: str = "",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.backbone_type = backbone
        
        # Text encoder
        self.text_encoder = LViTTextEncoder(
            model_name=text_encoder,
            embed_dim=embed_dim,
            freeze=freeze_text_encoder
        )
        
        # Hierarchical image encoder - choose based on backbone
        if backbone == 'convnext_base':
            print(f"[LViT] Using ConvNeXt-Base backbone (ImageNet pretrained)")
            self.image_encoder = HierarchicalConvNeXtEncoder(
                pretrained=True,
                img_size=img_size
            )
        elif backbone.startswith('dinov2_'):
            # DINOv2 backbones
            from .dinov2_encoder import HierarchicalDINOv2Encoder, create_dinov2_encoder
            print(f"[LViT] Using DINOv2 backbone: {backbone}" +
                  (" [gradient checkpointing ON]" if use_gradient_checkpointing else ""))
            
            # Use factory function to optionally load supervised pretrained weights
            if dinov2_pretrained_path:
                self.image_encoder = create_dinov2_encoder(
                    model_name=backbone,
                    pretrained=True,
                    img_size=img_size,
                    freeze_backbone=freeze_dinov2_backbone,
                    pretrained_path=dinov2_pretrained_path,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )
            else:
                self.image_encoder = HierarchicalDINOv2Encoder(
                    model_name=backbone,
                    pretrained=True,
                    img_size=img_size,
                    freeze_backbone=freeze_dinov2_backbone,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )
        elif backbone.startswith('swin_'):
            from .swin_encoder import HierarchicalSwinEncoder
            print(f"[LViT] Using Swin backbone: {backbone}" +
                  (" [gradient checkpointing ON]" if use_gradient_checkpointing else ""))
            self.image_encoder = HierarchicalSwinEncoder(
                model_name=backbone,
                pretrained=True,
                img_size=img_size,
                freeze_backbone=freeze_dinov2_backbone,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:  # default: vit
            print(f"[LViT] Using ViT-B/16 backbone (ImageNet pretrained)")
            self.image_encoder = HierarchicalViTEncoder(
                pretrained=True,
                img_size=img_size
            )
        
        # Language-guided feature enhancement at bottleneck
        self.lfe = LanguageGuidedFeatureEnhancement(
            visual_dim=768,  # ViT dimension
            text_dim=embed_dim,
            hidden_dim=256,
            num_heads=8
        )
        
        # Decoder with language-modulated skips
        self.decoder = LViTDecoder(
            encoder_dims=[64, 128, 256, 768],
            decoder_dims=[512, 256, 128, 64],
            text_dim=embed_dim
        )
        
        # Output heads
        self.output_heads = LViTOutputHeads(
            in_channels=64,
            num_classes=num_classes
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W] input images
            texts: List of text instructions
            
        Returns:
            Dictionary with np, hv, type predictions
        """
        # Encode text
        text_sentence, text_tokens = self.text_encoder(texts)
        
        # Encode image (hierarchical features)
        encoder_features = self.image_encoder(images)
        
        # Language-guided feature enhancement at bottleneck
        enhanced_vit = self.lfe(
            encoder_features['vit_deep'],
            text_tokens,
            text_sentence
        )
        encoder_features['vit_deep'] = enhanced_vit
        
        # Decode with language guidance
        decoded = self.decoder(encoder_features, text_sentence)
        
        # Output predictions
        outputs = self.output_heads(decoded)
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_lvit_model(
    num_classes: int = 6,
    freeze_text_encoder: bool = True,
    img_size: int = 256,
    backbone: str = "vit",
    freeze_dinov2_backbone: bool = False,
    dinov2_pretrained_path: str = "",
    use_gradient_checkpointing: bool = False,
    **kwargs
) -> LViTNucleiSegmenter:
    """
    Create LViT model for nuclei segmentation.
    
    Args:
        num_classes: Number of nucleus classes (default: 6)
        freeze_text_encoder: Whether to freeze text encoder
        img_size: Input image size
        backbone: Backbone type - 'vit', 'convnext_base', 'dinov2_vit_b_14', 
                  'dinov2_vit_l_14', 'dinov2_vit_s_14', 'dinov2_vit_g_14'
        freeze_dinov2_backbone: Whether to freeze DINOv2 backbone (only for DINOv2)
        dinov2_pretrained_path: Path to supervised pretrained DINOv2 checkpoint
        use_gradient_checkpointing: Enable gradient checkpointing on DINOv2 backbone
        
    Returns:
        LViTNucleiSegmenter model
    """
    model = LViTNucleiSegmenter(
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=512,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder,
        img_size=img_size,
        backbone=backbone,
        freeze_dinov2_backbone=freeze_dinov2_backbone,
        dinov2_pretrained_path=dinov2_pretrained_path,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LViT] Total parameters: {total_params:,}")
    print(f"[LViT] Trainable parameters: {trainable_params:,}")
    
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LViT Nuclei Segmenter Test")
    print("=" * 70)
    
    # Create model
    model = create_lvit_model(num_classes=6)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nLViT Nuclei Segmenter Configuration")
    print(f"  - Text Encoder: BioClinicalBERT")
    print(f"  - Image Encoder: Hierarchical ViT-B/16")
    print(f"  - Language-Guided Enhancement: Enabled")
    print(f"  - Language-Modulated Skips: Enabled")
    print(f"  - Output: HoVer-Net style (NP + HV + Type)")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    texts = [
        "Segment neoplastic nuclei in breast tissue.",
        "Identify inflammatory cells in colon sample."
    ]
    
    with torch.no_grad():
        outputs = model(images, texts)
    
    print(f"\nOutputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Verify output shapes
    assert outputs['np'].shape == (batch_size, 2, 256, 256)
    assert outputs['hv'].shape == (batch_size, 2, 256, 256)
    assert outputs['type'].shape == (batch_size, 6, 256, 256)
    
    print(f"\n✅ LViT Nuclei Segmenter test passed!")
