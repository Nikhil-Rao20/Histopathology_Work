"""
LViT3: Enhanced Language-guided Vision Transformer for Nuclei Instance Segmentation
=====================================================================================

Based on LViT with Phase 1 improvements from CRIS and LAVT papers:

Key Improvements:
    1. Instance Normalization in cross-attention modules (from LAVT paper - +2% IoU)
    2. Text-to-Pixel Contrastive Loss support (from CRIS paper)
    3. Improved text-visual alignment via better feature normalization

Architecture Changes from LViT:
    - Replace BatchNorm with InstanceNorm in LanguageGuidedFeatureEnhancement
    - Add instance normalization in fusion projections
    - Expose intermediate features for contrastive loss computation

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
# Text Encoder (BioClinicalBERT for medical domain)
# ============================================================

class LViT3TextEncoder(nn.Module):
    """
    Text encoder using BioClinicalBERT.
    Returns sentence embedding, token-level features, and projected features for contrastive loss.
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
        
        # Contrastive projection head (for text-to-pixel contrastive loss)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text.
        
        Args:
            texts: List of text strings
            
        Returns:
            sentence_embed: [B, embed_dim] - sentence embedding
            token_embeds: [B, L, embed_dim] - token embeddings
            contrastive_embed: [B, embed_dim] - normalized embedding for contrastive loss
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
        
        # Contrastive embedding (L2 normalized for cosine similarity)
        contrastive_embed = self.contrastive_proj(sentence_embed)
        contrastive_embed = F.normalize(contrastive_embed, p=2, dim=-1)  # [B, embed_dim]
        
        return sentence_embed, token_embeds, contrastive_embed


# ============================================================
# ViT Encoder with Hierarchical Features
# ============================================================

class HierarchicalViTEncoder(nn.Module):
    """
    ViT encoder that extracts hierarchical (multi-scale) features.
    Simulates U-Net style encoder with skip connections.
    
    Uses Instance Normalization in CNN skip layers for better cross-modal fusion.
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
        # Using Instance Normalization for better text-visual alignment
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),      # /2
            nn.InstanceNorm2d(64, affine=True),  # Instance Norm instead of BatchNorm
            nn.ReLU(inplace=True)
        )
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),    # /4
            nn.InstanceNorm2d(128, affine=True),  # Instance Norm
            nn.ReLU(inplace=True)
        )
        
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),   # /8
            nn.InstanceNorm2d(256, affine=True),  # Instance Norm
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
# Language-Guided Feature Enhancement (LFE) Module with Instance Norm
# ============================================================

class LanguageGuidedFeatureEnhancementV3(nn.Module):
    """
    Enhanced Language-Guided Feature Enhancement module.
    
    Key improvements from LAVT paper:
    - Instance Normalization in projection layers for better cross-modal alignment
    - This provides ~2% IoU improvement over BatchNorm
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        contrastive_dim: int = 512  # Dimension for contrastive features
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.contrastive_dim = contrastive_dim
        
        # Project to common dimension - using Instance Norm for better alignment
        self.visual_proj = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, 1),
            nn.InstanceNorm2d(hidden_dim, affine=True),  # Instance Norm instead of BatchNorm
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            # Note: LayerNorm is similar to InstanceNorm for 1D - keeps it
        )
        
        # Cross-attention: visual queries, text keys/values
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Gate for controlled fusion - using Instance Norm
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection with Instance Norm
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, visual_dim, 1),
            nn.InstanceNorm2d(visual_dim, affine=True),  # Instance Norm
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Contrastive projection - project visual features for contrastive loss
        self.contrastive_visual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(visual_dim, contrastive_dim),
            nn.ReLU(inplace=True),
            nn.Linear(contrastive_dim, contrastive_dim)
        )
    
    def forward(
        self,
        visual: torch.Tensor,
        text_tokens: torch.Tensor,
        text_sentence: torch.Tensor,
        return_contrastive_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhance visual features with language guidance.
        
        Args:
            visual: [B, C_v, H, W] visual features
            text_tokens: [B, L, C_t] text token embeddings
            text_sentence: [B, C_t] sentence embedding
            return_contrastive_features: Whether to return features for contrastive loss
            
        Returns:
            enhanced: [B, C_v, H, W] enhanced visual features
            contrastive_visual: [B, contrastive_dim] (optional) normalized features for contrastive loss
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
        
        # Compute contrastive features if requested
        contrastive_visual = None
        if return_contrastive_features:
            contrastive_visual = self.contrastive_visual_proj(output)
            contrastive_visual = F.normalize(contrastive_visual, p=2, dim=-1)  # L2 normalize
        
        return output, contrastive_visual


# ============================================================
# Language-Modulated Skip Connection with Instance Norm
# ============================================================

class LanguageModulatedSkipV3(nn.Module):
    """
    Skip connection modulated by language embedding.
    Uses Instance Normalization for better cross-modal fusion.
    """
    
    def __init__(
        self,
        skip_dim: int,
        decoder_dim: int,
        text_dim: int
    ):
        super().__init__()
        
        # Spatial attention from text (kept as is - not conv)
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
        
        # Fusion conv with Instance Norm
        self.fusion = nn.Sequential(
            nn.Conv2d(skip_dim + decoder_dim, decoder_dim, 3, 1, 1),
            nn.InstanceNorm2d(decoder_dim, affine=True),  # Instance Norm
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
# U-Net Style Decoder with Language Guidance (Instance Norm)
# ============================================================

class LViT3Decoder(nn.Module):
    """
    U-Net style decoder with language-modulated skip connections.
    Uses Instance Normalization throughout.
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [64, 128, 256, 768],
        decoder_dims: List[int] = [512, 256, 128, 64],
        text_dim: int = 512
    ):
        super().__init__()
        
        # Initial projection from ViT features - Instance Norm
        self.init_proj = nn.Sequential(
            nn.Conv2d(encoder_dims[-1], decoder_dims[0], 1),
            nn.InstanceNorm2d(decoder_dims[0], affine=True),  # Instance Norm
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with language-modulated skips
        self.up1 = nn.ConvTranspose2d(decoder_dims[0], decoder_dims[1], 4, 2, 1)  # /16 -> /8
        self.skip1 = LanguageModulatedSkipV3(encoder_dims[2], decoder_dims[1], text_dim)
        
        self.up2 = nn.ConvTranspose2d(decoder_dims[1], decoder_dims[2], 4, 2, 1)  # /8 -> /4
        self.skip2 = LanguageModulatedSkipV3(encoder_dims[1], decoder_dims[2], text_dim)
        
        self.up3 = nn.ConvTranspose2d(decoder_dims[2], decoder_dims[3], 4, 2, 1)  # /4 -> /2
        self.skip3 = LanguageModulatedSkipV3(encoder_dims[0], decoder_dims[3], text_dim)
        
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
# HoVer-Net Style Output Heads (with Instance Norm)
# ============================================================

class LViT3OutputHeads(nn.Module):
    """
    Output heads for nuclei instance segmentation.
    Uses Instance Normalization.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 6
    ):
        super().__init__()
        
        # Shared refinement with Instance Norm
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.InstanceNorm2d(64, affine=True),  # Instance Norm
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32, affine=True),  # Instance Norm
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
# Main LViT3 Model for Nuclei Segmentation
# ============================================================

class LViT3NucleiSegmenter(nn.Module):
    """
    LViT3: Enhanced Language-guided Vision Transformer for Nuclei Instance Segmentation.
    
    Key improvements over LViT:
        1. Instance Normalization in all cross-modal fusion modules (~2% IoU improvement)
        2. Contrastive feature projection for text-to-pixel contrastive loss
        3. Better text-visual alignment through improved normalization
    
    Features:
        - Hierarchical ViT encoder with Instance Norm skip connections
        - Language-guided feature enhancement with Instance Norm
        - Language-modulated skip connections with Instance Norm
        - HoVer-Net style instance segmentation heads
        - Support for contrastive loss training
    """
    
    def __init__(
        self,
        text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        num_classes: int = 6,
        freeze_text_encoder: bool = True,
        img_size: int = 256,
        enable_contrastive: bool = True  # Enable contrastive loss features
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.enable_contrastive = enable_contrastive
        
        # Text encoder with contrastive projection
        self.text_encoder = LViT3TextEncoder(
            model_name=text_encoder,
            embed_dim=embed_dim,
            freeze=freeze_text_encoder
        )
        
        # Hierarchical image encoder with Instance Norm
        print(f"[LViT3] Using ViT-B/16 backbone with Instance Normalization")
        self.image_encoder = HierarchicalViTEncoder(
            pretrained=True,
            img_size=img_size
        )
        
        # Language-guided feature enhancement with Instance Norm and contrastive support
        self.lfe = LanguageGuidedFeatureEnhancementV3(
            visual_dim=768,  # ViT dimension
            text_dim=embed_dim,
            hidden_dim=256,
            num_heads=8,
            contrastive_dim=embed_dim
        )
        
        # Decoder with Instance Norm and language-modulated skips
        self.decoder = LViT3Decoder(
            encoder_dims=[64, 128, 256, 768],
            decoder_dims=[512, 256, 128, 64],
            text_dim=embed_dim
        )
        
        # Output heads with Instance Norm
        self.output_heads = LViT3OutputHeads(
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
            images: [B, 3, H, W] input images
            texts: List of text instructions
            return_contrastive_features: Whether to return features for contrastive loss
            
        Returns:
            Dictionary with:
                - np: [B, 2, H, W] nuclei presence predictions
                - hv: [B, 2, H, W] horizontal-vertical predictions
                - type: [B, num_classes, H, W] type predictions
                - contrastive_visual: [B, embed_dim] (optional) visual features for contrastive loss
                - contrastive_text: [B, embed_dim] (optional) text features for contrastive loss
        """
        # Encode text
        text_sentence, text_tokens, contrastive_text = self.text_encoder(texts)
        
        # Encode image (hierarchical features)
        encoder_features = self.image_encoder(images)
        
        # Language-guided feature enhancement at bottleneck
        enhanced_vit, contrastive_visual = self.lfe(
            encoder_features['vit_deep'],
            text_tokens,
            text_sentence,
            return_contrastive_features=return_contrastive_features and self.enable_contrastive
        )
        encoder_features['vit_deep'] = enhanced_vit
        
        # Decode with language guidance
        decoded = self.decoder(encoder_features, text_sentence)
        
        # Output predictions
        outputs = self.output_heads(decoded)
        
        # Add contrastive features if requested
        if return_contrastive_features and self.enable_contrastive:
            outputs['contrastive_visual'] = contrastive_visual
            outputs['contrastive_text'] = contrastive_text
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_lvit3_model(
    num_classes: int = 6,
    freeze_text_encoder: bool = True,
    img_size: int = 256,
    enable_contrastive: bool = True,
    **kwargs
) -> LViT3NucleiSegmenter:
    """
    Create LViT3 model for nuclei segmentation.
    
    LViT3 improvements:
        1. Instance Normalization for better cross-modal fusion (~2% IoU improvement)
        2. Contrastive feature support for text-to-pixel alignment
    
    Args:
        num_classes: Number of nucleus classes (default: 6)
        freeze_text_encoder: Whether to freeze text encoder
        img_size: Input image size
        enable_contrastive: Whether to enable contrastive loss features
        
    Returns:
        LViT3NucleiSegmenter model
    """
    model = LViT3NucleiSegmenter(
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=512,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder,
        img_size=img_size,
        enable_contrastive=enable_contrastive
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LViT3] Total parameters: {total_params:,}")
    print(f"[LViT3] Trainable parameters: {trainable_params:,}")
    print(f"[LViT3] Instance Normalization: Enabled (replacing BatchNorm)")
    print(f"[LViT3] Contrastive Loss Support: {'Enabled' if enable_contrastive else 'Disabled'}")
    
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LViT3 Nuclei Segmenter Test")
    print("=" * 70)
    
    # Create model
    model = create_lvit3_model(num_classes=6, enable_contrastive=True)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nLViT3 Nuclei Segmenter Configuration")
    print(f"  - Text Encoder: BioClinicalBERT")
    print(f"  - Image Encoder: Hierarchical ViT-B/16 with Instance Norm")
    print(f"  - Language-Guided Enhancement: Instance Norm enabled")
    print(f"  - Language-Modulated Skips: Instance Norm enabled")
    print(f"  - Contrastive Loss Support: Enabled")
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
    
    # Test without contrastive features
    with torch.no_grad():
        outputs = model(images, texts, return_contrastive_features=False)
    
    print(f"\nOutputs (without contrastive):")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Test with contrastive features
    with torch.no_grad():
        outputs = model(images, texts, return_contrastive_features=True)
    
    print(f"\nOutputs (with contrastive):")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Verify output shapes
    assert outputs['np'].shape == (batch_size, 2, 256, 256)
    assert outputs['hv'].shape == (batch_size, 2, 256, 256)
    assert outputs['type'].shape == (batch_size, 6, 256, 256)
    assert outputs['contrastive_visual'].shape == (batch_size, 512)
    assert outputs['contrastive_text'].shape == (batch_size, 512)
    
    # Verify contrastive features are normalized
    visual_norm = outputs['contrastive_visual'].norm(dim=-1)
    text_norm = outputs['contrastive_text'].norm(dim=-1)
    print(f"\nContrastive feature norms (should be ~1.0):")
    print(f"  Visual: {visual_norm.tolist()}")
    print(f"  Text: {text_norm.tolist()}")
    
    print(f"\n✅ LViT3 Nuclei Segmenter test passed!")
