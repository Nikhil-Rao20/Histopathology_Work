"""
LViT2: Enhanced Language-guided Vision Transformer for Nuclei Instance Segmentation
====================================================================================

IMPROVEMENTS over LViT:
    1. Deep Supervision - Auxiliary losses at decoder levels
    2. Auxiliary Classification Branch - Global class prediction
    3. Dice + CE Combined Loss - Better per-class optimization (integrated in loss)
    4. Label Smoothing - Reduces overconfidence (integrated in loss)
    5. Copy-Paste Augmentation support - For rare class handling

Architecture Enhancements:
    - AuxiliaryClassificationBranch: Predicts which classes are present globally
    - DeepSupervision: Auxiliary heads at decoder levels 1, 2, 3
    - Enhanced output that includes aux_type predictions for deep supervision
    - Global class prediction for multi-label image classification

Based on: "LViT: Language meets Vision Transformer in Medical Image Segmentation"
Enhanced with techniques from:
    - Deep Supervision (U-Net++, CE-Net)
    - Auxiliary Branch (HoVer-Net, PanNuke baseline)
    - Dice Loss (nnU-Net)

Author: Enhanced for CIPS-Net V2 ablation study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============================================================
# Text Encoder (BioClinicalBERT for medical domain)
# ============================================================

class LViT2TextEncoder(nn.Module):
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
        
        # Additional layers to match expected dimensions
        self.stem_upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # /4 -> /2
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
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
        
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(1024, 768, 1),
            nn.BatchNorm2d(768),
            nn.GELU()
        )
        
        self.bottleneck_upsample = nn.Sequential(
            nn.ConvTranspose2d(768, 768, 4, 2, 1),  # /32 -> /16
            nn.BatchNorm2d(768),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract hierarchical features from ConvNeXt."""
        B, C, H, W = x.shape
        
        x = self.convnext.features[0](x)  # Stem
        stem_features = x
        
        x = self.convnext.features[1](x)  # Stage 1
        stage1_features = x
        
        x = self.convnext.features[2](x)  # Downsample
        x = self.convnext.features[3](x)  # Stage 2
        stage2_features = x
        
        x = self.convnext.features[4](x)  # Downsample
        x = self.convnext.features[5](x)  # Stage 3
        stage3_features = x
        
        x = self.convnext.features[6](x)  # Downsample
        x = self.convnext.features[7](x)  # Stage 4
        stage4_features = x
        
        # Map to LViT expected format
        skip1 = self.stem_upsample(stem_features)
        skip2 = self.stage2_proj(stage2_features)
        skip2 = F.interpolate(skip2, scale_factor=2, mode='bilinear', align_corners=False)
        skip3 = self.stage3_proj(stage3_features)
        skip3 = F.interpolate(skip3, scale_factor=2, mode='bilinear', align_corners=False)
        bottleneck = self.bottleneck_proj(stage4_features)
        vit_deep = self.bottleneck_upsample(bottleneck)
        
        return {
            'skip1': skip1,
            'skip2': skip2,
            'skip3': skip3,
            'vit_mid': vit_deep,
            'vit_deep': vit_deep,
            'bottleneck': bottleneck,  # For auxiliary branch
        }


# ============================================================
# ViT Encoder with Hierarchical Features
# ============================================================

class HierarchicalViTEncoder(nn.Module):
    """ViT encoder that extracts hierarchical (multi-scale) features."""
    
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
        
        # Modify ViT to accept 256x256 images
        # Update the pos_embedding for the new image size
        num_patches_new = (img_size // self.patch_size) ** 2  # 16x16 = 256 patches
        num_patches_orig = (224 // self.patch_size) ** 2  # 14x14 = 196 patches
        
        if num_patches_new != num_patches_orig:
            # Interpolate position embeddings
            pos_embed = self.vit.encoder.pos_embedding  # [1, 197, 768] (196 patches + 1 cls)
            cls_token = pos_embed[:, :1, :]  # [1, 1, 768]
            patch_pos_embed = pos_embed[:, 1:, :]  # [1, 196, 768]
            
            # Reshape for interpolation
            h_orig = w_orig = int(num_patches_orig ** 0.5)  # 14
            h_new = w_new = int(num_patches_new ** 0.5)  # 16
            
            patch_pos_embed = patch_pos_embed.reshape(1, h_orig, w_orig, self.hidden_dim)
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, 768, 14, 14]
            patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed, size=(h_new, w_new), mode='bicubic', align_corners=False
            )  # [1, 768, 16, 16]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches_new, self.hidden_dim)
            
            # Combine cls + patch embeddings
            new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=1)
            self.vit.encoder.pos_embedding = nn.Parameter(new_pos_embed)
            
            # Update image_size attribute
            self.vit.image_size = img_size
        
        # CNN branch for skip connections
        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.skip_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.skip_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Project ViT features for mid-level fusion
        self.vit_proj_mid = nn.Sequential(
            nn.Conv2d(768, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        
        # CNN skip connections
        f1 = self.skip_conv1(x)
        f2 = self.skip_conv2(f1)
        f3 = self.skip_conv3(f2)
        
        # ViT encoding - manual forward to extract intermediate features
        # Patch embedding
        x_patch = self.vit.conv_proj(x)  # [B, 768, H/16, W/16]
        x_patch = x_patch.flatten(2).transpose(1, 2)  # [B, N, 768]
        
        # Add positional embedding
        n = x_patch.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x_vit = torch.cat([batch_class_token, x_patch], dim=1)
        x_vit = x_vit + self.vit.encoder.pos_embedding
        x_vit = self.vit.encoder.dropout(x_vit)
        
        # Extract intermediate features from transformer blocks
        for i, block in enumerate(self.vit.encoder.layers):
            x_vit = block(x_vit)
            if i == 5:
                mid_features = x_vit[:, 1:, :]  # Exclude cls token
        
        vit_out = x_vit[:, 1:, :]  # Final features (exclude cls)
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Reshape ViT features to spatial
        vit_deep = vit_out.transpose(1, 2).view(B, 768, num_patches_h, num_patches_w)
        vit_mid = mid_features.transpose(1, 2).view(B, 768, num_patches_h, num_patches_w)
        vit_mid = self.vit_proj_mid(vit_mid)
        
        return {
            'skip1': f1,
            'skip2': f2,
            'skip3': f3,
            'vit_mid': vit_mid,
            'vit_deep': vit_deep,
            'bottleneck': vit_deep,  # For auxiliary branch
        }


# ============================================================
# Language-Guided Feature Enhancement (LFE) Module
# ============================================================

class LanguageGuidedFeatureEnhancement(nn.Module):
    """Enhances visual features using language guidance."""
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.visual_proj = nn.Conv2d(visual_dim, hidden_dim, 1)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Conv2d(hidden_dim, visual_dim, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        visual: torch.Tensor,
        text_tokens: torch.Tensor,
        text_sentence: torch.Tensor
    ) -> torch.Tensor:
        B, C_v, H, W = visual.shape
        
        visual_proj = self.visual_proj(visual)
        visual_flat = visual_proj.flatten(2).transpose(1, 2)
        visual_flat = self.norm1(visual_flat)
        
        text_proj = self.text_proj(text_tokens)
        text_proj = self.norm2(text_proj)
        
        attended, _ = self.cross_attn(visual_flat, text_proj, text_proj)
        
        sentence_proj = self.text_proj(text_sentence)
        sentence_expanded = sentence_proj.unsqueeze(1).expand(-1, H*W, -1)
        
        gate_input = torch.cat([attended, sentence_expanded], dim=-1)
        gate = self.gate(gate_input)
        
        enhanced = visual_flat + gate * attended
        enhanced = enhanced.transpose(1, 2).view(B, -1, H, W)
        enhanced = self.out_proj(enhanced)
        
        output = visual + enhanced
        return output


# ============================================================
# Language-Modulated Skip Connection
# ============================================================

class LanguageModulatedSkip(nn.Module):
    """Skip connection modulated by language embedding."""
    
    def __init__(
        self,
        skip_dim: int,
        decoder_dim: int,
        text_dim: int
    ):
        super().__init__()
        
        self.text_to_channel = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, skip_dim),
            nn.Sigmoid()
        )
        
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
        B, C_s, H, W = skip.shape
        
        channel_attn = self.text_to_channel(text_sentence)
        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1)
        
        skip_modulated = skip * channel_attn
        
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=(H, W), mode='bilinear', align_corners=False)
        
        fused = torch.cat([skip_modulated, decoder], dim=1)
        output = self.fusion(fused)
        
        return output


# ============================================================
# NEW: Auxiliary Classification Branch (Global Class Prediction)
# ============================================================

class AuxiliaryClassificationBranch(nn.Module):
    """
    Auxiliary branch for global class prediction.
    
    Forces encoder to learn class-discriminative features by predicting
    which classes are present in the image (multi-label classification).
    
    Architecture:
        bottleneck features -> Global Average Pool -> FC -> sigmoid
        
    Output: [B, num_classes] multi-label probabilities
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bottleneck: [B, C, H, W] bottleneck features
            
        Returns:
            [B, num_classes] class presence logits
        """
        x = self.gap(bottleneck)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        x = self.classifier(x)  # [B, num_classes]
        return x


# ============================================================
# NEW: Deep Supervision Heads
# ============================================================

class DeepSupervisionHead(nn.Module):
    """
    Auxiliary output head for deep supervision.
    
    Takes intermediate decoder features and produces type prediction.
    These auxiliary predictions are upsampled to full resolution and
    contribute to the loss during training.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 6,
        scale_factor: int = 2
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] intermediate features
            target_size: (H, W) target spatial size
            
        Returns:
            [B, num_classes, target_H, target_W] type predictions
        """
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


# ============================================================
# U-Net Style Decoder with Deep Supervision
# ============================================================

class LViT2Decoder(nn.Module):
    """
    U-Net style decoder with language-modulated skip connections
    AND deep supervision outputs at each level.
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [64, 128, 256, 768],
        decoder_dims: List[int] = [512, 256, 128, 64],
        text_dim: int = 512,
        num_classes: int = 6,
        deep_supervision: bool = True
    ):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Initial projection from ViT features
        self.init_proj = nn.Sequential(
            nn.Conv2d(encoder_dims[-1], decoder_dims[0], 1),
            nn.BatchNorm2d(decoder_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with language-modulated skips
        self.up1 = nn.ConvTranspose2d(decoder_dims[0], decoder_dims[1], 4, 2, 1)
        self.skip1 = LanguageModulatedSkip(encoder_dims[2], decoder_dims[1], text_dim)
        
        self.up2 = nn.ConvTranspose2d(decoder_dims[1], decoder_dims[2], 4, 2, 1)
        self.skip2 = LanguageModulatedSkip(encoder_dims[1], decoder_dims[2], text_dim)
        
        self.up3 = nn.ConvTranspose2d(decoder_dims[2], decoder_dims[3], 4, 2, 1)
        self.skip3 = LanguageModulatedSkip(encoder_dims[0], decoder_dims[3], text_dim)
        
        self.up4 = nn.ConvTranspose2d(decoder_dims[3], decoder_dims[3], 4, 2, 1)
        
        # Deep supervision heads (auxiliary type predictions at each level)
        if deep_supervision:
            self.ds_head1 = DeepSupervisionHead(decoder_dims[1], num_classes)  # /8
            self.ds_head2 = DeepSupervisionHead(decoder_dims[2], num_classes)  # /4
            self.ds_head3 = DeepSupervisionHead(decoder_dims[3], num_classes)  # /2
    
    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        text_sentence: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode features with language guidance.
        
        Returns:
            decoded: [B, 64, H, W] decoded features
            deep_outputs: Dict of auxiliary type predictions (if deep_supervision)
        """
        # Get target size from skip1
        target_size = (encoder_features['skip1'].shape[2] * 2,
                       encoder_features['skip1'].shape[3] * 2)  # Full resolution
        
        # Initial projection
        x = self.init_proj(encoder_features['vit_deep'])
        
        # Decoder level 1 (/8)
        x = self.up1(x)
        x = self.skip1(encoder_features['skip3'], x, text_sentence)
        
        deep_outputs = {}
        if self.deep_supervision:
            deep_outputs['ds1'] = self.ds_head1(x, target_size)
        
        # Decoder level 2 (/4)
        x = self.up2(x)
        x = self.skip2(encoder_features['skip2'], x, text_sentence)
        
        if self.deep_supervision:
            deep_outputs['ds2'] = self.ds_head2(x, target_size)
        
        # Decoder level 3 (/2)
        x = self.up3(x)
        x = self.skip3(encoder_features['skip1'], x, text_sentence)
        
        if self.deep_supervision:
            deep_outputs['ds3'] = self.ds_head3(x, target_size)
        
        # Final upsample (/1)
        x = self.up4(x)
        
        return x, deep_outputs


# ============================================================
# HoVer-Net Style Output Heads
# ============================================================

class LViT2OutputHeads(nn.Module):
    """Output heads for nuclei instance segmentation."""
    
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 6
    ):
        super().__init__()
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
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
# Main LViT2 Model for Nuclei Segmentation
# ============================================================

class LViT2NucleiSegmenter(nn.Module):
    """
    LViT2: Enhanced Language-guided Vision Transformer for Nuclei Segmentation.
    
    IMPROVEMENTS over LViT:
        1. Deep Supervision - Auxiliary losses at decoder levels
        2. Auxiliary Classification Branch - Global class prediction
        3. Supports ConvNeXt-Base or ViT backbone
    
    Training outputs:
        - np, hv, type: Main predictions
        - ds1, ds2, ds3: Deep supervision type predictions
        - aux_class: Global class presence prediction
    """
    
    def __init__(
        self,
        text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        num_classes: int = 6,
        freeze_text_encoder: bool = True,
        img_size: int = 256,
        backbone: str = "vit",
        deep_supervision: bool = True,
        aux_classification: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.backbone_type = backbone
        self.deep_supervision = deep_supervision
        self.use_aux_classification = aux_classification
        
        # Text encoder
        self.text_encoder = LViT2TextEncoder(
            model_name=text_encoder,
            embed_dim=embed_dim,
            freeze=freeze_text_encoder
        )
        
        # Hierarchical image encoder
        if backbone == 'convnext_base':
            print(f"[LViT2] Using ConvNeXt-Base backbone (ImageNet pretrained)")
            self.image_encoder = HierarchicalConvNeXtEncoder(
                pretrained=True,
                img_size=img_size
            )
            bottleneck_dim = 768  # After projection
        else:
            print(f"[LViT2] Using ViT-B/16 backbone (ImageNet pretrained)")
            self.image_encoder = HierarchicalViTEncoder(
                pretrained=True,
                img_size=img_size
            )
            bottleneck_dim = 768
        
        # Language-guided feature enhancement
        self.lfe = LanguageGuidedFeatureEnhancement(
            visual_dim=768,
            text_dim=embed_dim,
            hidden_dim=256,
            num_heads=8
        )
        
        # Decoder with deep supervision
        self.decoder = LViT2Decoder(
            encoder_dims=[64, 128, 256, 768],
            decoder_dims=[512, 256, 128, 64],
            text_dim=embed_dim,
            num_classes=num_classes,
            deep_supervision=deep_supervision
        )
        
        # Output heads
        self.output_heads = LViT2OutputHeads(
            in_channels=64,
            num_classes=num_classes
        )
        
        # Auxiliary classification branch
        if aux_classification:
            self.aux_classifier = AuxiliaryClassificationBranch(
                in_channels=bottleneck_dim,
                hidden_dim=256,
                num_classes=num_classes,
                dropout=0.3
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
        instructions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W] input images
            instructions: List of text instructions (also accepts 'texts' for compatibility)
            
        Returns:
            Dictionary with:
                - np: [B, 2, H, W] nuclei presence
                - hv: [B, 2, H, W] horizontal-vertical maps
                - type: [B, num_classes, H, W] type classification
                - ds1, ds2, ds3: Deep supervision outputs (if enabled)
                - aux_class: [B, num_classes] global class prediction (if enabled)
        """
        # Encode text
        text_sentence, text_tokens = self.text_encoder(instructions)
        
        # Encode image (hierarchical features)
        encoder_features = self.image_encoder(images)
        
        # Auxiliary classification from bottleneck (before enhancement)
        outputs = {}
        if self.use_aux_classification:
            # Use bottleneck features for global classification
            aux_class = self.aux_classifier(encoder_features['bottleneck'])
            outputs['aux_class'] = aux_class
        
        # Language-guided feature enhancement at bottleneck
        enhanced_vit = self.lfe(
            encoder_features['vit_deep'],
            text_tokens,
            text_sentence
        )
        encoder_features['vit_deep'] = enhanced_vit
        
        # Decode with language guidance and deep supervision
        decoded, deep_outputs = self.decoder(encoder_features, text_sentence)
        
        # Add deep supervision outputs
        if self.deep_supervision:
            outputs.update(deep_outputs)
        
        # Main output predictions
        main_outputs = self.output_heads(decoded)
        outputs.update(main_outputs)
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_lvit2_model(
    num_classes: int = 6,
    freeze_text_encoder: bool = True,
    img_size: int = 256,
    backbone: str = "vit",
    deep_supervision: bool = True,
    aux_classification: bool = True,
    **kwargs
) -> LViT2NucleiSegmenter:
    """
    Create LViT2 model for nuclei segmentation.
    
    Args:
        num_classes: Number of nucleus classes (default: 6)
        freeze_text_encoder: Whether to freeze text encoder
        img_size: Input image size
        backbone: Backbone type - 'vit' (default) or 'convnext_base'
        deep_supervision: Enable deep supervision (default: True)
        aux_classification: Enable auxiliary classification branch (default: True)
        
    Returns:
        LViT2NucleiSegmenter model
    """
    model = LViT2NucleiSegmenter(
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=512,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder,
        img_size=img_size,
        backbone=backbone,
        deep_supervision=deep_supervision,
        aux_classification=aux_classification
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LViT2] Total parameters: {total_params:,}")
    print(f"[LViT2] Trainable parameters: {trainable_params:,}")
    print(f"[LViT2] Deep Supervision: {deep_supervision}")
    print(f"[LViT2] Auxiliary Classification: {aux_classification}")
    
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LViT2 Nuclei Segmenter Test")
    print("=" * 70)
    
    # Create model
    model = create_lvit2_model(num_classes=6, backbone='vit')
    model.eval()
    
    # Test forward pass
    images = torch.randn(2, 3, 256, 256)
    texts = ["neoplastic cells", "inflammatory and dead cells"]
    
    with torch.no_grad():
        outputs = model(images, texts)
    
    print("\nOutput shapes:")
    for k, v in outputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
    
    print("\n✅ LViT2 test passed!")
