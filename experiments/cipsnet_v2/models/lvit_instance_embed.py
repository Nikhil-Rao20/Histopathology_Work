"""
LViT-IE: Language-guided Vision Transformer with Instance Embedding Decoder
=============================================================================

Novel decoder for nuclei instance segmentation that replaces HoVer-Net's
3 independent heads (NP + HV + Type) with a mathematically principled
instance-aware design:

1. NP Head — Binary foreground (kept, proven effective)
2. Distance Transform Head — Normalized EDT replaces HV maps
   - Peaks at 1.0 at each nucleus center, 0 at boundaries
   - Local maxima → instance markers (no Sobel needed)
   - Single channel vs HV's 2 channels
3. Instance Embedding Head — D-dim vector per pixel
   - Pull-push discriminative loss clusters same-instance pixels
   - Mean-shift clustering at inference (no watershed)
   - Handles arbitrary shapes (no star-convexity assumption)
4. Instance-Pooled Classification — Classify whole nuclei, not pixels
   - Pool decoder features per GT instance during training
   - Pool per predicted instance at inference
   - Captures whole-nucleus morphology (size, shape, texture)

Key Mathematical Components:
   - Pull-Push Embedding Loss (De Brabandere et al., 2017)
   - Normalized Distance Transform Regression
   - Instance Feature Pooling with Text Conditioning
   - PCGrad for multi-task gradient balancing (Yu et al., NeurIPS 2020)

Reference:
   - De Brabandere et al., "Semantic Instance Segmentation with a Discriminative Loss Function"
   - Naylor et al., "Segmentation of Nuclei by Deep Regression of the Distance Map" IEEE TMI 2019
   - Yu et al., "Gradient Surgery for Multi-Task Learning" NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

# Reuse the encoder and text components from LViT
from .lvit_nuclei import (
    LViTTextEncoder,
    HierarchicalViTEncoder,
    HierarchicalConvNeXtEncoder,
    LanguageGuidedFeatureEnhancement,
    LanguageModulatedSkip,
)


# ============================================================
# Instance Embedding Decoder
# ============================================================

class LViTIEDecoder(nn.Module):
    """
    U-Net style decoder with language-modulated skip connections.
    Identical structure to LViTDecoder but with different output heads.
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
        self.up1 = nn.ConvTranspose2d(decoder_dims[0], decoder_dims[1], 4, 2, 1)
        self.skip1 = LanguageModulatedSkip(encoder_dims[2], decoder_dims[1], text_dim)
        
        self.up2 = nn.ConvTranspose2d(decoder_dims[1], decoder_dims[2], 4, 2, 1)
        self.skip2 = LanguageModulatedSkip(encoder_dims[1], decoder_dims[2], text_dim)
        
        self.up3 = nn.ConvTranspose2d(decoder_dims[2], decoder_dims[3], 4, 2, 1)
        self.skip3 = LanguageModulatedSkip(encoder_dims[0], decoder_dims[3], text_dim)
        
        self.up4 = nn.ConvTranspose2d(decoder_dims[3], decoder_dims[3], 4, 2, 1)
    
    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        text_sentence: torch.Tensor
    ) -> torch.Tensor:
        x = self.init_proj(encoder_features['vit_deep'])
        
        x = self.up1(x)
        x = self.skip1(encoder_features['skip3'], x, text_sentence)
        
        x = self.up2(x)
        x = self.skip2(encoder_features['skip2'], x, text_sentence)
        
        x = self.up3(x)
        x = self.skip3(encoder_features['skip1'], x, text_sentence)
        
        x = self.up4(x)  # [B, 64, H, W]
        
        return x


# ============================================================
# Output Heads for Instance Embedding Decoder
# ============================================================

class IENPHead(nn.Module):
    """NP Head — Binary foreground segmentation (identical to HoVer-Net NP)."""
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 2, H, W] logits (background/foreground)."""
        return self.head(x)


class IEDistHead(nn.Module):
    """
    Distance Transform Head — Normalized EDT regression.
    
    Predicts a single-channel map where each foreground pixel's value
    is its normalized distance-to-boundary:
        D_norm(p) = D(p) / max_{q in instance} D(q)
    
    Peaks at 1.0 at instance centers, 0 at boundaries.
    Local maxima serve as instance markers (replaces Sobel + HV).
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1, H, W] normalized distance transform prediction."""
        return self.head(x)


class IEEmbedHead(nn.Module):
    """
    Instance Embedding Head — D-dimensional per-pixel embeddings.
    
    Learns to cluster pixels of the same instance together in embedding
    space and push different instances apart. Trained with pull-push
    discriminative loss (De Brabandere et al., 2017).
    
    At inference: mean-shift clustering on foreground embeddings.
    """
    
    def __init__(self, in_channels: int = 64, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embed_dim, 1),
            # No activation — embeddings are unconstrained in R^D
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, embed_dim, H, W] per-pixel embeddings."""
        return self.head(x)


class IEInstancePooledTypeHead(nn.Module):
    """
    Instance-Pooled Classification Head.
    
    Instead of classifying each pixel independently (HoVer-Net style),
    this head pools decoder features per instance and classifies the
    whole nucleus. This captures morphological context (size, shape,
    texture) that per-pixel features miss.
    
    During training: uses GT instance masks for pooling.
    During inference: uses predicted instances from embedding head.
    
    Optionally conditioned on text embedding for text-guided classification.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 6,
        text_dim: int = 512,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Feature refinement before pooling
        self.feature_refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Text conditioning: project text to same space
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Classifier: takes pooled instance features (+ text)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # instance_feat + text_feat
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Fallback per-pixel head for dense prediction (needed for eval compatibility)
        self.pixel_head = nn.Sequential(
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
    
    def forward(
        self,
        decoder_features: torch.Tensor,
        text_sentence: torch.Tensor,
        instance_maps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            decoder_features: [B, C, H, W]
            text_sentence: [B, text_dim]
            instance_maps: [B, H, W] GT instance maps (int32, 0=bg).
                          If None, returns per-pixel prediction only.
        
        Returns:
            dict with:
                'type': [B, num_classes, H, W] per-pixel logits (always)
                'inst_type_logits': [N_total, num_classes] per-instance logits (training only)
                'inst_type_labels': [N_total] GT labels per instance (training only)
        """
        B, _, H, W = decoder_features.shape
        
        # Refine features
        feat = self.feature_refine(decoder_features)  # [B, hidden_dim, H, W]
        
        # Per-pixel prediction (always available, for eval compatibility)
        pixel_logits = self.pixel_head(feat)  # [B, num_classes, H, W]
        
        result = {'type': pixel_logits}
        
        # Instance-pooled classification (training with GT instances)
        if instance_maps is not None:
            inst_logits_list = []
            inst_labels_list = []
            
            text_feat = self.text_proj(text_sentence)  # [B, hidden_dim]
            
            for b in range(B):
                inst_map = instance_maps[b]  # [H, W]
                feat_b = feat[b]  # [hidden_dim, H, W]
                text_b = text_feat[b]  # [hidden_dim]
                
                unique_ids = torch.unique(inst_map)
                unique_ids = unique_ids[unique_ids > 0]  # Exclude background
                
                if len(unique_ids) == 0:
                    continue
                
                for inst_id in unique_ids:
                    mask = (inst_map == inst_id)  # [H, W]
                    if mask.sum() < 5:
                        continue  # Skip tiny fragments
                    
                    # Average pool features for this instance
                    mask_float = mask.float().unsqueeze(0)  # [1, H, W]
                    inst_feat = (feat_b * mask_float).sum(dim=(1, 2)) / (mask_float.sum() + 1e-8)
                    # inst_feat: [hidden_dim]
                    
                    # Concatenate with text features
                    combined = torch.cat([inst_feat, text_b], dim=0)  # [hidden_dim * 2]
                    logit = self.classifier(combined)  # [num_classes]
                    
                    inst_logits_list.append(logit)
                    
                    # Get GT type for this instance (majority vote from pixel-level type_map)
                    # This will be provided via the loss function
            
            if inst_logits_list:
                result['inst_type_logits'] = torch.stack(inst_logits_list, dim=0)
        
        return result


# ============================================================
# Complete LViT-IE Model
# ============================================================

class LViTInstanceEmbedSegmenter(nn.Module):
    """
    LViT-IE: Language-guided Vision Transformer with Instance Embedding Decoder.
    
    Same encoder as LViT (ViT-B/16 + BioClinicalBERT + LFE + modulated skips).
    Novel decoder with:
      - NP head (binary foreground)
      - Distance transform head (normalized EDT, replaces HV maps)
      - Instance embedding head (D-dim per-pixel, pull-push clustering)
      - Instance-pooled type classification (whole-nucleus features)
    """
    
    def __init__(
        self,
        text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        num_classes: int = 6,
        freeze_text_encoder: bool = True,
        img_size: int = 256,
        backbone: str = "vit",
        instance_embed_dim: int = 16,
        freeze_dinov2_backbone: bool = False,
        dinov2_pretrained_path: str = "",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.backbone_type = backbone
        self.instance_embed_dim = instance_embed_dim
        
        # ========== Encoder (SAME as LViT) ==========
        
        # Text encoder
        self.text_encoder = LViTTextEncoder(
            model_name=text_encoder,
            embed_dim=embed_dim,
            freeze=freeze_text_encoder
        )
        
        # Image encoder - support DINOv2 backbones
        if backbone == 'convnext_base':
            print(f"[LViT-IE] Using ConvNeXt-Base backbone (ImageNet pretrained)")
            self.image_encoder = HierarchicalConvNeXtEncoder(pretrained=True, img_size=img_size)
        elif backbone.startswith('dinov2_'):
            # DINOv2 backbones
            from .dinov2_encoder import HierarchicalDINOv2Encoder, create_dinov2_encoder
            print(f"[LViT-IE] Using DINOv2 backbone: {backbone}" +
                  (" [gradient checkpointing ON]" if use_gradient_checkpointing else ""))
            
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
            print(f"[LViT-IE] Using Swin backbone: {backbone}" +
                  (" [gradient checkpointing ON]" if use_gradient_checkpointing else ""))
            self.image_encoder = HierarchicalSwinEncoder(
                model_name=backbone,
                pretrained=True,
                img_size=img_size,
                freeze_backbone=freeze_dinov2_backbone,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            print(f"[LViT-IE] Using ViT-B/16 backbone (ImageNet pretrained)")
            self.image_encoder = HierarchicalViTEncoder(pretrained=True, img_size=img_size)
        
        # Language-guided feature enhancement at bottleneck
        self.lfe = LanguageGuidedFeatureEnhancement(
            visual_dim=768, text_dim=embed_dim, hidden_dim=256, num_heads=8
        )
        
        # ========== Novel Decoder ==========
        
        # U-Net decoder with language-modulated skips (same structure as LViT)
        self.decoder = LViTIEDecoder(
            encoder_dims=[64, 128, 256, 768],
            decoder_dims=[512, 256, 128, 64],
            text_dim=embed_dim,
        )
        
        # Shared feature refinement
        self.shared_refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # ========== Output Heads ==========
        
        # 1. NP Head — binary foreground
        self.np_head = IENPHead(in_channels=64)
        
        # 2. Distance Transform Head — replaces HV maps
        self.dist_head = IEDistHead(in_channels=64)
        
        # 3. Instance Embedding Head — pull-push clustering
        self.embed_head = IEEmbedHead(in_channels=64, embed_dim=instance_embed_dim)
        
        # 4. Instance-Pooled Type Classification
        self.type_head = IEInstancePooledTypeHead(
            in_channels=64,
            num_classes=num_classes,
            text_dim=embed_dim,
            hidden_dim=128,
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
        texts: List[str],
        instance_maps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W]
            texts: List of B text instructions
            instance_maps: [B, H, W] GT instance maps (training only, for pooled classification)
        
        Returns:
            dict with:
                'np': [B, 2, H, W] — foreground logits
                'dist': [B, 1, H, W] — normalized distance transform
                'embed': [B, D, H, W] — instance embeddings
                'type': [B, num_classes, H, W] — per-pixel type logits
                'inst_type_logits': [N, num_classes] — per-instance logits (training only)
        """
        # Encode text
        text_sentence, text_tokens = self.text_encoder(texts)
        
        # Encode image
        encoder_features = self.image_encoder(images)
        
        # Language-guided feature enhancement at bottleneck
        enhanced_vit = self.lfe(
            encoder_features['vit_deep'], text_tokens, text_sentence
        )
        encoder_features['vit_deep'] = enhanced_vit
        
        # Decode with language guidance
        decoded = self.decoder(encoder_features, text_sentence)
        
        # Shared refinement
        decoded = self.shared_refine(decoded)
        
        # ========== Heads ==========
        
        # 1. NP Head
        np_logits = self.np_head(decoded)  # [B, 2, H, W]
        
        # 2. Distance Transform Head
        dist_pred = self.dist_head(decoded)  # [B, 1, H, W]
        
        # 3. Instance Embedding Head
        embed_pred = self.embed_head(decoded)  # [B, D, H, W]
        
        # 4. Instance-Pooled Type Head
        type_result = self.type_head(decoded, text_sentence, instance_maps)
        
        outputs = {
            'np': np_logits,
            'dist': dist_pred,
            'embed': embed_pred,
            'type': type_result['type'],
        }
        
        # Add instance-level logits if available (training)
        if 'inst_type_logits' in type_result:
            outputs['inst_type_logits'] = type_result['inst_type_logits']
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_lvit_ie_model(
    num_classes: int = 6,
    freeze_text_encoder: bool = True,
    img_size: int = 256,
    backbone: str = "vit",
    instance_embed_dim: int = 16,
    freeze_dinov2_backbone: bool = False,
    dinov2_pretrained_path: str = "",
    use_gradient_checkpointing: bool = False,
    **kwargs,
) -> LViTInstanceEmbedSegmenter:
    """
    Create LViT-IE model for nuclei instance segmentation.
    
    Args:
        num_classes: Number of nucleus classes (default: 6)
        freeze_text_encoder: Whether to freeze text encoder
        img_size: Input image size
        backbone: Backbone type - 'vit', 'convnext_base', 'dinov2_vit_b_14', 
                  'dinov2_vit_l_14', 'dinov2_vit_s_14', 'dinov2_vit_g_14'
        instance_embed_dim: Dimension of instance embeddings
        freeze_dinov2_backbone: Whether to freeze DINOv2 backbone
        dinov2_pretrained_path: Path to supervised pretrained DINOv2 checkpoint
        use_gradient_checkpointing: Enable gradient checkpointing on DINOv2
            backbone to save activation memory (recommended for ViT-L/14)
        
    Returns:
        LViTInstanceEmbedSegmenter model
    """
    model = LViTInstanceEmbedSegmenter(
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=512,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder,
        img_size=img_size,
        backbone=backbone,
        instance_embed_dim=instance_embed_dim,
        freeze_dinov2_backbone=freeze_dinov2_backbone,
        dinov2_pretrained_path=dinov2_pretrained_path,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LViT-IE] Total parameters: {total_params:,}")
    print(f"[LViT-IE] Trainable parameters: {trainable_params:,}")
    
    return model
