"""
SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation
Paper: https://arxiv.org/abs/2311.15537 (CVPR 2024)
Official Implementation: https://github.com/xb534/SED

Architecture:
- Simple encoder-decoder design
- Uses hierarchical CLIP features
- Category-guided decoder with text conditioning
- Efficient and effective baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU block."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class CategoryGuidedBlock(nn.Module):
    """
    Category-guided feature modulation block.
    Uses text embeddings to modulate visual features.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Visual feature processing
        self.visual_conv = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Text conditioning
        self.text_proj = nn.Linear(text_dim, hidden_dim * 2)  # gamma and beta
        
        # Output conv
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, D, H, W)
        text_features: torch.Tensor,    # (C, D_text)
    ) -> torch.Tensor:
        """
        Returns category-guided features.
        """
        B, _, H, W = visual_features.shape
        C = text_features.shape[0]
        
        # Process visual features
        x = self.visual_conv(visual_features)  # (B, hidden, H, W)
        hidden = x.shape[1]
        
        # Get text conditioning
        text_cond = self.text_proj(text_features)  # (C, hidden*2)
        gamma, beta = text_cond.chunk(2, dim=-1)   # (C, hidden) each
        
        # Apply conditioning per class
        # Expand for batch: (B, C, hidden, H, W)
        x_expanded = x.unsqueeze(1).expand(-1, C, -1, -1, -1)
        gamma = gamma.view(1, C, hidden, 1, 1)
        beta = beta.view(1, C, hidden, 1, 1)
        
        x_modulated = gamma * x_expanded + beta  # (B, C, hidden, H, W)
        
        # Process each class
        outputs = []
        for c in range(C):
            out = self.output_conv(x_modulated[:, c])
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  # (B, C, hidden, H, W)


class HierarchicalDecoder(nn.Module):
    """
    Hierarchical decoder for multi-scale feature fusion.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 256,
        out_dim: int = 256,
    ):
        super().__init__()
        
        # Lateral connections
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1)
            for in_ch in in_channels
        ])
        
        # Top-down path
        self.smooth = nn.ModuleList([
            ConvBNReLU(hidden_dim, hidden_dim)
            for _ in in_channels
        ])
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * len(in_channels), hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of features from different scales (low to high resolution)
        """
        # Lateral connections
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        
        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Smooth
        smoothed = [smooth(lat) for smooth, lat in zip(self.smooth, laterals)]
        
        # Upsample all to largest resolution and concatenate
        target_size = smoothed[0].shape[2:]
        upsampled = [
            F.interpolate(s, size=target_size, mode='bilinear', align_corners=False)
            for s in smoothed
        ]
        
        # Fuse
        fused = torch.cat(upsampled, dim=1)
        output = self.fusion(fused)
        
        return output


@register_model("sed")
class SED(TextGuidedSegmentationBase):
    """
    SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation
    
    Key features:
    - Simple yet effective encoder-decoder
    - Hierarchical CLIP feature extraction
    - Category-guided decoding
    - Strong baseline performance
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
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
        
        self.hidden_dim = hidden_dim
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Extract from multiple layers
        self.extract_layers = [3, 6, 9, 12]
        
        # Hierarchical decoder
        self.decoder = HierarchicalDecoder(
            in_channels=[self.visual_dim] * len(self.extract_layers),
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        
        # Category-guided block
        self.cat_guided = CategoryGuidedBlock(
            visual_dim=hidden_dim,
            text_dim=self.clip_embed_dim,
            hidden_dim=hidden_dim,
        )
        
        # Output head
        self.output_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        
        self.to(device)
    
    def extract_hierarchical_features(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from multiple CLIP layers."""
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
        
        features = []
        for i, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            if (i + 1) in self.extract_layers:
                feat = x[1:].permute(1, 2, 0)  # Remove CLS, (B, D, N)
                feat = feat.reshape(B, self.visual_dim, patch_h, patch_w)
                features.append(feat)
        
        return features
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image using hierarchical features.
        """
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            hierarchical_features = self.extract_hierarchical_features(image)
        
        # Decode
        decoded = self.decoder(hierarchical_features)
        
        return decoded
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text using CLIP."""
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
        
        # Encode image
        visual_features = self.encode_image(image)  # (B, hidden, H', W')
        
        # Encode text
        text_features = self.encode_text(text_prompts)  # (C, D)
        
        # Category-guided decoding
        cat_features = self.cat_guided(visual_features, text_features)  # (B, C, hidden, H', W')
        
        C = cat_features.shape[1]
        
        # Generate per-class logits
        logits_list = []
        for c in range(C):
            logit = self.output_head(cat_features[:, c])  # (B, 1, H', W')
            logits_list.append(logit)
        
        logits = torch.cat(logits_list, dim=1)  # (B, C, H', W')
        
        # Upsample to original size
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'visual_features': visual_features,
            'text_features': text_features,
        }
