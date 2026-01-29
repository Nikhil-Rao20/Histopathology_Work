"""
LSeg: Language-driven Semantic Segmentation
Paper: https://arxiv.org/abs/2201.03546 (ICLR 2022)
Official Implementation: https://github.com/isl-org/lang-seg

Architecture:
- Uses CLIP text encoder for language features
- DPT (Dense Prediction Transformer) decoder for dense features
- Computes pixel-text similarity for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class Readout(nn.Module):
    """Readout operation for DPT."""
    
    def __init__(self, in_features: int, readout_type: str = "project"):
        super().__init__()
        self.readout_type = readout_type
        if readout_type == "project":
            self.readout_proj = nn.Sequential(
                nn.Linear(2 * in_features, in_features),
                nn.GELU(),
            )
    
    def forward(self, x: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch tokens (B, N, D)
            readout: CLS token (B, 1, D)
        """
        if self.readout_type == "project":
            readout = readout.expand_as(x)
            x = torch.cat([x, readout], dim=-1)
            x = self.readout_proj(x)
        elif self.readout_type == "add":
            x = x + readout
        return x


class Resample(nn.Module):
    """Resample features to target resolution."""
    
    def __init__(self, in_features: int, out_features: int, scale: int):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(in_features, out_features, kernel_size=1)
        
        if scale == 4:
            self.resample = nn.ConvTranspose2d(
                out_features, out_features, kernel_size=4, stride=4
            )
        elif scale == 2:
            self.resample = nn.ConvTranspose2d(
                out_features, out_features, kernel_size=2, stride=2
            )
        elif scale == 1:
            self.resample = nn.Identity()
        elif scale == 0.5:
            self.resample = nn.Conv2d(
                out_features, out_features, kernel_size=3, stride=2, padding=1
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.resample(x)
        return x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block for DPT decoder."""
    
    def __init__(self, features: int, use_bn: bool = False):
        super().__init__()
        
        self.resConfUnit1 = ResidualConvUnit(features, use_bn)
        self.resConfUnit2 = ResidualConvUnit(features, use_bn)
        
    def forward(self, *xs) -> torch.Tensor:
        output = xs[0]
        
        if len(xs) == 2:
            # Upsample and add skip connection
            output = F.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=True
            )
            output = output + xs[1]
        
        output = self.resConfUnit1(output)
        output = self.resConfUnit2(output)
        
        return output


class ResidualConvUnit(nn.Module):
    """Residual convolution unit."""
    
    def __init__(self, features: int, use_bn: bool = False):
        super().__init__()
        
        layers = [
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(features))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not use_bn),
        ])
        if use_bn:
            layers.append(nn.BatchNorm2d(features))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class DPTHead(nn.Module):
    """
    DPT (Dense Prediction Transformer) decoder head.
    Based on official LSeg implementation.
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        features: int = 256,
        use_bn: bool = False,
        out_channels: List[int] = [256, 512, 1024, 1024],
        readout: str = "project",
    ):
        super().__init__()
        
        self.features = features
        
        # Readout modules
        self.readout_ops = nn.ModuleList([
            Readout(in_channels, readout) for _ in range(4)
        ])
        
        # Resample modules for each layer
        self.resample_ops = nn.ModuleList([
            Resample(in_channels, out_ch, scale)
            for out_ch, scale in zip(out_channels, [4, 2, 1, 0.5])
        ])
        
        # Project to common feature dimension
        self.projects = nn.ModuleList([
            nn.Conv2d(out_ch, features, kernel_size=1)
            for out_ch in out_channels
        ])
        
        # Fusion blocks
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(features, use_bn) for _ in range(4)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        features: List[torch.Tensor],
        cls_token: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """
        Args:
            features: List of 4 feature maps from different layers
            cls_token: CLS token (B, 1, D)
            patch_h, patch_w: Spatial dimensions of patches
            
        Returns:
            Fused features (B, features, H, W)
        """
        B = features[0].shape[0]
        
        layer_outputs = []
        for i, (feat, readout_op, resample_op, project) in enumerate(zip(
            features, self.readout_ops, self.resample_ops, self.projects
        )):
            # Apply readout
            feat = readout_op(feat, cls_token)
            
            # Reshape to spatial
            feat = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
            
            # Resample and project
            feat = resample_op(feat)
            feat = project(feat)
            
            layer_outputs.append(feat)
        
        # Progressive fusion (from coarse to fine)
        path = self.fusion_blocks[3](layer_outputs[3])
        path = self.fusion_blocks[2](path, layer_outputs[2])
        path = self.fusion_blocks[1](path, layer_outputs[1])
        path = self.fusion_blocks[0](path, layer_outputs[0])
        
        # Final head
        out = self.head(path)
        
        return out


@register_model("lseg")
class LSeg(TextGuidedSegmentationBase):
    """
    LSeg: Language-driven Semantic Segmentation
    
    Key features:
    - CLIP text encoder for language features
    - DPT decoder for dense visual features
    - Pixel-text similarity for open-vocabulary segmentation
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        features: int = 256,
        use_bn: bool = False,
        readout: str = "project",
        freeze_clip: bool = True,
        scale_factor: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            clip_model=clip_model,
            freeze_clip=freeze_clip,
            device=device,
        )
        
        self.features = features
        self.scale_factor = scale_factor
        
        # Get CLIP dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768
        
        self.patch_size = 16
        
        # Extract from 4 layers (similar to DPT)
        self.extract_layers = [3, 6, 9, 12]
        
        # DPT decoder
        self.dpt_head = DPTHead(
            in_channels=self.visual_dim,
            features=features,
            use_bn=use_bn,
            readout=readout,
        )
        
        # Output projection (to match text embedding dimension)
        self.out_proj = nn.Conv2d(features, self.clip_embed_dim, kernel_size=1)
        
        self.to(device)
    
    def extract_visual_features(self, image: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract features from multiple layers of CLIP ViT.
        
        Returns:
            Tuple of (list of layer features, CLS token)
        """
        visual = self.clip_model.visual
        
        # Initial processing
        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        # Add CLS and positional embedding
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
        
        x = x.permute(1, 0, 2)  # (N+1, B, D)
        
        layer_features = []
        for i, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            if (i + 1) in self.extract_layers:
                layer_features.append(x[1:].permute(1, 0, 2))  # (B, N, D) without CLS
        
        cls_out = x[0:1].permute(1, 0, 2)  # (B, 1, D)
        
        return layer_features, cls_out
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Extract dense visual features using DPT decoder."""
        B, _, H, W = image.shape
        
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            layer_features, cls_token = self.extract_visual_features(image)
        
        # Compute patch dimensions
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # Apply DPT decoder
        dense_features = self.dpt_head(layer_features, cls_token, patch_h, patch_w)
        
        # Project to CLIP embedding dimension
        dense_features = self.out_proj(dense_features)
        
        return dense_features  # (B, D, H', W')
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text prompts using CLIP."""
        return self.encode_text_clip(text_prompts)
    
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image: Input image (B, 3, H, W)
            text_prompts: List of text prompts
            
        Returns:
            Dictionary with 'logits' (B, C, H, W)
        """
        B, _, H, W = image.shape
        
        # Encode image to dense features
        visual_features = self.encode_image(image)  # (B, D, H', W')
        
        # Encode text
        text_features = self.encode_text(text_prompts)  # (C, D)
        
        # Normalize features
        visual_features = F.normalize(visual_features, dim=1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity (pixel-text)
        logits = self.compute_similarity(visual_features, text_features, normalize=False)
        
        # Scale logits
        logits = logits * self.scale_factor
        
        # Upsample to original size
        logits = self.upsample_logits(logits, (H, W))
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'visual_features': visual_features,
            'text_features': text_features,
        }


@register_model("lseg_vit_l")
class LSegViTL(LSeg):
    """LSeg with ViT-L/14 backbone."""
    
    def __init__(self, **kwargs):
        kwargs['clip_model'] = 'ViT-L/14'
        super().__init__(**kwargs)
