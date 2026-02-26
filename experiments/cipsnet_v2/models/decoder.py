"""
Decoder Modules for CIPS-Net V2
================================

Contains:
1. HoVerNetDecoder: Multi-head decoder for NP, HV, and Type maps
2. TextConditionedTypeHead: Novel type prediction head conditioned on text attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsampling block with skip connection support."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SharedDecoder(nn.Module):
    """
    Shared decoder backbone that upsamples features to target resolution.
    
    Takes patch features (from ViT at 224x224) and upsamples to target img_size.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_channels: List[int] = [512, 256, 128, 64],
        img_size: int = 256,
        patch_size: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels
        self.img_size = img_size
        self.patch_size = patch_size
        
        # ViT uses 224x224 internally, so feature_size is always 224/patch_size
        self.vit_img_size = 224
        self.feature_size = self.vit_img_size // patch_size  # 14 for patch_size=16
        
        # Initial projection from patch features
        self.init_proj = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels[0], 1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling stages
        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(UpsampleBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        
        # Calculate size after upsample blocks
        current_size = self.feature_size * (2 ** len(decoder_channels[1:]))
        # With feature_size=14 and 3 upsample blocks (2x each): 14 -> 28 -> 56 -> 112
        
        # Final upsampling to target resolution using interpolation
        self.target_size = img_size
        self.current_size = current_size
        
        self.out_channels = decoder_channels[-1]
    
    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            patch_features: [B, num_patches, embed_dim]
            
        Returns:
            decoder_features: [B, out_channels, target_size, target_size]
        """
        B, N, C = patch_features.shape
        H = W = int(N ** 0.5)
        
        # Reshape to spatial format
        x = patch_features.transpose(1, 2).view(B, C, H, W)
        # [B, embed_dim, H', W']
        
        # Initial projection
        x = self.init_proj(x)
        
        # Upsampling stages
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # Final upsampling to target size using interpolation
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), 
                             mode='bilinear', align_corners=True)
        
        return x


class NPHead(nn.Module):
    """
    Nuclei Presence Head: Binary segmentation (foreground/background).
    """
    
    def __init__(self, in_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, in_channels, dropout=dropout)
        self.conv2 = ConvBlock(in_channels, in_channels // 2, dropout=dropout)
        self.out_conv = nn.Conv2d(in_channels // 2, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            np_map: [B, 2, H, W] - logits for background/foreground
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out_conv(x)


class HVHead(nn.Module):
    """
    Horizontal-Vertical Head: Predicts distance maps for instance separation.
    
    Output: 2 channels for horizontal and vertical gradients.
    """
    
    def __init__(self, in_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, in_channels, dropout=dropout)
        self.conv2 = ConvBlock(in_channels, in_channels // 2, dropout=dropout)
        self.out_conv = nn.Conv2d(in_channels // 2, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            hv_map: [B, 2, H, W] - horizontal and vertical distance maps
        """
        x = self.conv1(x)
        x = self.conv2(x)
        # Tanh to bound output to [-1, 1]
        return torch.tanh(self.out_conv(x))


class TypeHead(nn.Module):
    """
    Standard Type Head: Per-pixel nucleus classification.
    """
    
    def __init__(self, in_channels: int, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes  # Including background
        
        self.conv1 = ConvBlock(in_channels, in_channels, dropout=dropout)
        self.conv2 = ConvBlock(in_channels, in_channels // 2, dropout=dropout)
        self.out_conv = nn.Conv2d(in_channels // 2, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            type_map: [B, num_classes, H, W] - logits for each class
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out_conv(x)


class TextConditionedTypeHead(nn.Module):
    """
    Novel: Text-Conditioned Type Head
    
    Uses class guidance vectors and attention scores from CGR module
    to weight the type predictions based on text instructions.
    
    This is a KEY NOVELTY: The type prediction is influenced by what
    classes are mentioned in the text instruction.
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        num_classes: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes  # Including background
        self.embed_dim = embed_dim
        
        # Feature processing
        self.conv1 = ConvBlock(in_channels, in_channels, dropout=dropout)
        self.conv2 = ConvBlock(in_channels, in_channels // 2, dropout=dropout)
        
        # Class guidance integration
        # Project class guidance to spatial features
        self.class_proj = nn.Linear(embed_dim, in_channels // 2)
        
        # Attention-based fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 2 + num_classes - 1, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.out_conv = nn.Conv2d(in_channels // 2, num_classes, 1)
        
        # Learnable class bias (can be weighted by attention)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
    
    def forward(
        self,
        x: torch.Tensor,
        class_guidance: Optional[torch.Tensor] = None,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional text conditioning.
        
        Args:
            x: Decoder features [B, in_channels, H, W]
            class_guidance: Class guidance vectors [B, num_classes-1, embed_dim]
            attention_scores: Class attention weights [B, num_classes-1]
            
        Returns:
            type_map: [B, num_classes, H, W] - text-conditioned class logits
        """
        B, C, H, W = x.shape
        
        # Feature processing
        x = self.conv1(x)
        x = self.conv2(x)  # [B, C//2, H, W]
        
        if class_guidance is not None and attention_scores is not None:
            # Project class guidance
            class_feat = self.class_proj(class_guidance)  # [B, num_classes-1, C//2]
            
            # Weight by attention scores
            class_feat = class_feat * attention_scores.unsqueeze(-1)
            
            # Average pool class features and expand to spatial
            class_feat_pooled = class_feat.mean(dim=1)  # [B, C//2]
            class_feat_spatial = class_feat_pooled.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            
            # Also create per-class attention maps
            attn_maps = attention_scores.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            # [B, num_classes-1, H, W]
            
            # Fuse features
            x = torch.cat([x, attn_maps], dim=1)
            x = self.fusion(x)
            
            # Add class-weighted modulation
            x = x + class_feat_spatial * 0.1
        
        # Output
        logits = self.out_conv(x)
        
        # Add learnable class bias weighted by attention
        if attention_scores is not None:
            # Background gets base bias, other classes get attention-weighted bias
            bias = self.class_bias.clone()
            bias[1:] = bias[1:] + attention_scores.mean(dim=0) * 0.5
            logits = logits + bias.view(1, -1, 1, 1)
        
        return logits


class HoVerNetDecoder(nn.Module):
    """
    Complete HoVer-Net style decoder with three output heads:
    1. NP Head: Nuclei Presence (binary segmentation)
    2. HV Head: Horizontal-Vertical distance maps
    3. Type Head: Nucleus classification
    
    Can optionally use TextConditionedTypeHead instead of standard TypeHead.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_classes: int = 6,
        img_size: int = 256,
        patch_size: int = 16,
        use_text_conditioned_type: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_text_conditioned_type = use_text_conditioned_type
        
        # Shared decoder backbone
        self.shared_decoder = SharedDecoder(
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            img_size=img_size,
            patch_size=patch_size,
            dropout=dropout
        )
        
        out_channels = self.shared_decoder.out_channels
        
        # Output heads
        self.np_head = NPHead(out_channels, dropout=dropout)
        self.hv_head = HVHead(out_channels, dropout=dropout)
        
        if use_text_conditioned_type:
            self.type_head = TextConditionedTypeHead(
                out_channels, embed_dim, num_classes, dropout=dropout
            )
        else:
            self.type_head = TypeHead(out_channels, num_classes, dropout=dropout)
    
    def forward(
        self,
        patch_features: torch.Tensor,
        class_guidance: Optional[torch.Tensor] = None,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            patch_features: [B, num_patches, embed_dim]
            class_guidance: [B, num_classes-1, embed_dim] (optional, for text conditioning)
            attention_scores: [B, num_classes-1] (optional, for text conditioning)
            
        Returns:
            Dictionary with:
                - np: [B, 2, H, W] nuclei presence logits
                - hv: [B, 2, H, W] horizontal-vertical maps
                - type: [B, num_classes, H, W] type logits
        """
        # Shared decoder features
        decoder_features = self.shared_decoder(patch_features)
        
        # Output heads
        np_map = self.np_head(decoder_features)
        hv_map = self.hv_head(decoder_features)
        
        if self.use_text_conditioned_type and class_guidance is not None:
            type_map = self.type_head(decoder_features, class_guidance, attention_scores)
        else:
            type_map = self.type_head(decoder_features)
        
        return {
            'np': np_map,
            'hv': hv_map,
            'type': type_map
        }
