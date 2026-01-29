"""
FC-CLIP: Fully Convolutional CLIP for Dense Vision-Language Inference
Paper: https://arxiv.org/abs/2308.02487 (NeurIPS 2023)
Official Implementation: https://github.com/bytedance/fc-clip

Architecture:
- Converts CLIP attention pooling to convolutional operations
- Enables dense prediction directly from CLIP
- Multi-scale feature extraction with convolutional neck
- Simple yet effective design for open-vocabulary segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class ConvAttnPooling(nn.Module):
    """
    Convolutional attention pooling to replace CLIP's attention pooling.
    Enables spatial output instead of single vector.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        
        # Project to QKV
        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)
        
        # Output projection
        self.out_proj = nn.Linear(in_dim, out_dim)
        
        # Learnable query (replaces CLS token query)
        self.query = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, N, D)
            
        Returns:
            Pooled features (B, N, out_dim) - spatial pooling
        """
        B, N, D = x.shape
        
        # Expand query for batch
        q = self.query.expand(B, N, -1)
        q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        return out


class ConvNeck(nn.Module):
    """
    Convolutional neck for multi-scale feature processing.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Multi-scale convolutions
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ))
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, D, H, W)
            
        Returns:
            Processed features (B, out_dim, H, W)
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        x = self.output_proj(x)
        return x


class DenseDecoder(nn.Module):
    """
    Dense decoder for upsampling features.
    """
    
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        num_upsamples: int = 4,
    ):
        super().__init__()
        
        self.upsamples = nn.ModuleList()
        for i in range(num_upsamples):
            if i == 0:
                self.upsamples.append(nn.Sequential(
                    nn.ConvTranspose2d(in_dim, hidden_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.upsamples.append(nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ))
        
        self.final = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for upsample in self.upsamples:
            x = upsample(x)
        x = self.final(x)
        return x


@register_model("fc_clip")
class FCCLIP(TextGuidedSegmentationBase):
    """
    FC-CLIP: Fully Convolutional CLIP for Dense Vision-Language Inference
    
    Key features:
    - Converts CLIP to fully convolutional architecture
    - Maintains spatial information throughout
    - Simple and efficient design
    - Strong open-vocabulary segmentation performance
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_dim: int = 256,
        num_neck_layers: int = 4,
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
        
        # Convolutional neck
        self.conv_neck = ConvNeck(
            in_dim=self.visual_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_neck_layers,
        )
        
        # Dense decoder
        self.dense_decoder = DenseDecoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_upsamples=4,  # 16x upsampling
        )
        
        # Project to CLIP dimension for similarity
        self.output_proj = nn.Conv2d(hidden_dim, self.clip_embed_dim, kernel_size=1)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        
        self.to(device)
    
    def extract_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract dense features from CLIP visual encoder.
        """
        visual = self.clip_model.visual
        B = image.shape[0]
        
        # Patch embedding
        x = visual.conv1(image)  # (B, D, H/16, W/16)
        patch_h, patch_w = x.shape[2], x.shape[3]
        
        x = x.reshape(B, self.visual_dim, -1).permute(0, 2, 1)  # (B, N, D)
        
        # Add CLS and positional embedding
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
        
        # Transformer
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        
        # Remove CLS and reshape to spatial
        x = x[:, 1:]  # (B, N, D)
        x = x.permute(0, 2, 1).reshape(B, self.visual_dim, patch_h, patch_w)
        
        return x
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to dense features.
        """
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            dense_features = self.extract_dense_features(image)
        
        # Process through convolutional neck
        dense_features = self.conv_neck(dense_features)
        
        # Upsample
        dense_features = self.dense_decoder(dense_features)
        
        # Project to CLIP dimension
        dense_features = self.output_proj(dense_features)
        
        return dense_features  # (B, D, H, W)
    
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
        visual_features = self.encode_image(image)  # (B, D, H', W')
        
        # Encode text
        text_features = self.encode_text(text_prompts)  # (C, D)
        
        # Normalize
        visual_features = F.normalize(visual_features, dim=1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        logits = self.compute_similarity(visual_features, text_features, normalize=False)
        logits = logits * self.logit_scale.exp()
        
        # Ensure correct output size
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'visual_features': visual_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale,
        }


@register_model("fc_clip_convnext")
class FCCLIPConvNeXt(FCCLIP):
    """FC-CLIP variant with ConvNeXt-style neck."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('num_neck_layers', 6)
        super().__init__(**kwargs)
