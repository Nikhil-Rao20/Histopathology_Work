"""
CLIPSeg: Image Segmentation Using Text and Image Prompts
Paper: https://arxiv.org/abs/2112.10003 (CVPR 2022)
Official Implementation: https://github.com/timojl/clipseg

Architecture:
- Uses CLIP ViT encoder with feature extraction at multiple layers
- Transformer-based decoder with FiLM conditioning from text
- Supports both text and image-based conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class Interpolate(nn.Module):
    """Interpolation module for upsampling."""
    
    def __init__(self, scale_factor: float, mode: str = "bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, 
                           align_corners=False if self.mode == "bilinear" else None)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder block for CLIPSeg.
    Based on official implementation.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 4,
        depth: int = 3,
        reduce_dim: int = 64,
    ):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.reduce_dim = reduce_dim
        
        # Reduce dimension projection
        self.reduce_proj = nn.Linear(embed_dim, reduce_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=reduce_dim,
                nhead=num_heads,
                dim_feedforward=reduce_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(reduce_dim, 1)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,  # (B, N, embed_dim)
        conditional: torch.Tensor,     # (B, C, embed_dim)
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: Visual features from CLIP encoder
            conditional: Text conditioning features
            
        Returns:
            Decoded features (B, C, N)
        """
        B, N, _ = visual_tokens.shape
        C = conditional.shape[1]
        
        # Reduce dimension
        visual_tokens = self.reduce_proj(visual_tokens)  # (B, N, reduce_dim)
        conditional = self.reduce_proj(conditional)       # (B, C, reduce_dim)
        
        outputs = []
        for c in range(C):
            # Get conditioning for this class
            cond = conditional[:, c:c+1, :]  # (B, 1, reduce_dim)
            
            # Decode with transformer
            x = visual_tokens
            for layer in self.layers:
                x = layer(x, cond)
            
            # Project to single channel
            out = self.output_proj(x).squeeze(-1)  # (B, N)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  # (B, C, N)


class FiLMLayer(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) layer.
    Applies affine transformation conditioned on input.
    """
    
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gamma = nn.Linear(cond_dim, out_dim)
        self.beta = nn.Linear(cond_dim, out_dim)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return gamma * x + beta


@register_model("clipseg")
class CLIPSeg(TextGuidedSegmentationBase):
    """
    CLIPSeg: Image Segmentation Using Text and Image Prompts
    
    Key features:
    - Extracts features from multiple CLIP ViT layers
    - Uses FiLM conditioning for text-visual fusion
    - Transformer-based decoder for dense prediction
    - Supports both text and visual prompts
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        extract_layers: Tuple[int, ...] = (3, 6, 9),
        reduce_dim: int = 64,
        decoder_depth: int = 3,
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
        
        self.extract_layers = extract_layers
        self.reduce_dim = reduce_dim
        
        # Get CLIP visual dimensions
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_dim = self.clip_model.visual.width
        else:
            self.visual_dim = 768  # Default for ViT-B
        
        # Patch size for spatial reconstruction
        self.patch_size = 16
        self.num_patches = (image_size // self.patch_size) ** 2
        
        # Feature projection from each extracted layer
        self.layer_projections = nn.ModuleList([
            nn.Linear(self.visual_dim, reduce_dim)
            for _ in extract_layers
        ])
        
        # FiLM conditioning layers
        self.film_layers = nn.ModuleList([
            FiLMLayer(reduce_dim, reduce_dim, self.clip_embed_dim)
            for _ in extract_layers
        ])
        
        # Combine multi-layer features
        self.combine = nn.Sequential(
            nn.Linear(reduce_dim * len(extract_layers), reduce_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduce_dim, reduce_dim),
        )
        
        # Decoder head
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(reduce_dim, reduce_dim, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim // 2, 1, kernel_size=1),
        )
        
        self.to(device)
    
    def extract_visual_features(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from multiple layers of CLIP visual encoder.
        Based on official CLIPSeg implementation.
        """
        visual = self.clip_model.visual
        
        # Initial convolution
        x = visual.conv1(image)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid^2)
        x = x.permute(0, 2, 1)  # (B, grid^2, width)
        
        # Add class token and positional embedding
        cls_token = visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, grid^2 + 1, width)
        # Interpolate positional embeddings if needed
        pos_embed = visual.positional_embedding
        if x.shape[1] != pos_embed.shape[0]:
            old_grid_size = int((pos_embed.shape[0] - 1) ** 0.5)
            new_grid_size = int((x.shape[1] - 1) ** 0.5)
            pos_embed = self.interpolate_pos_embed(pos_embed, new_grid_size, old_grid_size)
        x = x + pos_embed
        x = visual.ln_pre(x)
        
        # Extract features at specified layers
        x = x.permute(1, 0, 2)  # (N+1, B, width)
        
        extracted_features = []
        for i, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            if i + 1 in self.extract_layers:
                extracted_features.append(x.permute(1, 0, 2))  # (B, N+1, width)
        
        return extracted_features
    
    def encode_image(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-layer visual features."""
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            return self.extract_visual_features(image)
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text prompts using CLIP."""
        return self.encode_text_clip(text_prompts)
    
    def compute_conditional(
        self,
        features_list: List[torch.Tensor],
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute FiLM-conditioned features.
        
        Args:
            features_list: List of visual features from each layer
            text_features: Text embeddings (C, D)
            
        Returns:
            Conditioned features (B, C, reduce_dim, H, W)
        """
        B = features_list[0].shape[0]
        C = text_features.shape[0]
        
        # Grid size from patches
        grid_size = int(math.sqrt(features_list[0].shape[1] - 1))  # Exclude CLS
        
        outputs = []
        for c in range(C):
            text_cond = text_features[c:c+1]  # (1, D)
            text_cond = text_cond.expand(B, -1)  # (B, D)
            
            layer_outputs = []
            for i, (features, proj, film) in enumerate(zip(
                features_list, self.layer_projections, self.film_layers
            )):
                # Remove CLS token and project
                feat = features[:, 1:, :]  # (B, N, width)
                feat = proj(feat)           # (B, N, reduce_dim)
                
                # Apply FiLM conditioning
                feat = film(feat, text_cond)  # (B, N, reduce_dim)
                
                layer_outputs.append(feat)
            
            # Combine multi-layer features
            combined = torch.cat(layer_outputs, dim=-1)  # (B, N, reduce_dim * num_layers)
            combined = self.combine(combined)            # (B, N, reduce_dim)
            
            # Reshape to spatial
            combined = combined.permute(0, 2, 1)  # (B, reduce_dim, N)
            combined = combined.reshape(B, self.reduce_dim, grid_size, grid_size)
            
            outputs.append(combined)
        
        return torch.stack(outputs, dim=1)  # (B, C, reduce_dim, H, W)
    
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image: Input image (B, 3, H, W)
            text_prompts: List of text prompts (one per class)
            
        Returns:
            Dictionary with 'logits' (B, C, H, W) and intermediate outputs
        """
        B, _, H, W = image.shape
        
        # Encode image (multi-layer features)
        visual_features = self.encode_image(image)
        
        # Encode text
        text_features = self.encode_text(text_prompts)
        
        # Compute conditioned features
        conditioned = self.compute_conditional(visual_features, text_features)
        # conditioned: (B, C, reduce_dim, h, w)
        
        # Decode each class
        C = conditioned.shape[1]
        logits_list = []
        for c in range(C):
            feat = conditioned[:, c]  # (B, reduce_dim, h, w)
            logit = self.decoder(feat)  # (B, 1, H', W')
            logits_list.append(logit)
        
        logits = torch.cat(logits_list, dim=1)  # (B, C, H', W')
        
        # Upsample to original size
        logits = self.upsample_logits(logits, (H, W))
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'visual_features': visual_features,
            'text_features': text_features,
        }


@register_model("clipseg_rd64")
class CLIPSegRD64(CLIPSeg):
    """CLIPSeg with reduce_dim=64 (default configuration)."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('reduce_dim', 64)
        super().__init__(**kwargs)


@register_model("clipseg_rd128")
class CLIPSegRD128(CLIPSeg):
    """CLIPSeg with reduce_dim=128."""
    
    def __init__(self, **kwargs):
        kwargs['reduce_dim'] = 128
        super().__init__(**kwargs)
