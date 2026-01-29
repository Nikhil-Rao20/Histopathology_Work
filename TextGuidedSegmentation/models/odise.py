"""
ODISE: Open-vocabulary DIffusion-based Segmentation
Paper: https://arxiv.org/abs/2303.04803 (CVPR 2023)
Official Implementation: https://github.com/NVlabs/ODISE

Architecture:
- Leverages diffusion model features for segmentation
- Uses Stable Diffusion internal representations
- CLIP text encoder for language conditioning
- Mask prediction with diffusion features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class DiffusionFeatureExtractor(nn.Module):
    """
    Simulated diffusion feature extractor.
    In practice, ODISE uses Stable Diffusion's UNet features.
    Here we simulate with a multi-scale CNN.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_scales: int = 4,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Encoder (simulating diffusion UNet encoder)
        self.encoders = nn.ModuleList()
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(num_scales)]
        
        for i in range(num_scales):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1, stride=2),
                nn.GroupNorm(32, channels[i+1]),
                nn.SiLU(),
                nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                nn.GroupNorm(32, channels[i+1]),
                nn.SiLU(),
            ))
        
        # Store output channels
        self.out_channels = channels[1:]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        return features


class FeatureFusionModule(nn.Module):
    """
    Fuses CLIP and diffusion features.
    """
    
    def __init__(
        self,
        clip_dim: int,
        diffusion_dims: List[int],
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Project diffusion features
        self.diff_projs = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, kernel_size=1)
            for dim in diffusion_dims
        ])
        
        # Project CLIP features
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * len(diffusion_dims), hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        diffusion_features: List[torch.Tensor],
        clip_features: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        """
        Fuse features from diffusion and CLIP.
        """
        # Project and upsample diffusion features
        target_size = diffusion_features[0].shape[2:]
        
        projected = []
        for proj, feat in zip(self.diff_projs, diffusion_features):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)
        
        # Concatenate and fuse
        concat = torch.cat(projected, dim=1)
        fused = self.fusion(concat)
        
        return fused


class MaskTransformerDecoder(nn.Module):
    """
    Transformer decoder for mask prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        num_queries: int = 100,
    ):
        super().__init__()
        
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        memory: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        B = memory.shape[0]
        queries = self.query_embed.expand(B, -1, -1)
        
        for layer in self.layers:
            queries = layer(queries, memory)
        
        return self.norm(queries)


@register_model("odise")
class ODISE(TextGuidedSegmentationBase):
    """
    ODISE: Open-vocabulary DIffusion-based Segmentation
    
    Key features:
    - Uses diffusion model features (simulated here)
    - CLIP text encoder for language understanding
    - Multi-scale feature fusion
    - Strong zero-shot capabilities
    
    Note: This is a simplified version. Full ODISE uses
    actual Stable Diffusion UNet features.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        hidden_dim: int = 256,
        num_queries: int = 100,
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
        
        # Diffusion feature extractor (simulated)
        self.diffusion_extractor = DiffusionFeatureExtractor(
            in_channels=3,
            base_channels=64,
            num_scales=4,
        )
        
        # CLIP visual encoder (for dense features)
        self.patch_size = 16
        
        # Feature fusion
        self.fusion = FeatureFusionModule(
            clip_dim=self.visual_dim,
            diffusion_dims=self.diffusion_extractor.out_channels,
            hidden_dim=hidden_dim,
        )
        
        # Add CLIP features
        self.clip_proj = nn.Sequential(
            nn.Conv2d(self.visual_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Combine diffusion and CLIP
        self.combine = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Mask decoder
        self.mask_decoder = MaskTransformerDecoder(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=3,
            num_queries=num_queries,
        )
        
        # Mask head
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Class head
        self.class_embed = nn.Linear(hidden_dim, hidden_dim)
        
        # Text projection
        self.text_proj = nn.Linear(self.clip_embed_dim, hidden_dim)
        
        # Upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.to(device)
    
    def extract_clip_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract dense CLIP features."""
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
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        
        x = x[:, 1:].permute(0, 2, 1).reshape(B, self.visual_dim, patch_h, patch_w)
        
        return x
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image with diffusion and CLIP features."""
        # Diffusion features
        diff_features = self.diffusion_extractor(image)
        
        # CLIP features
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            clip_features = self.extract_clip_features(image)
        
        # Fuse diffusion features
        fused_diff = self.fusion(diff_features, None)
        
        # Project CLIP features
        clip_proj = self.clip_proj(clip_features)
        
        # Match sizes
        if fused_diff.shape[2:] != clip_proj.shape[2:]:
            fused_diff = F.interpolate(
                fused_diff, size=clip_proj.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # Combine
        combined = torch.cat([fused_diff, clip_proj], dim=1)
        combined = self.combine(combined)
        
        return combined
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode and project text."""
        text_feats = self.encode_text_clip(text_prompts)
        text_feats = self.text_proj(text_feats)
        return text_feats
    
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        B, _, H, W = image.shape
        
        # Encode
        pixel_features = self.encode_image(image)  # (B, D, H', W')
        text_features = self.encode_text(text_prompts)  # (C, D)
        
        # Flatten for decoder
        h, w = pixel_features.shape[2:]
        memory = pixel_features.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        
        # Decode
        queries = self.mask_decoder(memory)  # (B, Q, D)
        
        # Upsample pixel features for mask prediction
        pixel_up = self.upsampler(pixel_features)
        
        # Predict masks
        mask_embeds = self.mask_embed(queries)  # (B, Q, D)
        masks = torch.einsum('bqd,bdhw->bqhw', mask_embeds, pixel_up)
        
        # Class predictions
        class_embeds = self.class_embed(queries)  # (B, Q, D)
        class_embeds = F.normalize(class_embeds, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        
        class_logits = torch.einsum('bqd,cd->bqc', class_embeds, text_features_norm)
        
        # Combine
        masks_up = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        class_probs = class_logits.softmax(dim=-1)
        
        logits = torch.einsum('bqhw,bqc->bchw', masks_up.sigmoid(), class_probs)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'masks': masks_up,
            'queries': queries,
            'class_logits': class_logits,
            'text_features': text_features,
        }
