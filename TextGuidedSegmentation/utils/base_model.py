"""
Base model class for text-guided segmentation models.
Provides common interface for all 14 models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import clip


class TextGuidedSegmentationBase(nn.Module, ABC):
    """
    Abstract base class for text-guided segmentation models.
    All models must implement:
    - encode_image: Extract visual features from image
    - encode_text: Extract text features from prompts
    - forward: Full forward pass returning segmentation mask
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        freeze_clip: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.clip_model_name = clip_model
        self.freeze_clip = freeze_clip
        self._device = device
        
        # Load CLIP model (shared by most models)
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=device)
        
        # Convert CLIP model to float32 for compatibility
        # (CLIP loads in fp16 on CUDA by default in newer versions)
        self.clip_model = self.clip_model.float()
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Common dimensions
        self.clip_embed_dim = self.clip_model.visual.output_dim
        self.clip_width = self.clip_model.visual.width if hasattr(self.clip_model.visual, 'width') else 768
    
    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from image.
        
        Args:
            image: Input image tensor (B, 3, H, W)
            
        Returns:
            Visual features (B, D, H', W') or (B, N, D)
        """
        pass
    
    @abstractmethod
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Extract text features from prompts.
        
        Args:
            text_prompts: List of text prompts
            
        Returns:
            Text features (N, D)
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            image: Input image (B, 3, H, W)
            text_prompts: List of text prompts (one per class)
            
        Returns:
            Dictionary containing:
            - 'logits': Segmentation logits (B, C, H, W)
            - 'pred_mask': Predicted mask (B, H, W) (optional)
            - Other model-specific outputs
        """
        pass
    
    def predict(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> torch.Tensor:
        """
        Predict segmentation mask.
        
        Args:
            image: Input image (B, 3, H, W)
            text_prompts: List of text prompts
            
        Returns:
            Predicted mask (B, H, W)
        """
        output = self.forward(image, text_prompts)
        logits = output['logits']
        return logits.argmax(dim=1)
    
    def encode_text_clip(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text using CLIP text encoder.
        Common utility used by most models.
        
        Args:
            text_prompts: List of text prompts
            
        Returns:
            Text features (N, D)
        """
        text_tokens = clip.tokenize(text_prompts).to(self._device)
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        return text_features
    
    def interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        new_grid_size: int,
        old_grid_size: int = 14,  # CLIP ViT-B/16 default is 224/16=14
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings for different input resolutions.
        
        CLIP models are trained on 224x224 images, but we may use different sizes.
        This function interpolates the positional embeddings to match the new size.
        
        Args:
            pos_embed: Original positional embeddings (N, D) where N = old_grid_size**2 + 1
            new_grid_size: New spatial grid size (e.g., 16 for 256x256 with patch_size=16)
            old_grid_size: Original grid size (default 14 for CLIP ViT-B/16)
            
        Returns:
            Interpolated positional embeddings (new_grid_size**2 + 1, D)
        """
        if new_grid_size == old_grid_size:
            return pos_embed
        
        # Separate cls token and positional embeddings
        cls_pos = pos_embed[:1]  # (1, D)
        patch_pos = pos_embed[1:]  # (old_grid**2, D)
        
        # Reshape to 2D grid
        D = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, D).permute(0, 3, 1, 2)
        
        # Interpolate
        patch_pos = F.interpolate(
            patch_pos,
            size=(new_grid_size, new_grid_size),
            mode='bicubic',
            align_corners=False,
        )
        
        # Reshape back
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid_size * new_grid_size, D)
        
        # Concatenate with cls token
        new_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        
        return new_pos_embed
    
    def get_clip_visual_features(
        self, 
        image: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract visual features using CLIP visual encoder.
        
        Args:
            image: Input image (B, 3, H, W)
            return_all_tokens: If True, return all patch tokens, else just CLS
            
        Returns:
            Visual features (B, D) or (B, N+1, D) if return_all_tokens
        """
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            if return_all_tokens:
                # Get all tokens from visual transformer
                visual = self.clip_model.visual
                x = visual.conv1(image)  # (B, width, grid, grid)
                B, D, h, w = x.shape
                x = x.reshape(B, D, -1)  # (B, width, grid**2)
                x = x.permute(0, 2, 1)  # (B, grid**2, width)
                
                # Add class token
                cls_token = visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)  # (B, grid**2 + 1, width)
                
                # Interpolate positional embeddings if needed
                pos_embed = visual.positional_embedding
                if x.shape[1] != pos_embed.shape[0]:
                    # Determine old grid size from pos_embed
                    old_grid_size = int((pos_embed.shape[0] - 1) ** 0.5)
                    new_grid_size = int((x.shape[1] - 1) ** 0.5)
                    pos_embed = self.interpolate_pos_embed(pos_embed, new_grid_size, old_grid_size)
                
                x = x + pos_embed
                x = visual.ln_pre(x)
                
                x = x.permute(1, 0, 2)  # (N+1, B, width)
                x = visual.transformer(x)
                x = x.permute(1, 0, 2)  # (B, N+1, width)
                x = visual.ln_post(x)
                
                if visual.proj is not None:
                    x = x @ visual.proj
                
                return x  # (B, N+1, embed_dim)
            else:
                return self.clip_model.encode_image(image)  # (B, embed_dim)
    
    def compute_similarity(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between visual and text features.
        
        Args:
            visual_features: (B, D, H, W) or (B, N, D)
            text_features: (C, D)
            normalize: Whether to normalize features
            
        Returns:
            Similarity scores (B, C, H, W) or (B, C, N)
        """
        if normalize:
            visual_features = F.normalize(visual_features, dim=1 if visual_features.dim() == 4 else -1)
            text_features = F.normalize(text_features, dim=-1)
        
        if visual_features.dim() == 4:
            # (B, D, H, W) case
            B, D, H, W = visual_features.shape
            visual_flat = visual_features.permute(0, 2, 3, 1).reshape(B * H * W, D)
            sim = visual_flat @ text_features.t()  # (B*H*W, C)
            sim = sim.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            # (B, N, D) case
            sim = torch.einsum('bnd,cd->bnc', visual_features, text_features)
            sim = sim.permute(0, 2, 1)  # (B, C, N)
        
        return sim
    
    def upsample_logits(
        self,
        logits: torch.Tensor,
        target_size: Tuple[int, int],
        mode: str = 'bilinear',
    ) -> torch.Tensor:
        """
        Upsample logits to target size.
        
        Args:
            logits: Input logits (B, C, H, W)
            target_size: Target (H, W)
            mode: Interpolation mode
            
        Returns:
            Upsampled logits (B, C, target_H, target_W)
        """
        return F.interpolate(
            logits, 
            size=target_size, 
            mode=mode, 
            align_corners=False if mode == 'bilinear' else None
        )
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class ConvBlock(nn.Module):
    """Standard convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block for upsampling with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_bn: bool = True,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBlock(in_channels // 2 + skip_channels, out_channels, use_bn=use_bn),
            ConvBlock(out_channels, out_channels, use_bn=use_bn),
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from different levels
            
        Returns:
            List of FPN feature maps
        """
        # Build laterals
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[2:], 
                mode='nearest'
            )
        
        # Output convs
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        
        return outputs


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, **kwargs) -> TextGuidedSegmentationBase:
    """Build a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def get_model(name: str, **kwargs) -> TextGuidedSegmentationBase:
    """
    Get a model by name.
    
    Args:
        name: Model name (case-insensitive)
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        Instantiated model
    """
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name_lower](**kwargs)


def list_models() -> list:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())
