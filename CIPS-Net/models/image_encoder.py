"""
Image Encoder: ViT-B/ViT-L for Histopathology Images
Extracts dense patch embeddings from input images
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Tuple, Optional


class ImageEncoder(nn.Module):
    """
    Vision Transformer encoder for histopathology images.
    
    Args:
        model_name: 'vit_b_16' (uses ViT-small 384-dim) or 'vit_l_16' (uses ViT-base 768-dim)
        pretrained: Whether to use ImageNet pretrained weights
        img_size: Input image size (default: 224)
        patch_size: Patch size for ViT (default: 16)
        embed_dim: Embedding dimension (will be projected to this size)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
    """
    
    def __init__(
        self,
        model_name: str = 'vit_b_16',
        pretrained: bool = True,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Load Hugging Face ViT model (using smaller models for faster download)
        if model_name == 'vit_b_16':
            # Using ViT-small instead of base for faster download (~80MB vs 340MB)
            model_id = "WinKawaks/vit-small-patch16-224"
            self.embed_dim = 384  # ViT-small has 384 dim
        elif model_name == 'vit_l_16':
            # For large, use base model (~340MB)
            model_id = "google/vit-base-patch16-224"
            self.embed_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Load pretrained or initialize from scratch
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_id)
        else:
            config = ViTConfig.from_pretrained(model_id)
            self.vit = ViTModel(config)
        
        # Projection layer to ensure consistent embedding dimension
        self.proj = nn.Linear(self.embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For extracting intermediate features (for skip connections)
        self.feature_layers = [3, 6, 9, 12] if num_layers == 12 else [6, 12, 18, 24]
        
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Extract patch embeddings and intermediate features.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
            intermediate_features: List of feature maps from intermediate layers
        """
        # Get ViT outputs with hidden states
        outputs = self.vit(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract patch embeddings (exclude CLS token)
        # Last hidden state: [B, num_patches+1, hidden_dim]
        patch_embeddings = outputs.last_hidden_state[:, 1:]  # [B, num_patches, hidden_dim]
        
        # Get intermediate features for skip connections
        intermediate_features = []
        hidden_states = outputs.hidden_states  # Tuple of all layer outputs
        for layer_idx in self.feature_layers:
            if layer_idx < len(hidden_states):
                # Exclude CLS token from intermediate features
                intermediate_features.append(hidden_states[layer_idx][:, 1:])
        
        # Project and normalize
        patch_embeddings = self.proj(patch_embeddings)
        patch_embeddings = self.norm(patch_embeddings)
        patch_embeddings = self.dropout(patch_embeddings)
        
        return patch_embeddings, intermediate_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only patch embeddings.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            patch_embeddings: [B, num_patches, embed_dim]
        """
        patch_embeddings, _ = self.forward_features(x)
        return patch_embeddings
    
    def get_patch_grid_size(self) -> Tuple[int, int]:
        """Returns the spatial dimensions of the patch grid."""
        grid_size = self.img_size // self.patch_size
        return (grid_size, grid_size)
    
    def reshape_to_spatial(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reshape patch embeddings to spatial feature map.
        
        Args:
            patch_embeddings: [B, num_patches, embed_dim]
            
        Returns:
            spatial_features: [B, embed_dim, H, W]
        """
        B, N, D = patch_embeddings.shape
        H = W = int(N ** 0.5)
        
        spatial_features = patch_embeddings.transpose(1, 2)  # [B, embed_dim, num_patches]
        spatial_features = spatial_features.reshape(B, D, H, W)
        
        return spatial_features


if __name__ == "__main__":
    # Test the image encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = ImageEncoder(
        model_name='vit_b_16',
        pretrained=False,
        img_size=224,
        embed_dim=768
    ).to(device)
    
    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)
    
    # Forward pass
    patch_embeddings = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
    # Test spatial reshape
    spatial = encoder.reshape_to_spatial(patch_embeddings)
    print(f"Spatial features shape: {spatial.shape}")
    
    # Test with features
    patch_embeddings, features = encoder.forward_features(x)
    print(f"\nIntermediate features:")
    for i, feat in enumerate(features):
        print(f"  Layer {i}: {feat.shape}")
