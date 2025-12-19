"""
Segmentation Decoder: UNet-style Transformer Decoder
Conditioned on grounded visual-text features
Outputs binary masks for activated classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class DecoderBlock(nn.Module):
    """Single decoder block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )
        
        # Convolution blocks
        conv_in_channels = out_channels + skip_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(out_channels)
            # Projection for guidance to match decoder dimension
            self.guidance_proj = nn.Linear(out_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            skip: Skip connection features [B, C_skip, H*2, W*2]
            guidance: Class guidance [B, embed_dim]
            
        Returns:
            Output features [B, C_out, H*2, W*2]
        """
        # Upsample
        x = self.upsample(x)
        
        # Concatenate skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Apply guidance through attention
        if self.use_attention and guidance is not None:
            B, C, H, W = x.shape
            
            # Reshape for attention: [B, H*W, C]
            x_flat = x.flatten(2).transpose(1, 2)
            
            # Project guidance to match decoder dimension if needed
            if guidance.shape[-1] != C:
                if not hasattr(self, 'guidance_proj'):
                    self.guidance_proj = nn.Linear(guidance.shape[-1], C).to(x.device)
                guidance = self.guidance_proj(guidance)
            
            # Guidance as query
            query = guidance.unsqueeze(1)  # [B, 1, C]
            
            # Cross-attention
            attended, _ = self.attention(
                query=query,
                key=x_flat,
                value=x_flat
            )
            
            # Reshape back and add residual
            attended = attended.expand(-1, H * W, -1)
            x_flat = x_flat + attended
            x_flat = self.norm(x_flat)
            
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class SegmentationDecoder(nn.Module):
    """
    UNet-style decoder with class-specific heads.
    
    Args:
        embed_dim: Embedding dimension from encoder
        decoder_channels: List of channel dimensions for decoder layers
        num_classes: Number of pathology classes
        img_size: Output image size
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_classes: int = 5,
        img_size: int = 224,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Initial projection from patch embeddings to feature map
        self.input_proj = nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1)
        
        # Decoder blocks (with increasing spatial resolution)
        self.decoder_blocks = nn.ModuleList()
        self.guidance_projections = nn.ModuleList()  # Project guidance to each decoder level
        
        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1] if i + 1 < len(decoder_channels) else 64
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=0,  # Can add skip connections from encoder
                    use_attention=True,
                    dropout=dropout
                )
            )
            
            # Add projection layer for guidance at this decoder level
            self.guidance_projections.append(
                nn.Linear(embed_dim, out_ch)
            )
        
        # Final output channels
        final_channels = decoder_channels[-1]
        
        # Class-specific segmentation heads
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(final_channels, final_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(final_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(final_channels // 2, 1, kernel_size=1)
            )
            for _ in range(num_classes)
        ])
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        grounded_features: torch.Tensor,
        class_guidance: torch.Tensor,
        class_presence: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode grounded features to segmentation masks.
        
        Args:
            grounded_features: [B, num_patches, embed_dim]
            class_guidance: [B, num_classes, embed_dim]
            class_presence: Binary class presence [B, num_classes]
            return_features: Whether to return intermediate features
            
        Returns:
            masks: Segmentation masks [B, num_classes+1, H, W] (last channel is background)
            features: Intermediate features (optional)
        """
        B, N, D = grounded_features.shape
        H = W = int(N ** 0.5)
        
        # Reshape to spatial: [B, embed_dim, H, W]
        x = grounded_features.transpose(1, 2).reshape(B, D, H, W)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Store intermediate features for skip connections
        intermediate_features = []
        
        # Decoder blocks with class-specific guidance
        for i, (decoder_block, guidance_proj) in enumerate(zip(self.decoder_blocks, self.guidance_projections)):
            # Get guidance for this layer (average over active classes)
            active_mask = class_presence.unsqueeze(-1)  # [B, num_classes, 1]
            active_guidance = (class_guidance * active_mask).sum(dim=1)
            active_guidance = active_guidance / (active_mask.sum(dim=1) + 1e-8)
            # [B, embed_dim]
            
            # Project guidance to match decoder channels at this level
            active_guidance = guidance_proj(active_guidance)  # [B, out_channels]
            
            x = decoder_block(x, skip=None, guidance=active_guidance)
            intermediate_features.append(x)
        
        # Feature refinement
        x = self.refinement(x)
        
        # Upsample to target size
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Generate class-specific masks
        class_masks = []
        for i in range(self.num_classes):
            # Only compute mask if class is present
            if class_presence[:, i].sum() > 0:
                mask = self.class_heads[i](x)
            else:
                mask = torch.zeros(B, 1, self.img_size, self.img_size).to(x.device)
            
            class_masks.append(mask)
        
        # Concatenate all masks: [B, num_classes, H, W]
        masks = torch.cat(class_masks, dim=1)
        
        if return_features:
            return masks, intermediate_features
        else:
            return masks, None


if __name__ == "__main__":
    # Test the decoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    decoder = SegmentationDecoder(
        embed_dim=768,
        decoder_channels=[512, 256, 128, 64],
        num_classes=5,
        img_size=224
    ).to(device)
    
    # Test inputs
    B, num_patches = 2, 196
    grounded_features = torch.randn(B, num_patches, 768).to(device)
    class_guidance = torch.randn(B, 5, 768).to(device)
    class_presence = torch.tensor([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0]
    ], dtype=torch.float32).to(device)
    
    # Forward pass
    masks, features = decoder(
        grounded_features=grounded_features,
        class_guidance=class_guidance,
        class_presence=class_presence,
        return_features=True
    )
    
    print(f"Output masks shape: {masks.shape}")  # [B, 6, 224, 224]
    print(f"Number of intermediate features: {len(features)}")
    
    # Test sigmoid activation
    masks_prob = torch.sigmoid(masks)
    print(f"\nMask probability range: [{masks_prob.min():.3f}, {masks_prob.max():.3f}]")
