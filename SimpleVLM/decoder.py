"""
UNet-style Decoder for Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling and skip connections
    """
    
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_skip=True
    ):
        super().__init__()
        
        self.use_skip = use_skip
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolutions
        conv_in_channels = in_channels + skip_channels if use_skip else in_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip=None):
        """
        Args:
            x: [B, in_channels, H, W]
            skip: [B, skip_channels, H*2, W*2] or None
        
        Returns:
            out: [B, out_channels, H*2, W*2]
        """
        # Upsample
        x = self.upsample(x)
        
        # Concatenate skip connection
        if self.use_skip and skip is not None:
            # Match spatial dimensions if needed
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class UNetDecoder(nn.Module):
    """
    UNet-style decoder with skip connections
    """
    
    def __init__(
        self,
        encoder_channels=[64, 256, 512, 1024, 2048],  # From encoder
        decoder_channels=[512, 256, 128, 64],
        num_classes=5,
        use_skip_connections=True
    ):
        """
        Args:
            encoder_channels: List of channel dimensions from encoder (low to high resolution)
            decoder_channels: List of channel dimensions for decoder blocks
            num_classes: Number of output classes
            use_skip_connections: Whether to use skip connections from encoder
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_skip_connections = use_skip_connections
        
        # Bottleneck: process the deepest feature
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i+1] if i+1 < len(decoder_channels) else decoder_channels[-1]
            
            # Skip connection from encoder (if available)
            skip_idx = -(i+2)  # -2, -3, -4, ...
            skip_ch = encoder_channels[skip_idx] if abs(skip_idx) <= len(encoder_channels) and use_skip_connections else 0
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    use_skip=use_skip_connections
                )
            )
        
        # Final upsampling and output
        final_channels = decoder_channels[-1]
        
        # Extra upsampling to match input resolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(final_channels, final_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(final_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Output head: one mask per class
        self.output_head = nn.Conv2d(final_channels // 2, num_classes, 1)
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of features from encoder
                             [low_res_to_high_res]: e.g., [f1, f2, f3, f4, f5]
        
        Returns:
            masks: [B, num_classes, H, W]
        """
        # Start from bottleneck (deepest feature)
        x = self.bottleneck(encoder_features[-1])
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get corresponding skip connection
            skip_idx = -(i+2)  # -2, -3, -4, ...
            skip = encoder_features[skip_idx] if abs(skip_idx) <= len(encoder_features) and self.use_skip_connections else None
            
            x = decoder_block(x, skip)
        
        # Final upsampling
        x = self.final_upsample(x)
        
        # Output masks
        masks = self.output_head(x)
        
        return masks


class SimpleDecoder(nn.Module):
    """
    Simpler decoder without skip connections (lighter)
    """
    
    def __init__(
        self,
        in_channels=512,
        decoder_channels=[256, 128, 64],
        num_classes=5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Decoder blocks
        channels = [in_channels] + decoder_channels
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(channels[i], channels[i+1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Extra upsampling to match input resolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1] // 2),
            nn.ReLU(inplace=True)
        )
        
        # Output head
        self.output_head = nn.Conv2d(decoder_channels[-1] // 2, num_classes, 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W] - fused features
        
        Returns:
            masks: [B, num_classes, H_out, W_out]
        """
        for block in self.blocks:
            x = block(x)
        
        x = self.final_upsample(x)
        masks = self.output_head(x)
        
        return masks


if __name__ == "__main__":
    # Test UNet decoder
    print("Testing UNetDecoder...")
    
    encoder_channels = [64, 256, 512, 1024, 2048]
    decoder = UNetDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=[512, 256, 128, 64],
        num_classes=5,
        use_skip_connections=True
    )
    
    # Create dummy encoder features
    encoder_features = [
        torch.randn(2, 64, 56, 56),    # 1/4 resolution
        torch.randn(2, 256, 28, 28),   # 1/8
        torch.randn(2, 512, 14, 14),   # 1/16
        torch.randn(2, 1024, 7, 7),    # 1/32
        torch.randn(2, 2048, 7, 7),    # 1/32
    ]
    
    masks = decoder(encoder_features)
    print(f"Output masks: {masks.shape}")
    
    # Test simple decoder
    print("\nTesting SimpleDecoder...")
    decoder = SimpleDecoder(in_channels=512, decoder_channels=[256, 128, 64], num_classes=5)
    
    x = torch.randn(2, 512, 14, 14)
    masks = decoder(x)
    print(f"Output masks: {masks.shape}")
