"""
UNet Architecture for Image Segmentation
=========================================

Classic UNet implementation with encoder-decoder structure and skip connections.
Suitable for binary and multi-class segmentation tasks.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv2d -> BatchNorm -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: ConvTranspose2d -> Concatenate with skip connection -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: input from previous layer (to be upsampled)
        x2: skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch due to pooling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution: 1x1 Conv to produce final segmentation mask
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet Architecture
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB)
        out_channels (int): Number of output classes
        base_filters (int): Number of filters in the first layer (default: 64)
        depth (int): Number of downsampling/upsampling levels (default: 4)
        bilinear (bool): Use bilinear upsampling instead of transposed conv (default: False)
    
    Input:
        x: [B, in_channels, H, W]
    
    Output:
        logits: [B, out_channels, H, W]
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, depth=4, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.bilinear = bilinear
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_filters)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = base_filters
        for i in range(depth):
            out_ch = in_ch * 2
            self.down_blocks.append(Down(in_ch, out_ch))
            in_ch = out_ch
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            out_ch = in_ch // 2
            self.up_blocks.append(Up(in_ch, out_ch, bilinear))
            in_ch = out_ch
        
        # Output convolution
        self.outc = OutConv(base_filters, out_channels)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, in_channels, H, W]
        
        Returns:
            logits: [B, out_channels, H, W]
        """
        # Encoder with skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.inc(x)
        skip_connections.append(x)
        
        # Downsampling
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        
        # Remove last skip connection (it's the bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Upsampling with skip connections
        for up in self.up_blocks:
            skip = skip_connections.pop()
            x = up(x, skip)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'UNet',
            'input_channels': self.in_channels,
            'output_channels': self.out_channels,
            'base_filters': self.base_filters,
            'depth': self.depth,
            'bilinear_upsampling': self.bilinear,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }


# Test code
if __name__ == '__main__':
    print("="*80)
    print("UNet Architecture Test")
    print("="*80)
    
    # Create model
    model = UNet(in_channels=3, out_channels=5, base_filters=64, depth=4)
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        if 'parameters' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    H, W = 224, 224
    x = torch.randn(batch_size, 3, H, W)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {model.out_channels}, {H}, {W}]")
    
    assert output.shape == (batch_size, model.out_channels, H, W), "Shape mismatch!"
    print("\n✓ Test passed!")
    
    # Model summary
    print("\n" + "="*80)
    print("Model Architecture Summary")
    print("="*80)
    print(model)
    print("="*80)