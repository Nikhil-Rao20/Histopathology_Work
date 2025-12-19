"""
Simple Vision Encoder using pretrained models
Supports: ResNet, ViT, EfficientNet
"""

import torch
import torch.nn as nn
import timm


class VisionEncoder(nn.Module):
    """
    Vision encoder that extracts multi-scale features from images.
    Uses timm library for flexible backbone selection.
    """
    
    def __init__(
        self,
        model_name='resnet50',
        pretrained=True,
        freeze_backbone=False,
        extract_features=True
    ):
        """
        Args:
            model_name: Name of the timm model (e.g., 'resnet50', 'vit_base_patch16_224')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            extract_features: Whether to extract intermediate features (for skip connections)
        """
        super().__init__()
        
        self.model_name = model_name
        self.extract_features = extract_features
        
        # Load pretrained model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=extract_features,  # Return feature pyramid
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimensions
        if extract_features:
            # Feature pyramid: list of features at different scales
            # E.g., ResNet50: [64, 256, 512, 1024, 2048] at [1/2, 1/4, 1/8, 1/16, 1/32] resolution
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.feature_dims = [f.shape[1] for f in features]
                print(f"Vision Encoder '{model_name}' feature dimensions: {self.feature_dims}")
        else:
            # Single feature vector
            self.feature_dims = [self.backbone.num_features]
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Vision backbone frozen: {model_name}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images
        
        Returns:
            features: List of feature maps at different scales
                     E.g., [B, 64, H/2, W/2], [B, 256, H/4, W/4], ...
        """
        features = self.backbone(x)
        return features


class SimpleViTEncoder(nn.Module):
    """
    Simplified ViT encoder using Hugging Face transformers
    """
    
    def __init__(
        self,
        model_name='google/vit-base-patch16-224',
        pretrained=True,
        freeze_backbone=False,
        output_stride=16
    ):
        super().__init__()
        
        from transformers import ViTModel
        
        self.vit = ViTModel.from_pretrained(model_name) if pretrained else ViTModel.from_config(
            ViTModel.config_class.from_pretrained(model_name)
        )
        
        self.hidden_dim = self.vit.config.hidden_size
        self.patch_size = self.vit.config.patch_size
        self.output_stride = output_stride
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print(f"ViT backbone frozen: {model_name}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        
        Returns:
            features: [B, N, D] where N = (H/patch_size) * (W/patch_size)
        """
        B, C, H, W = x.shape
        
        # ViT forward pass
        outputs = self.vit(pixel_values=x)
        
        # Get patch tokens (exclude CLS token)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # [B, N, D]
        
        # Reshape to spatial format
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        
        # [B, N, D] -> [B, D, H', W']
        features = patch_embeddings.transpose(1, 2).reshape(B, self.hidden_dim, n_patches_h, n_patches_w)
        
        return [features]  # Return as list for consistency


if __name__ == "__main__":
    # Test ResNet encoder
    print("Testing ResNet50 encoder...")
    encoder = VisionEncoder(model_name='resnet50', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features = encoder(x)
    print(f"Number of feature levels: {len(features)}")
    for i, f in enumerate(features):
        print(f"  Level {i}: {f.shape}")
    
    print("\nTesting ViT encoder...")
    # Test ViT encoder
    # encoder = SimpleViTEncoder(model_name='google/vit-base-patch16-224', pretrained=False)
    # features = encoder(x)
    # print(f"ViT features: {features[0].shape}")
