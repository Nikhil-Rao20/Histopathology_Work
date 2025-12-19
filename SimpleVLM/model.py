"""
Main SimpleVLM Model for Multi-Class Segmentation
Combines Vision Encoder + Text Encoder + Fusion + Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_encoder import VisionEncoder
from text_encoder import TextEncoder
from fusion import SimpleFusion, MultiScaleFusion
from decoder import UNetDecoder, SimpleDecoder


class SimpleVLMSegmenter(nn.Module):
    """
    Simple Vision-Language Model for Multi-Class Segmentation
    
    Architecture:
    1. Vision Encoder: Extract visual features (ResNet/ViT)
    2. Text Encoder: Encode natural language instructions (BERT/DistilBERT)
    3. Fusion: Combine visual and textual features (FiLM modulation)
    4. Decoder: Generate multi-class segmentation masks (UNet)
    """
    
    def __init__(
        self,
        # Vision encoder
        vision_backbone='resnet50',
        vision_pretrained=True,
        freeze_vision=False,
        
        # Text encoder
        text_model='distilbert-base-uncased',
        freeze_text=False,
        max_text_length=77,
        
        # Fusion
        fusion_dim=512,
        
        # Decoder
        decoder_channels=[256, 128, 64],
        num_classes=5,
        use_skip_connections=True,
        
        # Model mode
        use_simple_decoder=False
    ):
        """
        Args:
            vision_backbone: Vision encoder backbone ('resnet50', 'resnet101', etc.)
            vision_pretrained: Use ImageNet pretrained weights
            freeze_vision: Freeze vision encoder
            
            text_model: Text encoder model name (HuggingFace)
            freeze_text: Freeze text encoder
            max_text_length: Maximum text sequence length
            
            fusion_dim: Dimension for fused features
            
            decoder_channels: Channel dimensions for decoder blocks
            num_classes: Number of segmentation classes
            use_skip_connections: Use skip connections in decoder
            
            use_simple_decoder: Use simpler decoder without skip connections
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_simple_decoder = use_simple_decoder
        
        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_backbone,
            pretrained=vision_pretrained,
            freeze_backbone=freeze_vision,
            extract_features=True
        )
        
        # Get feature dimensions from vision encoder
        self.vis_feature_dims = self.vision_encoder.feature_dims
        
        # 2. Text Encoder
        self.text_encoder = TextEncoder(
            model_name=text_model,
            freeze_encoder=freeze_text,
            max_length=max_text_length
        )
        
        self.text_dim = self.text_encoder.hidden_dim
        
        # 3. Fusion Module
        if use_simple_decoder:
            # Simple fusion: just use the deepest feature
            self.fusion = SimpleFusion(
                vis_dim=self.vis_feature_dims[-1],
                text_dim=self.text_dim,
                hidden_dim=fusion_dim
            )
        else:
            # Multi-scale fusion: fuse at each encoder level
            self.fusion = MultiScaleFusion(
                vis_dims=self.vis_feature_dims,
                text_dim=self.text_dim,
                hidden_dim=fusion_dim
            )
        
        # 4. Segmentation Decoder
        if use_simple_decoder:
            self.decoder = SimpleDecoder(
                in_channels=fusion_dim,
                decoder_channels=decoder_channels,
                num_classes=num_classes
            )
        else:
            # Adjust encoder channels to fusion_dim
            fused_encoder_channels = [fusion_dim] * len(self.vis_feature_dims)
            
            self.decoder = UNetDecoder(
                encoder_channels=fused_encoder_channels,
                decoder_channels=decoder_channels,
                num_classes=num_classes,
                use_skip_connections=use_skip_connections
            )
        
        print(f"\n{'='*80}")
        print(f"SimpleVLM Model Initialized")
        print(f"{'='*80}")
        print(f"Vision: {vision_backbone} (pretrained={vision_pretrained}, frozen={freeze_vision})")
        print(f"Text: {text_model} (frozen={freeze_text})")
        print(f"Fusion dim: {fusion_dim}")
        print(f"Decoder: {'Simple' if use_simple_decoder else 'UNet'} (classes={num_classes})")
        print(f"{'='*80}\n")
    
    def forward(self, images, texts):
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W] - input images
            texts: List of strings (batch) - natural language instructions
        
        Returns:
            masks: [B, num_classes, H, W] - predicted segmentation masks (logits)
        """
        B, C, H, W = images.shape
        
        # 1. Vision encoding
        vis_features = self.vision_encoder(images)  # List of multi-scale features
        
        # 2. Text encoding
        text_embeddings, text_pooled, text_mask = self.text_encoder(texts)
        # text_embeddings: [B, L, text_dim]
        # text_pooled: [B, text_dim]
        
        # 3. Fusion
        if self.use_simple_decoder:
            # Simple: fuse only the deepest feature
            fused = self.fusion(vis_features[-1], text_pooled)  # [B, fusion_dim, H', W']
            
            # 4. Decode
            masks = self.decoder(fused)  # [B, num_classes, H_out, W_out]
        else:
            # Multi-scale: fuse at each level
            fused_features = self.fusion(vis_features, text_pooled)  # List of [B, fusion_dim, H_i, W_i]
            
            # 4. Decode with skip connections
            masks = self.decoder(fused_features)  # [B, num_classes, H_out, W_out]
        
        # Resize to match input resolution
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        
        return masks
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'vision_backbone': self.vision_encoder.model_name,
            'text_model': self.text_encoder.model_name,
            'decoder_type': 'Simple' if self.use_simple_decoder else 'UNet'
        }


if __name__ == "__main__":
    # Test the model
    print("Testing SimpleVLM with ResNet50 + DistilBERT...")
    
    model = SimpleVLMSegmenter(
        vision_backbone='resnet50',
        vision_pretrained=False,
        freeze_vision=False,
        text_model='distilbert-base-uncased',
        freeze_text=False,
        fusion_dim=512,
        decoder_channels=[256, 128, 64],
        num_classes=5,
        use_skip_connections=True,
        use_simple_decoder=False
    )
    
    # Test input
    images = torch.randn(2, 3, 224, 224)
    texts = [
        "Segment neoplastic tissue in this histopathology image",
        "Identify inflammatory regions and dead tissue"
    ]
    
    # Forward pass
    masks = model(images, texts)
    print(f"\nOutput masks: {masks.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        if 'parameters' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
