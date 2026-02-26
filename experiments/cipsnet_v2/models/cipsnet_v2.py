"""
CIPS-Net V2: Compositional Instruction-Guided Panoptic Segmentation Network
============================================================================

Main model file with support for multiple ablation variants.

Model Variants:
1. BASELINE: ImageEncoder + HoVerNetDecoder (no text)
2. WITH_TEXT: + TextEncoder with simple fusion
3. WITH_CGR: + CompositionalGraphReasoning module
4. WITH_TEXT_CONDITIONED_TYPE: + TextConditionedTypeHead
5. FULL: All components combined

Key Novelties:
- Text-guided nucleus instance segmentation
- Compositional Graph Reasoning for class relationships
- Cross-modal visual-text-class attention
- Text-conditioned type prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto
from typing import Dict, Optional, Tuple, List

from .encoders import ImageEncoder, TextEncoder
from .cgr_module import CompositionalGraphReasoning
from .decoder import HoVerNetDecoder


class ModelVariant(Enum):
    """Supported model variants for ablation study."""
    BASELINE = auto()                    # ViT + HoVer decoder only
    WITH_TEXT = auto()                   # + Text encoder (simple fusion)
    WITH_CGR = auto()                    # + CGR module
    WITH_TEXT_CONDITIONED_TYPE = auto()  # + Text-conditioned type head
    FULL = auto()                        # All components


class SimpleFusion(nn.Module):
    """
    Simple fusion module for WITH_TEXT variant.
    
    Fuses visual features with text embedding through attention.
    """
    
    def __init__(self, visual_dim: int, text_dim: int, output_dim: int):
        super().__init__()
        
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse visual and text features.
        
        Args:
            visual_features: [B, N, visual_dim]
            text_embedding: [B, text_dim]
            
        Returns:
            fused_features: [B, N, output_dim]
        """
        # Project features
        visual = self.visual_proj(visual_features)  # [B, N, output_dim]
        text = self.text_proj(text_embedding)  # [B, output_dim]
        text = text.unsqueeze(1)  # [B, 1, output_dim]
        
        # Cross attention: visual attends to text
        fused, _ = self.cross_attn(visual, text, text)
        
        # Residual connection
        fused = self.fusion_norm(visual + fused)
        
        return fused


class CIPSNetV2(nn.Module):
    """
    CIPS-Net V2: Compositional Instruction-Guided Panoptic Segmentation Network
    
    A modular architecture supporting multiple variants for ablation studies.
    
    Architecture Overview:
    ----------------------
    1. Image Encoder (ViT): Extracts visual features from histopathology images
    2. Text Encoder (DistilBERT): Encodes natural language instructions
    3. CGR Module: Compositional Graph Reasoning for class relationships
    4. HoVer Decoder: Multi-head decoder for NP, HV, Type outputs
    
    Output Format (HoVer-Net style):
    --------------------------------
    - NP Map: [B, 2, H, W] - Binary nuclei presence (background/foreground)
    - HV Map: [B, 2, H, W] - Horizontal-Vertical distance maps for instance separation
    - Type Map: [B, C, H, W] - Per-pixel nucleus classification
    """
    
    # PanNuke nucleus classes
    NUCLEUS_CLASSES = [
        'background',
        'neoplastic',
        'inflammatory',
        'connective',
        'dead',
        'epithelial'
    ]
    
    def __init__(
        self,
        variant: ModelVariant = ModelVariant.FULL,
        backbone: str = 'vit_b_16',
        pretrained: bool = True,
        num_classes: int = 6,
        img_size: int = 256,
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_cgr_layers: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize CIPS-Net V2.
        
        Args:
            variant: Model variant for ablation study
            backbone: ViT backbone type ('vit_b_16', 'vit_l_16', 'vit_b_32')
            pretrained: Use pretrained backbone weights
            num_classes: Number of nucleus classes (including background)
            img_size: Input image size
            decoder_channels: Channel sizes for decoder stages
            num_cgr_layers: Number of CGR graph attention layers
            dropout: Dropout rate
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.variant = variant
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Determine which components to use based on variant
        self.use_text = variant in [
            ModelVariant.WITH_TEXT,
            ModelVariant.WITH_CGR,
            ModelVariant.WITH_TEXT_CONDITIONED_TYPE,
            ModelVariant.FULL
        ]
        self.use_cgr = variant in [
            ModelVariant.WITH_CGR,
            ModelVariant.FULL
        ]
        self.use_text_conditioned_type = variant in [
            ModelVariant.WITH_TEXT_CONDITIONED_TYPE,
            ModelVariant.FULL
        ]
        
        # ==================== Image Encoder ====================
        self.image_encoder = ImageEncoder(
            model_name=backbone,
            pretrained=pretrained,
            img_size=img_size,
            freeze_encoder=freeze_backbone
        )
        embed_dim = self.image_encoder.embed_dim
        
        # ==================== Text Encoder ====================
        if self.use_text:
            self.text_encoder = TextEncoder()
            text_dim = self.text_encoder.embed_dim
        else:
            self.text_encoder = None
            text_dim = embed_dim
        
        # ==================== Fusion Modules ====================
        if self.use_text and not self.use_cgr:
            # Simple fusion for WITH_TEXT variant
            self.simple_fusion = SimpleFusion(embed_dim, text_dim, embed_dim)
        else:
            self.simple_fusion = None
        
        # ==================== CGR Module ====================
        if self.use_cgr:
            self.cgr = CompositionalGraphReasoning(
                num_classes=num_classes - 1,  # Exclude background
                embed_dim=embed_dim,
                num_graph_layers=num_cgr_layers,
                dropout=dropout
            )
        else:
            self.cgr = None
        
        # ==================== Decoder ====================
        # Note: ViT internally uses 224x224, producing 14x14=196 patches
        # The decoder needs to know this internal size
        patch_size = self.image_encoder.patch_size
        encoder_img_size = 224  # ViT's fixed internal size
        
        self.decoder = HoVerNetDecoder(
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            img_size=img_size,  # Final output size
            patch_size=patch_size,
            use_text_conditioned_type=self.use_text_conditioned_type,
            dropout=dropout
        )
        
        # Log configuration
        self._log_config()
    
    def _log_config(self):
        """Log model configuration."""
        print(f"\n{'='*60}")
        print(f"CIPS-Net V2 Configuration")
        print(f"{'='*60}")
        print(f"Variant: {self.variant.name}")
        print(f"Components:")
        print(f"  - Image Encoder: {self.image_encoder.model_name}")
        print(f"  - Text Encoder: {self.use_text}")
        print(f"  - CGR Module: {self.use_cgr}")
        print(f"  - Text-Conditioned Type: {self.use_text_conditioned_type}")
        print(f"  - Embed Dim: {self.image_encoder.embed_dim}")
        print(f"  - Num Classes: {self.num_classes}")
        print(f"{'='*60}\n")
    
    def forward(
        self,
        images: torch.Tensor,
        instructions: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images [B, 3, H, W]
            instructions: Optional list of text instructions (required for text variants)
            
        Returns:
            Dictionary with:
                - np: [B, 2, H, W] nuclei presence logits
                - hv: [B, 2, H, W] horizontal-vertical maps
                - type: [B, num_classes, H, W] type logits
                - attention_scores: (optional) [B, num_classes-1] class attention
        """
        B = images.shape[0]
        device = images.device
        
        # ==================== Image Encoding ====================
        patch_features, cls_token = self.image_encoder(images)
        # patch_features: [B, N, embed_dim]
        # cls_token: [B, embed_dim]
        
        # Initialize text-related variables
        class_guidance = None
        attention_scores = None
        
        # ==================== Text Processing ====================
        if self.use_text:
            if instructions is None:
                # Default instruction if none provided
                instructions = ["Segment all nuclei in the image."] * B
            
            # Text encoding
            token_embeddings, sentence_embedding = self.text_encoder(instructions)
            # token_embeddings: [B, seq_len, text_dim]
            # sentence_embedding: [B, text_dim]
            
            if self.use_cgr:
                # ==================== CGR Processing ====================
                # Extract mentioned classes from text
                class_presence = self.text_encoder.extract_mentioned_classes(
                    instructions, device=patch_features.device
                )
                
                patch_features, class_guidance, attention_scores = self.cgr(
                    visual_features=patch_features,
                    text_features=token_embeddings,
                    sentence_embedding=sentence_embedding,
                    class_presence=class_presence
                )
            elif self.simple_fusion is not None:
                # ==================== Simple Fusion ====================
                patch_features = self.simple_fusion(
                    patch_features, sentence_embedding
                )
        
        # ==================== Decoding ====================
        outputs = self.decoder(
            patch_features,
            class_guidance=class_guidance,
            attention_scores=attention_scores
        )
        
        # Add attention scores to output if available
        if attention_scores is not None:
            outputs['attention_scores'] = attention_scores
        
        return outputs
    
    def get_trainable_params(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get trainable parameters grouped by component.
        
        Useful for applying different learning rates to different components.
        
        Returns:
            Dictionary mapping component names to parameter lists
        """
        params = {
            'image_encoder': list(self.image_encoder.parameters()),
            'decoder': list(self.decoder.parameters())
        }
        
        if self.text_encoder is not None:
            params['text_encoder'] = list(self.text_encoder.parameters())
        
        if self.cgr is not None:
            params['cgr'] = list(self.cgr.parameters())
        
        if self.simple_fusion is not None:
            params['simple_fusion'] = list(self.simple_fusion.parameters())
        
        return params
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters per component.
        
        Returns:
            Dictionary mapping component names to parameter counts
        """
        counts = {}
        
        # Image encoder
        counts['image_encoder'] = sum(p.numel() for p in self.image_encoder.parameters())
        
        # Decoder
        counts['decoder'] = sum(p.numel() for p in self.decoder.parameters())
        
        # Text encoder
        if self.text_encoder is not None:
            counts['text_encoder'] = sum(p.numel() for p in self.text_encoder.parameters())
        
        # CGR
        if self.cgr is not None:
            counts['cgr'] = sum(p.numel() for p in self.cgr.parameters())
        
        # Simple fusion
        if self.simple_fusion is not None:
            counts['simple_fusion'] = sum(p.numel() for p in self.simple_fusion.parameters())
        
        # Total
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return counts


def create_model(
    variant: str = 'FULL',
    **kwargs
) -> CIPSNetV2:
    """
    Factory function to create CIPS-Net V2 models.
    
    Args:
        variant: Model variant name ('BASELINE', 'WITH_TEXT', 'WITH_CGR', 
                 'WITH_TEXT_CONDITIONED_TYPE', 'FULL')
        **kwargs: Additional arguments passed to CIPSNetV2
        
    Returns:
        CIPSNetV2 model instance
    """
    variant_map = {
        'BASELINE': ModelVariant.BASELINE,
        'WITH_TEXT': ModelVariant.WITH_TEXT,
        'WITH_CGR': ModelVariant.WITH_CGR,
        'WITH_TEXT_CONDITIONED_TYPE': ModelVariant.WITH_TEXT_CONDITIONED_TYPE,
        'FULL': ModelVariant.FULL
    }
    
    if variant.upper() not in variant_map:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variant_map.keys())}")
    
    return CIPSNetV2(variant=variant_map[variant.upper()], **kwargs)


# ============================================================
# Example Usage and Testing
# ============================================================

if __name__ == '__main__':
    # Test all model variants
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    instructions = [
        "Segment neoplastic and inflammatory cells.",
        "Find all epithelial nuclei in the image."
    ]
    
    variants = ['BASELINE', 'WITH_TEXT', 'WITH_CGR', 'WITH_TEXT_CONDITIONED_TYPE', 'FULL']
    
    for variant_name in variants:
        print(f"\n{'='*60}")
        print(f"Testing variant: {variant_name}")
        print(f"{'='*60}")
        
        model = create_model(variant_name, pretrained=False).to(device)
        
        # Count parameters
        param_counts = model.count_parameters()
        print(f"Parameters:")
        for name, count in param_counts.items():
            print(f"  {name}: {count:,}")
        
        # Forward pass
        if variant_name == 'BASELINE':
            outputs = model(images)
        else:
            outputs = model(images, instructions)
        
        # Check output shapes
        print(f"\nOutput shapes:")
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")
        
        # Verify output shapes
        assert outputs['np'].shape == (batch_size, 2, 256, 256), f"NP shape mismatch"
        assert outputs['hv'].shape == (batch_size, 2, 256, 256), f"HV shape mismatch"
        assert outputs['type'].shape == (batch_size, 6, 256, 256), f"Type shape mismatch"
        
        print(f"\n✓ Variant {variant_name} passed all tests!")
        
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("All variants tested successfully!")
    print(f"{'='*60}")
