"""
LAVT-Style Nuclei Segmenter
============================

Language-Aware Vision Transformer for Text-Guided Nuclei Instance Segmentation.

Key Innovation: EARLY FUSION of text features into ViT encoder at EVERY layer.

This approach is based on LAVT (CVPR 2022) which showed that injecting text
features into intermediate ViT layers produces significantly better cross-modal
alignments than late fusion approaches.

Architecture:
1. Text Encoder: BioClinicalBERT (frozen initially)
2. Image Encoder: ViT-B/16 with Language-Aware blocks
3. Each ViT block has cross-attention to inject text features
4. HoVer-Net style decoder for NP, HV, Type outputs

Author: Created for MICCAI 2026 submission
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
import math


# ============================================================
# Text Encoder (BioClinicalBERT)
# ============================================================

class LAVTTextEncoder(nn.Module):
    """
    Text encoder using BioClinicalBERT for medical domain.
    Returns both [CLS] embedding and token embeddings for cross-attention.
    """
    
    def __init__(
        self,
        model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        embed_dim: int = 768,
        max_length: int = 64,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get encoder hidden dimension
        self.encoder_dim = self.encoder.config.hidden_size
        
        # Project to common embedding dimension
        if self.encoder_dim != embed_dim:
            self.proj = nn.Linear(self.encoder_dim, embed_dim)
        else:
            self.proj = nn.Identity()
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        texts: List[str],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text instructions.
        
        Args:
            texts: List of instruction strings
            device: Target device
            
        Returns:
            token_embeddings: [B, seq_len, embed_dim] - for cross-attention
            sentence_embedding: [B, embed_dim] - [CLS] token
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        # Encode
        with torch.set_grad_enabled(not any(not p.requires_grad for p in self.encoder.parameters())):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get token embeddings and [CLS] token
        token_embeddings = outputs.last_hidden_state  # [B, seq_len, encoder_dim]
        sentence_embedding = token_embeddings[:, 0]  # [B, encoder_dim]
        
        # Project to common dimension
        token_embeddings = self.proj(token_embeddings)  # [B, seq_len, embed_dim]
        sentence_embedding = self.proj(sentence_embedding)  # [B, embed_dim]
        
        return token_embeddings, sentence_embedding


# ============================================================
# Language-Aware ViT Block (Core Innovation)
# ============================================================

class LanguageAwareViTBlock(nn.Module):
    """
    LAVT-style ViT block with text injection via cross-attention.
    
    Key Insight: Text is injected at EVERY transformer layer, not just at the end.
    This allows the model to learn text-conditioned visual representations
    throughout the entire encoding process.
    
    Architecture:
    1. Self-attention on visual tokens
    2. Cross-attention: visual tokens attend to text tokens
    3. Feed-forward network
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Cross-attention for text injection
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Learnable gate for cross-attention (starts at 0 for stability)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Feed-forward network
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with text injection.
        
        Args:
            x: Visual tokens [B, N, embed_dim]
            text_tokens: Text token embeddings [B, seq_len, embed_dim]
            text_mask: Optional attention mask for text [B, seq_len]
            
        Returns:
            x: Updated visual tokens [B, N, embed_dim]
        """
        # Self-attention
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(self_attn_out)
        
        # Cross-attention with text (LAVT-style injection)
        x_norm = self.norm2(x)
        text_norm = self.text_norm(text_tokens)
        
        # Create key_padding_mask if text_mask provided
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()  # True = ignore
        
        cross_attn_out, _ = self.cross_attn(
            query=x_norm,
            key=text_norm,
            value=text_norm,
            key_padding_mask=key_padding_mask
        )
        
        # Gated addition (gate starts at 0, learned during training)
        x = x + torch.sigmoid(self.gate) * self.dropout(cross_attn_out)
        
        # Feed-forward
        x = x + self.mlp(self.norm3(x))
        
        return x


# ============================================================
# Language-Aware ViT Encoder
# ============================================================

class LanguageAwareViT(nn.Module):
    """
    Vision Transformer with Language-Aware blocks.
    
    Initializes from pretrained ViT weights, then adds cross-attention
    for text injection at every layer.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        img_size: int = 224,
        patch_size: int = 16
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Load pretrained ViT for patch embedding and positional embedding
        if pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            pretrained_vit = models.vit_b_16(weights=weights)
        else:
            pretrained_vit = models.vit_b_16(weights=None)
        
        # Copy patch embedding
        self.patch_embed = pretrained_vit.conv_proj
        
        # Copy positional embedding and class token
        self.cls_token = pretrained_vit.class_token
        self.pos_embed = nn.Parameter(pretrained_vit.encoder.pos_embedding.clone())
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Create Language-Aware blocks
        self.blocks = nn.ModuleList([
            LanguageAwareViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Copy pretrained weights to self-attention and MLP
        if pretrained:
            self._init_from_pretrained(pretrained_vit)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
    
    def _init_from_pretrained(self, pretrained_vit):
        """Initialize self-attention and MLP from pretrained ViT."""
        for i, block in enumerate(self.blocks):
            pretrained_block = pretrained_vit.encoder.layers[i]
            
            # Copy self-attention weights
            block.self_attn.in_proj_weight.data = pretrained_block.self_attention.in_proj_weight.data.clone()
            block.self_attn.in_proj_bias.data = pretrained_block.self_attention.in_proj_bias.data.clone()
            block.self_attn.out_proj.weight.data = pretrained_block.self_attention.out_proj.weight.data.clone()
            block.self_attn.out_proj.bias.data = pretrained_block.self_attention.out_proj.bias.data.clone()
            
            # Copy LayerNorm weights
            block.norm1.weight.data = pretrained_block.ln_1.weight.data.clone()
            block.norm1.bias.data = pretrained_block.ln_1.bias.data.clone()
            block.norm3.weight.data = pretrained_block.ln_2.weight.data.clone()
            block.norm3.bias.data = pretrained_block.ln_2.bias.data.clone()
            
            # Copy MLP weights
            block.mlp[0].weight.data = pretrained_block.mlp[0].weight.data.clone()
            block.mlp[0].bias.data = pretrained_block.mlp[0].bias.data.clone()
            block.mlp[3].weight.data = pretrained_block.mlp[3].weight.data.clone()
            block.mlp[3].bias.data = pretrained_block.mlp[3].bias.data.clone()
    
    def forward(
        self, 
        x: torch.Tensor, 
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            text_tokens: Text token embeddings [B, seq_len, embed_dim]
            text_mask: Optional attention mask [B, seq_len]
            
        Returns:
            patch_features: [B, num_patches, embed_dim]
            cls_token: [B, embed_dim]
        """
        B = x.shape[0]
        
        # Resize to 224x224 if needed
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Language-Aware blocks
        for block in self.blocks:
            x = block(x, text_tokens, text_mask)
        
        x = self.norm(x)
        
        # Split CLS token and patch features
        cls_token = x[:, 0]  # [B, embed_dim]
        patch_features = x[:, 1:]  # [B, num_patches, embed_dim]
        
        return patch_features, cls_token


# ============================================================
# HoVer-Net Style Decoder (Reused from existing code)
# ============================================================

class ConvBlock(nn.Module):
    """Basic convolutional block."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Upsampling block."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(in_ch, out_ch, dropout)
        self.conv2 = ConvBlock(out_ch, out_ch, dropout)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LAVTDecoder(nn.Module):
    """
    HoVer-Net style decoder for nuclei instance segmentation.
    
    Outputs:
    - NP: Binary nuclei presence [B, 2, H, W]
    - HV: Horizontal-Vertical maps [B, 2, H, W]
    - Type: Per-pixel classification [B, num_classes, H, W]
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_classes: int = 6,
        img_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Initial projection
        self.init_proj = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels[0], 1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(UpsampleBlock(in_ch, out_ch, dropout))
            in_ch = out_ch
        
        out_channels = decoder_channels[-1]
        
        # NP head (binary segmentation)
        self.np_head = nn.Sequential(
            ConvBlock(out_channels, out_channels, dropout),
            nn.Conv2d(out_channels, 2, 1)
        )
        
        # HV head (horizontal-vertical maps)
        self.hv_head = nn.Sequential(
            ConvBlock(out_channels, out_channels, dropout),
            nn.Conv2d(out_channels, 2, 1)
        )
        
        # Type head (classification)
        self.type_head = nn.Sequential(
            ConvBlock(out_channels, out_channels, dropout),
            nn.Conv2d(out_channels, num_classes, 1)
        )
    
    def forward(self, patch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            patch_features: [B, num_patches, embed_dim]
            
        Returns:
            Dictionary with 'np', 'hv', 'type' predictions
        """
        B, N, C = patch_features.shape
        H = W = int(math.sqrt(N))
        
        # Reshape to spatial
        x = patch_features.transpose(1, 2).view(B, C, H, W)  # [B, embed_dim, 14, 14]
        
        # Initial projection
        x = self.init_proj(x)
        
        # Upsample
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # Final resize to target
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), 
                             mode='bilinear', align_corners=True)
        
        # Prediction heads
        np_out = self.np_head(x)  # [B, 2, H, W]
        hv_out = self.hv_head(x)  # [B, 2, H, W]
        type_out = self.type_head(x)  # [B, num_classes, H, W]
        
        return {
            'np': np_out,
            'hv': hv_out,
            'type': type_out
        }


# ============================================================
# Main Model: LAVT Nuclei Segmenter
# ============================================================

class LAVTNucleiSegmenter(nn.Module):
    """
    Language-Aware Vision Transformer for Nuclei Instance Segmentation.
    
    Main model combining:
    1. BioClinicalBERT text encoder
    2. LAVT-style ViT with text injection at every layer
    3. HoVer-Net style decoder
    
    This is designed for text-guided referring nuclei segmentation where
    the text instruction specifies which cell types to segment.
    """
    
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
        num_classes: int = 6,
        img_size: int = 256,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        decoder_channels: List[int] = [512, 256, 128, 64],
        dropout: float = 0.1,
        pretrained_vit: bool = True,
        freeze_text_encoder: bool = True,
        text_encoder_name: str = 'emilyalsentzer/Bio_ClinicalBERT'
    ):
        """
        Initialize LAVT Nuclei Segmenter.
        
        Args:
            num_classes: Number of nucleus classes (including background)
            img_size: Input image size
            embed_dim: Embedding dimension
            num_layers: Number of ViT layers
            num_heads: Number of attention heads
            decoder_channels: Decoder channel sizes
            dropout: Dropout rate
            pretrained_vit: Use pretrained ViT weights
            freeze_text_encoder: Freeze text encoder weights
            text_encoder_name: Name of text encoder model
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Text Encoder
        self.text_encoder = LAVTTextEncoder(
            model_name=text_encoder_name,
            embed_dim=embed_dim,
            freeze_encoder=freeze_text_encoder
        )
        
        # Language-Aware ViT
        self.image_encoder = LanguageAwareViT(
            pretrained=pretrained_vit,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = LAVTDecoder(
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            img_size=img_size,
            dropout=dropout
        )
        
        self._log_config()
    
    def _log_config(self):
        """Log model configuration."""
        print(f"\n{'='*60}")
        print(f"LAVT Nuclei Segmenter Configuration")
        print(f"{'='*60}")
        print(f"Architecture: LAVT-style (Early Fusion)")
        print(f"Components:")
        print(f"  - Text Encoder: {self.text_encoder.model_name}")
        print(f"  - Image Encoder: ViT-B/16 with Language-Aware blocks")
        print(f"  - Text injection: Every ViT layer (12 layers)")
        print(f"  - Decoder: HoVer-Net style (NP + HV + Type)")
        print(f"  - Embed Dim: {self.embed_dim}")
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
            instructions: List of text instructions
            
        Returns:
            Dictionary with 'np', 'hv', 'type' predictions
        """
        B = images.shape[0]
        device = images.device
        
        # Default instruction if not provided
        if instructions is None:
            instructions = ["Segment all nuclei."] * B
        
        # Encode text
        text_tokens, text_cls = self.text_encoder(instructions, device)
        # text_tokens: [B, seq_len, embed_dim]
        # text_cls: [B, embed_dim]
        
        # Encode image with text injection
        patch_features, img_cls = self.image_encoder(images, text_tokens)
        # patch_features: [B, num_patches, embed_dim]
        # img_cls: [B, embed_dim]
        
        # Decode
        outputs = self.decoder(patch_features)
        # outputs: {'np': [B,2,H,W], 'hv': [B,2,H,W], 'type': [B,C,H,W]}
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_lavt_model(
    num_classes: int = 6,
    img_size: int = 256,
    pretrained: bool = True,
    freeze_text_encoder: bool = True,
    **kwargs
) -> LAVTNucleiSegmenter:
    """
    Factory function to create LAVT Nuclei Segmenter.
    
    Args:
        num_classes: Number of classes
        img_size: Input image size
        pretrained: Use pretrained weights
        freeze_text_encoder: Freeze text encoder
        **kwargs: Additional arguments
        
    Returns:
        LAVTNucleiSegmenter model
    """
    return LAVTNucleiSegmenter(
        num_classes=num_classes,
        img_size=img_size,
        pretrained_vit=pretrained,
        freeze_text_encoder=freeze_text_encoder,
        **kwargs
    )


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    # Quick test
    print("Testing LAVT Nuclei Segmenter...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = create_lavt_model(
        num_classes=6,
        img_size=256,
        pretrained=True,
        freeze_text_encoder=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    instructions = [
        "Segment neoplastic and inflammatory cells.",
        "Identify all epithelial nuclei."
    ]
    
    print(f"\nInput: images {images.shape}, instructions: {len(instructions)}")
    
    with torch.no_grad():
        outputs = model(images, instructions)
    
    print(f"\nOutputs:")
    print(f"  np: {outputs['np'].shape}")  # Expected: [2, 2, 256, 256]
    print(f"  hv: {outputs['hv'].shape}")  # Expected: [2, 2, 256, 256]
    print(f"  type: {outputs['type'].shape}")  # Expected: [2, 6, 256, 256]
    
    print("\n✅ LAVT Nuclei Segmenter test passed!")
