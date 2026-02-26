"""
CRIS: CLIP-Driven Referring Image Segmentation for Nuclei Instance Segmentation
================================================================================

Based on: "CRIS: CLIP-Driven Referring Image Segmentation" (CVPR 2022)
Paper: https://arxiv.org/abs/2111.15174

Key Innovation:
    - Text-to-pixel contrastive learning for fine-grained alignment
    - CLIP encoders for robust vision-language representations
    - Vision-Language Decoder with multi-scale fusion
    - Contrastive loss between text and pixel embeddings

Architecture:
    1. CLIP Image Encoder (ViT or ResNet backbone)
    2. CLIP Text Encoder (Transformer)
    3. Vision-Language Decoder with cross-attention
    4. Text-to-Pixel Contrastive Learning module
    5. HoVer-Net style output heads (NP, HV, Type)

Adapted for nuclei instance segmentation on PanNuke dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============================================================
# Text Encoder (BioClinicalBERT for medical domain)
# ============================================================

class CRISTextEncoder(nn.Module):
    """
    Text encoder using BioClinicalBERT for medical text understanding.
    Returns both sentence embedding and token embeddings for contrastive learning.
    """
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        freeze: bool = True,
        max_length: int = 128
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for BERT
        self.max_length = max_length
        
        # Project to common embedding dimension
        self.text_proj = nn.Linear(self.hidden_size, embed_dim)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            sentence_embed: [B, embed_dim] - sentence-level embedding
            token_embeds: [B, L, embed_dim] - token-level embeddings
        """
        device = next(self.parameters()).device
        
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        
        # Encode
        outputs = self.encoder(**tokens)
        hidden_states = outputs.last_hidden_state  # [B, L, 768]
        
        # Project to common dimension
        token_embeds = self.text_proj(hidden_states)  # [B, L, embed_dim]
        
        # Sentence embedding (CLS token)
        sentence_embed = token_embeds[:, 0, :]  # [B, embed_dim]
        
        return sentence_embed, token_embeds


# ============================================================
# Image Encoder (ViT with multi-scale features)
# ============================================================

class CRISImageEncoder(nn.Module):
    """
    Image encoder using ViT backbone with multi-scale feature extraction.
    Supports any input resolution by interpolating positional embeddings.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        pretrained: bool = True,
        freeze: bool = False,
        img_size: int = 256
    ):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        # Load pretrained ViT
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.hidden_dim = 768  # ViT-B hidden dimension
        self.patch_size = 16
        self.img_size = img_size
        
        # Store target grid size
        self.target_h = self.target_w = img_size // self.patch_size
        
        # Remove classification head
        self.vit.heads = nn.Identity()
        
        # Multi-scale feature projection
        self.proj_high = nn.Conv2d(self.hidden_dim, embed_dim, 1)  # High-level
        self.proj_mid = nn.Conv2d(self.hidden_dim, embed_dim, 1)   # Mid-level
        self.proj_low = nn.Conv2d(self.hidden_dim, embed_dim, 1)   # Low-level
        
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def _interpolate_pos_embed(self, device):
        """Interpolate positional embeddings for new image size (lazy, on same device)."""
        old_pos_embed = self.vit.encoder.pos_embedding.to(device)  # [1, 197, 768]
        
        new_h, new_w = self.target_h, self.target_w
        num_patches = new_h * new_w
        
        # Separate CLS token and patch embeddings
        cls_embed = old_pos_embed[:, 0:1, :]  # [1, 1, 768]
        patch_embed = old_pos_embed[:, 1:, :]  # [1, 196, 768]
        
        # Reshape and interpolate
        old_h = old_w = 14
        patch_embed = patch_embed.reshape(1, old_h, old_w, self.hidden_dim)
        patch_embed = patch_embed.permute(0, 3, 1, 2)  # [1, 768, 14, 14]
        
        patch_embed = F.interpolate(
            patch_embed, size=(new_h, new_w), mode='bicubic', align_corners=False
        )  # [1, 768, 16, 16]
        
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, num_patches, self.hidden_dim)
        
        # Concatenate CLS and new patch embeddings
        new_pos_embed = torch.cat([cls_embed, patch_embed], dim=1)  # [1, 257, 768]
        
        return new_pos_embed
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from image.
        
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            Dictionary with multi-scale features
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Patch embedding
        x = self.vit.conv_proj(x)  # [B, 768, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, N, 768]
        
        # Add class token and interpolated position embedding
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Get interpolated pos embed on correct device
        pos_embed = self._interpolate_pos_embed(device)
        x = x + pos_embed
        
        # Extract features at different layers
        features = []
        for i, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            if i in [3, 7, 11]:  # Early, middle, late layers
                features.append(x[:, 1:, :])  # Remove CLS token
        
        # Reshape to spatial format
        h, w = H // 16, W // 16
        
        feat_low = features[0].transpose(1, 2).view(B, self.hidden_dim, h, w)
        feat_mid = features[1].transpose(1, 2).view(B, self.hidden_dim, h, w)
        feat_high = features[2].transpose(1, 2).view(B, self.hidden_dim, h, w)
        
        # Project to common dimension
        feat_low = self.proj_low(feat_low)    # [B, embed_dim, h, w]
        feat_mid = self.proj_mid(feat_mid)    # [B, embed_dim, h, w]
        feat_high = self.proj_high(feat_high)  # [B, embed_dim, h, w]
        
        return {
            'low': feat_low,
            'mid': feat_mid,
            'high': feat_high,
            'cls': x[:, 0, :]  # CLS token
        }


# ============================================================
# Vision-Language Decoder
# ============================================================

class VLDecoderBlock(nn.Module):
    """
    Vision-Language Decoder block with cross-attention.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention for visual features
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention with text
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual: [B, N, C] visual features
            text: [B, L, C] text features
            
        Returns:
            [B, N, C] refined visual features
        """
        # Self-attention
        x = visual
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Cross-attention with text
        x = x + self.cross_attn(self.norm2(x), text, text)[0]
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


class VisionLanguageDecoder(nn.Module):
    """
    Multi-scale Vision-Language Decoder.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Decoder blocks for each scale
        self.decoder_high = nn.ModuleList([
            VLDecoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_mid = nn.ModuleList([
            VLDecoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_low = nn.ModuleList([
            VLDecoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode multi-scale features with text guidance.
        
        Args:
            features: Dict with 'low', 'mid', 'high' features [B, C, H, W]
            text_embeds: [B, L, C] text token embeddings
            
        Returns:
            [B, C, H, W] fused features
        """
        B, C, H, W = features['high'].shape
        
        # Flatten spatial dimensions
        feat_high = features['high'].flatten(2).transpose(1, 2)  # [B, N, C]
        feat_mid = features['mid'].flatten(2).transpose(1, 2)
        feat_low = features['low'].flatten(2).transpose(1, 2)
        
        # Apply decoder blocks
        for block in self.decoder_high:
            feat_high = block(feat_high, text_embeds)
        
        for block in self.decoder_mid:
            feat_mid = block(feat_mid, text_embeds)
        
        for block in self.decoder_low:
            feat_low = block(feat_low, text_embeds)
        
        # Reshape back to spatial
        feat_high = feat_high.transpose(1, 2).view(B, C, H, W)
        feat_mid = feat_mid.transpose(1, 2).view(B, C, H, W)
        feat_low = feat_low.transpose(1, 2).view(B, C, H, W)
        
        # Fuse multi-scale features
        fused = torch.cat([feat_low, feat_mid, feat_high], dim=1)
        fused = self.fusion(fused)
        
        return fused


# ============================================================
# Text-to-Pixel Contrastive Module
# ============================================================

class TextPixelContrastive(nn.Module):
    """
    Text-to-Pixel contrastive learning module.
    Aligns text embeddings with corresponding pixel embeddings.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Projection heads for contrastive learning
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )
    
    def forward(
        self,
        pixel_features: torch.Tensor,
        text_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute text-pixel similarity for contrastive learning.
        
        Args:
            pixel_features: [B, C, H, W] pixel embeddings
            text_embed: [B, C] sentence embedding
            
        Returns:
            pixel_text_sim: [B, H, W] similarity map
            contrastive_features: [B, C, H, W] for contrastive loss
        """
        B, C, H, W = pixel_features.shape
        
        # Project features
        text_proj = self.text_proj(text_embed)  # [B, C]
        pixel_proj = self.pixel_proj(pixel_features)  # [B, C, H, W]
        
        # Normalize
        text_proj = F.normalize(text_proj, dim=-1)
        pixel_proj = F.normalize(pixel_proj, dim=1)
        
        # Compute similarity
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        similarity = (pixel_proj * text_proj).sum(dim=1)  # [B, H, W]
        similarity = similarity / self.temperature
        
        return similarity, pixel_proj


# ============================================================
# HoVer-Net Style Output Heads
# ============================================================

class CRISOutputHeads(nn.Module):
    """
    Output heads for nuclei instance segmentation.
    Produces NP (nuclear pixels), HV (horizontal-vertical), and Type maps.
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 6
    ):
        super().__init__()
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.np_head = nn.Conv2d(32, 2, 1)  # Binary: nucleus/background
        self.hv_head = nn.Conv2d(32, 2, 1)  # Horizontal-Vertical gradients
        self.type_head = nn.Conv2d(32, num_classes, 1)  # Class probabilities
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H/16, W/16] encoded features
            
        Returns:
            Dictionary with 'np', 'hv', 'type' predictions
        """
        # Upsample to full resolution
        x = self.up1(x)  # [B, 256, H/8, W/8]
        x = self.up2(x)  # [B, 128, H/4, W/4]
        x = self.up3(x)  # [B, 64, H/2, W/2]
        x = self.up4(x)  # [B, 32, H, W]
        
        # Output predictions
        np_out = self.np_head(x)
        hv_out = self.hv_head(x)
        type_out = self.type_head(x)
        
        return {
            'np': np_out,
            'hv': hv_out,
            'type': type_out
        }


# ============================================================
# Main CRIS Model for Nuclei Segmentation
# ============================================================

class CRISNucleiSegmenter(nn.Module):
    """
    CRIS: CLIP-Driven Referring Image Segmentation for Nuclei.
    
    Key features:
        - Text-to-pixel contrastive learning
        - Multi-scale vision-language fusion
        - HoVer-Net style instance segmentation heads
    """
    
    def __init__(
        self,
        text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512,
        num_classes: int = 6,
        freeze_text_encoder: bool = True,
        freeze_image_encoder: bool = False
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Encoders
        self.text_encoder = CRISTextEncoder(
            model_name=text_encoder,
            embed_dim=embed_dim,
            freeze=freeze_text_encoder
        )
        
        self.image_encoder = CRISImageEncoder(
            embed_dim=embed_dim,
            pretrained=True,
            freeze=freeze_image_encoder
        )
        
        # Vision-Language Decoder
        self.vl_decoder = VisionLanguageDecoder(
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=3
        )
        
        # Text-Pixel Contrastive Module
        self.contrastive = TextPixelContrastive(
            embed_dim=embed_dim,
            temperature=0.07
        )
        
        # Output heads
        self.output_heads = CRISOutputHeads(
            in_channels=embed_dim,
            num_classes=num_classes
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W] input images
            texts: List of text instructions
            
        Returns:
            Dictionary with:
                - 'np': [B, 2, H, W] nuclear pixel predictions
                - 'hv': [B, 2, H, W] horizontal-vertical predictions
                - 'type': [B, num_classes, H, W] type predictions
                - 'text_pixel_sim': [B, H/16, W/16] similarity map
                - 'contrastive_features': [B, C, H/16, W/16] for loss
        """
        # Encode text
        text_sentence, text_tokens = self.text_encoder(texts)
        
        # Encode image (multi-scale)
        image_features = self.image_encoder(images)
        
        # Vision-language decoding
        fused_features = self.vl_decoder(image_features, text_tokens)
        
        # Text-pixel contrastive
        text_pixel_sim, contrastive_feats = self.contrastive(
            fused_features, text_sentence
        )
        
        # Output predictions
        outputs = self.output_heads(fused_features)
        
        # Add contrastive outputs
        outputs['text_pixel_sim'] = text_pixel_sim
        outputs['contrastive_features'] = contrastive_feats
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_cris_model(
    num_classes: int = 6,
    freeze_text_encoder: bool = True,
    freeze_image_encoder: bool = False,
    **kwargs
) -> CRISNucleiSegmenter:
    """
    Create CRIS model for nuclei segmentation.
    
    Args:
        num_classes: Number of nucleus classes (default: 6)
        freeze_text_encoder: Whether to freeze text encoder
        freeze_image_encoder: Whether to freeze image encoder
        
    Returns:
        CRISNucleiSegmenter model
    """
    model = CRISNucleiSegmenter(
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=512,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder,
        freeze_image_encoder=freeze_image_encoder
    )
    
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CRIS Nuclei Segmenter Test")
    print("=" * 70)
    
    # Create model
    model = create_cris_model(num_classes=6)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nCRIS Nuclei Segmenter Configuration")
    print(f"  - Text Encoder: BioClinicalBERT")
    print(f"  - Image Encoder: ViT-B/16")
    print(f"  - Embed Dimension: 512")
    print(f"  - Vision-Language Decoder: 3 layers")
    print(f"  - Text-Pixel Contrastive: Enabled")
    print(f"  - Output: HoVer-Net style (NP + HV + Type)")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    texts = [
        "Segment neoplastic nuclei in breast tissue.",
        "Identify inflammatory cells in colon sample."
    ]
    
    with torch.no_grad():
        outputs = model(images, texts)
    
    print(f"\nOutputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Verify output shapes
    assert outputs['np'].shape == (batch_size, 2, 256, 256)
    assert outputs['hv'].shape == (batch_size, 2, 256, 256)
    assert outputs['type'].shape == (batch_size, 6, 256, 256)
    
    print(f"\n✅ CRIS Nuclei Segmenter test passed!")
