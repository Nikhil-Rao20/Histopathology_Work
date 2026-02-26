"""
Grounding DINO: Text-Grounded Detection for Nuclei Instance Segmentation
=========================================================================

Based on: "Grounding DINO: Marrying DINO with Grounded Pre-Training" (ECCV 2023)
Paper: https://arxiv.org/abs/2303.05499

Key Innovation:
    - Text-guided object queries for text-aware detection
    - Multi-scale deformable attention with text fusion
    - Language-aware query selection
    - Bi-directional vision-language fusion

Adaptation for Instance Segmentation:
    Instead of box outputs, we use query-based mask prediction
    combined with HoVer-Net style outputs for instance segmentation.

Architecture:
    1. Swin/ViT backbone for multi-scale image features
    2. Text encoder (BioClinicalBERT)
    3. Feature Enhancer with bi-directional fusion
    4. Language-Aware Query Selection
    5. Decoder with text-modulated cross-attention
    6. Instance Segmentation Heads (NP + HV + Type)

Adapted for nuclei instance segmentation on PanNuke dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============================================================
# Text Encoder (BioClinicalBERT)
# ============================================================

class GroundingTextEncoder(nn.Module):
    """
    Text encoder for grounding-based models.
    Returns both word-level and sentence-level embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 256,
        freeze: bool = True,
        max_length: int = 128
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768
        self.max_length = max_length
        
        # Project to common dimension
        self.text_proj = nn.Linear(self.hidden_size, embed_dim)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text.
        
        Returns:
            word_embeds: [B, L, embed_dim] word-level embeddings
            sentence_embed: [B, embed_dim] sentence embedding
            attention_mask: [B, L] for masking padding
        """
        device = next(self.parameters()).device
        
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        
        outputs = self.encoder(**tokens)
        hidden = outputs.last_hidden_state
        
        word_embeds = self.text_proj(hidden)
        sentence_embed = word_embeds[:, 0, :]  # CLS token
        
        return word_embeds, sentence_embed, tokens.attention_mask


# ============================================================
# Multi-Scale Image Encoder
# ============================================================

class MultiScaleViTEncoder(nn.Module):
    """
    Multi-scale feature extraction from ViT.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        pretrained: bool = True,
        img_size: int = 256
    ):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.hidden_dim = 768
        self.patch_size = 16
        self.img_size = img_size
        
        # Store target grid size for lazy pos embed interpolation
        self.target_h = self.target_w = img_size // self.patch_size
        
        self.vit.heads = nn.Identity()
        
        # Multi-scale projections
        self.proj_s8 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_s16 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_s32 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Downsample for s32
        self.downsample = nn.MaxPool2d(2)
    
    def _interpolate_pos_embed(self, device):
        """Interpolate positional embeddings (lazy, on same device)."""
        old_pos_embed = self.vit.encoder.pos_embedding.to(device)
        
        new_h, new_w = self.target_h, self.target_w
        num_patches = new_h * new_w
        
        cls_embed = old_pos_embed[:, 0:1, :]
        patch_embed = old_pos_embed[:, 1:, :]
        
        old_h = old_w = 14
        patch_embed = patch_embed.reshape(1, old_h, old_w, self.hidden_dim)
        patch_embed = patch_embed.permute(0, 3, 1, 2)
        
        patch_embed = F.interpolate(
            patch_embed, size=(new_h, new_w), mode='bicubic', align_corners=False
        )
        
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, num_patches, self.hidden_dim)
        new_pos_embed = torch.cat([cls_embed, patch_embed], dim=1)
        
        return new_pos_embed
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features.
        
        Returns:
            Dictionary with s8, s16, s32 scale features
        """
        B, C, H, W = x.shape
        h, w = H // 16, W // 16
        device = x.device
        
        # Patch embedding
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Get interpolated pos embed on correct device
        pos_embed = self._interpolate_pos_embed(device)
        x = x + pos_embed
        
        # Extract at different layers
        features = {}
        for i, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            if i == 5:  # Layer 6 - mid level
                feat = x[:, 1:, :].transpose(1, 2).view(B, self.hidden_dim, h, w)
                # Upsample to s8
                feat_s8 = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
                features['s8'] = self.proj_s8(feat_s8)
            elif i == 8:  # Layer 9
                feat = x[:, 1:, :].transpose(1, 2).view(B, self.hidden_dim, h, w)
                features['s16'] = self.proj_s16(feat)
            elif i == 11:  # Layer 12 - last
                feat = x[:, 1:, :].transpose(1, 2).view(B, self.hidden_dim, h, w)
                feat_s32 = self.downsample(feat)
                features['s32'] = self.proj_s32(feat_s32)
        
        return features


# ============================================================
# Bi-Directional Vision-Language Fusion
# ============================================================

class BiDirectionalFusion(nn.Module):
    """
    Bi-directional fusion between vision and language features.
    Vision enhances language, and language enhances vision.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Vision to Language
        self.v2l_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.v2l_norm = nn.LayerNorm(embed_dim)
        
        # Language to Vision
        self.l2v_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.l2v_norm = nn.LayerNorm(embed_dim)
        
        # FFNs
        self.v_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.v_norm = nn.LayerNorm(embed_dim)
        
        self.l_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.l_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        vision: torch.Tensor,
        language: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bi-directional fusion.
        
        Args:
            vision: [B, N_v, C] vision features
            language: [B, N_l, C] language features
            lang_mask: [B, N_l] attention mask for language
            
        Returns:
            enhanced_vision: [B, N_v, C]
            enhanced_language: [B, N_l, C]
        """
        # Create key padding mask for attention
        key_padding_mask = None
        if lang_mask is not None:
            key_padding_mask = (lang_mask == 0)  # True for padding
        
        # Vision to Language: language queries attend to vision
        l_enhanced, _ = self.v2l_attn(
            self.v2l_norm(language), 
            vision, 
            vision
        )
        language = language + l_enhanced
        language = language + self.l_ffn(self.l_norm(language))
        
        # Language to Vision: vision queries attend to language
        v_enhanced, _ = self.l2v_attn(
            self.l2v_norm(vision), 
            language, 
            language,
            key_padding_mask=key_padding_mask
        )
        vision = vision + v_enhanced
        vision = vision + self.v_ffn(self.v_norm(vision))
        
        return vision, language


class FeatureEnhancer(nn.Module):
    """
    Feature enhancer with multiple bi-directional fusion layers.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BiDirectionalFusion(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        vision: torch.Tensor,
        language: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multiple bi-directional fusion layers."""
        for layer in self.layers:
            vision, language = layer(vision, language, lang_mask)
        
        return vision, language


# ============================================================
# Language-Aware Query Selection
# ============================================================

class LanguageAwareQueries(nn.Module):
    """
    Generates object queries conditioned on text.
    These queries will attend to image features to predict instances.
    """
    
    def __init__(
        self,
        num_queries: int = 100,
        embed_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # Cross-attention with text
        self.text_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Query refinement
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        text_embeds: torch.Tensor,
        text_sentence: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate text-conditioned queries.
        
        Args:
            text_embeds: [B, L, C] text token embeddings
            text_sentence: [B, C] sentence embedding
            
        Returns:
            queries: [B, num_queries, C] text-conditioned queries
        """
        B = text_embeds.shape[0]
        
        # Expand learnable queries
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        
        # Attend to text
        queries_attended, _ = self.text_attn(
            self.norm(queries),
            text_embeds,
            text_embeds
        )
        
        # Add sentence-level context
        sentence_expanded = text_sentence.unsqueeze(1)  # [B, 1, C]
        queries = queries + queries_attended + sentence_expanded
        
        # Refine
        queries = queries + self.refine(queries)
        
        return queries


# ============================================================
# Grounding Decoder
# ============================================================

class GroundingDecoderLayer(nn.Module):
    """
    Decoder layer with self-attention, cross-attention to image, and FFN.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention to image features
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, C] object queries
            memory: [B, N, C] image memory
            
        Returns:
            refined queries: [B, Q, C]
        """
        # Self-attention
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        
        # Cross-attention
        queries = queries + self.cross_attn(self.norm2(queries), memory, memory)[0]
        
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        
        return queries


class GroundingDecoder(nn.Module):
    """
    Full grounding decoder with multiple layers.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            GroundingDecoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """Apply decoder layers."""
        for layer in self.layers:
            queries = layer(queries, memory)
        
        return queries


# ============================================================
# Instance Segmentation Head
# ============================================================

class GroundingSegmentationHead(nn.Module):
    """
    Converts query features to instance segmentation outputs.
    Produces dense NP/HV/Type predictions rather than per-query masks.
    s16 (H/16) -> full resolution (H): 4 upsampling steps (2^4 = 16x)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 6
    ):
        super().__init__()
        
        # Query aggregation to dense features
        self.query_to_dense = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Upsample path: s16 -> s1 (4 upsampling steps)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1),  # s16 -> s8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # s8 -> s4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # s4 -> s2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # s2 -> s1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.np_head = nn.Conv2d(32, 2, 1)
        self.hv_head = nn.Conv2d(32, 2, 1)
        self.type_head = nn.Conv2d(32, num_classes, 1)
    
    def forward(
        self,
        queries: torch.Tensor,
        multi_scale_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate dense segmentation from queries and features.
        
        Args:
            queries: [B, Q, C] refined object queries
            multi_scale_features: Dict with s8, s16, s32 features
            
        Returns:
            Dictionary with np, hv, type predictions
        """
        B = queries.shape[0]
        
        # Aggregate queries to dense feature
        query_feat = self.query_to_dense(queries)  # [B, Q, C]
        
        # Dot product attention with s16 features
        s16 = multi_scale_features['s16']  # [B, C, H, W]
        _, C, H, W = s16.shape
        
        s16_flat = s16.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # Attention: each spatial location attends to queries
        attn = torch.bmm(s16_flat, query_feat.transpose(1, 2))  # [B, HW, Q]
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate query information
        aggregated = torch.bmm(attn, query_feat)  # [B, HW, C]
        aggregated = aggregated.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        # Combine with original features
        x = s16 + aggregated
        
        # Upsample to full resolution (s16 -> s1 in 4 steps)
        x = self.up1(x)  # [B, 256, 2H, 2W] = s8
        
        # Add s8 features if available
        s8 = multi_scale_features['s8']
        if s8.shape[-2:] != x.shape[-2:]:
            s8 = F.interpolate(s8, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Match channels and add
        x = x + F.pad(s8, (0, 0, 0, 0, 0, 256 - s8.shape[1]))[:, :256]
        
        x = self.up2(x)  # [B, 128, 4H, 4W] = s4
        x = self.up3(x)  # [B, 64, 8H, 8W] = s2
        x = self.up4(x)  # [B, 32, 16H, 16W] = s1 (full resolution)
        
        return {
            'np': self.np_head(x),
            'hv': self.hv_head(x),
            'type': self.type_head(x)
        }


# ============================================================
# Main Grounding DINO Model for Nuclei Segmentation
# ============================================================

class GroundingDINONucleiSegmenter(nn.Module):
    """
    Grounding DINO adapted for Nuclei Instance Segmentation.
    
    Key features:
        - Text-grounded object queries
        - Bi-directional vision-language fusion
        - Multi-scale feature extraction
        - HoVer-Net style instance segmentation heads
    """
    
    def __init__(
        self,
        text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 6,
        freeze_text_encoder: bool = True,
        img_size: int = 256
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Text encoder
        self.text_encoder = GroundingTextEncoder(
            model_name=text_encoder,
            embed_dim=embed_dim,
            freeze=freeze_text_encoder
        )
        
        # Image encoder
        self.image_encoder = MultiScaleViTEncoder(
            embed_dim=embed_dim,
            pretrained=True,
            img_size=img_size
        )
        
        # Feature enhancer (bi-directional fusion)
        self.feature_enhancer = FeatureEnhancer(
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=3
        )
        
        # Language-aware queries
        self.query_generator = LanguageAwareQueries(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=8
        )
        
        # Grounding decoder
        self.decoder = GroundingDecoder(
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=6
        )
        
        # Segmentation head
        self.seg_head = GroundingSegmentationHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
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
            Dictionary with np, hv, type predictions
        """
        # Encode text
        text_tokens, text_sentence, text_mask = self.text_encoder(texts)
        
        # Encode image (multi-scale)
        image_features = self.image_encoder(images)
        
        # Flatten multi-scale features for enhancer
        B = images.shape[0]
        s8 = image_features['s8'].flatten(2).transpose(1, 2)   # [B, N8, C]
        s16 = image_features['s16'].flatten(2).transpose(1, 2)  # [B, N16, C]
        s32 = image_features['s32'].flatten(2).transpose(1, 2)  # [B, N32, C]
        
        # Concatenate for feature enhancer
        vision_flat = torch.cat([s8, s16, s32], dim=1)  # [B, N_total, C]
        
        # Bi-directional enhancement
        vision_enhanced, text_enhanced = self.feature_enhancer(
            vision_flat, text_tokens, text_mask
        )
        
        # Generate language-aware queries
        queries = self.query_generator(text_enhanced, text_sentence)
        
        # Decode
        queries_refined = self.decoder(queries, vision_enhanced)
        
        # Generate segmentation outputs
        outputs = self.seg_head(queries_refined, image_features)
        
        return outputs


# ============================================================
# Factory Function
# ============================================================

def create_grounding_dino_model(
    num_classes: int = 6,
    num_queries: int = 100,
    freeze_text_encoder: bool = True,
    img_size: int = 256,
    **kwargs
) -> GroundingDINONucleiSegmenter:
    """
    Create Grounding DINO model for nuclei segmentation.
    
    Args:
        num_classes: Number of nucleus classes (default: 6)
        num_queries: Number of object queries
        freeze_text_encoder: Whether to freeze text encoder
        img_size: Input image size
        
    Returns:
        GroundingDINONucleiSegmenter model
    """
    model = GroundingDINONucleiSegmenter(
        text_encoder="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=256,
        num_queries=num_queries,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder,
        img_size=img_size
    )
    
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Grounding DINO Nuclei Segmenter Test")
    print("=" * 70)
    
    # Create model
    model = create_grounding_dino_model(num_classes=6, num_queries=100)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nGrounding DINO Nuclei Segmenter Configuration")
    print(f"  - Text Encoder: BioClinicalBERT")
    print(f"  - Image Encoder: Multi-Scale ViT-B/16")
    print(f"  - Bi-Directional Fusion: 3 layers")
    print(f"  - Object Queries: 100")
    print(f"  - Decoder Layers: 6")
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
    
    print(f"\n✅ Grounding DINO Nuclei Segmenter test passed!")
