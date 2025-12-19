"""
Vision-Language Fusion Module
Combines visual and textual features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between vision and language features
    """
    
    def __init__(
        self,
        vis_dim=2048,
        text_dim=768,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project visual and text features to same dimension
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, vis_features, text_features, text_mask=None):
        """
        Args:
            vis_features: [B, H*W, vis_dim] - flattened spatial features
            text_features: [B, L, text_dim] - text token features
            text_mask: [B, L] - attention mask (1 = valid, 0 = padding)
        
        Returns:
            fused_features: [B, H*W, hidden_dim]
        """
        # Project to same dimension
        vis = self.vis_proj(vis_features)  # [B, H*W, hidden_dim]
        text = self.text_proj(text_features)  # [B, L, hidden_dim]
        
        # Cross attention: vision queries, text keys/values
        # key_padding_mask: True = ignore
        if text_mask is not None:
            key_padding_mask = (text_mask == 0)
        else:
            key_padding_mask = None
        
        attn_out, _ = self.cross_attn(
            query=vis,
            key=text,
            value=text,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + norm
        vis = self.norm1(vis + attn_out)
        
        # FFN
        ffn_out = self.ffn(vis)
        vis = self.norm2(vis + ffn_out)
        
        return vis


class SimpleFusion(nn.Module):
    """
    Simple fusion by modulating visual features with text embedding
    """
    
    def __init__(
        self,
        vis_dim=2048,
        text_dim=768,
        hidden_dim=512
    ):
        super().__init__()
        
        # Project text to modulation parameters (scale and shift)
        self.text_to_gamma = nn.Linear(text_dim, vis_dim)
        self.text_to_beta = nn.Linear(text_dim, vis_dim)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(vis_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, vis_features, text_pooled):
        """
        Args:
            vis_features: [B, C, H, W] - visual features
            text_pooled: [B, D] - pooled text embedding
        
        Returns:
            fused_features: [B, hidden_dim, H, W]
        """
        # Generate modulation parameters
        gamma = self.text_to_gamma(text_pooled).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.text_to_beta(text_pooled).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Modulate: FiLM (Feature-wise Linear Modulation)
        vis_modulated = gamma * vis_features + beta
        
        # Project to output dimension
        out = self.out_proj(vis_modulated)
        
        return out


class MultiScaleFusion(nn.Module):
    """
    Fuse multi-scale visual features with text
    """
    
    def __init__(
        self,
        vis_dims=[256, 512, 1024, 2048],
        text_dim=768,
        hidden_dim=512
    ):
        super().__init__()
        
        self.num_levels = len(vis_dims)
        
        # Fusion module for each scale
        self.fusions = nn.ModuleList([
            SimpleFusion(vis_dim=vis_dims[i], text_dim=text_dim, hidden_dim=hidden_dim)
            for i in range(self.num_levels)
        ])
    
    def forward(self, vis_features_list, text_pooled):
        """
        Args:
            vis_features_list: List of [B, C_i, H_i, W_i] at different scales
            text_pooled: [B, D] - pooled text embedding
        
        Returns:
            fused_list: List of [B, hidden_dim, H_i, W_i] at different scales
        """
        fused_list = []
        for i, vis_feat in enumerate(vis_features_list):
            fused = self.fusions[i](vis_feat, text_pooled)
            fused_list.append(fused)
        
        return fused_list


if __name__ == "__main__":
    # Test cross attention fusion
    print("Testing CrossAttentionFusion...")
    fusion = CrossAttentionFusion(vis_dim=2048, text_dim=768, hidden_dim=512)
    
    vis = torch.randn(2, 196, 2048)  # [B, H*W, vis_dim]
    text = torch.randn(2, 20, 768)  # [B, L, text_dim]
    text_mask = torch.ones(2, 20)  # [B, L]
    
    fused = fusion(vis, text, text_mask)
    print(f"Fused features: {fused.shape}")
    
    # Test simple fusion
    print("\nTesting SimpleFusion...")
    fusion = SimpleFusion(vis_dim=2048, text_dim=768, hidden_dim=512)
    
    vis = torch.randn(2, 2048, 14, 14)  # [B, C, H, W]
    text = torch.randn(2, 768)  # [B, D]
    
    fused = fusion(vis, text)
    print(f"Fused features: {fused.shape}")
