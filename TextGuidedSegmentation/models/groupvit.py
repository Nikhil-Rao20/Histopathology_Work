"""
GroupViT: Semantic Segmentation Emerges from Text Supervision
Paper: https://arxiv.org/abs/2202.11094 (CVPR 2022)
Official Implementation: https://github.com/NVlabs/GroupViT

Architecture:
- Hierarchical grouping mechanism that learns to group patches
- Contrastive learning with CLIP-style text encoder
- Group tokens that attend to patch tokens and merge them
- Zero-shot transfer to semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import clip
import math
from functools import partial

from ..utils.base_model import TextGuidedSegmentationBase, register_model


class Mlp(nn.Module):
    """MLP module for transformer."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttnBlock(nn.Module):
    """Cross-attention block for group-patch interaction."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
    
    def forward(
        self,
        query: torch.Tensor,  # Group tokens
        key_value: torch.Tensor,  # Patch tokens
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = query.shape
        M = key_value.shape[1]
        
        q = self.q(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(key_value).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn:
            return x, attn.mean(dim=1)  # Average over heads
        return x, None


class AssignAttention(nn.Module):
    """
    Assign attention for grouping patches into groups.
    Key component of GroupViT.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        hard: bool = True,
        gumbel: bool = False,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
    
    def forward(
        self,
        query: torch.Tensor,  # Group tokens (B, G, D)
        key: torch.Tensor,    # Patch tokens (B, N, D)
        value: torch.Tensor = None,  # Optional, uses key if None
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute assignment attention.
        
        Returns:
            Updated group tokens and optional attention map
        """
        if value is None:
            value = key
        
        B, G, C = query.shape
        N = key.shape[1]
        
        q = self.q(query).reshape(B, G, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention: (B, heads, G, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if self.gumbel and self.training:
            attn = F.gumbel_softmax(attn, tau=self.gumbel_tau, hard=self.hard, dim=-1)
        else:
            attn = attn.softmax(dim=-1)
            if self.hard:
                # Hard assignment (argmax)
                index = attn.argmax(dim=-1, keepdim=True)
                hard_attn = torch.zeros_like(attn).scatter_(-1, index, 1.0)
                attn = hard_attn - attn.detach() + attn  # Straight-through
        
        out = (attn @ v).transpose(1, 2).reshape(B, G, C)
        
        if return_attn:
            # Return attention map: (B, G, N)
            return out, attn.mean(dim=1)
        return out, None


class GroupingBlock(nn.Module):
    """
    GroupingBlock for merging patches into groups.
    Core building block of GroupViT.
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        num_group_tokens: int,
        num_output_groups: int,
        hard: bool = True,
        gumbel: bool = False,
    ):
        super().__init__()
        
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        
        # Group tokens (learnable)
        self.group_tokens = nn.Parameter(torch.randn(1, num_group_tokens, dim))
        
        # Cross attention: group tokens attend to patch tokens
        self.cross_attn = CrossAttnBlock(dim, num_heads)
        self.cross_norm = nn.LayerNorm(dim)
        
        # Assignment attention: assign patches to groups
        self.assign_attn = AssignAttention(dim, num_heads, hard=hard, gumbel=gumbel)
        
        # MLP for group token refinement
        self.mlp = Mlp(dim, dim * 4, out_dim)
        self.mlp_norm = nn.LayerNorm(dim)
        
        # Project to output dimension if different
        if dim != out_dim:
            self.proj = nn.Linear(dim, out_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,  # Patch tokens (B, N, D)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input patch tokens
            
        Returns:
            Updated group tokens and optional attention map
        """
        B = x.shape[0]
        
        # Expand group tokens for batch
        group_tokens = self.group_tokens.expand(B, -1, -1)
        
        # Cross attention: groups attend to patches
        group_tokens = group_tokens + self.cross_attn(
            self.cross_norm(group_tokens), x
        )[0]
        
        # Assignment: patches assigned to groups
        new_groups, attn = self.assign_attn(
            group_tokens, x, return_attn=return_attn
        )
        
        # MLP refinement
        new_groups = self.proj(new_groups)
        new_groups = new_groups + self.mlp(self.mlp_norm(new_groups))
        
        return new_groups, attn


class GroupViTEncoder(nn.Module):
    """
    GroupViT encoder with hierarchical grouping.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_group_tokens: List[int] = [64, 8],
        num_output_groups: List[int] = [64, 8],
        hard_assignment: bool = True,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # Transformer blocks (split by grouping stages)
        self.blocks_stage1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth // 2)
        ])
        
        # First grouping
        self.grouping1 = GroupingBlock(
            embed_dim, embed_dim, num_heads,
            num_group_tokens[0], num_output_groups[0],
            hard=hard_assignment
        )
        
        self.blocks_stage2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth // 4)
        ])
        
        # Second grouping
        self.grouping2 = GroupingBlock(
            embed_dim, embed_dim, num_heads,
            num_group_tokens[1], num_output_groups[1],
            hard=hard_assignment
        )
        
        self.blocks_stage3 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth // 4)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Group tokens and list of attention maps
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed
        
        attn_maps = []
        
        # Stage 1
        for block in self.blocks_stage1:
            x = block(x)
        
        # First grouping
        x, attn1 = self.grouping1(x, return_attn=return_attn)
        if attn1 is not None:
            attn_maps.append(attn1)
        
        # Stage 2
        for block in self.blocks_stage2:
            x = block(x)
        
        # Second grouping
        x, attn2 = self.grouping2(x, return_attn=return_attn)
        if attn2 is not None:
            attn_maps.append(attn2)
        
        # Stage 3
        for block in self.blocks_stage3:
            x = block(x)
        
        x = self.norm(x)
        
        return x, attn_maps


class TransformerBlock(nn.Module):
    """Standard transformer block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


@register_model("groupvit")
class GroupViT(TextGuidedSegmentationBase):
    """
    GroupViT: Semantic Segmentation Emerges from Text Supervision
    
    Key features:
    - Hierarchical grouping of patches into semantic groups
    - CLIP-style contrastive learning
    - Zero-shot transfer via group-text similarity
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 256,
        clip_model: str = "ViT-B/16",
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_group_tokens: List[int] = [64, 8],
        num_output_groups: List[int] = [64, 8],
        freeze_clip: bool = True,
        hard_assignment: bool = True,
        device: str = "cuda",
    ):
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            clip_model=clip_model,
            freeze_clip=freeze_clip,
            device=device,
        )
        
        self.embed_dim = embed_dim
        self.num_output_groups = num_output_groups[-1]  # Final number of groups
        
        # GroupViT encoder
        self.encoder = GroupViTEncoder(
            img_size=image_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_group_tokens=num_group_tokens,
            num_output_groups=num_output_groups,
            hard_assignment=hard_assignment,
        )
        
        # Project to CLIP dimension for compatibility
        self.visual_proj = nn.Linear(embed_dim, self.clip_embed_dim)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)  # ln(100)
        
        self.to(device)
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode image to group tokens.
        
        Returns:
            Group tokens and attention maps
        """
        group_tokens, attn_maps = self.encoder(image, return_attn=True)
        
        # Project to CLIP dimension
        group_tokens = self.visual_proj(group_tokens)
        
        return group_tokens, attn_maps
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encode text using CLIP."""
        return self.encode_text_clip(text_prompts)
    
    def compute_seg_from_attn(
        self,
        group_tokens: torch.Tensor,  # (B, G, D)
        text_features: torch.Tensor,  # (C, D)
        attn_maps: List[torch.Tensor],
        orig_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Compute segmentation from group-text similarity and attention maps.
        
        Args:
            group_tokens: Final group token embeddings
            text_features: Text embeddings
            attn_maps: List of attention maps from grouping stages
            orig_size: Original image size (H, W)
            
        Returns:
            Segmentation logits (B, C, H, W)
        """
        B, G, D = group_tokens.shape
        C = text_features.shape[0]
        
        # Normalize
        group_tokens = F.normalize(group_tokens, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Group-text similarity: (B, G, C)
        similarity = torch.einsum('bgd,cd->bgc', group_tokens, text_features)
        similarity = similarity * self.logit_scale.exp()
        
        # Apply attention maps to get spatial segmentation
        # Chain attention maps: patches -> groups (stage 1) -> groups (stage 2)
        if len(attn_maps) == 2:
            # attn_maps[0]: (B, G1, N) - patches to first groups
            # attn_maps[1]: (B, G2, G1) - first groups to final groups
            
            # Compose: (B, G2, G1) @ (B, G1, N) -> (B, G2, N)
            combined_attn = torch.bmm(attn_maps[1], attn_maps[0])  # (B, G, N)
        elif len(attn_maps) == 1:
            combined_attn = attn_maps[0]
        else:
            # No attention maps, use uniform
            N = (self.image_size // 16) ** 2
            combined_attn = torch.ones(B, G, N, device=group_tokens.device) / N
        
        # Map similarity through attention: (B, C, N)
        # similarity: (B, G, C), combined_attn: (B, G, N)
        seg_logits = torch.einsum('bgc,bgn->bcn', similarity.softmax(dim=-1), combined_attn)
        
        # Reshape to spatial
        H = W = int(math.sqrt(seg_logits.shape[-1]))
        seg_logits = seg_logits.reshape(B, C, H, W)
        
        # Upsample
        seg_logits = F.interpolate(seg_logits, size=orig_size, mode='bilinear', align_corners=False)
        
        return seg_logits
    
    def forward(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        B, _, H, W = image.shape
        
        # Encode image to groups
        group_tokens, attn_maps = self.encode_image(image)
        
        # Encode text
        text_features = self.encode_text(text_prompts)
        
        # Compute segmentation
        logits = self.compute_seg_from_attn(
            group_tokens, text_features, attn_maps, (H, W)
        )
        
        # Global image-text similarity (for contrastive loss)
        image_features = group_tokens.mean(dim=1)  # (B, D)
        image_features = F.normalize(image_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        
        return {
            'logits': logits,
            'pred_mask': logits.argmax(dim=1),
            'group_tokens': group_tokens,
            'image_features': image_features,
            'text_features': text_features,
            'attn_maps': attn_maps,
            'logit_scale': self.logit_scale,
        }
