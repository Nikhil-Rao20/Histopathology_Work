"""
DINOv2 Encoder for CIPS-Net V2
==============================

Provides hierarchical image encoders using DINOv2 pretrained backbones.
DINOv2 offers self-supervised ViT models trained on 142M images with
excellent dense prediction capabilities.

Key Features:
    - ViT-B/14 (Base): 86M params, patch size 14, embed dim 768
    - ViT-L/14 (Large): 304M params, patch size 14, embed dim 1024
    - ViT-g/14 (Giant): 1.1B params, patch size 14, embed dim 1536 (optional)
    
Note: Patch size 14 provides higher spatial resolution than ViT-B/16:
    - 256×256 image → 18×18 patches (vs 16×16 for patch16)
    - Better for dense prediction tasks like instance segmentation

Supports:
    1. Frozen backbone (default) - use DINOv2 features directly
    2. Full finetuning - adapt to histopathology domain
    3. Supervised pretraining - pretrain on all PanNuke images
    
Reference:
    - Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
    - https://github.com/facebookresearch/dinov2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from functools import partial


# ============================================================
# DINOv2 Model Registry
# ============================================================

DINOV2_CONFIGS = {
    'dinov2_vit_s_14': {
        'hub_name': 'dinov2_vits14',
        'embed_dim': 384,
        'patch_size': 14,
        'num_heads': 6,
        'num_layers': 12,
    },
    'dinov2_vit_b_14': {
        'hub_name': 'dinov2_vitb14',
        'embed_dim': 768,
        'patch_size': 14,
        'num_heads': 12,
        'num_layers': 12,
    },
    'dinov2_vit_l_14': {
        'hub_name': 'dinov2_vitl14',
        'embed_dim': 1024,
        'patch_size': 14,
        'num_heads': 16,
        'num_layers': 24,
    },
    'dinov2_vit_g_14': {
        'hub_name': 'dinov2_vitg14',
        'embed_dim': 1536,
        'patch_size': 14,
        'num_heads': 24,
        'num_layers': 40,
    },
}


# ============================================================
# Hierarchical DINOv2 Encoder
# ============================================================

class HierarchicalDINOv2Encoder(nn.Module):
    """
    DINOv2 encoder that extracts hierarchical (multi-scale) features.
    Compatible with U-Net style decoders expecting skip connections.
    
    DINOv2 is a ViT without native hierarchy, so we:
    1. Extract features at different transformer layers (mid, deep)
    2. Create CNN-based skip connections for low-level features
    3. Project DINOv2 features to match decoder expectations
    
    Output format matches HierarchicalViTEncoder:
        - skip1: [B, 64, H/2, W/2]  - Low-level (from CNN branch)
        - skip2: [B, 128, H/4, W/4] - Low-level (from CNN branch)
        - skip3: [B, 256, H/8, W/8] - Low-level (from CNN branch)
        - vit_mid: [B, 768, H/16, W/16] - Mid-depth ViT features
        - vit_deep: [B, 768, H/16, W/16] - Deep ViT features
        
    Args:
        model_name: DINOv2 variant ('dinov2_vit_b_14', 'dinov2_vit_l_14', etc.)
        pretrained: Load pretrained weights from torch hub
        img_size: Input image size (default 256)
        freeze_backbone: Freeze DINOv2 backbone parameters
        output_dim: Output feature dimension (768 to match existing decoder)
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vit_b_14',
        pretrained: bool = True,
        img_size: int = 256,
        freeze_backbone: bool = False,
        output_dim: int = 768,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        if model_name not in DINOV2_CONFIGS:
            raise ValueError(
                f"Unknown DINOv2 model: {model_name}. "
                f"Choose from: {list(DINOV2_CONFIGS.keys())}"
            )
        
        self.model_name = model_name
        self.config = DINOV2_CONFIGS[model_name]
        self.embed_dim = self.config['embed_dim']
        self.patch_size = self.config['patch_size']
        self.num_layers = self.config['num_layers']
        self.img_size = img_size
        self.output_dim = output_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Calculate grid dimensions
        # DINOv2 uses patch size 14, so 256×256 → 18×18 patches
        self.num_patches_h = img_size // self.patch_size
        self.num_patches_w = img_size // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Load DINOv2 from torch hub
        if pretrained:
            print(f"[DINOv2] Loading pretrained {model_name} from torch hub...")
            self.dinov2 = torch.hub.load(
                'facebookresearch/dinov2',
                self.config['hub_name'],
                pretrained=True
            )
        else:
            print(f"[DINOv2] Initializing {model_name} without pretrained weights...")
            # Load architecture only (for supervised pretraining)
            self.dinov2 = torch.hub.load(
                'facebookresearch/dinov2',
                self.config['hub_name'],
                pretrained=False
            )
        
        # Freeze backbone if specified
        if freeze_backbone:
            print(f"[DINOv2] Freezing backbone parameters")
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        # Project DINOv2 features to output_dim if different
        if self.embed_dim != output_dim:
            self.feature_proj = nn.Sequential(
                nn.Linear(self.embed_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.feature_proj = nn.Identity()
        
        # CNN branch for low-level hierarchical features (skip connections)
        # These complement the ViT features with local details
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),      # /2
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),    # /4
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),   # /8
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        
        # DINOv2 layer indices for feature extraction
        # Mid-depth: ~50% of layers, Deep: ~100% of layers
        self.mid_layer_idx = self.num_layers // 2  # 6 for B/14, 12 for L/14
        self.deep_layer_idx = self.num_layers - 1  # 11 for B/14, 23 for L/14
        
        print(f"[DINOv2] Initialized {model_name}:")
        print(f"  - Embed dim: {self.embed_dim}, Patch size: {self.patch_size}")
        print(f"  - Num layers: {self.num_layers}")
        print(f"  - Mid features from layer {self.mid_layer_idx}")
        print(f"  - Deep features from layer {self.deep_layer_idx}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Grid size: {self.num_patches_h}×{self.num_patches_w}")
        
    def _extract_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features at intermediate layers.
        
        DINOv2 forward pass with intermediate feature extraction.
        """
        B = x.shape[0]
        
        # DINOv2 requires image size to be multiple of patch_size (14)
        # Resize to nearest valid size (252 for 256 input, or 266)
        H_orig, W_orig = x.shape[2], x.shape[3]
        
        # Round up to nearest multiple of patch_size
        H_new = ((H_orig + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_new = ((W_orig + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        if H_new != H_orig or W_new != W_orig:
            x = F.interpolate(x, size=(H_new, W_new), mode='bilinear', align_corners=False)
        
        # Patch embedding
        # DINOv2 uses its own patch embed
        x = self.dinov2.prepare_tokens_with_masks(x)  # [B, 1+N, embed_dim]
        
        # Store features at different depths
        features = {}
        
        # Apply transformer blocks
        for i, blk in enumerate(self.dinov2.blocks):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            
            if i == self.mid_layer_idx:
                features['mid'] = x.clone()
            
            if i == self.deep_layer_idx:
                features['deep'] = x.clone()
        
        # Apply final layer norm
        x = self.dinov2.norm(x)
        features['final'] = x
        
        return features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract hierarchical features.
        
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            Dictionary with features at different scales:
                - skip1: [B, 64, H/2, W/2]
                - skip2: [B, 128, H/4, W/4]
                - skip3: [B, 256, H/8, W/8]
                - vit_mid: [B, output_dim, H', W']
                - vit_deep: [B, output_dim, H', W']
            where H'=W'=H/16 (to match existing decoder expectations)
        """
        B, C, H, W = x.shape
        
        # CNN-based hierarchical features (for skip connections)
        f1 = self.downsample1(x)   # [B, 64, H/2, W/2]
        f2 = self.downsample2(f1)  # [B, 128, H/4, W/4]
        f3 = self.downsample3(f2)  # [B, 256, H/8, W/8]
        
        # DINOv2 features
        vit_features = self._extract_intermediate_features(x)
        
        # Extract patch features (remove CLS token)
        mid_tokens = vit_features['mid'][:, 1:, :]   # [B, N, embed_dim]
        deep_tokens = vit_features['final'][:, 1:, :]  # [B, N, embed_dim]
        
        # Project to output dimension
        mid_tokens = self.feature_proj(mid_tokens)   # [B, N, output_dim]
        deep_tokens = self.feature_proj(deep_tokens) # [B, N, output_dim]
        
        # Reshape to spatial format
        # DINOv2 with patch14 on 256×256 (resized to 266×266) gives 19×19 patches
        # Calculate actual grid size from number of tokens
        num_tokens = mid_tokens.shape[1]
        h = w = int(num_tokens ** 0.5)  # Assuming square grid
        
        vit_mid = mid_tokens.transpose(1, 2).view(B, self.output_dim, h, w)
        vit_deep = deep_tokens.transpose(1, 2).view(B, self.output_dim, h, w)
        
        # Resize to H/16 to match existing decoder expectations (16×16)
        # This is necessary for compatibility with existing LViT decoders
        target_h, target_w = H // 16, W // 16  # 16×16 for 256×256 input
        if h != target_h or w != target_w:
            vit_mid = F.interpolate(vit_mid, size=(target_h, target_w), 
                                    mode='bilinear', align_corners=False)
            vit_deep = F.interpolate(vit_deep, size=(target_h, target_w), 
                                     mode='bilinear', align_corners=False)
        
        return {
            'skip1': f1,         # [B, 64, H/2, W/2]
            'skip2': f2,         # [B, 128, H/4, W/4]
            'skip3': f3,         # [B, 256, H/8, W/8]
            'vit_mid': vit_mid,  # [B, output_dim, H/16, W/16]
            'vit_deep': vit_deep,# [B, output_dim, H/16, W/16]
        }
    
    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get CLS token embedding for classification tasks.
        
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            cls_token: [B, output_dim]
        """
        vit_features = self._extract_intermediate_features(x)
        cls_token = vit_features['final'][:, 0, :]  # [B, embed_dim]
        cls_token = self.feature_proj(cls_token)     # [B, output_dim]
        return cls_token


# ============================================================
# DINOv2 Classification Head for Supervised Pretraining
# ============================================================

class DINOv2ClassificationHead(nn.Module):
    """
    Classification head for supervised pretraining of DINOv2.
    
    Predicts nucleus type from CLS token + pooled patch features.
    Used for pretraining on all PanNuke images before segmentation.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token: [B, embed_dim] from DINOv2 encoder
            
        Returns:
            logits: [B, num_classes]
        """
        return self.head(cls_token)


# ============================================================
# Full DINOv2 Model for Supervised Pretraining
# ============================================================

class DINOv2ForPretraining(nn.Module):
    """
    DINOv2 model with classification head for supervised pretraining.
    
    This model is used to pretrain DINOv2 on all PanNuke images
    (regardless of fold) before using it as a backbone for segmentation.
    
    Args:
        model_name: DINOv2 variant
        num_classes: Number of nucleus types (default 6)
        freeze_backbone: Whether to freeze DINOv2 (usually False for pretraining)
        img_size: Input image size
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vit_b_14',
        num_classes: int = 6,
        freeze_backbone: bool = False,
        img_size: int = 256,
    ):
        super().__init__()
        
        config = DINOV2_CONFIGS[model_name]
        
        # Load DINOv2 backbone
        self.encoder = HierarchicalDINOv2Encoder(
            model_name=model_name,
            pretrained=True,  # Start from DINOv2 pretrained weights
            img_size=img_size,
            freeze_backbone=freeze_backbone,
            output_dim=config['embed_dim'],  # Keep native dimension for pretraining
        )
        
        # Classification head
        self.classifier = DINOv2ClassificationHead(
            embed_dim=config['embed_dim'],
            num_classes=num_classes,
        )
        
        self.model_name = model_name
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: [B, 3, H, W] input images
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        cls_token = self.encoder.get_cls_token(x)
        logits = self.classifier(cls_token)
        return logits
    
    def get_encoder(self) -> HierarchicalDINOv2Encoder:
        """Return the encoder for use in segmentation model."""
        return self.encoder


# ============================================================
# Utility Functions
# ============================================================

def create_dinov2_encoder(
    model_name: str = 'dinov2_vit_b_14',
    pretrained: bool = True,
    img_size: int = 256,
    freeze_backbone: bool = False,
    pretrained_path: Optional[str] = None,
    use_gradient_checkpointing: bool = False,
) -> HierarchicalDINOv2Encoder:
    """
    Factory function to create DINOv2 encoder.
    
    Args:
        model_name: DINOv2 variant
        pretrained: Load DINOv2 pretrained weights
        img_size: Input image size
        freeze_backbone: Freeze backbone parameters
        pretrained_path: Path to supervised pretrained weights (optional)
        use_gradient_checkpointing: Enable gradient checkpointing to save memory
            (trades compute for memory — recommended for ViT-L/14 on 24GB GPUs)
        
    Returns:
        HierarchicalDINOv2Encoder instance
    """
    encoder = HierarchicalDINOv2Encoder(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    
    # Load supervised pretrained weights if provided
    if pretrained_path is not None:
        print(f"[DINOv2] Loading supervised pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'state_dict' in checkpoint:
            # Filter encoder keys
            encoder_dict = {
                k.replace('encoder.', ''): v 
                for k, v in checkpoint['state_dict'].items() 
                if k.startswith('encoder.')
            }
            encoder.load_state_dict(encoder_dict)
        else:
            encoder.load_state_dict(checkpoint)
        
        print(f"[DINOv2] Loaded supervised pretrained weights successfully")
    
    return encoder


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


# ============================================================
# Test / Sanity Check
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DINOv2 Encoder Sanity Check")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name in ['dinov2_vit_b_14', 'dinov2_vit_l_14']:
        print(f"\nTesting {model_name}...")
        
        # Create encoder
        encoder = HierarchicalDINOv2Encoder(
            model_name=model_name,
            pretrained=True,
            img_size=256,
            freeze_backbone=False,
        ).to(device)
        
        # Count parameters
        params = count_parameters(encoder)
        print(f"  Total params: {params['total'] / 1e6:.1f}M")
        print(f"  Trainable: {params['trainable'] / 1e6:.1f}M")
        
        # Test forward pass
        x = torch.randn(2, 3, 256, 256, device=device)
        with torch.no_grad():
            features = encoder(x)
        
        print(f"  Output shapes:")
        for k, v in features.items():
            print(f"    {k}: {list(v.shape)}")
        
        # Verify shapes
        assert features['skip1'].shape == (2, 64, 128, 128), "skip1 shape mismatch"
        assert features['skip2'].shape == (2, 128, 64, 64), "skip2 shape mismatch"
        assert features['skip3'].shape == (2, 256, 32, 32), "skip3 shape mismatch"
        assert features['vit_mid'].shape[1] == 768, "vit_mid channels mismatch"
        assert features['vit_deep'].shape[1] == 768, "vit_deep channels mismatch"
        
        print(f"  ✅ {model_name} passed all checks!")
        
        del encoder
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("All DINOv2 encoder tests passed! ✅")
    print("=" * 60)
