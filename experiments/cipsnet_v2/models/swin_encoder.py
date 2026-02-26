"""
Swin Transformer Encoder for CIPS-Net V2
=========================================

Provides hierarchical image encoders using Swin Transformer backbones via timm.
Swin is *natively* hierarchical (multi-scale), making it a natural fit for the
LViT-IE U-Net decoder — no workarounds needed unlike plain ViT.

Swin feature pyramid (256×256 input, patch4):
    stage0: [B, 64,  64, 128]  → stride 4   (H/4)
    stage1: [B, 32,  32, 256]  → stride 8   (H/8)
    stage2: [B, 16,  16, 512]  → stride 16  (H/16)
    stage3: [B,  8,   8, 1024] → stride 32  (H/32)

Output format (matches HierarchicalViTEncoder / DINOv2Encoder):
    skip1    : [B,  64, H/2,  W/2 ] = [B,  64, 128, 128]  — lightweight CNN stem
    skip2    : [B, 128, H/4,  W/4 ] = [B, 128,  64,  64]  — Swin stage0
    skip3    : [B, 256, H/8,  W/8 ] = [B, 256,  32,  32]  — Swin stage1
    vit_mid  : [B, 768, H/16, W/16] = [B, 768,  16,  16]  — Swin stage2
    vit_deep : [B, 768, H/16, W/16] = [B, 768,  16,  16]  — Swin stage3 (upsampled)

Supported variants:
    'swin_b'  → swinv2_base_window8_256   (88M, ImageNet-1k / 22k pretrained)
    'swin_l'  → swinv2_large_window12to16_192to256 (197M, ImageNet-22k pretrained)

Reference:
    - Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows", ICCV 2021
    - Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution", CVPR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# ============================================================
# Swin Model Registry
# ============================================================

SWIN_CONFIGS = {
    'swin_b': {
        'timm_name': 'swinv2_base_window8_256',
        'stage_channels': [128, 256, 512, 1024],
        'pretrained_tag': 'ms_in1k',   # ImageNet-1k pretrained
    },
    'swin_l': {
        'timm_name': 'swinv2_large_window12to16_192to256',
        'stage_channels': [192, 384, 768, 1536],
        'pretrained_tag': 'ms_in22k',  # ImageNet-22k pretrained
    },
}

OUTPUT_DIM = 768   # Match ViT/DINOv2 decoder dimension


# ============================================================
# Hierarchical Swin Encoder
# ============================================================

class HierarchicalSwinEncoder(nn.Module):
    """
    Swin Transformer backbone with U-Net-compatible skip connections.

    Since Swin starts at stride-4 (no stride-2 feature map), a lightweight
    2-layer CNN stem produces the skip1 [H/2] skip connection.

    Args:
        model_name : 'swin_b' or 'swin_l'
        pretrained : Load ImageNet pretrained weights via timm
        img_size   : Input spatial size (default 256)
        freeze_backbone : Freeze Swin backbone parameters
        use_gradient_checkpointing : Enable gradient checkpointing on Swin stages
    """

    def __init__(
        self,
        model_name: str = 'swin_b',
        pretrained: bool = True,
        img_size: int = 256,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        assert model_name in SWIN_CONFIGS, \
            f"Unknown Swin variant '{model_name}'. Choose from {list(SWIN_CONFIGS.keys())}"

        cfg = SWIN_CONFIGS[model_name]
        self.model_name = model_name
        self.img_size = img_size
        self.stage_channels = cfg['stage_channels']  # [C0, C1, C2, C3]
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ── Swin backbone (features_only mode → returns 4 stage feature maps) ──
        import timm
        self.backbone = timm.create_model(
            cfg['timm_name'],
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        if pretrained:
            print(f"[Swin] Loaded pretrained {cfg['timm_name']}")

        # Optional: activate gradient checkpointing in Swin stages
        if use_gradient_checkpointing:
            self._enable_swin_grad_checkpointing()

        # Freeze backbone if requested
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print(f"[Swin] Backbone frozen")

        # ── CNN stem for skip1 (H/2) ──
        # Swin starts at stride-4; we need a stride-2 feature for skip1.
        # A lightweight 2-conv stem provides early texture/edge features.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )  # → [B, 64, H/2, W/2]

        C0, C1, C2, C3 = self.stage_channels

        # ── Projection layers: Swin channels → decoder-expected channels ──
        # skip2: stage0 (C0) → 128
        self.proj_skip2 = nn.Sequential(
            nn.Conv2d(C0, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # skip3: stage1 (C1) → 256
        self.proj_skip3 = nn.Sequential(
            nn.Conv2d(C1, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # vit_mid: stage2 (C2) → OUTPUT_DIM
        self.proj_mid = nn.Sequential(
            nn.Conv2d(C2, OUTPUT_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(OUTPUT_DIM),
            nn.GELU(),
        )

        # vit_deep: stage3 (C3) → OUTPUT_DIM  (also upsample H/32 → H/16)
        self.proj_deep = nn.Sequential(
            nn.Conv2d(C3, OUTPUT_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(OUTPUT_DIM),
            nn.GELU(),
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gc_tag = " [gradient checkpointing ON]" if use_gradient_checkpointing else ""
        print(f"[Swin] Initialized {model_name}{gc_tag}:")
        print(f"  - Stage channels: {self.stage_channels}")
        print(f"  - Total params: {total/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M")

    def _enable_swin_grad_checkpointing(self):
        """Enable gradient checkpointing on each Swin stage's transformer blocks."""
        try:
            # timm SwinV2 exposes layers as backbone.layers
            for layer in self.backbone.layers:
                for block in layer.blocks:
                    block.grad_checkpointing = True
            print("[Swin] Gradient checkpointing enabled on transformer blocks")
        except AttributeError:
            print("[Swin] Warning: could not enable gradient checkpointing (API mismatch)")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            Dict with keys: skip1, skip2, skip3, vit_mid, vit_deep
        """
        # ── CNN stem → skip1 [B, 64, H/2, W/2] ──
        skip1 = self.stem(x)

        # ── Swin backbone → 4 stage feature maps (NHWC tensors) ──
        stage_feats = self.backbone(x)
        # stage_feats[i]: [B, H_i, W_i, C_i]  (NHWC from timm Swin)

        # Convert NHWC → NCHW for all stages
        s0 = stage_feats[0].permute(0, 3, 1, 2).contiguous()  # [B, C0, H/4,  W/4 ]
        s1 = stage_feats[1].permute(0, 3, 1, 2).contiguous()  # [B, C1, H/8,  W/8 ]
        s2 = stage_feats[2].permute(0, 3, 1, 2).contiguous()  # [B, C2, H/16, W/16]
        s3 = stage_feats[3].permute(0, 3, 1, 2).contiguous()  # [B, C3, H/32, W/32]

        # ── Project to target channel dims ──
        skip2   = self.proj_skip2(s0)  # [B, 128, H/4,  W/4 ]
        skip3   = self.proj_skip3(s1)  # [B, 256, H/8,  W/8 ]
        vit_mid = self.proj_mid(s2)    # [B, 768, H/16, W/16]

        # stage3 is at H/32; upsample 2× to H/16 to match decoder bottleneck
        s3_up   = F.interpolate(s3, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        vit_deep = self.proj_deep(s3_up)  # [B, 768, H/16, W/16]

        return {
            'skip1'   : skip1,    # [B,  64, H/2,  W/2 ]
            'skip2'   : skip2,    # [B, 128, H/4,  W/4 ]
            'skip3'   : skip3,    # [B, 256, H/8,  W/8 ]
            'vit_mid' : vit_mid,  # [B, 768, H/16, W/16]
            'vit_deep': vit_deep, # [B, 768, H/16, W/16]
        }


# ============================================================
# Factory function
# ============================================================

def create_swin_encoder(
    model_name: str = 'swin_b',
    pretrained: bool = True,
    img_size: int = 256,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False,
) -> HierarchicalSwinEncoder:
    """
    Create a HierarchicalSwinEncoder.

    Args:
        model_name : 'swin_b' or 'swin_l'
        pretrained : Load ImageNet pretrained weights
        img_size   : Input spatial size
        freeze_backbone : Freeze Swin parameters
        use_gradient_checkpointing : Save activation memory (recommended for swin_l)

    Returns:
        HierarchicalSwinEncoder
    """
    return HierarchicalSwinEncoder(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
