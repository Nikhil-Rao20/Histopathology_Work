"""
CIPS-Net: Compositional Instruction-conditioned Pathology Segmentation Network

Main model that integrates all components:
1. Image Encoder (ViT)
2. Text Encoder (Clinical-BERT)
3. Instruction Grounding (Compositional Graph Reasoning)
4. Segmentation Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .instruction_grounding import CompositionalGraphReasoning
from .decoder import SegmentationDecoder


class CIPSNet(nn.Module):
    """
    CIPS-Net: Compositional Instruction-conditioned Pathology Segmentation Network
    
    Args:
        img_encoder_name: Vision transformer model name ('vit_b_16' or 'vit_l_16')
        text_encoder_name: Text encoder model name (Clinical-BERT variants)
        embed_dim: Shared embedding dimension
        num_classes: Number of pathology classes
        img_size: Input image size
        num_graph_layers: Number of graph reasoning layers
        decoder_channels: Channel dimensions for decoder
        freeze_text_encoder: Whether to freeze text encoder weights
    """
    
    def __init__(
        self,
        img_encoder_name: str = 'vit_b_16',
        text_encoder_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 768,
        num_classes: int = 5,
        img_size: int = 224,
        num_graph_layers: int = 3,
        decoder_channels: List[int] = [512, 256, 128, 64],
        freeze_text_encoder: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.img_encoder_name = img_encoder_name
        self.text_encoder_name = text_encoder_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 1. Image Encoder
        self.image_encoder = ImageEncoder(
            model_name=img_encoder_name,
            pretrained=pretrained,
            img_size=img_size,
            embed_dim=embed_dim
        )
        
        # 2. Text Encoder
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            embed_dim=embed_dim,
            freeze_encoder=freeze_text_encoder
        )
        
        # 3. Instruction Grounding Module (KEY NOVELTY)
        self.grounding_module = CompositionalGraphReasoning(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_graph_layers=num_graph_layers
        )
        
        # 4. Segmentation Decoder
        self.decoder = SegmentationDecoder(
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            img_size=img_size
        )
        
        # Loss weights for different components
        self.register_buffer('loss_weights', torch.ones(num_classes + 1))
        
        self.class_names = [
            'Neoplastic',
            'Inflammatory',
            'Connective_Soft_tissue',
            'Epithelial',
            'Dead',
            'Background'
        ]
    
    def forward(
        self,
        images: torch.Tensor,
        instructions: List[str],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CIPS-Net.
        
        Args:
            images: Input images [B, 3, H, W]
            instructions: List of instruction strings
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - masks: Segmentation masks [B, num_classes+1, H, W]
                - class_presence: Detected class presence [B, num_classes]
                - attention_scores: Graph attention scores [B, num_classes] (optional)
        """
        B = images.shape[0]
        
        # 1. Encode images
        patch_embeddings = self.image_encoder(images)
        # [B, num_patches, embed_dim]
        
        # 2. Encode instructions
        token_embeddings, sentence_embedding = self.text_encoder(texts=instructions)
        # token_embeddings: [B, seq_len, embed_dim]
        # sentence_embedding: [B, embed_dim]
        
        # Extract class presence from instructions
        class_presence = self.text_encoder.extract_mentioned_classes(instructions)
        # [B, num_classes]
        
        # 3. Ground instruction in visual features using graph reasoning
        grounded_features, class_guidance, attention_scores = self.grounding_module(
            visual_features=patch_embeddings,
            text_features=token_embeddings,
            sentence_embedding=sentence_embedding,
            class_presence=class_presence
        )
        # grounded_features: [B, num_patches, embed_dim]
        # class_guidance: [B, num_classes, embed_dim]
        # attention_scores: [B, num_classes]
        
        # 4. Decode to segmentation masks
        masks, _ = self.decoder(
            grounded_features=grounded_features,
            class_guidance=class_guidance,
            class_presence=class_presence
        )
        # masks: [B, num_classes+1, H, W]
        
        # Prepare output
        output = {
            'masks': masks,
            'class_presence': class_presence,
            'grounded_features': grounded_features,
            'class_guidance': class_guidance
        }
        
        if return_attention:
            output['attention_scores'] = attention_scores
        
        return output
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_presence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth masks [B, num_classes+1, H, W]
            class_presence: Ground truth class presence [B, num_classes]
            
        Returns:
            Dictionary with loss components
        """
        pred_masks = predictions['masks']
        pred_class_presence = predictions['class_presence']
        
        # 1. Segmentation loss (BCE + Dice)
        seg_loss = self._compute_segmentation_loss(pred_masks, targets)
        
        # 2. Class presence loss (optional, if ground truth provided)
        if class_presence is not None:
            presence_loss = F.binary_cross_entropy_with_logits(
                pred_class_presence,
                class_presence
            )
        else:
            presence_loss = torch.tensor(0.0).to(pred_masks.device)
        
        # 3. Regularization losses
        # Encourage sparsity in attention
        if 'attention_scores' in predictions:
            attention_reg = self._compute_attention_regularization(
                predictions['attention_scores']
            )
        else:
            attention_reg = torch.tensor(0.0).to(pred_masks.device)
        
        # Total loss
        total_loss = (
            seg_loss +
            0.1 * presence_loss +
            0.01 * attention_reg
        )
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'presence_loss': presence_loss,
            'attention_reg': attention_reg
        }
    
    def _compute_segmentation_loss(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Combined BCE + Dice loss for segmentation.
        
        Args:
            pred_masks: Predicted masks [B, C, H, W]
            target_masks: Ground truth masks [B, C, H, W]
            
        Returns:
            Combined loss value
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks,
            target_masks,
            reduction='none'
        )
        
        # Weight by class importance
        bce_loss = (bce_loss * self.loss_weights.view(1, -1, 1, 1)).mean()
        
        # Dice loss
        pred_probs = torch.sigmoid(pred_masks)
        dice_loss = self._dice_loss(pred_probs, target_masks)
        
        return bce_loss + dice_loss
    
    def _dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-5
    ) -> torch.Tensor:
        """
        Dice loss for segmentation.
        
        Args:
            pred: Predicted probabilities [B, C, H, W]
            target: Ground truth masks [B, C, H, W]
            smooth: Smoothing factor
            
        Returns:
            Dice loss value
        """
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss
    
    def _compute_attention_regularization(
        self,
        attention_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Regularize attention scores to encourage focused attention.
        
        Args:
            attention_scores: [B, num_classes]
            
        Returns:
            Regularization loss
        """
        # Entropy regularization (encourage sparsity)
        entropy = -(attention_scores * torch.log(attention_scores + 1e-8)).sum(dim=-1)
        return entropy.mean()
    
    def predict(
        self,
        images: torch.Tensor,
        instructions: List[str],
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode prediction.
        
        Args:
            images: Input images [B, 3, H, W]
            instructions: List of instruction strings
            threshold: Threshold for binary masks
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(images, instructions, return_attention=True)
            
            # Convert logits to probabilities
            mask_probs = torch.sigmoid(outputs['masks'])
            
            # Apply threshold
            binary_masks = (mask_probs > threshold).float()
            
            outputs['mask_probs'] = mask_probs
            outputs['binary_masks'] = binary_masks
        
        return outputs
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model configuration and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CIPS-Net',
            'image_encoder': self.img_encoder_name,
            'text_encoder': self.text_encoder_name,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'class_names': self.class_names
        }


if __name__ == "__main__":
    # Test CIPS-Net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CIPSNet(
        img_encoder_name='vit_b_16',
        text_encoder_name="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=768,
        num_classes=5,
        img_size=224,
        pretrained=False  # Set to True when actually training
    ).to(device)
    
    # Test inputs
    images = torch.randn(2, 3, 224, 224).to(device)
    instructions = [
        "Segment Neoplastic and Inflammatory regions in this Breast tissue.",
        "Identify Connective_Soft_tissue and Epithelial structures."
    ]
    
    # Forward pass
    print("Running forward pass...")
    outputs = model(images, instructions, return_attention=True)
    
    print(f"\nOutput shapes:")
    print(f"  Masks: {outputs['masks'].shape}")
    print(f"  Class presence: {outputs['class_presence'].shape}")
    print(f"  Attention scores: {outputs['attention_scores'].shape}")
    
    # Test loss computation
    target_masks = torch.randint(0, 2, (2, 6, 224, 224)).float().to(device)
    target_presence = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]]).float().to(device)
    
    losses = model.compute_loss(outputs, target_masks, target_presence)
    print(f"\nLoss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in info.items():
        if key != 'class_names':
            print(f"  {key}: {value}")
