"""
Encoders for CIPS-Net V2
========================

Contains:
1. ImageEncoder: ViT-based image encoder with multi-scale features
2. TextEncoder: Transformer-based text encoder for instructions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    Vision Transformer (ViT) based image encoder.
    Extracts multi-scale features for segmentation.
    
    Supports: vit_b_16, vit_l_16, vit_b_32
    """
    
    def __init__(
        self,
        model_name: str = 'vit_b_16',
        pretrained: bool = True,
        img_size: int = 256,
        embed_dim: int = 768,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Load pretrained ViT
        if model_name == 'vit_b_16':
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            self.vit_embed_dim = 768
            self.patch_size = 16
        elif model_name == 'vit_l_16':
            weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_l_16(weights=weights)
            self.vit_embed_dim = 1024
            self.patch_size = 16
        elif model_name == 'vit_b_32':
            weights = models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_32(weights=weights)
            self.vit_embed_dim = 768
            self.patch_size = 32
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove classification head
        self.vit.heads = nn.Identity()
        
        # Calculate spatial dimensions
        self.num_patches_per_side = img_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Project to common embedding dimension if needed
        if self.vit_embed_dim != embed_dim:
            self.proj = nn.Linear(self.vit_embed_dim, embed_dim)
        else:
            self.proj = nn.Identity()
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            patch_features: Patch embeddings [B, num_patches, embed_dim]
            cls_token: CLS token embedding [B, embed_dim]
        """
        B, C, H, W = x.shape
        
        # Resize input to 224x224 (ViT's expected size)
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Get patch embeddings from ViT encoder
        # ViT forward: patch embed -> add pos embed -> transformer blocks
        x = self.vit._process_input(x)  # [B, num_patches, vit_embed_dim]
        
        # Add class token
        batch_class_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Add positional embedding
        x = x + self.vit.encoder.pos_embedding
        
        # Apply transformer blocks
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        
        # Split CLS token and patch features
        cls_token = x[:, 0]  # [B, vit_embed_dim]
        patch_features = x[:, 1:]  # [B, num_patches, vit_embed_dim]
        
        # Project to common dimension
        patch_features = self.proj(patch_features)  # [B, num_patches, embed_dim]
        cls_token = self.proj(cls_token)  # [B, embed_dim]
        
        return patch_features, cls_token
    
    def get_spatial_features(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Reshape patch features to spatial format.
        
        Args:
            patch_features: [B, num_patches, embed_dim]
            
        Returns:
            spatial_features: [B, embed_dim, H', W'] where H'=W'=num_patches_per_side
        """
        B, N, C = patch_features.shape
        H = W = self.num_patches_per_side
        return patch_features.transpose(1, 2).reshape(B, C, H, W)


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder for instruction understanding.
    
    Uses pretrained language models (DistilBERT, BioClinicalBERT, etc.)
    """
    
    # Class name mapping for instruction parsing
    CLASS_NAMES = {
        0: ['neoplastic', 'tumor', 'cancer', 'malignant', 'neoplasm'],
        1: ['inflammatory', 'inflammation', 'immune', 'lymphocyte', 'macrophage'],
        2: ['connective', 'soft tissue', 'stromal', 'fibroblast', 'stroma'],
        3: ['dead', 'necrotic', 'apoptotic', 'dying', 'necrosis'],
        4: ['epithelial', 'epithelium', 'epithelioid', 'glandular']
    }
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        embed_dim: int = 768,
        max_length: int = 64,
        freeze_encoder: bool = False
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
            token_embeddings: Token-level features [B, seq_len, embed_dim]
            sentence_embedding: Sentence-level feature [B, embed_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        token_embeddings = outputs.last_hidden_state  # [B, seq_len, encoder_dim]
        
        # Mean pooling for sentence embedding
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask  # [B, encoder_dim]
        
        # Project to common dimension
        token_embeddings = self.proj(token_embeddings)  # [B, seq_len, embed_dim]
        sentence_embedding = self.proj(sentence_embedding)  # [B, embed_dim]
        
        return token_embeddings, sentence_embedding
    
    def extract_mentioned_classes(
        self, 
        texts: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Extract which nucleus classes are mentioned in the instructions.
        
        Args:
            texts: List of instruction strings
            
        Returns:
            class_presence: Binary tensor [B, num_classes] indicating mentioned classes
        """
        import re
        
        if device is None:
            device = next(self.parameters()).device
            
        B = len(texts)
        num_classes = len(self.CLASS_NAMES)
        class_presence = torch.zeros(B, num_classes, device=device)
        
        # Words that indicate ALL classes (use word boundaries to avoid "falls" matching "all")
        ALL_CLASS_PATTERNS = [
            r'\ball\b',           # "all" as whole word
            r'\bevery\b',         # "every" as whole word  
            r'\bnuclei\b',        # "nuclei" as whole word
            r'\bcells\b',         # "cells" as whole word
            r'segment everything',
            r'all nuclei',
            r'all cells',
        ]
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            # Check for "all" or general segmentation requests (with word boundaries)
            if any(re.search(pattern, text_lower) for pattern in ALL_CLASS_PATTERNS):
                class_presence[i, :] = 1.0
                continue
            
            # Check for specific class mentions
            for class_idx, keywords in self.CLASS_NAMES.items():
                if any(keyword in text_lower for keyword in keywords):
                    class_presence[i, class_idx] = 1.0
        
        # If nothing mentioned, assume all classes
        no_mention_mask = class_presence.sum(dim=1) == 0
        class_presence[no_mention_mask, :] = 1.0
        
        return class_presence
