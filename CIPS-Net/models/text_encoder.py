"""
Text Encoder: Clinical-BERT / PubMedBERT for Medical Text
Extracts token embeddings and sentence embeddings from instructions
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Optional, Tuple


class TextEncoder(nn.Module):
    """
    Clinical text encoder using medical domain pretrained models.
    
    Supported models:
        - emilyalsentzer/Bio_ClinicalBERT
        - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
        - dmis-lab/biobert-base-cased-v1.1
    
    Args:
        model_name: HuggingFace model name
        embed_dim: Output embedding dimension
        max_length: Maximum sequence length
        freeze_encoder: Whether to freeze BERT weights
    """
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 768,
        max_length: int = 128,
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Load pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get BERT hidden size
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layers
        self.token_proj = nn.Linear(self.bert_hidden_size, embed_dim)
        self.sentence_proj = nn.Linear(self.bert_hidden_size, embed_dim)
        
        # Normalization and dropout
        self.token_norm = nn.LayerNorm(embed_dim)
        self.sentence_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For extracting class-specific representations
        self.class_names = [
            'Neoplastic',
            'Inflammatory',
            'Connective_Soft_tissue',
            'Epithelial',
            'Dead'
        ]
        
        # Class name embeddings (for graph reasoning)
        self.register_buffer(
            'class_name_embeddings',
            torch.zeros(len(self.class_names), embed_dim)
        )
        self._initialize_class_embeddings()
    
    def _initialize_class_embeddings(self):
        """Precompute embeddings for class names."""
        with torch.no_grad():
            for i, class_name in enumerate(self.class_names):
                tokens = self.tokenizer(
                    class_name,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=16
                )
                
                # Move to same device as model
                if next(self.parameters()).is_cuda:
                    tokens = {k: v.cuda() for k, v in tokens.items()}
                
                outputs = self.bert(**tokens)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                
                # Project and normalize
                cls_embedding = self.sentence_proj(cls_embedding)
                cls_embedding = self.sentence_norm(cls_embedding)
                
                self.class_name_embeddings[i] = cls_embedding.squeeze(0)
    
    def tokenize(self, texts: list) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of instruction strings
            
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        return tokens
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text instructions.
        
        Args:
            input_ids: Tokenized input [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            texts: Raw text strings (will be tokenized if input_ids not provided)
            
        Returns:
            token_embeddings: [B, seq_len, embed_dim]
            sentence_embedding: [B, embed_dim]
        """
        # Tokenize if raw texts provided
        if input_ids is None and texts is not None:
            tokens = self.tokenize(texts)
            input_ids = tokens['input_ids'].to(next(self.parameters()).device)
            attention_mask = tokens['attention_mask'].to(next(self.parameters()).device)
        
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract embeddings
        token_embeddings = outputs.last_hidden_state  # [B, seq_len, bert_hidden_size]
        cls_embedding = token_embeddings[:, 0, :]  # [B, bert_hidden_size]
        
        # Project to target dimension
        token_embeddings = self.token_proj(token_embeddings)
        token_embeddings = self.token_norm(token_embeddings)
        token_embeddings = self.dropout(token_embeddings)
        
        sentence_embedding = self.sentence_proj(cls_embedding)
        sentence_embedding = self.sentence_norm(sentence_embedding)
        sentence_embedding = self.dropout(sentence_embedding)
        
        return token_embeddings, sentence_embedding
    
    def get_class_embeddings(self) -> torch.Tensor:
        """
        Get precomputed class name embeddings.
        
        Returns:
            class_embeddings: [num_classes, embed_dim]
        """
        return self.class_name_embeddings
    
    def extract_mentioned_classes(self, texts: list) -> torch.Tensor:
        """
        Extract which classes are mentioned in the instruction.
        
        Args:
            texts: List of instruction strings
            
        Returns:
            class_presence: Binary matrix [B, num_classes]
        """
        B = len(texts)
        class_presence = torch.zeros(B, len(self.class_names))
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            for j, class_name in enumerate(self.class_names):
                # Check for class name (case-insensitive, with word boundaries)
                class_lower = class_name.lower().replace('_', ' ')
                if class_lower in text_lower or class_name in text:
                    class_presence[i, j] = 1.0
        
        return class_presence.to(next(self.parameters()).device)


if __name__ == "__main__":
    # Test the text encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = TextEncoder(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=768
    ).to(device)
    
    # Test instructions
    instructions = [
        "Segment Neoplastic and Inflammatory regions in this Breast tissue.",
        "Identify Connective_Soft_tissue in the sample.",
        "Mark all Dead cells and Epithelial structures."
    ]
    
    # Encode instructions
    token_embeddings, sentence_embeddings = encoder(texts=instructions)
    
    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"Sentence embeddings shape: {sentence_embeddings.shape}")
    
    # Get class embeddings
    class_embeddings = encoder.get_class_embeddings()
    print(f"\nClass embeddings shape: {class_embeddings.shape}")
    
    # Extract mentioned classes
    class_presence = encoder.extract_mentioned_classes(instructions)
    print(f"\nClass presence matrix:\n{class_presence}")
    print(f"\nClass names: {encoder.class_names}")
