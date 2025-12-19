"""
Text Encoder for processing natural language instructions
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """
    Text encoder using pretrained language models (BERT, DistilBERT, etc.)
    """
    
    def __init__(
        self,
        model_name='distilbert-base-uncased',
        freeze_encoder=False,
        max_length=77
    ):
        """
        Args:
            model_name: Hugging Face model name
            freeze_encoder: Whether to freeze the text encoder
            max_length: Maximum text sequence length
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        
        # Freeze if requested
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"Text encoder frozen: {model_name}")
        
        print(f"Text Encoder: {model_name}, Hidden dim: {self.hidden_dim}")
    
    def tokenize(self, texts):
        """
        Tokenize text inputs
        
        Args:
            texts: List of strings
        
        Returns:
            input_ids: [B, L]
            attention_mask: [B, L]
        """
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']
    
    def forward(self, texts):
        """
        Encode text inputs
        
        Args:
            texts: List of strings (batch)
        
        Returns:
            text_embeddings: [B, L, D] - token embeddings
            pooled_output: [B, D] - sentence embedding (CLS token or mean pooling)
        """
        # Tokenize
        input_ids, attention_mask = self.tokenize(texts)
        input_ids = input_ids.to(next(self.model.parameters()).device)
        attention_mask = attention_mask.to(next(self.model.parameters()).device)
        
        # Encode
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Token embeddings: [B, L, D]
        text_embeddings = outputs.last_hidden_state
        
        # Sentence embedding: use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output  # [B, D]
        else:
            # Mean pooling over tokens (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size()).float()
            pooled_output = torch.sum(text_embeddings * mask_expanded, 1) / torch.clamp(
                mask_expanded.sum(1), min=1e-9
            )
        
        return text_embeddings, pooled_output, attention_mask


class SimpleTextEncoder(nn.Module):
    """
    Even simpler text encoder using just embeddings + linear projection
    """
    
    def __init__(
        self,
        vocab_size=10000,
        embed_dim=512,
        max_length=77
    ):
        super().__init__()
        
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Simple embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_length, embed_dim))
        
        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.ln = nn.LayerNorm(embed_dim)
    
    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids: [B, L]
            attention_mask: [B, L]
        
        Returns:
            text_embeddings: [B, L, D]
            pooled_output: [B, D]
        """
        # Embed tokens
        x = self.token_embedding(token_ids)  # [B, L, D]
        
        # Add positional embeddings
        x = x + self.position_embedding[:token_ids.shape[1]]
        
        # Create padding mask for transformer
        if attention_mask is not None:
            # Transformer expects: True = ignore, False = attend
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Layer norm
        x = self.ln(x)
        
        # Pooled output (first token)
        pooled = x[:, 0, :]
        
        return x, pooled


if __name__ == "__main__":
    # Test text encoder
    encoder = TextEncoder(model_name='distilbert-base-uncased')
    
    texts = [
        "Segment neoplastic tissue in this histopathology image",
        "Identify inflammatory regions"
    ]
    
    text_emb, pooled, mask = encoder(texts)
    print(f"Text embeddings: {text_emb.shape}")
    print(f"Pooled output: {pooled.shape}")
    print(f"Attention mask: {mask.shape}")
