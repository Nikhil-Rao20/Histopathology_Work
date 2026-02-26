
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for modeling class relationships.
    
    Implements attention-based message passing between class nodes.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Edge feature projection (for encoding relationships)
        self.edge_proj = nn.Linear(embed_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of graph attention.
        
        Args:
            node_features: [B, num_nodes, embed_dim]
            edge_features: [B, num_nodes, num_nodes, embed_dim] (optional)
            mask: [B, num_nodes] binary mask for valid nodes
            
        Returns:
            updated_features: [B, num_nodes, embed_dim]
        """
        B, N, C = node_features.shape
        
        # Compute Q, K, V
        Q = self.q_proj(node_features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(node_features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(node_features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, N, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [B, num_heads, N, N]
        
        # Add edge bias if provided
        if edge_features is not None:
            edge_bias = self.edge_proj(edge_features)  # [B, N, N, num_heads]
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, num_heads, N, N]
            attn_scores = attn_scores + edge_bias
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, embed_dim]
        out = self.out_proj(out)
        
        return out


class VisualTextCrossAttention(nn.Module):
    """
    Cross-attention between visual features and text features.
    
    Aligns visual patch features with text token features.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Visual as query, text as key/value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention: visual queries attend to text keys/values.
        
        Args:
            visual_features: [B, num_patches, embed_dim]
            text_features: [B, seq_len, embed_dim]
            
        Returns:
            attended_features: [B, num_patches, embed_dim]
        """
        B, N_v, C = visual_features.shape
        N_t = text_features.shape[1]
        
        Q = self.q_proj(visual_features).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(text_features).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(text_features).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N_v, C)
        out = self.out_proj(out)
        
        return out


class CompositionalGraphReasoning(nn.Module):
    """
    Compositional Graph Reasoning Module (CGR)
    
    Novel contribution: Models nucleus class relationships using:
    1. Class nodes initialized from text embeddings
    2. Graph attention between classes (models co-occurrence)
    3. Cross-attention between visual features and class nodes
    4. Text-guided class attention weighting
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 768,
        num_graph_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_graph_layers = num_graph_layers
        
        # Learnable class node embeddings
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        
        # Class name embeddings (will be initialized from text encoder)
        self.class_names = [
            "neoplastic tumor cancer cells",
            "inflammatory immune lymphocyte cells", 
            "connective soft tissue stromal cells",
            "dead necrotic apoptotic cells",
            "epithelial glandular cells"
        ]
        
        # Graph attention layers for class relationship modeling
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_graph_layers)
        ])
        self.graph_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_graph_layers)
        ])
        
        # Visual-text cross attention
        self.visual_text_attn = VisualTextCrossAttention(embed_dim, num_heads, dropout)
        self.visual_text_norm = nn.LayerNorm(embed_dim)
        
        # Visual-class cross attention
        self.visual_class_attn = VisualTextCrossAttention(embed_dim, num_heads, dropout)
        self.visual_class_norm = nn.LayerNorm(embed_dim)
        
        # Class-visual cross attention (class nodes attend to visual features)
        self.class_visual_attn = VisualTextCrossAttention(embed_dim, num_heads, dropout)
        self.class_visual_norm = nn.LayerNorm(embed_dim)
        
        # Text-to-class attention for generating class weights
        self.text_class_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Class guidance projection
        self.class_guidance_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        sentence_embedding: torch.Tensor,
        class_presence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of CGR module.
        
        Args:
            visual_features: Patch features [B, num_patches, embed_dim]
            text_features: Token features [B, seq_len, embed_dim]
            sentence_embedding: Sentence feature [B, embed_dim]
            class_presence: Binary class indicators [B, num_classes]
            
        Returns:
            grounded_features: Text-grounded visual features [B, num_patches, embed_dim]
            class_guidance: Class-specific guidance vectors [B, num_classes, embed_dim]
            attention_scores: Class attention weights [B, num_classes]
        """
        B = visual_features.shape[0]
        
        # 1. Initialize class nodes with learnable embeddings
        class_nodes = self.class_embeddings.unsqueeze(0).expand(B, -1, -1)
        # [B, num_classes, embed_dim]
        
        # 2. Update class nodes with visual context
        class_nodes = class_nodes + self.class_visual_attn(class_nodes, visual_features)
        class_nodes = self.class_visual_norm(class_nodes)
        
        # 3. Graph reasoning between class nodes
        for graph_layer, norm in zip(self.graph_layers, self.graph_norms):
            class_nodes = class_nodes + graph_layer(class_nodes, mask=class_presence)
            class_nodes = norm(class_nodes)
        
        # 4. Cross-attention: visual features attend to text
        visual_text = self.visual_text_attn(visual_features, text_features)
        visual_features = visual_features + visual_text
        visual_features = self.visual_text_norm(visual_features)
        
        # 5. Cross-attention: visual features attend to class nodes
        visual_class = self.visual_class_attn(visual_features, class_nodes)
        
        # 6. Combine visual-text and visual-class features
        combined = torch.cat([visual_features, visual_class], dim=-1)
        grounded_features = self.output_proj(combined)
        grounded_features = self.output_norm(grounded_features)
        
        # 7. Generate class attention weights from text
        attention_scores = self.text_class_proj(sentence_embedding)  # [B, num_classes]
        attention_scores = torch.sigmoid(attention_scores)
        
        # Weight by class presence (mentioned classes get higher weight)
        attention_scores = attention_scores * class_presence + (1 - class_presence) * 0.1
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # 8. Generate class guidance vectors
        class_guidance = self.class_guidance_proj(class_nodes)
        # Weight class guidance by attention scores
        class_guidance = class_guidance * attention_scores.unsqueeze(-1)
        
        return grounded_features, class_guidance, attention_scores
