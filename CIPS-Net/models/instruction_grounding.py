"""
Instruction Grounding Module: Compositional Graph Reasoning
This is the KEY NOVELTY of CIPS-Net

Implements a graph-based reasoning mechanism where:
- Nodes represent pathology classes
- Edges represent co-occurrence and interaction
- Instructions activate a subgraph
- Decoder segments based on activated subgraph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class PathologyGraph(nn.Module):
    """
    Graph structure representing relationships between pathology classes.
    
    Nodes: Pathology classes (Neoplastic, Inflammatory, etc.)
    Edges: Co-occurrence and semantic relationships
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 768,
        num_relation_types: int = 3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_relation_types = num_relation_types
        
        # Node embeddings (will be updated during training)
        self.node_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))
        
        # Edge embeddings for different relation types
        # Type 0: co-occurrence, Type 1: hierarchical, Type 2: semantic similarity
        self.edge_embeddings = nn.Parameter(
            torch.randn(num_relation_types, embed_dim)
        )
        
        # Initialize adjacency matrix based on domain knowledge
        # This can be learned or fixed based on co-occurrence statistics
        self.register_buffer(
            'adjacency_matrix',
            torch.zeros(num_classes, num_classes, num_relation_types)
        )
        self._initialize_graph_structure()
        
        # Graph attention layers
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Node update layers
        self.node_update = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.edge_embeddings)
    
    def _initialize_graph_structure(self):
        """
        Initialize graph structure based on domain knowledge.
        Based on the co-occurrence analysis from the dataset.
        """
        # Co-occurrence relationships (from dataset analysis)
        # High co-occurrence pairs
        co_occur_pairs = [
            (0, 1),  # Neoplastic - Inflammatory
            (0, 2),  # Neoplastic - Connective_Soft_tissue
            (1, 2),  # Inflammatory - Connective_Soft_tissue
            (2, 3),  # Connective_Soft_tissue - Epithelial
        ]
        
        for i, j in co_occur_pairs:
            self.adjacency_matrix[i, j, 0] = 1.0  # Co-occurrence
            self.adjacency_matrix[j, i, 0] = 1.0
        
        # Hierarchical relationships
        hierarchical_pairs = [
            (3, 2),  # Epithelial -> Connective_Soft_tissue
        ]
        
        for i, j in hierarchical_pairs:
            self.adjacency_matrix[i, j, 1] = 1.0
        
        # All classes have self-loops
        for i in range(self.num_classes):
            self.adjacency_matrix[i, i, 2] = 1.0  # Self-similarity
    
    def forward(
        self,
        instruction_embedding: torch.Tensor,
        class_presence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform graph reasoning based on instruction.
        
        Args:
            instruction_embedding: [B, embed_dim]
            class_presence: Binary matrix indicating mentioned classes [B, num_classes]
            
        Returns:
            activated_nodes: Node features for activated subgraph [B, num_classes, embed_dim]
            attention_weights: Attention weights for graph [B, num_classes, num_classes]
        """
        B = instruction_embedding.shape[0]
        
        # Expand node embeddings for batch
        node_features = self.node_embeddings.unsqueeze(0).expand(B, -1, -1)
        # [B, num_classes, embed_dim]
        
        # Aggregate edge information
        edge_features = torch.zeros_like(node_features)
        for rel_type in range(self.num_relation_types):
            # Get adjacency for this relation type
            adj = self.adjacency_matrix[:, :, rel_type]  # [num_classes, num_classes]
            
            # Edge embedding
            edge_emb = self.edge_embeddings[rel_type]  # [embed_dim]
            
            # Aggregate neighbor features
            neighbor_features = torch.matmul(adj, self.node_embeddings)
            # [num_classes, embed_dim]
            
            # Weight by edge embedding
            edge_features = edge_features + (
                neighbor_features.unsqueeze(0) * edge_emb.unsqueeze(0).unsqueeze(0)
            )
        
        # Normalize
        edge_features = edge_features / self.num_relation_types
        
        # Combine node and edge features
        combined_features = torch.cat([node_features, edge_features], dim=-1)
        updated_nodes = self.node_update(combined_features)
        # [B, num_classes, embed_dim]
        
        # Instruction-guided attention
        instruction_query = instruction_embedding.unsqueeze(1)  # [B, 1, embed_dim]
        
        attended_nodes, attention_weights = self.graph_attention(
            query=instruction_query,
            key=updated_nodes,
            value=updated_nodes
        )
        # attended_nodes: [B, 1, embed_dim]
        # attention_weights: [B, 1, num_classes]
        
        # Activate nodes based on class presence and attention
        activation_scores = attention_weights.squeeze(1)  # [B, num_classes]
        
        # Mask by mentioned classes (soft masking)
        activation_scores = activation_scores * (class_presence + 0.1)
        activation_scores = F.softmax(activation_scores, dim=-1)
        
        # Weighted node features
        activated_nodes = updated_nodes * activation_scores.unsqueeze(-1)
        # [B, num_classes, embed_dim]
        
        return activated_nodes, activation_scores


class CompositionalGraphReasoning(nn.Module):
    """
    Main instruction grounding module using compositional graph reasoning.
    
    This module:
    1. Constructs a pathology class graph
    2. Activates relevant subgraph based on instruction
    3. Produces class-specific and global guidance features
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 768,
        num_graph_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_graph_layers = num_graph_layers
        
        # Pathology graph
        self.graph = PathologyGraph(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_relation_types=3
        )
        
        # Multi-layer graph reasoning
        self.graph_layers = nn.ModuleList([
            PathologyGraph(
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_relation_types=3
            )
            for _ in range(num_graph_layers)
        ])
        
        # Cross-modal fusion
        self.visual_text_fusion = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Class-specific feature generators
        self.class_feature_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
            for _ in range(num_classes)
        ])
        
        # Global feature fusion
        self.global_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        sentence_embedding: torch.Tensor,
        class_presence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ground instruction in visual features using graph reasoning.
        
        Args:
            visual_features: Patch embeddings [B, num_patches, embed_dim]
            text_features: Token embeddings [B, seq_len, embed_dim]
            sentence_embedding: Sentence embedding [B, embed_dim]
            class_presence: Binary class presence [B, num_classes]
            
        Returns:
            grounded_features: Fused visual-text features [B, num_patches, embed_dim]
            class_guidance: Class-specific guidance [B, num_classes, embed_dim]
            attention_scores: Graph attention scores [B, num_classes]
        """
        B = visual_features.shape[0]
        
        # Multi-layer graph reasoning
        activated_nodes = None
        attention_scores = None
        
        for layer in self.graph_layers:
            activated_nodes, attention_scores = layer(
                instruction_embedding=sentence_embedding,
                class_presence=class_presence
            )
            # Update sentence embedding with graph information
            sentence_embedding = sentence_embedding + activated_nodes.mean(dim=1)
        
        # Final activated nodes: [B, num_classes, embed_dim]
        # attention_scores: [B, num_classes]
        
        # Cross-modal fusion: align visual and text features
        fused_visual, _ = self.visual_text_fusion(
            query=visual_features,
            key=text_features,
            value=text_features
        )
        # [B, num_patches, embed_dim]
        
        # Combine with original visual features
        grounded_features = visual_features + fused_visual
        grounded_features = self.norm(grounded_features)
        
        # Generate class-specific guidance
        class_guidance = []
        for i in range(self.num_classes):
            class_feat = self.class_feature_generator[i](activated_nodes[:, i, :])
            class_guidance.append(class_feat)
        
        class_guidance = torch.stack(class_guidance, dim=1)
        # [B, num_classes, embed_dim]
        
        # Add global context
        global_context = sentence_embedding.unsqueeze(1).expand(-1, self.num_classes, -1)
        class_guidance = self.global_fusion(
            torch.cat([class_guidance, global_context], dim=-1)
        )
        
        # Final projection
        grounded_features = self.output_proj(grounded_features)
        
        return grounded_features, class_guidance, attention_scores


if __name__ == "__main__":
    # Test the instruction grounding module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    module = CompositionalGraphReasoning(
        num_classes=5,
        embed_dim=768,
        num_graph_layers=3
    ).to(device)
    
    # Test inputs
    B, num_patches, seq_len = 2, 196, 32
    visual_features = torch.randn(B, num_patches, 768).to(device)
    text_features = torch.randn(B, seq_len, 768).to(device)
    sentence_embedding = torch.randn(B, 768).to(device)
    class_presence = torch.tensor([
        [1, 1, 0, 0, 0],  # Neoplastic and Inflammatory
        [0, 1, 1, 1, 0]   # Inflammatory, Connective, Epithelial
    ], dtype=torch.float32).to(device)
    
    # Forward pass
    grounded_features, class_guidance, attention_scores = module(
        visual_features=visual_features,
        text_features=text_features,
        sentence_embedding=sentence_embedding,
        class_presence=class_presence
    )
    
    print(f"Grounded features shape: {grounded_features.shape}")
    print(f"Class guidance shape: {class_guidance.shape}")
    print(f"Attention scores shape: {attention_scores.shape}")
    print(f"\nAttention scores:\n{attention_scores}")
