import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding for token positions in the sequence.
    Implementation inspired by the Transformer architecture.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Hidden dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TokenImportanceAttention(nn.Module):
    """
    Custom attention mechanism to assess token importance.
    Uses a variation of self-attention with importance scoring.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize token importance attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, \
            "hidden_dim must be divisible by num_heads"
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Importance scoring layer
        self.importance_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply token importance attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, importance scores)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add head dimensions
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_len, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Apply output projection
        output = self.out_proj(context)
        
        # Compute importance scores
        importance = self.importance_layer(output).squeeze(-1)
        importance = torch.sigmoid(importance)  # Scale to [0, 1]
        
        return output, importance


class TokenCompressionPolicy(nn.Module):
    """
    Custom feature extractor for the token compression RL policy.
    This is used by the PPO agent to process token embeddings and context.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of input features (token embedding + context)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.features_dim = 64  # Output dimension for RL algorithm
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(hidden_dim, self.features_dim))
        
        # Combine layers
        self.features_extractor = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Features tensor of shape (batch_size, seq_length, features_dim)
        """
        return self.features_extractor(x)


class EnhancedTokenCompressionPolicy(nn.Module):
    """
    Enhanced policy network for token compression.
    Uses positional encoding, specialized attention mechanisms,
    and hierarchical features to assess token importance.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_context_gates: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_context_gates = use_context_gates
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Context gates - learn to focus on important context
        if use_context_gates:
            self.context_gates = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for modern PyTorch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Token importance attention
        self.token_importance = TokenImportanceAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Local context aggregation (convolutional)
        self.local_context = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Final decision layers
        self.decision_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Action probabilities (keep or mask tokens)
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Apply context gates if enabled
        if self.use_context_gates:
            context_weights = self.context_gates(features)
            features = features * context_weights
        
        # Process through transformer
        if mask is not None:
            features = self.transformer(features, src_key_padding_mask=mask)
        else:
            features = self.transformer(features)
            
        # Process through token importance attention
        attn_features, importance_scores = self.token_importance(features, mask)
        
        # Process through local context aggregation
        local_features = self.local_context(features.transpose(1, 2)).transpose(1, 2)
        
        # Concatenate global and local features
        combined_features = torch.cat([attn_features, local_features], dim=-1)
        
        # Get final decision probabilities
        action_probs = self.decision_layers(combined_features).squeeze(-1)
        
        return action_probs
        
    def get_importance_map(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed importance maps for analysis and visualization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary of importance metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Process through transformer
        if mask is not None:
            transformer_features = self.transformer(features, src_key_padding_mask=mask)
        else:
            transformer_features = self.transformer(features)
            
        # Process through token importance attention
        _, importance_scores = self.token_importance(transformer_features, mask)
        
        # Get final action probabilities
        action_probs = self.forward(x, mask)
        
        return {
            "raw_importance": importance_scores,
            "action_probs": action_probs,
            "feature_norm": torch.norm(features, dim=-1),
            "transformer_feature_norm": torch.norm(transformer_features, dim=-1)
        } 