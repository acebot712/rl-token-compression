"""
Simple policy network for token compression agent.

This module provides a lightweight, efficient policy network specifically designed
for token compression tasks. The network is kept simple and fast (1M parameters)
compared to full transformer architectures (100M+ parameters).

Key features:
- Feedforward architecture with local context
- Efficient embedding processing
- Fast inference for real-time compression decisions
- Modular design for easy experimentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class SimpleCompressionPolicy(nn.Module):
    """
    Lightweight policy network for token compression decisions.
    
    This network makes binary keep/drop decisions for each token based on:
    - Token embeddings from the reconstructor
    - Local context window
    - Position information
    
    Architecture: ~1M parameters (vs 100M+ for full transformers)
    - Embedding processing layers
    - Local context aggregation
    - Binary decision head
    """
    
    def __init__(self, embedding_dim: int = 768, context_window: int = 5, 
                 hidden_dim: int = 256, device: str = "cpu"):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Embedding processing layers
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Context aggregation (processes local window)
        self.context_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * context_window, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Position encoding
        self.position_embedding = nn.Embedding(512, hidden_dim // 4)  # Max sequence length 512
        
        # Decision head (binary keep/drop)
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # Binary decision
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Model initialized - parameter count logged in trainer
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, embeddings: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for compression decisions.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim]
            positions: Optional position indices [batch_size, seq_len]
            
        Returns:
            Logits for keep/drop decisions [batch_size, seq_len]
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Process embeddings
        processed_embeddings = self.embedding_processor(embeddings)  # [batch, seq, hidden]
        
        # Create context windows
        padded_embeddings = self._create_context_windows(processed_embeddings)  # [batch, seq, hidden*context]
        
        # Aggregate context
        context_features = self.context_aggregator(padded_embeddings)  # [batch, seq, hidden//2]
        
        # Add position information
        if positions is None:
            positions = torch.arange(seq_len, device=embeddings.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Clamp positions to avoid out-of-bounds
        positions = torch.clamp(positions, 0, 511)
        position_features = self.position_embedding(positions)  # [batch, seq, hidden//4]
        
        # Combine features
        combined_features = torch.cat([context_features, position_features], dim=-1)
        
        # Make decisions
        logits = self.decision_head(combined_features).squeeze(-1)  # [batch, seq]
        
        return logits
    
    def _create_context_windows(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Create local context windows around each token.
        
        Args:
            embeddings: Processed embeddings [batch, seq, hidden]
            
        Returns:
            Context windows [batch, seq, hidden * context_window]
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        context_radius = self.context_window // 2
        
        # Pad sequence for context windows
        padding = torch.zeros(batch_size, context_radius, hidden_dim, device=embeddings.device)
        padded = torch.cat([padding, embeddings, padding], dim=1)
        
        # Create sliding windows
        windows = []
        for i in range(seq_len):
            window_start = i
            window_end = i + self.context_window
            window = padded[:, window_start:window_end, :].reshape(batch_size, -1)
            windows.append(window)
        
        return torch.stack(windows, dim=1)  # [batch, seq, hidden * context_window]
    
    def get_compression_decisions(self, embeddings: torch.Tensor, 
                                threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary compression decisions.
        
        Args:
            embeddings: Token embeddings
            threshold: Decision threshold
            
        Returns:
            Binary decisions [batch_size, seq_len]
        """
        with torch.no_grad():
            logits = self.forward(embeddings)
            probabilities = torch.sigmoid(logits)
            decisions = (probabilities > threshold).float()
            return decisions
    
    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get raw logits for training."""
        return self.forward(embeddings)
    
    def get_compression_ratio(self, embeddings: torch.Tensor, 
                            threshold: float = 0.5) -> float:
        """
        Calculate compression ratio for given embeddings.
        
        Args:
            embeddings: Token embeddings
            threshold: Decision threshold
            
        Returns:
            Compression ratio (fraction of tokens kept)
        """
        decisions = self.get_compression_decisions(embeddings, threshold)
        return decisions.mean().item()
    
    def analyze_decisions(self, embeddings: torch.Tensor, 
                         tokens: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Analyze compression decisions for debugging.
        
        Args:
            embeddings: Token embeddings
            tokens: Optional token strings for analysis
            
        Returns:
            Analysis dictionary
        """
        with torch.no_grad():
            logits = self.forward(embeddings)
            probabilities = torch.sigmoid(logits)
            decisions = (probabilities > 0.5).float()
            
            batch_size, seq_len = decisions.shape
            
            analysis = {
                "compression_ratio": decisions.mean().item(),
                "keep_count": decisions.sum().item(),
                "total_tokens": batch_size * seq_len,
                "decision_entropy": self._calculate_entropy(probabilities),
                "probability_stats": {
                    "mean": probabilities.mean().item(),
                    "std": probabilities.std().item(),
                    "min": probabilities.min().item(),
                    "max": probabilities.max().item()
                }
            }
            
            if tokens and len(tokens) == seq_len:
                # Per-token analysis
                probs_list = probabilities[0].cpu().numpy()  # First batch item
                decisions_list = decisions[0].cpu().numpy()
                
                analysis["per_token"] = [
                    {
                        "token": token,
                        "probability": float(prob),
                        "decision": bool(decision),
                        "confidence": abs(prob - 0.5)
                    }
                    for token, prob, decision in zip(tokens, probs_list, decisions_list)
                ]
            
            return analysis
    
    def _calculate_entropy(self, probabilities: torch.Tensor) -> float:
        """Calculate decision entropy."""
        # Clamp probabilities to avoid log(0)
        probs = torch.clamp(probabilities, 1e-8, 1 - 1e-8)
        entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
        return entropy.mean().item()
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information for logging/debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "SimpleCompressionPolicy",
            "embedding_dim": self.embedding_dim,
            "context_window": self.context_window,
            "hidden_dim": self.hidden_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device
        }


# Factory function for easy model creation
def create_compression_policy(embedding_dim: int = 768, context_window: int = 5,
                            hidden_dim: int = 256, device: str = "auto") -> SimpleCompressionPolicy:
    """
    Create a compression policy with automatic device selection.
    
    Args:
        embedding_dim: Dimension of input embeddings
        context_window: Size of local context window
        hidden_dim: Hidden layer dimension
        device: Device ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Initialized compression policy
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return SimpleCompressionPolicy(
        embedding_dim=embedding_dim,
        context_window=context_window,
        hidden_dim=hidden_dim,
        device=device
    )


