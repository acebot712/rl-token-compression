import gymnasium as gym
import numpy as np
import torch
import json
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Tuple, Dict, Any, Optional


class TokenCompressionEnv(gym.Env):
    """
    RL Environment for token compression.
    The agent decides which tokens to keep or mask in a sequence.
    """
    
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        reconstructor: GPT2LMHeadModel,
        data_path: str,
        max_seq_length: int = 1024,
        context_window: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.reconstructor = reconstructor
        self.max_seq_length = max_seq_length
        self.context_window = context_window
        self.device = device
        
        # Load dataset
        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        
        # Filter sequences to remove empty or too short ones
        self.sequences = [
            seq["tokens"] for seq in self.dataset
            if len(seq["tokens"]) >= context_window
        ]
        
        if not self.sequences:
            raise ValueError(f"No valid sequences found in {data_path}")
            
        print(f"Loaded {len(self.sequences)} sequences for training")
        
        # Action space: binary decision for each token (0: mask, 1: keep)
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(max_seq_length,), dtype=np.float32
        )
        
        # Observation space: token embeddings and context
        # Get GPT-2 embedding size (768 for base model) + context window
        # This needs to match the dimensions from _get_observation
        embedding_size = self.reconstructor.transformer.wte.weight.shape[1]  # Usually 768
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_seq_length, embedding_size + context_window),
            dtype=np.float32
        )
        
        self.current_sequence = None
        self.current_mask = None
        self.episode_count = 0
        
    def _get_observation(self) -> np.ndarray:
        """Convert current sequence and context into observation."""
        if self.current_sequence is None:
            return np.zeros(self.observation_space.shape)
            
        # Get token embeddings
        token_embeddings = self.reconstructor.transformer.wte(
            torch.tensor(self.current_sequence).to(self.device)
        ).detach().cpu().numpy()
        
        # Add context window features
        context_features = np.zeros((self.max_seq_length, self.context_window))
        for i in range(self.max_seq_length):
            start = max(0, i - self.context_window // 2)
            end = min(self.max_seq_length, i + self.context_window // 2)
            context_features[i, :end-start] = self.current_sequence[start:end]
            
        return np.concatenate([token_embeddings, context_features], axis=1)
        
    def _calculate_reward(
        self,
        masked_sequence: List[int],
        original_sequence: List[int]
    ) -> float:
        """Calculate reward based on compression ratio and reconstruction loss."""
        # Calculate compression ratio
        original_length = len(original_sequence)
        masked_length = len(masked_sequence)
        compression_ratio = 1 - (masked_length / original_length)
        
        # Calculate reconstruction loss
        with torch.no_grad():
            inputs = torch.tensor(masked_sequence).unsqueeze(0).to(self.device)
            outputs = self.reconstructor(inputs, labels=inputs)
            reconstruction_loss = outputs.loss.item()
            
        # Combine into final reward
        # Higher compression and lower loss = better
        reward = compression_ratio - 0.1 * reconstruction_loss
        
        return reward
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with a new sequence from the dataset."""
        super().reset(seed=seed)
        
        # Select a random sequence from the dataset
        seq_idx = random.randint(0, len(self.sequences) - 1)
        tokens = self.sequences[seq_idx]
        
        # If sequence is too long, select a random window
        if len(tokens) > self.max_seq_length:
            start_idx = random.randint(0, len(tokens) - self.max_seq_length)
            tokens = tokens[start_idx:start_idx + self.max_seq_length]
        # If sequence is too short, pad with EOS tokens
        elif len(tokens) < self.max_seq_length:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.max_seq_length - len(tokens))
        
        self.current_sequence = tokens
        self.current_mask = np.ones(self.max_seq_length)
        self.episode_count += 1
        
        info = {
            "episode": self.episode_count,
            "sequence_length": len(tokens),
            "sequence_idx": seq_idx
        }
        
        return self._get_observation(), info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array of decisions for each token. Values may be continuous in
                the range ``[0, 1]``; any value greater than ``0.5`` is treated as
                ``"keep"`` while lower values are treated as ``"mask"``.
            
        Returns:
            observation: Next state
            reward: Reward for the action
            terminated: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Apply mask to sequence
        # Convert action to boolean mask and use it to filter the sequence
        mask = action > 0.5
        masked_indices = np.where(mask)[0]
        masked_sequence = [self.current_sequence[i] for i in masked_indices]
        
        # Calculate reward
        reward = self._calculate_reward(masked_sequence, self.current_sequence)
        
        # Update current mask
        self.current_mask = action
        
        # For now, we'll end the episode after one step
        # In practice, you might want to continue until certain conditions are met
        terminated = True
        truncated = False
        
        info = {
            "compression_ratio": 1 - len(masked_sequence) / len(self.current_sequence),
            "original_length": len(self.current_sequence),
            "masked_length": len(masked_sequence)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
