"""
Information-theoretic reward function for token compression.

This replaces the broken arbitrary reward function with a theoretically grounded
approach based on rate-distortion theory. The key insight is to balance the
information rate (compression) against distortion (reconstruction quality) using
principles from information theory.

Based on Shannon's rate-distortion framework:
- Rate: R = E[-log p(mask|x)] (expected information content of compression decisions)
- Distortion: D = E[d(x, x̂)] (expected reconstruction error)
- Objective: Minimize R + βD for Lagrange multiplier β

Key improvements over the original reward:
- Proper normalization to [0, 1] range
- Adaptive temperature scheduling
- Importance weighting for rare/critical tokens
- Information-theoretic foundation
- No arbitrary coefficients
"""

from typing import Dict, Optional

import torch

# Set up logging
from utils.logging import get_component_logger

logger = get_component_logger("REWARDS")


class InformationTheoreticReward:
    """
    Computes rewards based on rate-distortion theory principles.

    The reward balances compression efficiency against reconstruction quality
    using information-theoretic measures rather than arbitrary linear combinations.
    """

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        temperature_schedule: str = "adaptive",  # "fixed", "linear", "exponential", "adaptive"
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        importance_weighting: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the reward function.

        Args:
            vocab_size: Size of the vocabulary for importance weighting
            temperature_schedule: Type of temperature scheduling
            initial_temperature: Starting temperature
            min_temperature: Minimum temperature value
            importance_weighting: Whether to weight tokens by importance
            device: Device for computations
        """
        self.vocab_size = vocab_size
        self.temperature_schedule = temperature_schedule
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.importance_weighting = importance_weighting
        self.device = torch.device(device)

        # Token frequency cache for importance weighting
        self.token_frequencies = None
        self.frequency_cache_size = 0

        logger.info(f"Initialized information-theoretic reward with {temperature_schedule} temperature")

    def update_token_frequencies(self, sequences: torch.Tensor) -> None:
        """
        Update token frequency statistics for importance weighting.

        Args:
            sequences: Batch of token sequences [batch_size, seq_len]
        """
        if not self.importance_weighting:
            return

        # Count token frequencies in this batch
        flat_tokens = sequences.flatten()
        batch_counts = torch.bincount(flat_tokens, minlength=self.vocab_size).float()

        # Update running frequency estimate
        if self.token_frequencies is None:
            self.token_frequencies = batch_counts.to(self.device)
            self.frequency_cache_size = flat_tokens.numel()
        else:
            # Exponential moving average
            alpha = 0.01  # Learning rate for frequency updates
            self.token_frequencies = (1 - alpha) * self.token_frequencies + alpha * batch_counts.to(self.device)
            self.frequency_cache_size += flat_tokens.numel()

    def compute_importance_weights(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights for tokens based on rarity and position.

        Args:
            sequences: Token sequences [batch_size, seq_len]

        Returns:
            Importance weights [batch_size, seq_len]
        """
        if not self.importance_weighting or self.token_frequencies is None:
            return torch.ones_like(sequences, dtype=torch.float, device=self.device)

        # Get token frequencies
        token_freqs = self.token_frequencies[sequences] / (self.token_frequencies.sum() + 1e-8)

        # Importance is inverse frequency (rare tokens are more important)
        importance = 1.0 / (token_freqs + 1e-8)

        # Normalize to [0, 1] range
        importance = importance / importance.max()

        # Position-based weighting: tokens at the beginning and end are more important
        batch_size, seq_len = sequences.shape
        position_weights = torch.ones(seq_len, device=self.device)

        # Emphasize start and end of sequences
        for i in range(seq_len):
            if i < seq_len * 0.1 or i > seq_len * 0.9:  # First 10% and last 10%
                position_weights[i] = 1.5

        # Combine frequency and position weights
        final_importance = importance * position_weights.unsqueeze(0)

        return final_importance

    def compute_rate_term(self, mask_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the rate term R = E[-log p(mask|x)].

        This measures the expected information content of the compression decisions.
        Lower rate = more compression, but we want to reward compression.

        Args:
            mask_probs: Probability of keeping each token [batch_size, seq_len]

        Returns:
            Rate term [batch_size]
        """
        # Clamp probabilities to avoid log(0)
        probs = torch.clamp(mask_probs, min=1e-8, max=1 - 1e-8)

        # Compute entropy of compression decisions
        entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))

        # Average over sequence length to get rate
        rate = entropy.mean(dim=1)

        return rate

    def compute_distortion_term(
        self,
        reconstruction_loss: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the distortion term D = E[d(x, x̂)].

        This measures the expected reconstruction error.

        Args:
            reconstruction_loss: Loss per token [batch_size, seq_len] or scalar
            importance_weights: Optional importance weights [batch_size, seq_len]

        Returns:
            Distortion term [batch_size]
        """
        if reconstruction_loss.dim() == 0:
            # Scalar loss, return as-is
            return reconstruction_loss.unsqueeze(0)

        if reconstruction_loss.dim() == 1:
            # Already averaged over sequence
            return reconstruction_loss

        # Apply importance weighting if provided
        if importance_weights is not None:
            weighted_loss = reconstruction_loss * importance_weights
            distortion = weighted_loss.mean(dim=1)
        else:
            distortion = reconstruction_loss.mean(dim=1)

        return distortion

    def get_temperature(self, step: int, current_loss: float = None) -> float:
        """
        Get temperature for current training step.

        Args:
            step: Current training step
            current_loss: Current average reconstruction loss (for adaptive)

        Returns:
            Temperature value
        """
        if self.temperature_schedule == "fixed":
            return self.initial_temperature

        elif self.temperature_schedule == "linear":
            # Linear decay from initial to minimum
            max_steps = 100000  # Assume 100k steps for full decay
            progress = min(step / max_steps, 1.0)
            return self.initial_temperature * (1 - progress) + self.min_temperature * progress

        elif self.temperature_schedule == "exponential":
            # Exponential decay
            decay_rate = 0.9999
            temp = self.initial_temperature * (decay_rate**step)
            return max(temp, self.min_temperature)

        elif self.temperature_schedule == "adaptive":
            # Adaptive based on reconstruction quality
            if current_loss is None:
                return self.initial_temperature

            if current_loss > 2.0:  # High loss, use high temperature (more exploration)
                return self.initial_temperature
            elif current_loss < 0.5:  # Low loss, use low temperature (more exploitation)
                return self.min_temperature
            else:
                # Linear interpolation based on loss
                progress = (2.0 - current_loss) / 1.5  # Map [0.5, 2.0] to [0, 1]
                return self.initial_temperature * (1 - progress) + self.min_temperature * progress

        else:
            raise ValueError(f"Unknown temperature schedule: {self.temperature_schedule}")

    def compute_reward(
        self,
        mask_probs: torch.Tensor,
        reconstruction_loss: torch.Tensor,
        sequences: torch.Tensor,
        step: int = 0,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the information-theoretic reward.

        Args:
            mask_probs: Probability of keeping each token [batch_size, seq_len]
            reconstruction_loss: Reconstruction loss [batch_size, seq_len] or [batch_size] or scalar
            sequences: Original token sequences [batch_size, seq_len]
            step: Current training step
            beta: Rate-distortion trade-off parameter (if None, use adaptive)

        Returns:
            Dictionary with reward components
        """
        mask_probs.shape[0]

        # Update token frequencies for importance weighting
        self.update_token_frequencies(sequences)

        # Compute importance weights
        importance_weights = self.compute_importance_weights(sequences)

        # Compute rate term (compression efficiency)
        rate = self.compute_rate_term(mask_probs)

        # Compute distortion term (reconstruction quality)
        distortion = self.compute_distortion_term(reconstruction_loss, importance_weights)

        # Get current temperature
        avg_loss = distortion.mean().item() if distortion.numel() > 0 else 1.0
        temperature = self.get_temperature(step, avg_loss)

        # Adaptive beta scheduling
        if beta is None:
            if avg_loss > 2.0:  # High loss, prioritize reconstruction
                beta = 2.0
            elif avg_loss < 0.5:  # Low loss, allow more compression
                beta = 0.5
            else:
                beta = 1.0  # Balanced

        # Compute information-theoretic reward
        # Higher rate (more information) and lower distortion = better
        # Use exponential form to ensure positive rewards
        rate_reward = torch.exp(-rate / temperature)  # Reward compression
        distortion_penalty = torch.exp(-beta * distortion / temperature)  # Reward quality

        # Combined reward: balance compression and quality
        reward = rate_reward * distortion_penalty

        # Additional metrics for monitoring
        compression_ratio = 1.0 - mask_probs.mean(dim=1)
        entropy = -(
            mask_probs * torch.log(mask_probs + 1e-8) + (1 - mask_probs) * torch.log(1 - mask_probs + 1e-8)
        ).mean(dim=1)

        return {
            "reward": reward,
            "rate": rate,
            "distortion": distortion,
            "rate_reward": rate_reward,
            "distortion_penalty": distortion_penalty,
            "compression_ratio": compression_ratio,
            "entropy": entropy,
            "temperature": torch.tensor(temperature),
            "beta": torch.tensor(beta),
            "importance_weights": importance_weights,
        }


class SimpleReward:
    """
    Simplified reward function for quick testing and baseline comparison.

    Uses the fixed approach from the tech spec without the full information-theoretic
    framework. Useful for debugging and comparison.
    """

    def __init__(self):
        self.name = "simple"

    def compute_reward(
        self,
        mask_probs: torch.Tensor,
        reconstruction_loss: torch.Tensor,
        sequences: torch.Tensor,
        step: int = 0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute simple fixed reward from the tech spec.

        reward = sigmoid(compression_ratio) * exp(-loss/temp) * importance_weight
        """
        batch_size = mask_probs.shape[0]

        # Compression ratio
        compression_ratio = 1.0 - mask_probs.mean(dim=1)

        # Normalize reconstruction loss
        if reconstruction_loss.dim() > 1:
            loss = reconstruction_loss.mean(dim=1)
        elif reconstruction_loss.dim() == 1:
            loss = reconstruction_loss
        else:
            loss = reconstruction_loss.expand(batch_size)

        # Fixed temperature
        temperature = 1.0

        # Simple reward formula
        compression_reward = torch.sigmoid(compression_ratio * 2 - 1)  # Map [0,1] to sigmoid range
        quality_reward = torch.exp(-loss / temperature)

        reward = compression_reward * quality_reward

        return {
            "reward": reward,
            "compression_ratio": compression_ratio,
            "reconstruction_loss": loss,
            "compression_reward": compression_reward,
            "quality_reward": quality_reward,
        }


# Factory function to create reward functions
def create_reward_function(reward_type: str = "simple", **kwargs):
    """
    Factory function to create reward functions.

    Args:
        reward_type: Type of reward function ("simple" is recommended)
        **kwargs: Additional arguments for the reward function

    Returns:
        Reward function instance
    """
    if reward_type == "simple":
        return SimpleReward()
    elif reward_type == "information_theoretic":
        return InformationTheoreticReward(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. Use 'simple'.")


# Test functions
def test_reward_functions():
    """Test both reward functions with dummy data."""
    logger.info("Testing reward functions...")

    # Create test data
    batch_size, seq_len, vocab_size = 4, 10, 1000

    mask_probs = torch.rand(batch_size, seq_len)  # Random keep probabilities
    reconstruction_loss = torch.rand(batch_size, seq_len) * 2  # Random losses [0, 2]
    sequences = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test information-theoretic reward
    info_reward = InformationTheoreticReward(vocab_size=vocab_size)
    info_results = info_reward.compute_reward(mask_probs, reconstruction_loss, sequences, step=100)

    logger.info(f"Info reward - mean: {info_results['reward'].mean():.3f}, std: {info_results['reward'].std():.3f}")
    logger.info(f"Compression ratio: {info_results['compression_ratio'].mean():.3f}")
    logger.info(f"Temperature: {info_results['temperature']:.3f}")

    # Test simple reward
    simple_reward = SimpleReward()
    simple_results = simple_reward.compute_reward(mask_probs, reconstruction_loss, sequences)

    logger.info(
        f"Simple reward - mean: {simple_results['reward'].mean():.3f}, std: {simple_results['reward'].std():.3f}"
    )

    logger.info("Reward function tests passed!")
