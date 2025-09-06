"""
Baseline compression methods for comparison.

These are the simple, non-ML baselines that any reasonable RL system should beat.
If your fancy RL approach can't beat these, it's probably broken.

Baselines implemented:
1. RandomBaseline - Random token removal (sanity check)
2. FrequencyBaseline - Remove rare tokens based on corpus frequency
3. LengthBaseline - Remove longest tokens first
4. EntropyBaseline - Remove tokens with lowest estimated entropy
5. PositionBaseline - Remove tokens from middle of sequence

This is not just academic completeness - these baselines will likely be
surprisingly competitive and will keep you honest about whether your
RL approach is actually useful.
"""

import json
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set up logging
from utils.logging import get_component_logger

logger = get_component_logger("BASELINES")


class CompressionBaseline(ABC):
    """Abstract base class for compression baselines."""

    def __init__(self, name: str):
        self.name = name
        self.stats = defaultdict(list)

    @abstractmethod
    def compress(self, tokens: List[str], target_ratio: float) -> List[bool]:
        """
        Return boolean mask indicating which tokens to keep.

        Args:
            tokens: List of token strings
            target_ratio: Target compression ratio (0.5 = keep 50% of tokens)

        Returns:
            Boolean mask (True = keep, False = mask)
        """
        pass

    def get_compression_stats(self) -> Dict[str, float]:
        """Return interpretable statistics about compression strategy."""
        if not self.stats:
            return {}

        return {
            f"{self.name}_mean_ratio": np.mean(self.stats["compression_ratios"]),
            f"{self.name}_std_ratio": np.std(self.stats["compression_ratios"]),
            f"{self.name}_num_sequences": len(self.stats["compression_ratios"]),
        }

    def _record_stats(self, tokens: List[str], mask: List[bool]):
        """Record statistics for this compression."""
        kept_tokens = sum(mask)
        compression_ratio = kept_tokens / len(tokens)
        self.stats["compression_ratios"].append(compression_ratio)
        self.stats["sequence_lengths"].append(len(tokens))
        self.stats["kept_tokens"].append(kept_tokens)


class RandomBaseline(CompressionBaseline):
    """
    Random token removal baseline.

    This is the absolute minimum baseline. If your RL system can't beat this,
    something is fundamentally wrong.
    """

    def __init__(self, seed: int = 42):
        super().__init__("random")
        self.rng = np.random.RandomState(seed)

    def compress(self, tokens: List[str], target_ratio: float) -> List[bool]:
        """Randomly select tokens to keep."""
        n_tokens = len(tokens)
        n_keep = int(n_tokens * target_ratio)

        # Randomly select which tokens to keep
        indices = self.rng.choice(n_tokens, size=n_keep, replace=False)
        mask = [False] * n_tokens
        for idx in indices:
            mask[idx] = True

        self._record_stats(tokens, mask)
        return mask


class FrequencyBaseline(CompressionBaseline):
    """
    Remove rare tokens based on corpus frequency.

    This is often surprisingly effective. Rare tokens contribute less to
    overall meaning and are harder to predict anyway.
    """

    def __init__(self, corpus_path: Optional[str] = None):
        super().__init__("frequency")
        self.token_frequencies = {}
        self.corpus_loaded = False

        if corpus_path:
            self.load_corpus_frequencies(corpus_path)

    def load_corpus_frequencies(self, corpus_path: str):
        """Load token frequencies from corpus."""
        logger.info(f"Loading corpus frequencies from {corpus_path}")

        try:
            with open(corpus_path) as f:
                data = json.load(f)

            # Count token frequencies across all sequences
            all_tokens = []
            for item in data:
                if "tokens" in item:
                    all_tokens.extend(item["tokens"])
                elif "text" in item:
                    # If we have text, split by spaces (simple tokenization)
                    all_tokens.extend(item["text"].split())

            # Convert to frequency dictionary
            token_counts = Counter(all_tokens)
            total_tokens = sum(token_counts.values())

            self.token_frequencies = {token: count / total_tokens for token, count in token_counts.items()}

            self.corpus_loaded = True
            logger.info(f"Loaded frequencies for {len(self.token_frequencies)} unique tokens")

        except Exception as e:
            logger.warning(f"Failed to load corpus frequencies: {e}")
            logger.warning("Will use uniform frequencies as fallback")

    def get_token_frequency(self, token: str) -> float:
        """Get frequency of a token (with fallback for unknown tokens)."""
        if not self.corpus_loaded:
            return 0.5  # Default frequency for unknown tokens

        return self.token_frequencies.get(token, 1e-6)  # Very low freq for unknown

    def compress(self, tokens: List[str], target_ratio: float) -> List[bool]:
        """Keep most frequent tokens."""
        n_tokens = len(tokens)
        n_keep = int(n_tokens * target_ratio)

        # Get frequency scores for each token
        frequencies = [self.get_token_frequency(token) for token in tokens]

        # Sort by frequency (descending) and keep top n_keep
        sorted_indices = sorted(range(n_tokens), key=lambda i: frequencies[i], reverse=True)

        mask = [False] * n_tokens
        for i in range(n_keep):
            if i < len(sorted_indices):
                mask[sorted_indices[i]] = True

        self._record_stats(tokens, mask)
        return mask


class LengthBaseline(CompressionBaseline):
    """
    Remove longest tokens first.

    Motivation: Longer tokens often carry less information per character
    and removing them gives better compression ratios.
    """

    def __init__(self):
        super().__init__("length")

    def compress(self, tokens: List[str], target_ratio: float) -> List[bool]:
        """Keep shortest tokens."""
        n_tokens = len(tokens)
        n_keep = int(n_tokens * target_ratio)

        # Sort by token length (ascending) and keep shortest
        sorted_indices = sorted(range(n_tokens), key=lambda i: len(tokens[i]))

        mask = [False] * n_tokens
        for i in range(n_keep):
            if i < len(sorted_indices):
                mask[sorted_indices[i]] = True

        self._record_stats(tokens, mask)
        return mask


class EntropyBaseline(CompressionBaseline):
    """
    Remove tokens with lowest estimated entropy.

    This is more sophisticated - we estimate the information content of each
    token based on its context and remove the least informative ones.
    """

    def __init__(self, context_window: int = 3):
        super().__init__("entropy")
        self.context_window = context_window
        self.n_gram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.trained = False

    def train_on_corpus(self, corpus_data: List[List[str]]):
        """Train n-gram model on corpus for entropy estimation."""
        logger.info(f"Training {self.context_window}-gram model on {len(corpus_data)} sequences")

        for tokens in corpus_data:
            # Add padding
            padded_tokens = ["<PAD>"] * self.context_window + tokens + ["<PAD>"] * self.context_window

            # Count n-grams
            for i in range(len(padded_tokens) - self.context_window):
                context = tuple(padded_tokens[i : i + self.context_window])
                target = padded_tokens[i + self.context_window]

                self.context_counts[context] += 1
                self.n_gram_counts[(context, target)] += 1

        self.trained = True
        logger.info(f"Trained model with {len(self.context_counts)} contexts")

    def estimate_token_entropy(self, tokens: List[str], position: int) -> float:
        """Estimate entropy of token at given position."""
        if not self.trained:
            return 1.0  # Default entropy for untrained model

        # Get context
        start = max(0, position - self.context_window)
        end = min(len(tokens), position)

        context_tokens = ["<PAD>"] * (self.context_window - (end - start)) + tokens[start:end]
        context = tuple(context_tokens)

        # Get token
        token = tokens[position] if position < len(tokens) else "<PAD>"

        # Estimate probability
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 1.0  # High entropy for unknown contexts

        n_gram_count = self.n_gram_counts.get((context, token), 0)
        prob = (n_gram_count + 1) / (context_count + len(self.n_gram_counts))  # Laplace smoothing

        # Return negative log probability (entropy)
        return -math.log(prob)

    def compress(self, tokens: List[str], target_ratio: float) -> List[bool]:
        """Keep tokens with highest entropy (most informative)."""
        n_tokens = len(tokens)
        n_keep = int(n_tokens * target_ratio)

        if not self.trained:
            logger.warning("Entropy model not trained, using uniform entropy")
            entropies = [1.0] * n_tokens
        else:
            entropies = [self.estimate_token_entropy(tokens, i) for i in range(n_tokens)]

        # Sort by entropy (descending) and keep highest entropy tokens
        sorted_indices = sorted(range(n_tokens), key=lambda i: entropies[i], reverse=True)

        mask = [False] * n_tokens
        for i in range(n_keep):
            if i < len(sorted_indices):
                mask[sorted_indices[i]] = True

        self._record_stats(tokens, mask)
        return mask


class PositionBaseline(CompressionBaseline):
    """
    Remove tokens from middle of sequence.

    Motivation: Beginning and end of sequences often contain the most
    important information (topic, conclusion).
    """

    def __init__(self):
        super().__init__("position")

    def compress(self, tokens: List[str], target_ratio: float) -> List[bool]:
        """Keep tokens from beginning and end of sequence."""
        n_tokens = len(tokens)
        n_keep = int(n_tokens * target_ratio)

        # Calculate how many to keep from start and end
        n_start = n_keep // 2
        n_end = n_keep - n_start

        mask = [False] * n_tokens

        # Keep tokens from start
        for i in range(min(n_start, n_tokens)):
            mask[i] = True

        # Keep tokens from end
        for i in range(max(0, n_tokens - n_end), n_tokens):
            mask[i] = True

        self._record_stats(tokens, mask)
        return mask


class AdaptiveBaseline(CompressionBaseline):
    """
    Adaptive baseline that combines multiple strategies.

    This is what a reasonable heuristic approach might look like.
    Your RL system should definitely beat this.
    """

    def __init__(self, corpus_path: Optional[str] = None):
        super().__init__("adaptive")

        # Initialize component baselines
        self.frequency_baseline = FrequencyBaseline(corpus_path)
        self.length_baseline = LengthBaseline()
        self.position_baseline = PositionBaseline()
        self.entropy_baseline = EntropyBaseline()

    def train_on_corpus(self, corpus_data: List[List[str]]):
        """Train components that need training."""
        self.entropy_baseline.train_on_corpus(corpus_data)

    def compress(self, tokens: List[str], target_ratio: float) -> List[bool]:
        """Combine multiple strategies with weighted voting."""
        n_tokens = len(tokens)

        # Get scores from different baselines
        freq_mask = self.frequency_baseline.compress(tokens, 1.0)  # Get all scores
        length_mask = self.length_baseline.compress(tokens, 1.0)
        position_mask = self.position_baseline.compress(tokens, 1.0)
        entropy_mask = self.entropy_baseline.compress(tokens, 1.0)

        # Combine scores with weights
        scores = np.zeros(n_tokens)
        for i in range(n_tokens):
            scores[i] = (
                0.3 * float(freq_mask[i])  # Frequency weight
                + 0.2 * float(length_mask[i])  # Length weight
                + 0.2 * float(position_mask[i])  # Position weight
                + 0.3 * float(entropy_mask[i])  # Entropy weight
            )

        # Select top tokens
        n_keep = int(n_tokens * target_ratio)
        sorted_indices = sorted(range(n_tokens), key=lambda i: scores[i], reverse=True)

        mask = [False] * n_tokens
        for i in range(n_keep):
            if i < len(sorted_indices):
                mask[sorted_indices[i]] = True

        self._record_stats(tokens, mask)
        return mask


# Factory function and utilities


def create_baseline(baseline_type: str, **kwargs) -> CompressionBaseline:
    """Factory function to create baseline methods."""
    baselines = {
        "random": RandomBaseline,
        "frequency": FrequencyBaseline,
        "length": LengthBaseline,
        "entropy": EntropyBaseline,
        "position": PositionBaseline,
        "adaptive": AdaptiveBaseline,
    }

    if baseline_type not in baselines:
        raise ValueError(f"Unknown baseline type: {baseline_type}. Available: {list(baselines.keys())}")

    return baselines[baseline_type](**kwargs)


def evaluate_baselines(
    baselines: List[CompressionBaseline],
    test_data: List[List[str]],
    target_ratios: List[float] = [0.3, 0.5, 0.7],
    compute_quality: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate multiple baselines on test data.

    Args:
        baselines: List of baseline methods
        test_data: List of token sequences
        target_ratios: List of compression ratios to test
        compute_quality: Whether to compute reconstruction quality metrics

    Returns:
        Dictionary with evaluation results
    """
    results = {}

    logger.info(f"Evaluating {len(baselines)} baselines on {len(test_data)} sequences")

    # Initialize quality metrics if needed
    quality_metrics = None
    if compute_quality:
        try:
            from evaluation.evaluator import CompressionMetrics

            quality_metrics = CompressionMetrics()
        except ImportError:
            logger.warning("Could not import quality metrics, reconstruction quality will be disabled")
            compute_quality = False

    for baseline in baselines:
        baseline_results = {}

        for ratio in target_ratios:
            compression_ratios = []
            reconstruction_qualities = []

            for tokens in test_data:
                if isinstance(tokens[0], int):
                    # Convert token IDs to strings for baseline processing
                    tokens = [str(t) for t in tokens]

                # Get compression mask
                mask = baseline.compress(tokens, ratio)
                actual_ratio = sum(mask) / len(mask)
                compression_ratios.append(actual_ratio)

                # Compute reconstruction quality if enabled
                if compute_quality and quality_metrics:
                    # Get compressed tokens
                    compressed_tokens = [token for token, keep in zip(tokens, mask) if keep]

                    # Compute BLEU score as reconstruction quality
                    bleu_score = quality_metrics.compute_bleu(tokens, compressed_tokens)
                    reconstruction_qualities.append(bleu_score)

            # Store results
            ratio_results = {
                "mean": np.mean(compression_ratios),
                "std": np.std(compression_ratios),
                "target": ratio,
            }

            if compute_quality and reconstruction_qualities:
                ratio_results["reconstruction_quality"] = {
                    "mean": np.mean(reconstruction_qualities),
                    "std": np.std(reconstruction_qualities),
                    "values": reconstruction_qualities,
                }

            baseline_results[f"ratio_{ratio}"] = ratio_results

        # Add overall statistics
        baseline_results["stats"] = baseline.get_compression_stats()
        results[baseline.name] = baseline_results

    return results


def load_corpus_for_baselines(data_path: str) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Load corpus data and prepare it for baseline training."""
    with open(data_path) as f:
        data = json.load(f)

    sequences = []
    metadata = {"total_sequences": len(data), "total_tokens": 0}

    for item in data:
        if "text" in item:
            # Prefer raw text for meaningful baseline evaluation
            tokens = item["text"].split()
        elif "tokens" in item:
            # Token IDs - convert to strings (fallback)
            tokens = [str(t) for t in item["tokens"]]
        else:
            continue

        sequences.append(tokens)
        metadata["total_tokens"] += len(tokens)

    metadata["avg_sequence_length"] = metadata["total_tokens"] / len(sequences)

    logger.info(f"Loaded {len(sequences)} sequences, {metadata['total_tokens']} total tokens")
    return sequences, metadata


# Test and demonstration functions


def test_baselines():
    """Test all baseline methods with sample data."""
    logger.info("Testing baseline compression methods...")

    # Sample data
    test_sequences = [
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        [
            "machine",
            "learning",
            "is",
            "a",
            "subset",
            "of",
            "artificial",
            "intelligence",
        ],
        ["neural", "networks", "learn", "from", "data", "to", "make", "predictions"],
        [
            "this",
            "is",
            "a",
            "longer",
            "sequence",
            "with",
            "more",
            "tokens",
            "for",
            "testing",
            "purposes",
        ],
    ]

    # Create baselines
    baselines = [
        RandomBaseline(seed=42),
        FrequencyBaseline(),
        LengthBaseline(),
        EntropyBaseline(),
        PositionBaseline(),
    ]

    # Train entropy baseline
    baselines[3].train_on_corpus(test_sequences)

    # Test compression
    target_ratio = 0.5

    for baseline in baselines:
        logger.info(f"\nTesting {baseline.name} baseline:")

        for i, tokens in enumerate(test_sequences[:2]):  # Test first 2 sequences
            mask = baseline.compress(tokens, target_ratio)
            kept_tokens = [token for token, keep in zip(tokens, mask) if keep]

            logger.info(f"  Sequence {i + 1}: {len(tokens)} -> {len(kept_tokens)} tokens")
            logger.info(f"    Original: {' '.join(tokens)}")
            logger.info(f"    Compressed: {' '.join(kept_tokens)}")

    logger.info("Baseline tests completed successfully!")
