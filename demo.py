#!/usr/bin/env python3
"""
Quick demonstration of RL-based token compression.

This script demonstrates the baseline compression methods on sample text.
Complete end-to-end demo with trained RL agent requires running training first.

Usage:
    python demo.py

Expected runtime: < 5 seconds
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.baselines import (
    RandomBaseline,
    FrequencyBaseline,
    LengthBaseline,
    PositionBaseline,
)

def demo_compression():
    """Demonstrate token compression on sample text."""

    # Sample text
    sample_text = """
    Machine learning is transforming how we process and understand text data.
    Modern language models can generate human-like text, translate between languages,
    and answer complex questions. However, these models are computationally expensive.
    Token compression techniques aim to reduce inference costs while preserving semantic meaning.
    This research explores using reinforcement learning to learn optimal compression strategies.
    """

    # Tokenize (simple word-based for demo)
    tokens = sample_text.split()
    tokens = [t.strip() for t in tokens if t.strip()]

    print("=" * 70)
    print("TOKEN COMPRESSION DEMO")
    print("=" * 70)
    print(f"\nOriginal text ({len(tokens)} tokens):")
    print(" ".join(tokens))
    print()

    # Target compression ratio
    target_ratio = 0.5  # Keep 50% of tokens

    # Create baselines
    baselines = [
        ("Random", RandomBaseline()),
        ("Length", LengthBaseline()),
        ("Position", PositionBaseline()),
    ]

    print(f"Target compression: Keep {target_ratio*100:.0f}% of tokens")
    print("-" * 70)

    # Run each baseline
    for name, baseline in baselines:
        mask = baseline.compress(tokens, target_ratio)
        compressed_tokens = [t for t, keep in zip(tokens, mask) if keep]
        compression_achieved = len(compressed_tokens) / len(tokens)

        print(f"\n{name} Baseline:")
        print(f"  Tokens kept: {len(compressed_tokens)}/{len(tokens)} ({compression_achieved*100:.1f}%)")
        print(f"  Compressed: {' '.join(compressed_tokens)}")

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nTo train and evaluate the full RL agent:")
    print("  1. python data/prepare.py --config configs/data/sample.json")
    print("  2. python training/train.py --config configs/training/default.json")
    print("  3. python evaluation/evaluate.py --config configs/evaluation/default.json")
    print()
    print("For baseline evaluation on processed data:")
    print("  python evaluation/evaluate.py --config configs/evaluation/baseline_only.json")
    print()

if __name__ == "__main__":
    demo_compression()
