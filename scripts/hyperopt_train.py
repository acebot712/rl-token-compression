#!/usr/bin/env python3
"""
Hyperparameter optimization for RL Token Compression.

Usage:
    python scripts/hyperopt_train.py --config configs/training/default.json --max-trials 50
"""

import argparse
import sys

from training.train import train as base_train_function
from utils.config import setup_validated_config
from utils.errors import handle_errors
from utils.hyperopt import run_hyperparameter_optimization


@handle_errors("hyperparameter optimization")
def main():
    """Main hyperparameter optimization function."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for RL Token Compression")

    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Base configuration file")

    # Hyperopt arguments
    parser.add_argument("--max-trials", type=int, default=20, help="Maximum number of trials")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hyperopt",
        help="Output directory for optimization results",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel trials (not fully implemented)",
    )

    # Early stopping
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="Minimum improvement threshold",
    )

    args = parser.parse_args()

    print("🔍 RL Token Compression - Hyperparameter Optimization")
    print(f"   Base config: {args.config}")
    print(f"   Max trials: {args.max_trials}")
    print(f"   Output directory: {args.output_dir}")
    print()

    # Load base configuration
    default_config = {
        "data_path": "data/processed/processed_data.json",
        "output_dir": "results/joint_training",
        "reconstructor_path": "models/reconstructor/fine-tuned/checkpoint-5000",
        "val_data_path": None,
        "max_epochs": 50,
        "batch_size": 16,
        "learning_rate_policy": 3e-4,
        "learning_rate_reconstructor": 1e-4,
        "context_window": 5,
        "reward_type": "simple",
        "device": "auto",
        "resume": False,
        "debug": False,
    }

    # Override with command line config
    import json

    with open(args.config) as f:
        file_config = json.load(f)

    base_config = default_config.copy()
    base_config.update(file_config)
    base_config = setup_validated_config(base_config, "Hyperparameter Optimization", "training")

    print("Base configuration loaded and validated ✓")
    print()

    # Run hyperparameter optimization
    best_config = run_hyperparameter_optimization(
        base_config=base_config,
        train_function=base_train_function,
        max_trials=args.max_trials,
        output_dir=args.output_dir,
    )

    if best_config:
        print("\n🎉 Hyperparameter optimization complete!")
        print(f"   Best configuration saved to {args.output_dir}/best_config.json")
        print("\n   To use the optimized config:")
        print(f"   python training/train.py --config {args.output_dir}/best_config.json")
    else:
        print("\n❌ Hyperparameter optimization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
