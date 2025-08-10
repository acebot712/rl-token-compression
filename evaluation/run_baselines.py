#!/usr/bin/env python3
"""
Baseline evaluation script with CLI and config file support.

This script runs baseline compression methods for comparison with
the trained RL model.

Usage:
    # Quick baseline test
    python eval/baselines_main.py --data_path data/test.json --output_dir baseline_results
    
    # With config file
    python eval/baselines_main.py --config configs/prod/baselines.json
    
    # Specific baselines only
    python eval/baselines_main.py --data_path data/test.json --baselines random frequency
"""

import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
from typing import List, Dict, Any

from utils.config import setup_config, save_config
from utils.common import setup_output_dir, print_section_header, print_config_summary, handle_common_errors, print_summary_table
from evaluation.baselines import create_baseline, evaluate_baselines, load_corpus_for_baselines


@handle_common_errors
def run_baselines(config):
    """Run baseline compression methods evaluation."""
    print_section_header("BASELINE COMPRESSION EVALUATION")
    print_config_summary(config)
    print()
    
    setup_output_dir(config['output_dir'])
    
    # Load test data
    test_data, tokenizer = load_corpus_for_baselines(config['data_path'])
    print(f"Loaded {len(test_data)} sequences")
    
    if len(test_data) > config['num_sequences']:
        test_data = test_data[:config['num_sequences']]
        print(f"Limited to {config['num_sequences']} sequences for evaluation")
    
    # Create and evaluate baselines
    all_results = {}
    for baseline_name in config['baselines']:
        print(f"\nEvaluating {baseline_name}...")
        start_time = time.time()
        
        try:
            baseline = create_baseline(baseline_name)
            results = evaluate_baselines({baseline_name: baseline}, test_data)
            end_time = time.time()
            
            # Add timing
            results[baseline_name]['evaluation_time'] = end_time - start_time
            results[baseline_name]['sequences_per_second'] = len(test_data) / (end_time - start_time)
            all_results[baseline_name] = results[baseline_name]
            
            # Print progress
            comp_ratio = results[baseline_name].get('compression_ratio', 'N/A')
            quality = results[baseline_name].get('reconstruction_quality', 'N/A')
            speed = results[baseline_name].get('sequences_per_second', 'N/A')            
            print(f"  Results: compression={comp_ratio:.3f}, quality={quality}, speed={speed:.1f} seq/s")
            
        except Exception as e:
            print(f"  Failed: {e}")
            all_results[baseline_name] = {"error": str(e)}
    
    # Save results
    results_path = os.path.join(config['output_dir'], "baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save config and show summary
    config_path = os.path.join(config['output_dir'], "baseline_config.json")
    save_config(config, config_path)
    
    # Print summary
    clean_results = {k: v for k, v in all_results.items() if "error" not in v}
    if clean_results:
        print_summary_table(
            clean_results, 
            ['compression_ratio', 'reconstruction_quality', 'sequences_per_second'], 
            "Baseline Evaluation Summary"
        )
    
    print("\nBaseline evaluation complete!")


if __name__ == "__main__":
    default_config = {
        "data_path": "data/processed/test_data.json",
        "output_dir": "eval/baseline_results",
        "num_sequences": 1000,
        "baselines": ["random", "frequency", "length", "position", "entropy"],
        "compression_targets": [0.3, 0.5, 0.7],
        "random_seed": 42
    }
    
    config = setup_config(default_config, "Baseline Compression Evaluation")
    run_baselines(config)