#!/usr/bin/env python3
"""
Main evaluation script with comprehensive CLI and config file support.

This script evaluates trained token compression models against baselines
with rigorous statistical testing and multiple quality metrics.

Usage:
    # With config file
    python eval/eval_main.py --config configs/evaluation.json
    
    # With CLI overrides
    python eval/eval_main.py --config configs/evaluation.json --num_sequences 500
    
    # Pure CLI
    python eval/eval_main.py --model_path results/best_model.pt --data_path data/test.json
"""

import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import json
import numpy as np
from typing import Dict, List, Any

from utils.config import setup_config, save_config, resolve_device
from utils.common import setup_output_dir, print_section_header, print_config_summary, handle_common_errors, print_summary_table
from eval.evaluation import run_full_evaluation, EvaluationConfig
from eval.baselines import create_baseline, evaluate_baselines, load_corpus_for_baselines


def run_baseline_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run baseline evaluations and return results."""
    print("Running baseline evaluations...")
    
    # Load test data
    test_data, tokenizer = load_corpus_for_baselines(config['data_path'])
    
    if len(test_data) > config['num_sequences']:
        test_data = test_data[:config['num_sequences']]
        print(f"Limited to {config['num_sequences']} sequences for evaluation")
    
    # Create and evaluate baselines
    baseline_results = {}
    for baseline_name in config['baselines']:
        print(f"Evaluating {baseline_name} baseline...")
        try:
            baseline = create_baseline(baseline_name)
            results = evaluate_baselines({baseline_name: baseline}, test_data)
            baseline_results[baseline_name] = results[baseline_name]
            
            comp_ratio = results[baseline_name].get('compression_ratio', 'N/A')
            quality = results[baseline_name].get('reconstruction_quality', 'N/A')
            print(f"  {baseline_name}: compression={comp_ratio:.3f}, quality={quality}")
            
        except Exception as e:
            print(f"  Failed to evaluate {baseline_name}: {e}")
            baseline_results[baseline_name] = {"error": str(e)}
    
    # Save results
    baseline_path = os.path.join(config['output_dir'], "baseline_results.json")
    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print(f"Baseline results saved to: {baseline_path}")
    
    return baseline_results


@handle_common_errors
def evaluate_model(config):
    """Run comprehensive evaluation of token compression model."""
    print_section_header("TOKEN COMPRESSION EVALUATION")
    print_config_summary(config)
    print()
    
    # Setup
    setup_output_dir(config['output_dir'])
    config['device'] = resolve_device(config['device'])
    print(f"Using device: {config['device']}")
    
    baseline_results = {}
    if config['include_baselines']:
        baseline_results = run_baseline_evaluation(config)
    
    # Run model evaluation if available
    model_results = {}
    if config.get('model_path'):
        model_results = run_model_evaluation(config)
    
    # Save config and print summary
    config_path = os.path.join(config['output_dir'], "evaluation_config.json")
    save_config(config, config_path)
    
    # Print summary
    print_evaluation_summary(baseline_results, model_results, config['output_dir'])


def run_model_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run model evaluation if model path is provided."""
    print("Running trained model evaluation...")
    
    try:
        eval_config = EvaluationConfig(
            n_seeds=config.get('n_seeds', 3),
            significance_level=config.get('significance_level', 0.05),
            target_ratios=config.get('target_ratios', [0.3, 0.5, 0.7]),
            compute_bleu=config.get('compute_bleu', True),
            compute_rouge=config.get('compute_rouge', True),
            compute_perplexity=config.get('compute_perplexity', True),
            measure_speed=config.get('measure_speed', True),
            measure_memory=config.get('measure_memory', True),
            save_plots=config.get('save_plots', True),
            output_dir=config['output_dir']
        )
        
        model_results = run_full_evaluation(
            model_path=config['model_path'],
            data_path=config['data_path'],
            reconstructor_path=config['reconstructor_path'],
            eval_config=eval_config,
            device=config['device']
        )
        
        model_path = os.path.join(config['output_dir'], "model_results.json")
        with open(model_path, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f"Model results saved to: {model_path}")
        
        return model_results
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return {"error": str(e)}


def print_evaluation_summary(baseline_results: Dict[str, Any], model_results: Dict[str, Any], output_dir: str):
    """Print evaluation summary."""
    print_section_header("EVALUATION COMPLETE")
    
    if baseline_results:
        # Filter out error results for summary
        clean_results = {k: v for k, v in baseline_results.items() if "error" not in v}
        if clean_results:
            print_summary_table(
                clean_results, 
                ['compression_ratio', 'reconstruction_quality'], 
                "Baseline Results"
            )
    
    print(f"\nDetailed results saved to: {output_dir}")
    print(f"Check {output_dir}/ for plots and detailed analysis.")


if __name__ == "__main__":
    default_config = {
        "model_path": "results/joint_training/best_model.pt",
        "data_path": "data/processed/test_data.json",
        "output_dir": "eval/results",
        "reconstructor_path": "models/reconstructor/fine-tuned/checkpoint-5000",
        "num_sequences": 1000,
        "include_baselines": True,
        "baselines": ["random", "frequency", "length", "position"],
        "device": "auto",
        "n_seeds": 3,
        "target_ratios": [0.3, 0.5, 0.7],
        "compute_bleu": True,
        "compute_rouge": True,  
        "compute_perplexity": True,
        "measure_speed": True,
        "measure_memory": True,
        "save_plots": True
    }
    
    config = setup_config(default_config, "Token Compression Evaluation")
    evaluate_model(config)