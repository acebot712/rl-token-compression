"""
Common utilities for RL Token Compression.

Extracted patterns that were duplicated across multiple main scripts.
Single responsibility: eliminate code duplication.
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def setup_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def load_tokenizer_and_model(model_path: str) -> tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    """
    Load tokenizer and model from path.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (tokenizer, model)
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Setup special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model


def load_json_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data file.
    
    Args:
        data_path: Path to JSON file
        
    Returns:
        List of data items
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {data_path}, got {type(data)}")
    
    return data


def extract_sequences_from_data(data: List[Dict[str, Any]], tokenizer: GPT2Tokenizer) -> List[List[int]]:
    """
    Extract token sequences from data.
    
    Args:
        data: List of data items
        tokenizer: Tokenizer for encoding text
        
    Returns:
        List of token sequences
    """
    sequences = []
    
    for item in data:
        if 'tokens' in item:
            # Already tokenized
            sequences.append(item['tokens'])
        elif 'text' in item:
            # Need to tokenize
            tokens = tokenizer.encode(item['text'], max_length=1024, truncation=True)
            sequences.append(tokens)
        else:
            print(f"Warning: skipping item without 'tokens' or 'text': {item}")
    
    return sequences


def print_section_header(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("=" * width)
    print(title.upper())
    print("=" * width)


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of important config values."""
    important_keys = ['data_path', 'output_dir', 'batch_size', 'learning_rate', 'max_epochs', 'device']
    
    for key in important_keys:
        if key in config:
            print(f"{key}: {config[key]}")


def validate_paths(config: Dict[str, Any], required_paths: List[str]) -> None:
    """
    Validate that required file paths exist.
    
    Args:
        config: Configuration dictionary
        required_paths: List of config keys that should be valid file paths
    """
    for path_key in required_paths:
        if path_key not in config:
            continue
            
        path = config[path_key]
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path_key}={path}")


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    setup_output_dir(os.path.dirname(output_path))
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_summary_table(data: Dict[str, Dict[str, Any]], columns: List[str], title: str = "Summary") -> None:
    """
    Print a formatted summary table.
    
    Args:
        data: Dict of row_name -> {column_name: value}
        columns: List of column names to display
        title: Table title
    """
    print(f"\n{title.upper()}")
    print("-" * 60)
    
    # Header
    header = f"{'Method':<12}"
    for col in columns:
        header += f" {col:<12}"
    print(header)
    print("-" * 60)
    
    # Rows
    for name, row_data in data.items():
        row = f"{name:<12}"
        for col in columns:
            value = row_data.get(col, 'N/A')
            if isinstance(value, float):
                row += f" {value:<12.3f}"
            else:
                row += f" {str(value):<12}"
        print(row)


class SimpleLogger:
    """Simple logger that prints to console with prefixes."""
    
    def __init__(self, name: str):
        self.name = name
    
    def info(self, message: str) -> None:
        print(f"[{self.name}] {message}")
    
    def warning(self, message: str) -> None:
        print(f"[{self.name}] WARNING: {message}")
    
    def error(self, message: str) -> None:
        print(f"[{self.name}] ERROR: {message}")


def handle_common_errors(func):
    """Decorator for common error handling in main scripts."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            print(f"ERROR: File not found - {e}")
            return 1
        except ValueError as e:
            print(f"ERROR: Invalid value - {e}")
            return 1
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 1
        except Exception as e:
            print(f"ERROR: Unexpected error - {e}")
            return 1
    return wrapper