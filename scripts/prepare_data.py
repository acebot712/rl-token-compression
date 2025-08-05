#!/usr/bin/env python3
"""
Data preparation script for token compression training.

Prepares datasets with proper train/val/test splits.
"""

import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import random
import numpy as np
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.utils.config import setup_config, save_config
from src.utils.common import setup_output_dir, print_section_header, print_config_summary, handle_common_errors


def load_dataset_with_fallback(dataset_name: str, max_sequences: int):
    """Load dataset with fallback to dummy data if needed."""
    print(f"Loading dataset: {dataset_name}")
    
    try:
        if dataset_name == "reddit":
            return load_dataset("reddit", split="train", trust_remote_code=True)
        elif dataset_name == "wikitext":
            return load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        elif dataset_name == "bookcorpus":
            return load_dataset("bookcorpus", split="train")
        elif dataset_name == "openwebtext":
            return load_dataset("openwebtext", split="train")
        else:
            return load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}': {e}")
        print("Using fallback dummy dataset...")
        return [
            {"text": "This is a sample text for testing the token compression system."},
            {"text": "Another example sentence that can be used for training purposes."},
            {"text": "The quick brown fox jumps over the lazy dog multiple times."},
        ] * (max_sequences // 3)


def process_sequences(dataset, tokenizer: GPT2Tokenizer, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process and tokenize sequences from dataset."""
    processed_sequences = []
    seen_texts = set() if config['filter_duplicates'] else None
    processed_count = 0
    
    print("Processing sequences...")
    for i, example in enumerate(tqdm(dataset)):
        if processed_count >= config['max_sequences']:
            break
        
        # Extract text
        text = extract_text_from_example(example, config['dataset_name'])
        if not text or len(text.strip()) < config['min_length']:
            continue
        
        # Filter duplicates
        if config['filter_duplicates']:
            if text in seen_texts:
                continue
            seen_texts.add(text)
        
        # Tokenize
        try:
            tokens = tokenizer.encode(text, max_length=config['max_length'], truncation=True)
            if len(tokens) < config['min_length'] // 4:
                continue
            
            processed_sequences.append({
                "text": text,
                "tokens": tokens,
                "length": len(tokens)
            })
            processed_count += 1
            
        except Exception as e:
            continue  # Skip failed tokenization
    
    return processed_sequences


def split_and_save_data(sequences: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    """Split data into train/val/test and save."""
    if len(sequences) == 0:
        raise ValueError("No sequences were processed successfully!")
    
    # Shuffle
    random.shuffle(sequences)
    
    # Split
    n_total = len(sequences)
    n_train = int(n_total * config['train_split'])
    n_val = int(n_total * config['val_split'])
    
    train_data = sequences[:n_train]
    val_data = sequences[n_train:n_train + n_val]
    test_data = sequences[n_train + n_val:]
    
    print(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save splits
    output_dir = config['output_dir']
    
    files_to_save = [
        (train_data, "processed_data.json", "Training data"),
        (val_data, "val_data.json", "Validation data"),
        (test_data, "test_data.json", "Test data")
    ]
    
    for data, filename, description in files_to_save:
        if data:
            path = os.path.join(output_dir, filename)
            with open(path, 'w') as f:
                json.dump(data, f)
            print(f"  {description} saved to: {path}")
    
    # Save statistics
    stats = {
        "total_sequences": len(sequences),
        "train_sequences": len(train_data),
        "val_sequences": len(val_data),
        "test_sequences": len(test_data),
        "avg_length": float(np.mean([seq["length"] for seq in sequences])),
        "max_length": max([seq["length"] for seq in sequences]),
        "min_length": min([seq["length"] for seq in sequences]),
        "tokenizer": config['tokenizer'],
        "dataset_name": config['dataset_name']
    }
    
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved to: {stats_path}")


@handle_common_errors
def prepare_data(config):
    """Prepare dataset for token compression training."""
    print_section_header("DATA PREPARATION FOR TOKEN COMPRESSION")
    print_config_summary(config)
    print()
    
    # Setup
    setup_output_dir(config['output_dir'])
    random.seed(config['random_seed'])
    
    # Load tokenizer
    print(f"Loading tokenizer: {config['tokenizer']}")
    tokenizer = GPT2Tokenizer.from_pretrained(config['tokenizer'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and process dataset
    dataset = load_dataset_with_fallback(config['dataset_name'], config['max_sequences'])
    sequences = process_sequences(dataset, tokenizer, config)
    
    print(f"Processed {len(sequences)} sequences")
    
    # Split and save
    split_and_save_data(sequences, config)
    
    # Save config
    config_path = os.path.join(config['output_dir'], "preparation_config.json")
    save_config(config, config_path)
    
    print("\nData preparation complete!")


def extract_text_from_example(example, dataset_name: str) -> str:
    """Extract text from dataset example based on dataset type."""
    if isinstance(example, str):
        return example
    elif isinstance(example, dict):
        # Try common text fields
        for field in ['text', 'content', 'body', 'title', 'summary']:
            if field in example:
                text = example[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()
        
        # For Reddit specifically
        if dataset_name == "reddit":
            title = example.get("title", "")
            body = example.get("body", "")
            if title and body:
                return f"{title}\n\n{body}"
            return title or body
        
        # Return first string value found
        for key, value in example.items():
            if isinstance(value, str) and len(value.strip()) > 20:
                return value.strip()
    
    return str(example)


if __name__ == "__main__":
    default_config = {
        "output_dir": "data/processed",
        "max_sequences": 50000,
        "max_length": 1024,
        "min_length": 50,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "tokenizer": "gpt2",
        "dataset_name": "wikitext",
        "filter_duplicates": True,
        "random_seed": 42
    }
    
    config = setup_config(default_config, "Token Compression Data Preparation")
    
    # Validate splits
    total_split = config['train_split'] + config['val_split'] + config['test_split']
    if abs(total_split - 1.0) > 0.01:
        print(f"WARNING: Data splits sum to {total_split}, not 1.0. Normalizing...")
        config['train_split'] = config['train_split'] / total_split
        config['val_split'] = config['val_split'] / total_split  
        config['test_split'] = config['test_split'] / total_split
    
    prepare_data(config)