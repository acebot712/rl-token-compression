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
import time
from datetime import datetime
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from utils.config import setup_validated_config, save_config
from utils.common import setup_output_dir, print_section_header, print_config_summary
from utils.errors import handle_errors, validate_directory_writable, DataError
from utils.logging import get_component_logger

logger = get_component_logger("DATA-PREP")


def load_dataset_with_fallback(dataset_name: str, max_sequences: int):
    """Load dataset with fallback to dummy data if needed."""
    logger.info(f"Starting dataset loading process: {dataset_name}")
    logger.info(f"Target sequences to process: {max_sequences}")
    print(f"Loading dataset: {dataset_name}")
    
    start_time = time.time()
    try:
        logger.info(f"Attempting to load dataset from HuggingFace: {dataset_name}")
        if dataset_name == "reddit":
            logger.info("Loading Reddit dataset with trust_remote_code=True")
            dataset = load_dataset("reddit", split="train", trust_remote_code=True)
        elif dataset_name == "wikitext":
            logger.info("Loading WikiText-103 raw dataset")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        elif dataset_name == "bookcorpus":
            logger.info("Loading BookCorpus dataset")
            dataset = load_dataset("bookcorpus", split="train")
        elif dataset_name == "openwebtext":
            logger.info("Loading OpenWebText dataset")
            dataset = load_dataset("openwebtext", split="train")
        else:
            logger.info(f"Loading custom dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")
        
        load_time = time.time() - start_time
        logger.info(f"Dataset loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Dataset size: {len(dataset)} examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise DataError(f"Failed to load dataset '{dataset_name}': {e}. "
                       f"Check dataset name or install required datasets library.")


def process_sequences(dataset, tokenizer: GPT2Tokenizer, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process and tokenize sequences from dataset."""
    logger.info("Starting sequence processing...")
    logger.info(f"Configuration: max_sequences={config['max_sequences']}, max_length={config['max_length']}")
    logger.info(f"Filter duplicates: {config['filter_duplicates']}")
    
    processed_sequences = []
    seen_texts = set() if config['filter_duplicates'] else None
    processed_count = 0
    skipped_short = 0
    skipped_duplicates = 0
    skipped_tokenization = 0
    
    start_time = time.time()
    print("Processing sequences...")
    
    for i, example in enumerate(tqdm(dataset)):
        # Let tqdm handle progress display
        
        if processed_count >= config['max_sequences']:
            logger.info(f"Reached target of {config['max_sequences']} sequences")
            break
        
        # Extract text
        text = extract_text_from_example(example, config['dataset_name'])
        if not text or len(text.strip()) < config['min_length']:
            skipped_short += 1
            continue
        
        # Filter duplicates
        if config['filter_duplicates']:
            if text in seen_texts:
                skipped_duplicates += 1
                continue
            seen_texts.add(text)
        
        # Tokenize
        try:
            tokens = tokenizer.encode(text, max_length=config['max_length'], truncation=True)
            if len(tokens) < config['min_length'] // 4:
                skipped_short += 1
                continue
            
            processed_sequences.append({
                "text": text,
                "tokens": tokens,
                "length": len(tokens)
            })
            processed_count += 1
            
            # Let tqdm handle progress display
            
        except Exception as e:
            skipped_tokenization += 1
            if skipped_tokenization % 100 == 0:
                logger.warning(f"Tokenization failures: {skipped_tokenization}")
            continue
    
    total_time = time.time() - start_time
    logger.info(f"Sequence processing complete in {total_time:.2f} seconds")
    logger.info(f"Final stats: {processed_count} kept, {skipped_short} too short, {skipped_duplicates} duplicates, {skipped_tokenization} tokenization errors")
    
    return processed_sequences


def split_and_save_data(sequences: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    """Split data into train/val/test and save."""
    logger.info("Starting data splitting and saving process...")
    
    if len(sequences) == 0:
        logger.error("No sequences were processed successfully!")
        raise ValueError("No sequences were processed successfully!")
    
    logger.info(f"Total sequences to split: {len(sequences)}")
    logger.info(f"Split ratios - Train: {config['train_split']}, Val: {config['val_split']}, Test: {1 - config['train_split'] - config['val_split']}")
    
    # Shuffle
    logger.info("Shuffling sequences...")
    random.shuffle(sequences)
    
    # Split
    n_total = len(sequences)
    n_train = int(n_total * config['train_split'])
    n_val = int(n_total * config['val_split'])
    
    logger.info(f"Calculating splits: {n_train} train, {n_val} val, {n_total - n_train - n_val} test")
    
    train_data = sequences[:n_train]
    val_data = sequences[n_train:n_train + n_val]
    test_data = sequences[n_train + n_val:]
    
    logger.info(f"Split complete: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save splits
    output_dir = config['output_dir']
    logger.info(f"Saving data splits to directory: {output_dir}")
    
    files_to_save = [
        (train_data, "processed_data.json", "Training data"),
        (val_data, "val_data.json", "Validation data"),
        (test_data, "test_data.json", "Test data")
    ]
    
    save_start = time.time()
    for data, filename, description in files_to_save:
        if data:
            path = os.path.join(output_dir, filename)
            logger.info(f"Saving {description} ({len(data)} sequences) to {path}")
            
            file_start = time.time()
            with open(path, 'w') as f:
                json.dump(data, f)
            file_time = time.time() - file_start
            
            logger.info(f"  {description} saved in {file_time:.2f} seconds ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")
            print(f"  {description} saved to: {path}")
    
    # Save statistics
    logger.info("Calculating and saving dataset statistics...")
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
    
    total_save_time = time.time() - save_start
    logger.info(f"All data saved in {total_save_time:.2f} seconds")
    logger.info(f"Statistics: avg_len={stats['avg_length']:.1f}, min={stats['min_length']}, max={stats['max_length']}")
    print(f"  Statistics saved to: {stats_path}")


@handle_errors("data preparation")
def prepare_data(config):
    """Prepare dataset for token compression training."""
    logger.info("="*60)
    logger.info("STARTING DATA PREPARATION FOR TOKEN COMPRESSION")
    logger.info("="*60)
    
    print_section_header("DATA PREPARATION FOR TOKEN COMPRESSION")
    print_config_summary(config)
    print()
    
    logger.info(f"Configuration received: {json.dumps(config, indent=2)}")
    
    # Setup and validate
    logger.info("Setting up output directory and validation...")
    validate_directory_writable(config['output_dir'], "output")
    setup_output_dir(config['output_dir'])
    logger.info(f"Output directory ready: {config['output_dir']}")
    
    logger.info(f"Setting random seed: {config['random_seed']}")
    random.seed(config['random_seed'])
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['tokenizer']}")
    print(f"Loading tokenizer: {config['tokenizer']}")
    
    tokenizer_start = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained(config['tokenizer'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    tokenizer_time = time.time() - tokenizer_start
    logger.info(f"Tokenizer loaded in {tokenizer_time:.2f} seconds")
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Load and process dataset
    prep_start = time.time()
    dataset = load_dataset_with_fallback(config['dataset_name'], config['max_sequences'])
    sequences = process_sequences(dataset, tokenizer, config)
    
    logger.info(f"Final processed count: {len(sequences)} sequences")
    print(f"Processed {len(sequences)} sequences")
    
    # Split and save
    split_and_save_data(sequences, config)
    
    # Save config
    logger.info("Saving preparation configuration...")
    config_path = os.path.join(config['output_dir'], "preparation_config.json")
    save_config(config, config_path)
    logger.info(f"Configuration saved to: {config_path}")
    
    total_time = time.time() - prep_start
    logger.info(f"Data preparation completed successfully in {total_time:.2f} seconds")
    logger.info("="*60)
    
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
    
    config = setup_validated_config(default_config, "Token Compression Data Preparation", "data")
    
    # Validate splits
    total_split = config['train_split'] + config['val_split'] + config['test_split']
    if abs(total_split - 1.0) > 0.01:
        print(f"WARNING: Data splits sum to {total_split}, not 1.0. Normalizing...")
        config['train_split'] = config['train_split'] / total_split
        config['val_split'] = config['val_split'] / total_split  
        config['test_split'] = config['test_split'] / total_split
    
    prepare_data(config)