"""
Simple configuration system for RL Token Compression.

No bullshit. Just load JSON configs and merge with CLI args.
Does one thing: configuration. Does it well.
"""

import json
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass
import os


@dataclass
class Config:
    """Simple config class. Use dataclasses like a normal person."""
    pass


def load_json_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file. That's it."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_config(default_config: Dict[str, Any], script_name: str = "Script") -> Dict[str, Any]:
    """
    Setup configuration with JSON file and CLI overrides.
    
    No YAML. No nested argument generation. No recursive nonsense.
    Just JSON config + simple CLI overrides.
    
    Args:
        default_config: Default values
        script_name: Name for help text
        
    Returns:
        Final configuration dictionary
    """
    parser = argparse.ArgumentParser(description=f"{script_name} configuration")
    
    # Config file argument
    parser.add_argument('--config', type=str, help='JSON configuration file')
    
    # Simple CLI overrides for common parameters
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], help='Device')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--data_path', type=str, help='Data path')
    
    args = parser.parse_args()
    
    # Start with defaults
    config = default_config.copy()
    
    # Load config file if provided
    if args.config:
        try:
            file_config = load_json_config(args.config)
            config.update(file_config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"Failed to load config file: {e}")
            print("Using default configuration")
    
    # Apply CLI overrides
    cli_overrides = {}
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            cli_overrides[key] = value
            config[key] = value
    
    if cli_overrides:
        print(f"CLI overrides: {cli_overrides}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to: {output_path}")


def auto_device() -> str:
    """Auto-detect best available device. Extracted to avoid duplication."""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device."""
    if device == "auto":
        return auto_device()
    return device


if __name__ == "__main__":
    # Simple test
    test_config = {
        'batch_size': 16,
        'learning_rate': 3e-4,
        'data_path': 'data/processed/data.json'
    }
    
    config = setup_config(test_config, "Test Script")
    print("Final configuration:")
    print(json.dumps(config, indent=2))