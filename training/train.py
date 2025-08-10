import os
import sys

# MPS memory configuration is handled by Apple Silicon memory manager

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from training.trainer import create_simple_trainer
from utils.config import setup_validated_config, save_config, resolve_device
from utils.common import setup_output_dir, print_section_header, print_config_summary
from utils.errors import (handle_errors, validate_file_exists, validate_directory_writable, 
                         validate_device_available, check_memory_requirements, ModelError)
from utils.deterministic import create_reproducible_trainer_config, save_reproducibility_info
from training.distributed import create_distributed_trainer, get_distributed_config, cleanup_distributed


@handle_errors("training")
def train(config):
    """Train the token compression system using joint training."""
    print_section_header("RL TOKEN COMPRESSION - JOINT TRAINING")
    
    # Ensure reproducible configuration
    config = create_reproducible_trainer_config(config)
    
    print_config_summary(config)
    print()
    
    # Validate and resolve device
    config['device'] = validate_device_available(config['device'])
    print(f"Using device: {config['device']}")
    
    # Check memory requirements
    check_memory_requirements(config['device'], config.get('batch_size', 16), 512)
    
    # Validate and setup output directory
    validate_directory_writable(config['output_dir'], "output")
    setup_output_dir(config['output_dir'])
    
    # Validate input files
    validate_file_exists(config['data_path'], "Training data")
    validate_file_exists(config['reconstructor_path'] + '/config.json', "Reconstructor model")
    
    # Check for existing checkpoints if resume is enabled
    resume_from = None
    if config.get('resume', False):
        # Look for the latest checkpoint
        checkpoint_files = []
        if os.path.exists(config['output_dir']):
            for file in os.listdir(config['output_dir']):
                if file.startswith('checkpoint_step_') and file.endswith('.pt'):
                    checkpoint_files.append(file)
        
        if checkpoint_files:
            # Sort by step number and get the latest
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            resume_from = os.path.join(config['output_dir'], checkpoint_files[-1])
            print(f"Found checkpoint to resume from: {resume_from}")
        else:
            print("No checkpoint found, starting fresh training")
    
    # Load data
    import json
    with open(config['data_path'], 'r') as f:
        train_data = json.load(f)
    train_sequences = [item['tokens'] for item in train_data]
    
    # Load reconstructor
    from transformers import GPT2LMHeadModel
    try:
        reconstructor = GPT2LMHeadModel.from_pretrained(config['reconstructor_path'])
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        raise ModelError(f"Failed to load reconstructor from {config['reconstructor_path']}: {e}")
    
    # Check if distributed training is requested
    rank, world_size, is_distributed = get_distributed_config()
    
    # Create trainer (distributed or single-process)
    if is_distributed:
        print(f"Setting up distributed training - Rank {rank}/{world_size}")
        trainer = create_distributed_trainer(config, train_sequences, reconstructor, tokenizer)
    else:
        trainer = create_simple_trainer(config, train_sequences, reconstructor, tokenizer)
    
    print("Starting simplified joint training...")
    
    # Train with simple interface
    trainer.train(
        train_data=train_sequences,
        batch_size=config.get('batch_size', 16),
        max_epochs=config.get('max_epochs', 50),
        output_dir=config['output_dir']
    )
    
    print(f"Joint training complete. Models saved to {config['output_dir']}")
    
    # Save configuration and reproducibility info (main process only for distributed)
    if not is_distributed or rank == 0:
        config_path = os.path.join(config['output_dir'], "training_config.json")
        save_config(config, config_path)
        save_reproducibility_info(config['output_dir'], config)
    
    # Cleanup distributed training
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
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
        "debug": False
    }
    
    config = setup_validated_config(default_config, "RL Token Compression Training", "training")
    train(config)