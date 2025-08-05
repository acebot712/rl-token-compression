import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.training.trainer import create_joint_trainer, TrainingConfig
from src.utils.config import setup_config, save_config, resolve_device
from src.utils.common import setup_output_dir, print_section_header, print_config_summary, handle_common_errors


@handle_common_errors
def train(config):
    """Train the token compression system using joint training."""
    print_section_header("RL TOKEN COMPRESSION - JOINT TRAINING")
    print_config_summary(config)
    print()
    
    # Resolve device
    config['device'] = resolve_device(config['device'])
    print(f"Using device: {config['device']}")
    
    # Setup output directory
    setup_output_dir(config['output_dir'])
    
    # Create training configuration dict for joint trainer
    config_dict = {
        'batch_size': config['batch_size'],
        'learning_rate_policy': config['learning_rate_policy'],
        'learning_rate_reconstructor': config['learning_rate_reconstructor'],
        'max_epochs': config['max_epochs'],
        'max_steps_per_epoch': config.get('max_steps_per_epoch', 1000),
        'context_window': config['context_window'],
        'reward_type': config['reward_type'],
        'device': config['device'],
        'log_freq': config.get('log_freq', 100),
        'eval_freq': config.get('eval_freq', 500),
        'checkpoint_freq': config.get('checkpoint_freq', 1000)
    }
    
    # Create joint trainer
    trainer = create_joint_trainer(
        config_dict=config_dict,
        data_path=config['data_path'],
        reconstructor_path=config['reconstructor_path'],
        val_data_path=config.get('val_data_path')
    )
    
    print("Starting joint training...")
    
    # Train the system
    trainer.train(config['output_dir'])
    
    print(f"Joint training complete. Models saved to {config['output_dir']}")
    
    # Save configuration for reproducibility
    config_path = os.path.join(config['output_dir'], "training_config.json")
    save_config(config, config_path)


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
        "device": "auto"
    }
    
    config = setup_config(default_config, "RL Token Compression Training")
    train(config)