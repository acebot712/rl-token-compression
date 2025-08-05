#!/usr/bin/env python3
"""
Simple monitoring demo for RL Token Compression.

Demonstrates the basic logging and metrics system without over-engineering.
"""

import os
import sys
import time
import torch

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import simplified monitoring systems
from src.utils.logging import get_logger, get_metrics_logger, TrainingContextManager
from src.utils.metrics import record_training_metrics, record_gradient_health
from src.utils.checkpoints import CheckpointManager

# Set up logging
logger = get_logger(__name__)
metrics_logger = get_metrics_logger(__name__)

def demo_basic_logging():
    """Demo basic logging functionality."""
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

def demo_training_context():
    """Demo training context manager."""
    with TrainingContextManager("demo_experiment") as ctx:
        logger.info("Inside training context")
        time.sleep(1)
        logger.info("Training step completed")

def demo_metrics_recording():
    """Demo metrics recording."""
    for step in range(5):
        metrics = {
            'loss': 0.5 - step * 0.1,
            'accuracy': 0.8 + step * 0.02
        }
        record_training_metrics(step, 0, metrics)
        
        # Demo gradient health monitoring
        record_gradient_health('model', grad_norm=1.2, nan_count=0, inf_count=0)
        time.sleep(0.5)

def demo_checkpoints():
    """Demo checkpoint management."""
    checkpoint_manager = CheckpointManager("./demo_checkpoints")
    
    # Create dummy checkpoint
    dummy_checkpoint = {
        'model_state': {'weight': torch.tensor([1.0, 2.0, 3.0])},
        'step': 100,
        'loss': 0.25
    }
    
    # Save checkpoint
    success = checkpoint_manager.save_checkpoint(dummy_checkpoint, "demo_checkpoint")
    if success:
        logger.info("Checkpoint saved successfully")
    
    # Load checkpoint
    loaded = checkpoint_manager.load_checkpoint("demo_checkpoint")
    if loaded:
        logger.info(f"Checkpoint loaded: step {loaded['step']}, loss {loaded['loss']}")

def main():
    """Run monitoring system demo."""
    logger.info("Starting monitoring system demo")
    logger.info("=" * 50)
    
    logger.info("1. Basic Logging Demo")
    demo_basic_logging()
    
    logger.info("\n2. Training Context Demo")
    demo_training_context() 
    
    logger.info("\n3. Metrics Recording Demo")
    demo_metrics_recording()
    
    logger.info("\n4. Checkpoint Demo")
    demo_checkpoints()
    
    logger.info("=" * 50)
    logger.info("Monitoring demo completed successfully!")

if __name__ == "__main__":
    main()