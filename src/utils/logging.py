"""
Simple logging utilities for RL token compression.

Provides basic logging functionality without over-engineering.
"""

import logging
import sys
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    """Get a simple logger with basic formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

def get_metrics_logger(name: str) -> logging.Logger:
    """Get logger for metrics (same as regular logger for simplicity)."""
    return get_logger(name)

class TrainingContextManager:
    """Simple context manager for training sessions."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"Experiment {self.experiment_name} failed: {exc_val}")
        else:
            self.logger.info(f"Experiment {self.experiment_name} completed")