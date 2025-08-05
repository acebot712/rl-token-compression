"""
Simple checkpoint management.

Basic checkpoint save/load without over-engineering.
"""

import os
import torch
from .logging import get_logger

logger = get_logger(__name__)

class CheckpointManager:
    """Simple checkpoint manager."""
    
    def __init__(self, checkpoint_dir: str, max_backups: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_backups = max_backups
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, checkpoint: dict, checkpoint_name: str, is_best: bool = False) -> bool:
        """Save checkpoint."""
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
                torch.save(checkpoint, best_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_name: str):
        """Load checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path)
        return None