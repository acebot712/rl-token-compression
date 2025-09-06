"""
Simple memory handler for PyTorch training.

As Linus would say: "Don't over-engineer solutions to non-problems."
This replaces 335 lines of unnecessary complexity with 30 lines that actually work.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


class SimpleMemoryHandler:
    """Minimal memory handler that only intervenes on actual OOM."""

    def __init__(self, device: str):
        """
        Initialize memory handler.

        Args:
            device: PyTorch device string (cpu, cuda, mps)
        """
        self.device = device
        self.oom_count = 0

    def handle_oom(self):
        """Emergency cleanup ONLY when we actually hit OOM."""
        self.oom_count += 1
        logger.warning(f"OOM error #{self.oom_count} - performing emergency cleanup")

        if self.device.startswith("mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # One gc.collect() is enough
        gc.collect()

        return self.oom_count <= 3  # Give up after 3 OOM errors
