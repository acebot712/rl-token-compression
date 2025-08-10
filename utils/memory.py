"""
Memory management utilities for efficient training.

Provides MPS-specific optimizations, memory monitoring, and cleanup functions
to prevent out-of-memory issues during training.
"""

import torch
import gc
import psutil
import os
import logging
from typing import Dict, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Comprehensive memory management for PyTorch training.
    
    Handles MPS-specific optimizations, memory monitoring, and cleanup
    to prevent OOM issues during training.
    """
    
    def __init__(self, device: str = "auto", target_memory_mb: float = 8000):
        """
        Initialize memory manager.
        
        Args:
            device: Target device ("auto", "cuda", "mps", "cpu")
            target_memory_mb: Target memory usage in MB (default 8GB)
        """
        self.device = self._resolve_device(device)
        self.target_memory_mb = target_memory_mb
        self.is_mps = self.device.startswith("mps")
        self.is_cuda = self.device.startswith("cuda")
        
        # Setup MPS optimizations
        if self.is_mps:
            self._setup_mps_optimizations()
        
        logger.info(f"MemoryManager initialized - Device: {self.device}, Target: {target_memory_mb}MB")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _setup_mps_optimizations(self):
        """Setup MPS-specific memory optimizations."""
        try:
            # Set environment variables for MPS optimization
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable MPS memory caching
            
            # Set memory fraction if available
            if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
                memory_fraction = min(0.7, self.target_memory_mb / available_memory)
                torch.mps.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"MPS memory fraction set to {memory_fraction:.2f}")
            
        except Exception as e:
            logger.warning(f"Failed to setup MPS optimizations: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        gpu_memory = 0.0
        if self.is_cuda and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        elif self.is_mps:
            # MPS doesn't have direct memory query, estimate from system memory
            gpu_memory = cpu_memory * 0.6  # Rough estimate for unified memory
        
        return {
            'cpu_memory_mb': cpu_memory,
            'gpu_memory_mb': gpu_memory,
            'total_memory_mb': cpu_memory + gpu_memory,
            'memory_percent': (cpu_memory / psutil.virtual_memory().total) * 100
        }
    
    def cleanup_memory(self, force_gc: bool = True):
        """Perform comprehensive memory cleanup."""
        try:
            # Clear GPU caches
            if self.is_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.is_mps and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            
            # Force Python garbage collection
            if force_gc:
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def adaptive_batch_size(self, base_batch_size: int = 4, max_batch_size: int = 32) -> int:
        """
        Calculate adaptive batch size based on current memory usage.
        
        Args:
            base_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            
        Returns:
            Recommended batch size
        """
        memory_info = self.get_memory_usage()
        current_memory = memory_info['total_memory_mb']
        
        if current_memory < self.target_memory_mb * 0.5:
            # Low memory usage, can increase batch size
            return min(base_batch_size * 2, max_batch_size)
        elif current_memory > self.target_memory_mb * 0.8:
            # High memory usage, reduce batch size
            return max(base_batch_size // 2, 1)
        else:
            # Normal memory usage
            return base_batch_size
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        memory_info = self.get_memory_usage()
        return memory_info['memory_percent'] > 85.0  # Above 85% system memory
    
    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient operations."""
        initial_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            # Always cleanup after operations
            self.cleanup_memory()
            
            final_memory = self.get_memory_usage()
            memory_diff = final_memory['total_memory_mb'] - initial_memory['total_memory_mb']
            
            if abs(memory_diff) > 100:  # Log if memory changed by more than 100MB
                logger.debug(f"Memory change: {memory_diff:+.1f}MB "
                           f"({initial_memory['total_memory_mb']:.1f} -> {final_memory['total_memory_mb']:.1f})")


class CPUOffloadManager:
    """
    Manages CPU offloading for large models to reduce GPU memory usage.
    """
    
    def __init__(self, model: torch.nn.Module, device: str):
        """
        Initialize CPU offload manager.
        
        Args:
            model: Model to manage
            device: Target device for training
        """
        self.model = model
        self.device = device
        self.cpu_device = "cpu"
        self.is_on_gpu = False
    
    def move_to_gpu(self):
        """Move model to GPU for training."""
        if not self.is_on_gpu:
            self.model.to(self.device)
            self.is_on_gpu = True
    
    def move_to_cpu(self):
        """Move model to CPU to free GPU memory."""
        if self.is_on_gpu:
            self.model.to(self.cpu_device)
            self.is_on_gpu = False
    
    @contextmanager
    def gpu_context(self):
        """Context manager for temporary GPU usage."""
        was_on_gpu = self.is_on_gpu
        
        try:
            self.move_to_gpu()
            yield
        finally:
            if not was_on_gpu:
                self.move_to_cpu()


def setup_memory_efficient_training():
    """Setup global memory efficiency settings."""
    # Optimize Python memory management
    gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
    
    # Set environment variables for memory efficiency
    os.environ['PYTHONMALLOC'] = 'malloc'  # Use system malloc instead of Python's
    os.environ['OMP_NUM_THREADS'] = '4'    # Limit CPU threading overhead
    
    # Reduce PyTorch memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    logger.info("Memory efficient training setup completed")


def log_memory_summary(memory_manager: MemoryManager, prefix: str = ""):
    """Log a comprehensive memory usage summary."""
    memory_info = memory_manager.get_memory_usage()
    
    logger.info(f"{prefix}Memory Summary:")
    logger.info(f"  CPU Memory: {memory_info['cpu_memory_mb']:.1f}MB")
    logger.info(f"  GPU Memory: {memory_info['gpu_memory_mb']:.1f}MB") 
    logger.info(f"  Total Memory: {memory_info['total_memory_mb']:.1f}MB")
    logger.info(f"  System Usage: {memory_info['memory_percent']:.1f}%")
    
    if memory_manager.check_memory_pressure():
        logger.warning("⚠️  System under memory pressure!")