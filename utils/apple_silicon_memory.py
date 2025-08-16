"""
Apple Silicon MPS-optimized memory management.

Designed specifically for Apple Silicon's unified memory architecture where 
CPU and GPU share the same memory pool. Avoids CUDA-centric assumptions.

Key principles:
- Unified memory: CPU and GPU share single memory pool (no separate VRAM)
- No host/device transfers needed (data stays in place)
- Focus on total system memory pressure monitoring
- Optimize for data locality and minimize memory fragmentation
- Use MPS-specific APIs where appropriate
"""

import torch
import gc
import psutil
import os
import logging
from typing import Dict, Optional, Any, List, Tuple
from contextlib import contextmanager
import threading
import time

logger = logging.getLogger(__name__)


class UnifiedMemoryManager:
    """
    Memory manager optimized for Apple Silicon's unified memory architecture.
    
    Handles the unique characteristics of M-series chips:
    - Unified memory between CPU and GPU
    - Memory bandwidth-optimized compute
    - Metal Performance Shaders integration
    """
    
    def __init__(self, device: str = "auto", memory_target_gb: float = 32.0):
        """
        Initialize unified memory manager for Apple Silicon.
        
        Args:
            device: Target device ("auto", "mps", "cpu")
            memory_target_gb: Target memory usage in GB (recommend 70% of 48GB = 32GB)
        """
        self.device = self._resolve_device(device)
        self.memory_target_gb = memory_target_gb
        self.memory_target_bytes = int(memory_target_gb * 1024 * 1024 * 1024)
        self.is_mps = self.device.startswith("mps")
        
        # Get system memory info
        self.system_memory = psutil.virtual_memory()
        self.total_memory_gb = self.system_memory.total / (1024 ** 3)
        
        # Track memory usage history for optimization
        self._memory_history: List[float] = []
        self._memory_lock = threading.Lock()
        
        if self.is_mps:
            self._setup_unified_memory_optimizations()
        
        logger.info(f"UnifiedMemoryManager initialized for Apple Silicon")
        logger.info(f"  Total unified memory: {self.total_memory_gb:.1f}GB")
        logger.info(f"  Target memory usage: {memory_target_gb:.1f}GB ({(memory_target_gb/self.total_memory_gb*100):.1f}%)")
        logger.info(f"  Device: {self.device}")
        
        # Verify MPS availability and capabilities
        if self.is_mps and torch.backends.mps.is_available():
            logger.info("  MPS backend available and optimized")
        elif self.is_mps:
            logger.warning("  MPS requested but not available, falling back to CPU")
            self.device = "cpu"
            self.is_mps = False
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string optimized for Apple Silicon."""
        if device == "auto":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _setup_unified_memory_optimizations(self):
        """Setup Apple Silicon unified memory optimizations."""
        try:
            # UNIFIED MEMORY OPTIMIZATION: Disable artificial memory limits
            # Default PyTorch MPS limits to 60% of system memory (28.8GB of 48GB)
            # We need to disable this to use the full unified memory available
            
            # Set environment variables for optimal MPS performance
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit (use all available memory)
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'   # Disable lower limit (manual cleanup)
            
            # Enable MPS optimizations
            if hasattr(torch.backends.mps, 'is_built'):
                logger.info("  MPS backend built with unified memory support")
                
        except Exception as e:
            logger.warning(f"Failed to setup unified memory optimizations: {e}")
    
    def get_unified_memory_info(self) -> Dict[str, float]:
        """
        Get unified memory statistics for Apple Silicon.
        
        Unlike discrete GPU systems, we track total system memory since
        CPU and GPU share the same pool.
        """
        current_memory = psutil.virtual_memory()
        
        # Calculate PyTorch tensor memory usage
        torch_memory_gb = 0.0
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        
        # For MPS, estimate based on resident memory of current process
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024 ** 3)
        
        memory_info = {
            'total_memory_gb': self.total_memory_gb,
            'available_memory_gb': current_memory.available / (1024 ** 3),
            'used_memory_gb': (current_memory.total - current_memory.available) / (1024 ** 3),
            'process_memory_gb': process_memory_gb,
            'torch_memory_gb': torch_memory_gb,
            'memory_pressure_percent': ((current_memory.total - current_memory.available) / current_memory.total) * 100,
            'target_memory_gb': self.memory_target_gb,
            'within_target': process_memory_gb < self.memory_target_gb
        }
        
        return memory_info
    
    def cleanup_unified_memory(self, aggressive: bool = False):
        """
        Perform unified memory cleanup optimized for Apple Silicon.
        
        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        try:
            # UNIFIED MEMORY CLEANUP: Focus on reducing memory pressure, not discrete transfers
            
            # MPS-specific cleanup
            if self.is_mps:
                # Use MPS-specific memory management
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                # Synchronize MPS operations to ensure cleanup
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            
            # Python garbage collection for tensor cleanup
            if aggressive:
                # More aggressive cleanup for memory pressure
                for _ in range(3):  # Multiple GC passes
                    gc.collect()
                    
                # Compact memory allocator if available
                if hasattr(gc, 'compact'):
                    gc.compact()
            else:
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Unified memory cleanup failed: {e}")
    
    def adaptive_batch_sizing(self, base_batch_size: int = 4, max_batch_size: int = 32) -> int:
        """
        Calculate adaptive batch size based on unified memory pressure.
        
        Uses Apple Silicon's unified memory characteristics for optimization.
        """
        memory_info = self.get_unified_memory_info()
        
        # Use memory pressure percentage instead of discrete GPU memory
        memory_pressure = memory_info['memory_pressure_percent']
        
        if memory_pressure < 50:
            # Low memory pressure, can increase batch size
            return min(base_batch_size * 2, max_batch_size)
        elif memory_pressure > 80:
            # High memory pressure, reduce batch size
            return max(base_batch_size // 2, 1)
        else:
            # Normal memory pressure
            return base_batch_size
    
    def check_memory_pressure(self, threshold_percent: float = 85.0) -> bool:
        """Check if unified memory is under pressure."""
        memory_info = self.get_unified_memory_info()
        return memory_info['memory_pressure_percent'] > threshold_percent
    
    def suggest_gradient_accumulation(self, desired_batch_size: int) -> Tuple[int, int]:
        """
        Suggest gradient accumulation parameters based on unified memory constraints.
        
        Returns:
            Tuple of (micro_batch_size, accumulation_steps)
        """
        memory_info = self.get_unified_memory_info()
        available_gb = memory_info['available_memory_gb']
        
        # Estimate memory per sample (rough heuristic for transformer models)
        estimated_memory_per_sample_gb = 0.1  # 100MB per sample (conservative)
        max_micro_batch = int(available_gb / (2 * estimated_memory_per_sample_gb))  # Factor of 2 for gradients
        
        # Calculate optimal micro-batch size
        micro_batch_size = min(max_micro_batch, 8, desired_batch_size)  # Cap at 8 for efficiency
        accumulation_steps = max(1, desired_batch_size // micro_batch_size)
        
        logger.info(f"Suggested gradient accumulation: micro_batch={micro_batch_size}, "
                   f"accumulation_steps={accumulation_steps} for desired_batch={desired_batch_size}")
        
        return micro_batch_size, accumulation_steps
    
    @contextmanager
    def unified_memory_context(self, cleanup_threshold_gb: float = 30.0):
        """
        Context manager for unified memory operations.
        
        Monitors memory usage and performs cleanup when needed.
        """
        initial_memory = self.get_unified_memory_info()
        
        try:
            yield initial_memory
        finally:
            final_memory = self.get_unified_memory_info()
            
            # Check if cleanup is needed
            if final_memory['used_memory_gb'] > cleanup_threshold_gb:
                self.cleanup_unified_memory(aggressive=True)
                
                # Re-check memory after cleanup
                post_cleanup_memory = self.get_unified_memory_info()
                cleanup_freed_gb = final_memory['used_memory_gb'] - post_cleanup_memory['used_memory_gb']
                
                if cleanup_freed_gb > 0.5:  # Log if we freed more than 500MB
                    logger.info(f"Unified memory cleanup freed {cleanup_freed_gb:.1f}GB")
    
    def log_unified_memory_summary(self, prefix: str = ""):
        """Log comprehensive unified memory usage summary."""
        memory_info = self.get_unified_memory_info()
        
        logger.info(f"{prefix}Apple Silicon Unified Memory Summary:")
        logger.info(f"  Total unified memory: {memory_info['total_memory_gb']:.1f}GB")
        logger.info(f"  Available memory: {memory_info['available_memory_gb']:.1f}GB") 
        logger.info(f"  Used memory: {memory_info['used_memory_gb']:.1f}GB")
        logger.info(f"  Process memory: {memory_info['process_memory_gb']:.1f}GB")
        logger.info(f"  Memory pressure: {memory_info['memory_pressure_percent']:.1f}%")
        logger.info(f"  Within target: {'✓' if memory_info['within_target'] else '✗'}")
        
        if memory_info['memory_pressure_percent'] > 85:
            logger.warning("⚠️  High unified memory pressure detected!")
        elif memory_info['memory_pressure_percent'] > 70:
            logger.info("ℹ️  Moderate unified memory usage")


class MPSOptimizedTraining:
    """
    Training optimizations specific to Apple Silicon MPS.
    
    Focuses on data locality and efficient use of unified memory bandwidth.
    """
    
    def __init__(self, memory_manager: UnifiedMemoryManager):
        self.memory_manager = memory_manager
        self.device = memory_manager.device
        
    def optimize_tensor_creation(self, tensor_shape: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        """
        Create tensors optimized for MPS unified memory.
        
        Ensures tensors are created directly in unified memory space
        without unnecessary copies.
        """
        # Create tensor directly on MPS device to avoid CPU->GPU transfer
        if self.memory_manager.is_mps:
            return torch.empty(tensor_shape, dtype=dtype, device=self.device)
        else:
            return torch.empty(tensor_shape, dtype=dtype)
    
    def batch_tensor_operations(self, operations: List[callable]) -> List[Any]:
        """
        Batch tensor operations for better memory bandwidth utilization.
        
        Apple Silicon benefits from batching operations to make efficient
        use of the high memory bandwidth.
        """
        results = []
        
        with self.memory_manager.unified_memory_context():
            for op in operations:
                result = op()
                results.append(result)
                
        return results
    
    @contextmanager
    def memory_efficient_forward_pass(self):
        """Context manager for memory-efficient forward passes."""
        # Pre-cleanup before forward pass
        self.memory_manager.cleanup_unified_memory()
        
        try:
            yield
        finally:
            # Post-cleanup after forward pass
            self.memory_manager.cleanup_unified_memory()


def setup_apple_silicon_optimizations():
    """Setup global optimizations for Apple Silicon."""
    # CRITICAL: Disable artificial memory limit to use full unified memory
    # Default PyTorch limits to 60% (28.8GB of 48GB), we need more for training
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'   # Disable lower limit
    
    # Optimize threading for Apple Silicon
    os.environ['OMP_NUM_THREADS'] = '8'  # M4 Pro has 8 performance cores
    os.environ['MKL_NUM_THREADS'] = '8'
    
    # Use optimized memory allocator
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'native'
    
    # More aggressive Python GC for unified memory
    gc.set_threshold(500, 5, 5)
    
    logger.info("Apple Silicon optimizations configured")
    logger.info("  Unified memory optimizations: enabled")
    logger.info("  MPS high watermark: DISABLED (using full unified memory)")
    logger.info("  MPS low watermark: DISABLED (manual memory management)")
    logger.info("  Threading optimized for M4 Pro (8 performance cores)")
    logger.warning("  ⚠️ MPS memory limits disabled - monitor system memory carefully")