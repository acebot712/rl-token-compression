"""
Simple logging utilities for RL token compression.

Provides basic logging functionality without over-engineering.
"""

import logging
import sys
import os
import pickle
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import psutil

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

def get_tensor_diagnostics(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
    """Get comprehensive diagnostics for a tensor."""
    if tensor is None:
        return {f"{name}_status": "None"}
    
    try:
        diagnostics = {
            f"{name}_shape": list(tensor.shape),
            f"{name}_dtype": str(tensor.dtype),
            f"{name}_device": str(tensor.device),
            f"{name}_requires_grad": tensor.requires_grad,
            f"{name}_numel": tensor.numel(),
        }
        
        # Only compute statistics for non-empty tensors
        if tensor.numel() > 0:
            with torch.no_grad():
                diagnostics.update({
                    f"{name}_min": float(tensor.min().cpu()),
                    f"{name}_max": float(tensor.max().cpu()),
                    f"{name}_mean": float(tensor.mean().cpu()),
                    f"{name}_std": float(tensor.std().cpu()) if tensor.numel() > 1 else 0.0,
                    f"{name}_has_nan": bool(torch.isnan(tensor).any().cpu()),
                    f"{name}_has_inf": bool(torch.isinf(tensor).any().cpu()),
                })
        
        return diagnostics
    except Exception as e:
        return {f"{name}_error": str(e)}

def get_memory_diagnostics() -> Dict[str, Any]:
    """Get comprehensive memory diagnostics."""
    diagnostics = {}
    
    # System memory
    mem = psutil.virtual_memory()
    diagnostics['system_memory_gb'] = mem.total / (1024**3)
    diagnostics['system_memory_used_gb'] = mem.used / (1024**3)
    diagnostics['system_memory_available_gb'] = mem.available / (1024**3)
    diagnostics['system_memory_percent'] = mem.percent
    
    # Process memory
    process = psutil.Process()
    mem_info = process.memory_info()
    diagnostics['process_memory_rss_gb'] = mem_info.rss / (1024**3)
    diagnostics['process_memory_vms_gb'] = mem_info.vms / (1024**3)
    
    # PyTorch memory (if using CUDA/MPS)
    if torch.cuda.is_available():
        diagnostics['cuda_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        diagnostics['cuda_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have detailed memory tracking, estimate from process
        diagnostics['mps_estimated_gb'] = mem_info.rss / (1024**3)
    
    return diagnostics

def save_crash_dump(
    step: int,
    epoch: int,
    batch_indices: Optional[list] = None,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    exception: Optional[Exception] = None,
    output_dir: str = "crash_dumps"
) -> str:
    """
    Save comprehensive crash dump for debugging.
    
    Returns path to crash dump file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = os.path.join(output_dir, f"crash_dump_step{step}_epoch{epoch}_{timestamp}.pkl")
    
    crash_info = {
        'timestamp': timestamp,
        'step': step,
        'epoch': epoch,
        'batch_indices': batch_indices,
        'exception': str(exception) if exception else None,
        'traceback': traceback.format_exc() if exception else None,
        'memory_diagnostics': get_memory_diagnostics(),
        'tensor_diagnostics': {}
    }
    
    # Get diagnostics for all provided tensors
    if tensors:
        for name, tensor in tensors.items():
            crash_info['tensor_diagnostics'][name] = get_tensor_diagnostics(tensor, name)
    
    # Save the crash dump
    with open(dump_path, 'wb') as f:
        pickle.dump(crash_info, f)
    
    # Also save a human-readable text version
    text_path = dump_path.replace('.pkl', '.txt')
    with open(text_path, 'w') as f:
        f.write(f"CRASH DUMP - Step {step}, Epoch {epoch}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Batch indices: {batch_indices}\n\n")
        
        if exception:
            f.write(f"Exception: {exception}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
        
        f.write("Memory Diagnostics:\n")
        for key, value in crash_info['memory_diagnostics'].items():
            f.write(f"  {key}: {value:.2f} GB\n" if 'gb' in key else f"  {key}: {value}\n")
        
        f.write("\nTensor Diagnostics:\n")
        for tensor_name, diag in crash_info['tensor_diagnostics'].items():
            f.write(f"\n  {tensor_name}:\n")
            for key, value in diag.items():
                f.write(f"    {key}: {value}\n")
    
    logger = get_logger(__name__)
    logger.error(f"💀 CRASH DUMP SAVED: {dump_path}")
    logger.error(f"📝 Human-readable version: {text_path}")
    
    return dump_path

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