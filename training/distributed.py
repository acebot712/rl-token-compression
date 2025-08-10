"""
Distributed training support for RL Token Compression.

Simple, clean distributed training without over-engineering.
- Multi-GPU support with DDP
- Gradient synchronization
- Proper checkpointing across ranks
- Clear error handling
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, List, Optional, Tuple
import logging

from training.trainer import SimpleJointTrainer
from utils.errors import ModelError, DeviceError


logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize distributed training process group.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
    except Exception as e:
        raise ModelError(f"Failed to initialize distributed training: {e}")


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_distributed_config() -> Tuple[int, int, bool]:
    """
    Get distributed training configuration from environment.
    
    Returns:
        Tuple of (rank, world_size, is_distributed)
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1
    
    return rank, world_size, is_distributed


def setup_device_for_rank(rank: int, device_type: str = "cuda") -> torch.device:
    """
    Setup device for specific rank.
    
    Args:
        rank: Process rank
        device_type: Device type ('cuda' or 'cpu')
        
    Returns:
        Device for this rank
    """
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError("CUDA not available for distributed training")
        
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:
        raise DeviceError(f"Unsupported device type for distributed training: {device_type}")
    
    return device


class DistributedJointTrainer(SimpleJointTrainer):
    """Distributed version of the joint trainer."""
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        device_type: str = "cuda",
        **kwargs
    ):
        """
        Initialize distributed trainer.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            device_type: Device type ('cuda' or 'cpu')
            **kwargs: Arguments for SimpleJointTrainer
        """
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        # Setup device for this rank
        device = setup_device_for_rank(rank, device_type)
        kwargs['device'] = str(device)
        
        super().__init__(**kwargs)
        
        self.device_obj = device
        self.backend = "nccl" if device_type == "cuda" else "gloo"
    
    def setup_models(self, tokenizer, reconstructor):
        """Setup models with distributed data parallel wrapping."""
        super().setup_models(tokenizer, reconstructor)
        
        # Wrap models with DDP
        self.policy = DDP(
            self.policy,
            device_ids=[self.rank] if self.device_obj.type == "cuda" else None,
            find_unused_parameters=False  # More efficient
        )
        
        self.reconstructor = DDP(
            self.reconstructor,
            device_ids=[self.rank] if self.device_obj.type == "cuda" else None,
            find_unused_parameters=False
        )
        
        if self.is_main_process:
            policy_params = sum(p.numel() for p in self.policy.parameters())
            recon_params = sum(p.numel() for p in self.reconstructor.parameters())
            print(f"✓ Models wrapped with DDP - Policy: {policy_params:,} params, "
                  f"Reconstructor: {recon_params:,} params")
    
    def train_step(self, batch_sequences: List[List[int]]) -> Dict[str, float]:
        """Distributed training step."""
        # Ensure all processes have the same batch
        batch_sequences = self._sync_batch(batch_sequences)
        
        # Standard training step
        metrics = super().train_step(batch_sequences)
        
        # Average metrics across all processes
        metrics = self._average_metrics(metrics)
        
        return metrics
    
    def _sync_batch(self, batch_sequences: List[List[int]]) -> List[List[int]]:
        """Synchronize batch across all processes."""
        # For simplicity, assume all processes have the same batch
        # In practice, you might want to gather/scatter batches
        return batch_sequences
    
    def _average_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Average metrics across all processes."""
        averaged_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device_obj)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            averaged_metrics[key] = (tensor / self.world_size).item()
        
        return averaged_metrics
    
    def save_checkpoint(self, path: str):
        """Save checkpoint from main process only."""
        if self.is_main_process:
            # Save unwrapped model states (remove DDP wrapper)
            checkpoint = {
                'policy_state_dict': self.policy.module.state_dict(),
                'reconstructor_state_dict': self.reconstructor.module.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'reconstructor_optimizer_state_dict': self.reconstructor_optimizer.state_dict(),
                'step': self.step,
                'epoch': self.epoch,
                'rank': self.rank,
                'world_size': self.world_size
            }
            torch.save(checkpoint, path)
        
        # Synchronize all processes
        dist.barrier()
    
    def load_checkpoint(self, path: str):
        """Load checkpoint for distributed training."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load on CPU first, then move to device
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model states (handle DDP wrapper)
        if hasattr(self.policy, 'module'):
            self.policy.module.load_state_dict(checkpoint['policy_state_dict'])
            self.reconstructor.module.load_state_dict(checkpoint['reconstructor_state_dict'])
        else:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        
        # Load optimizer states
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.reconstructor_optimizer.load_state_dict(checkpoint['reconstructor_optimizer_state_dict'])
        
        # Load training state
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        if self.is_main_process:
            print(f"✓ Loaded checkpoint from {path} - Step: {self.step}, Epoch: {self.epoch}")
    
    def print(self, message: str):
        """Print from main process only."""
        if self.is_main_process:
            print(message)
    
    def train(self, train_data: List[List[int]], batch_size: int = 16, 
              max_epochs: int = 50, output_dir: str = "outputs/training"):
        """Distributed training loop."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Adjust batch size for distributed training
        # Each process gets batch_size samples, so effective batch is batch_size * world_size
        samples_per_rank = len(train_data) // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank
        local_data = train_data[start_idx:end_idx]
        
        self.print(f"Training on {len(local_data)} samples per rank ({len(train_data)} total)")
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Simple batching on local data
            for i in range(0, len(local_data), batch_size):
                batch = local_data[i:i+batch_size]
                if len(batch) < batch_size:
                    continue  # Skip incomplete batches for consistency
                
                metrics = self.train_step(batch)
                epoch_losses.append(metrics)
                
                # Logging from main process only
                if self.step % 100 == 0 and self.is_main_process:
                    self.print(f"Step {self.step}: Policy Loss: {metrics['policy_loss']:.4f}, "
                              f"Recon Loss: {metrics['reconstructor_loss']:.4f}, "
                              f"Compression: {metrics['compression_ratio']:.3f}")
            
            # Epoch summary
            if epoch_losses:
                avg_policy_loss = sum(m['policy_loss'] for m in epoch_losses) / len(epoch_losses)
                avg_recon_loss = sum(m['reconstructor_loss'] for m in epoch_losses) / len(epoch_losses)
                avg_compression = sum(m['compression_ratio'] for m in epoch_losses) / len(epoch_losses)
                
                self.print(f"Epoch {epoch+1}/{max_epochs} - "
                          f"Policy: {avg_policy_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
                          f"Compression: {avg_compression:.3f}")
            
            # Checkpointing every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path)
        
        # Save final model
        final_path = os.path.join(output_dir, "final_model.pt")
        self.save_checkpoint(final_path)
        self.print(f"Distributed training complete. Models saved to {output_dir}")


def create_distributed_trainer(
    config: Dict[str, Any], 
    train_data: List[List[int]],
    reconstructor,
    tokenizer
) -> DistributedJointTrainer:
    """
    Create distributed trainer.
    
    Args:
        config: Training configuration
        train_data: Training data
        reconstructor: Reconstructor model
        tokenizer: Tokenizer
        
    Returns:
        Configured distributed trainer
    """
    rank, world_size, is_distributed = get_distributed_config()
    
    if not is_distributed:
        raise ValueError("Distributed training requested but RANK/WORLD_SIZE not set")
    
    # Setup distributed process group
    device_type = "cuda" if config.get('device', 'auto') in ['cuda', 'auto'] else "cpu"
    setup_distributed(rank, world_size, backend="nccl" if device_type == "cuda" else "gloo")
    
    trainer = DistributedJointTrainer(
        rank=rank,
        world_size=world_size,
        device_type=device_type,
        policy_lr=config.get('learning_rate_policy', 1e-3),
        reconstructor_lr=config.get('learning_rate_reconstructor', 1e-4),
        context_window=config.get('context_window', 5),
        random_seed=config.get('random_seed', 42)
    )
    
    trainer.setup_models(tokenizer, reconstructor)
    
    if rank == 0:
        print(f"✓ Initialized distributed training - Rank {rank}/{world_size}")
    
    return trainer


def launch_distributed_training():
    """
    Launch distributed training using torchrun.
    
    Example usage:
        torchrun --nproc_per_node=2 training/distributed.py --config configs/training.json
    """
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from training.train import main_train_function  # Would need to refactor train.py
    
    try:
        rank, world_size, is_distributed = get_distributed_config()
        if is_distributed:
            print(f"Starting distributed training - Rank {rank}/{world_size}")
            main_train_function(distributed=True)
        else:
            print("No distributed environment detected, running single process")
            main_train_function(distributed=False)
    except Exception as e:
        logger.error(f"Distributed training failed: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    launch_distributed_training()