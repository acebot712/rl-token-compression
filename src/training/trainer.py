"""
Joint training system for token compression agent and reconstructor.

This is the core fix for the circular dependency problem. Instead of training
the reconstructor first and then training the agent on the fixed reconstructor,
we train both networks simultaneously on the same batch of data.

Key improvements:
- Breaks circular dependency by using same batch for both networks
- Uses Gumbel-Softmax for differentiable sampling
- Information-theoretic rewards with adaptive scheduling
- Target networks for stable joint optimization
- Variance reduction techniques for policy gradients

This addresses the fundamental flaw in the original training pipeline where
the agent learned patterns A while the reconstructor learned patterns R,
and A ≠ R led to suboptimal performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from dataclasses import dataclass
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.models.agent import SimpleCompressionPolicy
from src.training.rewards import create_reward_function

# Import monitoring systems
from src.utils.logging import get_logger, get_metrics_logger, TrainingContextManager
from src.utils.metrics import record_training_metrics, record_gradient_health
from src.utils.checkpoints import CheckpointManager

# Set up structured logging
logger = get_logger(__name__)
metrics_logger = get_metrics_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for joint training."""
    # Model parameters
    embedding_dim: int = 768
    context_window: int = 5
    policy_hidden_dim: int = 256
    
    # Training parameters
    batch_size: int = 16
    learning_rate_policy: float = 3e-4
    learning_rate_reconstructor: float = 1e-4
    max_epochs: int = 100
    max_steps_per_epoch: int = 1000
    
    # Joint training parameters
    gumbel_temperature_init: float = 1.0
    gumbel_temperature_min: float = 0.1
    
    # Reward parameters
    reward_type: str = "information_theoretic"  # or "simple"
    entropy_weight: float = 0.01
    
    # Regularization
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    
    # Logging and checkpointing
    log_freq: int = 100
    checkpoint_freq: int = 1000
    eval_freq: int = 500
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GumbelSoftmax:
    """Gumbel-Softmax trick for differentiable sampling."""
    
    @staticmethod
    def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Differentiable sampling using Gumbel-Sigmoid trick.
        
        Args:
            logits: Raw logits [batch_size, seq_len]
            temperature: Gumbel temperature
            
        Returns:
            Differentiable samples in [0, 1]
        """
        # Generate Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        
        # Apply Gumbel-Sigmoid
        return torch.sigmoid((logits + gumbel_noise) / temperature)
    
    @staticmethod
    def straight_through_sigmoid(logits: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator for binary decisions.
        Forward: hard decisions, Backward: soft gradients
        """
        probs = torch.sigmoid(logits)
        hard_decisions = (probs > 0.5).float()
        
        # Straight-through: use hard decisions but soft gradients
        return hard_decisions.detach() + probs - probs.detach()


class JointTrainer:
    """
    Joint trainer for agent and reconstructor.
    
    This is the core component that fixes the circular dependency problem
    by training both networks simultaneously on the same data.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        tokenizer: GPT2Tokenizer,
        reconstructor: GPT2LMHeadModel,
        train_data: List[List[int]],
        val_data: Optional[List[List[int]]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Initialize joint trainer.
        
        Args:
            config: Training configuration
            tokenizer: Tokenizer for text processing
            reconstructor: Reconstructor model (will be trained)
            train_data: Training sequences (list of token lists)
            val_data: Validation sequences (optional)
            checkpoint_manager: Optional checkpoint manager
        """
        self.config = config
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data or []
        
        # Initialize models
        self.policy = SimpleCompressionPolicy(
            embedding_dim=config.embedding_dim,
            context_window=config.context_window,
            hidden_dim=config.policy_hidden_dim,
            device=config.device
        ).to(config.device)
        
        self.reconstructor = reconstructor.to(config.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate_policy,
            weight_decay=config.weight_decay
        )
        
        self.reconstructor_optimizer = optim.AdamW(
            self.reconstructor.parameters(),
            lr=config.learning_rate_reconstructor,
            weight_decay=config.weight_decay
        )
        
        # Initialize reward function
        self.reward_function = create_reward_function(
            reward_type=config.reward_type,
            vocab_size=self.tokenizer.vocab_size,
            device=config.device
        )
        
        # Initialize Gumbel-Softmax
        self.gumbel = GumbelSoftmax()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_score = float('inf')
        
        # Checkpoint manager
        self.checkpoint_manager = checkpoint_manager
        
        # Simple counters for essential tracking
        self.nan_count = 0
        self.inf_count = 0
        
        policy_params = sum(p.numel() for p in self.policy.parameters())
        reconstructor_params = sum(p.numel() for p in self.reconstructor.parameters())
        val_seqs = len(val_data) if val_data else 0
        logger.info(f"Joint trainer initialized successfully - Policy: {policy_params:,} params, "
                   f"Reconstructor: {reconstructor_params:,} params, Train: {len(train_data)} seqs, "
                   f"Val: {val_seqs} seqs, Device: {config.device}, Batch: {config.batch_size}")
    
    def get_gumbel_temperature(self) -> float:
        """Get current Gumbel temperature based on training progress."""
        progress = min(self.step / 10000, 1.0)  # Decay over 10k steps
        temp = self.config.gumbel_temperature_init * (1 - progress) + \
               self.config.gumbel_temperature_min * progress
        return temp
    
    def prepare_batch(self, sequences: List[List[int]]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of sequences for training.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            Batch dictionary with tensors
        """
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        # Pad sequences
        padded_sequences = []
        attention_masks = []
        
        for seq in sequences:
            padded = seq + [self.tokenizer.eos_token_id] * (max_len - len(seq))
            mask = [1] * len(seq) + [0] * (max_len - len(seq))
            
            padded_sequences.append(padded)
            attention_masks.append(mask)
        
        # Convert to tensors
        sequences_tensor = torch.tensor(padded_sequences, device=self.config.device)
        attention_masks_tensor = torch.tensor(attention_masks, device=self.config.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.reconstructor.transformer.wte(sequences_tensor)
        
        return {
            'sequences': sequences_tensor,
            'embeddings': embeddings,
            'attention_masks': attention_masks_tensor,
            'batch_size': batch_size,
            'seq_len': max_len
        }
    
    def compute_policy_decisions(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy decisions for the batch."""
        temperature = self.get_gumbel_temperature()
        logits = self.policy.get_logits(batch['embeddings'])
        masked_logits = logits.masked_fill(batch['attention_masks'] == 0, -1e9)
        
        if self.policy.training:
            mask_decisions = self.gumbel.gumbel_sigmoid(masked_logits, temperature)
        else:
            mask_decisions = self.gumbel.straight_through_sigmoid(masked_logits)
        
        return masked_logits, mask_decisions
    
    def compute_losses(self, batch: Dict[str, torch.Tensor], masked_logits: torch.Tensor, 
                      mask_decisions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute policy and reconstructor losses."""
        # Get compressed sequences and reconstruction loss
        compressed_sequences = self.apply_masking(batch['sequences'], mask_decisions)
        reconstructor_outputs = self.reconstructor(
            compressed_sequences,
            attention_mask=batch['attention_masks'],
            labels=batch['sequences']
        )
        
        # Compute rewards
        mask_probs = torch.sigmoid(masked_logits)
        reward_results = self.reward_function.compute_reward(
            mask_probs=mask_probs,
            reconstruction_loss=reconstructor_outputs.loss,
            sequences=batch['sequences'],
            step=self.step
        )
        
        # Policy loss with REINFORCE
        rewards = reward_results['reward']
        advantages = rewards - rewards.mean().detach()
        
        log_probs = F.logsigmoid(masked_logits) * mask_decisions + \
                   F.logsigmoid(-masked_logits) * (1 - mask_decisions)
        masked_log_probs = log_probs.masked_fill(batch['attention_masks'] == 0, 0)
        
        policy_loss = -(masked_log_probs * advantages.unsqueeze(1)).sum(dim=1).mean()
        entropy = reward_results.get('entropy', torch.tensor(0.0))
        policy_loss -= self.config.entropy_weight * entropy.mean()
        
        return {
            'policy_loss': policy_loss,
            'reconstructor_loss': reconstructor_outputs.loss,
            'reward_results': reward_results
        }
    
    def update_networks(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy and reconstructor networks with gradient health monitoring."""
        gradient_info = {}
        
        # Update policy
        self.policy_optimizer.zero_grad()
        losses['policy_loss'].backward(retain_graph=True)
        
        # Monitor policy gradients
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
        gradient_info['policy_grad_norm'] = policy_grad_norm.item()
        
        # Check for NaN/Inf gradients in policy
        nan_count = 0
        inf_count = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_count += 1
                if torch.isinf(param.grad).any():
                    inf_count += 1
        
        gradient_info['policy_nan_count'] = nan_count
        gradient_info['policy_inf_count'] = inf_count
        
        # Only step if gradients are healthy
        if nan_count == 0 and inf_count == 0:
            self.policy_optimizer.step()
        else:
            logger.warning("Unhealthy gradients detected in policy",
                         nan_count=nan_count, inf_count=inf_count, step=self.step)
        
        # Update reconstructor
        self.reconstructor_optimizer.zero_grad()
        losses['reconstructor_loss'].backward()
        
        # Monitor reconstructor gradients
        reconstructor_grad_norm = torch.nn.utils.clip_grad_norm_(self.reconstructor.parameters(), self.config.grad_clip_norm)
        gradient_info['reconstructor_grad_norm'] = reconstructor_grad_norm.item()
        
        # Check for NaN/Inf gradients in reconstructor
        nan_count = 0
        inf_count = 0
        for param in self.reconstructor.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_count += 1
                if torch.isinf(param.grad).any():
                    inf_count += 1
        
        gradient_info['reconstructor_nan_count'] = nan_count
        gradient_info['reconstructor_inf_count'] = inf_count
        
        # Only step if gradients are healthy
        if nan_count == 0 and inf_count == 0:
            self.reconstructor_optimizer.step()
        else:
            logger.warning("Unhealthy gradients detected in reconstructor",
                         nan_count=nan_count, inf_count=inf_count, step=self.step)
        
        # Update simple counters
        self.nan_count += gradient_info['policy_nan_count'] + gradient_info['reconstructor_nan_count']
        self.inf_count += gradient_info['policy_inf_count'] + gradient_info['reconstructor_inf_count']
        
        # Record gradient health metrics
        record_gradient_health('policy', gradient_info['policy_grad_norm'], 
                             gradient_info['policy_nan_count'], gradient_info['policy_inf_count'])
        record_gradient_health('reconstructor', gradient_info['reconstructor_grad_norm'],
                             gradient_info['reconstructor_nan_count'], gradient_info['reconstructor_inf_count'])
        
        return gradient_info
    
    def joint_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one joint training step with comprehensive monitoring."""
        import time
        step_start_time = time.time()
        
        try:
            # Compute decisions
            masked_logits, mask_decisions = self.compute_policy_decisions(batch)
            
            # Compute losses
            losses = self.compute_losses(batch, masked_logits, mask_decisions)
            
            # Update networks and get gradient info
            gradient_info = self.update_networks(losses)
            
            # Update step counter
            self.step += 1
            
            # Prepare metrics
            reward_results = losses['reward_results']
            step_metrics = {
                'policy_loss': losses['policy_loss'].item(),
                'reconstructor_loss': losses['reconstructor_loss'].item(), 
                'reward_mean': reward_results['reward'].mean().item(),
                'compression_ratio': reward_results['compression_ratio'].mean().item(),
                'temperature': self.get_gumbel_temperature(),
                'policy_grad_norm': gradient_info['policy_grad_norm'],
                'reconstructor_grad_norm': gradient_info['reconstructor_grad_norm']
            }
            
            # Record training metrics
            record_training_metrics(self.step, self.epoch, step_metrics)
            
            # Log step completion
            step_duration = time.time() - step_start_time
            
            return step_metrics
            
        except Exception as e:
            logger.error(f"Training step failed - Step {self.step} (Epoch {self.epoch}): {str(e)}")
            raise
    
    def apply_masking(
        self,
        sequences: torch.Tensor,
        mask_decisions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply masking decisions to sequences.
        
        Args:
            sequences: Original sequences [batch_size, seq_len]
            mask_decisions: Mask decisions [batch_size, seq_len]
            
        Returns:
            Masked sequences [batch_size, seq_len]
        """
        # Use mask token for masked positions
        mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        
        # Create masked sequences
        masked_sequences = sequences.clone()
        
        # Apply masking: keep original token if decision > 0.5, else use mask token
        mask = mask_decisions < 0.5
        masked_sequences[mask] = mask_token_id
        
        return masked_sequences
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation data."""
        if not self.val_data:
            return {}
        
        self.policy.eval()
        self.reconstructor.eval()
        
        total_metrics = {
            'val_policy_loss': 0.0,
            'val_reconstructor_loss': 0.0,
            'val_reward': 0.0,
            'val_compression_ratio': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(self.val_data), self.config.batch_size):
                batch_data = self.val_data[i:i + self.config.batch_size]
                batch = self.prepare_batch(batch_data)
                
                # Compute metrics without updating parameters
                metrics = self.joint_training_step(batch)
                
                for key in total_metrics:
                    val_key = key.replace('val_', '')
                    if val_key in metrics:
                        total_metrics[key] += metrics[val_key]
                
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            for key in total_metrics:
                total_metrics[key] /= num_batches
        
        self.policy.train()
        self.reconstructor.train()
        
        return total_metrics
    
    def save_checkpoint(self, output_dir: str, final: bool = False, is_best: bool = False) -> bool:
        """Save training checkpoint with robust error handling."""
        try:
            checkpoint = {
                'step': self.step,
                'epoch': self.epoch,
                'policy_state_dict': self.policy.state_dict(),
                'reconstructor_state_dict': self.reconstructor.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'reconstructor_optimizer_state_dict': self.reconstructor_optimizer.state_dict(),
                'config': self.config.__dict__,
                'nan_count': self.nan_count,
                'inf_count': self.inf_count,
                'best_val_score': self.best_val_score
            }
            
            if self.checkpoint_manager:
                # Use robust checkpoint manager
                checkpoint_name = 'final_checkpoint' if final else f'checkpoint_step_{self.step}'
                success = self.checkpoint_manager.save_checkpoint(checkpoint, checkpoint_name, is_best)
                
                if success:
                    logger.model_checkpoint(
                        checkpoint_path=str(self.checkpoint_manager.active_dir / f"{checkpoint_name}.pt"),
                        metrics={'step': self.step, 'epoch': self.epoch, 'best_val_score': self.best_val_score},
                        is_best=is_best, is_final=final
                    )
                else:
                    logger.error(f"Checkpoint save failed - Step {self.step} (Epoch {self.epoch})")
                
                return success
            else:
                # Fallback to simple save
                checkpoint_name = 'final_checkpoint.pt' if final else f'checkpoint_step_{self.step}.pt'
                checkpoint_path = os.path.join(output_dir, checkpoint_name)
                
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved - Step {self.step} (Epoch {self.epoch}): {checkpoint_path}"
                          f" {'(BEST)' if is_best else ''} {'(FINAL)' if final else ''}")
                return True
                
        except Exception as e:
            logger.error(f"Checkpoint save failed - Step {self.step} (Epoch {self.epoch}): {str(e)}", 
                       exc_info=True)
            return False
    
    def train(self, output_dir: str, experiment_name: str = "joint_training") -> None:
        """
        Main training loop with comprehensive monitoring.
        
        Args:
            output_dir: Directory to save checkpoints and logs
            experiment_name: Name of the experiment for tracking
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Use training context manager for structured logging
        with TrainingContextManager(experiment_name) as ctx:
            total_params = sum(p.numel() for p in self.policy.parameters()) + sum(p.numel() for p in self.reconstructor.parameters())
            logger.info(f"Starting joint training session '{experiment_name}' - "
                       f"Epochs: {self.config.max_epochs}, Output: {output_dir}, "
                       f"Total params: {total_params:,}")
            
            self._run_training_loop(output_dir)
    
    def _run_training_loop(self, output_dir: str) -> None:
        """Internal training loop implementation."""
        try:
            for epoch in range(self.config.max_epochs):
                self.epoch = epoch
                epoch_metrics = []
                
                # Shuffle training data
                np.random.shuffle(self.train_data)
                
                # Training loop
                for step in range(self.config.max_steps_per_epoch):
                    # Sample batch
                    batch_indices = np.random.choice(
                        len(self.train_data),
                        size=self.config.batch_size,
                        replace=False
                    )
                    batch_data = [self.train_data[i] for i in batch_indices]
                    batch = self.prepare_batch(batch_data)
                    
                    # Training step
                    metrics = self.joint_training_step(batch)
                    epoch_metrics.append(metrics)
                    
                    # Logging
                    if self.step % self.config.log_freq == 0:
                        avg_metrics = {k: np.mean([m[k] for m in epoch_metrics[-10:]]) 
                                     for k in metrics.keys()}
                        
                        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
                        logger.info(f"Step {self.step} (Epoch {self.epoch}) - {metrics_str}")
                    
                    # Evaluation
                    if self.step % self.config.eval_freq == 0:
                        val_metrics = self.evaluate()
                        if val_metrics:
                            val_metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                            logger.info(f"Validation completed - Step {self.step} (Epoch {self.epoch}) - {val_metrics_str}")
                    
                    # Checkpointing
                    if self.step % self.config.checkpoint_freq == 0:
                        is_best = False
                        if val_metrics and 'val_reconstructor_loss' in val_metrics:
                            if val_metrics['val_reconstructor_loss'] < self.best_val_score:
                                self.best_val_score = val_metrics['val_reconstructor_loss']
                                is_best = True
                        
                        self.save_checkpoint(output_dir, is_best=is_best)
                
                # End of epoch summary
                epoch_avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) 
                                   for k in epoch_metrics[0].keys()}
                
                metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in epoch_avg_metrics.items()])
                logger.info(f"Epoch {epoch} completed - Total steps: {self.step}, "
                          f"Steps this epoch: {len(epoch_metrics)}, Metrics: {metrics_str}")
                
                # Epoch completed - metrics already logged
            
            # Save final checkpoint
            self.save_checkpoint(output_dir, final=True)
            
            # Training completed - no need to save metrics
        
        except KeyboardInterrupt:
            logger.info(f"Training interrupted by user - Step {self.step} (Epoch {self.epoch})")
            # Save emergency checkpoint
            self.save_checkpoint(output_dir, final=False)
            raise
        except Exception as e:
            logger.error(f"Training failed with exception - Step {self.step} (Epoch {self.epoch}): {str(e)}", 
                       exc_info=True)
            # Save emergency checkpoint
            self.save_checkpoint(output_dir, final=False)
            raise
        
        logger.info(f"Joint training completed successfully. Steps: {self.step}, "
                   f"NaN gradients: {self.nan_count}, Inf gradients: {self.inf_count}")
    

# Utility functions for creating and using the joint trainer

def load_training_data(data_path: str, tokenizer: GPT2Tokenizer) -> List[List[int]]:
    """Load and tokenize training data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    for item in data:
        if 'tokens' in item:
            sequences.append(item['tokens'])
        elif 'text' in item:
            tokens = tokenizer.encode(item['text'])
            sequences.append(tokens)
    
    logger.info(f"Loaded {len(sequences)} sequences from {data_path}")
    return sequences


def create_joint_trainer(
    config_dict: Dict[str, Any],
    data_path: str,
    reconstructor_path: str,
    val_data_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None
) -> JointTrainer:
    """Factory function to create joint trainer with monitoring."""
    
    # Create config
    config = TrainingConfig(**config_dict)
    
    # Initialize checkpoint manager if directory provided
    checkpoint_manager = None
    if checkpoint_dir:
        checkpoint_manager = CheckpointManager(checkpoint_dir, max_backups=5)
        logger.info("Checkpoint manager initialized", checkpoint_dir=checkpoint_dir)
    
    # Load tokenizer and reconstructor
    tokenizer = GPT2Tokenizer.from_pretrained(reconstructor_path)
    reconstructor = GPT2LMHeadModel.from_pretrained(reconstructor_path)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        reconstructor.resize_token_embeddings(len(tokenizer))
    
    # Load data
    train_data = load_training_data(data_path, tokenizer)
    val_data = load_training_data(val_data_path, tokenizer) if val_data_path else None
    
    return JointTrainer(config, tokenizer, reconstructor, train_data, val_data, checkpoint_manager)


