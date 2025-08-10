"""
Simplified joint trainer - joint training without over-engineering.

Core idea: Train both networks on the same batch. Period.
- Policy network learns to compress  
- Reconstructor learns to reconstruct compressed sequences
- Simple reward: balance compression vs reconstruction quality
- No target networks, no Gumbel-Softmax, no variance reduction tricks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Any
import os
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.agent import SimpleCompressionPolicy
from utils.deterministic import set_global_seed, make_model_deterministic


class SimpleJointTrainer:
    """Simplified joint trainer for policy and reconstructor."""
    
    def __init__(
        self,
        policy_lr: float = 1e-3,
        reconstructor_lr: float = 1e-4, 
        device: str = "cpu",
        context_window: int = 5,
        random_seed: int = 42,
        micro_batch_size: int = 2,
        gradient_accumulation_steps: int = 8
    ):
        self.device = device
        self.context_window = context_window
        self.random_seed = random_seed
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Set global seed for deterministic behavior
        set_global_seed(random_seed, deterministic_algorithms=True)
        
        # Simple optimizers
        self.policy_lr = policy_lr
        self.reconstructor_lr = reconstructor_lr
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        print(f"✓ Initialized trainer with micro_batch_size={micro_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
        print(f"  Effective batch size: {micro_batch_size * gradient_accumulation_steps}")
        
    def setup_models(self, tokenizer: GPT2Tokenizer, reconstructor: GPT2LMHeadModel):
        """Initialize models and optimizers."""
        # Policy network - keep it simple and deterministic
        self.policy = SimpleCompressionPolicy(
            embedding_dim=768,  # GPT-2 embedding size
            context_window=self.context_window,
            device=self.device
        ).to(self.device)
        
        # Make models deterministic
        self.policy = make_model_deterministic(self.policy, self.random_seed)
        self.reconstructor = make_model_deterministic(reconstructor, self.random_seed + 1)
        self.reconstructor = self.reconstructor.to(self.device)
        self.tokenizer = tokenizer
        
        # Simple optimizers - no fancy scheduling
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.reconstructor_optimizer = optim.Adam(self.reconstructor.parameters(), lr=self.reconstructor_lr)
    
    def clear_memory(self):
        """Clear device memory - essential for MPS memory management."""
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_info(self):
        """Get current memory usage for monitoring."""
        if self.device == "mps" and torch.backends.mps.is_available():
            return f"MPS device memory usage monitoring not available"
        elif self.device == "cuda" and torch.cuda.is_available():
            return f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB max"
        else:
            return "CPU device - no GPU memory tracking"
    
    def train_step(self, batch_sequences: List[List[int]]) -> Dict[str, float]:
        """One joint training step with proper gradient accumulation."""
        
        # Initialize optimizers
        self.policy_optimizer.zero_grad()
        self.reconstructor_optimizer.zero_grad()
        
        # Initialize accumulators
        total_policy_loss = 0.0
        total_reconstructor_loss = 0.0
        total_reward = 0.0
        total_compression_ratio = 0.0
        num_micro_batches = 0
        
        # Process micro-batches for gradient accumulation
        for i in range(0, len(batch_sequences), self.micro_batch_size):
            micro_batch = batch_sequences[i:i + self.micro_batch_size]
            
            # Memory management - clear cache before each micro-batch
            self.clear_memory()
            
            # Convert micro-batch to tensors - create directly on device for efficiency
            max_len = max(len(seq) for seq in micro_batch)
            sequences = torch.zeros(len(micro_batch), max_len, dtype=torch.long, device=self.device)
            attention_masks = torch.zeros(len(micro_batch), max_len, dtype=torch.long, device=self.device)
            
            # Efficient tensor creation - avoid intermediate CPU tensors
            for j, seq in enumerate(micro_batch):
                seq_len = len(seq)
                sequences[j, :seq_len] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attention_masks[j, :seq_len] = 1
            
            # Get policy decisions - simple thresholding, no Gumbel-Softmax
            with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for MPS compatibility
                embeddings = self.reconstructor.transformer.wte(sequences)  # Get embeddings
                policy_logits = self.policy(embeddings)  # [micro_batch_size, seq_len]
                
                # Handle policy output shape (squeeze extra dimensions if needed)
                if policy_logits.dim() == 3 and policy_logits.size(-1) == 1:
                    policy_logits = policy_logits.squeeze(-1)
                
                # Binary decisions: > 0.5 means keep token
                keep_probs = torch.sigmoid(policy_logits)
                keep_mask = (keep_probs > 0.5).float()
                
                # Apply masking - efficient in-place masking without clone
                mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
                masked_sequences = torch.where(keep_mask == 1, sequences, mask_token_id)
                
                # Compute reconstruction loss
                reconstructor_outputs = self.reconstructor(
                    input_ids=masked_sequences,
                    attention_mask=attention_masks,
                    labels=sequences  # Original sequences as targets
                )
                reconstructor_loss = reconstructor_outputs.loss
                
                # Simple reward: balance compression vs reconstruction
                # Ensure we handle edge cases with proper dimension handling
                kept_tokens = keep_mask.sum(dim=1)  # [batch_size]
                total_tokens = attention_masks.sum(dim=1)  # [batch_size]
                # Avoid division by zero
                compression_ratio = kept_tokens / (total_tokens + 1e-8)
                compression_reward = -compression_ratio.mean()
                
                # Policy loss: REINFORCE with reconstruction loss as reward signal
                reward = compression_reward - reconstructor_loss.detach()
                # Broadcast reward properly for element-wise multiplication
                policy_loss = -(torch.log(keep_probs + 1e-8) * keep_mask * reward).mean()
                
                # Total loss for joint optimization
                total_loss = policy_loss + reconstructor_loss
                
                # Scale loss by gradient accumulation steps
                scaled_loss = total_loss / self.gradient_accumulation_steps
                
            # Backward pass with gradient accumulation
            scaled_loss.backward()
            
            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_reconstructor_loss += reconstructor_loss.item()
            total_reward += reward.item()
            total_compression_ratio += compression_ratio.mean().item()
            num_micro_batches += 1
            
            # Clear intermediate tensors to save memory
            del sequences, attention_masks, embeddings, masked_sequences
            del policy_logits, keep_probs, keep_mask, reconstructor_outputs
            del compression_ratio, reward, policy_loss, reconstructor_loss
            
            # Memory management - clear cache after processing
            self.clear_memory()
        
        # Step optimizers after gradient accumulation
        self.policy_optimizer.step()
        self.reconstructor_optimizer.step()
        
        self.step += 1
        
        # Return averaged metrics
        return {
            'policy_loss': total_policy_loss / num_micro_batches,
            'reconstructor_loss': total_reconstructor_loss / num_micro_batches, 
            'total_loss': (total_policy_loss + total_reconstructor_loss) / num_micro_batches,
            'reward': total_reward / num_micro_batches,
            'compression_ratio': total_compression_ratio / num_micro_batches
        }
    
    def train(self, train_data: List[List[int]], batch_size: int = 16, max_epochs: int = 50, 
              output_dir: str = "outputs/training"):
        """Simple training loop."""
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Simple batching
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                metrics = self.train_step(batch)
                epoch_losses.append(metrics)
                
                # Enhanced logging every 100 steps with memory monitoring
                if self.step % 100 == 0:
                    memory_info = self.get_memory_info()
                    print(f"Step {self.step}: Policy Loss: {metrics['policy_loss']:.4f}, "
                          f"Recon Loss: {metrics['reconstructor_loss']:.4f}, "
                          f"Compression: {metrics['compression_ratio']:.3f}")
                    print(f"  Memory: {memory_info}")
            
            # Epoch summary
            avg_policy_loss = np.mean([m['policy_loss'] for m in epoch_losses])
            avg_recon_loss = np.mean([m['reconstructor_loss'] for m in epoch_losses])
            avg_compression = np.mean([m['compression_ratio'] for m in epoch_losses])
            
            print(f"Epoch {epoch+1}/{max_epochs} - "
                  f"Policy: {avg_policy_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
                  f"Compression: {avg_compression:.3f}")
            
            # Simple checkpointing every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Save final model
        self.save_checkpoint(os.path.join(output_dir, "final_model.pt"))
        print(f"Training complete. Models saved to {output_dir}")
    
    def save_checkpoint(self, path: str):
        """Simple checkpoint saving."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'reconstructor_state_dict': self.reconstructor.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'reconstructor_optimizer_state_dict': self.reconstructor_optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch
        }, path)


def create_simple_trainer(config: Dict[str, Any], train_data: List[List[int]], 
                         reconstructor: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> SimpleJointTrainer:
    """Create simplified trainer from config with deterministic behavior."""
    # Extract gradient accumulation parameters from config
    micro_batch_size = config.get('micro_batch_size', 2)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)
    
    trainer = SimpleJointTrainer(
        policy_lr=config.get('learning_rate_policy', 1e-3),
        reconstructor_lr=config.get('learning_rate_reconstructor', 1e-4),
        device=config.get('device', 'cpu'),
        context_window=config.get('context_window', 5),
        random_seed=config.get('random_seed', 42),
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    trainer.setup_models(tokenizer, reconstructor)
    return trainer