"""
Simplified joint trainer - joint training without over-engineering.

Core idea: Train both networks on the same batch. Period.
- Policy network learns to compress
- Reconstructor learns to reconstruct compressed sequences
- Simple reward: balance compression vs reconstruction quality
- No target networks, no Gumbel-Softmax, no variance reduction tricks
"""

import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from models.agent import SimpleCompressionPolicy
from utils.deterministic import make_model_deterministic, set_global_seed
from utils.logging import get_component_logger
from utils.simple_memory import SimpleMemoryHandler

logger = get_component_logger("TRAINING")


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
        gradient_accumulation_steps: int = 8,
        use_accelerate: bool = False,
        mixed_precision: str = None,
        gradient_checkpointing: bool = False,
        deepspeed_config: str = None,
    ):
        logger.info("=" * 60)
        logger.info("INITIALIZING SIMPLE JOINT TRAINER")
        logger.info("=" * 60)

        self.device = device
        self.context_window = context_window
        self.random_seed = random_seed
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_accelerate = use_accelerate
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.deepspeed_config = deepspeed_config

        # Initialize accelerator if requested
        if use_accelerate:
            deepspeed_plugin = None
            if deepspeed_config:
                from accelerate import DeepSpeedPlugin

                deepspeed_plugin = DeepSpeedPlugin(config_file=deepspeed_config)

            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=mixed_precision,
                log_with=["tensorboard"],
                deepspeed_plugin=deepspeed_plugin,
            )
            self.device = self.accelerator.device
            logger.info(f"Accelerate initialized - Device: {self.device}, Mixed precision: {mixed_precision}")
            if deepspeed_config:
                logger.info(f"DeepSpeed config: {deepspeed_config}")
        else:
            self.accelerator = None

        logger.info("Training configuration:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Policy LR: {policy_lr}")
        logger.info(f"  Reconstructor LR: {reconstructor_lr}")
        logger.info(f"  Context window: {context_window}")
        logger.info(f"  Random seed: {random_seed}")
        logger.info(f"  Micro batch size: {micro_batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {micro_batch_size * gradient_accumulation_steps}")

        # Set global seed for deterministic behavior
        logger.info("Setting global random seed for deterministic training...")
        set_global_seed(random_seed, deterministic_algorithms=True)

        # Simple optimizers
        self.policy_lr = policy_lr
        self.reconstructor_lr = reconstructor_lr

        # Training state
        self.step = 0
        self.epoch = 0

        # Simple memory handler for OOM situations only
        self.memory_handler = SimpleMemoryHandler(device)
        logger.info(f"Initialized simple memory handler for {device}")

        print(
            f"✓ Initialized trainer with micro_batch_size={micro_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}"
        )
        print(f"  Effective batch size: {micro_batch_size * gradient_accumulation_steps}")
        logger.info("Trainer initialization complete")

    def setup_models(self, tokenizer: GPT2Tokenizer, reconstructor: GPT2LMHeadModel):
        """Initialize models and optimizers."""
        # Policy network - keep it simple and deterministic
        self.policy = SimpleCompressionPolicy(
            embedding_dim=768,  # GPT-2 embedding size
            context_window=self.context_window,
            device=self.device,
        ).to(self.device)

        # Make models deterministic
        self.policy = make_model_deterministic(self.policy, self.random_seed)
        self.reconstructor = make_model_deterministic(reconstructor, self.random_seed + 1)
        self.reconstructor = self.reconstructor.to(self.device)
        self.tokenizer = tokenizer

        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing:
            if hasattr(self.policy, "gradient_checkpointing_enable"):
                self.policy.gradient_checkpointing_enable()
                logger.info("Policy gradient checkpointing enabled")
            if hasattr(self.reconstructor, "gradient_checkpointing_enable"):
                self.reconstructor.gradient_checkpointing_enable()
                logger.info("Reconstructor gradient checkpointing enabled")

        # Simple optimizers - no fancy scheduling
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.reconstructor_optimizer = optim.Adam(self.reconstructor.parameters(), lr=self.reconstructor_lr)

        # Prepare models with accelerate if using it
        if self.use_accelerate:
            (
                self.policy,
                self.reconstructor,
                self.policy_optimizer,
                self.reconstructor_optimizer,
            ) = self.accelerator.prepare(
                self.policy,
                self.reconstructor,
                self.policy_optimizer,
                self.reconstructor_optimizer,
            )
            logger.info("Models and optimizers prepared with accelerate")

    def get_memory_info(self):
        """Get current memory usage for monitoring."""
        if self.device == "cuda" and torch.cuda.is_available():
            return f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
        elif self.device.startswith("mps") and hasattr(torch.mps, "driver_allocated_memory"):
            try:
                allocated = torch.mps.driver_allocated_memory() / (1024**3)
                return f"MPS memory: {allocated:.2f}GB"
            except Exception:
                return "MPS device"
        else:
            return "CPU device"

    def _forward_pass(self, micro_batch: List[List[int]]):
        """Forward pass for a micro-batch - returns losses and metrics."""
        # Convert to tensors
        max_len = max(len(seq) for seq in micro_batch)
        sequences = torch.zeros(len(micro_batch), max_len, dtype=torch.long, device=self.device)
        attention_masks = torch.zeros(len(micro_batch), max_len, dtype=torch.long, device=self.device)

        for j, seq in enumerate(micro_batch):
            seq_len = len(seq)
            sequences[j, :seq_len] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attention_masks[j, :seq_len] = 1

        # Forward pass
        embeddings = self.reconstructor.transformer.wte(sequences)
        policy_logits = self.policy(embeddings)

        if policy_logits.dim() == 3 and policy_logits.size(-1) == 1:
            policy_logits = policy_logits.squeeze(-1)

        # Binary decisions
        keep_probs = torch.sigmoid(policy_logits)
        keep_mask = (keep_probs > 0.5).float()

        # Apply masking
        mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        masked_sequences = torch.where(keep_mask == 1, sequences, mask_token_id)

        # Compute losses
        reconstructor_outputs = self.reconstructor(
            input_ids=masked_sequences, attention_mask=attention_masks, labels=sequences
        )
        reconstructor_loss = reconstructor_outputs.loss

        # Compute compression ratio and reward
        kept_tokens = keep_mask.sum(dim=1)
        total_tokens = attention_masks.sum(dim=1)
        compression_ratio = kept_tokens / (total_tokens + 1e-8)
        compression_reward = -compression_ratio.mean()

        # Policy loss
        reward = compression_reward - reconstructor_loss.detach()
        policy_loss = -(torch.log(keep_probs + 1e-8) * keep_mask * reward).mean()

        return (
            policy_loss,
            reconstructor_loss,
            reward.item(),
            compression_ratio.mean().item(),
        )

    def train_step(self, batch_sequences: List[List[int]]) -> Dict[str, float]:
        """Simple training step - with optional accelerate support."""

        # Initialize accumulators
        total_policy_loss = 0.0
        total_reconstructor_loss = 0.0
        total_reward = 0.0
        total_compression_ratio = 0.0
        num_micro_batches = 0

        # Use accelerate's gradient accumulation if available
        if self.use_accelerate:
            with self.accelerator.accumulate(self.policy, self.reconstructor):
                # Process micro-batches for gradient accumulation
                for i in range(0, len(batch_sequences), self.micro_batch_size):
                    micro_batch = batch_sequences[i : i + self.micro_batch_size]

                    policy_loss, reconstructor_loss, reward, compression_ratio = self._forward_pass(micro_batch)

                    # Backward pass with accelerate
                    total_loss = policy_loss + reconstructor_loss
                    self.accelerator.backward(total_loss)

                    # Accumulate metrics
                    total_policy_loss += policy_loss.item()
                    total_reconstructor_loss += reconstructor_loss.item()
                    total_reward += reward
                    total_compression_ratio += compression_ratio
                    num_micro_batches += 1

                # Step optimizers
                self.accelerator.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.accelerator.clip_grad_norm_(self.reconstructor.parameters(), max_norm=1.0)
                self.policy_optimizer.step()
                self.reconstructor_optimizer.step()
                self.policy_optimizer.zero_grad()
                self.reconstructor_optimizer.zero_grad()
        else:
            # Original training loop without accelerate
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.reconstructor_optimizer.zero_grad(set_to_none=True)

            # Process micro-batches for gradient accumulation
            for i in range(0, len(batch_sequences), self.micro_batch_size):
                micro_batch = batch_sequences[i : i + self.micro_batch_size]

                try:
                    policy_loss, reconstructor_loss, reward, compression_ratio = self._forward_pass(micro_batch)

                    # Total loss
                    total_loss = policy_loss + reconstructor_loss
                    scaled_loss = total_loss / self.gradient_accumulation_steps

                    # Backward pass
                    scaled_loss.backward()

                    # Accumulate metrics
                    total_policy_loss += policy_loss.item()
                    total_reconstructor_loss += reconstructor_loss.item()
                    total_reward += reward
                    total_compression_ratio += compression_ratio
                    num_micro_batches += 1

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"OOM in micro-batch {i // self.micro_batch_size + 1}: {e}")
                        # Only intervene on actual OOM
                        if self.memory_handler.handle_oom():
                            continue  # Try next batch
                        else:
                            raise  # Too many OOM errors, give up
                    else:
                        raise  # Re-raise non-memory errors

            # Step optimizers
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.reconstructor.parameters(), max_norm=1.0)

            self.policy_optimizer.step()
            self.reconstructor_optimizer.step()

        self.step += 1

        # Return averaged metrics
        return {
            "policy_loss": total_policy_loss / num_micro_batches,
            "reconstructor_loss": total_reconstructor_loss / num_micro_batches,
            "total_loss": (total_policy_loss + total_reconstructor_loss) / num_micro_batches,
            "reward": total_reward / num_micro_batches,
            "compression_ratio": total_compression_ratio / num_micro_batches,
        }

    def train(
        self,
        train_data: List[List[int]],
        batch_size: int = 16,
        max_epochs: int = 50,
        output_dir: str = "outputs/training",
    ):
        """Simple training loop."""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING LOOP")
        logger.info("=" * 60)
        logger.info("Training parameters:")
        logger.info(f"  Training sequences: {len(train_data)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max epochs: {max_epochs}")
        logger.info(f"  Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        # Calculate training info
        total_batches = (len(train_data) + batch_size - 1) // batch_size
        total_steps = total_batches * max_epochs
        logger.info(f"Training will run {total_batches} batches per epoch, {total_steps} total steps")

        training_start = time.time()

        for epoch in range(max_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{max_epochs}")
            epoch_start = time.time()
            self.epoch = epoch
            epoch_losses = []

            # Simple batching
            for i in range(0, len(train_data), batch_size):
                batch_start = time.time()
                batch = train_data[i : i + batch_size]

                # Log batch info
                batch_num = i // batch_size + 1
                if batch_num % 10 == 1:  # Log every 10th batch
                    logger.info(
                        f"Processing batch {batch_num}/{total_batches} (sequences {i}-{min(i + batch_size, len(train_data))})"
                    )

                metrics = self.train_step(batch)
                epoch_losses.append(metrics)

                batch_time = time.time() - batch_start

                # Enhanced logging every 100 steps with memory monitoring
                if self.step % 100 == 0:
                    memory_info = self.get_memory_info()
                    elapsed = time.time() - training_start
                    steps_per_sec = self.step / elapsed if elapsed > 0 else 0
                    eta = (total_steps - self.step) / steps_per_sec if steps_per_sec > 0 else 0

                    logger.info(
                        f"Step {self.step}/{total_steps}: "
                        f"Policy={metrics['policy_loss']:.4f}, "
                        f"Recon={metrics['reconstructor_loss']:.4f}, "
                        f"Comp={metrics['compression_ratio']:.3f}, "
                        f"Batch_time={batch_time:.2f}s, "
                        f"ETA={eta / 3600:.1f}h"
                    )
                    logger.info(f"  Memory: {memory_info}")

                    print(
                        f"Step {self.step}: Policy Loss: {metrics['policy_loss']:.4f}, "
                        f"Recon Loss: {metrics['reconstructor_loss']:.4f}, "
                        f"Compression: {metrics['compression_ratio']:.3f}"
                    )
                    print(f"  Memory: {memory_info}")

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_policy_loss = np.mean([m["policy_loss"] for m in epoch_losses])
            avg_recon_loss = np.mean([m["reconstructor_loss"] for m in epoch_losses])
            avg_compression = np.mean([m["compression_ratio"] for m in epoch_losses])

            logger.info(
                f"Epoch {epoch + 1} complete in {epoch_time:.2f}s - "
                f"Avg Policy: {avg_policy_loss:.4f}, "
                f"Avg Recon: {avg_recon_loss:.4f}, "
                f"Avg Compression: {avg_compression:.3f}"
            )

            print(
                f"Epoch {epoch + 1}/{max_epochs} - "
                f"Policy: {avg_policy_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
                f"Compression: {avg_compression:.3f}"
            )

            # Simple checkpointing every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

        # Save final model
        self.save_checkpoint(os.path.join(output_dir, "final_model.pt"))
        print(f"Training complete. Models saved to {output_dir}")

    def save_checkpoint(self, path: str):
        """Simple checkpoint saving."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "reconstructor_state_dict": self.reconstructor.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "reconstructor_optimizer_state_dict": self.reconstructor_optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            },
            path,
        )


def create_simple_trainer(
    config: Dict[str, Any],
    train_data: List[List[int]],
    reconstructor: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
) -> SimpleJointTrainer:
    """Create simplified trainer from config with deterministic behavior."""
    # Extract gradient accumulation parameters from config
    micro_batch_size = config.get("micro_batch_size", 2)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)

    trainer = SimpleJointTrainer(
        policy_lr=config.get("learning_rate_policy", 1e-3),
        reconstructor_lr=config.get("learning_rate_reconstructor", 1e-4),
        device=config.get("device", "cpu"),
        context_window=config.get("context_window", 5),
        random_seed=config.get("random_seed", 42),
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_accelerate=config.get("use_accelerate", False),
        mixed_precision=config.get("mixed_precision", None),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        deepspeed_config=config.get("deepspeed_config", None),
    )
    trainer.setup_models(tokenizer, reconstructor)
    return trainer
