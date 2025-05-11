import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import random


class MaskedSequenceDataset(Dataset):
    """Dataset for fine-tuning GPT-2 on the task of reconstructing masked sequences."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: GPT2Tokenizer,
        max_length: int = 1024,
        mask_ratio: float = 0.3
    ):
        """
        Args:
            data_path: Path to tokenized data
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
            mask_ratio: Ratio of tokens to mask
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        
        # Load data
        with open(data_path, "r") as f:
            self.data = json.load(f)
            
        # Filter sequences and ensure they are valid
        self.sequences = []
        for seq in self.data:
            try:
                tokens = seq["tokens"]
                if not isinstance(tokens, list):
                    print(f"Warning: tokens is not a list: {type(tokens)}")
                    continue
                if not all(isinstance(t, int) for t in tokens):
                    print(f"Warning: tokens contains non-integer values")
                    continue
                if len(tokens) >= 10:  # Ensure minimum length
                    self.sequences.append(tokens)
            except (KeyError, TypeError) as e:
                print(f"Warning: Skipping invalid sequence: {e}")
        
        print(f"Loaded {len(self.sequences)} valid sequences")
        
        # Special tokens
        self.mask_token_id = tokenizer.eos_token_id  # Using EOS as a mask token
        self.pad_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            tokens = self.sequences[idx]
            
            # Truncate if necessary
            if len(tokens) > self.max_length:
                start_idx = random.randint(0, len(tokens) - self.max_length)
                tokens = tokens[start_idx:start_idx + self.max_length]
            
            # Create random mask
            mask = torch.rand(len(tokens)) < self.mask_ratio
            
            # Apply mask
            original_tokens = torch.tensor(tokens, dtype=torch.long)
            masked_tokens = original_tokens.clone()
            masked_tokens[mask] = self.mask_token_id
            
            # Pad sequences to max_length
            padding_length = self.max_length - len(tokens)
            if padding_length > 0:
                original_tokens = torch.cat([
                    original_tokens,
                    torch.full((padding_length,), fill_value=self.pad_token_id, dtype=torch.long)
                ])
                masked_tokens = torch.cat([
                    masked_tokens,
                    torch.full((padding_length,), fill_value=self.pad_token_id, dtype=torch.long)
                ])
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones_like(masked_tokens)
            if padding_length > 0:
                attention_mask[-padding_length:] = 0
            
            return {
                "input_ids": masked_tokens,
                "labels": original_tokens,
                "attention_mask": attention_mask
            }
        except Exception as e:
            print(f"Error processing sequence at index {idx}: {e}")
            print(f"Sequence length: {len(tokens) if 'tokens' in locals() else 'N/A'}")
            raise


def train_reconstructor(
    data_path: str,
    output_dir: str,
    model_name: str = "gpt2",
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    max_grad_norm: float = 1.0,
    mask_ratio: float = 0.3,
    warmup_steps: int = 500,
    device: str = None,
    num_workers: int = 4,
    gradient_accumulation_steps: int = 1,
    max_length: int = 512,  # Reduced from 1024
    use_amp: bool = True,  # Automatic Mixed Precision
    memory_efficient_attention: bool = True
):
    """
    Fine-tune GPT-2 for token reconstruction.
    
    Args:
        data_path: Path to tokenized data
        output_dir: Directory to save the fine-tuned model
        model_name: Pretrained model name
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        mask_ratio: Ratio of tokens to mask
        warmup_steps: Number of warmup steps for learning rate scheduler
        device: Device to run training on (cuda, mps, or cpu)
        num_workers: Number of workers for data loading
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_length: Maximum sequence length
        use_amp: Whether to use Automatic Mixed Precision
        memory_efficient_attention: Whether to use memory-efficient attention
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    
    # Initialize model with memory-efficient options
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_amp and device in ["cuda", "mps"] else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    if memory_efficient_attention:
        model.config.use_cache = False
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None
    
    # Create dataset and dataloader
    dataset = MaskedSequenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        mask_ratio=mask_ratio,
        max_length=max_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device in ["cuda", "mps"] else False
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    global_step = 0
    training_stats = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass with mixed precision
                if use_amp and device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if use_amp and device == "cuda":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    if use_amp and device == "cuda":
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress
                epoch_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
                global_step += 1
                
                # Save checkpoint
                if global_step % 1000 == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if device == "mps":
                        print("MPS out of memory. Trying to recover...")
                        torch.mps.empty_cache()
                    elif device == "cuda":
                        print("CUDA out of memory. Trying to recover...")
                        torch.cuda.empty_cache()
                    continue
                raise e
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        training_stats.append({
            "epoch": epoch + 1,
            "average_loss": avg_loss
        })
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training stats
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to tokenized data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.3,
                        help="Ratio of tokens to mask")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run training on (cuda, mps, or cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use Automatic Mixed Precision")
    parser.add_argument("--no_memory_efficient_attention", action="store_true",
                        help="Disable memory-efficient attention")
    
    args = parser.parse_args()
    
    train_reconstructor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        device=args.device,
        num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        use_amp=args.use_amp,
        memory_efficient_attention=not args.no_memory_efficient_attention
    ) 