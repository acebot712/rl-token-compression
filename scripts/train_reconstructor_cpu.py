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
            
            # CRITICAL FIX: Ensure at least one token is always unmasked
            # If all tokens are masked, randomly unmask one
            if mask.all() and len(tokens) > 0:
                unmask_idx = random.randint(0, len(tokens) - 1)
                mask[unmask_idx] = False
            
            # Also ensure we don't mask too few tokens (for learning)
            # If no tokens are masked, randomly mask at least one
            if not mask.any() and len(tokens) > 1:
                mask_idx = random.randint(0, len(tokens) - 1)
                mask[mask_idx] = True
            
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
            
            # Validation: Ensure we have valid data
            assert not torch.isnan(original_tokens).any(), "NaN found in original tokens"
            assert not torch.isnan(masked_tokens).any(), "NaN found in masked tokens"
            assert not torch.isnan(attention_mask.float()).any(), "NaN found in attention mask"
            assert (original_tokens >= 0).all(), "Negative token IDs found"
            assert (masked_tokens >= 0).all(), "Negative masked token IDs found"
            
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        device: Device to run training on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Create dataset and dataloader
    dataset = MaskedSequenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        mask_ratio=mask_ratio
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Changed from 4 to 0 for debugging
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
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
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # CRITICAL FIX: Validate loss to catch NaN early
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Invalid loss detected at step {global_step}")
                print(f"Loss value: {loss.item()}")
                print(f"Input IDs shape: {input_ids.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Input IDs range: {input_ids.min().item()} to {input_ids.max().item()}")
                print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
                print(f"Number of non-padding tokens: {attention_mask.sum().item()}")
                
                # Skip this batch instead of propagating NaN
                print("Skipping batch due to invalid loss")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update progress
            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({"loss": current_loss})
            global_step += 1
            
            # Save checkpoint
            if global_step % 1000 == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        
        # CRITICAL FIX: Handle case where loss might be NaN due to all batches being skipped
        if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
            print(f"WARNING: Invalid average loss for epoch {epoch+1}: {avg_loss}")
            avg_loss = float('inf')  # Set to infinity to indicate failure
        
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
                        help="Device to run training on")
    
    args = parser.parse_args()
    
    train_reconstructor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ) 