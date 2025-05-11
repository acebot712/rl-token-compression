import os
import json
import torch
import random
from transformers import GPT2Tokenizer
from train import train_reconstructor

def create_test_data(output_path, num_samples=10, max_length=100):
    """Create a small test dataset for training."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Generate random token sequences
    data = []
    for i in range(num_samples):
        # Random length between 20 and max_length
        length = random.randint(20, max_length)
        # Random token IDs (valid for GPT-2 vocabulary)
        tokens = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(length)]
        data.append({"tokens": tokens})
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    print(f"Created test data with {len(data)} samples at {output_path}")
    return output_path

if __name__ == "__main__":
    # Create test data
    test_data_path = "data/test/test_data.json"
    create_test_data(test_data_path)
    
    # Run training with small settings
    train_reconstructor(
        data_path=test_data_path,
        output_dir="models/reconstructor/test_output",
        batch_size=2,
        epochs=1,
        model_name="gpt2",  # Use smallest GPT-2 model
        mask_ratio=0.3,
        warmup_steps=10
    )
    
    print("Test complete!") 