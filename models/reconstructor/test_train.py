import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
from transformers import GPT2Tokenizer
from models.reconstructor.train_gpu import train_reconstructor


def create_test_data(output_path, num_samples=10, max_length=100):
    """Create a small test dataset for training."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Generate more realistic token sequences using actual text
    # This creates more coherent sequences that are easier to reconstruct
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a powerful tool for data analysis.",
        "Natural language processing helps computers understand text.",
        "Artificial intelligence will transform many industries.",
        "Deep learning models require large amounts of data.",
        "Python is a popular programming language for AI.",
        "Transformers have revolutionized natural language processing.",
        "Token compression can improve model efficiency.",
        "Reinforcement learning trains agents through rewards.",
        "Neural networks are inspired by biological brains."
    ]
    
    data = []
    for i in range(num_samples):
        # Use actual text instead of random tokens
        text = sample_texts[i % len(sample_texts)]
        tokens = tokenizer.encode(text)
        
        # Ensure minimum length by repeating if necessary
        while len(tokens) < 20:
            next_text = sample_texts[(i + 1) % len(sample_texts)]
            tokens = tokens + tokenizer.encode(" " + next_text)
        
        # Truncate to max_length if needed
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            
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

    # Determine the best training approach based on device
    device = None
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA with GPU training")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS with GPU training (AMP disabled)")
    else:
        device = "cpu"
        print("Using CPU training")

    # Run training with small settings and device-appropriate parameters
    try:
        train_reconstructor(
            data_path=test_data_path,
            output_dir="models/reconstructor/test_output",
            batch_size=2,
            epochs=1,
            model_name="gpt2",  # Use smallest GPT-2 model
            mask_ratio=0.2,  # Lower mask ratio to avoid all-masked sequences
            warmup_steps=10,
            use_amp=False,  # Disable AMP for compatibility
            num_workers=0,  # Reduce to 0 for debugging/MPS compatibility
            max_length=128,  # Smaller max length for test
            device=device
        )
        print("Test complete!")
    
    except Exception as e:
        print(f"Error during training: {e}")
        print("If you're on MPS/Mac and encountering issues, "
              "try using CPU training instead:")
        cpu_cmd = ("python models/reconstructor/train_cpu.py "
                   "--data_path data/test/test_data.json "
                   "--output_dir models/reconstructor/test_output "
                   "--batch_size 2 --epochs 1")
        print(cpu_cmd)
