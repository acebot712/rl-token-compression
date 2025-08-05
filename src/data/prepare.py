import os
import json
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm


def prepare_reddit_data(
    output_dir: str,
    max_sequences: int = 10000,
    max_length: int = 1024
) -> None:
    """
    Prepare Reddit data for training by downloading and tokenizing.
    
    Args:
        output_dir: Directory to save processed data
        max_sequences: Maximum number of sequences to process
        max_length: Maximum sequence length
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load Reddit dataset
    print("Loading Reddit dataset...")
    dataset = load_dataset("reddit", split="train", trust_remote_code=True)
    
    # Print dataset structure
    print("\nDataset structure:")
    print(dataset.features)
    
    # Get first example to inspect structure
    first_example = dataset[0]
    print("\nFirst example structure:")
    print(first_example)
    
    # Process and tokenize sequences
    processed_sequences: List[Dict[str, Any]] = []
    
    print("\nProcessing sequences...")
    for i, example in enumerate(tqdm(dataset)):
        if i >= max_sequences:
            break
            
        # Extract text based on available fields
        # The Reddit dataset has different fields
        text = (
            example.get('content', '') or 
            example.get('text', '') or 
            example.get('body', '')
        )
        if not text:
            continue
            
        # Tokenize
        tokens = tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )[0].tolist()
        
        processed_sequences.append({
            "tokens": tokens,
            "length": len(tokens),
            "source": "reddit"
        })
    
    # Save processed data
    output_path = os.path.join(output_dir, "processed_data.json")
    with open(output_path, "w") as f:
        json.dump(processed_sequences, f)
    
    print(f"\nProcessed {len(processed_sequences)} sequences")
    print(f"Saved to {output_path}")


