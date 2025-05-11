import os
import json
from typing import List, Dict, Any
from transformers import GPT2Tokenizer

def prepare_test_data(
    output_dir: str,
    num_samples: int = 10,
    max_length: int = 100
) -> None:
    """
    Prepare a small test dataset for testing purposes.
    
    Args:
        output_dir: Directory to save processed data
        num_samples: Number of test samples to generate
        max_length: Maximum sequence length
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Create sample texts
    sample_texts = [
        "This is a sample text for testing purposes.",
        "Token compression is an interesting research area.",
        "Deep reinforcement learning can be used for language tasks.",
        "The goal is to compress sequences while maintaining meaning.",
        "GPT models have shown impressive capabilities in NLP.",
        "Transformers are the foundation of modern language models.",
        "Attention mechanisms help models focus on relevant information.",
        "Fine-tuning allows adapting models to specific domains.",
        "Token prediction helps reconstruct masked sequences.",
        "Natural language processing continues to advance rapidly."
    ]
    
    # Process and tokenize sequences
    processed_sequences: List[Dict[str, Any]] = []
    
    print("Processing test sequences...")
    for i, text in enumerate(sample_texts[:num_samples]):
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
            "source": "test_sample"
        })
    
    # Save processed data
    output_path = os.path.join(output_dir, "test_data.json")
    with open(output_path, "w") as f:
        json.dump(processed_sequences, f)
    
    print(f"Processed {len(processed_sequences)} test sequences")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/test",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
        help="Number of test samples to generate"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=100,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    prepare_test_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_length=args.max_length
    ) 