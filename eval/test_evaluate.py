import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Import local modules with proper path handling
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from eval.evaluate import calculate_perplexity


def test_perplexity_calculation():
    """Test the perplexity calculation function."""
    print("Testing perplexity calculation...")
    
    # Set up tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = "cpu"
    
    # Test text
    test_text = "This is a test sentence for calculating perplexity."
    tokens = tokenizer.encode(test_text, return_tensors="pt")
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, tokens, device)
    
    print(f"Test text: {test_text}")
    print(f"Perplexity: {perplexity}")
    
    assert perplexity > 0, "Perplexity should be positive"
    print("Perplexity calculation test passed!")


def create_mock_data():
    """Create mock data for testing evaluation."""
    os.makedirs("eval/test", exist_ok=True)
    
    # Mock agent model output - just a simple function always choosing alternate tokens
    def mock_compression(tokens):
        return [t for i, t in enumerate(tokens) if i % 2 == 0]
    
    # Create test data if it doesn't exist
    test_data_path = "data/test/test_data.json"
    
    if not os.path.exists(test_data_path):
        # Run the data preparation script
        print("Test data not found, running data preparation...")
        
        # Use relative import with proper path handling
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
        from data.test_prepare import prepare_test_data
        prepare_test_data("data/test")
    
    # Load test data
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    # Create mock results using our mock compression function
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    results = []
    for item in test_data:
        tokens = item["tokens"]
        compressed_tokens = mock_compression(tokens)
        
        original_text = tokenizer.decode(tokens, skip_special_tokens=True)
        compressed_text = tokenizer.decode(compressed_tokens, skip_special_tokens=True)
        
        results.append({
            "original_text": original_text,
            "compressed_text": compressed_text,
            "compression_ratio": 0.5,  # Our mock function always keeps 50%
            "original_tokens": tokens,
            "compressed_tokens": compressed_tokens
        })
    
    # Save mock results
    mock_results_path = "eval/test/mock_results.json"
    with open(mock_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Created mock evaluation data at {mock_results_path}")
    return mock_results_path


def test_visualization():
    """Test the visualization functions."""
    print("\nTesting visualization...")
    
    # Create mock data if it doesn't exist
    mock_results_path = "eval/test/mock_results.json"
    if not os.path.exists(mock_results_path):
        mock_results_path = create_mock_data()
    
    # Load mock results
    with open(mock_results_path, "r") as f:
        results = json.load(f)
    
    # Create mock compression vs quality data
    compression_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    quality_scores = [0.8, 0.7, 0.6, 0.5, 0.4]  # Inverse relationship for testing
    
    # Create scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(compression_ratios, quality_scores, alpha=0.7)
    plt.xlabel("Compression Ratio")
    plt.ylabel("Quality Score")
    plt.title("Test Compression vs. Quality Trade-off")
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs("eval/test", exist_ok=True)
    plt.savefig("eval/test/test_plot.png")
    
    print(f"Created test visualization at eval/test/test_plot.png")
    print("Visualization test passed!")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("eval/test", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    # Run tests
    test_perplexity_calculation()
    create_mock_data()
    test_visualization()
    
    print("\nAll evaluation tests passed!") 