import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Import local modules with proper path handling
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.agent.policy import TokenCompressionPolicy
from rl.env import TokenCompressionEnv


def test_policy_network():
    """Test the policy network with sample data."""
    print("Testing policy network...")
    
    # Set up test params
    input_dim = 768 + 32  # GPT2 embedding size + context window
    hidden_dim = 128
    num_layers = 2
    
    # Create policy network
    policy = TokenCompressionPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    # Generate random input
    batch_size = 2
    seq_length = 20
    sample_input = torch.rand(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = policy(sample_input)
    
    # Check output shape
    expected_output_shape = (batch_size, seq_length, policy.features_dim)
    actual_output_shape = output.shape
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {expected_output_shape}")
    
    assert actual_output_shape == expected_output_shape, \
        f"Output shape {actual_output_shape} does not match expected {expected_output_shape}"
    
    print("Policy network test passed!")


def test_policy_with_environment():
    """Test the policy in the context of the environment."""
    print("\nTesting policy with environment...")
    
    # Set up tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Create a test data file if it doesn't exist
    test_data_path = "data/test/test_data.json"
    
    if not os.path.exists(test_data_path):
        # Run the data preparation script
        print("Test data not found, running data preparation...")
        
        from data.test_prepare import prepare_test_data
        prepare_test_data("data/test")
    
    # Create environment
    env = TokenCompressionEnv(
        tokenizer=tokenizer,
        reconstructor=model,
        data_path=test_data_path,
        max_seq_length=50,
        context_window=8
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Check observation shape
    print(f"Observation shape: {obs.shape}")
    
    # Create policy network
    input_dim = obs.shape[1]  # Should match tokenizer.vocab_size + context_window
    policy = TokenCompressionPolicy(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2
    )
    
    # Generate action
    obs_tensor = torch.FloatTensor(obs)
    features = policy(obs_tensor.unsqueeze(0))
    
    # For simplicity, create a random action
    action = np.random.randint(0, 2, size=(env.max_seq_length,))
    
    # Step environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Action shape: {action.shape}")
    print(f"Reward: {reward}")
    print(f"Environment info: {info}")
    
    print("Policy and environment integration test passed!")


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("models/agent", exist_ok=True)
    
    # Run tests
    test_policy_network()
    test_policy_with_environment() 