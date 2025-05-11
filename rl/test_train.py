import os
import torch
import numpy as np
from stable_baselines3 import PPO
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Import local modules with proper path handling
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from rl.env import TokenCompressionEnv


def test_environment():
    """Test the TokenCompressionEnv with a small sample dataset."""
    print("Testing RL environment...")
    
    # Create test data if it doesn't exist
    test_data_path = "data/test/test_data.json"
    
    if not os.path.exists(test_data_path):
        # Run the data preparation script
        print("Test data not found, running data preparation...")
        
        # Import test data preparation
        from data.test_prepare import prepare_test_data
        prepare_test_data("data/test")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Create environment with small dimensions for testing
    env = TokenCompressionEnv(
        tokenizer=tokenizer,
        reconstructor=model,
        data_path=test_data_path,
        max_seq_length=50,
        context_window=8,
        device="cpu"
    )
    
    # Test reset
    observation, info = env.reset()
    print(f"Reset observation shape: {observation.shape}")
    print(f"Reset info: {info}")
    
    # Test step with a simple action (mask every other token)
    action = np.zeros(env.max_seq_length)
    action[::2] = 1  # Keep every other token
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step reward: {reward}")
    print(f"Step info: {info}")
    print(f"Terminated: {terminated}")
    
    print("Environment test passed!")


def test_agent_training():
    """Test the PPO agent with a very small training run."""
    print("\nTesting RL agent training...")
    
    # Create test data if it doesn't exist
    test_data_path = "data/test/test_data.json"
    
    if not os.path.exists(test_data_path):
        from data.test_prepare import prepare_test_data
        prepare_test_data("data/test")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Create environment
    env = TokenCompressionEnv(
        tokenizer=tokenizer,
        reconstructor=model,
        data_path=test_data_path,
        max_seq_length=50,
        context_window=8,
        device="cpu"
    )
    
    # Create a very simple PPO agent for testing
    ppo_agent = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=4,  # Very small for testing
        batch_size=2,
        n_epochs=1
    )
    
    # Train for a minimal number of steps
    print("Training agent for 8 steps...")
    ppo_agent.learn(total_timesteps=8)
    
    # Save the model
    os.makedirs("rl/test_output", exist_ok=True)
    test_model_path = "rl/test_output/test_model.zip"
    ppo_agent.save(test_model_path)
    
    print(f"Agent trained and saved to {test_model_path}")
    print("Agent training test passed!")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("rl/test_output", exist_ok=True)
    
    # Run tests
    test_environment()
    test_agent_training()
    
    print("\nAll RL tests passed!") 