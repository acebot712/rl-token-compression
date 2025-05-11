import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from rl.env import TokenCompressionEnv
from models.agent.policy import TokenCompressionPolicy


def train(
    data_path: str,
    output_dir: str,
    reconstructor_path: str,
    num_timesteps: int = 1_000_000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    gamma: float = 0.99,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the token compression agent using PPO.
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save checkpoints and logs
        reconstructor_path: Path to fine-tuned reconstructor model
        num_timesteps: Number of training timesteps
        batch_size: Training batch size
        learning_rate: Learning rate for PPO
        n_steps: Number of steps to run for each environment per update
        gamma: Discount factor
        device: Device to run training on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer and reconstructor
    print("Loading tokenizer and reconstructor...")
    tokenizer = GPT2Tokenizer.from_pretrained(
        reconstructor_path if os.path.exists(reconstructor_path) else "gpt2"
    )
    reconstructor = GPT2LMHeadModel.from_pretrained(
        reconstructor_path if os.path.exists(reconstructor_path) else "gpt2"
    ).to(device)
    
    print(f"Using reconstructor from: {reconstructor_path}")
    print("Reconstructor model loaded successfully")
    
    # Create environment
    print("Creating environment...")
    env = TokenCompressionEnv(
        tokenizer=tokenizer,
        reconstructor=reconstructor,
        data_path=data_path,
        max_seq_length=1024,
        context_window=32,
        device=device
    )
    env = DummyVecEnv([lambda: env])
    
    # Initialize policy
    policy_kwargs = dict(
        features_extractor_class=TokenCompressionPolicy,
        features_extractor_kwargs=dict(
            input_dim=tokenizer.vocab_size + 32,  # vocab_size + context_window
            hidden_dim=256,
            num_layers=3,
            dropout=0.1
        )
    )
    
    # Initialize PPO agent
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs")
    )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="rl_model"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval_logs"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    print(f"Starting training for {num_timesteps} timesteps...")
    model.learn(
        total_timesteps=num_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--reconstructor_path",
        type=str,
        required=True,
        help="Path to fine-tuned reconstructor model"
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1_000_000,
        help="Number of training timesteps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps per update"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on"
    )
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        reconstructor_path=args.reconstructor_path,
        num_timesteps=args.num_timesteps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ) 