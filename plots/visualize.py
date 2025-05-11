import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from scipy.stats import pearsonr
from sklearn.manifold import TSNE


def load_tensorboard_data(log_dir: str) -> pd.DataFrame:
    """
    Load TensorBoard logs into a pandas DataFrame.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        
    Returns:
        DataFrame with training data
    """
    # Find event file
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
        
    # Load data
    event_file = os.path.join(log_dir, event_files[0])
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0}  # Load all scalar events
    )
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()["scalars"]
    
    # Create DataFrame
    data = {}
    steps = []
    
    for tag in tags:
        events = ea.Scalars(tag)
        if not steps:
            steps = [e.step for e in events]
        data[tag] = [e.value for e in events]
    
    df = pd.DataFrame({"step": steps, **data})
    return df


def plot_training_curves(
    log_dir: str,
    output_dir: str
) -> None:
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load TensorBoard data
        df = load_tensorboard_data(log_dir)
        
        # Set style
        sns.set(style="whitegrid")
        
        # Plot training metrics
        metrics = {
            "rollout/ep_rew_mean": "Reward",
            "rollout/ep_len_mean": "Episode Length",
            "train/loss": "Loss",
            "train/entropy_loss": "Entropy Loss",
            "train/policy_gradient_loss": "Policy Gradient Loss",
            "train/value_loss": "Value Loss",
            "train/approx_kl": "Approx KL Divergence",
            "train/clip_fraction": "Clip Fraction",
            "train/explained_variance": "Explained Variance"
        }
        
        # Create a 3x3 grid for all metrics
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for i, (tag, title) in enumerate(metrics.items()):
            if tag in df.columns:
                ax = axes[i]
                sns.lineplot(x="step", y=tag, data=df, ax=ax)
                ax.set_title(title)
                ax.set_xlabel("Steps")
                ax.set_ylabel(title)
                
                # Add moving average
                if len(df) > 10:
                    window_size = min(10, len(df) // 5)
                    df[f"{tag}_ma"] = df[tag].rolling(window=window_size).mean()
                    sns.lineplot(
                        x="step", y=f"{tag}_ma", data=df, 
                        ax=ax, color="red", alpha=0.7, label="Moving Avg"
                    )
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"))
        plt.close()
        
        # Plot reward vs. entropy
        if "rollout/ep_rew_mean" in df.columns and "train/entropy_loss" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(
                df["train/entropy_loss"], 
                df["rollout/ep_rew_mean"],
                alpha=0.7
            )
            plt.xlabel("Entropy Loss")
            plt.ylabel("Mean Reward")
            plt.title("Reward vs. Entropy Trade-off")
            
            # Add trend line
            try:
                z = np.polyfit(df["train/entropy_loss"], df["rollout/ep_rew_mean"], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(
                    min(df["train/entropy_loss"]), 
                    max(df["train/entropy_loss"]), 
                    100
                )
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
                
                # Add correlation
                corr, _ = pearsonr(df["train/entropy_loss"], df["rollout/ep_rew_mean"])
                plt.text(
                    0.05, 0.95, 
                    f"Correlation: {corr:.3f}", 
                    transform=plt.gca().transAxes
                )
            except Exception as e:
                print(f"Could not calculate trend line: {e}")
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "reward_vs_entropy.png"))
            plt.close()
    except Exception as e:
        print(f"Error loading TensorBoard data: {e}")
        print("Falling back to metrics.json if available")
        
        # Fallback to JSON file if available
        metrics_path = os.path.join(log_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                
            # Plot basic metrics
            plt.figure(figsize=(12, 8))
            
            # Plot reward
            plt.subplot(2, 2, 1)
            plt.plot(metrics["timesteps"], metrics["rewards"])
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title("Training Reward")
            
            # Plot compression ratio
            plt.subplot(2, 2, 2)
            plt.plot(metrics["timesteps"], metrics["compression_ratios"])
            plt.xlabel("Timesteps")
            plt.ylabel("Compression Ratio")
            plt.title("Compression Ratio")
            
            # Plot reconstruction loss
            plt.subplot(2, 2, 3)
            plt.plot(metrics["timesteps"], metrics["reconstruction_losses"])
            plt.xlabel("Timesteps")
            plt.ylabel("Reconstruction Loss")
            plt.title("Reconstruction Loss")
            
            # Plot entropy
            plt.subplot(2, 2, 4)
            plt.plot(metrics["timesteps"], metrics["entropy"])
            plt.xlabel("Timesteps")
            plt.ylabel("Entropy")
            plt.title("Policy Entropy")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_curves_basic.png"))
            plt.close()
        else:
            print(f"No metrics data found in {log_dir}")


def plot_evaluation_results(
    results_path: str,
    output_dir: str
) -> None:
    """
    Plot evaluation results.
    
    Args:
        results_path: Path to evaluation results JSON
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot metrics
    metrics = [
        "compression_ratio", 
        "reconstruction_loss", 
        "perplexity",
        "perplexity_ratio",
        "reward", 
        "bleu_score",
        "sentiment_accuracy"
    ]
    labels = [m.replace("_", " ").title() for m in metrics]
    
    values = [results.get(f"avg_{m}", 0) for m in metrics]
    stds = [results.get(f"std_{m}", 0) for m in metrics]
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    colors = sns.color_palette("viridis", len(metrics))
    bars = plt.bar(x, values, yerr=stds, capsize=5, color=colors, alpha=0.8)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Evaluation Metrics")
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.1,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
    plt.close()
    
    # Create scatter plot of compression vs. quality
    try:
        viz_data_path = os.path.join(os.path.dirname(results_path), "visualization_data.json")
        if os.path.exists(viz_data_path):
            with open(viz_data_path, "r") as f:
                viz_data = json.load(f)
                
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot
            if "compression_ratios" in viz_data and "bleu_scores" in viz_data:
                plt.scatter(
                    viz_data["compression_ratios"],
                    viz_data["bleu_scores"],
                    alpha=0.7
                )
                plt.xlabel("Compression Ratio")
                plt.ylabel("BLEU Score")
                plt.title("Compression vs. Quality Trade-off")
                
                # Add trend line
                try:
                    z = np.polyfit(
                        viz_data["compression_ratios"],
                        viz_data["bleu_scores"],
                        1
                    )
                    p = np.poly1d(z)
                    x_trend = np.linspace(
                        min(viz_data["compression_ratios"]),
                        max(viz_data["compression_ratios"]),
                        100
                    )
                    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
                    
                    # Add correlation
                    corr, _ = pearsonr(
                        viz_data["compression_ratios"],
                        viz_data["bleu_scores"]
                    )
                    plt.text(
                        0.05, 0.95,
                        f"Correlation: {corr:.3f}",
                        transform=plt.gca().transAxes
                    )
                except Exception as e:
                    print(f"Could not calculate trend line: {e}")
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "compression_quality_scatter.png"))
                plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")


def plot_token_masking_patterns(
    viz_data_path: str,
    output_dir: str,
    num_examples: int = 5
) -> None:
    """
    Plot heatmaps of token masking patterns.
    
    Args:
        viz_data_path: Path to visualization data JSON
        output_dir: Directory to save plots
        num_examples: Number of examples to plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load visualization data
    with open(viz_data_path, "r") as f:
        viz_data = json.load(f)
        
    mask_patterns = viz_data.get("mask_patterns", [])
    original_texts = viz_data.get("original_texts", [])
    reconstructed_texts = viz_data.get("reconstructed_texts", [])
    
    if not mask_patterns:
        print("No mask patterns found in visualization data")
        return
        
    # Plot individual patterns
    for i in range(min(num_examples, len(mask_patterns))):
        plt.figure(figsize=(15, 3))
        
        # Create heatmap
        pattern = np.array(mask_patterns[i])
        heatmap = np.zeros((1, len(pattern)))
        heatmap[0] = pattern
        
        plt.imshow(heatmap, aspect="auto", cmap="RdYlGn_r")
        plt.colorbar(label="Keep Probability")
        plt.xlabel("Token Position")
        plt.title(f"Token Masking Pattern - Example {i+1}")
        
        # Add text summary
        if i < len(original_texts) and i < len(reconstructed_texts):
            plt.figtext(
                0.5, 0.01,
                f"Compression: {1 - pattern.mean():.2f}",
                ha="center",
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"masking_pattern_{i+1}.png"))
        plt.close()
    
    # Create aggregate heatmap of all patterns
    if len(mask_patterns) > 1:
        # Find shortest pattern length
        min_length = min(len(p) for p in mask_patterns)
        
        # Truncate all patterns to minimum length
        truncated_patterns = [p[:min_length] for p in mask_patterns]
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        heatmap_data = np.array(truncated_patterns)
        
        sns.heatmap(
            heatmap_data,
            cmap="RdYlGn_r",
            cbar_kws={"label": "Keep Probability"}
        )
        plt.xlabel("Token Position")
        plt.ylabel("Sequence Index")
        plt.title(f"Aggregate Token Masking Patterns (n={len(mask_patterns)})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aggregate_masking_patterns.png"))
        plt.close()
        
        # Calculate position-wise average masking probability
        avg_pattern = np.mean(heatmap_data, axis=0)
        
        plt.figure(figsize=(15, 5))
        plt.plot(avg_pattern, color="blue", alpha=0.7)
        plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.3)
        plt.fill_between(
            range(len(avg_pattern)),
            avg_pattern,
            0.5,
            where=(avg_pattern >= 0.5),
            color="green",
            alpha=0.3,
            interpolate=True
        )
        plt.fill_between(
            range(len(avg_pattern)),
            avg_pattern,
            0.5,
            where=(avg_pattern <= 0.5),
            color="red",
            alpha=0.3,
            interpolate=True
        )
        plt.xlabel("Token Position")
        plt.ylabel("Average Keep Probability")
        plt.title("Position-wise Average Masking Decision")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "average_masking_pattern.png"))
        plt.close()


def plot_token_embeddings(
    tokens_data_path: str,
    output_dir: str
) -> None:
    """
    Plot t-SNE visualization of token embeddings.
    
    Args:
        tokens_data_path: Path to token embeddings data
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load token embeddings
        with open(tokens_data_path, "r") as f:
            data = json.load(f)
            
        embeddings = np.array(data.get("embeddings", []))
        labels = np.array(data.get("keep_decisions", []))
        
        if len(embeddings) == 0:
            print("No embeddings found in data")
            return
            
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        
        # Plot kept tokens (1) and masked tokens (0)
        kept = labels > 0.5
        masked = ~kept
        
        plt.scatter(
            embeddings_2d[kept, 0],
            embeddings_2d[kept, 1],
            alpha=0.7,
            color="green",
            label="Kept Tokens"
        )
        plt.scatter(
            embeddings_2d[masked, 0],
            embeddings_2d[masked, 1],
            alpha=0.7,
            color="red",
            label="Masked Tokens"
        )
        
        plt.legend()
        plt.title("t-SNE Visualization of Token Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "token_embeddings_tsne.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating token embeddings plot: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True,
                       help="Directory containing training logs")
    parser.add_argument("--results_path", type=str, required=True,
                       help="Path to evaluation results")
    parser.add_argument("--viz_data_path", type=str, required=False,
                       help="Path to visualization data (if not in same dir as results)")
    parser.add_argument("--tokens_data_path", type=str, required=False,
                       help="Path to token embeddings data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Plot training curves
    plot_training_curves(args.log_dir, args.output_dir)
    
    # Plot evaluation results
    plot_evaluation_results(args.results_path, args.output_dir)
    
    # Plot token masking patterns
    viz_data_path = args.viz_data_path or os.path.join(
        os.path.dirname(args.results_path),
        "visualization_data.json"
    )
    if os.path.exists(viz_data_path):
        plot_token_masking_patterns(viz_data_path, args.output_dir)
    
    # Plot token embeddings if data is available
    if args.tokens_data_path and os.path.exists(args.tokens_data_path):
        plot_token_embeddings(args.tokens_data_path, args.output_dir) 