import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
from stable_baselines3 import PPO
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

from rl.env import TokenCompressionEnv


def calculate_perplexity(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    device: str
) -> float:
    """
    Calculate perplexity of a sequence.
    
    Args:
        model: Language model
        input_ids: Token IDs
        device: Device to run on
        
    Returns:
        Perplexity value
    """
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    return torch.exp(loss).item()


def evaluate_downstream_tasks(
    original_text: str,
    reconstructed_text: str
) -> Dict[str, float]:
    """
    Evaluate downstream task performance.
    
    Args:
        original_text: Original text
        reconstructed_text: Reconstructed text
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # BLEU score
    try:
        original_tokens = original_text.split()
        reconstructed_tokens = reconstructed_text.split()
        
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [original_tokens],
            reconstructed_tokens,
            smoothing_function=smoothing
        )
        metrics["bleu_score"] = bleu_score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        metrics["bleu_score"] = 0.0
    
    # Sentiment Analysis
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
        original_sentiment = sentiment_analyzer(original_text)[0]
        reconstructed_sentiment = sentiment_analyzer(reconstructed_text)[0]
        
        metrics["sentiment_accuracy"] = int(
            original_sentiment["label"] == reconstructed_sentiment["label"]
        )
        metrics["sentiment_score_diff"] = abs(
            original_sentiment["score"] - reconstructed_sentiment["score"]
        )
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        metrics["sentiment_accuracy"] = 0.0
        metrics["sentiment_score_diff"] = 1.0
        
    return metrics


def evaluate_model(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_sequences: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Evaluate the trained model on a held-out test set.
    
    Args:
        model_path: Path to trained model
        data_path: Path to evaluation data
        output_dir: Directory to save results
        num_sequences: Number of sequences to evaluate
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Load tokenizer and reconstructor
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    reconstructor = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    # Create environment
    env = TokenCompressionEnv(
        tokenizer=tokenizer,
        reconstructor=reconstructor,
        data_path=data_path,
        max_seq_length=1024,
        context_window=32,
        device=device
    )
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Load evaluation data
    with open(data_path, "r") as f:
        eval_data = json.load(f)
    
    # Initialize metrics
    metrics = {
        "compression_ratios": [],
        "reconstruction_losses": [],
        "perplexities": [],
        "perplexity_ratios": [],  # perplexity(reconstructed) / perplexity(original)
        "rewards": [],
        "bleu_scores": [],
        "sentiment_accuracies": [],
        "sentiment_score_diffs": []
    }
    
    # Visualization data
    mask_patterns = []
    original_texts = []
    reconstructed_texts = []
    
    print("Evaluating model...")
    for i, sequence_data in enumerate(tqdm(eval_data[:num_sequences])):
        sequence = sequence_data["tokens"]
        
        # Set environment state
        env.current_sequence = sequence
        
        # Get model's action
        obs = env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        _, reward, _, _, info = env.step(action)
        
        # Record metrics
        metrics["compression_ratios"].append(info["compression_ratio"])
        metrics["rewards"].append(reward)
        
        # Get masked sequence
        masked_sequence = [t for i, t in enumerate(sequence) if action[i] > 0.5]
        
        # Calculate reconstruction loss
        with torch.no_grad():
            masked_tensor = torch.tensor(masked_sequence).unsqueeze(0).to(device)
            outputs = reconstructor(masked_tensor, labels=masked_tensor)
            loss = outputs.loss.item()
        metrics["reconstruction_losses"].append(loss)
        
        # Calculate perplexity
        original_tensor = torch.tensor(sequence).unsqueeze(0)
        masked_tensor = torch.tensor(masked_sequence).unsqueeze(0)
        
        original_perplexity = calculate_perplexity(reconstructor, original_tensor, device)
        masked_perplexity = calculate_perplexity(reconstructor, masked_tensor, device)
        
        metrics["perplexities"].append(masked_perplexity)
        metrics["perplexity_ratios"].append(masked_perplexity / original_perplexity)
        
        # Decode texts for downstream tasks
        original_text = tokenizer.decode(sequence, skip_special_tokens=True)
        reconstructed_text = tokenizer.decode(masked_sequence, skip_special_tokens=True)
        
        # Evaluate downstream tasks
        downstream_metrics = evaluate_downstream_tasks(original_text, reconstructed_text)
        
        metrics["bleu_scores"].append(downstream_metrics["bleu_score"])
        metrics["sentiment_accuracies"].append(downstream_metrics["sentiment_accuracy"])
        metrics["sentiment_score_diffs"].append(downstream_metrics["sentiment_score_diff"])
        
        # Save visualization data
        mask_patterns.append(action)
        original_texts.append(original_text)
        reconstructed_texts.append(reconstructed_text)
        
        # Save examples
        if i < 10:  # Save first 10 examples
            example = {
                "original_text": original_text,
                "reconstructed_text": reconstructed_text,
                "compression_ratio": info["compression_ratio"],
                "perplexity_original": original_perplexity,
                "perplexity_masked": masked_perplexity,
                "bleu_score": downstream_metrics["bleu_score"],
                "sentiment_accuracy": downstream_metrics["sentiment_accuracy"]
            }
            
            example_path = os.path.join(output_dir, f"example_{i+1}.json")
            with open(example_path, "w") as f:
                json.dump(example, f, indent=2)
    
    # Calculate average metrics
    results = {
        "avg_compression_ratio": np.mean(metrics["compression_ratios"]),
        "avg_reconstruction_loss": np.mean(metrics["reconstruction_losses"]),
        "avg_perplexity": np.mean(metrics["perplexities"]),
        "avg_perplexity_ratio": np.mean(metrics["perplexity_ratios"]),
        "avg_reward": np.mean(metrics["rewards"]),
        "avg_bleu_score": np.mean(metrics["bleu_scores"]),
        "avg_sentiment_accuracy": np.mean(metrics["sentiment_accuracies"]),
        "avg_sentiment_score_diff": np.mean(metrics["sentiment_score_diffs"]),
        "std_compression_ratio": np.std(metrics["compression_ratios"]),
        "std_reconstruction_loss": np.std(metrics["reconstruction_losses"]),
        "std_perplexity": np.std(metrics["perplexities"]),
        "std_reward": np.std(metrics["rewards"]),
        "std_bleu_score": np.std(metrics["bleu_scores"])
    }
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save visualization data
    viz_data = {
        "mask_patterns": [m.tolist() for m in mask_patterns],
        "original_texts": original_texts,
        "reconstructed_texts": reconstructed_texts
    }
    viz_path = os.path.join(output_dir, "visualization_data.json")
    with open(viz_path, "w") as f:
        json.dump(viz_data, f, indent=2)
    
    # Create compression vs. quality scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics["compression_ratios"], metrics["bleu_scores"], alpha=0.7)
    plt.xlabel("Compression Ratio")
    plt.ylabel("BLEU Score")
    plt.title("Compression vs. Quality Trade-off")
    plt.grid(True, alpha=0.3)
    
    # Calculate and plot trend line
    z = np.polyfit(metrics["compression_ratios"], metrics["bleu_scores"], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(metrics["compression_ratios"]), max(metrics["compression_ratios"]), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Calculate correlation
    corr, _ = pearsonr(metrics["compression_ratios"], metrics["bleu_scores"])
    plt.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=plt.gca().transAxes)
    
    plt.savefig(os.path.join(output_dir, "compression_vs_quality.png"))
    plt.close()
    
    print("Evaluation results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--num_sequences", type=int, default=100,
                       help="Number of sequences to evaluate")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ) 