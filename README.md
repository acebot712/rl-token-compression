# Learning to Forget — RL-Based Token Compression for Efficient LLM Inference

This repository contains the implementation for our research paper on RL-based token compression for efficient LLM inference. The system learns to selectively mask predictable tokens in long-context sequences while preserving semantic meaning, enabling faster and more efficient inference with large language models.

## Abstract

Large Language Models (LLMs) excel at processing lengthy input contexts, but this comes at significant computational cost due to the quadratic scaling of self-attention with sequence length. We propose a novel approach that uses reinforcement learning to train an agent to selectively compress input token sequences by identifying and masking predictable tokens. Our system maintains downstream task performance while achieving substantial compression ratios, effectively reducing inference time and computational requirements.

## Architecture

Our system consists of three main components:

1. **Agent**: A sophisticated policy network that decides which tokens to keep or mask, using specialized attention mechanisms to assess token importance.

2. **Reconstructor**: A fine-tuned GPT-2 model that learns to reconstruct masked sequences, trained specifically on the task of filling in masked tokens.

3. **Environment**: An RL environment where the agent interacts with sequences, receiving rewards based on compression ratio and reconstruction loss.

## Project Structure

```
.
├── data/                    # Data preparation and tokenization
│   ├── prepare.py          # Main data preparation script
│   ├── test_prepare.py     # Test script for data preparation
│   ├── test/              # Test data directory
│   └── processed/         # Processed data directory
├── models/
│   ├── agent/             # RL policy network with attention mechanisms
│   │   ├── policy.py      # Main policy implementation
│   │   ├── test_policy.py # Test script for policy
│   │   └── __init__.py
│   └── reconstructor/     # Fine-tuned GPT-2 for token reconstruction
│       ├── train_gpu.py   # GPU training script
│       ├── train_cpu.py   # CPU training script
│       ├── test_train.py  # Test script
│       ├── output/        # Training outputs
│       └── fine-tuned/    # Fine-tuned model checkpoints
├── rl/                    # Reinforcement learning components
│   ├── env.py            # RL environment implementation
│   ├── train.py          # PPO training script
│   ├── test_train.py     # Test script
│   └── test_output/      # Test outputs
├── eval/                 # Evaluation scripts
│   ├── evaluate.py       # Main evaluation script
│   ├── test_evaluate.py  # Test script
│   └── test/            # Test evaluation data
├── plots/               # Visualization scripts
│   └── visualize.py     # Main visualization script
├── requirements.txt     # Project dependencies
└── setup.py            # Package setup file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The project requires the following main dependencies:
- PyTorch (>=1.10.0)
- Transformers (>=4.12.0)
- Gymnasium (>=0.27.0)
- Stable-Baselines3 (>=1.6.0)
- TensorBoard (>=2.5.0)
- NLTK (>=3.6.0)
- Datasets (>=1.11.0)
- SentencePiece (>=0.1.96)
- And other scientific computing libraries (numpy, pandas, scikit-learn, etc.)

## Usage

### 1. Data Preparation

Tokenize and prepare the dataset:
```bash
python data/prepare.py \
    --output_dir data/processed \
    --max_sequences 50000 \
    --max_length 1024
```

### 2. Reconstructor Model Training

Train the GPT-2 reconstructor on masked sequences:

CPU version:
```bash
python models/reconstructor/train_cpu.py \
    --data_path data/processed/processed_data.json \
    --output_dir models/reconstructor/fine-tuned \
    --epochs 3 \
    --batch_size 8 \
    --mask_ratio 0.3
```

GPU version (CUDA):
```bash
python models/reconstructor/train_gpu.py \
    --data_path data/processed/processed_data.json \
    --output_dir models/reconstructor/fine-tuned \
    --device cuda \
    --max_length 512 \
    --use_amp \
    --gradient_accumulation_steps 4 \
    --batch_size 4
```

MPS (Apple Silicon) version:
```bash
python models/reconstructor/train_gpu.py \
    --data_path data/processed/processed_data.json \
    --output_dir models/reconstructor/fine-tuned \
    --device mps \
    --max_length 512 \
    --gradient_accumulation_steps 4 \
    --batch_size 4
```

**Note:** AMP (Automatic Mixed Precision) is automatically disabled for MPS devices to ensure training stability.

### 3. RL Training

Train the RL agent with PPO:
```bash
python rl/train.py \
    --data_path data/processed/processed_data.json \
    --output_dir models/agent/output \
    --reconstructor_path models/reconstructor/fine-tuned \
    --num_timesteps 1000000 \
    --n_steps 2048 \
    --batch_size 64 \
    --gamma 0.99
```

### 4. Evaluation

Evaluate the trained model with comprehensive metrics:
```bash
python eval/evaluate.py \
    --model_path models/agent/output/best_model.zip \
    --data_path data/processed/test_data.json \
    --output_dir eval/test_output \
    --num_sequences 100
```

### 5. Visualization

Generate visualization plots:
```bash
python plots/visualize.py \
    --log_dir models/agent/output/tb_logs \
    --results_path eval/test_output/evaluation_results.json \
    --output_dir plots/output
```

## Testing

The codebase includes test scripts to verify each component with small-scale examples. These tests correspond to the main usage steps and help ensure each component works correctly.

### Component Tests

Each main component has its corresponding test script:

1. **Data Preparation Test**
```bash
python data/test_prepare.py
```
Tests the data preparation pipeline with a small dataset.

2. **Reconstructor Model Test**
```bash
python models/reconstructor/test_train.py
```

Tests the reconstructor model training with a small number of epochs.

3. **Agent Policy Test**
```bash
python models/agent/test_policy.py
```
Tests the policy network's decision-making capabilities.

4. **RL Training Test**
```bash
python rl/test_train.py
```
Tests the RL training loop with a small number of timesteps.

5. **Evaluation Test**
```bash
python eval/test_evaluate.py
```
Tests the evaluation pipeline with a small number of sequences.

Note: The visualization component (`plots/visualize.py`) doesn't have a dedicated test script as it's primarily used for generating plots from existing results.

## Technical Details

### Policy Network Architecture

Our policy network leverages:
- Positional encoding for token position awareness
- Custom token importance attention mechanism
- Local context aggregation via convolutional layers
- Context gates to focus on relevant tokens
- Hierarchical feature extraction for token importance assessment

### Training Process

The training process uses PPO (Proximal Policy Optimization) with:
- Adaptive learning rate scheduling
- Entropy bonus for exploration
- Multiple parallel environments for diverse experience
- Importance weighted updates for stable learning

### Evaluation Metrics

We evaluate our approach using:
- Compression ratio (percentage of tokens removed)
- Reconstruction loss and perplexity
- BLEU score between original and reconstructed text
- Downstream task performance (e.g., sentiment analysis)
- Fine-grained token importance analysis

## Results

Our approach achieves an average compression ratio of 30-40% while maintaining over 90% semantic preservation on downstream tasks. The system demonstrates a clear trade-off between compression ratio and reconstruction quality that can be controlled through hyperparameter tuning.

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{rl_token_compression,
  title={Learning to Forget: RL-Based Token Compression for Efficient LLM Inference},
  author={[Authors]},
  journal={arXiv preprint},
  year={2023}
}
```

## License

MIT License
