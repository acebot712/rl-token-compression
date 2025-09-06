# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing RL-based token compression for efficient LLM inference. The system uses reinforcement learning to train an agent that selectively masks predictable tokens in sequences while preserving semantic meaning.

## Architecture

The system consists of three main components:

1. **Agent** (`models/agent/`): A sophisticated policy network with attention mechanisms that decides which tokens to keep or mask
2. **Reconstructor** (`models/reconstructor/`): A fine-tuned GPT-2 model that reconstructs masked sequences
3. **Environment** (`rl/`): RL environment where the agent interacts with token sequences

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```bash
python data/prepare.py --config configs/data/sample.json
```

### Training Commands

**Reconstructor Training:**
```bash
# Debug training (quick test)
python training/train.py --config configs/training/debug.json

# CUDA training (high-performance)
python training/train.py --config configs/training/cuda.json

# MPS training (Apple Silicon optimized)
python training/train.py --config configs/training/mps.json

# Default training (conservative settings)
python training/train.py --config configs/training/default.json
```

**RL Agent Training:**
```bash
python training/train.py --config configs/training/default.json
```

### Evaluation and Testing
```bash
# Main evaluation
python evaluation/evaluate.py --config configs/evaluation/default.json

# Generate plots
python plots/visualize.py --results_path outputs/evaluation/evaluation_results.json --output_dir plots/output
```

### Component Testing
Each component has a dedicated test script for verification:
```bash
python tests/test_data_prepare.py     # Test data preparation
python tests/test_models.py          # Test model components
python tests/test_trainer.py         # Test training pipeline
python tests/test_evaluation.py      # Test evaluation pipeline
```

## Key Implementation Details

### Policy Network Architecture
- Uses positional encoding for token position awareness (`models/agent/policy.py:8-49`)
- Custom TokenImportanceAttention mechanism for assessing token importance (`models/agent/policy.py:51-151`)
- Enhanced policy with transformer layers, context gates, and local context aggregation (`models/agent/policy.py:210-376`)

### RL Environment
- Gym-based environment for token compression decisions (`rl/env.py:10-181`)
- Binary action space (keep/mask) with continuous action values converted using 0.5 threshold
- Reward function balances compression ratio and reconstruction loss (`rl/env.py:86-107`)

### Training Process
- Uses PPO (Proximal Policy Optimization) with custom feature extractor
- Supports CUDA, MPS, and CPU training with device-specific optimizations
- Includes checkpointing, evaluation callbacks, and TensorBoard logging

### Device Support
The codebase supports three compute backends:
- **CUDA**: Full GPU acceleration with optional AMP
- **MPS** (Apple Silicon): GPU acceleration, AMP automatically disabled for stability
- **CPU**: Fallback for systems without GPU support

### Data Processing
- Processes Reddit dataset or custom text data
- Tokenizes using GPT-2 tokenizer with configurable sequence lengths
- Creates masked training data for reconstructor fine-tuning

## File Structure Notes

- Fine-tuned models are saved in `models/reconstructor/fine-tuned/` with checkpoint directories
- RL training outputs go to `models/agent/output/` including TensorBoard logs
- Test outputs are saved in respective `test_output/` directories within each component
- The `venv/` directory contains the Python virtual environment (ignored in version control)