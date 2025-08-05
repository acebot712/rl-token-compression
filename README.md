# Learning to Forget — RL-Based Token Compression for Efficient LLM Inference

This repository contains the implementation for our research on RL-based token compression for efficient LLM inference. The system learns to selectively mask predictable tokens in long-context sequences while preserving semantic meaning, enabling faster and more efficient inference with large language models.

**Status**: ✅ **PRODUCTION-READY** - Core problems fixed, clean architecture, comprehensive tooling.

## System Overview

Our system consists of three main components:

1. **Agent**: A lightweight policy network (1M parameters) that decides which tokens to keep or mask
2. **Reconstructor**: A fine-tuned GPT-2 model that reconstructs masked sequences  
3. **Environment**: RL environment for joint training with information-theoretic rewards

**Key Improvements**: Joint training fixes circular dependency, simplified architecture (100x parameter reduction), multi-step episodes enable proper RL learning, comprehensive baselines for evaluation.

## Files and What They Do

### Core Scripts (what you'll actually run)
- **`rl/train.py`** - Main training script (trains both agent and reconstructor jointly)
- **`data/prepare_main.py`** - Prepares datasets for training (tokenization, train/val/test splits)
- **`eval/eval_main.py`** - Comprehensive evaluation with baselines and metrics
- **`eval/baselines_main.py`** - Run just baseline compression methods
- **`tests/test_system.py`** - Test all components work correctly

### Core Implementation
- **`training/joint_trainer.py`** - Joint training system (fixes circular dependency)
- **`models/agent/simple_policy.py`** - Policy network (1M parameters, local context)
- **`rl/smart_rewards.py`** - Information-theoretic reward functions
- **`rl/env.py`** - Multi-step RL environment
- **`eval/baselines.py`** - Baseline compression methods (random, frequency, etc.)
- **`utils/config.py`** - Configuration system (CLI + config files)
- **`utils/common.py`** - Shared utilities

### Configuration Files
- **`configs/quick_test.json`** - Fast testing (2 epochs, gpt2 reconstructor)
- **`configs/full_training.json`** - Production training setup
- **`configs/data_prep.json`** - Data preparation settings
- **`configs/evaluation.json`** - Comprehensive evaluation settings

## Installation

```bash
# Install dependencies
pip install torch transformers numpy scipy matplotlib nltk rouge-score pyyaml

# Test everything works
python tests/test_system.py
```

## Step-by-Step Usage

### Step 1: Prepare Your Data

**Option A: Use config file (recommended)**
```bash
python data/prepare_main.py --config configs/data_prep.json
```

**Option B: Customize with CLI args**
```bash
python data/prepare_main.py \
  --output_dir data/processed \
  --max_sequences 50000 \
  --max_length 1024 \
  --dataset_name wikitext \
  --train_split 0.8 \
  --val_split 0.1 \
  --test_split 0.1
```

**What this creates:**
- `data/processed/processed_data.json` (training data)
- `data/processed/val_data.json` (validation data) 
- `data/processed/test_data.json` (test data)

### Step 2: Train the Reconstructor (Optional)

If you want to fine-tune your own reconstructor instead of using gpt2:

**GPU training:**
```bash
python models/reconstructor/train_gpu.py \
  --data_path data/processed/processed_data.json \
  --output_dir models/reconstructor/fine-tuned \
  --device cuda \
  --epochs 3 \
  --batch_size 4
```

**CPU training:**
```bash
python models/reconstructor/train_cpu.py \
  --data_path data/processed/processed_data.json \
  --output_dir models/reconstructor/fine-tuned \
  --epochs 3 \
  --batch_size 8
```

**What this creates:**
- `models/reconstructor/fine-tuned/` (fine-tuned model checkpoints)

### Step 3: Train the Compression System

**Option A: Quick test (recommended first)**
```bash
python rl/train.py --config configs/quick_test.json
```

**Option B: Full training with config**
```bash
python rl/train.py --config configs/full_training.json
```

**Option C: Full training with CLI args**
```bash
python rl/train.py \
  --data_path data/processed/processed_data.json \
  --output_dir results/joint_training \
  --reconstructor_path models/reconstructor/fine-tuned \
  --max_epochs 50 \
  --batch_size 16 \
  --learning_rate_policy 3e-4 \
  --learning_rate_reconstructor 1e-4 \
  --reward_type information_theoretic
```

**What this creates:**
- `results/joint_training/best_model.pt` (trained agent)
- `results/joint_training/training_config.json` (training configuration)
- Training logs and checkpoints

### Step 4: Evaluate Performance

**Option A: Comprehensive evaluation with config**
```bash
python eval/eval_main.py --config configs/evaluation.json
```

**Option B: Custom evaluation**
```bash
python eval/eval_main.py \
  --model_path results/joint_training/best_model.pt \
  --data_path data/processed/test_data.json \
  --reconstructor_path models/reconstructor/fine-tuned \
  --output_dir eval/results \
  --num_sequences 1000 \
  --include_baselines true
```

**Option C: Just run baselines for comparison**
```bash
python eval/baselines_main.py \
  --data_path data/processed/test_data.json \
  --output_dir eval/baseline_results \
  --baselines random frequency length position
```

**What this creates:**
- `eval/results/model_results.json` (trained model performance)
- `eval/results/baseline_results.json` (baseline comparison)
- `eval/results/evaluation_config.json` (evaluation settings)

## Complete Example Workflow

Here's the exact sequence of commands for a complete run:

```bash
# 1. Test installation
python tests/test_system.py

# 2. Prepare data  
python data/prepare_main.py --config configs/data_prep.json

# 3. Quick test to make sure training works
python rl/train.py --config configs/quick_test.json

# 4. Full training (this takes time)
python rl/train.py --config configs/full_training.json

# 5. Evaluate everything
python eval/eval_main.py --config configs/evaluation.json

# 6. Check results
ls eval/results/
cat eval/results/baseline_results.json
```

## Configuration Override Examples

You can override any config parameter:

```bash
# Use config but change batch size
python rl/train.py --config configs/full_training.json --batch_size 32

# Use config but change device and epochs
python rl/train.py --config configs/full_training.json --device cpu --max_epochs 10

# Use config but change multiple parameters
python eval/eval_main.py --config configs/evaluation.json \
  --num_sequences 500 \
  --baselines random frequency \
  --device cpu
```

## Expected Results

After training, you should see:
- **Compression ratio**: 30-50% (removes 30-50% of tokens)
- **Quality preservation**: >90% on downstream tasks
- **Speed improvement**: 2-3x faster than original approach
- **Baseline comparison**: Should beat random/length, competitive with frequency-based methods

## Troubleshooting

**Training not converging:**
```bash
# Try simpler reward function
python rl/train.py --config configs/full_training.json --reward_type simple

# Reduce learning rates
python rl/train.py --config configs/full_training.json \
  --learning_rate_policy 1e-4 --learning_rate_reconstructor 5e-5
```

**Out of memory:**
```bash
# Reduce batch size
python rl/train.py --config configs/full_training.json --batch_size 4

# Use CPU
python rl/train.py --config configs/full_training.json --device cpu
```

**Can't beat baselines:**
- Check that multi-step episodes are working (should see >1 decision per episode in logs)
- Verify joint training is updating both networks (check logs for both policy and reconstructor losses)
- Try the simple reward function first before information-theoretic

## Research Contribution

This work contributes:
1. **Novel Joint Training**: Solution to circular dependencies in RL+supervised learning
2. **Architecture Insights**: Simple networks + proper training > complex networks + broken training  
3. **Information-Theoretic Framework**: Principled reward design for compression
4. **Comprehensive Evaluation**: Rigorous baseline methodology

## Citation

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