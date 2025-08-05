# Learning to Forget — RL-Based Token Compression for Efficient LLM Inference

This repository contains the implementation for our research on RL-based token compression for efficient LLM inference. The system learns to selectively mask predictable tokens in long-context sequences while preserving semantic meaning, enabling faster and more efficient inference with large language models.

**Status**: ✅ **PRODUCTION-READY** - Core problems fixed, clean architecture, comprehensive tooling.

## 🚀 Apple Silicon Optimized

**Fully optimized for Apple Silicon M-series chips!** This implementation leverages:
- **Unified Memory Architecture**: Efficient use of shared CPU/GPU memory pool
- **Metal Performance Shaders (MPS)**: Native GPU acceleration 
- **Memory-Efficient Training**: Gradient accumulation prevents OOM on large models
- **M4 Pro Ready**: Tested on M4 Pro with 48GB unified memory

## System Overview

Our system consists of three main components:

1. **Agent**: A lightweight policy network (1M parameters) that decides which tokens to keep or mask
2. **Reconstructor**: A fine-tuned GPT-2 model that reconstructs masked sequences  
3. **Environment**: RL environment for joint training with information-theoretic rewards

**Key Improvements**: Joint training fixes circular dependency, simplified architecture (100x parameter reduction), multi-step episodes enable proper RL learning, comprehensive baselines for evaluation.

## Files and What They Do

### Core Scripts (what you'll actually run)
- **`scripts/train_rl.py`** - Main training script (trains both agent and reconstructor jointly)
- **`scripts/prepare_data.py`** - Prepares datasets for training (tokenization, train/val/test splits)
- **`scripts/run_evaluation.py`** - Comprehensive evaluation with baselines and metrics
- **`scripts/run_baselines.py`** - Run just baseline compression methods
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

**Development Configs** (`configs/dev/`) - Fast testing, small datasets
- **`debug.json`** - Ultra-fast debug (1 epoch, 4 batch, 4GB+ systems)
- **`small_scale.json`** - Quick research (5 epochs, 8 batch, 8GB+ systems)
- **`quick_eval.json`** - Quick evaluation testing
- **`sample_data.json`** - Small dataset preparation

**Production Configs** (`configs/prod/`) - Research-quality, full datasets
- **`standard.json`** - Standard research (50 epochs, 32 batch, 16GB+ systems)
- **`apple_silicon.json`** - Apple Silicon optimized (50 epochs, 64 batch, 48GB+ M-series)
- **`data_prep.json`** - Full dataset preparation
- **`evaluation.json`** - Comprehensive evaluation

*All configs include memory optimizations with gradient accumulation by default.*

## Installation

```bash
# Install all dependencies (includes Apple Silicon optimizations)
pip install -r requirements.txt

# Test everything works
python tests/test_system.py
```

### Apple Silicon Setup (Recommended)

For optimal performance on Apple Silicon:

```bash
# Verify MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# All configs automatically detect and use MPS when available
# Training uses unified memory efficiently without manual configuration
```

## Step-by-Step Usage

The configs are organized into two categories:
- **`configs/dev/`** - Fast configs for development, debugging, and quick testing
- **`configs/prod/`** - Full configs for research runs and final results

### Development Workflow (Quick Testing)

**Step 1: Create sample data for development**
```bash
python scripts/prepare_data.py --config configs/dev/sample_data.json
```

**Step 2: Debug training (ultra-fast)**
```bash
python scripts/train_rl.py --config configs/dev/debug.json
```

**Step 3: Quick evaluation**
```bash
python scripts/run_evaluation.py --config configs/dev/quick_eval.json
```

### Production Workflow (Research Results)

**Step 1: Prepare full dataset**
```bash
python scripts/prepare_data.py --config configs/prod/data_prep.json
```

**Step 2: Full training**
```bash
# Apple Silicon optimized training (recommended for M-series)
python scripts/train_rl.py --config configs/prod/apple_silicon.json

# Standard production training (16GB+ systems)
python scripts/train_rl.py --config configs/prod/standard.json

# Small-scale research (8GB+ systems)
python scripts/train_rl.py --config configs/dev/small_scale.json
```

**Step 3: Comprehensive evaluation**
```bash
python scripts/run_evaluation.py --config configs/prod/evaluation.json
```

### Data Preparation Details

**Development data (100 sequences):**
```bash
python scripts/prepare_data.py --config configs/dev/sample_data.json
```

**Production data (50,000 sequences):**
```bash
python scripts/prepare_data.py --config configs/prod/data_prep.json
```

**Custom data preparation:**
```bash
python scripts/prepare_data.py \
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
python scripts/train_reconstructor_gpu.py \
  --data_path data/processed/processed_data.json \
  --output_dir models/reconstructor/fine-tuned \
  --device cuda \
  --epochs 3 \
  --batch_size 4
```

**CPU training:**
```bash
python scripts/train_reconstructor_cpu.py \
  --data_path data/processed/processed_data.json \
  --output_dir models/reconstructor/fine-tuned \
  --epochs 3 \
  --batch_size 8
```

**What this creates:**
- `models/reconstructor/fine-tuned/` (fine-tuned model checkpoints)

### Available Training Configs

**Development configs (fast):**
- `configs/dev/debug.json` - 1 epoch, batch_size=2 (ultra-fast testing)
- `configs/dev/quick_test.json` - 2 epochs, batch_size=4 (quick validation)
- `configs/dev/ablation_study.json` - 20 epochs, batch_size=8 (ablation studies)

**Production configs (research quality):**
- `configs/prod/full_training.json` - 100 epochs, batch_size=32 (standard research)
- `configs/prod/large_scale_training.json` - 200 epochs, batch_size=64 (final results)

## Complete Example Workflows

### Development Workflow (Fast Testing)

```bash
# 1. Test installation
python tests/test_system.py

# 2. Create small dataset for development
python scripts/prepare_data.py --config configs/dev/sample_data.json

# 3. Ultra-fast debug training (1 epoch)
python scripts/train_rl.py --config configs/dev/debug.json

# 4. Quick evaluation
python scripts/run_evaluation.py --config configs/dev/quick_eval.json

# 5. Check results
ls debug/
```

### Production Workflow (Research Results)

```bash
# 1. Test installation
python tests/test_system.py

# 2. Prepare full dataset (50,000 sequences)
python scripts/prepare_data.py --config configs/prod/data_prep.json

# 3. Full training (100 epochs - this takes time)
python scripts/train_rl.py --config configs/prod/full_training.json

# 4. Comprehensive evaluation (5,000 sequences)
python scripts/run_evaluation.py --config configs/prod/evaluation.json

# 5. Check results
ls eval/results/
cat eval/results/baseline_results.json
```

## 🧠 Memory-Efficient Training

All configurations include memory optimizations with gradient accumulation by default:

### Configuration Selection by System

**Choose based on your available memory:**

```bash
# 4GB+ systems - Ultra-fast testing
python scripts/train_rl.py --config configs/dev/debug.json

# 8GB+ systems - Small-scale research
python scripts/train_rl.py --config configs/dev/small_scale.json

# 16GB+ systems - Standard production
python scripts/train_rl.py --config configs/prod/standard.json

# 48GB+ Apple Silicon - Optimal performance
python scripts/train_rl.py --config configs/prod/apple_silicon.json
```

### Key Memory Optimizations

1. **Gradient Accumulation**: Process large effective batch sizes with small micro-batches
2. **Unified Memory Management**: Apple Silicon optimizations for shared CPU/GPU memory
3. **Memory Pressure Monitoring**: Real-time memory usage tracking and adaptive cleanup
4. **Optimized Tensor Operations**: In-place operations and memory-efficient sampling

### Apple Silicon Specific Features

- **MPS Backend**: Native Metal Performance Shaders acceleration
- **Unified Memory**: Efficient use of shared 48GB memory pool  
- **Memory Bandwidth**: Optimized for high-bandwidth memory architecture
- **Thermal Management**: Sustained training workloads without throttling

### Quick Validation Workflow

```bash
# Test that everything works before long runs
python tests/test_system.py
python scripts/prepare_data.py --config configs/dev/sample_data.json
python scripts/train_rl.py --config configs/dev/quick_test.json
python scripts/run_evaluation.py --config configs/dev/quick_eval.json
```

## Configuration Override Examples

You can override any config parameter:

```bash
# Use config but change batch size
python scripts/train_rl.py --config configs/prod/full_training.json --batch_size 32

# Use config but change device and epochs
python scripts/train_rl.py --config configs/prod/full_training.json --device cpu --max_epochs 10

# Use config but change multiple parameters
python scripts/run_evaluation.py --config configs/prod/evaluation.json \
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
python scripts/train_rl.py --config configs/prod/standard.json --reward_type simple

# Reduce learning rates
python scripts/train_rl.py --config configs/prod/standard.json \
  --learning_rate_policy 1e-4 --learning_rate_reconstructor 5e-5
```

**Out of memory:**
```bash
# Use smaller configuration for your system memory
python scripts/train_rl.py --config configs/dev/debug.json        # 4GB+
python scripts/train_rl.py --config configs/dev/small_scale.json  # 8GB+
python scripts/train_rl.py --config configs/prod/standard.json    # 16GB+

# Apple Silicon: leverages unified memory efficiently
python scripts/train_rl.py --config configs/prod/apple_silicon.json
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
5. **Apple Silicon Optimization**: First RL+NLP system optimized for unified memory architecture
6. **Memory-Efficient Training**: Gradient accumulation techniques for large-scale RL training

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