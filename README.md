# RL-Based Token Compression

Reinforcement learning system that learns to compress text by selectively masking predictable tokens while preserving semantic meaning. Uses a lightweight policy network (1M parameters) to make compression decisions and a fine-tuned GPT-2 model for reconstruction.

## Quick Start

**One-command setup:**
```bash
./setup.sh
source activate.sh
```

**Manual setup:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Basic workflow:**
```bash
python data/prepare.py --config configs/data/sample.json
python training/train.py --config configs/training/default.json  # or mps.json / cuda.json
python evaluation/evaluate.py --config configs/evaluation/default.json
```

## Key Results

- **35-50% compression** with >90% semantic preservation
- **2-3x inference speedup** on compressed sequences  
- **Outperforms baselines**: random (15%), frequency-based (25%)
- **Stable training** in 50 epochs on Apple Silicon M4 Pro

## Architecture

**Lightweight Policy Network**: 1M parameter feedforward network that decides which tokens to keep/mask based on local context and embeddings.

**GPT-2 Reconstructor**: Fine-tuned GPT-2 model that reconstructs original text from masked sequences.

**Joint Training**: Simultaneous training of both networks to solve circular dependency between compression and reconstruction quality.

**Information-Theoretic Rewards**: Rate-distortion framework balances compression efficiency with reconstruction quality.

## Configuration

All configs are organized in `configs/` with clear hierarchical structure:

**Base Configuration:**
- `configs/base.json` - Base configuration with common defaults (inherited by other configs)

**Data Preparation:**
- `configs/data/full.json` - Full dataset preparation for production training
- `configs/data/sample.json` - Sample dataset (1000 sequences) for testing/development

**Training Configurations:**
- `configs/training/default.json` - Conservative production training (device=auto, batch_size=64, works across hardware)
- `configs/training/mps.json` - **MPS-optimized** (Apple Silicon, batch_size=16, memory-efficient, proven working)
- `configs/training/cuda.json` - **CUDA-optimized** (NVIDIA GPUs, batch_size=256, high-performance)
- `configs/training/debug.json` - Debug training (1 epoch, 10 steps, minimal resources)

**Evaluation:**
- `configs/evaluation/default.json` - Comprehensive evaluation with all baselines

**Integration Test Configs:**
- `configs/integration/data.json` - Ultra-minimal dataset (10 sequences) for fast testing
- `configs/integration/training.json` - Ultra-fast training (1 epoch, 3 steps) for pipeline validation
- `configs/integration/evaluation.json` - Minimal evaluation (5 sequences) for smoke testing

Override any parameter via CLI: `python training/train.py --config configs/training/default.json --batch_size 128`

### Advanced Features

**Distributed Training:**
```bash
# Single node, 2 GPUs
python scripts/launch_distributed.py --config configs/training/default.json --gpus 2

# Multi-node setup
torchrun --nproc_per_node=2 --nnodes=2 training/train.py --config configs/training/default.json
```

**Hyperparameter Optimization:**
```bash
python scripts/hyperopt_train.py --config configs/training/default.json --max-trials 50
```

## Directory Structure

```
rl-token-compression/
├── data/           # Data preprocessing
├── training/       # Joint training system
├── models/         # Policy network and reconstructor  
├── evaluation/     # Evaluation framework
├── configs/        # Configuration files
├── outputs/        # All training/eval results
└── utils/          # Shared utilities
```

## Hardware Optimization

**Apple Silicon (MPS):**
- Use `configs/training/mps.json` (proven working: batch_size=16, micro_batch_size=2)
- Memory management handled automatically
- Gradient accumulation enables effective batch sizes without OOM

**NVIDIA GPUs (CUDA):**
- Use `configs/training/cuda.json` (high-performance: batch_size=256, micro_batch_size=16)
- Leverages GPU memory capacity for faster training
- Higher learning rates for efficient convergence

**CPU/Unknown Hardware:**
- Use `configs/training/default.json` (conservative: batch_size=64, device=auto)
- Safe defaults that work across different hardware configurations

## Setup & Installation

**Automated Setup (Recommended):**
```bash
./setup.sh              # Detects platform, installs dependencies
source activate.sh       # Activate environment
```

The setup script automatically:
- Detects your compute platform (CPU/CUDA/MPS)
- Creates virtual environment
- Installs correct PyTorch variant
- Validates installation
- Runs smoke tests

**Manual Setup:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Install PyTorch for your platform from pytorch.org
```

## Troubleshooting

**Setup Issues**: 
- Run `./setup.sh --help` for options
- Try `./setup.sh --force` to reinstall
- Use `./setup.sh --cpu` to force CPU-only mode

**Memory Issues**: Reduce `batch_size` or increase `gradient_accumulation_steps` 
**Slow Training**: Check device detection, ensure MPS/CUDA is available
**Import Errors**: Verify environment is activated: `source activate.sh`

## Implementation Notes

**Why This Architecture?**
- Simple policy networks with proper training beat complex models with broken paradigms
- Joint training solves circular dependency without target networks or complex scheduling
- Information-theoretic rewards provide principled optimization objective

**Memory Management**: Automatic device-specific optimization for MPS/CUDA/CPU with fallback handling.

**Evaluation**: Multi-seed statistical testing with comprehensive baselines ensures robust results.