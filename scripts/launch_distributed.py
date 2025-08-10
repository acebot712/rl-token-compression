#!/usr/bin/env python3
"""
Launch script for distributed training.

Usage:
    # Single node, 2 GPUs
    python scripts/launch_distributed.py --config configs/training.json --gpus 2
    
    # Multi-node (run on each node with appropriate settings)
    python scripts/launch_distributed.py --config configs/training.json --gpus 2 \
        --master-addr 192.168.1.100 --master-port 12355 --node-rank 0 --num-nodes 2
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch distributed training")
    
    # Training arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Configuration file")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs per node")
    
    # Distributed arguments
    parser.add_argument("--master-addr", type=str, default="localhost",
                       help="Master node address")
    parser.add_argument("--master-port", type=str, default="12355", 
                       help="Master node port")
    parser.add_argument("--node-rank", type=int, default=0,
                       help="Node rank")
    parser.add_argument("--num-nodes", type=int, default=1,
                       help="Number of nodes")
    
    # Additional training arguments
    parser.add_argument("--batch-size", type=int,
                       help="Override batch size")
    parser.add_argument("--max-epochs", type=int,
                       help="Override max epochs")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.gpus < 1:
        print("Error: Number of GPUs must be >= 1")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Setup environment variables
    env = os.environ.copy()
    env['MASTER_ADDR'] = args.master_addr
    env['MASTER_PORT'] = args.master_port
    
    # Calculate world size
    world_size = args.gpus * args.num_nodes
    env['WORLD_SIZE'] = str(world_size)
    
    # Build torchrun command
    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        f"--nnodes={args.num_nodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "training/train.py",
        "--config", args.config,
        "--distributed"
    ]
    
    # Add optional overrides
    if args.batch_size:
        torchrun_cmd.extend(["--batch_size", str(args.batch_size)])
    if args.max_epochs:
        torchrun_cmd.extend(["--max_epochs", str(args.max_epochs)])
    
    print("Launching distributed training...")
    print(f"Command: {' '.join(torchrun_cmd)}")
    print(f"World size: {world_size} (GPUs per node: {args.gpus}, Nodes: {args.num_nodes})")
    print()
    
    try:
        result = subprocess.run(torchrun_cmd, env=env, check=True)
        print("\\n✓ Distributed training completed successfully")
        return result.returncode
    
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Distributed training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
        return 130
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())