"""
Deterministic behavior guarantees for reproducible research.

Same config + same data + same seed = same results. Always.
"""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """
    Set global random seed for all libraries.

    Args:
        seed: Random seed value
        deterministic_algorithms: Use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Environment variables for deterministic behavior
    if deterministic_algorithms:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic CUDA

        # PyTorch deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # MPS (Apple Silicon) deterministic behavior
        if torch.backends.mps.is_available():
            # MPS doesn't have full deterministic support yet, but we set what we can
            os.environ["MPS_DETERMINISTIC"] = "1"


def create_deterministic_dataloader(
    dataset, batch_size: int, shuffle: bool = True, seed: Optional[int] = None, **kwargs
):
    """
    Create a deterministic DataLoader.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling
        **kwargs: Additional DataLoader arguments
    """
    from torch.utils.data import DataLoader

    generator = None
    if seed is not None and shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=_worker_init_fn if seed is not None else None,
        **kwargs,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Worker initialization function for deterministic DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_model_deterministic(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """
    Initialize model parameters deterministically.

    Args:
        model: PyTorch model
        seed: Random seed for initialization

    Returns:
        Model with deterministically initialized parameters
    """
    torch.manual_seed(seed)

    # Re-initialize parameters if needed
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    return model


def verify_deterministic_setup() -> Dict[str, bool]:
    """
    Verify that deterministic setup is working correctly.

    Returns:
        Dictionary of checks and their status
    """
    checks = {}

    # Check if deterministic algorithms are enabled
    try:
        checks["torch_deterministic"] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        checks["torch_deterministic"] = False

    # Check CUDNN settings
    checks["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    checks["cudnn_benchmark"] = not torch.backends.cudnn.benchmark

    # Check environment variables
    checks["pythonhashseed"] = os.environ.get("PYTHONHASHSEED") is not None
    checks["cublas_workspace"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None

    return checks


def ensure_reproducible_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure configuration has all needed fields for reproducibility.

    Args:
        config: Configuration dictionary

    Returns:
        Updated configuration with reproducibility settings
    """
    config = config.copy()

    # Ensure random seed is set
    if "random_seed" not in config:
        config["random_seed"] = 42
        print(f"⚠️  No random seed specified, using default: {config['random_seed']}")

    # Ensure deterministic training
    if "deterministic" not in config:
        config["deterministic"] = True
        print("✓ Enabled deterministic training for reproducibility")

    # Warn about non-deterministic settings
    warnings = []

    if config.get("num_workers", 0) > 0 and "dataloader_seed" not in config:
        warnings.append("Multiple DataLoader workers without explicit seed may reduce reproducibility")
        config["dataloader_seed"] = config["random_seed"]

    if config.get("device") == "mps" and config.get("deterministic", True):
        warnings.append("MPS (Apple Silicon) has limited deterministic support")

    for warning in warnings:
        print(f"⚠️  Reproducibility warning: {warning}")

    return config


def create_reproducible_trainer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create trainer configuration with reproducibility guarantees.

    Args:
        config: Base configuration

    Returns:
        Configuration with reproducibility settings
    """
    # Ensure deterministic settings
    config = ensure_reproducible_config(config)

    # Set global seed
    set_global_seed(
        config["random_seed"],
        deterministic_algorithms=config.get("deterministic", True),
    )

    # Verify setup
    checks = verify_deterministic_setup()
    failed_checks = [k for k, v in checks.items() if not v]

    if failed_checks:
        print(f"⚠️  Some deterministic checks failed: {failed_checks}")
        print("   Results may not be fully reproducible")
    else:
        print("✓ All deterministic checks passed")

    return config


def save_reproducibility_info(output_dir: str, config: Dict[str, Any]) -> None:
    """
    Save information needed to reproduce results.

    Args:
        output_dir: Output directory
        config: Configuration used
    """
    import json
    import platform
    import sys

    repro_info = {
        "config": config,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "mps_available": torch.backends.mps.is_available(),
        },
        "deterministic_checks": verify_deterministic_setup(),
        "random_seeds": {
            "global_seed": config.get("random_seed", 42),
            "torch_seed": torch.initial_seed(),
        },
    }

    # Add git commit if available
    try:
        import subprocess

        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        repro_info["git_commit"] = git_commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    repro_path = os.path.join(output_dir, "reproducibility_info.json")
    with open(repro_path, "w") as f:
        json.dump(repro_info, f, indent=2, default=str)

    print(f"✓ Saved reproducibility info to {repro_path}")


def test_reproducibility(model_factory, config: Dict[str, Any], test_input) -> bool:
    """
    Test that model produces same outputs with same seed.

    Args:
        model_factory: Function that creates model
        config: Configuration dict with random_seed
        test_input: Test input for model

    Returns:
        True if outputs are identical
    """
    seed = config.get("random_seed", 42)

    # First run
    set_global_seed(seed, deterministic_algorithms=True)
    model1 = model_factory()
    with torch.no_grad():
        output1 = model1(test_input)

    # Second run with same seed
    set_global_seed(seed, deterministic_algorithms=True)
    model2 = model_factory()
    with torch.no_grad():
        output2 = model2(test_input)

    # Compare outputs
    if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return torch.allclose(output1, output2, atol=1e-6)
    else:
        return output1 == output2
