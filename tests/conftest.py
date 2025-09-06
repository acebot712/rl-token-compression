"""
Pytest configuration and shared fixtures.
"""

import shutil
import tempfile

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def sample_config():
    """Standard configuration for testing."""
    return {
        "batch_size": 4,
        "learning_rate_policy": 1e-3,
        "learning_rate_reconstructor": 1e-4,
        "max_epochs": 2,
        "context_window": 3,
        "device": "cpu",
    }


@pytest.fixture
def sample_sequences():
    """Sample token sequences for testing."""
    return [[1, 2, 3, 4, 5], [10, 11, 12, 13], [20, 21, 22, 23, 24, 25], [30, 31, 32]]
