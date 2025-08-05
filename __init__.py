"""
RL Token Compression - Clean, focused implementation.

No over-engineering. No unnecessary complexity.
Production-quality code that does one thing well: token compression via RL.
"""

__version__ = "0.2.0"
__author__ = "Refactored by Linus Torvalds principles"

# Core modules
from . import utils
from . import models
from . import training
from . import rl
from . import data
from . import eval

__all__ = [
    "utils",
    "models", 
    "training",
    "rl",
    "data",
    "eval"
]