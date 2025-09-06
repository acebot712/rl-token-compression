"""
Simple metrics collection for training.

Basic metrics tracking without over-engineering.
"""

from typing import Dict

from .logging import get_logger

logger = get_logger(__name__)


def record_training_metrics(step: int, epoch: int, metrics: Dict[str, float]):
    """Record basic training metrics."""
    logger.info(f"Step {step}, Epoch {epoch}: {metrics}")


def record_gradient_health(model_name: str, grad_norm: float, nan_count: int, inf_count: int):
    """Record gradient health metrics."""
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"{model_name} unhealthy gradients: {nan_count} NaN, {inf_count} Inf")
    else:
        logger.debug(f"{model_name} gradient norm: {grad_norm:.4f}")


def get_metrics_service():
    """Get metrics service (simplified - just returns None)."""
    return None
