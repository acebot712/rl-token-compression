"""
Lightweight evaluation utilities expected by tests.

Provides a minimal `calculate_perplexity` that works with Hugging Face
causal language models such as `GPT2LMHeadModel`.
"""

from typing import Union

import torch
from torch import Tensor


def calculate_perplexity(model, input_ids: Tensor, device: Union[str, torch.device] = "cpu") -> float:
    """
    Compute perplexity for given tokenized inputs using an autoregressive LM.

    Args:
        model: Hugging Face causal LM with a `loss` when labels are provided
        input_ids: Tensor of token ids shaped (batch, seq_len)
        device: Device to use (e.g., "cpu", "cuda", "mps")

    Returns:
        Perplexity as a float
    """
    if isinstance(device, str):
        device = torch.device(device)

    model_device = next(model.parameters()).device
    if model_device != device:
        model = model.to(device)

    input_ids = input_ids.to(device)

    model.eval()
    with torch.no_grad():
        # For causal LMs, providing labels equal to input_ids computes next-token loss
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss: Tensor = outputs.loss  # shape: ()
        perplexity = torch.exp(loss).item()

    return float(perplexity)


