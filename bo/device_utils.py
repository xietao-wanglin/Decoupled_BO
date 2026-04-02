"""Centralised device and dtype management.

Import ``DEVICE`` and ``DTYPE`` from here instead of hardcoding
``torch.device("cpu")`` in every module.

CUDA is used automatically when available.
"""

import torch

DTYPE = torch.double

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
