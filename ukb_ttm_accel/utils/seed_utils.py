# FILE: ukb_ttm_accel/utils/seed_utils.py

"""
Random seed utilities for reproducible experiments.

Ensures reproducibility across random, numpy, and PyTorch.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Seeds:
    - Python random module
    - NumPy random generator
    - PyTorch CPU and GPU random generators

    Args:
        seed: Random seed value
        deterministic: If True, set PyTorch to deterministic mode.
                      Warning: This may reduce performance.

    Example:
        >>> set_global_seed(42)
        Random seed set to 42 for reproducibility
        >>> # All subsequent random operations will be reproducible
    """
    # Python random module
    random.seed(seed)

    # NumPy random generator
    np.random.seed(seed)

    # PyTorch CPU random generator
    torch.manual_seed(seed)

    # PyTorch GPU random generators (if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set deterministic mode if requested
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: This may reduce performance, especially for variable input sizes

    print(f"Random seed set to {seed} for reproducibility")
    if deterministic:
        print("  PyTorch deterministic mode enabled (may reduce performance)")


def get_rng_state() -> dict:
    """
    Get the current state of all random number generators.

    Returns:
        Dictionary containing RNG states for all libraries

    Example:
        >>> state = get_rng_state()
        >>> # ... do some random operations ...
        >>> restore_rng_state(state)  # Restore to previous state
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: dict) -> None:
    """
    Restore random number generator states.

    Args:
        state: Dictionary containing RNG states (from get_rng_state)

    Example:
        >>> state = get_rng_state()
        >>> # ... do some random operations ...
        >>> restore_rng_state(state)
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
