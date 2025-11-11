"""
Reproducibility utilities for setting random seeds across libraries.

Ensures deterministic behavior for PyTorch, NumPy, and Python's random module.
"""
import os
import random
from typing import Optional

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA operations (cuDNN)

    Args:
        seed: Random seed value (integer)
        deterministic: If True, enables deterministic CUDA operations
                      (may reduce performance but ensures full reproducibility)

    Examples:
        >>> set_all_seeds(42)
        >>> # Now all random operations will be deterministic
        >>> torch.randn(3, 3)  # Reproducible
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Environment variables for better reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Configure PyTorch for deterministic behavior
    if deterministic:
        # CuDNN deterministic mode (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where available (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError as e:
                # Some operations don't have deterministic implementations
                print(f"Warning: Could not enable all deterministic algorithms: {e}")
    else:
        # Enable cudnn autotuner for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True


def make_reproducible(seed: int = 42, deterministic: bool = True) -> None:
    """
    Wrapper around set_all_seeds with default seed value.

    Args:
        seed: Random seed (default: 42)
        deterministic: Enable deterministic operations (default: True)

    Examples:
        >>> make_reproducible()
        >>> # Training will now be reproducible
    """
    set_all_seeds(seed, deterministic)
    print(f"Random seed set to {seed} (deterministic={deterministic})")


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """
    Initialize worker processes with different seeds for DataLoader.

    Use this as worker_init_fn in PyTorch DataLoader to ensure each worker
    has a different but reproducible seed.

    Args:
        worker_id: Worker ID (automatically passed by DataLoader)
        seed: Base seed (if None, uses a default)

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init_fn
        ... )
    """
    if seed is None:
        seed = torch.initial_seed() % 2**32

    worker_seed = seed + worker_id

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch random number generator with a specific seed.

    Useful for creating reproducible random splits and data augmentation.

    Args:
        seed: Random seed

    Returns:
        torch.Generator with the specified seed

    Examples:
        >>> generator = get_generator(42)
        >>> # Use for reproducible splits
        >>> from torch.utils.data import random_split
        >>> train, val = random_split(dataset, [0.8, 0.2], generator=generator)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_context(seed: int, deterministic: bool = True):
    """
    Context manager for temporarily setting random seeds.

    Useful for specific operations that need reproducibility without
    affecting the global random state.

    Args:
        seed: Random seed
        deterministic: Enable deterministic operations

    Examples:
        >>> with seed_context(42):
        ...     x = torch.randn(3, 3)  # Reproducible
        >>> # Outside context, previous random state is restored
    """
    # Save current states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()

    # Save current cudnn settings
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        # Set new seed
        set_all_seeds(seed, deterministic)
        yield
    finally:
        # Restore previous states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

        # Restore cudnn settings
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


class ReproducibilityConfig:
    """
    Configuration class for reproducibility settings.

    Can be saved/loaded for experiment tracking.
    """
    def __init__(
        self,
        seed: int = 42,
        deterministic: bool = True,
        num_workers: int = 4,
        persistent_workers: bool = True
    ):
        """
        Args:
            seed: Random seed
            deterministic: Enable deterministic operations
            num_workers: Number of DataLoader workers
            persistent_workers: Keep workers alive between epochs
        """
        self.seed = seed
        self.deterministic = deterministic
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def apply(self) -> None:
        """Apply reproducibility settings."""
        set_all_seeds(self.seed, self.deterministic)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'seed': self.seed,
            'deterministic': self.deterministic,
            'num_workers': self.num_workers,
            'persistent_workers': self.persistent_workers,
            'cuda_available': torch.cuda.is_available(),
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        }

    def __repr__(self) -> str:
        return (
            f"ReproducibilityConfig(seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"num_workers={self.num_workers})"
        )
