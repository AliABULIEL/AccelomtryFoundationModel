"""
Preprocessing operations for accelerometry data.

Implements instance standardization (RevIN-style) for time series normalization.
"""
from typing import Tuple, Optional

import torch
import numpy as np


def instance_standardize(
    x: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Per-window per-channel z-score normalization (RevIN-style).

    Computes mean and std along the time dimension for each window and channel,
    then standardizes to zero mean and unit variance.

    Args:
        x: Input tensor of shape (B, C, T) where
           B = batch size
           C = number of channels (typically 3 for x,y,z)
           T = time steps (window length)
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Tuple of (normalized_tensor, (mean, std)) where:
        - normalized_tensor: Standardized tensor of same shape as input
        - mean: Tensor of shape (B, C, 1) containing per-window per-channel means
        - std: Tensor of shape (B, C, 1) containing per-window per-channel stds

    Examples:
        >>> x = torch.randn(32, 3, 819)  # Batch of 32 windows, 3 channels, 819 samples
        >>> x_norm, (mean, std) = instance_standardize(x)
        >>> x_norm.shape
        torch.Size([32, 3, 819])
        >>> mean.shape
        torch.Size([32, 3, 1])
        >>> # Verify normalization
        >>> torch.allclose(x_norm.mean(dim=-1), torch.zeros(32, 3), atol=1e-6)
        True
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3-D input (B, C, T), got shape {x.shape}")

    # Compute mean and std along time dimension (dim=-1)
    # keepdim=True maintains shape (B, C, 1) for broadcasting
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    # Add eps to avoid division by zero
    std = std + eps

    # Standardize
    x_normalized = (x - mean) / std

    return x_normalized, (mean, std)


def inverse_standardize(
    y_pred: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    """
    Reverse instance standardization to recover original scale.

    Args:
        y_pred: Predicted/normalized tensor of shape (B, C, T)
        mean: Mean tensor of shape (B, C, 1) from instance_standardize
        std: Std tensor of shape (B, C, 1) from instance_standardize

    Returns:
        Tensor in original scale of shape (B, C, T)

    Examples:
        >>> x = torch.randn(32, 3, 819)
        >>> x_norm, (mean, std) = instance_standardize(x)
        >>> x_reconstructed = inverse_standardize(x_norm, mean, std)
        >>> torch.allclose(x, x_reconstructed, atol=1e-5)
        True
    """
    if y_pred.dim() != 3:
        raise ValueError(f"Expected 3-D input (B, C, T), got shape {y_pred.shape}")

    # Reverse the standardization: x = (x_norm * std) + mean
    y_original = (y_pred * std) + mean

    return y_original


def batch_instance_standardize(
    x: np.ndarray,
    eps: float = 1e-5
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    NumPy version of instance_standardize for preprocessing pipelines.

    Args:
        x: Input array of shape (B, C, T)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (normalized_array, (mean, std))

    Examples:
        >>> x = np.random.randn(32, 3, 819)
        >>> x_norm, (mean, std) = batch_instance_standardize(x)
        >>> x_norm.shape
        (32, 3, 819)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3-D input (B, C, T), got shape {x.shape}")

    # Compute mean and std along time dimension (axis=-1)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)

    # Add eps to avoid division by zero
    std = std + eps

    # Standardize
    x_normalized = (x - mean) / std

    return x_normalized, (mean, std)


def batch_inverse_standardize(
    y_pred: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    NumPy version of inverse_standardize.

    Args:
        y_pred: Predicted/normalized array of shape (B, C, T)
        mean: Mean array of shape (B, C, 1)
        std: Std array of shape (B, C, 1)

    Returns:
        Array in original scale of shape (B, C, T)
    """
    if y_pred.ndim != 3:
        raise ValueError(f"Expected 3-D input (B, C, T), got shape {y_pred.shape}")

    y_original = (y_pred * std) + mean

    return y_original


def robust_standardize(
    x: torch.Tensor,
    quantile_range: Tuple[float, float] = (0.25, 0.75),
    eps: float = 1e-5
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Robust standardization using median and IQR instead of mean and std.

    More robust to outliers and artifacts in accelerometry data.

    Args:
        x: Input tensor of shape (B, C, T)
        quantile_range: Tuple of (lower, upper) quantiles for IQR (default: 0.25, 0.75)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (normalized_tensor, (median, iqr))

    Examples:
        >>> x = torch.randn(32, 3, 819)
        >>> x_norm, (median, iqr) = robust_standardize(x)
        >>> x_norm.shape
        torch.Size([32, 3, 819])
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3-D input (B, C, T), got shape {x.shape}")

    # Compute median along time dimension
    median = x.median(dim=-1, keepdim=True)[0]

    # Compute IQR
    q_low = x.quantile(quantile_range[0], dim=-1, keepdim=True)
    q_high = x.quantile(quantile_range[1], dim=-1, keepdim=True)
    iqr = q_high - q_low + eps

    # Standardize using median and IQR
    x_normalized = (x - median) / iqr

    return x_normalized, (median, iqr)


def inverse_robust_standardize(
    y_pred: torch.Tensor,
    median: torch.Tensor,
    iqr: torch.Tensor
) -> torch.Tensor:
    """
    Reverse robust standardization.

    Args:
        y_pred: Predicted/normalized tensor of shape (B, C, T)
        median: Median tensor of shape (B, C, 1)
        iqr: IQR tensor of shape (B, C, 1)

    Returns:
        Tensor in original scale
    """
    if y_pred.dim() != 3:
        raise ValueError(f"Expected 3-D input (B, C, T), got shape {y_pred.shape}")

    y_original = (y_pred * iqr) + median

    return y_original
