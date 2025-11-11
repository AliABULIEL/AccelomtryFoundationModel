"""
Unit tests for NHANES 80 Hz processing.
"""

import numpy as np
import pytest

from src.dataio.nhanes.parse_80hz import (
    compute_window_params,
    to_windows,
    standardize_windows
)


def test_compute_window_params():
    """Test window parameter computation for 80 Hz."""
    # Test default parameters
    win_n, hop_n = compute_window_params(fs=80)

    # 80 Hz * 8.192s = 655.36 -> floor = 655
    assert win_n == 655, f"Expected win_n=655, got {win_n}"

    # 80 Hz * 4.096s = 327.68 -> floor = 327
    assert hop_n == 327, f"Expected hop_n=327, got {hop_n}"


def test_compute_window_params_rounding():
    """Test different rounding modes."""
    # Floor
    win_n_floor, hop_n_floor = compute_window_params(80, 8.192, 4.096, 'floor')
    assert win_n_floor == 655
    assert hop_n_floor == 327

    # Nearest
    win_n_nearest, hop_n_nearest = compute_window_params(80, 8.192, 4.096, 'nearest')
    assert win_n_nearest == 655
    assert hop_n_nearest == 328

    # Ceil
    win_n_ceil, hop_n_ceil = compute_window_params(80, 8.192, 4.096, 'ceil')
    assert win_n_ceil == 656
    assert hop_n_ceil == 328


def test_to_windows():
    """Test windowing with synthetic data."""
    # Create synthetic signal: 10000 samples, 3 channels
    signals = np.random.randn(10000, 3).astype(np.float32)

    # Create windows
    win_n, hop_n = 655, 327
    windows, spans = to_windows(signals, win_n, hop_n, pad_mode='none')

    # Check shapes
    expected_num_windows = (10000 - 655) // 327 + 1
    assert windows.shape == (expected_num_windows, 3, 655), \
        f"Expected shape ({expected_num_windows}, 3, 655), got {windows.shape}"

    assert spans.shape == (expected_num_windows, 2), \
        f"Expected spans shape ({expected_num_windows}, 2), got {spans.shape}"

    # Check spans length matches windows
    assert len(spans) == len(windows)

    # Check first and last spans
    assert spans[0][0] == 0
    assert spans[0][1] == 655

    # Check that windows have correct overlap
    assert spans[1][0] == hop_n
    assert spans[1][1] == hop_n + win_n


def test_to_windows_exact_calculation():
    """Test exact window calculation."""
    # T=10000, win_n=655, hop_n=327
    # num_windows = (10000 - 655) // 327 + 1 = 9345 // 327 + 1 = 28 + 1 = 29
    signals = np.random.randn(10000, 3).astype(np.float32)
    windows, spans = to_windows(signals, 655, 327, pad_mode='none')

    assert len(windows) == 29, f"Expected 29 windows, got {len(windows)}"


def test_to_windows_padding():
    """Test padding modes."""
    # Short signal
    signals = np.random.randn(500, 3).astype(np.float32)

    # No padding - should return empty
    windows_none, spans_none = to_windows(signals, 655, 327, pad_mode='none')
    assert len(windows_none) == 0

    # Edge padding - should work
    windows_edge, spans_edge = to_windows(signals, 655, 327, pad_mode='edge')
    assert len(windows_edge) > 0

    # Zero padding - should work
    windows_zero, spans_zero = to_windows(signals, 655, 327, pad_mode='zero')
    assert len(windows_zero) > 0


def test_standardize_windows():
    """Test window standardization."""
    # Create windows with known statistics
    np.random.seed(42)
    windows = np.random.randn(10, 3, 655).astype(np.float32) * 10 + 5

    # Standardize
    windows_norm, (means, stds) = standardize_windows(windows)

    # Check shapes
    assert windows_norm.shape == windows.shape
    assert means.shape == (10, 3)
    assert stds.shape == (10, 3)

    # Check that normalized windows have mean ~0 and std ~1
    for i in range(10):
        for c in range(3):
            assert np.abs(windows_norm[i, c].mean()) < 1e-5, \
                f"Mean not close to 0: {windows_norm[i, c].mean()}"
            assert np.abs(windows_norm[i, c].std() - 1.0) < 0.01, \
                f"Std not close to 1: {windows_norm[i, c].std()}"


def test_standardize_invertibility():
    """Test that standardization is invertible."""
    np.random.seed(42)
    windows = np.random.randn(5, 3, 655).astype(np.float32)

    # Standardize
    windows_norm, (means, stds) = standardize_windows(windows)

    # Inverse
    windows_reconstructed = windows_norm * stds[:, :, np.newaxis] + means[:, :, np.newaxis]

    # Check reconstruction
    assert np.allclose(windows, windows_reconstructed, atol=1e-5), \
        "Standardization is not invertible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
