"""
Unit tests for preprocessing operations.

Tests instance standardization, inverse operations, and numerical correctness.
"""
import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataio.preprocess import (
    instance_standardize,
    inverse_standardize,
    batch_instance_standardize,
    batch_inverse_standardize,
    robust_standardize,
    inverse_robust_standardize
)


class TestInstanceStandardize:
    """Tests for instance_standardize function."""

    def test_basic_standardization(self):
        """Test basic standardization produces zero mean and unit variance."""
        # Create batch of windows
        x = torch.randn(32, 3, 819) * 10 + 5  # Random data with mean~5, std~10

        x_norm, (mean, std) = instance_standardize(x)

        # Check shapes
        assert x_norm.shape == x.shape
        assert mean.shape == (32, 3, 1)
        assert std.shape == (32, 3, 1)

        # Check normalization: mean should be ~0, std should be ~1
        # Use larger tolerance due to finite samples
        assert torch.allclose(x_norm.mean(dim=-1), torch.zeros(32, 3), atol=1e-6)
        assert torch.allclose(x_norm.std(dim=-1), torch.ones(32, 3), atol=1e-2)

    def test_per_channel_normalization(self):
        """Test that normalization is per-channel."""
        # Create data with different statistics per channel
        x = torch.zeros(16, 3, 819)
        x[:, 0, :] = torch.randn(16, 819) * 1.0  # std=1
        x[:, 1, :] = torch.randn(16, 819) * 5.0  # std=5
        x[:, 2, :] = torch.randn(16, 819) * 10.0  # std=10

        x_norm, (mean, std) = instance_standardize(x)

        # All channels should have unit variance after normalization
        for c in range(3):
            assert torch.allclose(x_norm[:, c, :].std(dim=-1), torch.ones(16), atol=1e-2)

    def test_constant_signal(self):
        """Test handling of constant signals (zero variance)."""
        # Create constant signals
        x = torch.ones(16, 3, 819) * 5.0

        x_norm, (mean, std) = instance_standardize(x, eps=1e-5)

        # Should not raise error and should produce finite values
        assert torch.all(torch.isfinite(x_norm))

        # Mean should be captured
        assert torch.allclose(mean, torch.ones(16, 3, 1) * 5.0)

    def test_eps_prevents_division_by_zero(self):
        """Test that eps prevents division by zero."""
        x = torch.ones(16, 3, 819)

        # With eps, should work
        x_norm, _ = instance_standardize(x, eps=1e-5)
        assert torch.all(torch.isfinite(x_norm))

        # With very small eps, should still work
        x_norm, _ = instance_standardize(x, eps=1e-10)
        assert torch.all(torch.isfinite(x_norm))

    def test_incorrect_input_shape(self):
        """Test that incorrect input shape raises error."""
        x = torch.randn(32, 819)  # 2-D instead of 3-D

        with pytest.raises(ValueError, match="Expected 3-D input"):
            instance_standardize(x)


class TestInverseStandardize:
    """Tests for inverse_standardize function."""

    def test_perfect_reconstruction(self):
        """Test that inverse operation perfectly reconstructs original."""
        x_original = torch.randn(32, 3, 819) * 10 + 5

        x_norm, (mean, std) = instance_standardize(x_original)
        x_reconstructed = inverse_standardize(x_norm, mean, std)

        # Should reconstruct original within numerical precision
        assert torch.allclose(x_original, x_reconstructed, atol=1e-5)

    def test_multiple_rounds(self):
        """Test that multiple standardize-inverse rounds are consistent."""
        x = torch.randn(16, 3, 819)

        # Round 1
        x_norm1, (mean1, std1) = instance_standardize(x)
        x_recon1 = inverse_standardize(x_norm1, mean1, std1)

        # Round 2
        x_norm2, (mean2, std2) = instance_standardize(x_recon1)
        x_recon2 = inverse_standardize(x_norm2, mean2, std2)

        # Should be identical
        assert torch.allclose(x_recon1, x_recon2, atol=1e-5)

    def test_incorrect_input_shape(self):
        """Test that incorrect input shape raises error."""
        x = torch.randn(32, 819)
        mean = torch.randn(32, 1)
        std = torch.randn(32, 1)

        with pytest.raises(ValueError, match="Expected 3-D input"):
            inverse_standardize(x, mean, std)


class TestBatchInstanceStandardize:
    """Tests for NumPy version of instance standardization."""

    def test_numpy_standardization(self):
        """Test NumPy version produces correct results."""
        x = np.random.randn(32, 3, 819) * 10 + 5

        x_norm, (mean, std) = batch_instance_standardize(x)

        # Check shapes
        assert x_norm.shape == x.shape
        assert mean.shape == (32, 3, 1)
        assert std.shape == (32, 3, 1)

        # Check normalization
        assert np.allclose(x_norm.mean(axis=-1), 0, atol=1e-6)
        assert np.allclose(x_norm.std(axis=-1), 1, atol=1e-2)

    def test_numpy_reconstruction(self):
        """Test NumPy inverse reconstruction."""
        x_original = np.random.randn(32, 3, 819) * 10 + 5

        x_norm, (mean, std) = batch_instance_standardize(x_original)
        x_reconstructed = batch_inverse_standardize(x_norm, mean, std)

        assert np.allclose(x_original, x_reconstructed, atol=1e-5)

    def test_torch_numpy_consistency(self):
        """Test that PyTorch and NumPy versions give same results."""
        x_np = np.random.randn(16, 3, 819)
        x_torch = torch.from_numpy(x_np).float()

        # NumPy version
        x_norm_np, (mean_np, std_np) = batch_instance_standardize(x_np)

        # PyTorch version
        x_norm_torch, (mean_torch, std_torch) = instance_standardize(x_torch)

        # Results should match
        assert np.allclose(x_norm_np, x_norm_torch.numpy(), atol=1e-6)
        assert np.allclose(mean_np, mean_torch.numpy(), atol=1e-6)
        assert np.allclose(std_np, std_torch.numpy(), atol=1e-6)


class TestRobustStandardize:
    """Tests for robust standardization."""

    def test_robust_standardization(self):
        """Test robust standardization with median and IQR."""
        x = torch.randn(32, 3, 819)

        x_norm, (median, iqr) = robust_standardize(x)

        # Check shapes
        assert x_norm.shape == x.shape
        assert median.shape == (32, 3, 1)
        assert iqr.shape == (32, 3, 1)

        # Check that IQR is positive
        assert torch.all(iqr > 0)

    def test_robust_with_outliers(self):
        """Test that robust method handles outliers better."""
        # Create data with outliers
        x = torch.randn(16, 3, 819)

        # Add extreme outliers
        x[:, 0, 0] = 1000
        x[:, 0, 1] = -1000

        # Standard normalization
        x_norm_standard, _ = instance_standardize(x)

        # Robust normalization
        x_norm_robust, _ = robust_standardize(x)

        # Robust should be less affected by outliers
        # (bulk of data should be better normalized)
        bulk_indices = slice(10, 800)  # Exclude outliers

        robust_std = x_norm_robust[:, 0, bulk_indices].std()
        standard_std = x_norm_standard[:, 0, bulk_indices].std()

        # Robust should have std closer to 1 for bulk data
        assert abs(robust_std - 1.0) < abs(standard_std - 1.0)

    def test_inverse_robust(self):
        """Test inverse robust standardization."""
        x_original = torch.randn(16, 3, 819)

        x_norm, (median, iqr) = robust_standardize(x_original)
        x_reconstructed = inverse_robust_standardize(x_norm, median, iqr)

        assert torch.allclose(x_original, x_reconstructed, atol=1e-5)


class TestNumericalStability:
    """Test numerical stability across different scales."""

    def test_large_values(self):
        """Test standardization with large values."""
        x = torch.randn(16, 3, 819) * 1e6

        x_norm, (mean, std) = instance_standardize(x)

        # Should produce finite values
        assert torch.all(torch.isfinite(x_norm))
        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(std))

        # Should still normalize correctly
        assert torch.allclose(x_norm.mean(dim=-1), torch.zeros(16, 3), atol=1e-4)

    def test_small_values(self):
        """Test standardization with small values."""
        x = torch.randn(16, 3, 819) * 1e-6

        x_norm, (mean, std) = instance_standardize(x)

        # Should produce finite values
        assert torch.all(torch.isfinite(x_norm))

        # Reconstruction should work
        x_recon = inverse_standardize(x_norm, mean, std)
        assert torch.allclose(x, x_recon, atol=1e-9)

    def test_mixed_scales(self):
        """Test with different scales per channel."""
        x = torch.zeros(16, 3, 819)
        x[:, 0, :] = torch.randn(16, 819) * 1e6
        x[:, 1, :] = torch.randn(16, 819) * 1e0
        x[:, 2, :] = torch.randn(16, 819) * 1e-6

        x_norm, (mean, std) = instance_standardize(x)

        # All channels should normalize to unit variance
        for c in range(3):
            assert torch.allclose(x_norm[:, c, :].std(dim=-1), torch.ones(16), atol=1e-2)

        # Reconstruction should work
        x_recon = inverse_standardize(x_norm, mean, std)
        assert torch.allclose(x, x_recon, rtol=1e-5)


class TestBatchSizes:
    """Test with different batch sizes."""

    def test_single_sample(self):
        """Test with batch size 1."""
        x = torch.randn(1, 3, 819)

        x_norm, (mean, std) = instance_standardize(x)

        assert x_norm.shape == (1, 3, 819)
        assert torch.allclose(x_norm[0].mean(dim=-1), torch.zeros(3), atol=1e-6)

    def test_large_batch(self):
        """Test with large batch."""
        x = torch.randn(512, 3, 819)

        x_norm, (mean, std) = instance_standardize(x)

        assert x_norm.shape == (512, 3, 819)
        # Check random samples
        for i in [0, 100, 511]:
            assert torch.allclose(x_norm[i].mean(dim=-1), torch.zeros(3), atol=1e-6)


class TestReproducibility:
    """Test reproducibility of operations."""

    def test_deterministic(self):
        """Test that operations are deterministic."""
        torch.manual_seed(42)
        x1 = torch.randn(32, 3, 819)

        torch.manual_seed(42)
        x2 = torch.randn(32, 3, 819)

        x_norm1, (mean1, std1) = instance_standardize(x1)
        x_norm2, (mean2, std2) = instance_standardize(x2)

        assert torch.allclose(x_norm1, x_norm2)
        assert torch.allclose(mean1, mean2)
        assert torch.allclose(std1, std2)


class TestGradientFlow:
    """Test that operations allow gradient flow."""

    def test_forward_backward(self):
        """Test that gradients flow through standardization."""
        x = torch.randn(16, 3, 819, requires_grad=True)

        x_norm, (mean, std) = instance_standardize(x)

        # Compute loss
        loss = x_norm.sum()
        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_inverse_backward(self):
        """Test gradients through inverse operation."""
        x = torch.randn(16, 3, 819)

        x_norm, (mean, std) = instance_standardize(x)

        # Make normalized data require gradients
        x_norm_grad = x_norm.detach().requires_grad_(True)

        x_recon = inverse_standardize(x_norm_grad, mean, std)

        loss = x_recon.sum()
        loss.backward()

        assert x_norm_grad.grad is not None
        assert torch.all(torch.isfinite(x_norm_grad.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
