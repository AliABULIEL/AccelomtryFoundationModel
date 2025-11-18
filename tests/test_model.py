"""
Tests for TTM classifier model
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.ttm_classifier import (
    TTMAccelerometryClassifier,
    FocalLoss,
    create_class_weights,
)


class TestTTMClassifier:
    """Test TTM classifier functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return TTMAccelerometryClassifier(
            n_classes=4,
            n_channels=3,
            context_length=512,
            hidden_dim=256,
            dropout=0.3,
            freeze_encoder=True,
        )

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.n_classes == 4
        assert model.n_channels == 3
        assert model.context_length == 512

        # Check frozen encoder
        assert model.freeze_encoder == True

        # Count parameters
        stats = model.get_parameter_stats()
        assert stats['total'] > 0
        assert stats['trainable'] >= stats['classifier']

        print(f"✓ Model initialization: {stats['total']:,} total params, "
              f"{stats['trainable']:,} trainable")

    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 8
        time_steps = 820  # 8.192 seconds at 100Hz
        channels = 3

        # Create dummy input
        x = torch.randn(batch_size, time_steps, channels)

        # Forward pass
        output = model(x, return_embeddings=True)

        # Check output
        assert 'logits' in output
        assert 'embeddings' in output

        logits = output['logits']
        embeddings = output['embeddings']

        assert logits.shape == (batch_size, 4)
        assert embeddings.shape[0] == batch_size

        print(f"✓ Forward pass: input {x.shape} -> logits {logits.shape}")

    def test_gradient_flow(self, model):
        """Test gradient flow through model."""
        x = torch.randn(4, 820, 3)
        y = torch.randint(0, 4, (4,))

        # Forward pass
        output = model(x)
        logits = output['logits']

        # Loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()

        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        assert has_grad, "No gradients computed"

        print(f"✓ Gradient flow: loss={loss.item():.4f}")

    def test_freeze_unfreeze(self, model):
        """Test freezing and unfreezing encoder."""
        # Initially frozen
        model.freeze_encoder()
        frozen_params = sum(1 for p in model.ttm_model.parameters() if not p.requires_grad)
        assert frozen_params > 0

        # Unfreeze
        model.unfreeze_encoder()
        frozen_params = sum(1 for p in model.ttm_model.parameters() if not p.requires_grad)
        assert frozen_params == 0

        print(f"✓ Freeze/unfreeze encoder")

    def test_uncertainty_estimation(self, model):
        """Test Monte Carlo dropout uncertainty estimation."""
        x = torch.randn(4, 820, 3)

        # Predict with uncertainty
        mean_probs, uncertainty = model.predict_with_uncertainty(x, n_samples=10)

        assert mean_probs.shape == (4, 4)
        assert uncertainty.shape == (4, 4)
        assert (uncertainty >= 0).all()

        print(f"✓ Uncertainty estimation: mean uncertainty={uncertainty.mean().item():.4f}")

    def test_variable_sequence_length(self, model):
        """Test handling variable sequence lengths."""
        # Shorter sequence (should be padded)
        x_short = torch.randn(2, 256, 3)
        output_short = model(x_short)
        assert output_short['logits'].shape == (2, 4)

        # Longer sequence (should be truncated)
        x_long = torch.randn(2, 1024, 3)
        output_long = model(x_long)
        assert output_long['logits'].shape == (2, 4)

        print(f"✓ Variable sequence length")

    def test_batch_sizes(self, model):
        """Test different batch sizes."""
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 820, 3)
            output = model(x)
            assert output['logits'].shape == (batch_size, 4)

        print(f"✓ Variable batch sizes")


class TestFocalLoss:
    """Test Focal Loss implementation."""

    def test_focal_loss(self):
        """Test focal loss computation."""
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        # Basic focal loss
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

        print(f"✓ Focal loss: {loss.item():.4f}")

    def test_focal_loss_with_weights(self):
        """Test focal loss with class weights."""
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        # Class weights
        alpha = torch.tensor([1.0, 0.8, 0.6, 0.4])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        loss = loss_fn(logits, targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

        print(f"✓ Focal loss with weights: {loss.item():.4f}")


class TestClassWeights:
    """Test class weight computation."""

    def test_create_class_weights(self):
        """Test class weight creation."""
        class_counts = {0: 300, 1: 400, 2: 200, 3: 100}

        # Inverse frequency
        weights_inv = create_class_weights(class_counts, mode='inverse')
        assert len(weights_inv) == 4
        assert weights_inv[3] > weights_inv[0]  # Smaller class gets higher weight

        # Sqrt inverse
        weights_sqrt = create_class_weights(class_counts, mode='sqrt_inverse')
        assert len(weights_sqrt) == 4

        print(f"✓ Class weights: {weights_inv.numpy()}")


def test_memory_usage():
    """Test memory usage."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model = TTMAccelerometryClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Forward pass
    x = torch.randn(64, 820, 3).to(device)
    output = model(x)

    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1e9
        print(f"✓ Memory usage: {memory_gb:.2f} GB for batch_size=64")
        assert memory_gb < 5.0, f"Memory usage too high: {memory_gb:.2f} GB"
    else:
        print(f"✓ Memory usage: CPU mode")


if __name__ == "__main__":
    print("Testing TTM Classifier...")
    print("=" * 50)

    # Model tests
    test_cls = TestTTMClassifier()
    model = test_cls.model.__wrapped__(test_cls)  # Get model from fixture

    # Create model directly for testing
    model = TTMAccelerometryClassifier(
        n_classes=4,
        n_channels=3,
        context_length=512,
        hidden_dim=256,
        dropout=0.3,
        freeze_encoder=True,
    )

    test_cls.test_model_initialization(model)
    test_cls.test_forward_pass(model)
    test_cls.test_gradient_flow(model)
    test_cls.test_freeze_unfreeze(model)
    test_cls.test_uncertainty_estimation(model)
    test_cls.test_variable_sequence_length(model)
    test_cls.test_batch_sizes(model)

    # Loss tests
    loss_test = TestFocalLoss()
    loss_test.test_focal_loss()
    loss_test.test_focal_loss_with_weights()

    # Weight tests
    weight_test = TestClassWeights()
    weight_test.test_create_class_weights()

    # Memory test
    test_memory_usage()

    print("=" * 50)
    print("✓ All model tests passed!")
