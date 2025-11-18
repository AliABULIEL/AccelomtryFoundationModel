"""
TTM-Based Accelerometry Classifier
Adapts IBM's Tiny Time Mixer for activity classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

logger = logging.getLogger(__name__)


class TTMAccelerometryClassifier(nn.Module):
    """
    TTM-based classifier for accelerometry activity recognition.

    Architecture:
    - Pre-trained TTM encoder (frozen initially)
    - Classification head (Linear→ReLU→Dropout→Linear)
    - Optional LoRA adapters for parameter-efficient fine-tuning
    - Monte Carlo dropout for uncertainty estimation
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-timeseries-ttm-r2",
        n_classes: int = 4,  # Sleep, Sedentary, Light, MVPA
        n_channels: int = 3,  # X, Y, Z accelerometer
        context_length: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        freeze_encoder: bool = True,
    ):
        """
        Initialize TTM classifier.

        Args:
            model_name: HuggingFace model name
            n_classes: Number of activity classes
            n_channels: Number of input channels (3 for X,Y,Z)
            context_length: TTM context length
            hidden_dim: Hidden dimension for classification head
            dropout: Dropout probability
            use_lora: Whether to use LoRA adapters
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            freeze_encoder: Whether to freeze TTM encoder
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.context_length = context_length
        self.use_lora = use_lora
        self.freeze_encoder = freeze_encoder

        # Load pre-trained TTM
        logger.info(f"Loading TTM model: {model_name}")
        try:
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

            # Load config
            config = AutoConfig.from_pretrained(model_name)

            # Modify config for accelerometry
            config.num_input_channels = n_channels
            config.context_length = context_length

            # Load model
            self.ttm_model = TinyTimeMixerForPrediction.from_pretrained(
                model_name,
                config=config,
            )

            # Get encoder output dimension
            self.encoder_dim = config.d_model

        except ImportError:
            logger.warning("Could not import TinyTimeMixer, using dummy model")
            # Fallback for testing
            self.encoder_dim = 256
            self.ttm_model = nn.Sequential(
                nn.Linear(context_length * n_channels, self.encoder_dim),
                nn.ReLU(),
            )

        # Freeze encoder if requested
        if freeze_encoder:
            logger.info("Freezing TTM encoder")
            for param in self.ttm_model.parameters():
                param.requires_grad = False

        # Apply LoRA if requested
        if use_lora and not freeze_encoder:
            logger.info(f"Applying LoRA adapters (rank={lora_rank}, alpha={lora_alpha})")
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=dropout,
                target_modules=["query", "value"],  # Apply to attention layers
            )
            self.ttm_model = get_peft_model(self.ttm_model, lora_config)
            self.ttm_model.print_trainable_parameters()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # For uncertainty estimation
        self.dropout_layer = nn.Dropout(dropout)
        self.mc_dropout_enabled = False

        logger.info(f"Initialized TTM classifier: {self.count_parameters()} parameters")

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, time, channels)
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with 'logits' and optionally 'embeddings'
        """
        batch_size = x.shape[0]

        # Ensure correct input shape for TTM
        # TTM expects (batch, channels, time)
        if x.shape[-1] == self.n_channels:
            x = x.transpose(1, 2)  # (batch, time, channels) -> (batch, channels, time)

        # Pad or truncate to context_length
        if x.shape[2] < self.context_length:
            # Pad with zeros
            padding = self.context_length - x.shape[2]
            x = F.pad(x, (0, padding))
        elif x.shape[2] > self.context_length:
            # Truncate
            x = x[:, :, :self.context_length]

        # Extract features with TTM encoder
        try:
            # TTM forward pass
            ttm_output = self.ttm_model(x)

            # Extract embeddings (use prediction or hidden states)
            if hasattr(ttm_output, 'prediction'):
                embeddings = ttm_output.prediction.mean(dim=-1)  # Pool over time
            elif hasattr(ttm_output, 'last_hidden_state'):
                embeddings = ttm_output.last_hidden_state.mean(dim=1)  # Pool over time
            else:
                # Fallback: use output directly
                embeddings = ttm_output.reshape(batch_size, -1)[:, :self.encoder_dim]

        except Exception as e:
            logger.warning(f"TTM forward error: {e}, using fallback")
            # Fallback: simple embedding
            embeddings = x.reshape(batch_size, -1)
            embeddings = self.ttm_model(embeddings)

        # Apply Monte Carlo dropout if enabled
        if self.mc_dropout_enabled:
            embeddings = self.dropout_layer(embeddings)

        # Classification
        logits = self.classifier(embeddings)

        output = {'logits': logits}
        if return_embeddings:
            output['embeddings'] = embeddings

        return output

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo dropout.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean_probs: Mean predicted probabilities
            uncertainty: Uncertainty (std of probabilities)
        """
        self.mc_dropout_enabled = True
        self.eval()  # Keep dropout active via mc_dropout_enabled flag

        all_probs = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = self.forward(x)
                probs = F.softmax(output['logits'], dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs)  # (n_samples, batch, n_classes)

        mean_probs = all_probs.mean(dim=0)
        uncertainty = all_probs.std(dim=0)

        self.mc_dropout_enabled = False

        return mean_probs, uncertainty

    def freeze_encoder(self):
        """Freeze TTM encoder for linear probing."""
        logger.info("Freezing encoder")
        for param in self.ttm_model.parameters():
            param.requires_grad = False
        self.freeze_encoder = True

    def unfreeze_encoder(self):
        """Unfreeze TTM encoder for fine-tuning."""
        logger.info("Unfreezing encoder")
        for param in self.ttm_model.parameters():
            param.requires_grad = True
        self.freeze_encoder = False

    def enable_lora(self, rank: int = 8, alpha: int = 16):
        """
        Enable LoRA adapters for parameter-efficient fine-tuning.

        Args:
            rank: LoRA rank
            alpha: LoRA alpha
        """
        if self.use_lora:
            logger.info("LoRA already enabled")
            return

        logger.info(f"Enabling LoRA (rank={rank}, alpha={alpha})")

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )

        self.ttm_model = get_peft_model(self.ttm_model, lora_config)
        self.use_lora = True
        self.ttm_model.print_trainable_parameters()

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_parameter_stats(self) -> Dict[str, int]:
        """Get detailed parameter statistics."""
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)

        encoder_params = sum(p.numel() for p in self.ttm_model.parameters())
        encoder_trainable = sum(
            p.numel() for p in self.ttm_model.parameters() if p.requires_grad
        )

        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'encoder_total': encoder_params,
            'encoder_trainable': encoder_trainable,
            'classifier': classifier_params,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples and down-weights easy examples.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights (tensor of size n_classes)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits (batch, n_classes)
            targets: Ground truth labels (batch,)

        Returns:
            loss: Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_class_weights(
    class_counts: Dict[int, int],
    mode: str = 'inverse',
) -> torch.Tensor:
    """
    Create class weights for imbalanced dataset.

    Args:
        class_counts: Dict mapping class_id -> count
        mode: 'inverse' or 'sqrt_inverse'

    Returns:
        weights: Tensor of class weights
    """
    n_classes = len(class_counts)
    counts = np.array([class_counts.get(i, 1) for i in range(n_classes)])

    if mode == 'inverse':
        weights = 1.0 / counts
    elif mode == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize
    weights = weights / weights.sum() * n_classes

    return torch.FloatTensor(weights)
