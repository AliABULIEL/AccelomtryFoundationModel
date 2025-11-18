"""
Self-Supervised Learning (SSL) Augmentations for Accelerometry
Based on Yuan et al. 2024 multi-task SSL framework
"""

import numpy as np
import torch
from typing import Tuple, Optional
from scipy import interpolate
import logging

logger = logging.getLogger(__name__)


class SSLAugmentations:
    """
    Multi-task SSL augmentations for accelerometry data.

    Implements:
    1. Arrow of Time (AOT): Temporal direction prediction
    2. Permutation: Detect shuffled segments
    3. Time Warping: Smooth temporal distortion (15-30%)
    """

    def __init__(
        self,
        time_warp_range: Tuple[float, float] = (0.85, 1.15),
        n_permutation_segments: int = 4,
        p_augment: float = 0.5,
    ):
        """
        Initialize SSL augmentations.

        Args:
            time_warp_range: Range for time warping factor (0.85-1.15 = ±15%)
            n_permutation_segments: Number of segments for permutation
            p_augment: Probability of applying each augmentation
        """
        self.time_warp_range = time_warp_range
        self.n_permutation_segments = n_permutation_segments
        self.p_augment = p_augment

    def arrow_of_time(
        self,
        x: np.ndarray,
        apply: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Arrow of Time augmentation: randomly reverse time direction.

        Args:
            x: Input array of shape (time, channels)
            apply: Whether to apply augmentation

        Returns:
            augmented: Augmented array
            label: 0 = forward, 1 = reversed
        """
        if not apply or np.random.rand() > self.p_augment:
            return x, 0

        if np.random.rand() < 0.5:
            # Reverse time
            return x[::-1].copy(), 1
        else:
            # Keep forward
            return x, 0

    def permutation(
        self,
        x: np.ndarray,
        apply: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Permutation augmentation: shuffle temporal segments.

        Args:
            x: Input array of shape (time, channels)
            apply: Whether to apply augmentation

        Returns:
            augmented: Augmented array
            label: 0 = original order, 1 = permuted
        """
        if not apply or np.random.rand() > self.p_augment:
            return x, 0

        if np.random.rand() < 0.5:
            # Permute segments
            T = len(x)
            segment_size = T // self.n_permutation_segments

            # Create segments
            segments = []
            for i in range(self.n_permutation_segments):
                start = i * segment_size
                end = start + segment_size if i < self.n_permutation_segments - 1 else T
                segments.append(x[start:end])

            # Shuffle segments
            np.random.shuffle(segments)

            # Concatenate
            permuted = np.concatenate(segments, axis=0)
            return permuted, 1
        else:
            # Keep original
            return x, 0

    def time_warping(
        self,
        x: np.ndarray,
        apply: bool = True
    ) -> np.ndarray:
        """
        Time warping: smooth temporal distortion.

        Uses cubic spline interpolation to warp time axis smoothly.

        Args:
            x: Input array of shape (time, channels)
            apply: Whether to apply augmentation

        Returns:
            warped: Time-warped array of same shape
        """
        if not apply or np.random.rand() > self.p_augment:
            return x

        T, C = x.shape

        # Random warping factor
        warp_factor = np.random.uniform(*self.time_warp_range)

        # Create smooth warping curve using cubic spline
        n_knots = 5  # Number of control points
        knot_indices = np.linspace(0, T-1, n_knots)
        knot_values = knot_indices.copy()

        # Add random perturbation to middle knots
        for i in range(1, n_knots-1):
            max_shift = T * (warp_factor - 1.0) / (n_knots - 2)
            knot_values[i] += np.random.uniform(-max_shift, max_shift)

        # Ensure monotonicity
        knot_values = np.sort(knot_values)
        knot_values[0] = 0
        knot_values[-1] = T - 1

        # Create cubic spline
        spline = interpolate.CubicSpline(knot_indices, knot_values)

        # Warp time axis
        old_time = np.arange(T)
        new_time = spline(old_time)
        new_time = np.clip(new_time, 0, T-1)

        # Interpolate each channel
        warped = np.zeros_like(x)
        for c in range(C):
            interpolator = interpolate.interp1d(
                old_time,
                x[:, c],
                kind='cubic',
                fill_value='extrapolate'
            )
            warped[:, c] = interpolator(new_time)

        return warped

    def apply_all(
        self,
        x: np.ndarray,
        return_labels: bool = True
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Apply all augmentations with random selection.

        Args:
            x: Input array of shape (time, channels)
            return_labels: Whether to return SSL task labels

        Returns:
            augmented: Augmented array
            labels: Dict with SSL task labels (if return_labels=True)
        """
        # Apply time warping (continuous transformation)
        x_aug = self.time_warping(x, apply=True)

        # Apply arrow of time
        x_aug, aot_label = self.arrow_of_time(x_aug, apply=True)

        # Apply permutation
        x_aug, perm_label = self.permutation(x_aug, apply=True)

        if return_labels:
            labels = {
                'arrow_of_time': aot_label,
                'permutation': perm_label,
            }
            return x_aug, labels
        else:
            return x_aug, None


class ColabAugmentations:
    """
    Colab-optimized augmentations with batched operations.
    """

    @staticmethod
    def add_gaussian_noise(
        x: torch.Tensor,
        noise_std: float = 0.01
    ) -> torch.Tensor:
        """Add Gaussian noise for robustness."""
        noise = torch.randn_like(x) * noise_std
        return x + noise

    @staticmethod
    def channel_dropout(
        x: torch.Tensor,
        p_drop: float = 0.1
    ) -> torch.Tensor:
        """Randomly drop entire channels (simulate sensor failure)."""
        if np.random.rand() < p_drop:
            # Randomly select channel to drop
            channel_idx = np.random.randint(x.shape[-1])
            x_aug = x.clone()
            x_aug[..., channel_idx] = 0
            return x_aug
        return x

    @staticmethod
    def rotation_augmentation(
        x: torch.Tensor,
        max_angle: float = 15.0
    ) -> torch.Tensor:
        """
        Random 3D rotation augmentation for accelerometry.
        Simulates different device orientations.

        Args:
            x: Input tensor of shape (..., 3) with X, Y, Z channels
            max_angle: Maximum rotation angle in degrees

        Returns:
            Rotated tensor
        """
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.deg2rad(angle)

        # Random rotation axis
        axis = np.random.choice(['x', 'y', 'z'])

        # Rotation matrices
        if axis == 'x':
            R = torch.tensor([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ], dtype=x.dtype, device=x.device)
        elif axis == 'y':
            R = torch.tensor([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ], dtype=x.dtype, device=x.device)
        else:  # z
            R = torch.tensor([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ], dtype=x.dtype, device=x.device)

        # Apply rotation
        return torch.matmul(x, R.T)

    @staticmethod
    def mixup(
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup augmentation for time series.

        Args:
            x1, x2: Input tensors
            y1, y2: Labels
            alpha: Mixup parameter

        Returns:
            Mixed input and labels
        """
        lam = np.random.beta(alpha, alpha)
        x_mixed = lam * x1 + (1 - lam) * x2

        # For classification, return both labels with mixing ratio
        # Model should compute: lam * loss(y1) + (1-lam) * loss(y2)
        y_mixed = (y1, y2, lam)

        return x_mixed, y_mixed


class TTMDataAugmentor:
    """
    Integrated augmentation pipeline for TTM training.
    Combines SSL augmentations with standard augmentations.
    """

    def __init__(
        self,
        use_ssl: bool = True,
        use_standard: bool = True,
        ssl_prob: float = 0.5,
        aug_prob: float = 0.5,
    ):
        """
        Initialize augmentor.

        Args:
            use_ssl: Use SSL augmentations
            use_standard: Use standard augmentations
            ssl_prob: Probability of SSL augmentations
            aug_prob: Probability of standard augmentations
        """
        self.use_ssl = use_ssl
        self.use_standard = use_standard
        self.ssl_prob = ssl_prob
        self.aug_prob = aug_prob

        if use_ssl:
            self.ssl_aug = SSLAugmentations()

    def __call__(
        self,
        x: np.ndarray,
        return_ssl_labels: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Apply augmentation pipeline.

        Args:
            x: Input array of shape (time, channels)
            return_ssl_labels: Return SSL task labels

        Returns:
            augmented: Augmented array
            ssl_labels: Optional SSL labels
        """
        ssl_labels = None

        # SSL augmentations
        if self.use_ssl and np.random.rand() < self.ssl_prob:
            x, ssl_labels = self.ssl_aug.apply_all(x, return_labels=return_ssl_labels)

        # Standard augmentations (applied to tensor later in training)
        # These are applied in the training loop for efficiency

        return x, ssl_labels
