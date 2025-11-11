"""
Non-wear detection algorithms for accelerometry data.

Implements:
- Choi et al. (2011) algorithm
- Troiano et al. (2008) algorithm
- Outlier clipping
- High-pass filtering
- Window-level non-wear marking
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def clip_outliers(
    acc: np.ndarray,
    threshold_g: float = 4.0
) -> np.ndarray:
    """
    Clip acceleration outliers.

    WARNING: Assumes input is in units of g (not m/s²).
    If data is in m/s², divide by 9.81 first.

    Args:
        acc: Acceleration array of shape (T, 3) or (T,) in units of g
        threshold_g: Clipping threshold in g (default: 4.0g)

    Returns:
        Clipped acceleration array
    """
    return np.clip(acc, -threshold_g, threshold_g)


def highpass_filter(
    acc: np.ndarray,
    fs: int = 80,
    cutoff: float = 0.25,
    order: int = 4
) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to remove drift.

    Args:
        acc: Acceleration array of shape (T, 3) or (T, C)
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz (default: 0.25 Hz)
        order: Filter order (default: 4)

    Returns:
        Filtered acceleration array
    """
    # Design Butterworth filter
    sos = signal.butter(order, cutoff, btype='highpass', fs=fs, output='sos')

    # Apply filter to each channel
    if acc.ndim == 1:
        return signal.sosfiltfilt(sos, acc)
    elif acc.ndim == 2:
        filtered = np.zeros_like(acc)
        for i in range(acc.shape[1]):
            filtered[:, i] = signal.sosfiltfilt(sos, acc[:, i])
        return filtered
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {acc.shape}")


@jit(nopython=True if HAS_NUMBA else False)
def _choi_core(
    vm: np.ndarray,
    fs: int,
    min_period_min: int,
    spike_tolerance: int,
    min_spike_separation_min: int,
    activity_threshold: float
) -> np.ndarray:
    """
    Core Choi algorithm implementation.

    Args:
        vm: Vector magnitude (T,)
        fs: Sampling frequency
        min_period_min: Minimum non-wear period in minutes
        spike_tolerance: Number of allowed spikes per window
        min_spike_separation_min: Minimum separation between spikes in minutes
        activity_threshold: Activity threshold for zero counts

    Returns:
        Boolean mask where True = non-wear
    """
    T = len(vm)
    nonwear_mask = np.zeros(T, dtype=np.bool_)

    # Convert minutes to samples
    min_period_samples = min_period_min * 60 * fs
    min_spike_sep_samples = min_spike_separation_min * 60 * fs

    # Find zero-count samples
    zero_counts = vm <= activity_threshold

    i = 0
    while i < T:
        if zero_counts[i]:
            # Start of potential non-wear period
            period_start = i
            period_end = i

            # Extend period while zero counts continue
            j = i
            spike_count = 0
            last_spike_idx = -min_spike_sep_samples - 1

            while j < T:
                if zero_counts[j]:
                    period_end = j
                    j += 1
                else:
                    # Spike detected
                    if j - last_spike_idx >= min_spike_sep_samples:
                        spike_count += 1
                        last_spike_idx = j

                    if spike_count > spike_tolerance:
                        # Too many spikes, end period
                        break

                    # Skip this spike (up to 2 minutes of activity allowed)
                    spike_window = min(2 * 60 * fs, T - j)
                    j += spike_window

            # Check if period is long enough
            period_length = period_end - period_start + 1
            if period_length >= min_period_samples:
                nonwear_mask[period_start:period_end+1] = True

            i = period_end + 1
        else:
            i += 1

    return nonwear_mask


def choi_algorithm(
    acc: np.ndarray,
    fs: int = 80,
    min_period: int = 90,
    spike_tolerance: int = 2,
    min_spike_separation: int = 30,
    activity_threshold: float = 0.01
) -> np.ndarray:
    """
    Choi et al. (2011) non-wear detection algorithm.

    Reference:
    Choi et al. (2011). "Assessment of wear/nonwear time classification
    algorithms for triaxial accelerometer." Med Sci Sports Exerc 43(2):357-364.

    Args:
        acc: Acceleration array of shape (T, 3) in any units
        fs: Sampling frequency in Hz (default: 80)
        min_period: Minimum non-wear period in minutes (default: 90)
        spike_tolerance: Number of allowed spikes per window (default: 2)
        min_spike_separation: Minimum separation between spikes in minutes (default: 30)
        activity_threshold: Activity threshold (default: 0.01)

    Returns:
        Boolean mask of shape (T,) where True = non-wear
    """
    if acc.ndim == 1:
        vm = np.abs(acc)
    elif acc.ndim == 2:
        # Compute vector magnitude
        vm = np.sqrt(np.sum(acc**2, axis=1))
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {acc.shape}")

    return _choi_core(
        vm,
        fs,
        min_period,
        spike_tolerance,
        min_spike_separation,
        activity_threshold
    )


@jit(nopython=True if HAS_NUMBA else False)
def _troiano_core(
    vm: np.ndarray,
    fs: int,
    min_period_min: int,
    activity_threshold: float
) -> np.ndarray:
    """
    Core Troiano algorithm implementation.

    Args:
        vm: Vector magnitude (T,)
        fs: Sampling frequency
        min_period_min: Minimum non-wear period in minutes
        activity_threshold: Activity threshold for zero counts

    Returns:
        Boolean mask where True = non-wear
    """
    T = len(vm)
    nonwear_mask = np.zeros(T, dtype=np.bool_)

    # Convert minutes to samples
    min_period_samples = min_period_min * 60 * fs

    # Find zero-count samples
    zero_counts = vm <= activity_threshold

    # Find consecutive runs of zeros
    i = 0
    while i < T:
        if zero_counts[i]:
            # Start of potential non-wear period
            period_start = i
            period_end = i

            # Extend period while zero counts continue
            while period_end < T and zero_counts[period_end]:
                period_end += 1

            period_end -= 1  # Adjust to last zero index

            # Check if period is long enough
            period_length = period_end - period_start + 1
            if period_length >= min_period_samples:
                nonwear_mask[period_start:period_end+1] = True

            i = period_end + 1
        else:
            i += 1

    return nonwear_mask


def troiano_algorithm(
    acc: np.ndarray,
    fs: int = 80,
    min_period: int = 60,
    activity_threshold: float = 0.01
) -> np.ndarray:
    """
    Troiano et al. (2008) non-wear detection algorithm.

    Simpler than Choi - detects consecutive periods of near-zero activity.

    Reference:
    Troiano et al. (2008). "Physical activity in the United States measured
    by accelerometer." Med Sci Sports Exerc 40(1):181-188.

    Args:
        acc: Acceleration array of shape (T, 3) in any units
        fs: Sampling frequency in Hz (default: 80)
        min_period: Minimum non-wear period in minutes (default: 60)
        activity_threshold: Activity threshold (default: 0.01)

    Returns:
        Boolean mask of shape (T,) where True = non-wear
    """
    if acc.ndim == 1:
        vm = np.abs(acc)
    elif acc.ndim == 2:
        # Compute vector magnitude
        vm = np.sqrt(np.sum(acc**2, axis=1))
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {acc.shape}")

    return _troiano_core(
        vm,
        fs,
        min_period,
        activity_threshold
    )


def mark_nonwear_windows_from_spans(
    spans: np.ndarray,
    nonwear_mask: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Mark windows as invalid if they contain too much non-wear time.

    Args:
        spans: Array of shape (W, 2) with [start, end) indices for each window
        nonwear_mask: Boolean array of shape (T,) where True = non-wear
        threshold: Maximum allowed fraction of non-wear (default: 0.5 = 50%)

    Returns:
        Boolean mask of shape (W,) where True = invalid window (too much non-wear)
    """
    W = spans.shape[0]
    invalid_windows = np.zeros(W, dtype=bool)

    for i in range(W):
        start, end = spans[i]

        # Ensure valid indices
        start = max(0, start)
        end = min(len(nonwear_mask), end)

        if start >= end:
            invalid_windows[i] = True
            continue

        # Calculate fraction of non-wear in this window
        window_nonwear = nonwear_mask[start:end]
        nonwear_fraction = window_nonwear.sum() / len(window_nonwear)

        if nonwear_fraction > threshold:
            invalid_windows[i] = True

    return invalid_windows


def compute_wear_time(
    nonwear_mask: np.ndarray,
    fs: int = 80
) -> Tuple[float, float]:
    """
    Compute total wear and non-wear time.

    Args:
        nonwear_mask: Boolean array where True = non-wear
        fs: Sampling frequency in Hz

    Returns:
        (wear_hours, nonwear_hours) tuple
    """
    total_samples = len(nonwear_mask)
    nonwear_samples = nonwear_mask.sum()
    wear_samples = total_samples - nonwear_samples

    wear_hours = wear_samples / (fs * 3600)
    nonwear_hours = nonwear_samples / (fs * 3600)

    return wear_hours, nonwear_hours


def preprocess_for_nonwear(
    acc: np.ndarray,
    fs: int = 80,
    clip_threshold_g: float = 4.0,
    apply_highpass: bool = True,
    highpass_cutoff: float = 0.25
) -> np.ndarray:
    """
    Preprocessing pipeline for non-wear detection.

    Args:
        acc: Acceleration array of shape (T, 3) or (T,)
        fs: Sampling frequency in Hz
        clip_threshold_g: Outlier clipping threshold in g
        apply_highpass: Whether to apply high-pass filter
        highpass_cutoff: High-pass cutoff frequency in Hz

    Returns:
        Preprocessed acceleration array
    """
    # Clip outliers
    acc_clipped = clip_outliers(acc, clip_threshold_g)

    # Apply high-pass filter if requested
    if apply_highpass:
        acc_filtered = highpass_filter(acc_clipped, fs, highpass_cutoff)
    else:
        acc_filtered = acc_clipped

    return acc_filtered
