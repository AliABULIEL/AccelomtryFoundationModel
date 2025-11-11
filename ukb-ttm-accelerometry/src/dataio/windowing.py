"""
Windowing operations for accelerometry data.

Implements 8.192s windowing with configurable overlap and padding strategies.
"""
import math
from typing import Literal, Tuple

import numpy as np


def compute_window_params(
    fs: int,
    win_sec: float,
    hop_sec: float,
    rounding: Literal["floor", "nearest", "ceil"] = "floor"
) -> Tuple[int, int]:
    """
    Compute window and hop sizes in samples.

    Args:
        fs: Sampling frequency in Hz
        win_sec: Window duration in seconds
        hop_sec: Hop duration in seconds (for overlap)
        rounding: How to round fractional samples
            - "floor": Round down (default)
            - "nearest": Round to nearest integer
            - "ceil": Round up

    Returns:
        Tuple of (win_n, hop_n) in samples

    Examples:
        >>> compute_window_params(100, 8.192, 4.096, "floor")
        (819, 409)
        >>> compute_window_params(100, 8.192, 4.096, "nearest")
        (819, 410)
    """
    win_samples = fs * win_sec
    hop_samples = fs * hop_sec

    if rounding == "floor":
        win_n = int(win_samples)
        hop_n = int(hop_samples)
    elif rounding == "nearest":
        win_n = int(round(win_samples))
        hop_n = int(round(hop_samples))
    elif rounding == "ceil":
        win_n = int(math.ceil(win_samples))
        hop_n = int(math.ceil(hop_samples))
    else:
        raise ValueError(f"Invalid rounding mode: {rounding}. Use 'floor', 'nearest', or 'ceil'")

    return win_n, hop_n


def segment_stream(
    x: np.ndarray,
    win_n: int,
    hop_n: int,
    pad_mode: Literal["none", "reflect", "zero"] = "none"
) -> np.ndarray:
    """
    Create overlapping windows from time series data using strided views.

    Args:
        x: Input array of shape (T, C) where T is time steps, C is channels (typically 3 for x,y,z)
        win_n: Window size in samples
        hop_n: Hop size in samples (stride between windows)
        pad_mode: Padding strategy for edges
            - "none": No padding, last incomplete window discarded (default for training)
            - "reflect": Reflect padding at edges (for inference continuity)
            - "zero": Zero padding at edges

    Returns:
        Array of shape (N, C, win_n) where N is number of windows

    Raises:
        ValueError: If win_n < 64 (too small for meaningful windows)
        ValueError: If input is not 2-D
        AssertionError: If win_n is divisible by 7 (catches accidental config errors)

    Examples:
        >>> x = np.random.randn(1000, 3)
        >>> windows = segment_stream(x, 819, 409)
        >>> windows.shape
        (2, 3, 819)  # 2 windows with 50% overlap
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D input (T, C), got shape {x.shape}")

    if win_n < 64:
        raise ValueError(
            f"Window size {win_n} is too small (< 64 samples). "
            "This likely indicates a configuration error. "
            "For 100 Hz and 8.192s windows, win_n should be ~819."
        )

    # Catch accidental divisibility by 7 (common config mistake)
    assert win_n % 7 != 0, (
        f"Window size {win_n} is divisible by 7, which is unusual and may indicate "
        "a configuration error. Please verify your window parameters."
    )

    T, C = x.shape

    # Apply padding if requested
    if pad_mode != "none":
        x = _apply_padding(x, win_n, hop_n, pad_mode)
        T = x.shape[0]

    # Calculate number of windows that fit
    if T < win_n:
        raise ValueError(
            f"Input length {T} is shorter than window size {win_n}. "
            "Cannot create any windows."
        )

    n_windows = (T - win_n) // hop_n + 1

    # Create strided view for efficient windowing
    # Output shape: (n_windows, win_n, C)
    shape = (n_windows, win_n, C)
    strides = (x.strides[0] * hop_n, x.strides[0], x.strides[1])
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Transpose to (n_windows, C, win_n) for consistent channel-first format
    windows = windows.transpose(0, 2, 1)

    # Create a copy to avoid stride_tricks issues with mutability
    windows = windows.copy()

    return windows


def _apply_padding(
    x: np.ndarray,
    win_n: int,
    hop_n: int,
    pad_mode: Literal["reflect", "zero"]
) -> np.ndarray:
    """
    Apply padding to ensure complete coverage of input signal.

    Args:
        x: Input array of shape (T, C)
        win_n: Window size
        hop_n: Hop size
        pad_mode: Padding mode ("reflect" or "zero")

    Returns:
        Padded array
    """
    T = x.shape[0]

    # Calculate how much padding is needed to ensure last window fits
    remainder = (T - win_n) % hop_n
    if remainder != 0:
        pad_needed = hop_n - remainder

        if pad_mode == "reflect":
            # Reflect padding on both ends for smoother transitions
            pad_left = min(pad_needed // 2, T - 1)
            pad_right = pad_needed - pad_left
            x = np.pad(x, ((pad_left, pad_right), (0, 0)), mode='reflect')
        elif pad_mode == "zero":
            # Zero padding on right side only
            x = np.pad(x, ((0, pad_needed), (0, 0)), mode='constant', constant_values=0)

    return x


def compute_window_timestamps(
    timestamps: np.ndarray,
    hop_n: int,
    win_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute start and end timestamps for each window.

    Args:
        timestamps: Array of timestamps (length T)
        hop_n: Hop size in samples
        win_n: Window size in samples

    Returns:
        Tuple of (start_times, end_times) arrays of length N (number of windows)

    Examples:
        >>> timestamps = pd.date_range('2020-01-01', periods=1000, freq='10ms')
        >>> starts, ends = compute_window_timestamps(timestamps.values, 409, 819)
        >>> len(starts)
        2
    """
    n_windows = (len(timestamps) - win_n) // hop_n + 1

    start_indices = np.arange(n_windows) * hop_n
    end_indices = start_indices + win_n - 1  # Inclusive end index

    start_times = timestamps[start_indices]
    end_times = timestamps[end_indices]

    return start_times, end_times


def filter_windows_by_gaps(
    windows: np.ndarray,
    gap_flags: np.ndarray,
    hop_n: int,
    win_n: int,
    max_gap_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out windows with too many gap samples.

    Args:
        windows: Array of shape (N, C, win_n)
        gap_flags: Array of gap flags of shape (T,) where 1 indicates gap
        hop_n: Hop size used to create windows
        win_n: Window size
        max_gap_ratio: Maximum ratio of gap samples allowed (default 0.1 = 10%)

    Returns:
        Tuple of (filtered_windows, valid_mask) where valid_mask is boolean array
        indicating which windows were kept

    Examples:
        >>> windows = np.random.randn(10, 3, 819)
        >>> gap_flags = np.zeros(1000)
        >>> gap_flags[500:600] = 1  # Large gap
        >>> filtered, mask = filter_windows_by_gaps(windows, gap_flags, 409, 819, 0.1)
        >>> filtered.shape[0] <= windows.shape[0]
        True
    """
    n_windows = windows.shape[0]
    valid_mask = np.ones(n_windows, dtype=bool)

    for i in range(n_windows):
        start_idx = i * hop_n
        end_idx = start_idx + win_n

        # Count gap samples in this window
        window_gaps = gap_flags[start_idx:end_idx].sum()
        gap_ratio = window_gaps / win_n

        if gap_ratio > max_gap_ratio:
            valid_mask[i] = False

    filtered_windows = windows[valid_mask]

    return filtered_windows, valid_mask
