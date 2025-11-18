# FILE: ukb_ttm_accel/data/windowing.py

"""
Accelerometry data windowing and preprocessing functions.

Handles:
- Loading and resampling accelerometry data from various sources
- Calibration and filtering
- Window extraction with configurable overlap
- Normalization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Literal
from datetime import datetime


def resample_and_calibrate(
    file_path: str,
    source_type: Literal["cwa", "csv", "parquet"] = "cwa",
    target_sampling_rate_hz: int = 50,
    lowpass_hz: Optional[float] = 20.0,
    calibrate: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load, resample, and calibrate accelerometry data.

    Supports multiple input formats:
    - .cwa files (Axivity binary format) via actipy
    - .csv files with columns: time, x, y, z
    - .parquet files with columns: time, x, y, z

    Args:
        file_path: Path to accelerometry file
        source_type: Type of source file ("cwa", "csv", or "parquet")
        target_sampling_rate_hz: Target sampling rate after resampling
        lowpass_hz: Low-pass filter cutoff frequency (None to skip filtering)
        calibrate: Whether to apply gravity calibration (only for cwa)
        verbose: Whether to print processing information

    Returns:
        DataFrame with columns: time (index), x, y, z
        Units: g (gravitational acceleration)

    Example:
        >>> df = resample_and_calibrate("participant_123.cwa", target_sampling_rate_hz=50)
        >>> print(df.shape)
        (604800, 3)  # 7 days * 24 hours * 3600 sec * 50 Hz / 1
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if verbose:
        print(f"Loading {source_type} file: {file_path.name}")

    if source_type == "cwa":
        # Use actipy to read and process .cwa files
        try:
            import actipy
        except ImportError:
            raise ImportError(
                "actipy is required for .cwa files. "
                "Install with: pip install actipy"
            )

        # Read device with actipy (handles calibration and filtering)
        data, info = actipy.read_device(
            str(file_path),
            lowpass_hz=lowpass_hz,
            calibrate_gravity=calibrate,
            detect_nonwear=False,  # We'll handle this separately if needed
            resample_hz=target_sampling_rate_hz,
        )

        # actipy returns a DataFrame with time index and x, y, z columns
        df = data[['x', 'y', 'z']].copy()

        if verbose:
            print(f"  Loaded {len(df):,} samples")
            print(f"  Duration: {info.get('Duration', 'unknown')}")

    elif source_type == "csv":
        # Load CSV file
        df = pd.read_csv(file_path)

        # Ensure required columns exist
        required_cols = ['time', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Parse time column
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # Resample to target frequency
        df = _resample_dataframe(df, target_sampling_rate_hz)

        if verbose:
            print(f"  Loaded {len(df):,} samples from CSV")

    elif source_type == "parquet":
        # Load Parquet file
        df = pd.read_parquet(file_path)

        # Ensure required columns exist
        required_cols = ['time', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Parquet must contain columns: {required_cols}")

        # Parse time column if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        df.set_index('time', inplace=True)

        # Resample to target frequency
        df = _resample_dataframe(df, target_sampling_rate_hz)

        if verbose:
            print(f"  Loaded {len(df):,} samples from Parquet")

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    return df


def _resample_dataframe(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """
    Resample time-indexed DataFrame to target frequency.

    Args:
        df: DataFrame with datetime index and x, y, z columns
        target_hz: Target sampling rate in Hz

    Returns:
        Resampled DataFrame
    """
    # Calculate resample period (e.g., 50 Hz -> '20ms')
    period_ms = int(1000 / target_hz)
    resample_rule = f"{period_ms}ms"

    # Resample using linear interpolation
    df_resampled = df.resample(resample_rule).mean().interpolate(method='linear')

    # Drop any NaN rows (beginning/end of time series)
    df_resampled.dropna(inplace=True)

    return df_resampled


def create_windows(
    accel_df: pd.DataFrame,
    window_seconds: int,
    sampling_rate_hz: int,
    overlap_fraction: float = 0.0,
    extract_time_of_day: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create fixed-length windows from continuous accelerometry data.

    Args:
        accel_df: DataFrame with time index and x, y, z columns
        window_seconds: Window length in seconds
        sampling_rate_hz: Sampling rate in Hz
        overlap_fraction: Fraction of overlap between consecutive windows (0.0 to 0.99)
        extract_time_of_day: Whether to extract time-of-day features

    Returns:
        Tuple of:
        - windows: np.ndarray of shape [num_windows, window_length, 3]
                  where 3 channels are [x, y, z]
        - time_of_day: np.ndarray of shape [num_windows] with values in [0, 1]
                      representing fraction of day (0 = midnight, 0.5 = noon)
                      Returns None if extract_time_of_day=False

    Example:
        >>> df = pd.DataFrame(...)  # 7 days at 50 Hz
        >>> windows, tod = create_windows(df, window_seconds=10, sampling_rate_hz=50)
        >>> print(windows.shape)
        (60480, 500, 3)  # 7 days * 24 hours * 3600 / 10 seconds
    """
    window_length = window_seconds * sampling_rate_hz
    stride = int(window_length * (1 - overlap_fraction))

    if stride < 1:
        stride = 1

    # Extract accelerometry values (x, y, z)
    accel_values = accel_df[['x', 'y', 'z']].values

    # Create sliding windows
    num_windows = (len(accel_values) - window_length) // stride + 1
    windows = np.zeros((num_windows, window_length, 3), dtype=np.float32)

    time_of_day_values = None
    if extract_time_of_day:
        time_of_day_values = np.zeros(num_windows, dtype=np.float32)

    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_length

        windows[i] = accel_values[start_idx:end_idx]

        if extract_time_of_day:
            # Get timestamp of window center
            center_idx = start_idx + window_length // 2
            timestamp = accel_df.index[center_idx]

            # Convert to time of day (0 to 1)
            seconds_since_midnight = (
                timestamp.hour * 3600 +
                timestamp.minute * 60 +
                timestamp.second
            )
            time_of_day_values[i] = seconds_since_midnight / 86400.0

    return windows, time_of_day_values


def zscore_normalize_windows(
    windows: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Apply z-score normalization per window and per channel.

    Each window's each channel is normalized to mean=0, std=1.

    Args:
        windows: np.ndarray of shape [num_windows, window_length, num_channels]
        epsilon: Small constant to avoid division by zero

    Returns:
        Normalized windows with same shape

    Example:
        >>> windows = np.random.randn(100, 500, 3)
        >>> normalized = zscore_normalize_windows(windows)
        >>> print(normalized[0, :, 0].mean(), normalized[0, :, 0].std())
        0.0 1.0
    """
    # Compute mean and std per window and per channel
    # Shape: [num_windows, 1, num_channels]
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)

    # Normalize
    normalized = (windows - mean) / (std + epsilon)

    return normalized.astype(np.float32)


def compute_magnitude(windows: np.ndarray) -> np.ndarray:
    """
    Compute magnitude (Euclidean norm) of 3-axis accelerometry.

    Args:
        windows: np.ndarray of shape [num_windows, window_length, 3]

    Returns:
        Magnitude array of shape [num_windows, window_length, 1]

    Example:
        >>> windows = np.random.randn(100, 500, 3)
        >>> magnitude = compute_magnitude(windows)
        >>> print(magnitude.shape)
        (100, 500, 1)
    """
    magnitude = np.sqrt(np.sum(windows ** 2, axis=-1, keepdims=True))
    return magnitude.astype(np.float32)
