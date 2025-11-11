"""
UK Biobank Axivity .cwa reader with robust handling of real-world data issues.

Handles:
- Daylight-saving time gaps (forward-fill timestamps, not signals)
- Non-monotonic clocks
- Duplicate packets
- Missing samples with gap detection
"""
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def read_cwa_to_segments(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read Axivity .cwa file via accelerometer package (or fallback to .csv.gz/.npy).

    Returns a tidy DataFrame with columns: ["participant_id", "timestamp", "x", "y", "z"]
    in UTC timezone.

    Handles:
    - Daylight-saving gaps: forward-fill timestamp only, not signals
    - Non-monotonic clocks: sort by timestamp and warn
    - Duplicate packets: drop with warning

    Args:
        path: Path to .cwa, .csv.gz, or .npy file

    Returns:
        pd.DataFrame with columns ["participant_id", "timestamp", "x", "y", "z"]
        timestamp is UTC datetime

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If file format is not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Extract participant ID from filename (assuming format like 1234567.cwa)
    participant_id = path.stem.split('.')[0]

    # Read based on file extension
    if path.suffix == '.cwa':
        df = _read_cwa_file(path)
    elif path.suffix == '.gz' and path.name.endswith('.csv.gz'):
        df = _read_csv_gz(path)
    elif path.suffix == '.npy':
        df = _read_npy(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Ensure required columns exist
    required_cols = ['timestamp', 'x', 'y', 'z']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert timestamp to UTC datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    elif df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    # Handle non-monotonic timestamps
    if not df['timestamp'].is_monotonic_increasing:
        warnings.warn(
            f"Non-monotonic timestamps detected in {path}. Sorting by timestamp.",
            UserWarning
        )
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Handle duplicate timestamps
    duplicates = df['timestamp'].duplicated()
    if duplicates.any():
        n_dup = duplicates.sum()
        warnings.warn(
            f"Dropping {n_dup} duplicate timestamp packets in {path}",
            UserWarning
        )
        df = df[~duplicates].reset_index(drop=True)

    # Add participant ID
    df.insert(0, 'participant_id', participant_id)

    # Keep only required columns in order
    df = df[['participant_id', 'timestamp', 'x', 'y', 'z']]

    return df


def _read_cwa_file(path: Path) -> pd.DataFrame:
    """Read .cwa file using accelerometer package."""
    try:
        import accelerometer
    except ImportError:
        raise ImportError(
            "accelerometer package required for .cwa files. "
            "Install with: pip install accelerometer"
        )

    # Read using accelerometer package
    # The read_device function returns a tuple: (data, info)
    data, info = accelerometer.read_device(
        str(path),
        lowpass_hz=None,  # No filtering at read stage
        calibrate_gravity=True,
        detect_nonwear=False,
        resample_hz=None,  # Will resample separately
    )

    # data is a DataFrame with time index and x, y, z columns
    df = pd.DataFrame({
        'timestamp': data.index,
        'x': data['x'].values,
        'y': data['y'].values,
        'z': data['z'].values,
    })

    return df


def _read_csv_gz(path: Path) -> pd.DataFrame:
    """Read compressed CSV file."""
    df = pd.read_csv(
        path,
        compression='gzip',
        parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(path, nrows=0).columns else None
    )
    return df


def _read_npy(path: Path) -> pd.DataFrame:
    """
    Read .npy file.

    Expected format: array with shape (N, 4) where columns are [timestamp, x, y, z]
    or dict with keys 'timestamp', 'x', 'y', 'z'
    """
    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray):
        if data.shape[1] == 4:
            df = pd.DataFrame(data, columns=['timestamp', 'x', 'y', 'z'])
        else:
            raise ValueError(f"Expected .npy array with 4 columns, got {data.shape[1]}")
    elif isinstance(data, dict) or isinstance(data.item(), dict):
        # Handle pickled dict
        if isinstance(data, np.ndarray):
            data = data.item()
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported .npy format: {type(data)}")

    return df


def resample_to_fs(df: pd.DataFrame, fs: int = 100) -> pd.DataFrame:
    """
    Enforce uniform sampling rate on exact time boundaries.

    For 100 Hz: samples at exact 10ms boundaries (0ms, 10ms, 20ms, ...)

    Missing samples:
    - Linear interpolation for gaps up to 250ms
    - Mark with is_gap=1 for longer gaps (used later for exclusion)

    Args:
        df: DataFrame with columns ["participant_id", "timestamp", "x", "y", "z"]
        fs: Target sampling frequency in Hz (default: 100)

    Returns:
        pd.DataFrame with uniform sampling at fs Hz, with additional column "is_gap"
        indicating samples that were synthesized from large gaps (>250ms)
    """
    if df.empty:
        warnings.warn("Empty DataFrame provided to resample_to_fs", UserWarning)
        return df.assign(is_gap=0)

    # Store participant_id
    participant_id = df['participant_id'].iloc[0]

    # Set timestamp as index for resampling
    df = df.set_index('timestamp')

    # Create uniform time grid with exact period boundaries
    start_time = df.index[0].floor(f'{1000//fs}ms')  # Floor to nearest sample boundary
    end_time = df.index[-1].ceil(f'{1000//fs}ms')

    period_ns = int(1e9 / fs)  # Period in nanoseconds
    uniform_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=pd.Timedelta(nanoseconds=period_ns)
    )

    # Reindex to uniform grid
    df_uniform = df.reindex(uniform_index)

    # Identify gaps: missing samples or gaps > 250ms
    is_missing = df_uniform[['x', 'y', 'z']].isna().any(axis=1)

    # For gap detection, compute time differences
    time_diff = pd.Series(uniform_index).diff().dt.total_seconds() * 1000  # in ms
    time_diff.index = uniform_index

    # Mark large gaps (>250ms worth of samples)
    max_interpolate_ms = 250
    gap_threshold_samples = int(max_interpolate_ms / (1000 / fs))

    # Forward fill the gap flag to mark all samples in a gap region
    is_gap = np.zeros(len(df_uniform), dtype=int)

    # Find runs of missing data
    missing_mask = is_missing.values
    gap_starts = np.where(np.diff(np.concatenate([[False], missing_mask])))[0]
    gap_ends = np.where(np.diff(np.concatenate([missing_mask, [False]])))[0]

    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start
        if gap_length > gap_threshold_samples:
            # Mark this entire gap region
            is_gap[start:end] = 1

    # Interpolate missing values (including large gaps, but marked for later exclusion)
    df_uniform[['x', 'y', 'z']] = df_uniform[['x', 'y', 'z']].interpolate(
        method='linear',
        limit_direction='both',
        limit_area=None  # Interpolate everywhere
    )

    # Fill any remaining NaNs at edges with forward/backward fill
    df_uniform[['x', 'y', 'z']] = df_uniform[['x', 'y', 'z']].ffill().bfill()

    # Add gap flag
    df_uniform['is_gap'] = is_gap

    # Reset index and add participant_id back
    df_uniform = df_uniform.reset_index().rename(columns={'index': 'timestamp'})
    df_uniform.insert(0, 'participant_id', participant_id)

    # Report statistics
    n_interpolated = is_missing.sum()
    n_gap_samples = is_gap.sum()
    if n_interpolated > 0:
        warnings.warn(
            f"Interpolated {n_interpolated} missing samples. "
            f"{n_gap_samples} samples in large gaps (>250ms) marked for exclusion.",
            UserWarning
        )

    return df_uniform
