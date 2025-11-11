#!/usr/bin/env python3
"""
Parse NHANES 80 Hz accelerometry data into windowed format.

Converts parquet files to HDF5 windows with:
- Window size: 8.192s at 80 Hz → 655 samples
- Hop size: 4.096s → 327 samples (50% overlap)
- RevIN-style per-window standardization
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import logging

import numpy as np
import h5py
from tqdm import tqdm

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    import pandas as pd
    HAS_POLARS = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def compute_window_params(
    fs: int = 80,
    win_sec: float = 8.192,
    hop_sec: float = 4.096,
    rounding: str = 'floor'
) -> Tuple[int, int]:
    """
    Compute window and hop sizes in samples.

    Args:
        fs: Sampling frequency in Hz
        win_sec: Window duration in seconds
        hop_sec: Hop duration in seconds
        rounding: Rounding mode ('floor', 'nearest', 'ceil')

    Returns:
        (win_n, hop_n) tuple of sample counts
    """
    win_n_exact = fs * win_sec
    hop_n_exact = fs * hop_sec

    if rounding == 'floor':
        win_n = int(np.floor(win_n_exact))
        hop_n = int(np.floor(hop_n_exact))
    elif rounding == 'nearest':
        win_n = int(np.round(win_n_exact))
        hop_n = int(np.round(hop_n_exact))
    elif rounding == 'ceil':
        win_n = int(np.ceil(win_n_exact))
        hop_n = int(np.ceil(hop_n_exact))
    else:
        raise ValueError(f"Unknown rounding mode: {rounding}")

    return win_n, hop_n


def to_windows(
    signals: np.ndarray,
    win_n: int,
    hop_n: int,
    pad_mode: str = 'none'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create strided windows from continuous signal.

    Args:
        signals: Array of shape (T, C) where T=time, C=channels
        win_n: Window size in samples
        hop_n: Hop size in samples
        pad_mode: Padding strategy ('none', 'edge', 'zero')

    Returns:
        windows: Array of shape (W, C, win_n)
        spans: Array of shape (W, 2) with [start, end) indices
    """
    T, C = signals.shape

    if pad_mode != 'none':
        # Calculate padding needed
        if T < win_n:
            pad_width = win_n - T
        else:
            remainder = (T - win_n) % hop_n
            pad_width = (hop_n - remainder) % hop_n

        if pad_width > 0:
            if pad_mode == 'edge':
                signals = np.pad(signals, ((0, pad_width), (0, 0)), mode='edge')
            elif pad_mode == 'zero':
                signals = np.pad(signals, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
            T = signals.shape[0]

    # Calculate number of windows
    if T < win_n:
        if pad_mode == 'none':
            return np.empty((0, C, win_n), dtype=signals.dtype), np.empty((0, 2), dtype=np.int64)

    num_windows = (T - win_n) // hop_n + 1

    # Create windows using stride tricks
    windows = np.zeros((num_windows, C, win_n), dtype=signals.dtype)
    spans = np.zeros((num_windows, 2), dtype=np.int64)

    for i in range(num_windows):
        start = i * hop_n
        end = start + win_n
        windows[i] = signals[start:end].T  # Shape: (C, win_n)
        spans[i] = [start, end]

    return windows, spans


def standardize_windows(
    windows: np.ndarray,
    eps: float = 1e-5
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Standardize windows using RevIN-style per-window per-channel z-score.

    Args:
        windows: Array of shape (W, C, win_n)
        eps: Small constant for numerical stability

    Returns:
        normalized: Standardized windows (W, C, win_n)
        stats: Tuple of (means, stds) each of shape (W, C)
    """
    # Compute per-window per-channel statistics
    means = windows.mean(axis=2, keepdims=True)  # Shape: (W, C, 1)
    stds = windows.std(axis=2, keepdims=True) + eps  # Shape: (W, C, 1)

    # Normalize
    normalized = (windows - means) / stds

    # Return with squeezed stats
    return normalized, (means.squeeze(axis=2), stds.squeeze(axis=2))


def load_parquet_participant(
    parquet_dir: Path,
    participant_id: str,
    cycle: str
) -> Optional[np.ndarray]:
    """
    Load parquet data for a single participant.

    Args:
        parquet_dir: Root parquet directory
        participant_id: Participant ID
        cycle: NHANES cycle

    Returns:
        Array of shape (T, 3) with x, y, z accelerations, or None if not found
    """
    participant_path = parquet_dir / f"cycle={cycle}" / f"participant_id={participant_id}"

    if not participant_path.exists():
        return None

    parquet_files = list(participant_path.glob("*.parquet"))
    if not parquet_files:
        return None

    try:
        if HAS_POLARS:
            df = pl.read_parquet(parquet_files[0])
            df = df.sort('timestamp')
            signals = np.column_stack([
                df['x'].to_numpy(),
                df['y'].to_numpy(),
                df['z'].to_numpy()
            ])
        else:
            df = pd.read_parquet(parquet_files[0])
            df = df.sort_values('timestamp')
            signals = df[['x', 'y', 'z']].values

        return signals.astype(np.float32)

    except Exception as e:
        logging.error(f"Error loading {participant_id}: {e}")
        return None


def process_participant_80hz(
    participant_id: str,
    cycle: str,
    parquet_dir: Path,
    output_dir: Path,
    win_n: int,
    hop_n: int,
    fs: int = 80,
    pad_mode: str = 'none',
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Process single participant's 80 Hz data into windows.

    Args:
        participant_id: Participant ID
        cycle: NHANES cycle
        parquet_dir: Input parquet directory
        output_dir: Output HDF5 directory
        win_n: Window size in samples
        hop_n: Hop size in samples
        fs: Sampling frequency
        pad_mode: Padding strategy
        logger: Logger instance

    Returns:
        True if successful
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Load data
    signals = load_parquet_participant(parquet_dir, participant_id, cycle)
    if signals is None:
        logger.warning(f"No data for {participant_id}")
        return False

    # Create windows
    windows, spans = to_windows(signals, win_n, hop_n, pad_mode)

    if windows.shape[0] == 0:
        logger.warning(f"No windows for {participant_id} (T={signals.shape[0]})")
        return False

    # Standardize
    windows_norm, (means, stds) = standardize_windows(windows)

    # Create output directory
    participant_dir = output_dir / participant_id
    participant_dir.mkdir(parents=True, exist_ok=True)

    # Save to HDF5
    output_file = participant_dir / "windows.h5"

    try:
        with h5py.File(output_file, 'w') as f:
            # Save windows
            f.create_dataset(
                'windows',
                data=windows_norm,
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            # Save spans
            f.create_dataset(
                'spans',
                data=spans,
                dtype='int64',
                compression='gzip',
                compression_opts=4
            )

            # Save stats for potential denormalization
            f.create_dataset(
                'means',
                data=means,
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            f.create_dataset(
                'stds',
                data=stds,
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            # Metadata
            f.attrs['participant_id'] = participant_id
            f.attrs['cycle'] = cycle
            f.attrs['fs'] = fs
            f.attrs['win_n'] = win_n
            f.attrs['hop_n'] = hop_n
            f.attrs['num_windows'] = windows.shape[0]
            f.attrs['total_samples'] = signals.shape[0]

        logger.info(f"  ✓ {participant_id}: {windows.shape[0]} windows")
        return True

    except Exception as e:
        logger.error(f"  ✗ Error saving {participant_id}: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse NHANES 80 Hz data to windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input parquet directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/nhanes/80hz_windows',
        help='Output HDF5 directory'
    )

    parser.add_argument(
        '--win-sec',
        type=float,
        default=8.192,
        help='Window duration in seconds'
    )

    parser.add_argument(
        '--hop-sec',
        type=float,
        default=4.096,
        help='Hop duration in seconds'
    )

    parser.add_argument(
        '--fs',
        type=int,
        default=80,
        help='Sampling frequency in Hz'
    )

    parser.add_argument(
        '--pad',
        type=str,
        choices=['none', 'edge', 'zero'],
        default='none',
        help='Padding strategy for partial windows'
    )

    parser.add_argument(
        '--rounding',
        type=str,
        choices=['floor', 'nearest', 'ceil'],
        default='floor',
        help='Rounding mode for window/hop sizes'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_level)
    parquet_dir = Path(args.input)
    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("NHANES 80 Hz Parser")
    logger.info("=" * 60)

    # Compute window parameters
    win_n, hop_n = compute_window_params(args.fs, args.win_sec, args.hop_sec, args.rounding)
    logger.info(f"Window size: {win_n} samples ({args.win_sec}s @ {args.fs} Hz)")
    logger.info(f"Hop size: {hop_n} samples ({args.hop_sec}s)")
    logger.info(f"Overlap: {100 * (1 - hop_n / win_n):.1f}%")

    # Find all participants
    participants = []
    for cycle_dir in parquet_dir.glob("cycle=*"):
        cycle = cycle_dir.name.split('=')[1]
        for participant_dir in cycle_dir.glob("participant_id=*"):
            participant_id = participant_dir.name.split('=')[1]
            participants.append((participant_id, cycle))

    if not participants:
        logger.error("No participants found")
        return 1

    logger.info(f"Found {len(participants)} participants")

    # Process participants
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for participant_id, cycle in tqdm(participants, desc="Processing"):
        success = process_participant_80hz(
            participant_id,
            cycle,
            parquet_dir,
            output_dir,
            win_n,
            hop_n,
            args.fs,
            args.pad,
            logger
        )
        if success:
            success_count += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Successful: {success_count} / {len(participants)}")
    logger.info(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
