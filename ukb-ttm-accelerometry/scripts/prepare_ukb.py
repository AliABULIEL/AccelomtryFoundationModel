#!/usr/bin/env python3
"""
Prepare UK Biobank accelerometry data for training.

Converts raw .cwa files to windowed HDF5/Zarr format with preprocessing.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataio.ukb_cwa_reader import read_cwa_to_segments, resample_to_fs
from src.dataio.windowing import (
    compute_window_params,
    segment_stream,
    compute_window_timestamps,
    filter_windows_by_gaps
)
from src.utils.io import save_windows_hdf5, save_windows_zarr


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def process_single_file(
    input_path: Path,
    output_dir: Path,
    win_sec: float,
    hop_sec: float,
    fs: int,
    rounding: str,
    max_gap_ratio: float,
    storage_format: str,
    logger: logging.Logger
) -> Optional[dict]:
    """
    Process a single accelerometry file.

    Args:
        input_path: Path to input .cwa file
        output_dir: Output directory
        win_sec: Window size in seconds
        hop_sec: Hop size in seconds
        fs: Sampling frequency
        rounding: Rounding mode
        max_gap_ratio: Maximum gap ratio
        storage_format: Output format (hdf5 or zarr)
        logger: Logger instance

    Returns:
        Dictionary with processing statistics, or None if failed
    """
    try:
        logger.info(f"Processing {input_path.name}...")

        # Read and resample
        logger.info("  Reading .cwa file...")
        df = read_cwa_to_segments(input_path)

        logger.info(f"  Loaded {len(df)} samples")

        logger.info("  Resampling to {fs} Hz...")
        df_resampled = resample_to_fs(df, fs=fs)

        logger.info(f"  Resampled to {len(df_resampled)} samples")

        # Extract signals and gap flags
        signals = df_resampled[['x', 'y', 'z']].values  # Shape: (T, 3)
        gap_flags = df_resampled['is_gap'].values
        timestamps = df_resampled['timestamp'].values
        participant_id = df_resampled['participant_id'].iloc[0]

        # Compute window parameters
        win_n, hop_n = compute_window_params(fs, win_sec, hop_sec, rounding)
        logger.info(f"  Window params: win_n={win_n}, hop_n={hop_n}")

        # Create windows
        logger.info("  Creating windows...")
        windows = segment_stream(signals, win_n, hop_n, pad_mode="none")
        logger.info(f"  Created {windows.shape[0]} windows")

        # Compute window timestamps
        start_times, end_times = compute_window_timestamps(timestamps, hop_n, win_n)

        # Filter windows by gap content
        logger.info("  Filtering windows by gap content...")
        windows_filtered, valid_mask = filter_windows_by_gaps(
            windows, gap_flags, hop_n, win_n, max_gap_ratio
        )
        start_times_filtered = start_times[valid_mask]
        end_times_filtered = end_times[valid_mask]

        n_filtered = len(windows) - len(windows_filtered)
        logger.info(f"  Filtered {n_filtered} windows with >={max_gap_ratio*100:.1f}% gaps")
        logger.info(f"  Retained {len(windows_filtered)} windows")

        if len(windows_filtered) == 0:
            logger.warning(f"  No valid windows for {input_path.name}")
            return None

        # Create output directory for this participant
        participant_dir = output_dir / participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)

        # Save windows
        if storage_format == "hdf5":
            output_path = participant_dir / "windows.h5"
            save_windows_hdf5(
                output_path,
                windows_filtered,
                start_times_filtered,
                end_times_filtered,
                gap_flags=None,  # Already filtered
                metadata={
                    'participant_id': participant_id,
                    'fs': fs,
                    'win_sec': win_sec,
                    'hop_sec': hop_sec,
                    'win_n': win_n,
                    'hop_n': hop_n,
                    'rounding': rounding,
                    'n_filtered': n_filtered,
                    'max_gap_ratio': max_gap_ratio
                }
            )
        elif storage_format == "zarr":
            output_path = participant_dir / "windows.zarr"
            save_windows_zarr(
                output_path,
                windows_filtered,
                start_times_filtered,
                end_times_filtered,
                gap_flags=None,
                metadata={
                    'participant_id': participant_id,
                    'fs': fs,
                    'win_sec': win_sec,
                    'hop_sec': hop_sec,
                    'win_n': win_n,
                    'hop_n': hop_n,
                    'rounding': rounding,
                    'n_filtered': n_filtered,
                    'max_gap_ratio': max_gap_ratio
                }
            )

        logger.info(f"  Saved to {output_path}")

        # Compute wear time
        total_duration = (timestamps[-1] - timestamps[0]).astype('timedelta64[h]').astype(float)

        return {
            'participant_id': participant_id,
            'input_file': input_path.name,
            'n_samples': len(df_resampled),
            'n_windows_total': len(windows),
            'n_windows_valid': len(windows_filtered),
            'n_windows_filtered': n_filtered,
            'duration_hours': total_duration,
            'output_path': str(output_path),
            'success': True
        }

    except Exception as e:
        logger.error(f"  Error processing {input_path.name}: {e}", exc_info=True)
        return {
            'participant_id': input_path.stem,
            'input_file': input_path.name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare UK Biobank accelerometry data for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input .cwa file or directory containing .cwa files"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for processed windows"
    )

    parser.add_argument(
        "--win-sec",
        type=float,
        default=8.192,
        help="Window duration in seconds"
    )

    parser.add_argument(
        "--hop-sec",
        type=float,
        default=4.096,
        help="Hop duration in seconds (for overlap)"
    )

    parser.add_argument(
        "--fs",
        type=int,
        default=100,
        help="Sampling frequency in Hz"
    )

    parser.add_argument(
        "--rounding",
        type=str,
        choices=["floor", "nearest", "ceil"],
        default="floor",
        help="Rounding mode for window size"
    )

    parser.add_argument(
        "--max-gap-ratio",
        type=float,
        default=0.1,
        help="Maximum ratio of gap samples per window"
    )

    parser.add_argument(
        "--min-wear-hours",
        type=float,
        default=24,
        help="Minimum wear time in hours to include participant"
    )

    parser.add_argument(
        "--storage-format",
        type=str,
        choices=["hdf5", "zarr"],
        default="hdf5",
        help="Storage format for output"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Path to save processing summary CSV"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Parse input
    input_path = Path(args.input)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of files to process
    if input_path.is_file():
        if input_path.suffix == '.cwa':
            input_files = [input_path]
        else:
            logger.error(f"Input file must be .cwa format: {input_path}")
            return 1
    elif input_path.is_dir():
        input_files = sorted(input_path.glob("*.cwa"))
        if not input_files:
            logger.error(f"No .cwa files found in {input_path}")
            return 1
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    logger.info(f"Found {len(input_files)} files to process")

    # Process files
    results = []
    for file_path in tqdm(input_files, desc="Processing files"):
        result = process_single_file(
            file_path,
            output_dir,
            args.win_sec,
            args.hop_sec,
            args.fs,
            args.rounding,
            args.max_gap_ratio,
            args.storage_format,
            logger
        )
        if result is not None:
            results.append(result)

    # Create summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)

    n_success = sum(1 for r in results if r['success'])
    n_failed = len(results) - n_success

    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"Successful: {n_success}")
    logger.info(f"Failed: {n_failed}")

    if n_success > 0:
        successful_results = [r for r in results if r['success']]
        total_windows = sum(r['n_windows_valid'] for r in successful_results)
        total_duration = sum(r['duration_hours'] for r in successful_results)

        logger.info(f"Total windows: {total_windows}")
        logger.info(f"Total duration: {total_duration:.1f} hours")
        logger.info(f"Average windows per participant: {total_windows/n_success:.1f}")

    # Save summary if requested
    if args.summary_file:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(args.summary_file, index=False)
        logger.info(f"\nSummary saved to: {args.summary_file}")

    logger.info("="*60)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
