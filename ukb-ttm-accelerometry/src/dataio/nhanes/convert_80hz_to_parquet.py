#!/usr/bin/env python3
"""
Convert NHANES raw 80Hz accelerometry from R format to Parquet.

Reads RData files downloaded by download_80hz.R and converts to
standardized columnar parquet format with schema:
    participant_id, timestamp (UTC ns), x, y, z, device_location, is_gap
"""
import argparse
import sys
from pathlib import Path
import logging
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def read_rdata_file(path: Path) -> Optional[pd.DataFrame]:
    """
    Read RData file and extract accelerometry data.

    Args:
        path: Path to RData file

    Returns:
        DataFrame with accelerometry data or None if error
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()

        # Load RData file
        robjects.r['load'](str(path))

        # Get the participant_data object (adjust name as needed)
        r_df = robjects.r['participant_data']

        # Convert to pandas
        df = pandas2ri.rpy2py(r_df)

        return df

    except ImportError:
        logging.error("rpy2 not installed. Install with: pip install rpy2")
        return None
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return None


def standardize_80hz_data(
    df: pd.DataFrame,
    participant_id: str,
    cycle: str,
    fs: int = 80
) -> pd.DataFrame:
    """
    Standardize accelerometry data to common schema.

    Args:
        df: Raw data from R
        participant_id: Participant ID
        cycle: NHANES cycle
        fs: Sampling frequency (Hz)

    Returns:
        Standardized DataFrame
    """
    # Detect column names (NHANES format may vary)
    # Common column patterns:
    # - PAXINTEN, PAXSTEP: activity counts/steps (not raw)
    # - PAXDAY, PAXN: day number, sample number
    # - Actual raw data columns depend on file structure

    # For 80Hz data, we expect timestamp + x,y,z axes
    # This is a template - adjust based on actual NHANES data structure

    standardized = pd.DataFrame({
        'participant_id': participant_id,
        'cycle': cycle,
        'device_location': 'hip',  # NHANES 2011-2014 typically hip
    })

    # Detect and extract axes
    # Look for common patterns
    axis_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'x' in col_lower and 'acc' in col_lower:
            axis_cols['x'] = col
        elif 'y' in col_lower and 'acc' in col_lower:
            axis_cols['y'] = col
        elif 'z' in col_lower and 'acc' in col_lower:
            axis_cols['z'] = col

    if not axis_cols:
        # Fallback: assume first 3 numeric columns are x,y,z
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            axis_cols = {'x': numeric_cols[0], 'y': numeric_cols[1], 'z': numeric_cols[2]}

    # Extract axes
    for axis, col in axis_cols.items():
        standardized[axis] = df[col].values

    # Generate timestamps
    # NHANES may have day + sample number
    if 'PAXDAY' in df.columns and 'PAXN' in df.columns:
        # Convert day + sample to timestamp
        # Assume start date from cycle
        base_date = pd.Timestamp(f'{cycle[:4]}-01-01', tz='UTC')
        days = df['PAXDAY'].values
        samples = df['PAXN'].values

        # Calculate timestamps
        timestamps = []
        for day, sample in zip(days, samples):
            ts = base_date + pd.Timedelta(days=day) + pd.Timedelta(seconds=sample/fs)
            timestamps.append(ts)

        standardized['timestamp'] = pd.DatetimeIndex(timestamps)
    else:
        # Generate sequential timestamps
        n_samples = len(df)
        timestamps = pd.date_range(
            start=pd.Timestamp(f'{cycle[:4]}-01-01', tz='UTC'),
            periods=n_samples,
            freq=f'{1000/fs}ms'
        )
        standardized['timestamp'] = timestamps

    return standardized


def enforce_80hz_timestamps(
    df: pd.DataFrame,
    fs: int = 80,
    max_gap_ms: float = 250.0
) -> pd.DataFrame:
    """
    Enforce exact 80 Hz timestamps with gap detection.

    Args:
        df: DataFrame with timestamp column
        fs: Sampling frequency
        max_gap_ms: Maximum gap for interpolation (ms)

    Returns:
        DataFrame with uniform timestamps and is_gap flag
    """
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create uniform time grid
    start_time = df['timestamp'].iloc[0].floor(f'{1000/fs}ms')
    end_time = df['timestamp'].iloc[-1].ceil(f'{1000/fs}ms')

    uniform_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f'{1000/fs}ms'
    )

    # Reindex to uniform grid
    df = df.set_index('timestamp')
    df_uniform = df.reindex(uniform_index)

    # Detect gaps
    is_missing = df_uniform[['x', 'y', 'z']].isna().any(axis=1)

    # Mark large gaps
    max_gap_samples = int(max_gap_ms / (1000 / fs))
    is_gap = np.zeros(len(df_uniform), dtype=int)

    # Find runs of missing data
    missing_mask = is_missing.values
    gap_starts = np.where(np.diff(np.concatenate([[False], missing_mask])))[0]
    gap_ends = np.where(np.diff(np.concatenate([missing_mask, [False]])))[0]

    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start
        if gap_length > max_gap_samples:
            is_gap[start:end] = 1

    # Interpolate small gaps
    df_uniform[['x', 'y', 'z']] = df_uniform[['x', 'y', 'z']].interpolate(
        method='linear',
        limit_direction='both'
    )

    # Fill remaining with forward/backward fill
    df_uniform[['x', 'y', 'z']] = df_uniform[['x', 'y', 'z']].ffill().bfill()

    # Add gap flag
    df_uniform['is_gap'] = is_gap

    # Reset index
    df_uniform = df_uniform.reset_index().rename(columns={'index': 'timestamp'})

    # Ensure other columns are propagated
    for col in ['participant_id', 'cycle', 'device_location']:
        if col in df.columns:
            df_uniform[col] = df[col].iloc[0]

    return df_uniform


def convert_participant(
    input_path: Path,
    output_dir: Path,
    participant_id: str,
    cycle: str,
    fs: int = 80,
    logger: logging.Logger = None
) -> bool:
    """
    Convert single participant's data to parquet.

    Args:
        input_path: Path to RData file
        output_dir: Output directory
        participant_id: Participant ID
        cycle: NHANES cycle
        fs: Sampling frequency
        logger: Logger instance

    Returns:
        True if successful
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Read RData
        df = read_rdata_file(input_path)
        if df is None:
            return False

        # Standardize
        df = standardize_80hz_data(df, participant_id, cycle, fs)

        # Enforce timestamps
        df = enforce_80hz_timestamps(df, fs)

        # Create output directory
        participant_dir = output_dir / f"cycle={cycle}" / f"participant_id={participant_id}"
        participant_dir.mkdir(parents=True, exist_ok=True)

        # Write to parquet
        output_file = participant_dir / "data.parquet"

        # Convert timestamp to int64 nanoseconds for parquet
        df['timestamp'] = df['timestamp'].astype('int64')

        # Write with compression
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            output_file,
            compression='snappy',
            use_dictionary=True
        )

        logger.info(f"  ✓ Converted {participant_id}: {len(df)} samples")

        return True

    except Exception as e:
        logger.error(f"  ✗ Error converting {participant_id}: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert NHANES 80Hz data to Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with RData files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/nhanes/80hz_parquet',
        help='Output parquet directory'
    )

    parser.add_argument(
        '--fs',
        type=int,
        default=80,
        help='Sampling frequency in Hz'
    )

    parser.add_argument(
        '--manifest',
        type=str,
        default=None,
        help='Path to manifest.csv (if None, uses input/manifest.csv)'
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
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    logger.info("="*60)
    logger.info("NHANES 80Hz → Parquet Converter")
    logger.info("="*60)

    # Read manifest
    manifest_path = Path(args.manifest) if args.manifest else input_dir / "manifest.csv"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.info("Looking for RData files in input directory...")

        # Fallback: scan directory
        rdata_files = list(input_dir.rglob("*.RData"))
        if not rdata_files:
            logger.error("No RData files found")
            return 1

        logger.info(f"Found {len(rdata_files)} RData files")

        # Process each file
        success_count = 0
        for rdata_file in tqdm(rdata_files, desc="Converting"):
            # Extract participant ID and cycle from path
            parts = rdata_file.parts
            cycle = parts[-3] if len(parts) >= 3 else "unknown"
            participant_id = parts[-2] if len(parts) >= 2 else rdata_file.stem

            success = convert_participant(
                rdata_file,
                output_dir,
                participant_id,
                cycle,
                args.fs,
                logger
            )

            if success:
                success_count += 1

    else:
        # Use manifest
        manifest = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest: {len(manifest)} participants")

        success_count = 0
        for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Converting"):
            success = convert_participant(
                Path(row['path']),
                output_dir,
                row['participant_id'],
                row['cycle'],
                args.fs,
                logger
            )

            if success:
                success_count += 1

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Conversion Complete")
    logger.info("="*60)
    logger.info(f"Successful: {success_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nNext step:")
    logger.info(f"  python -m src.dataio.nhanes.parse_80hz --input {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
