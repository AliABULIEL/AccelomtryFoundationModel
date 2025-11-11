"""
Parse NHANES minute-level accelerometry summaries.

Handles:
- PAXMIN_C/D (2003-2006): minute activity counts
- PAXMIN_G/H (2011-2014): minute summaries
- Timestamp construction
- Wear time flagging
- Windowing for TTM (1024-minute context, 96-minute prediction)
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def download_minute_data(
    cycles: List[str],
    download_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Load PAXMIN tables from XPT files (no R required!).

    Assumes XPT files have been downloaded by download_nhanes.py.
    If not found, provides instructions to download them.

    Args:
        cycles: List of cycles like '2003-2004', '2011-2012'
        download_dir: Directory containing downloaded XPT files

    Returns:
        Dictionary mapping cycle to DataFrame
    """
    logger = logging.getLogger(__name__)

    cycle_map = {
        '2003-2004': 'PAXMIN_C',
        '2005-2006': 'PAXMIN_D',
        '2011-2012': 'PAXMIN_G',
        '2013-2014': 'PAXMIN_H'
    }

    downloaded_data = {}

    for cycle in cycles:
        if cycle not in cycle_map:
            logger.warning(f"Unknown cycle: {cycle}")
            continue

        table_name = cycle_map[cycle]
        logger.info(f"Loading {table_name}...")

        # Find XPT file
        cycle_dir = download_dir / cycle.replace('-', '_')
        xpt_file = cycle_dir / f"{table_name}.XPT"

        if not xpt_file.exists():
            logger.warning(f"  ✗ {table_name}: Not found at {xpt_file}")
            logger.warning(f"     Download it first with:")
            logger.warning(f"     python -m src.dataio.nhanes.download_nhanes --cycles {cycle} --data-types minute")
            continue

        try:
            # Load XPT file using pyreadstat
            import pyreadstat
            df, meta = pyreadstat.read_xport(str(xpt_file))
            downloaded_data[cycle] = df
            logger.info(f"  ✓ {table_name}: {len(df)} rows")

        except Exception as e:
            logger.error(f"  ✗ {table_name}: Error loading XPT file: {e}")
            continue

    return downloaded_data


def parse_minute_summaries(
    minute_dataframes: Dict[str, pd.DataFrame],
    base_date: str = "2000-01-03"
) -> pd.DataFrame:
    """
    Parse minute-level data into normalized schema.

    Args:
        minute_dataframes: Dict mapping cycle to raw DataFrame
        base_date: Base date for timestamp anchoring

    Returns:
        Normalized DataFrame with columns:
            participant_id, timestamp, axis_summaries, wear_flag, cycle
    """
    logger = logging.getLogger(__name__)
    normalized_list = []

    for cycle, df in minute_dataframes.items():
        logger.info(f"Parsing {cycle}...")

        # Ensure SEQN exists
        if 'SEQN' not in df.columns:
            logger.warning(f"No SEQN column in {cycle}")
            continue

        # Get unique participants
        participants = df['SEQN'].unique()
        logger.info(f"  {len(participants)} participants")

        # Process each participant
        for pid in tqdm(participants, desc=f"  {cycle}", leave=False):
            participant_df = df[df['SEQN'] == pid].copy()
            participant_df = participant_df.sort_index()

            # Construct timestamps
            if 'PAXDAY' in participant_df.columns:
                # Method 1: Use PAXDAY/PAXHOUR/PAXMIN if available
                if 'PAXHOUR' in participant_df.columns and 'PAXMIN' in participant_df.columns:
                    timestamps = pd.to_datetime(base_date) + \
                        pd.to_timedelta(participant_df['PAXDAY'] - 1, unit='D') + \
                        pd.to_timedelta(participant_df['PAXHOUR'], unit='h') + \
                        pd.to_timedelta(participant_df['PAXMIN'], unit='m')
                else:
                    # Use PAXDAY and minute index
                    minute_offsets = participant_df.groupby('PAXDAY').cumcount()
                    timestamps = pd.to_datetime(base_date) + \
                        pd.to_timedelta(participant_df['PAXDAY'] - 1, unit='D') + \
                        pd.to_timedelta(minute_offsets, unit='m')
            else:
                # Method 2: Sequential timestamps (relative minutes)
                timestamps = pd.to_datetime(base_date) + \
                    pd.to_timedelta(np.arange(len(participant_df)), unit='m')

            # Extract activity summaries
            axis_summaries = None

            if 'PAXINTEN' in participant_df.columns:
                # 2003-2006: activity counts
                axis_summaries = participant_df['PAXINTEN'].values
            elif 'PAXSTEP' in participant_df.columns:
                # Steps as proxy
                axis_summaries = participant_df['PAXSTEP'].values
            else:
                # 2011-2014: aggregate numeric columns
                # Look for columns like AX*, PAX*
                numeric_cols = participant_df.select_dtypes(include=[np.number]).columns
                accel_cols = [col for col in numeric_cols if
                             col.startswith('AX') or
                             (col.startswith('PAX') and col not in ['PAXDAY', 'PAXHOUR', 'PAXMIN'])]

                if accel_cols:
                    axis_summaries = participant_df[accel_cols].sum(axis=1).values
                else:
                    # Fallback: zeros
                    axis_summaries = np.zeros(len(participant_df))

            # Wear flag
            wear_flag = np.ones(len(participant_df), dtype=bool)

            if 'PAXSTAT' in participant_df.columns:
                # Explicit wear status
                wear_flag = (participant_df['PAXSTAT'] == 1).values
            elif 'WEARFLAG' in participant_df.columns:
                wear_flag = (participant_df['WEARFLAG'] == 1).values
            elif axis_summaries is not None:
                # Infer: wear if activity > 0
                wear_flag = axis_summaries > 0

            # Create participant frame
            participant_normalized = pd.DataFrame({
                'participant_id': str(pid),
                'timestamp': timestamps,
                'axis_summaries': axis_summaries,
                'wear_flag': wear_flag,
                'cycle': cycle
            })

            normalized_list.append(participant_normalized)

    if not normalized_list:
        logger.warning("No data parsed")
        return pd.DataFrame()

    # Concatenate all
    all_data = pd.concat(normalized_list, ignore_index=True)

    # Sort by participant and timestamp
    all_data = all_data.sort_values(['participant_id', 'timestamp']).reset_index(drop=True)

    logger.info(f"Parsed {len(all_data)} minute-level rows")

    return all_data


def create_minute_windows(
    minute_df: pd.DataFrame,
    context_length: int = 1024,
    prediction_length: int = 96,
    min_wear_fraction: float = 0.8,
    output_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create rolling windows for minute-level forecasting.

    Args:
        minute_df: Normalized minute DataFrame
        context_length: Context window size in minutes (default: 1024 = 17.1h)
        prediction_length: Prediction window size in minutes (default: 96 = 1.6h)
        min_wear_fraction: Minimum wear fraction in context (default: 0.8)
        output_dir: Optional output directory to save arrays

    Returns:
        X_context: (B, 1, context_length) array
        Y_future: (B, 1, prediction_length) array
        windows_meta: DataFrame with (participant_id, start_timestamp)
    """
    logger = logging.getLogger(__name__)

    total_length = context_length + prediction_length

    X_list = []
    Y_list = []
    meta_list = []

    # Process each participant
    participants = minute_df['participant_id'].unique()

    for pid in tqdm(participants, desc="Creating windows"):
        participant_data = minute_df[minute_df['participant_id'] == pid]

        if len(participant_data) < total_length:
            continue

        # Sort by timestamp
        participant_data = participant_data.sort_values('timestamp').reset_index(drop=True)

        # Extract arrays
        values = participant_data['axis_summaries'].values
        wear = participant_data['wear_flag'].values
        timestamps = participant_data['timestamp'].values

        # Create sliding windows
        num_windows = len(values) - total_length + 1

        for i in range(num_windows):
            context_values = values[i:i+context_length]
            context_wear = wear[i:i+context_length]

            # Check wear requirement
            wear_fraction = context_wear.sum() / context_length
            if wear_fraction < min_wear_fraction:
                continue

            # Extract future
            future_values = values[i+context_length:i+total_length]

            # Append
            X_list.append(context_values)
            Y_list.append(future_values)
            meta_list.append({
                'participant_id': pid,
                'start_timestamp': str(timestamps[i])
            })

    if not X_list:
        logger.warning("No valid windows created")
        return (
            np.empty((0, 1, context_length), dtype=np.float32),
            np.empty((0, 1, prediction_length), dtype=np.float32),
            pd.DataFrame(columns=['participant_id', 'start_timestamp'])
        )

    # Stack into arrays
    X_context = np.stack(X_list).astype(np.float32)[:, np.newaxis, :]  # (B, 1, Tc)
    Y_future = np.stack(Y_list).astype(np.float32)[:, np.newaxis, :]  # (B, 1, Tp)

    # Metadata
    windows_meta = pd.DataFrame(meta_list)

    logger.info(f"Created {len(X_context)} windows")

    # Save if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / 'X_context.npy', X_context)
        np.save(output_dir / 'Y_future.npy', Y_future)
        windows_meta.to_csv(output_dir / 'windows_meta.csv', index=False)

        logger.info(f"Saved to {output_dir}")

    return X_context, Y_future, windows_meta


def save_minute_summaries(
    minute_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save minute summaries to CSV.

    Args:
        minute_df: Normalized minute DataFrame
        output_path: Output CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minute_df.to_csv(output_path, index=False)
    logger = logging.getLogger(__name__)
    logger.info(f"Saved minute summaries to {output_path}")


def compute_wear_stats(
    minute_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute wear time statistics per participant.

    Args:
        minute_df: Normalized minute DataFrame

    Returns:
        DataFrame with participant_id, total_minutes, wear_minutes, wear_fraction
    """
    stats = minute_df.groupby('participant_id').agg(
        total_minutes=('timestamp', 'count'),
        wear_minutes=('wear_flag', 'sum')
    ).reset_index()

    stats['wear_fraction'] = stats['wear_minutes'] / stats['total_minutes']

    return stats
