#!/usr/bin/env python3
"""
Generate synthetic accelerometry data for testing and development.

This script creates realistic .cwa-compatible files that mimic UK Biobank
accelerometry data patterns. Use this for testing the pipeline without
requiring actual UK Biobank access.
"""
import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

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


def generate_activity_pattern(
    duration_hours: float,
    fs: int = 100,
    activity_level: str = 'moderate'
) -> np.ndarray:
    """
    Generate realistic activity patterns mimicking human behavior.

    Args:
        duration_hours: Duration in hours
        fs: Sampling frequency
        activity_level: 'sedentary', 'light', 'moderate', or 'active'

    Returns:
        Array of shape (n_samples, 3) with x, y, z acceleration
    """
    n_samples = int(duration_hours * 3600 * fs)
    t = np.arange(n_samples) / fs

    # Activity level parameters
    activity_params = {
        'sedentary': {'base_std': 0.05, 'movement_freq': 0.1, 'movement_amp': 0.2},
        'light': {'base_std': 0.1, 'movement_freq': 0.5, 'movement_amp': 0.5},
        'moderate': {'base_std': 0.15, 'movement_freq': 1.5, 'movement_amp': 1.0},
        'active': {'base_std': 0.2, 'movement_freq': 2.5, 'movement_amp': 2.0}
    }

    params = activity_params.get(activity_level, activity_params['moderate'])

    # Generate base signals
    # Walking frequency components
    walking_freq = params['movement_freq']  # Hz

    # X-axis: lateral movement
    x = params['movement_amp'] * np.sin(2 * np.pi * walking_freq * t)
    x += np.random.randn(n_samples) * params['base_std']

    # Y-axis: anterior-posterior
    y = params['movement_amp'] * 0.7 * np.cos(2 * np.pi * walking_freq * t + np.pi/4)
    y += np.random.randn(n_samples) * params['base_std']

    # Z-axis: vertical (includes gravity ~9.8 m/s²)
    z = 9.8 + params['movement_amp'] * 0.5 * np.sin(4 * np.pi * walking_freq * t)
    z += np.random.randn(n_samples) * params['base_std']

    return np.column_stack([x, y, z])


def add_realistic_features(
    signals: np.ndarray,
    timestamps: np.ndarray,
    fs: int = 100
) -> tuple:
    """
    Add realistic features to synthetic data.

    Features:
    - Circadian rhythm (less activity at night)
    - Random non-wear periods
    - Occasional gaps

    Args:
        signals: Array of shape (n_samples, 3)
        timestamps: Timestamp array
        fs: Sampling frequency

    Returns:
        Tuple of (modified_signals, gap_mask)
    """
    signals = signals.copy()
    n_samples = len(signals)

    # Extract hour of day
    hours = pd.DatetimeIndex(timestamps).hour

    # Circadian modulation (reduced activity at night)
    circadian = 0.3 + 0.7 * np.cos((hours - 14) * np.pi / 12)  # Peak at 2 PM
    circadian = np.repeat(circadian, fs * 3600)[:n_samples]

    # Apply circadian modulation to non-gravity components
    signals[:, 0] *= circadian
    signals[:, 1] *= circadian
    signals[:, 2] = 9.8 + (signals[:, 2] - 9.8) * circadian

    # Add occasional gaps (3-5 gaps per day)
    gap_mask = np.zeros(n_samples, dtype=bool)
    n_gaps = np.random.randint(3, 6) * (n_samples // (24 * 3600 * fs))

    for _ in range(n_gaps):
        gap_start = np.random.randint(0, n_samples - fs * 600)  # Random position
        gap_duration = np.random.randint(fs * 30, fs * 600)  # 30s to 10min gaps
        gap_end = min(gap_start + gap_duration, n_samples)
        gap_mask[gap_start:gap_end] = True

    return signals, gap_mask


def generate_participant_data(
    participant_id: str,
    duration_days: float,
    fs: int = 100,
    output_format: str = 'csv.gz',
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Generate complete dataset for one participant.

    Args:
        participant_id: Participant identifier
        duration_days: Duration in days
        fs: Sampling frequency
        output_format: 'csv.gz' or 'npy'
        logger: Logger instance

    Returns:
        DataFrame with accelerometry data
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Generating {duration_days} days of data for {participant_id}...")

    # Generate timestamps
    start_time = pd.Timestamp('2020-01-01 08:00:00', tz='UTC')
    n_samples = int(duration_days * 24 * 3600 * fs)
    timestamps = pd.date_range(start_time, periods=n_samples, freq=f'{1000/fs}ms')

    # Generate activity patterns for different times of day
    activity_schedule = []
    current_hour = 0

    while current_hour < duration_days * 24:
        hour_of_day = current_hour % 24

        # Schedule: 0-6: sleep, 6-9: light, 9-12: moderate, 12-13: light,
        # 13-18: moderate, 18-20: light, 20-22: sedentary, 22-24: sleep
        if 0 <= hour_of_day < 6 or 22 <= hour_of_day < 24:
            activity = 'sedentary'  # Sleep
            duration = min(1.0, duration_days * 24 - current_hour)
        elif 6 <= hour_of_day < 9 or 18 <= hour_of_day < 20:
            activity = 'light'
            duration = min(1.0, duration_days * 24 - current_hour)
        elif 12 <= hour_of_day < 13 or 20 <= hour_of_day < 22:
            activity = 'light'
            duration = min(1.0, duration_days * 24 - current_hour)
        else:
            activity = 'moderate'
            duration = min(1.0, duration_days * 24 - current_hour)

        activity_schedule.append((activity, duration))
        current_hour += duration

    # Generate signals for each activity period
    all_signals = []

    for activity, duration in tqdm(activity_schedule, desc="Generating activity patterns"):
        signals = generate_activity_pattern(duration, fs, activity)
        all_signals.append(signals)

    signals = np.vstack(all_signals)[:n_samples]  # Ensure exact length

    # Add realistic features
    signals, gap_mask = add_realistic_features(signals, timestamps.values, fs)

    # Create DataFrame
    df = pd.DataFrame({
        'participant_id': participant_id,
        'timestamp': timestamps,
        'x': signals[:, 0],
        'y': signals[:, 1],
        'z': signals[:, 2],
        'is_gap': gap_mask.astype(int)
    })

    logger.info(f"Generated {len(df)} samples ({len(df)/fs/3600:.1f} hours)")
    logger.info(f"  Mean: x={df['x'].mean():.3f}, y={df['y'].mean():.3f}, z={df['z'].mean():.3f}")
    logger.info(f"  Std: x={df['x'].std():.3f}, y={df['y'].std():.3f}, z={df['z'].std():.3f}")
    logger.info(f"  Gaps: {gap_mask.sum()} samples ({gap_mask.sum()/len(df)*100:.2f}%)")

    return df


def save_participant_data(
    df: pd.DataFrame,
    output_dir: Path,
    output_format: str = 'csv.gz'
) -> Path:
    """
    Save participant data to disk.

    Args:
        df: DataFrame with participant data
        output_dir: Output directory
        output_format: 'csv.gz' or 'npy'

    Returns:
        Path to saved file
    """
    participant_id = df['participant_id'].iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == 'csv.gz':
        output_path = output_dir / f'{participant_id}.csv.gz'
        df.to_csv(output_path, index=False, compression='gzip')
    elif output_format == 'npy':
        output_path = output_dir / f'{participant_id}.npy'
        # Save as structured array
        data = np.array([
            (row['timestamp'], row['x'], row['y'], row['z'])
            for _, row in df.iterrows()
        ], dtype=[
            ('timestamp', 'datetime64[ns]'),
            ('x', 'float32'),
            ('y', 'float32'),
            ('z', 'float32')
        ])
        np.save(output_path, data)
    else:
        raise ValueError(f"Unknown format: {output_format}")

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic accelerometry data for testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
EXAMPLES:
  # Generate 5 participants with 7 days each
  python scripts/generate_demo_data.py \\
      --n-participants 5 \\
      --duration-days 7 \\
      --output-dir ./data/demo

  # Generate 1 week of data for testing
  python scripts/generate_demo_data.py \\
      --n-participants 3 \\
      --duration-days 7 \\
      --output-dir ./data/test_data

  # Generate small dataset for quick testing
  python scripts/generate_demo_data.py \\
      --n-participants 2 \\
      --duration-days 1 \\
      --output-dir ./data/quick_test

NEXT STEPS AFTER GENERATION:
  1. Process the synthetic data:
     python scripts/prepare_ukb.py \\
         --input ./data/demo \\
         --outdir ./data/processed

  2. Create splits:
     python scripts/make_splits.py \\
         --data-dir ./data/processed \\
         --output-dir ./data/splits

  3. Train models with the processed data
        """
    )

    parser.add_argument(
        '--n-participants',
        type=int,
        default=5,
        help='Number of participants to generate'
    )

    parser.add_argument(
        '--duration-days',
        type=float,
        default=7.0,
        help='Duration in days per participant'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/demo',
        help='Output directory for generated data'
    )

    parser.add_argument(
        '--fs',
        type=int,
        default=100,
        help='Sampling frequency in Hz'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        choices=['csv.gz', 'npy'],
        default='csv.gz',
        help='Output file format'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
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
    np.random.seed(args.seed)

    logger.info("="*60)
    logger.info("SYNTHETIC ACCELEROMETRY DATA GENERATOR")
    logger.info("="*60)
    logger.info(f"Participants: {args.n_participants}")
    logger.info(f"Duration: {args.duration_days} days per participant")
    logger.info(f"Sampling rate: {args.fs} Hz")
    logger.info(f"Output format: {args.output_format}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("")

    output_dir = Path(args.output_dir)

    # Generate data for each participant
    generated_files = []

    for i in range(args.n_participants):
        participant_id = f'demo_{i+1:06d}'

        logger.info(f"\n[{i+1}/{args.n_participants}] Generating participant {participant_id}")

        # Generate data
        df = generate_participant_data(
            participant_id,
            args.duration_days,
            args.fs,
            args.output_format,
            logger
        )

        # Save data
        output_path = save_participant_data(df, output_dir, args.output_format)
        generated_files.append(output_path)

        logger.info(f"  Saved to: {output_path}")

    # Save metadata
    metadata = {
        'n_participants': args.n_participants,
        'duration_days': args.duration_days,
        'fs': args.fs,
        'output_format': args.output_format,
        'seed': args.seed,
        'generated_files': [str(f) for f in generated_files],
        'total_samples': args.n_participants * args.duration_days * 24 * 3600 * args.fs,
        'total_hours': args.n_participants * args.duration_days * 24
    }

    import json
    metadata_file = output_dir / 'generation_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Generated {args.n_participants} participants")
    logger.info(f"Total duration: {args.n_participants * args.duration_days * 24:.0f} hours")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metadata: {metadata_file}")
    logger.info("")

    # Calculate approximate sizes
    samples_per_participant = args.duration_days * 24 * 3600 * args.fs
    mb_per_participant = samples_per_participant * 4 * 3 / (1024**2)  # 4 bytes per float, 3 channels
    total_mb = mb_per_participant * args.n_participants

    logger.info(f"Approximate size: {total_mb:.1f} MB")
    logger.info("")

    logger.info("NEXT STEPS:")
    logger.info("="*60)
    logger.info("1. Process the synthetic data:")
    logger.info(f"   python scripts/prepare_ukb.py \\")
    logger.info(f"       --input {output_dir} \\")
    logger.info(f"       --outdir ./data/processed \\")
    logger.info(f"       --win-sec 8.192 \\")
    logger.info(f"       --hop-sec 4.096")
    logger.info("")
    logger.info("2. Create train/val/test splits:")
    logger.info("   python scripts/make_splits.py \\")
    logger.info("       --data-dir ./data/processed \\")
    logger.info("       --output-dir ./data/splits")
    logger.info("")
    logger.info("3. Load and train:")
    logger.info("   See example_pipeline.py for usage examples")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
