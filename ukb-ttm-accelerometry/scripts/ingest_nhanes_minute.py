#!/usr/bin/env python3
"""
NHANES minute-level data ingestion pipeline.

Downloads, normalizes, creates windows, merges clinical data, and creates splits.
"""

import argparse
import sys
from pathlib import Path
import logging

from src.dataio.nhanes.parse_minute import (
    download_minute_data,
    parse_minute_summaries,
    create_minute_windows,
    save_minute_summaries,
    compute_wear_stats
)

from src.dataio.nhanes.merge_clinical import (
    download_clinical_data,
    normalize_clinical_variables,
    merge_with_accelerometry,
    save_clinical_csv
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NHANES minute-level ingestion pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/nhanes',
        help='Data directory'
    )

    parser.add_argument(
        '--cycles',
        type=str,
        default='2003-2004,2005-2006,2011-2012,2013-2014',
        help='Comma-separated cycles'
    )

    parser.add_argument(
        '--context-length',
        type=int,
        default=1024,
        help='Context window length in minutes'
    )

    parser.add_argument(
        '--prediction-length',
        type=int,
        default=96,
        help='Prediction window length in minutes'
    )

    parser.add_argument(
        '--min-wear-fraction',
        type=float,
        default=0.8,
        help='Minimum wear fraction in context'
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
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cycles = [c.strip() for c in args.cycles.split(',')]

    logger.info("=" * 60)
    logger.info("NHANES Minute-Level Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Cycles: {cycles}")
    logger.info(f"Context: {args.context_length} min, Prediction: {args.prediction_length} min")
    logger.info("")

    # Step 1: Download minute data
    logger.info("Step 1/6: Downloading minute-level data...")
    raw_minute_dir = data_dir / "minute_raw"
    minute_dataframes = download_minute_data(cycles, raw_minute_dir)

    if not minute_dataframes:
        logger.error("No minute data downloaded")
        return 1

    logger.info(f"Downloaded {len(minute_dataframes)} cycles")
    logger.info("")

    # Step 2: Parse and normalize
    logger.info("Step 2/6: Parsing and normalizing...")
    minute_df = parse_minute_summaries(minute_dataframes)

    if minute_df.empty:
        logger.error("No data parsed")
        return 1

    # Save normalized summaries
    minute_csv = data_dir / "minute_summaries.csv"
    save_minute_summaries(minute_df, minute_csv)
    logger.info("")

    # Step 3: Compute wear stats
    logger.info("Step 3/6: Computing wear statistics...")
    wear_stats = compute_wear_stats(minute_df)
    wear_stats_csv = data_dir / "wear_stats.csv"
    wear_stats.to_csv(wear_stats_csv, index=False)
    logger.info(f"Saved to {wear_stats_csv}")
    logger.info("")

    # Step 4: Create windows
    logger.info("Step 4/6: Creating windows...")
    windows_dir = data_dir / "minute_windows"
    X_context, Y_future, windows_meta = create_minute_windows(
        minute_df,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        min_wear_fraction=args.min_wear_fraction,
        output_dir=windows_dir
    )

    if len(X_context) == 0:
        logger.error("No windows created")
        return 1

    logger.info("")

    # Step 5: Download and merge clinical data
    logger.info("Step 5/6: Downloading and merging clinical data...")

    # Get participant IDs with minute data
    accel_participants = minute_df['participant_id'].unique().tolist()

    # Download clinical data
    clinical_data = download_clinical_data(cycles, data_dir / "clinical_raw")

    # Normalize
    clinical_df = normalize_clinical_variables(clinical_data)

    # Merge
    merged_df = merge_with_accelerometry(clinical_df, accel_participants)

    # Save
    clinical_csv = data_dir / "clinical.csv"
    save_clinical_csv(merged_df, clinical_csv)
    logger.info("")

    # Step 6: Create splits
    logger.info("Step 6/6: Creating stratified splits...")

    # Import make_splits functionality
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Get participant IDs from windows
    participant_ids = windows_meta['participant_id'].unique()

    # Merge with clinical for stratification
    splits_df = pd.DataFrame({'participant_id': participant_ids})
    splits_df = splits_df.merge(merged_df, on='participant_id', how='left')

    # Create stratification groups
    from src.dataio.nhanes.merge_clinical import create_stratification_groups
    strat_groups = create_stratification_groups(splits_df)

    # Split: 80% train, 10% val, 10% test
    # First split: 80-20
    train_ids, test_ids = train_test_split(
        participant_ids,
        test_size=0.2,
        random_state=42,
        stratify=strat_groups
    )

    # Second split: 10-10 from remaining 80
    train_strat = strat_groups[splits_df['participant_id'].isin(train_ids)]
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.125,  # 10% of total = 12.5% of 80%
        random_state=42,
        stratify=train_strat
    )

    # Save splits
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({'participant_id': train_ids}).to_csv(
        splits_dir / "train.csv", index=False
    )
    pd.DataFrame({'participant_id': val_ids}).to_csv(
        splits_dir / "val.csv", index=False
    )
    pd.DataFrame({'participant_id': test_ids}).to_csv(
        splits_dir / "test.csv", index=False
    )

    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    logger.info(f"Saved to {splits_dir}")
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Minute summaries: {minute_csv}")
    logger.info(f"Windows: {windows_dir}")
    logger.info(f"Clinical: {clinical_csv}")
    logger.info(f"Splits: {splits_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Train model:")
    logger.info(f"     python train.py --config src/configs/nhanes_minute.yaml")
    logger.info("  2. Evaluate:")
    logger.info("     python evaluate.py --checkpoint model.ckpt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
