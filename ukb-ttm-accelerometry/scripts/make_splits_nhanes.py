#!/usr/bin/env python3
"""
Create stratified train/val/test splits for NHANES data.

Stratifies by age bin, sex, and race/ethnicity using clinical data.
"""

import argparse
import sys
from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataio.nhanes.merge_clinical import create_stratification_groups


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
        description="Create stratified NHANES splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory with participant subdirectories'
    )

    parser.add_argument(
        '--clinical-csv',
        type=str,
        required=True,
        help='Path to clinical CSV with stratification variables'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/nhanes/splits',
        help='Output directory for split CSVs'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Test set ratio (0-1)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (0-1)'
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
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NHANES Split Creator")
    logger.info("=" * 60)

    # Find all participants
    logger.info(f"Scanning {data_dir}...")
    participant_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    participant_ids = [d.name for d in participant_dirs]

    logger.info(f"Found {len(participant_ids)} participants")

    if not participant_ids:
        logger.error("No participants found")
        return 1

    # Load clinical data
    logger.info(f"Loading clinical data from {args.clinical_csv}...")
    clinical_df = pd.read_csv(args.clinical_csv)
    clinical_df['participant_id'] = clinical_df['participant_id'].astype(str)

    # Create dataframe with participants
    splits_df = pd.DataFrame({'participant_id': participant_ids})

    # Merge with clinical
    splits_df = splits_df.merge(clinical_df, on='participant_id', how='left')

    logger.info(f"Matched {splits_df['participant_id'].notna().sum()} participants with clinical data")

    # Create stratification groups
    logger.info("Creating stratification groups...")
    strat_groups = create_stratification_groups(splits_df)

    logger.info(f"Found {strat_groups.nunique()} stratification groups")

    # Ensure we can stratify
    min_group_size = strat_groups.value_counts().min()
    if min_group_size < 2:
        logger.warning(f"Some stratification groups have < 2 samples (min: {min_group_size})")
        logger.warning("Falling back to random splitting without stratification")
        strat_groups = None

    # Split
    logger.info(f"Splitting: Train {100*(1-args.test_ratio-args.val_ratio):.1f}%, "
                f"Val {100*args.val_ratio:.1f}%, Test {100*args.test_ratio:.1f}%")

    # First split: separate test set
    if strat_groups is not None:
        train_val_ids, test_ids = train_test_split(
            participant_ids,
            test_size=args.test_ratio,
            random_state=args.seed,
            stratify=strat_groups
        )

        # Get stratification for remaining participants
        train_val_mask = splits_df['participant_id'].isin(train_val_ids)
        train_val_strat = strat_groups[train_val_mask]

        # Second split: separate val from train
        val_ratio_adjusted = args.val_ratio / (1 - args.test_ratio)

        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio_adjusted,
            random_state=args.seed,
            stratify=train_val_strat
        )
    else:
        # Random splitting
        train_val_ids, test_ids = train_test_split(
            participant_ids,
            test_size=args.test_ratio,
            random_state=args.seed
        )

        val_ratio_adjusted = args.val_ratio / (1 - args.test_ratio)

        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio_adjusted,
            random_state=args.seed
        )

    # Convert to lists
    train_ids = list(train_ids)
    val_ids = list(val_ids)
    test_ids = list(test_ids)

    # Save to CSV
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"

    pd.DataFrame({'participant_id': train_ids}).to_csv(train_csv, index=False)
    pd.DataFrame({'participant_id': val_ids}).to_csv(val_csv, index=False)
    pd.DataFrame({'participant_id': test_ids}).to_csv(test_csv, index=False)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Splits Created")
    logger.info("=" * 60)
    logger.info(f"Train: {len(train_ids)} ({100*len(train_ids)/len(participant_ids):.1f}%)")
    logger.info(f"Val:   {len(val_ids)} ({100*len(val_ids)/len(participant_ids):.1f}%)")
    logger.info(f"Test:  {len(test_ids)} ({100*len(test_ids)/len(participant_ids):.1f}%)")
    logger.info("")
    logger.info(f"Saved to {output_dir}")
    logger.info(f"  - {train_csv}")
    logger.info(f"  - {val_csv}")
    logger.info(f"  - {test_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
