#!/usr/bin/env python3
"""
Create train/validation/test splits for UK Biobank accelerometry data.

Supports participant-level and window-level splitting strategies.
"""
import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_all_seeds


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def find_participant_dirs(data_dir: Path) -> List[Path]:
    """
    Find all participant directories containing processed data.

    Args:
        data_dir: Root directory containing participant subdirectories

    Returns:
        List of participant directory paths
    """
    participant_dirs = []

    # Look for directories with windows.h5 or windows.zarr
    for path in data_dir.iterdir():
        if path.is_dir():
            if (path / "windows.h5").exists() or (path / "windows.zarr").exists():
                participant_dirs.append(path)

    return sorted(participant_dirs)


def create_participant_splits(
    participant_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: np.ndarray = None
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits at participant level.

    Args:
        participant_ids: List of participant IDs
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed
        stratify: Optional stratification array (e.g., age groups, sex)

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to participant ID lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Set seed for reproducibility
    set_all_seeds(seed)

    # First split: separate test set
    train_val_ids, test_ids = train_test_split(
        participant_ids,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify
    )

    # Second split: separate train and val from remaining
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    if stratify is not None:
        # Need to reindex stratify array for second split
        train_val_indices = [i for i, pid in enumerate(participant_ids) if pid in train_val_ids]
        stratify_train_val = stratify[train_val_indices]
    else:
        stratify_train_val = None

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=stratify_train_val
    )

    return {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
        'test': sorted(test_ids)
    }


def load_stratification_info(
    stratify_file: Path,
    participant_ids: List[str]
) -> np.ndarray:
    """
    Load stratification information from CSV.

    Args:
        stratify_file: Path to CSV with participant_id and stratification column
        participant_ids: List of participant IDs to match

    Returns:
        Array of stratification values aligned with participant_ids
    """
    df = pd.read_csv(stratify_file)

    # Assume first column is participant_id, second is stratification variable
    df = df.set_index(df.columns[0])

    # Extract stratification values in the same order as participant_ids
    stratify_values = []
    for pid in participant_ids:
        if pid in df.index:
            stratify_values.append(df.loc[pid].iloc[0])
        else:
            stratify_values.append(None)  # Missing value

    return np.array(stratify_values)


def save_splits(
    splits: Dict[str, List[str]],
    output_dir: Path,
    format: str = "json"
) -> None:
    """
    Save splits to disk.

    Args:
        splits: Dictionary of splits
        output_dir: Output directory
        format: Output format ('json' or 'txt')
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Save as single JSON file
        output_file = output_dir / "splits.json"
        with open(output_file, 'w') as f:
            json.dump(splits, f, indent=2)

    elif format == "txt":
        # Save as separate text files
        for split_name, ids in splits.items():
            output_file = output_dir / f"{split_name}.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(ids))

    else:
        raise ValueError(f"Unknown format: {format}")


def create_splits_with_metadata(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_file: Path = None,
    logger: logging.Logger = None
) -> Dict[str, List[str]]:
    """
    Create splits and save with metadata.

    Args:
        data_dir: Directory containing participant data
        output_dir: Output directory for splits
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        stratify_file: Optional stratification CSV
        logger: Logger instance

    Returns:
        Dictionary of splits
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Find participant directories
    logger.info(f"Scanning {data_dir} for participant data...")
    participant_dirs = find_participant_dirs(data_dir)

    if not participant_dirs:
        raise ValueError(f"No participant directories found in {data_dir}")

    participant_ids = [d.name for d in participant_dirs]
    logger.info(f"Found {len(participant_ids)} participants")

    # Load stratification if provided
    stratify = None
    if stratify_file is not None:
        logger.info(f"Loading stratification from {stratify_file}")
        stratify = load_stratification_info(stratify_file, participant_ids)
        logger.info(f"Loaded stratification for {len(stratify)} participants")

    # Create splits
    logger.info("Creating splits...")
    splits = create_participant_splits(
        participant_ids,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
        stratify
    )

    # Log split sizes
    logger.info(f"Train: {len(splits['train'])} participants ({len(splits['train'])/len(participant_ids)*100:.1f}%)")
    logger.info(f"Val:   {len(splits['val'])} participants ({len(splits['val'])/len(participant_ids)*100:.1f}%)")
    logger.info(f"Test:  {len(splits['test'])} participants ({len(splits['test'])/len(participant_ids)*100:.1f}%)")

    # Save splits
    logger.info(f"Saving splits to {output_dir}...")
    save_splits(splits, output_dir, format="json")
    save_splits(splits, output_dir, format="txt")

    # Save metadata
    metadata = {
        'data_dir': str(data_dir),
        'n_participants': len(participant_ids),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed,
        'stratify_file': str(stratify_file) if stratify_file else None,
        'splits': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        }
    }

    metadata_file = output_dir / "split_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_file}")

    return splits


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for UK Biobank accelerometry data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed participant data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for splits"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--stratify-file",
        type=str,
        default=None,
        help="CSV file with stratification variable (e.g., age group, sex)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Parse paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    stratify_file = Path(args.stratify_file) if args.stratify_file else None

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f"Ratios must sum to 1.0 (got {total_ratio})")
        return 1

    # Create splits
    try:
        splits = create_splits_with_metadata(
            data_dir,
            output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
            stratify_file,
            logger
        )

        logger.info("\n" + "="*60)
        logger.info("SPLITS CREATED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Splits saved to: {output_dir}")
        logger.info("Files created:")
        logger.info(f"  - splits.json (all splits in one file)")
        logger.info(f"  - train.txt, val.txt, test.txt (individual files)")
        logger.info(f"  - split_metadata.json (metadata)")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"Error creating splits: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
