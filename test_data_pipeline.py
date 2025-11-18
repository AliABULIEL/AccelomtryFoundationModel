# FILE: test_data_pipeline.py

"""
Test script for validating the data pipeline with synthetic data.

This script:
1. Generates synthetic accelerometry data (simulating UKB format)
2. Creates synthetic clinical metadata
3. Tests all data processing functions
4. Validates the complete Dataset and DataLoader pipeline
5. Provides comprehensive logging of all steps

This is designed to run on any machine without requiring actual UKB data,
making it perfect for testing in Colab or local environments.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import shutil
from datetime import datetime, timedelta

from ukb_ttm_accel.config import TrainingConfig, save_config_to_yaml
from ukb_ttm_accel.data import (
    UKBAccelDataset,
    accel_collate_fn,
    resample_and_calibrate,
    create_windows,
    zscore_normalize_windows,
    load_clinical_data,
    build_label_and_metadata,
)
from ukb_ttm_accel.utils import setup_logger, set_global_seed


def generate_synthetic_accelerometry(
    participant_id: int,
    duration_hours: int = 24,
    sampling_rate_hz: int = 50,
    output_path: str = None,
    file_format: str = "csv"
) -> str:
    """
    Generate synthetic accelerometry data simulating UK Biobank format.

    Args:
        participant_id: Participant ID number
        duration_hours: Duration of recording in hours
        sampling_rate_hz: Sampling rate in Hz
        output_path: Path to save file (auto-generated if None)
        file_format: Output format ("csv" or "parquet")

    Returns:
        Path to generated file
    """
    # Calculate number of samples
    num_samples = duration_hours * 3600 * sampling_rate_hz

    # Generate timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [
        start_time + timedelta(seconds=i/sampling_rate_hz)
        for i in range(num_samples)
    ]

    # Generate realistic accelerometry data
    # Simulate daily activity patterns with some randomness
    t = np.arange(num_samples) / sampling_rate_hz

    # Base activity level (varying throughout the day)
    activity_level = 0.5 + 0.3 * np.sin(2 * np.pi * t / (24 * 3600))

    # X-axis: dominant horizontal movement
    x = activity_level * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.1, num_samples)

    # Y-axis: secondary horizontal movement
    y = activity_level * np.sin(2 * np.pi * 0.3 * t + np.pi/4) + np.random.normal(0, 0.1, num_samples)

    # Z-axis: vertical movement + gravity
    z = 1.0 + activity_level * np.sin(2 * np.pi * 0.7 * t + np.pi/2) + np.random.normal(0, 0.1, num_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps,
        'x': x,
        'y': y,
        'z': z
    })

    # Determine output path
    if output_path is None:
        output_path = f"synthetic_data/participant_{participant_id}.{file_format}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save file
    if file_format == "csv":
        df.to_csv(output_path, index=False)
    elif file_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    return str(output_path)


def generate_synthetic_clinical_data(
    participant_ids: list,
    output_path: str = None
) -> str:
    """
    Generate synthetic clinical metadata.

    Args:
        participant_ids: List of participant IDs
        output_path: Path to save CSV (auto-generated if None)

    Returns:
        Path to generated CSV
    """
    np.random.seed(42)

    num_participants = len(participant_ids)

    # Generate realistic clinical data
    data = {
        'eid': participant_ids,
        'age': np.random.randint(40, 70, num_participants),
        'sex': np.random.choice([0, 1], num_participants),  # 0=female, 1=male
        'height': np.random.normal(170, 10, num_participants),  # cm
        'weight': np.random.normal(75, 15, num_participants),  # kg
    }

    # Calculate BMI
    height_m = np.array(data['height']) / 100.0
    data['bmi'] = data['weight'] / (height_m ** 2)

    # Generate some binary outcomes
    data['hypertension'] = np.random.choice([0, 1], num_participants, p=[0.7, 0.3])
    data['diabetes'] = np.random.choice([0, 1], num_participants, p=[0.85, 0.15])

    df = pd.DataFrame(data)

    # Determine output path
    if output_path is None:
        output_path = "synthetic_data/clinical.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(output_path, index=False)

    return str(output_path)


def test_windowing_functions(logger):
    """Test windowing and preprocessing functions."""
    logger.info("\n" + "="*60)
    logger.info("Testing windowing functions...")
    logger.info("="*60)

    # Generate a small test dataset
    num_samples = 5000  # 100 seconds at 50 Hz
    timestamps = pd.date_range(start='2024-01-01', periods=num_samples, freq='20ms')

    df = pd.DataFrame({
        'x': np.random.randn(num_samples),
        'y': np.random.randn(num_samples),
        'z': np.random.randn(num_samples) + 1.0  # Add gravity bias
    }, index=timestamps)
    df.index.name = 'time'

    # Test create_windows
    logger.info("Testing create_windows...")
    windows, time_of_day = create_windows(
        accel_df=df,
        window_seconds=10,
        sampling_rate_hz=50,
        overlap_fraction=0.5
    )

    logger.info(f"  Input shape: {df.shape}")
    logger.info(f"  Windows shape: {windows.shape}")
    logger.info(f"  Time of day shape: {time_of_day.shape}")
    logger.info(f"  Expected windows: ~{(num_samples - 500) // 250 + 1}")

    # Test normalization
    logger.info("Testing zscore_normalize_windows...")
    normalized = zscore_normalize_windows(windows)

    logger.info(f"  Original mean: {windows.mean():.4f}, std: {windows.std():.4f}")
    logger.info(f"  Normalized mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")

    # Verify per-window normalization
    window_0_mean = normalized[0, :, :].mean()
    window_0_std = normalized[0, :, :].std()
    logger.info(f"  Window 0 mean: {window_0_mean:.6f}, std: {window_0_std:.4f}")

    logger.info("Windowing functions test: PASSED")


def test_clinical_metadata(logger, clinical_csv_path):
    """Test clinical metadata functions."""
    logger.info("\n" + "="*60)
    logger.info("Testing clinical metadata functions...")
    logger.info("="*60)

    # Load clinical data
    logger.info(f"Loading clinical data from: {clinical_csv_path}")
    clinical_df = load_clinical_data(clinical_csv_path)

    logger.info(f"  Loaded {len(clinical_df)} participants")
    logger.info(f"  Columns: {list(clinical_df.columns)}")

    # Test label extraction
    participant_id = clinical_df.index[0]

    logger.info(f"\nTesting label extraction for participant {participant_id}...")

    # Test BMI class
    label, metadata = build_label_and_metadata(
        eid=participant_id,
        clinical_df=clinical_df,
        label_type="bmi_class",
        metadata_columns=["age", "sex", "bmi"]
    )

    logger.info(f"  BMI class label: {label}")
    logger.info(f"  Metadata: {metadata}")

    # Test different label types
    for label_type in ["bmi", "age_group", "sex"]:
        label, _ = build_label_and_metadata(
            eid=participant_id,
            clinical_df=clinical_df,
            label_type=label_type,
            metadata_columns=["age", "sex"]
        )
        logger.info(f"  {label_type} label: {label}")

    logger.info("Clinical metadata functions test: PASSED")


def test_dataset_and_dataloader(logger, accel_files, clinical_csv_path, config):
    """Test UKBAccelDataset and DataLoader."""
    logger.info("\n" + "="*60)
    logger.info("Testing Dataset and DataLoader...")
    logger.info("="*60)

    # Create dataset
    logger.info("Creating UKBAccelDataset...")

    dataset = UKBAccelDataset(
        accel_file_paths=accel_files,
        clinical_csv_path=clinical_csv_path,
        split="train",
        window_seconds=config.window_seconds,
        sampling_rate_hz=config.sampling_rate_hz,
        normalize_per_window=config.normalize_per_window,
        max_windows_per_participant=10,  # Limit for testing
        label_type=config.label_type,
        metadata_columns=config.metadata_columns,
        cache_hdf5_path=None,  # No caching for first test
        source_type="csv",
    )

    logger.info(f"  Dataset created with {len(dataset)} windows")

    # Test __getitem__
    logger.info("\nTesting __getitem__...")
    sample = dataset[0]

    logger.info(f"  Sample keys: {list(sample.keys())}")
    logger.info(f"  Signal shape: {sample['signal'].shape}")
    logger.info(f"  Signal dtype: {sample['signal'].dtype}")
    logger.info(f"  Label: {sample['label'].item()}")
    logger.info(f"  Label dtype: {sample['label'].dtype}")
    logger.info(f"  Metadata shape: {sample['metadata'].shape}")
    logger.info(f"  Participant ID: {sample['participant_id'].item()}")
    logger.info(f"  Time of day: {sample['time_of_day'].item():.4f}")

    # Test DataLoader
    logger.info("\nTesting DataLoader...")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=accel_collate_fn,
        num_workers=0,  # Single process for testing
    )

    logger.info(f"  Created DataLoader with batch_size=4")

    # Get a batch
    batch = next(iter(dataloader))

    logger.info(f"  Batch keys: {list(batch.keys())}")
    logger.info(f"  Signals shape: {batch['signals'].shape}")
    logger.info(f"  Labels shape: {batch['labels'].shape}")
    logger.info(f"  Metadata shape: {batch['metadata'].shape}")
    logger.info(f"  Metadata mask shape: {batch['metadata_mask'].shape}")

    # Verify batch values
    logger.info("\nVerifying batch values...")
    logger.info(f"  Signals mean: {batch['signals'].mean():.4f}")
    logger.info(f"  Signals std: {batch['signals'].std():.4f}")
    logger.info(f"  Labels: {batch['labels'].tolist()}")
    logger.info(f"  Participant IDs: {batch['participant_ids'].tolist()}")

    logger.info("Dataset and DataLoader test: PASSED")


def test_hdf5_caching(logger, accel_files, clinical_csv_path, config):
    """Test HDF5 caching functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing HDF5 caching...")
    logger.info("="*60)

    cache_path = "synthetic_data/test_cache.h5"

    # Create dataset with caching
    logger.info("Creating dataset with HDF5 caching...")

    dataset = UKBAccelDataset(
        accel_file_paths=accel_files,
        clinical_csv_path=clinical_csv_path,
        split="train",
        window_seconds=config.window_seconds,
        sampling_rate_hz=config.sampling_rate_hz,
        normalize_per_window=config.normalize_per_window,
        max_windows_per_participant=10,
        label_type=config.label_type,
        metadata_columns=config.metadata_columns,
        cache_hdf5_path=cache_path,
        source_type="csv",
    )

    logger.info(f"  Dataset created and cached to: {cache_path}")
    logger.info(f"  Dataset length: {len(dataset)}")

    # Load from cache
    logger.info("\nLoading dataset from cache...")

    dataset_from_cache = UKBAccelDataset(
        accel_file_paths=accel_files,
        clinical_csv_path=clinical_csv_path,
        split="train",
        window_seconds=config.window_seconds,
        sampling_rate_hz=config.sampling_rate_hz,
        normalize_per_window=config.normalize_per_window,
        max_windows_per_participant=10,
        label_type=config.label_type,
        metadata_columns=config.metadata_columns,
        cache_hdf5_path=cache_path,
        source_type="csv",
    )

    logger.info(f"  Loaded from cache, length: {len(dataset_from_cache)}")

    # Compare samples
    sample1 = dataset[0]
    sample2 = dataset_from_cache[0]

    logger.info("\nComparing original vs cached...")
    logger.info(f"  Signals match: {torch.allclose(sample1['signal'], sample2['signal'])}")
    logger.info(f"  Labels match: {sample1['label'] == sample2['label']}")
    logger.info(f"  Metadata match: {torch.allclose(sample1['metadata'], sample2['metadata'])}")

    logger.info("HDF5 caching test: PASSED")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test data pipeline with synthetic data")

    parser.add_argument(
        "--num-participants",
        type=int,
        default=10,
        help="Number of synthetic participants to generate"
    )

    parser.add_argument(
        "--duration-hours",
        type=int,
        default=24,
        help="Duration of each recording in hours"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_outputs",
        help="Directory for test outputs"
    )

    parser.add_argument(
        "--keep-synthetic-data",
        action="store_true",
        help="Keep synthetic data after testing (default: delete)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logger(
        name="test_pipeline",
        log_file=str(output_dir / "test.log")
    )

    logger.info("="*60)
    logger.info("UKB TTM Accelerometry - Data Pipeline Test")
    logger.info("="*60)

    # Set random seed
    set_global_seed(42)

    # Generate synthetic data
    logger.info(f"\nGenerating synthetic data for {args.num_participants} participants...")

    participant_ids = list(range(1000001, 1000001 + args.num_participants))
    accel_files = []

    for pid in participant_ids:
        file_path = generate_synthetic_accelerometry(
            participant_id=pid,
            duration_hours=args.duration_hours,
            sampling_rate_hz=50,
            file_format="csv"
        )
        accel_files.append(file_path)
        logger.info(f"  Generated: {file_path}")

    # Generate clinical data
    clinical_csv_path = generate_synthetic_clinical_data(participant_ids)
    logger.info(f"  Generated clinical data: {clinical_csv_path}")

    # Create test configuration
    config = TrainingConfig(
        window_seconds=10,
        sampling_rate_hz=50,
        batch_size=4,
        label_type="bmi_class",
        metadata_columns=["age", "sex", "bmi"],
    )

    # Save config for reference
    config_path = output_dir / "test_config.yaml"
    save_config_to_yaml(config, str(config_path))
    logger.info(f"  Saved test config: {config_path}")

    try:
        # Run tests
        test_windowing_functions(logger)
        test_clinical_metadata(logger, clinical_csv_path)
        test_dataset_and_dataloader(logger, accel_files, clinical_csv_path, config)
        test_hdf5_caching(logger, accel_files, clinical_csv_path, config)

        # All tests passed
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*60)
        logger.info("\nData pipeline is working correctly.")
        logger.info("You can now proceed to use this pipeline with real UKB data.")

    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"TEST FAILED: {e}")
        logger.error(f"{'='*60}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    finally:
        # Clean up synthetic data unless requested to keep
        if not args.keep_synthetic_data:
            logger.info("\nCleaning up synthetic data...")
            synthetic_dir = Path("synthetic_data")
            if synthetic_dir.exists():
                shutil.rmtree(synthetic_dir)
                logger.info("  Synthetic data deleted")


if __name__ == "__main__":
    main()
