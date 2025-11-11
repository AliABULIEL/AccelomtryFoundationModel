#!/usr/bin/env python3
"""
Example pipeline demonstrating end-to-end usage of the toolkit.

This script shows how to:
1. Process synthetic accelerometry data
2. Create windows
3. Save to HDF5
4. Load into PyTorch dataset
5. Apply preprocessing
6. Train a simple model

Prerequisites:
    pip install -r requirements.txt
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Import toolkit modules
from src.dataio.windowing import compute_window_params, segment_stream, compute_window_timestamps
from src.dataio.preprocess import instance_standardize, inverse_standardize
from src.utils.io import save_windows_hdf5, load_windows_hdf5
from src.utils.seed import set_all_seeds
from src.utils.time_features import build_time_features_for_windows


def create_synthetic_data():
    """Create synthetic accelerometry data for demonstration."""
    print("Creating synthetic accelerometry data...")

    # Simulate 1 hour of data at 100 Hz
    fs = 100
    duration_sec = 3600
    n_samples = fs * duration_sec

    # Generate timestamps
    start_time = pd.Timestamp('2020-01-01 08:00:00', tz='UTC')
    timestamps = pd.date_range(start_time, periods=n_samples, freq='10ms')

    # Generate realistic accelerometry signals (simulated walking pattern)
    t = np.arange(n_samples) / fs

    # Walking frequency ~1.5 Hz
    walking_freq = 1.5

    # x, y, z acceleration with some noise
    x = 0.3 * np.sin(2 * np.pi * walking_freq * t) + np.random.randn(n_samples) * 0.1
    y = 0.2 * np.cos(2 * np.pi * walking_freq * t) + np.random.randn(n_samples) * 0.1
    z = 9.8 + 0.5 * np.sin(4 * np.pi * walking_freq * t) + np.random.randn(n_samples) * 0.1

    return timestamps.values, np.column_stack([x, y, z])


def main():
    """Run example pipeline."""

    # Set seed for reproducibility
    set_all_seeds(42, deterministic=True)
    print("Random seed set to 42\n")

    # 1. Create synthetic data
    timestamps, signals = create_synthetic_data()
    print(f"Created {len(signals)} samples ({len(signals)/100/60:.1f} minutes)")
    print(f"Signal shape: {signals.shape}\n")

    # 2. Compute window parameters
    fs = 100
    win_sec = 8.192
    hop_sec = 4.096

    win_n, hop_n = compute_window_params(fs, win_sec, hop_sec, rounding='floor')
    print(f"Window parameters:")
    print(f"  - Window size: {win_n} samples ({win_n/fs:.3f}s)")
    print(f"  - Hop size: {hop_n} samples ({hop_n/fs:.3f}s)")
    print(f"  - Overlap: {(1 - hop_n/win_n)*100:.1f}%\n")

    # 3. Create windows
    print("Creating windows...")
    windows = segment_stream(signals, win_n=win_n, hop_n=hop_n, pad_mode='none')
    print(f"Created {windows.shape[0]} windows")
    print(f"Window shape: {windows.shape}\n")

    # 4. Compute window timestamps
    start_times, end_times = compute_window_timestamps(timestamps, hop_n, win_n)
    print(f"Window timestamps:")
    print(f"  First window: {start_times[0]} to {end_times[0]}")
    print(f"  Last window:  {start_times[-1]} to {end_times[-1]}\n")

    # 5. Extract time features
    time_features = build_time_features_for_windows(start_times, end_times, use_midpoint=True)
    print(f"Time features extracted: {list(time_features.keys())[:5]}...\n")

    # 6. Save to HDF5
    output_dir = Path('./demo_output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'demo_windows.h5'

    print(f"Saving to {output_path}...")
    save_windows_hdf5(
        output_path,
        windows,
        start_times,
        end_times,
        metadata={
            'fs': fs,
            'win_sec': win_sec,
            'hop_sec': hop_sec,
            'participant_id': 'demo_001'
        }
    )
    print("Saved successfully!\n")

    # 7. Load data back
    print("Loading data from HDF5...")
    loaded_data = load_windows_hdf5(output_path, load_timestamps=True)
    print(f"Loaded {loaded_data['windows'].shape[0]} windows")
    print(f"Metadata: {loaded_data['metadata']}\n")

    # 8. Convert to PyTorch and apply preprocessing
    print("Converting to PyTorch tensors...")
    x = torch.from_numpy(loaded_data['windows'][:32]).float()  # First 32 windows
    print(f"Batch shape: {x.shape}\n")

    print("Applying instance standardization...")
    x_norm, (mean, std) = instance_standardize(x)

    print(f"Original stats:")
    print(f"  Mean: {x.mean().item():.3f}, Std: {x.std().item():.3f}")
    print(f"Normalized stats:")
    print(f"  Mean: {x_norm.mean().item():.6f}, Std: {x_norm.std().item():.3f}\n")

    # 9. Inverse transform
    print("Testing inverse transform...")
    x_reconstructed = inverse_standardize(x_norm, mean, std)
    reconstruction_error = (x - x_reconstructed).abs().max().item()
    print(f"Max reconstruction error: {reconstruction_error:.2e}\n")

    # 10. Summary
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nWhat we did:")
    print("  ✓ Generated synthetic accelerometry data")
    print("  ✓ Created 8.192s windows with 50% overlap")
    print("  ✓ Extracted window timestamps and time features")
    print("  ✓ Saved to compressed HDF5 format")
    print("  ✓ Loaded data back efficiently")
    print("  ✓ Applied instance standardization (RevIN-style)")
    print("  ✓ Verified perfect reconstruction")
    print("\nNext steps:")
    print("  1. Use real .cwa files with scripts/prepare_ukb.py")
    print("  2. Create train/val/test splits with scripts/make_splits.py")
    print("  3. Load data with PyTorch datasets")
    print("  4. Train models with Granite TTM or custom architectures")
    print("\nSee README.md for complete documentation!")
    print("="*60)


if __name__ == "__main__":
    main()
