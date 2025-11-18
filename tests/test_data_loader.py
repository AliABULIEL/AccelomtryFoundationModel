"""
Tests for data loading pipeline
"""

import pytest
import numpy as np
import torch
import tempfile
import h5py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_loader import (
    AccelerometryDataLoader,
    StreamingDataset,
    create_stratified_splits,
)


class TestAccelerometryDataLoader:
    """Test data loader functionality."""

    def test_window_extraction(self):
        """Test window extraction from continuous data."""
        # Create synthetic data
        n_samples = 10000
        n_channels = 3
        data = np.random.randn(n_samples, n_channels).astype(np.float32)

        # Initialize loader
        loader = AccelerometryDataLoader(
            data_dir="./data",
            cache_dir="./cache",
            window_size=820,
            stride=410,
        )

        # Extract windows
        windows = loader.extract_windows(data, normalize=True)

        # Check shape
        expected_n_windows = (n_samples - 820) // 410 + 1
        assert windows.shape[0] == expected_n_windows
        assert windows.shape[1] == 820
        assert windows.shape[2] == 3

        # Check normalization
        for i in range(min(10, len(windows))):
            window = windows[i]
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            assert np.allclose(mean, 0, atol=1e-5)
            assert np.allclose(std, 1, atol=1e-1)

        print(f"✓ Window extraction: {windows.shape}")

    def test_hdf5_streaming(self):
        """Test HDF5 streaming dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock HDF5 file
            hdf5_path = Path(tmpdir) / "test.h5"

            n_windows = 1000
            window_size = 820
            n_channels = 3

            with h5py.File(hdf5_path, 'w') as f:
                f.create_dataset(
                    'windows',
                    data=np.random.randn(n_windows, window_size, n_channels).astype(np.float32),
                )
                f.create_dataset(
                    'labels',
                    data=np.random.randint(0, 4, n_windows).astype(np.int32),
                )

            # Create streaming dataset
            dataset = StreamingDataset(str(hdf5_path))

            assert len(dataset) == n_windows

            # Test __getitem__
            window, label = dataset[0]
            assert window.shape == (window_size, n_channels)
            assert 0 <= label < 4

            # Test with subset of indices
            indices = np.random.choice(n_windows, size=100, replace=False)
            subset = StreamingDataset(str(hdf5_path), indices=indices)
            assert len(subset) == 100

            print(f"✓ HDF5 streaming: {len(dataset)} samples")

    def test_stratified_splits(self):
        """Test stratified train/val/test splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock HDF5 file
            hdf5_path = Path(tmpdir) / "test.h5"

            n_windows = 1000
            # Create imbalanced labels
            labels = np.random.choice([0, 1, 2, 3], size=n_windows, p=[0.3, 0.4, 0.2, 0.1])

            with h5py.File(hdf5_path, 'w') as f:
                f.create_dataset(
                    'windows',
                    data=np.random.randn(n_windows, 820, 3).astype(np.float32),
                )
                f.create_dataset(
                    'labels',
                    data=labels.astype(np.int32),
                )

            # Create splits
            train_idx, val_idx, test_idx = create_stratified_splits(
                str(hdf5_path),
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            # Check sizes
            total = len(train_idx) + len(val_idx) + len(test_idx)
            assert total == n_windows

            # Check no overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0

            # Check stratification
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            test_labels = labels[test_idx]

            for label in [0, 1, 2, 3]:
                train_ratio = (train_labels == label).sum() / len(train_labels)
                val_ratio = (val_labels == label).sum() / len(val_labels)
                test_ratio = (test_labels == label).sum() / len(test_labels)

                # Ratios should be similar (within 20%)
                assert abs(train_ratio - val_ratio) < 0.2
                assert abs(train_ratio - test_ratio) < 0.2

            print(f"✓ Stratified splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    test = TestAccelerometryDataLoader()
    test.test_window_extraction()
    test.test_hdf5_streaming()
    test.test_stratified_splits()
    print("\n✓ All data loader tests passed!")
