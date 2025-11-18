"""
UK Biobank Accelerometry Data Loader
Optimized for Colab with HDF5 caching and streaming
"""

import os
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging
from tqdm import tqdm
import actipy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccelerometryDataLoader:
    """
    Production-grade data loader for UK Biobank accelerometry data.

    Features:
    - HDF5 caching for fast repeated access
    - Streaming to avoid memory overflow
    - Automatic .cwa processing with actipy
    - Window extraction optimized for TTM (8.192s = 820 samples @ 100Hz)
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        window_size: int = 820,  # 8.192 seconds at 100Hz
        stride: int = 410,  # 50% overlap
        sample_rate: int = 100,
        use_cache: bool = True,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing .cwa files
            cache_dir: Directory for HDF5 cache
            window_size: Number of samples per window (820 = 8.192s)
            stride: Stride between windows
            sample_rate: Target sample rate in Hz
            use_cache: Whether to use HDF5 caching
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.use_cache = use_cache

        logger.info(f"Initialized DataLoader: window={window_size} samples ({window_size/sample_rate:.3f}s)")

    def process_cwa_file(
        self,
        cwa_path: str,
        participant_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single .cwa file using actipy.

        Args:
            cwa_path: Path to .cwa file
            participant_id: Optional participant ID

        Returns:
            data: Array of shape (n_samples, 3) with X, Y, Z accelerometer data
            info: Dictionary with metadata (sample_rate, duration, etc.)
        """
        if participant_id is None:
            participant_id = Path(cwa_path).stem

        cache_path = self.cache_dir / f"{participant_id}.h5"

        # Check cache first
        if self.use_cache and cache_path.exists():
            logger.info(f"Loading cached data for {participant_id}")
            with h5py.File(cache_path, 'r') as f:
                data = f['data'][:]
                info = dict(f.attrs)
            return data, info

        # Process .cwa file with actipy
        logger.info(f"Processing {cwa_path} with actipy...")
        try:
            # Read and process with actipy
            # actipy handles: resampling to 100Hz, gravity calibration, filtering
            data, info_dict = actipy.read_device(
                cwa_path,
                lowpass_hz=20,  # 20Hz lowpass filter
                calibrate_gravity=True,
                detect_nonwear=True,
                resample_hz=self.sample_rate,
            )

            # Extract XYZ accelerometer data
            if isinstance(data, dict):
                acc_data = np.column_stack([
                    data['x'].values,
                    data['y'].values,
                    data['z'].values
                ])
            else:
                acc_data = data[['x', 'y', 'z']].values

            info = {
                'participant_id': participant_id,
                'sample_rate': self.sample_rate,
                'duration_hours': len(acc_data) / (self.sample_rate * 3600),
                'n_samples': len(acc_data),
            }

            # Cache processed data
            if self.use_cache:
                logger.info(f"Caching processed data to {cache_path}")
                with h5py.File(cache_path, 'w') as f:
                    f.create_dataset('data', data=acc_data, compression='gzip')
                    for k, v in info.items():
                        f.attrs[k] = v

            return acc_data, info

        except Exception as e:
            logger.error(f"Error processing {cwa_path}: {e}")
            raise

    def extract_windows(
        self,
        data: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract fixed-size windows from continuous accelerometer data.

        Args:
            data: Array of shape (n_samples, 3)
            normalize: Whether to apply z-normalization per window

        Returns:
            windows: Array of shape (n_windows, window_size, 3)
        """
        n_samples = len(data)
        n_windows = (n_samples - self.window_size) // self.stride + 1

        windows = np.zeros((n_windows, self.window_size, 3), dtype=np.float32)

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            window = data[start_idx:end_idx]

            if normalize:
                # Z-normalization per channel
                mean = window.mean(axis=0, keepdims=True)
                std = window.std(axis=0, keepdims=True) + 1e-8
                window = (window - mean) / std

            windows[i] = window

        logger.info(f"Extracted {n_windows} windows from {n_samples} samples")
        return windows

    def create_dataset_hdf5(
        self,
        cwa_files: List[str],
        output_path: str,
        labels: Optional[Dict[str, np.ndarray]] = None,
        max_files: Optional[int] = None,
    ):
        """
        Create HDF5 dataset from multiple .cwa files.
        Optimized for streaming to avoid memory overflow.

        Args:
            cwa_files: List of paths to .cwa files
            output_path: Path for output HDF5 file
            labels: Optional dict mapping participant_id -> label array
            max_files: Maximum number of files to process (for testing)
        """
        if max_files:
            cwa_files = cwa_files[:max_files]

        logger.info(f"Creating HDF5 dataset from {len(cwa_files)} files")

        # First pass: count total windows
        total_windows = 0
        file_windows = []

        for cwa_path in tqdm(cwa_files, desc="Counting windows"):
            try:
                data, _ = self.process_cwa_file(cwa_path)
                n_windows = (len(data) - self.window_size) // self.stride + 1
                file_windows.append((cwa_path, n_windows))
                total_windows += n_windows
            except Exception as e:
                logger.warning(f"Skipping {cwa_path}: {e}")

        logger.info(f"Total windows: {total_windows}")

        # Create HDF5 file with datasets
        with h5py.File(output_path, 'w') as f:
            # Create datasets
            ds_windows = f.create_dataset(
                'windows',
                shape=(total_windows, self.window_size, 3),
                dtype=np.float32,
                chunks=(64, self.window_size, 3),
                compression='gzip',
                compression_opts=4,
            )

            if labels is not None:
                ds_labels = f.create_dataset(
                    'labels',
                    shape=(total_windows,),
                    dtype=np.int32,
                    chunks=(64,),
                )

            # Store metadata
            f.attrs['window_size'] = self.window_size
            f.attrs['stride'] = self.stride
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['n_files'] = len(file_windows)

            # Second pass: write windows
            idx = 0
            for cwa_path, n_windows in tqdm(file_windows, desc="Writing windows"):
                try:
                    participant_id = Path(cwa_path).stem
                    data, _ = self.process_cwa_file(cwa_path)
                    windows = self.extract_windows(data, normalize=True)

                    # Write windows
                    ds_windows[idx:idx+n_windows] = windows

                    # Write labels if provided
                    if labels is not None and participant_id in labels:
                        label_array = labels[participant_id]
                        # Match labels to windows
                        window_labels = self._match_labels_to_windows(
                            label_array, n_windows
                        )
                        ds_labels[idx:idx+n_windows] = window_labels

                    idx += n_windows

                except Exception as e:
                    logger.warning(f"Error processing {cwa_path}: {e}")

        logger.info(f"Created HDF5 dataset: {output_path}")
        logger.info(f"Size: {os.path.getsize(output_path) / 1e9:.2f} GB")

    def _match_labels_to_windows(
        self,
        labels: np.ndarray,
        n_windows: int
    ) -> np.ndarray:
        """
        Match continuous labels to extracted windows.
        Uses majority voting within each window.
        """
        window_labels = np.zeros(n_windows, dtype=np.int32)

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size

            if end_idx <= len(labels):
                # Majority vote
                window_label = np.bincount(labels[start_idx:end_idx]).argmax()
                window_labels[i] = window_label
            else:
                # Use last available label
                window_labels[i] = labels[-1] if len(labels) > 0 else 0

        return window_labels


class StreamingDataset:
    """
    Memory-efficient dataset that streams from HDF5.
    Avoids loading all data into memory.
    """

    def __init__(
        self,
        hdf5_path: str,
        indices: Optional[np.ndarray] = None,
        transform=None,
    ):
        """
        Initialize streaming dataset.

        Args:
            hdf5_path: Path to HDF5 file
            indices: Optional subset of indices to use
            transform: Optional transform function
        """
        self.hdf5_path = hdf5_path
        self.transform = transform

        # Open file to get length
        with h5py.File(hdf5_path, 'r') as f:
            self.total_length = len(f['windows'])
            self.has_labels = 'labels' in f

        self.indices = indices if indices is not None else np.arange(self.total_length)
        self.length = len(self.indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get single item by streaming from HDF5."""
        real_idx = self.indices[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            window = f['windows'][real_idx]

            if self.has_labels:
                label = f['labels'][real_idx]
            else:
                label = -1  # No label

        if self.transform:
            window = self.transform(window)

        return window, label


def create_stratified_splits(
    hdf5_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train/val/test splits.

    Args:
        hdf5_path: Path to HDF5 dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility

    Returns:
        train_indices, val_indices, test_indices
    """
    from sklearn.model_selection import train_test_split

    np.random.seed(random_seed)

    with h5py.File(hdf5_path, 'r') as f:
        n_samples = len(f['windows'])

        if 'labels' in f:
            labels = f['labels'][:]
        else:
            labels = np.zeros(n_samples)

    indices = np.arange(n_samples)

    # Split train and temp (val+test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed,
    )

    # Split temp into val and test
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_seed,
    )

    logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return train_idx, val_idx, test_idx
