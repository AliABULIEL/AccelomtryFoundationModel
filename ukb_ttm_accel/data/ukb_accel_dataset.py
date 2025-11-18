# FILE: ukb_ttm_accel/data/ukb_accel_dataset.py

"""
PyTorch Dataset for UK Biobank accelerometry data.

Supports:
- Loading from multiple file formats (cwa, csv, parquet)
- HDF5 caching for faster repeated access
- Train/val/test splitting
- Clinical label and metadata integration
- Windowing with configurable overlap
- Per-window normalization
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Dict, Any
import h5py
from tqdm import tqdm

from ukb_ttm_accel.data.windowing import (
    resample_and_calibrate,
    create_windows,
    zscore_normalize_windows,
)
from ukb_ttm_accel.data.clinical_metadata import (
    load_clinical_data,
    build_label_and_metadata,
    create_train_val_test_split,
)


class UKBAccelDataset(Dataset):
    """
    PyTorch Dataset for UK Biobank accelerometry with clinical metadata.

    Features:
    - Lazy loading with HDF5 caching
    - Train/val/test splitting
    - Window extraction with overlap control
    - Per-window normalization
    - Clinical label and metadata integration
    - Support for max_windows_per_participant (for Colab constraints)

    Example:
        >>> dataset = UKBAccelDataset(
        ...     accel_file_paths=["p1.cwa", "p2.cwa"],
        ...     clinical_csv_path="clinical.csv",
        ...     split="train",
        ...     window_seconds=10,
        ...     sampling_rate_hz=50,
        ...     cache_hdf5_path="cache.h5"
        ... )
        >>> sample = dataset[0]
        >>> print(sample['signal'].shape)
        torch.Size([500, 3])
    """

    def __init__(
        self,
        accel_file_paths: List[str],
        clinical_csv_path: str,
        split: str,
        window_seconds: int = 10,
        sampling_rate_hz: int = 50,
        normalize_per_window: bool = True,
        max_windows_per_participant: Optional[int] = None,
        label_type: str = "bmi_class",
        metadata_columns: Optional[List[str]] = None,
        cache_hdf5_path: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        overlap_fraction: float = 0.0,
        source_type: str = "cwa",
        random_state: int = 42,
    ):
        """
        Initialize UKB Accelerometry Dataset.

        Args:
            accel_file_paths: List of paths to accelerometry files
            clinical_csv_path: Path to clinical metadata CSV
            split: Dataset split ("train", "val", or "test")
            window_seconds: Window length in seconds
            sampling_rate_hz: Target sampling rate in Hz
            normalize_per_window: Whether to apply z-score normalization per window
            max_windows_per_participant: Max windows to use per participant (None = all)
            label_type: Type of clinical label to use
            metadata_columns: List of metadata columns to extract
            cache_hdf5_path: Path to HDF5 cache file (None = no caching)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            overlap_fraction: Fraction of overlap between windows (0.0 to 0.99)
            source_type: Type of accelerometry files ("cwa", "csv", "parquet")
            random_state: Random seed for splits
        """
        super().__init__()

        self.accel_file_paths = [Path(p) for p in accel_file_paths]
        self.clinical_csv_path = clinical_csv_path
        self.split = split
        self.window_seconds = window_seconds
        self.sampling_rate_hz = sampling_rate_hz
        self.normalize_per_window = normalize_per_window
        self.max_windows_per_participant = max_windows_per_participant
        self.label_type = label_type
        self.metadata_columns = metadata_columns or []
        self.cache_hdf5_path = cache_hdf5_path
        self.overlap_fraction = overlap_fraction
        self.source_type = source_type

        # Validate split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got: {split}")

        # Load clinical data
        self.clinical_df = load_clinical_data(clinical_csv_path)

        # Extract participant IDs from file paths
        # Assumption: filename contains participant ID (eid)
        # Example: "1001234.cwa" or "participant_1001234.csv"
        self.participant_ids = self._extract_participant_ids()

        # Create train/val/test splits
        splits = create_train_val_test_split(
            self.participant_ids,
            self.clinical_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_column=None,  # Can enable stratification if needed
            random_state=random_state,
        )

        # Filter to current split
        self.split_participant_ids = splits[split]
        self.split_file_paths = [
            fp for fp, pid in zip(self.accel_file_paths, self.participant_ids)
            if pid in self.split_participant_ids
        ]

        # Build index mapping (global_idx -> (participant_idx, window_idx))
        self.index_map = []
        self.participant_data = {}  # Cache for loaded data

        # Check if we should use cache
        if cache_hdf5_path and Path(cache_hdf5_path).exists():
            self._load_from_cache()
        else:
            self._build_index()

            # Save to cache if path provided
            if cache_hdf5_path:
                self._save_to_cache()

    def _extract_participant_ids(self) -> List[int]:
        """
        Extract participant IDs from file paths.

        Assumption: Filename contains numeric participant ID.
        Example: "1001234.cwa" -> 1001234

        Returns:
            List of participant IDs
        """
        participant_ids = []

        for file_path in self.accel_file_paths:
            # Extract numeric ID from filename
            filename = file_path.stem  # Filename without extension

            # Try to extract first numeric sequence
            import re
            match = re.search(r'\d+', filename)

            if match:
                participant_id = int(match.group())
                participant_ids.append(participant_id)
            else:
                raise ValueError(
                    f"Could not extract participant ID from filename: {file_path.name}"
                )

        return participant_ids

    def _build_index(self):
        """
        Build index mapping from global index to (participant, window) pairs.

        Also processes and caches accelerometry data.
        """
        print(f"Building index for {self.split} split ({len(self.split_file_paths)} participants)...")

        for p_idx, file_path in enumerate(tqdm(self.split_file_paths, desc=f"Processing {self.split}")):
            participant_id = self.split_participant_ids[p_idx]

            try:
                # Load and process accelerometry data
                accel_df = resample_and_calibrate(
                    file_path=str(file_path),
                    source_type=self.source_type,
                    target_sampling_rate_hz=self.sampling_rate_hz,
                    verbose=False,
                )

                # Create windows
                windows, time_of_day = create_windows(
                    accel_df=accel_df,
                    window_seconds=self.window_seconds,
                    sampling_rate_hz=self.sampling_rate_hz,
                    overlap_fraction=self.overlap_fraction,
                    extract_time_of_day=True,
                )

                # Normalize if requested
                if self.normalize_per_window:
                    windows = zscore_normalize_windows(windows)

                # Limit windows per participant if specified
                if self.max_windows_per_participant is not None:
                    num_windows = min(len(windows), self.max_windows_per_participant)
                    windows = windows[:num_windows]
                    time_of_day = time_of_day[:num_windows]
                else:
                    num_windows = len(windows)

                # Get label and metadata
                label, metadata = build_label_and_metadata(
                    eid=participant_id,
                    clinical_df=self.clinical_df,
                    label_type=self.label_type,
                    metadata_columns=self.metadata_columns,
                )

                # Cache participant data
                self.participant_data[p_idx] = {
                    'windows': windows,
                    'time_of_day': time_of_day,
                    'label': label,
                    'metadata': metadata,
                    'participant_id': participant_id,
                }

                # Update index map
                for w_idx in range(num_windows):
                    self.index_map.append((p_idx, w_idx))

            except Exception as e:
                print(f"Warning: Failed to process {file_path.name}: {e}")
                continue

        print(f"Built index with {len(self.index_map)} windows from {len(self.participant_data)} participants")

    def _save_to_cache(self):
        """Save processed data to HDF5 cache."""
        print(f"Saving to cache: {self.cache_hdf5_path}")

        cache_path = Path(self.cache_hdf5_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache_path, 'w') as f:
            # Save metadata
            f.attrs['split'] = self.split
            f.attrs['window_seconds'] = self.window_seconds
            f.attrs['sampling_rate_hz'] = self.sampling_rate_hz
            f.attrs['label_type'] = self.label_type

            # Create group for this split
            split_group = f.create_group(self.split)

            # Save each participant's data
            for p_idx, data in self.participant_data.items():
                p_group = split_group.create_group(f"participant_{p_idx}")

                p_group.create_dataset('windows', data=data['windows'], compression='gzip')
                p_group.create_dataset('time_of_day', data=data['time_of_day'])
                p_group.create_dataset('metadata', data=data['metadata'])
                p_group.attrs['label'] = data['label']
                p_group.attrs['participant_id'] = data['participant_id']

            # Save index map
            index_map_array = np.array(self.index_map, dtype=np.int32)
            split_group.create_dataset('index_map', data=index_map_array)

        print(f"Cache saved successfully")

    def _load_from_cache(self):
        """Load processed data from HDF5 cache."""
        print(f"Loading from cache: {self.cache_hdf5_path}")

        with h5py.File(self.cache_hdf5_path, 'r') as f:
            if self.split not in f:
                raise ValueError(f"Split '{self.split}' not found in cache")

            split_group = f[self.split]

            # Load index map
            self.index_map = list(map(tuple, split_group['index_map'][...]))

            # Load participant data
            for p_key in split_group.keys():
                if p_key == 'index_map':
                    continue

                p_idx = int(p_key.split('_')[1])
                p_group = split_group[p_key]

                self.participant_data[p_idx] = {
                    'windows': p_group['windows'][...],
                    'time_of_day': p_group['time_of_day'][...],
                    'metadata': p_group['metadata'][...],
                    'label': p_group.attrs['label'],
                    'participant_id': p_group.attrs['participant_id'],
                }

        print(f"Loaded {len(self.index_map)} windows from cache")

    def __len__(self) -> int:
        """Return total number of windows in dataset."""
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single window sample.

        Args:
            idx: Global window index

        Returns:
            Dictionary with keys:
            - signal: FloatTensor[window_length, num_channels]
            - label: LongTensor[] or FloatTensor[] depending on label type
            - participant_id: LongTensor[]
            - metadata: FloatTensor[num_metadata_features]
            - time_of_day: FloatTensor[]
        """
        p_idx, w_idx = self.index_map[idx]
        data = self.participant_data[p_idx]

        # Get window signal
        signal = torch.from_numpy(data['windows'][w_idx]).float()

        # Get label
        label = data['label']
        if self.label_type in ['bmi_class', 'age_group', 'sex', 'hypertension', 'diabetes']:
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)

        # Get metadata
        metadata = torch.from_numpy(data['metadata']).float()

        # Get time of day
        time_of_day = torch.tensor(data['time_of_day'][w_idx], dtype=torch.float32)

        # Get participant ID
        participant_id = torch.tensor(data['participant_id'], dtype=torch.long)

        return {
            'signal': signal,
            'label': label,
            'participant_id': participant_id,
            'metadata': metadata,
            'time_of_day': time_of_day,
        }
