"""
PyTorch datasets for accelerometry data.

Provides datasets for forecasting pretext tasks and supervised learning.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.io import load_windows_hdf5, load_windows_zarr
from ..utils.time_features import build_time_features_for_windows


class AccelerometryForecastDataset(Dataset):
    """
    Dataset for forecasting pretext tasks.

    Yields (context_window, future_window, exog_features) for self-supervised learning.
    """

    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        context_length: int = 819,
        forecast_length: int = 200,
        include_time_features: bool = True,
        include_static_features: bool = False,
        static_features_path: Optional[Union[str, Path]] = None,
        storage_format: str = "hdf5",
        transform: Optional[callable] = None,
    ):
        """
        Args:
            data_paths: Path(s) to windowed data file(s) (.h5 or .zarr)
            context_length: Length of context window in samples
            forecast_length: Length of forecast horizon in samples
            include_time_features: Include temporal features (hour, day, etc.)
            include_static_features: Include participant static features
            static_features_path: Path to CSV with static features
            storage_format: "hdf5" or "zarr"
            transform: Optional transform function applied to windows

        Examples:
            >>> dataset = AccelerometryForecastDataset(
            ...     'data/participant_123/windows.h5',
            ...     context_length=819,
            ...     forecast_length=200
            ... )
            >>> x_context, y_future, exog = dataset[0]
            >>> x_context.shape
            torch.Size([3, 819])
            >>> y_future.shape
            torch.Size([3, 200])
        """
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.include_time_features = include_time_features
        self.include_static_features = include_static_features
        self.storage_format = storage_format
        self.transform = transform

        # Handle single path or list of paths
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.data_paths = [Path(p) for p in data_paths]

        # Load all data
        self._load_data()

        # Load static features if requested
        self.static_features = None
        if include_static_features and static_features_path is not None:
            self._load_static_features(static_features_path)

    def _load_data(self):
        """Load windowed data from all files."""
        all_windows = []
        all_timestamps_start = []
        all_timestamps_end = []
        all_participant_ids = []

        for path in self.data_paths:
            # Load based on format
            if self.storage_format == "hdf5":
                data = load_windows_hdf5(path, load_timestamps=True, load_gap_flags=False)
            elif self.storage_format == "zarr":
                data = load_windows_zarr(path, load_timestamps=True, load_gap_flags=False)
            else:
                raise ValueError(f"Unknown storage format: {self.storage_format}")

            # Extract participant ID from path (e.g., participant_123/windows.h5)
            participant_id = path.parent.name if path.parent.name.startswith('participant') else path.stem

            all_windows.append(data['windows'])
            all_timestamps_start.append(data['timestamps_start'])
            all_timestamps_end.append(data['timestamps_end'])
            all_participant_ids.extend([participant_id] * len(data['windows']))

        # Concatenate all data
        self.windows = np.concatenate(all_windows, axis=0)
        self.timestamps_start = np.concatenate(all_timestamps_start, axis=0)
        self.timestamps_end = np.concatenate(all_timestamps_end, axis=0)
        self.participant_ids = np.array(all_participant_ids)

        # Verify windows are large enough for context + forecast
        required_length = self.context_length + self.forecast_length
        if self.windows.shape[2] < required_length:
            raise ValueError(
                f"Window length {self.windows.shape[2]} is too short for "
                f"context_length={self.context_length} + forecast_length={self.forecast_length}"
            )

    def _load_static_features(self, path: Union[str, Path]):
        """Load static features from CSV."""
        df = pd.read_csv(path)
        # Assume first column is participant_id
        self.static_features = df.set_index(df.columns[0]).to_dict('index')

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Tuple of (context_window, future_window, exog_dict) where:
            - context_window: (C, context_length)
            - future_window: (C, forecast_length)
            - exog_dict: Dictionary of exogenous features
        """
        # Get full window
        window = self.windows[idx]  # Shape: (C, T)

        # Split into context and forecast
        x_context = window[:, :self.context_length]
        y_future = window[:, self.context_length:self.context_length + self.forecast_length]

        # Apply transform if provided
        if self.transform is not None:
            x_context = self.transform(x_context)
            y_future = self.transform(y_future)

        # Convert to tensors
        x_context = torch.from_numpy(x_context).float()
        y_future = torch.from_numpy(y_future).float()

        # Build exogenous features dictionary
        exog_dict = {}

        if self.include_time_features:
            # Get timestamp for this window (use start time)
            timestamp = self.timestamps_start[idx]
            time_features = build_time_features_for_windows(
                np.array([timestamp]),
                use_midpoint=False
            )

            # Convert to tensors and extract first element (since we have one window)
            for key, value in time_features.items():
                exog_dict[f"time_{key}"] = torch.tensor(value[0], dtype=torch.float32)

        if self.include_static_features and self.static_features is not None:
            participant_id = self.participant_ids[idx]
            if participant_id in self.static_features:
                static_feats = self.static_features[participant_id]
                for key, value in static_feats.items():
                    exog_dict[f"static_{key}"] = torch.tensor(value, dtype=torch.float32)

        return x_context, y_future, exog_dict


class AccelerometryLabelDataset(Dataset):
    """
    Dataset for supervised learning tasks (activity recognition, sleep classification, etc.).

    Yields (window, label_dict) for classification or regression.
    """

    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        labels_path: Union[str, Path],
        window_length: int = 819,
        include_time_features: bool = True,
        include_static_features: bool = False,
        static_features_path: Optional[Union[str, Path]] = None,
        storage_format: str = "hdf5",
        transform: Optional[callable] = None,
        label_columns: Optional[List[str]] = None
    ):
        """
        Args:
            data_paths: Path(s) to windowed data file(s)
            labels_path: Path to CSV with labels (must have matching indices)
            window_length: Length of window to use
            include_time_features: Include temporal features
            include_static_features: Include static features
            static_features_path: Path to static features CSV
            storage_format: "hdf5" or "zarr"
            transform: Optional transform function
            label_columns: List of label columns to use (None = all except ID/timestamp)

        Examples:
            >>> dataset = AccelerometryLabelDataset(
            ...     'data/participant_123/windows.h5',
            ...     'data/labels.csv',
            ...     window_length=819
            ... )
            >>> x_window, labels = dataset[0]
            >>> x_window.shape
            torch.Size([3, 819])
        """
        self.window_length = window_length
        self.include_time_features = include_time_features
        self.include_static_features = include_static_features
        self.storage_format = storage_format
        self.transform = transform

        # Handle single path or list of paths
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.data_paths = [Path(p) for p in data_paths]

        # Load data
        self._load_data()

        # Load labels
        self._load_labels(labels_path, label_columns)

        # Load static features if requested
        self.static_features = None
        if include_static_features and static_features_path is not None:
            self._load_static_features(static_features_path)

    def _load_data(self):
        """Load windowed data."""
        all_windows = []
        all_timestamps_start = []
        all_participant_ids = []

        for path in self.data_paths:
            if self.storage_format == "hdf5":
                data = load_windows_hdf5(path, load_timestamps=True)
            elif self.storage_format == "zarr":
                data = load_windows_zarr(path, load_timestamps=True)
            else:
                raise ValueError(f"Unknown storage format: {self.storage_format}")

            participant_id = path.parent.name if path.parent.name.startswith('participant') else path.stem

            all_windows.append(data['windows'])
            all_timestamps_start.append(data['timestamps_start'])
            all_participant_ids.extend([participant_id] * len(data['windows']))

        self.windows = np.concatenate(all_windows, axis=0)
        self.timestamps_start = np.concatenate(all_timestamps_start, axis=0)
        self.participant_ids = np.array(all_participant_ids)

    def _load_labels(self, labels_path: Union[str, Path], label_columns: Optional[List[str]]):
        """Load labels from CSV."""
        labels_df = pd.read_csv(labels_path)

        # If label_columns not specified, use all numeric columns
        if label_columns is None:
            # Exclude common ID/timestamp columns
            exclude_cols = ['participant_id', 'window_id', 'timestamp', 'timestamp_start', 'timestamp_end']
            label_columns = [col for col in labels_df.columns
                           if col not in exclude_cols and pd.api.types.is_numeric_dtype(labels_df[col])]

        self.label_columns = label_columns
        self.labels = labels_df[label_columns].values

        # Verify length matches
        if len(self.labels) != len(self.windows):
            raise ValueError(
                f"Number of labels ({len(self.labels)}) doesn't match "
                f"number of windows ({len(self.windows)})"
            )

    def _load_static_features(self, path: Union[str, Path]):
        """Load static features from CSV."""
        df = pd.read_csv(path)
        self.static_features = df.set_index(df.columns[0]).to_dict('index')

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Tuple of (window, label_dict) where:
            - window: (C, window_length)
            - label_dict: Dictionary of labels and features
        """
        # Get window
        window = self.windows[idx][:, :self.window_length]

        # Apply transform
        if self.transform is not None:
            window = self.transform(window)

        window = torch.from_numpy(window).float()

        # Build label dictionary
        label_dict = {}

        # Add labels
        for i, col_name in enumerate(self.label_columns):
            label_dict[col_name] = torch.tensor(self.labels[idx, i], dtype=torch.float32)

        # Add time features if requested
        if self.include_time_features:
            timestamp = self.timestamps_start[idx]
            time_features = build_time_features_for_windows(
                np.array([timestamp]),
                use_midpoint=False
            )
            for key, value in time_features.items():
                label_dict[f"time_{key}"] = torch.tensor(value[0], dtype=torch.float32)

        # Add static features if requested
        if self.include_static_features and self.static_features is not None:
            participant_id = self.participant_ids[idx]
            if participant_id in self.static_features:
                static_feats = self.static_features[participant_id]
                for key, value in static_feats.items():
                    label_dict[f"static_{key}"] = torch.tensor(value, dtype=torch.float32)

        return window, label_dict
