"""
PyTorch datasets for NHANES accelerometry data.

Provides:
- NHANES80HzForecastDataset: 80 Hz forecasting (context: 655, forecast: 164)
- NHANES80HzLabelDataset: 80 Hz supervised learning
- NHANESMinuteDataset: Minute-level forecasting (context: 1024, forecast: 96)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py


class NHANES80HzForecastDataset(Dataset):
    """
    PyTorch dataset for NHANES 80 Hz forecasting.

    Each sample contains:
    - x: Context window (3, 655)
    - y: Future window (3, 164)
    - exog: Exogenous features dict
    """

    def __init__(
        self,
        data_dirs: Union[str, Path, List[Union[str, Path]]],
        context_length: int = 655,
        forecast_length: int = 164,
        include_exog: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_dirs: Directory or list of directories containing windows.h5 files
            context_length: Context window length (default: 655 samples)
            forecast_length: Forecast window length (default: 164 samples)
            include_exog: Include exogenous features
        """
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [Path(data_dirs)]
        else:
            data_dirs = [Path(d) for d in data_dirs]

        self.context_length = context_length
        self.forecast_length = forecast_length
        self.include_exog = include_exog

        # Find all participant HDF5 files
        self.participant_files = []
        for data_dir in data_dirs:
            participant_files = list(data_dir.glob("*/windows.h5"))
            self.participant_files.extend(participant_files)

        if not self.participant_files:
            raise ValueError(f"No windows.h5 files found in {data_dirs}")

        # Build index: (file_idx, window_idx)
        self.index = []
        for file_idx, h5_path in enumerate(self.participant_files):
            with h5py.File(h5_path, 'r') as f:
                num_windows = f['windows'].shape[0]

                # Each window i can serve as context for window i+1's forecast
                # We need at least 2 windows to create one sample
                if num_windows < 2:
                    continue

                # For each valid context window
                for win_idx in range(num_windows - 1):
                    self.index.append((file_idx, win_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        file_idx, win_idx = self.index[idx]
        h5_path = self.participant_files[file_idx]

        with h5py.File(h5_path, 'r') as f:
            # Get context window (3, 655)
            x_context = f['windows'][win_idx]  # Shape: (3, win_n)

            # Get next window for forecast
            next_window = f['windows'][win_idx + 1]  # Shape: (3, win_n)

            # Extract first forecast_length samples as future
            y_future = next_window[:, :self.forecast_length]  # Shape: (3, forecast_length)

            # Load metadata
            participant_id = f.attrs.get('participant_id', 'unknown')
            cycle = f.attrs.get('cycle', 'unknown')
            fs = f.attrs.get('fs', 80)

        # Convert to torch tensors
        x = torch.from_numpy(x_context).float()  # (3, 655)
        y = torch.from_numpy(y_future).float()  # (3, 164)

        # Exogenous features
        exog = {}
        if self.include_exog:
            exog['participant_id'] = participant_id
            exog['cycle'] = cycle
            exog['device_location'] = 'hip'
            exog['fs'] = fs

        return x, y, exog


class NHANES80HzLabelDataset(Dataset):
    """
    PyTorch dataset for NHANES 80 Hz supervised learning.

    Each sample contains:
    - x: Window (3, 655)
    - labels: Dict of labels for this window
    """

    def __init__(
        self,
        data_dirs: Union[str, Path, List[Union[str, Path]]],
        labels_csv: Union[str, Path],
        window_length: int = 655
    ):
        """
        Initialize dataset.

        Args:
            data_dirs: Directory or list of directories containing windows.h5 files
            labels_csv: Path to CSV with columns: participant_id, win_idx, label columns...
            window_length: Window length (default: 655)
        """
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [Path(data_dirs)]
        else:
            data_dirs = [Path(d) for d in data_dirs]

        self.window_length = window_length

        # Find all participant files
        self.participant_files = []
        self.participant_id_to_file = {}

        for data_dir in data_dirs:
            for h5_path in data_dir.glob("*/windows.h5"):
                self.participant_files.append(h5_path)

                # Extract participant ID from path or HDF5 attributes
                with h5py.File(h5_path, 'r') as f:
                    pid = f.attrs.get('participant_id', h5_path.parent.name)
                    self.participant_id_to_file[str(pid)] = h5_path

        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df['participant_id'] = self.labels_df['participant_id'].astype(str)

        # Identify label columns (all except participant_id, win_idx)
        self.label_columns = [col for col in self.labels_df.columns
                              if col not in ['participant_id', 'win_idx']]

        # Build index
        self.index = []
        for _, row in self.labels_df.iterrows():
            pid = str(row['participant_id'])
            win_idx = int(row['win_idx'])

            if pid in self.participant_id_to_file:
                h5_path = self.participant_id_to_file[pid]
                self.index.append((h5_path, win_idx, row[self.label_columns].to_dict()))

        if not self.index:
            raise ValueError("No matching windows found for labels")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        h5_path, win_idx, labels_dict = self.index[idx]

        with h5py.File(h5_path, 'r') as f:
            # Get window
            x_window = f['windows'][win_idx]  # Shape: (3, win_n)

        # Convert to torch tensor
        x = torch.from_numpy(x_window).float()

        # Convert labels to tensors
        labels = {}
        for key, value in labels_dict.items():
            if pd.notna(value):
                labels[key] = torch.tensor(value).float()
            else:
                labels[key] = torch.tensor(float('nan')).float()

        return x, labels


class NHANESMinuteDataset(Dataset):
    """
    PyTorch dataset for NHANES minute-level forecasting.

    Each sample contains:
    - x: Context window (1, 1024)
    - y: Future window (1, 96)
    - meta: Dict with participant_id, timestamp
    """

    def __init__(
        self,
        X_context_path: Union[str, Path],
        Y_future_path: Union[str, Path],
        meta_csv_path: Union[str, Path],
        context_length: int = 1024,
        prediction_length: int = 96
    ):
        """
        Initialize dataset.

        Args:
            X_context_path: Path to X_context.npy file (B, 1, 1024)
            Y_future_path: Path to Y_future.npy file (B, 1, 96)
            meta_csv_path: Path to windows_meta.csv
            context_length: Context length (default: 1024 minutes)
            prediction_length: Prediction length (default: 96 minutes)
        """
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Load arrays
        self.X_context = np.load(X_context_path).astype(np.float32)
        self.Y_future = np.load(Y_future_path).astype(np.float32)

        # Load metadata
        self.meta_df = pd.read_csv(meta_csv_path)

        # Validate shapes
        assert self.X_context.shape[0] == self.Y_future.shape[0], \
            "X and Y must have same number of samples"
        assert self.X_context.shape == (self.X_context.shape[0], 1, context_length), \
            f"Expected X shape (B, 1, {context_length}), got {self.X_context.shape}"
        assert self.Y_future.shape == (self.Y_future.shape[0], 1, prediction_length), \
            f"Expected Y shape (B, 1, {prediction_length}), got {self.Y_future.shape}"
        assert len(self.meta_df) == self.X_context.shape[0], \
            "Metadata must have same length as arrays"

    def __len__(self) -> int:
        return self.X_context.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Get arrays
        x = torch.from_numpy(self.X_context[idx]).float()  # (1, 1024)
        y = torch.from_numpy(self.Y_future[idx]).float()  # (1, 96)

        # Get metadata
        meta = {
            'participant_id': self.meta_df.iloc[idx]['participant_id'],
            'start_timestamp': self.meta_df.iloc[idx]['start_timestamp']
        }

        return x, y, meta


class NHANESMinuteFromDataFrame(Dataset):
    """
    Alternative minute-level dataset that creates windows on-the-fly from DataFrame.

    Useful for dynamic window creation without pre-computing.
    """

    def __init__(
        self,
        minute_df: pd.DataFrame,
        context_length: int = 1024,
        prediction_length: int = 96,
        min_wear_fraction: float = 0.8
    ):
        """
        Initialize dataset from DataFrame.

        Args:
            minute_df: Normalized minute DataFrame with columns:
                        participant_id, timestamp, axis_summaries, wear_flag, cycle
            context_length: Context window size (default: 1024 minutes)
            prediction_length: Prediction window size (default: 96 minutes)
            min_wear_fraction: Minimum wear fraction in context (default: 0.8)
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.min_wear_fraction = min_wear_fraction

        total_length = context_length + prediction_length

        # Build index of valid windows
        self.index = []

        participants = minute_df['participant_id'].unique()
        for pid in participants:
            participant_data = minute_df[minute_df['participant_id'] == pid].copy()
            participant_data = participant_data.sort_values('timestamp').reset_index(drop=True)

            if len(participant_data) < total_length:
                continue

            # Extract arrays
            values = participant_data['axis_summaries'].values
            wear = participant_data['wear_flag'].values
            timestamps = participant_data['timestamp'].values

            # Create sliding windows
            num_windows = len(values) - total_length + 1

            for i in range(num_windows):
                context_wear = wear[i:i+context_length]

                # Check wear requirement
                wear_fraction = context_wear.sum() / context_length
                if wear_fraction >= min_wear_fraction:
                    self.index.append({
                        'participant_id': pid,
                        'start_idx': i,
                        'values': values,
                        'timestamp': timestamps[i]
                    })

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        item = self.index[idx]

        start_idx = item['start_idx']
        values = item['values']

        # Extract context and future
        context_values = values[start_idx:start_idx+self.context_length]
        future_values = values[start_idx+self.context_length:start_idx+self.context_length+self.prediction_length]

        # Convert to torch tensors
        x = torch.from_numpy(context_values).float().unsqueeze(0)  # (1, 1024)
        y = torch.from_numpy(future_values).float().unsqueeze(0)  # (1, 96)

        # Metadata
        meta = {
            'participant_id': item['participant_id'],
            'start_timestamp': str(item['timestamp'])
        }

        return x, y, meta
