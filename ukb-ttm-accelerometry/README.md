# UK Biobank TTM Accelerometry Toolkit

Production-ready toolkit for processing UK Biobank accelerometry data and integrating with IBM Granite TTM (Tiny Time Mixer) foundation models.

## Features

- **Robust Data Reading**: Handles Axivity .cwa files with automatic handling of:
  - Daylight saving time gaps
  - Non-monotonic timestamps
  - Duplicate packets
  - Missing samples with gap detection

- **Precise Windowing**: 8.192s windows (819 samples @ 100 Hz) with configurable overlap
  - Strided windowing for efficiency
  - Gap filtering to exclude low-quality windows
  - Multiple padding strategies

- **Instance Normalization**: RevIN-style per-window standardization
  - Per-channel z-score normalization
  - Invertible for reconstruction
  - Robust variant for outlier handling

- **Efficient Storage**: HDF5 and Zarr formats with compression
  - Optimized chunking for fast random access
  - ZSTD/Blosc compression
  - Metadata preservation

- **PyTorch Datasets**: Ready-to-use datasets for:
  - Forecasting pretext tasks
  - Supervised learning (activity, sleep classification)
  - Time feature extraction
  - Static covariate integration

- **Reproducibility**: Comprehensive seed management across libraries

## Installation

> **Important**: The IBM tsfm package must be installed from GitHub, not PyPI.
> See [INSTALL.md](INSTALL.md) for detailed installation instructions and troubleshooting.

### Quick Install (Recommended)

```bash
# Automated installation (Linux/macOS)
chmod +x install.sh
./install.sh
```

### Standard Installation

```bash
# Clone repository
cd ukb-ttm-accelerometry

# Install IBM tsfm from GitHub (REQUIRED FIRST)
pip install git+https://github.com/IBM/tsfm.git

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python3 test_basic.py
```

### One-Line Installation

```bash
pip install git+https://github.com/IBM/tsfm.git && pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1
- **IBM tsfm** (Granite Time Series Foundation Model) - must install from GitHub
- accelerometer ≥ 1.1 (for .cwa reading)
- See `requirements.txt` for full list

### Google Colab Setup

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install IBM tsfm (required for Granite TTM models)
!pip install git+https://github.com/IBM/tsfm.git

# Clone and install this package
!git clone https://github.com/yourusername/ukb-ttm-accelerometry.git
%cd ukb-ttm-accelerometry
!pip install -r requirements.txt

# Set high precision matmul for better performance
import torch
torch.set_float32_matmul_precision("high")

# Verify installation
!python test_basic.py
```

### Minimal Installation (without TTM models)

If you only need data processing without the Granite TTM models:

```bash
# Skip tsfm installation and install other dependencies
pip install numpy pandas scipy numba h5py zarr torch accelerometer pyyaml tqdm scikit-learn

# The toolkit will work for data processing, windowing, and preprocessing
# TTM model features will not be available
```

## Getting Data

### Option 1: UK Biobank Data (Real Data - Requires Access)

UK Biobank data requires approved access. See [docs/UKB_DATA_ACCESS.md](docs/UKB_DATA_ACCESS.md) for complete details.

**Quick Summary:**
1. Apply for access: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
2. Wait for approval (4-8 weeks)
3. Download your `.ukbkey` file
4. Use our download script:

```bash
python scripts/download_ukb_data.py \
    --key-file your.ukbkey \
    --dataset-id YOUR_APPLICATION_ID \
    --output-dir ./data/ukb_raw
```

### Option 2: Synthetic Demo Data (Testing - No Access Required)

Generate realistic synthetic data for development and testing:

```bash
# Quick test (2 participants, 1 day)
python scripts/generate_demo_data.py \
    --n-participants 2 \
    --duration-days 1 \
    --output-dir ./data/demo

# Realistic test (50 participants, 7 days)
python scripts/generate_demo_data.py \
    --n-participants 50 \
    --duration-days 7 \
    --output-dir ./data/demo
```

**Features of synthetic data:**
- ✅ Realistic activity patterns (walking, sleep, sedentary)
- ✅ Circadian rhythms
- ✅ Random gaps simulating non-wear
- ✅ Compatible with all processing scripts
- ⚠️ For testing only - use real data for research

## Quick Start

### 1. Prepare Raw Data

Convert .cwa files (or synthetic data) to windowed HDF5 format:

```bash
python scripts/prepare_ukb.py \
    --input /path/to/cwa/files/ \
    --outdir ./processed_data \
    --win-sec 8.192 \
    --hop-sec 4.096 \
    --max-gap-ratio 0.1 \
    --min-wear-hours 24 \
    --summary-file processing_summary.csv 
```

**Parameters:**
- `--win-sec`: Window duration in seconds (default: 8.192 for 819 samples @ 100Hz)
- `--hop-sec`: Hop duration for overlap (default: 4.096 for 50% overlap)
- `--max-gap-ratio`: Maximum ratio of gap samples per window (default: 0.1 = 10%)
- `--rounding`: Rounding mode - `floor`, `nearest`, or `ceil` (default: `floor`)
- `--storage-format`: Output format - `hdf5` or `zarr` (default: `hdf5`)

**Output Structure:**
```
processed_data/
  participant_001/
    windows.h5
  participant_002/
    windows.h5
  ...
```

### 2. Create Train/Val/Test Splits

```bash
python scripts/make_splits.py \
    --data-dir ./processed_data \
    --output-dir ./splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --stratify-file demographics.csv  # Optional
```

**Output:**
```
splits/
  splits.json              # All splits in one file
  train.txt               # Training participant IDs
  val.txt                 # Validation participant IDs
  test.txt                # Test participant IDs
  split_metadata.json     # Metadata and statistics
```

### 3. Load Data in PyTorch

#### For Forecasting (Pretext Task)

```python
from src.dataio.datasets import AccelerometryForecastDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = AccelerometryForecastDataset(
    data_paths='processed_data/participant_001/windows.h5',
    context_length=819,      # 8.192s context
    forecast_length=200,     # ~2s forecast
    include_time_features=True,
    storage_format='hdf5'
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate
for x_context, y_future, exog_features in loader:
    # x_context: (B, 3, 819) - input accelerometry
    # y_future: (B, 3, 200) - target to forecast
    # exog_features: dict with time features
    print(x_context.shape, y_future.shape)
    break
```

#### For Supervised Learning

```python
from src.dataio.datasets import AccelerometryLabelDataset

dataset = AccelerometryLabelDataset(
    data_paths='processed_data/participant_001/windows.h5',
    labels_path='labels.csv',
    window_length=819,
    include_time_features=True
)

loader = DataLoader(dataset, batch_size=256, shuffle=True)

for x_window, label_dict in loader:
    # x_window: (B, 3, 819)
    # label_dict: {'activity': tensor, 'sleep': tensor, ...}
    print(x_window.shape, label_dict.keys())
    break
```

### 4. Preprocessing

#### Instance Standardization (RevIN-style)

```python
import torch
from src.dataio.preprocess import instance_standardize, inverse_standardize

# Standardize windows
x = torch.randn(32, 3, 819)  # (batch, channels, time)
x_norm, (mean, std) = instance_standardize(x)

# x_norm has zero mean and unit variance per window per channel
print(x_norm.mean(dim=-1))  # ~ [0, 0, 0]
print(x_norm.std(dim=-1))   # ~ [1, 1, 1]

# Inverse transform (for reconstruction)
x_reconstructed = inverse_standardize(x_norm, mean, std)
assert torch.allclose(x, x_reconstructed)
```

#### Robust Standardization (for outliers)

```python
from src.dataio.preprocess import robust_standardize, inverse_robust_standardize

x_norm, (median, iqr) = robust_standardize(x)
x_reconstructed = inverse_robust_standardize(x_norm, median, iqr)
```

### 5. Time Features

```python
import pandas as pd
from src.utils.time_features import build_time_features_for_windows

# Window timestamps
timestamps = pd.date_range('2020-01-01 08:00', periods=100, freq='8.192s')

# Extract features
features = build_time_features_for_windows(
    timestamps.values,
    use_midpoint=True,
    include_holiday=True
)

# Available features:
# - hour_sin, hour_cos (24-hour cycle)
# - minute_sin, minute_cos
# - day_of_week_sin, day_of_week_cos (7-day cycle)
# - month_sin, month_cos (12-month cycle)
# - is_weekend, is_holiday
# - hour_category (morning/afternoon/evening/night)

print(features.keys())
```

## Configuration

Three preset configurations are provided in `conf/`:

### Base Configuration (`conf/base.yaml`)

Default settings for 8.192s windows:

```yaml
data:
  fs: 100
  window_sec: 8.192
  hop_sec: 4.096

model:
  context_length: 819
  forecast_length: 200

training:
  batch_size: 256
  learning_rate: 1.0e-4
```

### Fine-tuning Configuration (`conf/finetune.yaml`)

For fine-tuning pretrained TTM models with extended context:

```yaml
model:
  context_length: 1024  # Extended context for TTM-E
  model_name: "ibm-granite/granite-timeseries-ttm-v1"
  freeze_backbone: false

training:
  learning_rate: 5.0e-5  # Lower LR for fine-tuning
  batch_size: 128
```

### Training from Scratch (`conf/scratch.yaml`)

For training custom models from random initialization:

```yaml
model:
  context_length: 819  # Strict 8.192s
  architecture:
    type: "transformer"
    num_layers: 6
    hidden_dim: 256

training:
  learning_rate: 1.0e-3  # Higher LR
  max_epochs: 200
```

Load configuration in Python:

```python
import yaml

with open('conf/base.yaml') as f:
    config = yaml.safe_load(f)
```

## Advanced Usage

### Custom Windowing

```python
from src.dataio.windowing import compute_window_params, segment_stream

# Compute window parameters
win_n, hop_n = compute_window_params(
    fs=100,
    win_sec=8.192,
    hop_sec=4.096,
    rounding='floor'
)
# win_n = 819, hop_n = 409

# Create windows from continuous data
import numpy as np
signals = np.random.randn(10000, 3)  # (time, channels)

windows = segment_stream(
    signals,
    win_n=819,
    hop_n=409,
    pad_mode='none'  # or 'reflect', 'zero'
)
# Output: (N, 3, 819) where N = number of windows
```

### HDF5/Zarr I/O

```python
from src.utils.io import save_windows_hdf5, load_windows_hdf5
import numpy as np
import pandas as pd

# Save windows
windows = np.random.randn(1000, 3, 819)
timestamps_start = pd.date_range('2020-01-01', periods=1000, freq='4.096s')
timestamps_end = timestamps_start + pd.Timedelta('8.192s')

save_windows_hdf5(
    'output.h5',
    windows,
    timestamps_start.values,
    timestamps_end.values,
    metadata={'participant_id': '001', 'fs': 100}
)

# Load windows
data = load_windows_hdf5('output.h5')
# data['windows']: (1000, 3, 819)
# data['timestamps_start']: timestamps
# data['metadata']: dict
```

### Reproducibility

```python
from src.utils.seed import set_all_seeds, ReproducibilityConfig

# Set all seeds
set_all_seeds(42, deterministic=True)

# Or use configuration object
config = ReproducibilityConfig(
    seed=42,
    deterministic=True,
    num_workers=4
)
config.apply()

# For DataLoader workers
from src.utils.seed import worker_init_fn
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn
)
```

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
# Test windowing operations
pytest tests/test_windowing.py -v

# Test preprocessing
pytest tests/test_preprocess.py -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Pipeline Overview

```
Raw .cwa files
     ↓
[read_cwa_to_segments]
     ↓
Tidy DataFrame (timestamp, x, y, z)
     ↓
[resample_to_fs]
     ↓
Uniform 100 Hz data + gap flags
     ↓
[segment_stream]
     ↓
Windows (N, C, win_n)
     ↓
[filter_windows_by_gaps]
     ↓
Filtered windows
     ↓
[save_windows_hdf5/zarr]
     ↓
Compressed HDF5/Zarr files
     ↓
[PyTorch Dataset]
     ↓
Training batches
```

## Troubleshooting

### Issue: `tsfm` installation fails

The IBM tsfm package must be installed from GitHub, not PyPI.

```bash
# Correct installation
pip install git+https://github.com/IBM/tsfm.git

# If you get SSL errors
pip install --trusted-host github.com git+https://github.com/IBM/tsfm.git

# If you're behind a proxy
pip install --proxy http://proxy.example.com:8080 git+https://github.com/IBM/tsfm.git
```

**Common errors:**

1. **"Could not find a version that satisfies the requirement granite-tsfm"**
   - You tried to install from PyPI. Use GitHub URL instead.

2. **Git not installed**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git

   # macOS
   brew install git

   # Or use conda
   conda install git
   ```

3. **Permission denied**
   ```bash
   # Use --user flag
   pip install --user git+https://github.com/IBM/tsfm.git
   ```

4. **Working without tsfm**
   - The toolkit works for data processing without tsfm
   - Only TTM model features require tsfm
   - Use the "Minimal Installation" option above

### Issue: `accelerometer` package fails to install

```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get install python3-dev build-essential

# Or use conda
conda install -c conda-forge accelerometer
```

### Issue: Out of memory during processing

Reduce batch size or process fewer participants at once:

```bash
python scripts/prepare_ukb.py --input single_file.cwa ...
```

### Issue: Slow data loading

- Enable `persistent_workers=True` in DataLoader
- Increase `num_workers` (typically 4-8)
- Use `pin_memory=True` for GPU training
- Consider using Zarr format for cloud storage

### Issue: Windows have unexpected shape

Verify window parameters:

```python
from src.dataio.windowing import compute_window_params

win_n, hop_n = compute_window_params(100, 8.192, 4.096, 'floor')
print(f"win_n={win_n}, hop_n={hop_n}")  # Should be 819, 409
```

## Citation

If you use this toolkit, please cite:

```bibtex
@software{ukb_ttm_accelerometry,
  title={UK Biobank TTM Accelerometry Toolkit},
  author={Your Team},
  year={2025},
  url={https://github.com/yourusername/ukb-ttm-accelerometry}
}
```

Also cite the relevant papers:
- UK Biobank: Sudlow et al. (2015)
- Granite TTM: IBM Research (2024)
- Accelerometer package: Doherty et al. (2017)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact the team.

---

**Happy Modeling!** 🚀
