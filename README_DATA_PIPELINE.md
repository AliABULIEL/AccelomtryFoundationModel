# UKB TTM Accelerometry Foundation Model - Data Pipeline

This is **Phase 1** of the implementation: Project structure, configuration, Colab utilities, and complete data processing pipeline.

## Project Structure

```
ukb_ttm_accel/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration dataclasses and YAML I/O
├── data/
│   ├── __init__.py
│   ├── windowing.py           # Accelerometry windowing and preprocessing
│   ├── clinical_metadata.py   # Clinical data loading and label extraction
│   ├── ukb_accel_dataset.py   # PyTorch Dataset with HDF5 caching
│   └── collate_fns.py         # DataLoader collate functions
└── utils/
    ├── __init__.py
    ├── colab_env.py           # Colab environment setup
    ├── logging_utils.py       # Logging utilities
    └── seed_utils.py          # Reproducibility utilities

main_ssl_pretrain.py           # Self-supervised pretraining (stub)
main_supervised_finetune.py    # Supervised fine-tuning (stub)
main_evaluate.py               # Evaluation script (stub)
test_data_pipeline.py          # Complete pipeline test with synthetic data
```

## Installation

### In Google Colab

```python
# Install the package and dependencies
!pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.18" \
             accelerate transformers datasets actipy h5py pyarrow pyyaml scikit-learn

# Mount Google Drive (if your data is there)
from google.colab import drive
drive.mount('/content/drive')

# Or use the utility function
from ukb_ttm_accel.utils import setup_colab_env
setup_colab_env(mount_drive=True, install_deps=True)
```

### Local Installation

```bash
pip install torch torchvision torchaudio
pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.18"
pip install accelerate transformers datasets
pip install pandas numpy scikit-learn
pip install actipy h5py pyarrow pyyaml
```

## Quick Start: Test the Pipeline

Run the complete test suite with synthetic data:

```bash
python test_data_pipeline.py --num-participants 10 --duration-hours 24
```

This will:
1. Generate synthetic accelerometry data (CSV format)
2. Generate synthetic clinical metadata
3. Test all windowing and preprocessing functions
4. Test the Dataset and DataLoader
5. Test HDF5 caching functionality
6. Print comprehensive logs and verify correctness

Expected output:
```
=============================================================
ALL TESTS PASSED!
=============================================================

Data pipeline is working correctly.
You can now proceed to use this pipeline with real UKB data.
```

## Configuration

Create a YAML configuration file:

```yaml
# config.yaml

# Data configuration
window_seconds: 10
sampling_rate_hz: 50
batch_size: 32
num_workers: 2
max_windows_per_participant: null  # null = use all windows
overlap_fraction: 0.0
normalize_per_window: true

# Data paths
accel_data_dir: "/content/drive/MyDrive/ukb_accel"
clinical_csv_path: "/content/drive/MyDrive/ukb_accel/clinical.csv"
cache_hdf5_path: "/content/drive/MyDrive/ukb_accel/cache.h5"

# Data split ratios
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15

# Clinical metadata
label_type: "bmi_class"  # Options: bmi_class, bmi, age_group, sex, hypertension, diabetes
metadata_columns:
  - age
  - sex
  - bmi

# SSL pretraining
ssl_epochs: 50
ssl_learning_rate: 0.001
ssl_mask_ratio: 0.3

# Fine-tuning
ft_epochs: 30
ft_learning_rate: 0.0001
backbone_freeze_epochs: 5

# Optimization
weight_decay: 0.0001
gradient_accumulation_steps: 1
max_grad_norm: 1.0

# Colab/IO
use_mixed_precision: true
checkpoint_dir: "./checkpoints"
log_dir: "./logs"

# Random seed
seed: 42
```

Load and use configuration:

```python
from ukb_ttm_accel.config import load_config_from_yaml

config = load_config_from_yaml("config.yaml")
print(config.batch_size)  # 32
```

## Usage Examples

### 1. Basic Dataset Usage

```python
from ukb_ttm_accel.data import UKBAccelDataset
from ukb_ttm_accel.config import TrainingConfig

# Create configuration
config = TrainingConfig(
    window_seconds=10,
    sampling_rate_hz=50,
    label_type="bmi_class",
    metadata_columns=["age", "sex", "bmi"]
)

# Create dataset
dataset = UKBAccelDataset(
    accel_file_paths=[
        "/path/to/participant_1001.cwa",
        "/path/to/participant_1002.cwa",
    ],
    clinical_csv_path="/path/to/clinical.csv",
    split="train",
    window_seconds=config.window_seconds,
    sampling_rate_hz=config.sampling_rate_hz,
    label_type=config.label_type,
    metadata_columns=config.metadata_columns,
)

# Get a sample
sample = dataset[0]
print(sample['signal'].shape)      # torch.Size([500, 3])
print(sample['label'])             # tensor(2)
print(sample['metadata'].shape)    # torch.Size([3])
```

### 2. DataLoader with Batching

```python
from torch.utils.data import DataLoader
from ukb_ttm_accel.data import accel_collate_fn

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=accel_collate_fn,
    num_workers=2,
)

for batch in dataloader:
    signals = batch['signals']       # [32, 500, 3]
    labels = batch['labels']         # [32]
    metadata = batch['metadata']     # [32, num_features]

    # Your training code here
    break
```

### 3. HDF5 Caching for Faster Loading

```python
# First run: processes data and saves to HDF5
dataset = UKBAccelDataset(
    accel_file_paths=file_paths,
    clinical_csv_path="clinical.csv",
    split="train",
    cache_hdf5_path="cache_train.h5",  # Will be created
    # ... other parameters
)

# Subsequent runs: loads from cache (much faster)
dataset = UKBAccelDataset(
    accel_file_paths=file_paths,
    clinical_csv_path="clinical.csv",
    split="train",
    cache_hdf5_path="cache_train.h5",  # Loads from existing cache
    # ... other parameters
)
```

### 4. Limiting Windows for Colab

```python
# Useful for quick experiments in Colab with limited RAM
dataset = UKBAccelDataset(
    accel_file_paths=file_paths,
    clinical_csv_path="clinical.csv",
    split="train",
    max_windows_per_participant=100,  # Only use 100 windows per person
    # ... other parameters
)
```

## Data Format Requirements

### Accelerometry Files

Supported formats:

1. **CWA files** (Axivity binary format):
   - Filename: `{participant_id}.cwa`
   - Processed with `actipy` library
   - Automatic calibration and filtering

2. **CSV files**:
   - Columns: `time`, `x`, `y`, `z`
   - `time`: datetime string or timestamp
   - `x`, `y`, `z`: acceleration in g

3. **Parquet files**:
   - Same structure as CSV
   - Better compression and faster I/O

### Clinical Metadata CSV

Required columns:
- `eid`: Participant ID (must match accelerometry filenames)

Optional columns (depending on `label_type`):
- `age`: Age in years
- `sex`: 0=female, 1=male
- `bmi`: Body Mass Index
- `height`: Height in cm
- `weight`: Weight in kg
- `hypertension`: Binary (0/1)
- `diabetes`: Binary (0/1)

Example:
```csv
eid,age,sex,bmi,height,weight,hypertension,diabetes
1001234,45,1,27.3,175,83.7,0,0
1001235,52,0,24.1,162,63.2,1,0
```

## Label Types

The pipeline supports multiple label types for different downstream tasks:

| Label Type | Description | Use Case | Output Type |
|------------|-------------|----------|-------------|
| `bmi_class` | BMI categories (0-3) | Classification | Long tensor |
| `bmi` | Raw BMI value | Regression | Float tensor |
| `age_group` | Age categories (0-2) | Classification | Long tensor |
| `sex` | Binary sex (0/1) | Classification | Long tensor |
| `hypertension` | Hypertension status | Classification | Long tensor |
| `diabetes` | Diabetes status | Classification | Long tensor |

BMI Classes:
- 0: Underweight (< 18.5)
- 1: Normal (18.5-24.9)
- 2: Overweight (25-29.9)
- 3: Obese (≥ 30)

Age Groups:
- 0: Young (< 40)
- 1: Middle (40-60)
- 2: Old (> 60)

## Reproducibility

Set random seed for reproducible experiments:

```python
from ukb_ttm_accel.utils import set_global_seed

set_global_seed(42)  # Seeds Python, NumPy, and PyTorch
```

## Logging

Set up logging for your experiments:

```python
from ukb_ttm_accel.utils import setup_logger

logger = setup_logger(
    name="my_experiment",
    log_file="experiment.log",
)

logger.info("Starting experiment...")
```

## Next Steps

This Phase 1 implementation provides the complete data pipeline. Future phases will add:

- **Phase 2**: TTM model adaptation for accelerometry
- **Phase 3**: Self-supervised pretraining implementation
- **Phase 4**: Supervised fine-tuning implementation
- **Phase 5**: Evaluation and embedding extraction

## Troubleshooting

### Out of Memory in Colab

1. Reduce `batch_size` in config
2. Use `max_windows_per_participant` to limit data
3. Enable `use_mixed_precision` for lower memory usage
4. Process data in smaller chunks with HDF5 caching

### Slow Data Loading

1. Use HDF5 caching (creates cache once, loads quickly after)
2. Increase `num_workers` in DataLoader (not in Colab)
3. Use Parquet instead of CSV for faster I/O

### Missing Dependencies

```bash
# Install actipy for .cwa file support
pip install actipy

# Install HDF5 support
pip install h5py
```

## License

This project is designed for research purposes with UK Biobank accelerometry data.
