# Quick Start Guide - Phase 1

## 🚀 5-Minute Setup

### In Google Colab

```python
# 1. Install dependencies
!pip install -q numpy pandas scikit-learn torch pyyaml h5py tqdm

# 2. Clone/upload this repository to Colab
# (or copy the ukb_ttm_accel folder to your Colab environment)

# 3. Test the pipeline
!python test_data_pipeline.py --num-participants 5 --duration-hours 6

# Output should end with:
# =============================================================
# ALL TESTS PASSED!
# =============================================================
```

### Local Setup

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn torch pyyaml h5py tqdm

# 2. Verify imports
python verify_imports.py

# 3. Test pipeline
python test_data_pipeline.py --num-participants 5 --duration-hours 6
```

## 📦 What You Get

After running the test, you'll have validated:

✅ Configuration system (YAML-based)
✅ Data windowing and preprocessing
✅ Clinical metadata handling
✅ PyTorch Dataset with HDF5 caching
✅ DataLoader with proper batching
✅ All utilities (logging, seeding, Colab helpers)

## 📊 Using with Real Data

### Step 1: Prepare Your Data

**Accelerometry files** (one per participant):
- Format: `.cwa`, `.csv`, or `.parquet`
- Naming: Must contain participant ID (e.g., `1001234.cwa`)
- Location: Any directory

**Clinical metadata** (single CSV):
```csv
eid,age,sex,bmi,height,weight
1001234,45,1,27.3,175,83.7
1001235,52,0,24.1,162,63.2
```

### Step 2: Create Configuration

```bash
cp configs/example_config.yaml my_config.yaml
# Edit my_config.yaml with your paths
```

### Step 3: Create Dataset

```python
from ukb_ttm_accel.config import load_config_from_yaml
from ukb_ttm_accel.data import UKBAccelDataset
from torch.utils.data import DataLoader
import glob

# Load config
config = load_config_from_yaml("my_config.yaml")

# Get all accelerometry files
accel_files = glob.glob("/path/to/accel/*.cwa")

# Create dataset
dataset = UKBAccelDataset(
    accel_file_paths=accel_files,
    clinical_csv_path=config.clinical_csv_path,
    split="train",
    window_seconds=config.window_seconds,
    sampling_rate_hz=config.sampling_rate_hz,
    label_type=config.label_type,
    metadata_columns=config.metadata_columns,
    cache_hdf5_path="cache_train.h5",  # Recommended for large datasets
    max_windows_per_participant=100,   # Optional: limit for Colab
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=2,
)

# Use it!
for batch in dataloader:
    signals = batch['signals']      # Shape: [B, T, 3]
    labels = batch['labels']        # Shape: [B]
    metadata = batch['metadata']    # Shape: [B, M]
    # Your code here...
```

## 🔧 Common Configurations

### For Colab (Limited RAM)

```yaml
batch_size: 16                      # Smaller batches
max_windows_per_participant: 50     # Limit windows
cache_hdf5_path: "/content/cache.h5"  # Use caching
use_mixed_precision: true           # Lower memory
```

### For Full Dataset (Local/Server)

```yaml
batch_size: 64                      # Larger batches
max_windows_per_participant: null   # Use all windows
num_workers: 8                      # Parallel loading
cache_hdf5_path: "./cache.h5"       # Fast SSD
```

### For Different Tasks

**BMI Classification:**
```yaml
label_type: "bmi_class"
metadata_columns: [age, sex, bmi]
```

**Age Prediction:**
```yaml
label_type: "age_group"
metadata_columns: [sex, bmi, height, weight]
```

**Disease Classification:**
```yaml
label_type: "diabetes"  # or "hypertension"
metadata_columns: [age, sex, bmi]
```

## 📝 Label Types Reference

| Label Type | Task Type | Classes | Description |
|------------|-----------|---------|-------------|
| `bmi_class` | Classification | 4 | Underweight/Normal/Overweight/Obese |
| `bmi` | Regression | - | Raw BMI value |
| `age_group` | Classification | 3 | Young/Middle/Old |
| `sex` | Classification | 2 | Female/Male |
| `hypertension` | Classification | 2 | No/Yes |
| `diabetes` | Classification | 2 | No/Yes |

## 🐛 Troubleshooting

### "Participant ID not found"
- Check that your accelerometry filenames contain numeric IDs
- Ensure these IDs exist in the clinical CSV `eid` column

### "Out of memory in Colab"
- Set `max_windows_per_participant: 50`
- Reduce `batch_size: 16`
- Enable `use_mixed_precision: true`

### "Slow data loading"
- Use HDF5 caching: set `cache_hdf5_path`
- First run will be slow (creating cache)
- Subsequent runs will be 10-100x faster

### "Missing columns in clinical CSV"
- Ensure CSV has `eid` column
- For `bmi_class` label: need `bmi` column
- For metadata: need columns matching `metadata_columns`

## 📚 Next Steps

1. ✅ **Test pipeline** with synthetic data (`test_data_pipeline.py`)
2. ✅ **Verify imports** (`verify_imports.py`)
3. 📊 **Load your data** (use examples above)
4. 🔍 **Explore batches** (check shapes, values, distributions)
5. ⏸️ **Wait for Phase 2** (model training implementation)

## 💡 Tips

- **Always use HDF5 caching** for datasets > 100 participants
- **Test with subset first**: Use `max_windows_per_participant`
- **Check data quality**: Inspect first batch for NaNs/outliers
- **Monitor memory**: Use `nvidia-smi` or Colab runtime stats
- **Save configs**: Version control your YAML files

## 📖 Full Documentation

See `README_DATA_PIPELINE.md` for complete documentation.

See `PHASE1_SUMMARY.md` for implementation details.

## ✨ Features Highlights

- 🎯 **Multiple file formats**: CWA, CSV, Parquet
- 🗄️ **Smart caching**: HDF5 for fast repeated access
- 🎚️ **Flexible windowing**: Configurable size and overlap
- 🏷️ **Multiple labels**: 6 different clinical labels
- 📊 **Rich metadata**: Combine signals with clinical data
- 🔄 **Reproducible**: Seed management built-in
- 🚀 **Colab-ready**: Optimized for limited resources

---

**Phase 1 Complete** ✅
Ready for real-world UK Biobank data processing!
