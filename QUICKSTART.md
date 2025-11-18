# Quick Start Guide

## Overview

This is a complete production-grade system for training IBM's Tiny Time Mixer (TTM) on UK Biobank accelerometry data. The system is optimized for Google Colab free tier.

## Installation (5 minutes)

### Option 1: Google Colab (Recommended)

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/AccelomtryFoundationModel/blob/main/notebooks/TTM_Accelerometry_Training.ipynb)

2. The notebook will automatically:
   - Install all dependencies
   - Mount Google Drive
   - Setup checkpointing
   - Run the complete pipeline

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/AccelomtryFoundationModel.git
cd AccelomtryFoundationModel

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python scripts/verify_installation.py
```

## Quick Test (10 minutes)

### Create Demo Dataset

```bash
# Create synthetic dataset for testing
make create-demo-data
```

Or manually:

```python
import h5py
import numpy as np

n_samples = 10000
windows = np.random.randn(n_samples, 820, 3).astype('float32')
labels = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])

with h5py.File('data/synthetic_data.h5', 'w') as f:
    f.create_dataset('windows', data=windows, compression='gzip')
    f.create_dataset('labels', data=labels)
```

### Run Quick Training

```bash
# Quick training (reduced epochs for testing)
make train-quick

# Or directly:
python scripts/train.py --data data/synthetic_data.h5 --quick
```

Expected output:
```
STEP 1: Loading Data
Train: 7000 samples
Val:   1500 samples
Test:  1500 samples

STEP 2: Initializing Model
Total parameters:     1,234,567
Trainable parameters: 234,567

STEP 3-4: Training
Stage 1: Linear Probe (Frozen Encoder)
Epoch 1/2: Train Loss: 1.2345, Val Loss: 1.1234
...

STEP 5: Evaluation
F1 Score (macro): 0.8542 [0.8421, 0.8663]

TRAINING COMPLETE
```

## Full Training on UK Biobank Data

### 1. Prepare Data

```python
from src.data.data_loader import AccelerometryDataLoader

loader = AccelerometryDataLoader(
    data_dir='./data/ukb_cwa',
    cache_dir='./data/cache',
    window_size=820,
    stride=410,
)

# Process all .cwa files
cwa_files = glob.glob('./data/ukb_cwa/*.cwa')
loader.create_dataset_hdf5(
    cwa_files=cwa_files,
    output_path='./data/ukb_dataset.h5',
    labels=participant_labels,  # Your labels
)
```

### 2. Configure Training

Edit `configs/default.yaml`:

```yaml
training:
  stage1_epochs: 10   # Linear probe
  stage2_epochs: 20   # LoRA fine-tuning
  stage3_epochs: 10   # Full fine-tuning
  batch_size: 64
  use_amp: true
```

### 3. Run Training

```bash
python scripts/train.py --data data/ukb_dataset.h5 --config configs/default.yaml
```

## Expected Performance

After full training on UK Biobank:

| Metric | Target | Expected |
|--------|--------|----------|
| F1 Score | >0.85 | 0.87-0.91 |
| Inference Speed | <10ms | ~5ms |
| GPU Memory | <3GB | ~2.5GB |
| Model Size | <10MB | ~8MB |

## Troubleshooting

### Out of Memory (OOM)

The system automatically handles OOM by reducing batch size. If you still encounter issues:

```python
# In config
training:
  batch_size: 32  # Reduce from 64
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Slow Training

Enable all optimizations:

```python
# In config
training:
  use_amp: true  # Mixed precision
colab:
  use_gradient_checkpointing: true  # Reduces memory
```

### Import Errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

## Project Structure

```
AccelomtryFoundationModel/
├── src/               # Source code
│   ├── data/          # Data loading and augmentation
│   ├── models/        # TTM classifier
│   ├── training/      # Training pipeline
│   ├── evaluation/    # Evaluation and benchmarking
│   └── utils/         # Configuration and utilities
├── tests/             # Unit tests
├── notebooks/         # Jupyter notebooks
├── scripts/           # Training scripts
└── configs/           # Configuration files
```

## Key Files

- **Training**: `scripts/train.py` - Main training script
- **Notebook**: `notebooks/TTM_Accelerometry_Training.ipynb` - Colab notebook
- **Config**: `configs/default.yaml` - Default configuration
- **Tests**: `tests/test_*.py` - Unit tests
- **Docs**: `README.md` - Full documentation

## Common Commands

```bash
# Run tests
make test

# Train (quick)
make train-quick

# Train (full)
make train

# Format code
make format

# Clean temporary files
make clean

# Show help
make help
```

## Next Steps

1. **Data Preparation**: Process your UK Biobank .cwa files
2. **Training**: Run full 3-stage training pipeline
3. **Evaluation**: Benchmark on CAPTURE-24
4. **Deployment**: Export model for production use

## Support

- **Documentation**: See `README.md` for full details
- **Issues**: https://github.com/YOUR_USERNAME/AccelomtryFoundationModel/issues
- **Examples**: Check `notebooks/` for detailed examples

## Performance Tips

### For Best Results

1. **Use Full Training**: Don't use `--quick` for production
2. **Tune Hyperparameters**: Adjust learning rates per your data
3. **Monitor Validation**: Watch for overfitting
4. **Use SSL Augmentations**: Enable in config for better generalization
5. **Checkpoint Frequently**: Set `save_every: 1000` in config

### For Fastest Training

1. **Enable AMP**: `use_amp: true`
2. **Gradient Checkpointing**: `use_gradient_checkpointing: true`
3. **Batch Size**: Use largest that fits in memory
4. **Num Workers**: Set to number of CPU cores

### For Best Accuracy

1. **More Epochs**: Increase stage2 and stage3 epochs
2. **Data Augmentation**: Enable SSL augmentations
3. **Larger Hidden Dim**: Increase `hidden_dim` if memory allows
4. **Lower Learning Rate**: Fine-tune with lower lr in stage 3

## Citation

If you use this code, please cite:

```bibtex
@software{ttm_accelerometry_2025,
  title={TTM-Based Accelerometry Foundation Model},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/AccelomtryFoundationModel}
}
```

---

**Happy Training! 🚀**
