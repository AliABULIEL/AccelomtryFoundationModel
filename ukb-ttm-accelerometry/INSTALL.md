# Installation Guide

Quick reference for installing UK Biobank TTM Accelerometry Toolkit.

## Quick Install (Recommended)

### Option 1: Automated Script (Linux/macOS)

```bash
chmod +x install.sh
./install.sh
```

This script will:
1. Check Python version (≥3.9 required)
2. Install IBM tsfm from GitHub
3. Install all other dependencies
4. Run basic tests
5. Optionally install in development mode

### Option 2: Manual Installation

```bash
# Step 1: Install tsfm from GitHub (REQUIRED)
pip install git+https://github.com/IBM/tsfm.git

# Step 2: Install other dependencies
pip install -r requirements.txt

# Step 3: Verify installation
python3 test_basic.py
```

### Option 3: Google Colab

```python
# Run this in a Colab cell
!pip install git+https://github.com/IBM/tsfm.git
!git clone https://github.com/yourusername/ukb-ttm-accelerometry.git
%cd ukb-ttm-accelerometry
!pip install -r requirements.txt
!python test_basic.py

# Or use the automated script
!python install_colab.py
```

## Installation Options

### Full Installation (with TTM models)

Required for using Granite TTM foundation models:

```bash
pip install git+https://github.com/IBM/tsfm.git
pip install -r requirements.txt
```

**Size**: ~2-3 GB (includes PyTorch, transformers, tsfm)
**Time**: 5-10 minutes
**Features**: All features available

### Minimal Installation (data processing only)

If you only need data preprocessing without TTM models:

```bash
pip install numpy pandas scipy numba h5py zarr torch accelerometer pyyaml tqdm scikit-learn
```

**Size**: ~1-1.5 GB
**Time**: 2-5 minutes
**Features**: Data I/O, windowing, preprocessing, PyTorch datasets
**Limitations**: TTM models not available

### Development Installation

For contributors:

```bash
pip install git+https://github.com/IBM/tsfm.git
pip install -r requirements.txt
pip install -e .[dev]  # Install in editable mode with dev dependencies
```

This includes: pytest, black, ruff, mypy

## Verifying Installation

### Quick Check

```bash
python3 test_basic.py
```

Expected output:
```
Testing basic imports...
✓ compute_window_params works correctly
✓ segment_stream works correctly
All basic tests passed! ✓
```

### Full Test Suite

```bash
pytest tests/ -v
```

### Check tsfm Installation

```python
python3 -c "import tsfm; print(f'tsfm version: {tsfm.__version__}')"
```

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'tsfm'"

**Cause**: tsfm not installed from GitHub

**Solution**:
```bash
pip install git+https://github.com/IBM/tsfm.git
```

### Issue: "Could not find a version that satisfies the requirement granite-tsfm"

**Cause**: Trying to install from PyPI (it's not there)

**Solution**: Use GitHub URL (see above)

### Issue: Git not found

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git

# Windows
# Download from: https://git-scm.com/download/win
```

### Issue: SSL/Certificate errors

**Solution**:
```bash
pip install --trusted-host github.com git+https://github.com/IBM/tsfm.git
```

### Issue: Behind corporate proxy

**Solution**:
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
pip install git+https://github.com/IBM/tsfm.git
```

### Issue: Permission denied

**Solution**:
```bash
# Install to user directory
pip install --user git+https://github.com/IBM/tsfm.git
pip install --user -r requirements.txt
```

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, or Windows (WSL recommended for Windows)
- **Python**: 3.9 or higher
- **RAM**: 8 GB (16 GB recommended for training)
- **Disk**: 5 GB free space

### Recommended Requirements

- **OS**: Ubuntu 20.04+ or macOS 12+
- **Python**: 3.10 or 3.11
- **RAM**: 16-32 GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for training)
- **Disk**: 50+ GB for data storage

### Python Version Compatibility

| Python | Status | Notes |
|--------|--------|-------|
| 3.9    | ✓ Tested | Minimum required version |
| 3.10   | ✓ Tested | Recommended |
| 3.11   | ✓ Tested | Recommended |
| 3.12   | ⚠ Experimental | Some dependencies may not be available |
| 3.8    | ✗ Not supported | Use 3.9+ |

## Conda/Mamba Installation

If you prefer conda/mamba:

```bash
# Create environment
conda create -n ukb-ttm python=3.10
conda activate ukb-ttm

# Install PyTorch (choose appropriate version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other conda packages
conda install numpy pandas scipy h5py zarr scikit-learn pyyaml tqdm -c conda-forge

# Install remaining packages with pip
pip install git+https://github.com/IBM/tsfm.git
pip install accelerometer numba pyarrow einops tensorboard wandb
```

## Docker Installation (Advanced)

For reproducible environments:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential

# Install Python packages
RUN pip install git+https://github.com/IBM/tsfm.git

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "test_basic.py"]
```

Build and run:
```bash
docker build -t ukb-ttm .
docker run -it ukb-ttm
```

## Next Steps

After successful installation:

1. **Read the documentation**: See [README.md](README.md)
2. **Run example pipeline**: `python3 example_pipeline.py`
3. **Process your data**: `python scripts/prepare_ukb.py --help`
4. **Check configurations**: Review files in `conf/`

## Support

- **Documentation**: [README.md](README.md)
- **Issues**: https://github.com/yourusername/ukb-ttm-accelerometry/issues
- **IBM tsfm**: https://github.com/IBM/tsfm
- **Discussions**: https://github.com/yourusername/ukb-ttm-accelerometry/discussions

## Version Information

- **Toolkit Version**: 0.1.0
- **tsfm**: Install from GitHub (latest)
- **PyTorch**: ≥2.1
- **Python**: ≥3.9

Last updated: 2025-11-11
