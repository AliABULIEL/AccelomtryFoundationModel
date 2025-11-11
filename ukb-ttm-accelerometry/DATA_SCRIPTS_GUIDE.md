# Data Scripts Guide

Complete guide to the data download and generation scripts included in this repository.

## Overview

This toolkit includes scripts for **both** real UK Biobank data and synthetic demo data:

| Script | Purpose | Requires Access |
|--------|---------|----------------|
| `download_ukb_data.py` | Download real UK Biobank data | ✅ Yes |
| `generate_demo_data.py` | Generate synthetic data | ❌ No |
| `quick_demo.sh` | Complete pipeline demo | ❌ No |

## Scripts Reference

### 1. download_ukb_data.py

Downloads real UK Biobank accelerometry data using official tools.

**Prerequisites:**
- ✅ Approved UK Biobank application
- ✅ Downloaded `.ukbkey` file
- ✅ UK Biobank utilities installed (`ukbfetch`, `ukbunpack`)

**Usage:**
```bash
# Download all accelerometry data
python scripts/download_ukb_data.py \
    --key-file /path/to/your.ukbkey \
    --dataset-id YOUR_APPLICATION_ID \
    --output-dir ./data/ukb_raw

# Download specific participants
python scripts/download_ukb_data.py \
    --key-file your.ukbkey \
    --dataset-id YOUR_ID \
    --participant-ids 1000001 1000002 1000003 \
    --output-dir ./data/ukb_subset

# Download without auto-unpacking
python scripts/download_ukb_data.py \
    --key-file your.ukbkey \
    --dataset-id YOUR_ID \
    --output-dir ./data/ukb \
    --no-unpack
```

**Parameters:**
- `--key-file` (required): Path to your `.ukbkey` file
- `--dataset-id` (required): Your UK Biobank application/dataset ID
- `--output-dir` (default: `./data/ukb_raw`): Output directory
- `--participant-ids` (optional): Specific participant IDs to download
- `--no-unpack` (optional): Skip automatic unpacking
- `--log-level` (default: `INFO`): Logging verbosity

**Output Structure:**
```
output_dir/
├── downloaded/          # Encrypted .enc files
├── unpacked/           # Intermediate files
├── raw_cwa/            # Final .cwa files (USE THESE)
│   ├── 1000001.cwa
│   ├── 1000002.cwa
│   └── ...
└── download_stats.json # Statistics
```

**What It Does:**
1. ✅ Downloads encrypted files from UK Biobank
2. ✅ Automatically unpacks to .cwa format
3. ✅ Organizes files in `raw_cwa/` directory
4. ✅ Generates download statistics
5. ✅ Validates all downloads
6. ✅ Displays next steps

**Troubleshooting:**

*"Authentication failed"*
- Verify .ukbkey file is from your approved application
- Check dataset ID is correct
- Ensure key hasn't expired

*"ukbfetch: command not found"*
- Install UK Biobank utilities:
  ```bash
  wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbfetch
  wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbunpack
  chmod +x ukbfetch ukbunpack
  sudo mv ukbfetch ukbunpack /usr/local/bin/
  ```

*"Disk space full"*
- Each participant ≈ 50-100 MB
- 1000 participants ≈ 50-100 GB
- Check available space: `df -h`

---

### 2. generate_demo_data.py

Generates realistic synthetic accelerometry data for testing and development.

**Prerequisites:**
- ❌ No UK Biobank access needed
- ✅ Python packages: numpy, pandas, tqdm

**Usage:**
```bash
# Quick test (2 participants, 1 day)
python scripts/generate_demo_data.py \
    --n-participants 2 \
    --duration-days 1 \
    --output-dir ./data/quick_test

# Standard demo (5 participants, 7 days)
python scripts/generate_demo_data.py \
    --n-participants 5 \
    --duration-days 7 \
    --output-dir ./data/demo

# Realistic test (50 participants, 7 days)
python scripts/generate_demo_data.py \
    --n-participants 50 \
    --duration-days 7 \
    --output-dir ./data/realistic_demo \
    --seed 42

# Save as numpy format
python scripts/generate_demo_data.py \
    --n-participants 10 \
    --duration-days 7 \
    --output-dir ./data/demo \
    --output-format npy
```

**Parameters:**
- `--n-participants` (default: 5): Number of participants
- `--duration-days` (default: 7): Days per participant
- `--output-dir` (default: `./data/demo`): Output directory
- `--fs` (default: 100): Sampling frequency in Hz
- `--output-format` (default: `csv.gz`): Output format (`csv.gz` or `npy`)
- `--seed` (default: 42): Random seed for reproducibility
- `--log-level` (default: `INFO`): Logging verbosity

**Output Structure:**
```
output_dir/
├── demo_000001.csv.gz
├── demo_000002.csv.gz
├── demo_000003.csv.gz
└── generation_metadata.json
```

**What It Generates:**

✅ **Realistic Features:**
- Activity patterns (walking, sedentary, sleep)
- Circadian rhythms (less activity at night)
- Random gaps simulating non-wear periods
- Proper gravity component (9.8 m/s² on z-axis)
- Realistic noise characteristics

✅ **Activity Schedule:**
- 00:00-06:00: Sleep (sedentary)
- 06:00-09:00: Morning activities (light)
- 09:00-12:00: Daily activities (moderate)
- 12:00-13:00: Lunch (light)
- 13:00-18:00: Afternoon activities (moderate)
- 18:00-20:00: Evening (light)
- 20:00-22:00: Relaxation (sedentary)
- 22:00-24:00: Sleep (sedentary)

✅ **Data Quality:**
- 100 Hz sampling rate (exact 10ms periods)
- Realistic inter-individual variation
- 3-5 gaps per day (30s to 10min each)
- CSV.gz compressed format (~35 MB per week per participant)

**Size Estimates:**

| Participants | Days | Size (csv.gz) | Processing Time |
|-------------|------|---------------|-----------------|
| 2           | 1    | ~10 MB        | ~30 seconds     |
| 5           | 7    | ~175 MB       | ~2 minutes      |
| 50          | 7    | ~1.7 GB       | ~20 minutes     |
| 100         | 7    | ~3.5 GB       | ~40 minutes     |

**Use Cases:**
- ✅ Pipeline testing
- ✅ Algorithm development
- ✅ Tutorial demonstrations
- ✅ Unit testing
- ✅ Benchmarking processing speed
- ❌ **NOT for research** (use real UK Biobank data)

---

### 3. quick_demo.sh

Runs complete pipeline from data generation to splits.

**Prerequisites:**
- ❌ No UK Biobank access needed
- ✅ All Python packages installed

**Usage:**
```bash
# Run interactive demo
./quick_demo.sh

# Or make executable first
chmod +x quick_demo.sh
./quick_demo.sh
```

**What It Does:**
1. ✅ Generates 3 participants with 1 day each
2. ✅ Processes to 8.192s windows
3. ✅ Creates train/val/test splits
4. ✅ Shows data structure
5. ✅ Displays next steps

**Output:**
```
demo_quick_test/
├── raw/
│   ├── demo_000001.csv.gz
│   ├── demo_000002.csv.gz
│   └── demo_000003.csv.gz
├── processed/
│   ├── demo_000001/
│   │   └── windows.h5
│   ├── demo_000002/
│   │   └── windows.h5
│   └── demo_000003/
│       └── windows.h5
└── splits/
    ├── splits.json
    ├── train.txt
    ├── val.txt
    └── test.txt
```

**Time:** ~2 minutes
**Size:** ~30 MB

---

## Complete Workflows

### Workflow 1: Test with Demo Data (No Access Required)

```bash
# Step 1: Generate data
python scripts/generate_demo_data.py \
    --n-participants 10 \
    --duration-days 7 \
    --output-dir ./data/demo

# Step 2: Process to windows
python scripts/prepare_ukb.py \
    --input ./data/demo \
    --outdir ./data/processed \
    --win-sec 8.192 \
    --hop-sec 4.096

# Step 3: Create splits
python scripts/make_splits.py \
    --data-dir ./data/processed \
    --output-dir ./data/splits

# Step 4: Load and use
python3 -c "
from src.dataio.datasets import AccelerometryForecastDataset
dataset = AccelerometryForecastDataset('data/processed/demo_000001/windows.h5')
print(f'Dataset size: {len(dataset)} windows')
"
```

### Workflow 2: Real UK Biobank Data (Requires Access)

```bash
# Step 1: Download data
python scripts/download_ukb_data.py \
    --key-file your.ukbkey \
    --dataset-id YOUR_ID \
    --output-dir ./data/ukb_raw

# Step 2: Process to windows
python scripts/prepare_ukb.py \
    --input ./data/ukb_raw/raw_cwa \
    --outdir ./data/processed \
    --win-sec 8.192 \
    --hop-sec 4.096

# Step 3: Create splits
python scripts/make_splits.py \
    --data-dir ./data/processed \
    --output-dir ./data/splits

# Step 4: Train models
# (see example_pipeline.py)
```

### Workflow 3: Quick Demo (Automated)

```bash
# One command for complete demo
./quick_demo.sh
```

---

## Comparison: Real vs. Synthetic Data

| Feature | Real UK Biobank | Synthetic Demo |
|---------|-----------------|----------------|
| **Access** | Requires approval | No restrictions |
| **Time to get data** | 4-8 weeks | Instant |
| **Cost** | £100-500 | Free |
| **File format** | .cwa (binary) | .csv.gz or .npy |
| **Realism** | 100% real | ~80% realistic |
| **Participants** | Up to 100,000 | Unlimited |
| **Variability** | True human variation | Parametric |
| **Use for research** | ✅ Yes | ❌ No |
| **Use for testing** | ✅ Yes | ✅ Yes |
| **Processing** | Same pipeline | Same pipeline |

---

## Frequently Asked Questions

### Can I use synthetic data for publications?

**No.** Synthetic data is for:
- ✅ Testing the toolkit
- ✅ Developing algorithms
- ✅ Creating tutorials
- ✅ Benchmarking code

For actual research, you must use real UK Biobank data.

### How realistic is the synthetic data?

The synthetic data includes:
- ✅ Realistic activity patterns
- ✅ Circadian rhythms
- ✅ Proper gravity component
- ✅ Realistic noise
- ⚠️ Less inter-individual variability than real data
- ⚠️ Simplified activity transitions

**Similarity**: ~80% realistic for algorithm development

### Can I mix real and synthetic data?

**Not recommended.** Keep them separate:
- Real data → Research, publications
- Synthetic data → Development, testing

### How do I get UK Biobank access?

See [docs/UKB_DATA_ACCESS.md](docs/UKB_DATA_ACCESS.md) for complete guide.

**Quick summary:**
1. Apply: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
2. Wait 4-8 weeks for approval
3. Download `.ukbkey` file
4. Use `download_ukb_data.py` script

### What if I only want to test the toolkit?

Use synthetic data! No UK Biobank access needed:

```bash
python scripts/generate_demo_data.py \
    --n-participants 5 \
    --duration-days 7 \
    --output-dir ./data/demo
```

### Can I contribute more realistic synthetic patterns?

**Yes!** Pull requests welcome. The generator is in `scripts/generate_demo_data.py`.

Ideas for improvement:
- More varied activity patterns
- Age/sex-specific patterns
- Seasonal variations
- Activity bout detection
- More realistic artifacts

---

## Support

- **UK Biobank access questions**: https://www.ukbiobank.ac.uk/enable-your-research/contact-us
- **Toolkit issues**: https://github.com/yourusername/ukb-ttm-accelerometry/issues
- **Data processing questions**: See [README.md](README.md)
- **Complete guide**: [docs/UKB_DATA_ACCESS.md](docs/UKB_DATA_ACCESS.md)

---

## Quick Reference

**Generate demo data:**
```bash
python scripts/generate_demo_data.py --n-participants 5 --duration-days 7 --output-dir ./data/demo
```

**Download real data:**
```bash
python scripts/download_ukb_data.py --key-file your.ukbkey --dataset-id YOUR_ID --output-dir ./data/ukb
```

**Run complete demo:**
```bash
./quick_demo.sh
```

**Process data:**
```bash
python scripts/prepare_ukb.py --input ./data/demo --outdir ./data/processed
```

**Create splits:**
```bash
python scripts/make_splits.py --data-dir ./data/processed --output-dir ./data/splits
```

---

Last updated: 2025-11-11
