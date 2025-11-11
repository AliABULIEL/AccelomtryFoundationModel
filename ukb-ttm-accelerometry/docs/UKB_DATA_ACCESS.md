# UK Biobank Data Access Guide

Complete guide for accessing and downloading UK Biobank accelerometry data.

## Table of Contents

1. [Getting Access to UK Biobank](#getting-access)
2. [Downloading Real Data](#downloading-real-data)
3. [Using Demo Data for Testing](#using-demo-data)
4. [Data Processing Pipeline](#data-processing-pipeline)

---

## Getting Access

### Overview

UK Biobank is a **restricted access** research resource. You cannot simply download the data - you must apply for and be granted access.

### Requirements for Access

1. **Institutional Affiliation**
   - Must be affiliated with a research institution
   - Institution must have ethics approval processes

2. **Research Proposal**
   - Clear research question
   - Public health benefit
   - Ethical approval

3. **Costs**
   - Application fee: ~£100-500 (varies by institution type)
   - Academic institutions: lower fees
   - Commercial organizations: higher fees

### Application Process

#### Step 1: Register

1. Visit: https://www.ukbiobank.ac.uk/
2. Create an account in the Access Management System (AMS)
3. Provide institutional details

#### Step 2: Prepare Application

Required information:
- **Project Title**: Descriptive title of your research
- **Research Aims**: Clear objectives
- **Public Health Impact**: How will this benefit public health?
- **Data Fields Required**: Specify accelerometry data (Field 90001)
- **Duration**: How long you need access (typically 3 years, renewable)
- **Ethics Approval**: From your institution's IRB/ethics committee

#### Step 3: Submit Application

1. Complete online application form in AMS
2. Pay application fee
3. Submit ethics approval documentation
4. Wait for review

#### Step 4: Review Process

- **Timeline**: 4-8 weeks typically
- **Possible Outcomes**:
  - Approved
  - Approved with modifications
  - Rejected (can reapply with changes)

#### Step 5: Access Granted

Once approved:
1. Download your `.ukbkey` file from AMS
2. Download UK Biobank utilities (ukbfetch, ukbunpack)
3. Download your data using the scripts provided

### Accelerometry Data Field

- **Field ID**: 90001
- **Description**: Raw accelerometer data (.cwa files)
- **Duration**: Typically 7 days of continuous recording
- **Participants**: ~100,000 participants with accelerometry data
- **File Size**: ~50-100 MB per participant (compressed)

---

## Downloading Real Data

Once you have access, use our download script.

### Prerequisites

1. ✅ Approved UK Biobank application
2. ✅ Downloaded `.ukbkey` file
3. ✅ UK Biobank utilities installed

### Install UK Biobank Utilities

```bash
# Download utilities
wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbfetch
wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbunpack

# Make executable
chmod +x ukbfetch ukbunpack

# Move to system path
sudo mv ukbfetch ukbunpack /usr/local/bin/

# Verify installation
ukbfetch --help
ukbunpack --help
```

### Download All Accelerometry Data

```bash
python scripts/download_ukb_data.py \
    --key-file /path/to/your.ukbkey \
    --dataset-id YOUR_APPLICATION_ID \
    --output-dir ./data/ukb_raw
```

**Parameters:**
- `--key-file`: Path to your .ukbkey file from UK Biobank
- `--dataset-id`: Your application/dataset ID (e.g., 12345)
- `--output-dir`: Where to save downloaded data

### Download Specific Participants

```bash
python scripts/download_ukb_data.py \
    --key-file your.ukbkey \
    --dataset-id YOUR_ID \
    --participant-ids 1000001 1000002 1000003 \
    --output-dir ./data/ukb_subset
```

### Download Process

The script will:
1. ✅ Download encrypted .enc files from UK Biobank
2. ✅ Automatically unpack to .cwa files
3. ✅ Organize files in `output_dir/raw_cwa/`
4. ✅ Generate download statistics
5. ✅ Display next steps for processing

**Expected Output Structure:**
```
data/ukb_raw/
├── downloaded/           # Encrypted .enc files
├── unpacked/            # Intermediate unpacked files
├── raw_cwa/             # Final .cwa files (USE THESE)
│   ├── 1000001.cwa
│   ├── 1000002.cwa
│   └── ...
└── download_stats.json  # Download statistics
```

### Troubleshooting Download Issues

#### Issue: "Authentication failed"
**Cause**: Invalid .ukbkey file or wrong dataset ID

**Solution**:
- Verify key file is from your approved application
- Check dataset ID matches your application
- Ensure key file hasn't expired (typically 3 years)

#### Issue: "Connection timeout"
**Cause**: UK Biobank servers slow or network issues

**Solution**:
- Retry the download
- Use faster internet connection
- Download in smaller batches with `--participant-ids`

#### Issue: "Disk space full"
**Cause**: Each participant = ~50-100 MB

**Solution**:
- Ensure adequate disk space:
  - 100 participants = ~5-10 GB
  - 1000 participants = ~50-100 GB
  - 10000 participants = ~500 GB - 1 TB

---

## Using Demo Data for Testing

**Don't have UK Biobank access yet?** Use our synthetic data generator!

### Generate Synthetic Data

```bash
# Generate 5 participants with 7 days each
python scripts/generate_demo_data.py \
    --n-participants 5 \
    --duration-days 7 \
    --output-dir ./data/demo
```

**Parameters:**
- `--n-participants`: Number of participants (default: 5)
- `--duration-days`: Days per participant (default: 7)
- `--output-dir`: Output directory
- `--fs`: Sampling rate in Hz (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

### Quick Test Dataset

For rapid testing:

```bash
# 2 participants, 1 day each (~17 MB)
python scripts/generate_demo_data.py \
    --n-participants 2 \
    --duration-days 1 \
    --output-dir ./data/quick_test
```

### Realistic Demo Dataset

For thorough testing:

```bash
# 50 participants, 7 days each (~1.7 GB)
python scripts/generate_demo_data.py \
    --n-participants 50 \
    --duration-days 7 \
    --output-dir ./data/realistic_demo
```

### What Gets Generated

The synthetic data includes:
- ✅ Realistic activity patterns (walking, sleep, sedentary)
- ✅ Circadian rhythms (more activity during day)
- ✅ Random gaps (simulating non-wear periods)
- ✅ Proper gravity component (9.8 m/s² on z-axis)
- ✅ Realistic noise characteristics
- ✅ CSV.gz or .npy format (compatible with prepare_ukb.py)

### Differences from Real Data

| Feature | Real UKB Data | Synthetic Data |
|---------|--------------|----------------|
| File format | .cwa (binary) | .csv.gz or .npy |
| Participants | Real humans | Simulated patterns |
| Variability | High inter-individual | Parametric variation |
| Artifacts | Real-world noise | Simulated noise |
| **Processing** | ✅ Works | ✅ Works identically |

**Important**: Synthetic data is for **development and testing only**. For actual research, use real UK Biobank data.

---

## Data Processing Pipeline

### Complete Workflow

#### 1. Download/Generate Data

**Option A - Real Data:**
```bash
python scripts/download_ukb_data.py \
    --key-file your.ukbkey \
    --dataset-id YOUR_ID \
    --output-dir ./data/ukb_raw
```

**Option B - Demo Data:**
```bash
python scripts/generate_demo_data.py \
    --n-participants 10 \
    --duration-days 7 \
    --output-dir ./data/demo
```

#### 2. Process to Windows

```bash
python scripts/prepare_ukb.py \
    --input ./data/ukb_raw/raw_cwa \  # or ./data/demo
    --outdir ./data/processed \
    --win-sec 8.192 \
    --hop-sec 4.096 \
    --max-gap-ratio 0.1 \
    --summary-file processing_summary.csv
```

**Output**: HDF5 files with 8.192s windows
```
data/processed/
├── demo_000001/
│   └── windows.h5
├── demo_000002/
│   └── windows.h5
└── ...
```

#### 3. Create Splits

```bash
python scripts/make_splits.py \
    --data-dir ./data/processed \
    --output-dir ./data/splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

**Output**: Train/val/test participant lists
```
data/splits/
├── splits.json
├── train.txt
├── val.txt
├── test.txt
└── split_metadata.json
```

#### 4. Train Models

```python
from src.dataio.datasets import AccelerometryForecastDataset
from torch.utils.data import DataLoader

# Load training data
dataset = AccelerometryForecastDataset(
    data_paths='data/processed/demo_000001/windows.h5',
    context_length=819,
    forecast_length=200
)

# Train your model
loader = DataLoader(dataset, batch_size=256, shuffle=True)
# ... training code ...
```

---

## Data Size Estimates

### Real UK Biobank Data

| Participants | Raw (.cwa) | Processed (HDF5) | Storage Needed |
|-------------|-----------|------------------|----------------|
| 10          | ~500 MB   | ~200 MB          | 1 GB           |
| 100         | ~5 GB     | ~2 GB            | 10 GB          |
| 1,000       | ~50 GB    | ~20 GB           | 100 GB         |
| 10,000      | ~500 GB   | ~200 GB          | 1 TB           |
| 100,000     | ~5 TB     | ~2 TB            | 10 TB          |

**Note**: Includes space for raw, processed, and intermediate files.

### Synthetic Demo Data

| Participants | Days | CSV.gz | Processed (HDF5) |
|-------------|------|--------|------------------|
| 2           | 1    | ~10 MB | ~4 MB            |
| 5           | 7    | ~175 MB| ~70 MB           |
| 50          | 7    | ~1.7 GB| ~700 MB          |
| 100         | 7    | ~3.5 GB| ~1.4 GB          |

---

## Quick Start Examples

### Example 1: Test with Small Demo

```bash
# Generate
python scripts/generate_demo_data.py --n-participants 2 --duration-days 1 --output-dir ./demo

# Process
python scripts/prepare_ukb.py --input ./demo --outdir ./demo_processed

# Splits
python scripts/make_splits.py --data-dir ./demo_processed --output-dir ./demo_splits

# Time: ~2 minutes
# Size: ~20 MB
```

### Example 2: Realistic Testing

```bash
# Generate
python scripts/generate_demo_data.py --n-participants 50 --duration-days 7 --output-dir ./realistic

# Process
python scripts/prepare_ukb.py --input ./realistic --outdir ./realistic_processed

# Splits
python scripts/make_splits.py --data-dir ./realistic_processed --output-dir ./realistic_splits

# Time: ~30 minutes
# Size: ~2 GB
```

### Example 3: Real Data Subset

```bash
# Download specific participants
python scripts/download_ukb_data.py \
    --key-file your.ukbkey \
    --dataset-id YOUR_ID \
    --participant-ids 1000001 1000002 1000003 1000004 1000005 \
    --output-dir ./real_subset

# Process
python scripts/prepare_ukb.py --input ./real_subset/raw_cwa --outdir ./real_processed

# Splits
python scripts/make_splits.py --data-dir ./real_processed --output-dir ./real_splits

# Time: ~1 hour (depends on download speed)
# Size: ~500 MB
```

---

## Support and Resources

### UK Biobank Resources

- **Main Website**: https://www.ukbiobank.ac.uk/
- **Access Management**: https://bbams.ndph.ox.ac.uk/ams/
- **Data Showcase**: https://biobank.ndph.ox.ac.uk/showcase/
- **Forum**: https://www.ukbiobank.ac.uk/learn-more-about-uk-biobank/contact-us

### Toolkit Resources

- **Repository**: https://github.com/yourusername/ukb-ttm-accelerometry
- **Documentation**: See [README.md](../README.md)
- **Issues**: https://github.com/yourusername/ukb-ttm-accelerometry/issues

### Getting Help

1. **For UK Biobank access questions**: Contact UK Biobank directly
2. **For toolkit issues**: Open a GitHub issue
3. **For data processing questions**: See [README.md](../README.md) or example_pipeline.py

---

## Frequently Asked Questions

### Can I access UK Biobank data without approval?
**No.** UK Biobank is a restricted resource. You must apply and be approved. Use synthetic data for testing.

### How long does approval take?
**4-8 weeks** typically, but can vary. Plan ahead for your research timeline.

### Can students apply?
**Yes**, but usually through their supervisor's application. Check with your institution.

### Is there a free tier?
**Academic institutions** pay lower fees (~£100-300). Commercial use costs more.

### Can I share my UK Biobank data?
**No.** Data cannot be shared. Each researcher must apply for their own access.

### What if I only want to test the toolkit?
**Use synthetic data:**
```bash
python scripts/generate_demo_data.py --n-participants 5 --duration-days 7 --output-dir ./demo
```

### How do I cite UK Biobank?
See: https://www.ukbiobank.ac.uk/enable-your-research/publish-and-disseminate

Standard citation:
```
Sudlow, C., Gallacher, J., Allen, N., et al. (2015).
UK Biobank: An Open Access Resource for Identifying the
Causes of a Wide Range of Complex Diseases of Middle and Old Age.
PLoS Medicine, 12(3), e1001779.
```

---

Last updated: 2025-11-11
