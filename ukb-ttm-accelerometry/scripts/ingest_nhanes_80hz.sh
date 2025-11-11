#!/usr/bin/env bash
# NHANES 80 Hz data ingestion pipeline
# Downloads, converts, windows, merges clinical data, and creates splits

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default paths
DATA_DIR="${1:-data/nhanes}"
RAW80_DIR="${DATA_DIR}/raw80"
PARQUET_DIR="${DATA_DIR}/80hz_parquet"
WINDOWS_DIR="${DATA_DIR}/80hz_windows"
CLINICAL_CSV="${DATA_DIR}/clinical.csv"

# Parameters
CYCLES="${2:-2011-2012,2013-2014}"
WIN_SEC="${3:-8.192}"
HOP_SEC="${4:-4.096}"
FS="${5:-80}"

echo "========================================"
echo "NHANES 80 Hz Ingestion Pipeline"
echo "========================================"
echo "Data directory: ${DATA_DIR}"
echo "Cycles: ${CYCLES}"
echo "Window: ${WIN_SEC}s, Hop: ${HOP_SEC}s, FS: ${FS} Hz"
echo ""

# Step 1: Download raw 80 Hz data
echo "Step 1/5: Downloading raw 80 Hz data..."
if [ ! -d "${RAW80_DIR}/manifest.csv" ]; then
    Rscript src/dataio/nhanes/download_80hz.R \
        --out "${RAW80_DIR}" \
        --cycles "${CYCLES}" \
        --install-pkg
else
    echo "  → Raw data already exists, skipping download"
fi
echo ""

# Step 2: Convert to parquet
echo "Step 2/5: Converting RData to Parquet..."
if [ ! -d "${PARQUET_DIR}" ]; then
    python -m src.dataio.nhanes.convert_80hz_to_parquet \
        --input "${RAW80_DIR}" \
        --output "${PARQUET_DIR}" \
        --fs "${FS}"
else
    echo "  → Parquet files already exist, skipping conversion"
fi
echo ""

# Step 3: Parse to windows
echo "Step 3/5: Creating windows..."
python -m src.dataio.nhanes.parse_80hz \
    --input "${PARQUET_DIR}" \
    --output "${WINDOWS_DIR}" \
    --win-sec "${WIN_SEC}" \
    --hop-sec "${HOP_SEC}" \
    --fs "${FS}" \
    --pad none
echo ""

# Step 4: Merge clinical data
echo "Step 4/5: Downloading and merging clinical data..."
python -c "
from pathlib import Path
from src.dataio.nhanes.merge_clinical import (
    download_clinical_data,
    normalize_clinical_variables,
    merge_with_accelerometry,
    save_clinical_csv
)

# Get participant IDs with accelerometry
windows_dir = Path('${WINDOWS_DIR}')
accel_participants = [d.name for d in windows_dir.iterdir() if d.is_dir()]

# Download clinical data
cycles = '${CYCLES}'.split(',')
clinical_data = download_clinical_data(
    cycles,
    Path('${DATA_DIR}/clinical_raw')
)

# Normalize
clinical_df = normalize_clinical_variables(clinical_data)

# Merge
merged_df = merge_with_accelerometry(clinical_df, accel_participants)

# Save
save_clinical_csv(merged_df, Path('${CLINICAL_CSV}'))
"
echo ""

# Step 5: Create train/val/test splits
echo "Step 5/5: Creating stratified splits..."
python scripts/make_splits_nhanes.py \
    --data-dir "${WINDOWS_DIR}" \
    --clinical-csv "${CLINICAL_CSV}" \
    --output-dir "${DATA_DIR}/splits" \
    --test-ratio 0.2 \
    --val-ratio 0.1 \
    --seed 42
echo ""

echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Windows: ${WINDOWS_DIR}"
echo "Clinical: ${CLINICAL_CSV}"
echo "Splits: ${DATA_DIR}/splits/"
echo ""
echo "Next steps:"
echo "  1. Train model:"
echo "     python train.py --config src/configs/nhanes_80hz.yaml"
echo "  2. Evaluate:"
echo "     python evaluate.py --checkpoint model.ckpt"
