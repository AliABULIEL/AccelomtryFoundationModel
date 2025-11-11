#!/usr/bin/env bash
# NHANES 80 Hz data ingestion pipeline - Pure Python (No R!)
# Downloads XPT files directly, converts to windows, merges clinical data

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default paths
DATA_DIR="${1:-data/nhanes}"
DOWNLOAD_DIR="${DATA_DIR}/downloads"
WINDOWS_DIR="${DATA_DIR}/80hz_windows"
CLINICAL_CSV="${DATA_DIR}/clinical.csv"

# Parameters
CYCLES="${2:-2011-2012,2013-2014}"
WIN_SEC="${3:-8.192}"
HOP_SEC="${4:-4.096}"
FS="${5:-80}"

echo "========================================"
echo "NHANES 80 Hz Ingestion Pipeline (Python)"
echo "========================================"
echo "Data directory: ${DATA_DIR}"
echo "Cycles: ${CYCLES}"
echo "Window: ${WIN_SEC}s, Hop: ${HOP_SEC}s, FS: ${FS} Hz"
echo ""
echo "✅ No R dependency required!"
echo ""

# Step 1: Download XPT files using Python
echo "Step 1/4: Downloading NHANES data (Python)..."
python -m src.dataio.nhanes.download_nhanes \
    --output-dir "${DOWNLOAD_DIR}" \
    --cycles "${CYCLES}" \
    --data-types raw_80hz,demo,bmx,bpx
echo ""

# Step 2: Parse XPT to windows
echo "Step 2/4: Creating windows from XPT files..."
python -m src.dataio.nhanes.parse_80hz_xpt \
    --input "${DOWNLOAD_DIR}" \
    --output "${WINDOWS_DIR}" \
    --win-sec "${WIN_SEC}" \
    --hop-sec "${HOP_SEC}" \
    --fs "${FS}" \
    --pad none
echo ""

# Step 3: Merge clinical data
echo "Step 3/4: Merging clinical data..."
python -c "
from pathlib import Path
from src.dataio.nhanes.merge_clinical import (
    parse_xpt_file,
    normalize_clinical_variables,
    merge_with_accelerometry,
    save_clinical_csv
)

# Get participant IDs with accelerometry
windows_dir = Path('${WINDOWS_DIR}')
accel_participants = [d.name for d in windows_dir.iterdir() if d.is_dir()]

# Load clinical XPT files
download_dir = Path('${DOWNLOAD_DIR}')
clinical_dataframes = {}

for cycle_dir in download_dir.glob('*'):
    if not cycle_dir.is_dir():
        continue

    cycle = cycle_dir.name.replace('_', '-')

    for table_type in ['demo', 'bmx', 'bpx']:
        for xpt_file in cycle_dir.glob(f'{table_type.upper()}_*.XPT'):
            table_name = xpt_file.stem
            df = parse_xpt_file(xpt_file)
            if not df.empty:
                clinical_dataframes[table_name] = df

# Normalize
clinical_df = normalize_clinical_variables(clinical_dataframes)

# Merge
merged_df = merge_with_accelerometry(clinical_df, accel_participants)

# Save
save_clinical_csv(merged_df, Path('${CLINICAL_CSV}'))
"
echo ""

# Step 4: Create train/val/test splits
echo "Step 4/4: Creating stratified splits..."
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
