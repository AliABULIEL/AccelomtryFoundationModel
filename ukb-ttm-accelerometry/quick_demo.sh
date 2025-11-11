#!/bin/bash
# Quick demonstration of the complete pipeline using synthetic data
# This runs the entire workflow from data generation to processing

set -e  # Exit on error

echo "=========================================="
echo "UK Biobank TTM Accelerometry"
echo "Complete Pipeline Demo"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Generate synthetic accelerometry data"
echo "  2. Process it into 8.192s windows"
echo "  3. Create train/val/test splits"
echo "  4. Show usage examples"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Configuration
N_PARTICIPANTS=3
DURATION_DAYS=1
DATA_DIR="./demo_quick_test"

echo ""
echo "=========================================="
echo "Step 1: Generate Synthetic Data"
echo "=========================================="
echo "  Participants: $N_PARTICIPANTS"
echo "  Duration: $DURATION_DAYS days each"
echo "  Output: $DATA_DIR/raw"
echo ""

python3 scripts/generate_demo_data.py \
    --n-participants $N_PARTICIPANTS \
    --duration-days $DURATION_DAYS \
    --output-dir $DATA_DIR/raw \
    --seed 42

echo ""
echo "✓ Step 1 complete"
echo ""
echo "=========================================="
echo "Step 2: Process to Windows"
echo "=========================================="
echo "  Window: 8.192s (819 samples @ 100Hz)"
echo "  Overlap: 50% (4.096s hop)"
echo "  Output: $DATA_DIR/processed"
echo ""

python3 scripts/prepare_ukb.py \
    --input $DATA_DIR/raw \
    --outdir $DATA_DIR/processed \
    --win-sec 8.192 \
    --hop-sec 4.096 \
    --max-gap-ratio 0.1 \
    --log-level INFO

echo ""
echo "✓ Step 2 complete"
echo ""
echo "=========================================="
echo "Step 3: Create Splits"
echo "=========================================="
echo "  Train: 70%"
echo "  Val: 15%"
echo "  Test: 15%"
echo "  Output: $DATA_DIR/splits"
echo ""

python3 scripts/make_splits.py \
    --data-dir $DATA_DIR/processed \
    --output-dir $DATA_DIR/splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42

echo ""
echo "✓ Step 3 complete"
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
find $DATA_DIR -type f \( -name "*.h5" -o -name "*.json" -o -name "*.txt" \) | head -10

echo ""
echo "Data structure:"
tree $DATA_DIR -L 2 2>/dev/null || find $DATA_DIR -type d | head -10

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Explore the data:"
echo "   python3 -c \"from src.utils.io import load_windows_hdf5; \\"
echo "              data = load_windows_hdf5('$DATA_DIR/processed/demo_000001/windows.h5'); \\"
echo "              print(f'Windows shape: {data[\\\"windows\\\"].shape}')\""
echo ""
echo "2. Load in PyTorch:"
echo "   python3 example_pipeline.py"
echo ""
echo "3. Train a model:"
echo "   See README.md for training examples"
echo ""
echo "4. Process real UK Biobank data:"
echo "   python scripts/download_ukb_data.py --help"
echo "   See docs/UKB_DATA_ACCESS.md for access instructions"
echo ""
echo "=========================================="
echo "Demo completed successfully! ✓"
echo "=========================================="
