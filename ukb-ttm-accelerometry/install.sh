#!/bin/bash
# Installation script for UK Biobank TTM Accelerometry Toolkit
# Handles the proper installation order for tsfm and dependencies

set -e  # Exit on error

echo "=========================================="
echo "UK Biobank TTM Accelerometry Installation"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python >= 3.9
required_version="3.9"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "ERROR: Python 3.9 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Install IBM tsfm from GitHub
echo "Installing IBM tsfm from GitHub..."
echo "This may take a few minutes..."
pip install git+https://github.com/IBM/tsfm.git

if [ $? -eq 0 ]; then
    echo "✓ IBM tsfm installed successfully"
else
    echo "ERROR: Failed to install IBM tsfm"
    echo "Please check your internet connection and try again"
    exit 1
fi
echo ""

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully"
else
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo ""

# Optional: Install in development mode
read -p "Install package in development mode? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e .
    echo "✓ Package installed in development mode"
fi
echo ""

# Run basic tests
echo "Running basic functionality tests..."
python3 test_basic.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation completed successfully! ✓"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Prepare your data: python scripts/prepare_ukb.py --help"
    echo "  2. Create splits: python scripts/make_splits.py --help"
    echo "  3. See README.md for complete documentation"
    echo ""
else
    echo ""
    echo "WARNING: Basic tests failed"
    echo "Installation may be incomplete"
    exit 1
fi
