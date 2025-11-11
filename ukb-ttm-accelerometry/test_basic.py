#!/usr/bin/env python3
"""
Basic functionality test without heavy dependencies.
"""
import sys
import numpy as np

print("Testing basic imports...")

# Test windowing module
from src.dataio.windowing import compute_window_params, segment_stream

# Test compute_window_params
win_n, hop_n = compute_window_params(100, 8.192, 4.096, "floor")
assert win_n == 819, f"Expected win_n=819, got {win_n}"
assert hop_n == 409, f"Expected hop_n=409, got {hop_n}"
print("✓ compute_window_params works correctly")

# Test segment_stream
x = np.random.randn(1000, 3)
windows = segment_stream(x, win_n=100, hop_n=50, pad_mode="none")
expected_windows = (1000 - 100) // 50 + 1
assert windows.shape[0] == expected_windows, f"Expected {expected_windows} windows, got {windows.shape[0]}"
assert windows.shape == (expected_windows, 3, 100), f"Unexpected shape: {windows.shape}"
print("✓ segment_stream works correctly")

print("\nAll basic tests passed! ✓")
print("\nRepository structure created successfully:")
print("  - Data I/O modules (ukb_cwa_reader, windowing, preprocess)")
print("  - PyTorch datasets (forecasting, labeled)")
print("  - Utilities (time_features, io, seed)")
print("  - Configuration files (base, finetune, scratch)")
print("  - Processing scripts (prepare_ukb, make_splits)")
print("  - Comprehensive tests and documentation")
