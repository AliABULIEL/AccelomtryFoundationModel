# FILE: ukb_ttm_accel/data/__init__.py

"""
Data processing module for UKB TTM Accelerometry project.

Includes:
- Accelerometry data windowing and preprocessing
- Clinical metadata handling
- PyTorch Dataset implementation with HDF5 caching
- DataLoader collate functions
"""

from ukb_ttm_accel.data.windowing import (
    resample_and_calibrate,
    create_windows,
    zscore_normalize_windows,
)
from ukb_ttm_accel.data.clinical_metadata import (
    load_clinical_data,
    build_label_and_metadata,
    create_train_val_test_split,
)
from ukb_ttm_accel.data.ukb_accel_dataset import UKBAccelDataset
from ukb_ttm_accel.data.collate_fns import accel_collate_fn

__all__ = [
    "resample_and_calibrate",
    "create_windows",
    "zscore_normalize_windows",
    "load_clinical_data",
    "build_label_and_metadata",
    "create_train_val_test_split",
    "UKBAccelDataset",
    "accel_collate_fn",
]
