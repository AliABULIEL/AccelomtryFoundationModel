# FILE: ukb_ttm_accel/__init__.py

"""
UKB TTM Accelerometry Foundation Model

A modular implementation of Tiny Time-Mixer (TTM) adapted for
UK Biobank-like accelerometry data, designed for Google Colab.

This package provides:
- Data ingestion and preprocessing for accelerometry + clinical metadata
- Self-supervised domain-adaptive pretraining
- Supervised downstream fine-tuning on clinical labels
- Evaluation utilities for metrics and embedding analysis
"""

__version__ = "0.1.0"

from ukb_ttm_accel.config import TrainingConfig, load_config_from_yaml, save_config_to_yaml

__all__ = [
    "TrainingConfig",
    "load_config_from_yaml",
    "save_config_to_yaml",
]
