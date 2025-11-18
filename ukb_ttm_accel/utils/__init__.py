# FILE: ukb_ttm_accel/utils/__init__.py

"""
Utility functions for UKB TTM Accelerometry project.

Includes:
- Logging setup
- Random seed management
- Colab environment detection and setup
"""

from ukb_ttm_accel.utils.logging_utils import setup_logger, get_logger
from ukb_ttm_accel.utils.seed_utils import set_global_seed
from ukb_ttm_accel.utils.colab_env import (
    is_colab,
    install_dependencies,
    mount_drive_if_needed,
    setup_colab_env,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "set_global_seed",
    "is_colab",
    "install_dependencies",
    "mount_drive_if_needed",
    "setup_colab_env",
]
