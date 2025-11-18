# FILE: ukb_ttm_accel/utils/colab_env.py

"""
Google Colab environment utilities.

Functions to:
- Detect if running in Colab
- Install required dependencies
- Mount Google Drive
- Set up the complete Colab environment

Example usage in Colab notebook:
    from ukb_ttm_accel.utils.colab_env import setup_colab_env
    setup_colab_env(mount_drive=True, install_deps=True)
"""

import os
import sys
import subprocess
from typing import Optional


def is_colab() -> bool:
    """
    Detect if the current environment is Google Colab.

    Returns:
        True if running in Colab, False otherwise

    Example:
        >>> if is_colab():
        ...     print("Running in Colab")
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def install_dependencies(verbose: bool = True) -> None:
    """
    Install all required dependencies for the UKB TTM project.

    This includes:
    - tsfm_public (IBM Granite Time Series Foundation Models)
    - PyTorch and related libraries
    - Transformers, Accelerate, Datasets
    - Data processing libraries (pandas, numpy, scikit-learn)
    - Accelerometry-specific libraries (actipy)
    - Storage and serialization (h5py, pyarrow, pyyaml)

    Args:
        verbose: If True, print installation progress

    Example:
        >>> install_dependencies(verbose=True)
        Installing dependencies...
        ...
    """
    if verbose:
        print("Installing dependencies for UKB TTM Accelerometry project...")

    packages = [
        # IBM Granite TSFM (Tiny Time-Mixer)
        '"tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.18"',

        # PyTorch ecosystem (Colab typically has torch pre-installed, but ensure compatibility)
        "torch",
        "torchvision",
        "torchaudio",

        # Transformers ecosystem
        "transformers",
        "accelerate",
        "datasets",

        # Data processing
        "pandas",
        "numpy",
        "scikit-learn",

        # Accelerometry processing
        "actipy",

        # Storage and serialization
        "h5py",
        "pyarrow",
        "pyyaml",

        # Plotting (useful for debugging)
        "matplotlib",
        "seaborn",
    ]

    # Construct pip install command
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages

    if verbose:
        print(f"Running: pip install {' '.join(packages[:3])} ...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if verbose:
            print("Dependencies installed successfully!")
            if result.stdout:
                print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        raise


def mount_drive_if_needed(mount_path: str = "/content/drive", force: bool = False) -> None:
    """
    Mount Google Drive if running in Colab and not already mounted.

    Args:
        mount_path: Path where Drive should be mounted (default: /content/drive)
        force: If True, force remount even if already mounted

    Example:
        >>> mount_drive_if_needed()
        Google Drive mounted at /content/drive
    """
    if not is_colab():
        print("Not running in Colab, skipping Drive mount")
        return

    if os.path.exists(mount_path) and not force:
        print(f"Google Drive already mounted at {mount_path}")
        return

    try:
        from google.colab import drive
        drive.mount(mount_path, force_remount=force)
        print(f"Google Drive mounted at {mount_path}")
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        raise


def setup_colab_env(
    mount_drive: bool = True,
    install_deps: bool = True,
    drive_path: str = "/content/drive",
    verbose: bool = True
) -> None:
    """
    Complete Colab environment setup.

    This function:
    1. Checks if running in Colab
    2. Optionally installs all dependencies
    3. Optionally mounts Google Drive
    4. Displays environment information

    Args:
        mount_drive: Whether to mount Google Drive
        install_deps: Whether to install dependencies
        drive_path: Path for Drive mount point
        verbose: Whether to print progress information

    Example:
        >>> # At the top of your Colab notebook:
        >>> from ukb_ttm_accel.utils.colab_env import setup_colab_env
        >>> setup_colab_env(mount_drive=True, install_deps=True)
    """
    if not is_colab():
        print("Not running in Google Colab. Environment setup may differ.")
        return

    if verbose:
        print("=" * 60)
        print("UKB TTM Accelerometry - Colab Environment Setup")
        print("=" * 60)

    # Install dependencies
    if install_deps:
        if verbose:
            print("\n[1/3] Installing dependencies...")
        install_dependencies(verbose=verbose)
    else:
        if verbose:
            print("\n[1/3] Skipping dependency installation")

    # Mount Google Drive
    if mount_drive:
        if verbose:
            print("\n[2/3] Mounting Google Drive...")
        mount_drive_if_needed(mount_path=drive_path)
    else:
        if verbose:
            print("\n[2/3] Skipping Google Drive mount")

    # Display environment info
    if verbose:
        print("\n[3/3] Environment information:")
        print(f"  Python version: {sys.version.split()[0]}")

        try:
            import torch
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("  PyTorch: Not installed")

        try:
            import numpy as np
            print(f"  NumPy version: {np.__version__}")
        except ImportError:
            print("  NumPy: Not installed")

        try:
            import pandas as pd
            print(f"  Pandas version: {pd.__version__}")
        except ImportError:
            print("  Pandas: Not installed")

        print("\n" + "=" * 60)
        print("Setup complete! Ready to start training.")
        print("=" * 60)
