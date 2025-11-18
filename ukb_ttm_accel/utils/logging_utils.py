# FILE: ukb_ttm_accel/utils/logging_utils.py

"""
Logging utilities for UKB TTM Accelerometry project.

Provides a simple logger factory with console and file output support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ukb_ttm_accel",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file. If None, only console logging
        level: Logging level (default: INFO)
        format_string: Custom format string. If None, uses default format

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("my_experiment", log_file="train.log")
        >>> logger.info("Training started")
        2024-01-15 10:30:00 - my_experiment - INFO - Training started
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format with timestamp, name, level, and message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ukb_ttm_accel") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger()
        >>> logger.info("Processing data...")
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger
