# FILE: ukb_ttm_accel/config.py

"""
Configuration management for UKB TTM Accelerometry project.

Provides dataclass-based configuration for all training hyperparameters,
with YAML serialization support.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Complete configuration for training TTM on accelerometry data.

    Sections:
    - Data configuration
    - SSL pretraining hyperparameters
    - Fine-tuning hyperparameters
    - Optimization settings
    - Colab/IO settings
    - Evaluation settings
    """

    # Data configuration
    window_seconds: int = 10
    sampling_rate_hz: int = 50
    batch_size: int = 32
    num_workers: int = 2
    max_windows_per_participant: Optional[int] = None
    overlap_fraction: float = 0.0
    normalize_per_window: bool = True

    # Data paths (to be set by user)
    accel_data_dir: str = "/content/drive/MyDrive/ukb_accel"
    clinical_csv_path: str = "/content/drive/MyDrive/ukb_accel/clinical.csv"
    cache_hdf5_path: Optional[str] = None

    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Clinical metadata
    label_type: str = "bmi_class"
    metadata_columns: List[str] = field(default_factory=lambda: ["age", "sex", "bmi"])

    # SSL pretraining hyperparameters
    ssl_epochs: int = 50
    ssl_learning_rate: float = 1e-3
    ssl_mask_ratio: float = 0.3
    ssl_patch_length: int = 16
    ssl_context_length: int = 512

    # Fine-tuning hyperparameters
    ft_epochs: int = 30
    ft_learning_rate: float = 1e-4
    backbone_freeze_epochs: int = 5

    # Optimization settings
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    scheduler_type: str = "cosine"

    # Colab/IO settings
    use_mixed_precision: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every_n_epochs: int = 5
    device: str = "cuda"

    # Evaluation settings
    val_interval: int = 1
    early_stopping_patience: int = 10
    metric_for_best_model: str = "val_loss"

    # Model architecture (TTM specific)
    num_layers: int = 4
    d_model: int = 64
    num_heads: int = 4
    dropout: float = 0.1

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.window_seconds > 0, "window_seconds must be positive"
        assert self.sampling_rate_hz > 0, "sampling_rate_hz must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0.0 <= self.overlap_fraction < 1.0, "overlap_fraction must be in [0, 1)"
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "train/val/test ratios must sum to 1.0"
        assert 0.0 <= self.ssl_mask_ratio <= 1.0, "ssl_mask_ratio must be in [0, 1]"

        # Create directories if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


def load_config_from_yaml(path: str) -> TrainingConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        TrainingConfig instance

    Example:
        >>> config = load_config_from_yaml("config.yaml")
        >>> print(config.batch_size)
        32
    """
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return TrainingConfig(**config_dict)


def save_config_to_yaml(config: TrainingConfig, path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: TrainingConfig instance to save
        path: Path where YAML file will be written

    Example:
        >>> config = TrainingConfig(batch_size=64)
        >>> save_config_to_yaml(config, "my_config.yaml")
    """
    config_dict = asdict(config)

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {path}")
