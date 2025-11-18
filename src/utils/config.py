"""
Configuration management for TTM accelerometry system
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_dir: str = "./data"
    cache_dir: str = "./data/cache"
    window_size: int = 820  # 8.192 seconds at 100Hz
    stride: int = 410  # 50% overlap
    sample_rate: int = 100
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_cache: bool = True
    max_files: Optional[int] = None  # For testing


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "ibm-granite/granite-timeseries-ttm-r2"
    n_classes: int = 4
    n_channels: int = 3
    context_length: int = 512
    hidden_dim: int = 256
    dropout: float = 0.3
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    freeze_encoder: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Stage 1: Linear Probe
    stage1_epochs: int = 10
    stage1_lr: float = 1e-3
    stage1_weight_decay: float = 0.01

    # Stage 2: LoRA
    stage2_epochs: int = 20
    stage2_lr: float = 1e-4
    stage2_weight_decay: float = 0.01

    # Stage 3: Full fine-tuning
    stage3_epochs: int = 10
    stage3_lr: float = 1e-5
    stage3_weight_decay: float = 0.01

    # General
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_amp: bool = True
    num_workers: int = 2

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1000
    log_every: int = 100


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    use_ssl: bool = True
    use_standard: bool = True
    time_warp_range: tuple = (0.85, 1.15)
    n_permutation_segments: int = 4
    p_augment: float = 0.5
    noise_std: float = 0.01
    rotation_max_angle: float = 15.0


@dataclass
class ColabConfig:
    """Colab-specific configuration."""
    mount_drive: bool = True
    drive_path: str = "/content/drive"
    use_gradient_checkpointing: bool = True
    target_gpu_memory_gb: float = 3.0
    min_batch_size: int = 8


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)

    # Metadata
    experiment_name: str = "ttm_accelerometry"
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            augmentation=AugmentationConfig(**config_dict.get('augmentation', {})),
            colab=ColabConfig(**config_dict.get('colab', {})),
            experiment_name=config_dict.get('experiment_name', 'ttm_accelerometry'),
            seed=config_dict.get('seed', 42),
        )

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'augmentation': self.augmentation.__dict__,
            'colab': self.colab.__dict__,
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def __repr__(self) -> str:
        """String representation."""
        lines = [
            f"Config(experiment_name='{self.experiment_name}', seed={self.seed})",
            f"  Data: window_size={self.data.window_size}, sample_rate={self.data.sample_rate}",
            f"  Model: {self.model.model_name}, n_classes={self.model.n_classes}",
            f"  Training: batch_size={self.training.batch_size}, AMP={self.training.use_amp}",
        ]
        return '\n'.join(lines)
