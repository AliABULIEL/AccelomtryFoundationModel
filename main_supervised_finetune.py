# FILE: main_supervised_finetune.py

"""
Supervised fine-tuning script for TTM on clinical labels.

This script handles:
- Loading pretrained TTM checkpoint
- Fine-tuning on downstream clinical tasks
- Evaluation on validation set
- Checkpoint saving

Future implementation will include:
- TTM model loading from pretrained checkpoint
- Task-specific head attachment
- Training loop with backbone freezing options
- Metrics computation (accuracy, F1, AUC, etc.)
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ukb_ttm_accel.config import load_config_from_yaml, TrainingConfig
from ukb_ttm_accel.data import UKBAccelDataset, accel_collate_fn
from ukb_ttm_accel.utils import setup_logger, set_global_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning of pretrained TTM on clinical labels"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained TTM checkpoint from SSL phase"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume fine-tuning from"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/supervised_finetune",
        help="Directory for outputs (checkpoints, logs)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )

    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=None,
        help="Number of epochs to freeze backbone (overrides config)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, verbose logging)"
    )

    return parser.parse_args()


def main():
    """Main fine-tuning function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logger(
        name="supervised_finetune",
        log_file=str(output_dir / "finetune.log")
    )

    logger.info("="*60)
    logger.info("UKB TTM Accelerometry - Supervised Fine-Tuning")
    logger.info("="*60)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config_from_yaml(args.config)

    # Override freeze epochs if provided
    if args.freeze_backbone_epochs is not None:
        config.backbone_freeze_epochs = args.freeze_backbone_epochs

    # Set random seed
    set_global_seed(config.seed)

    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Pretrained checkpoint: {args.pretrained_checkpoint}")
    logger.info(f"  Label type: {config.label_type}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Fine-tuning epochs: {config.ft_epochs}")
    logger.info(f"  Learning rate: {config.ft_learning_rate}")
    logger.info(f"  Backbone freeze epochs: {config.backbone_freeze_epochs}")
    logger.info(f"  Device: {args.device}")

    # Future: Load pretrained model
    logger.info("Pretrained model loading will be implemented in next phase")

    # Future: Attach task-specific head
    logger.info("Task head attachment will be implemented in next phase")

    # Future: Create datasets and dataloaders
    logger.info("Dataset creation will be implemented in next phase")

    # Future: Set up optimizer and scheduler
    logger.info("Optimizer setup will be implemented in next phase")

    # Future: Training loop with optional backbone freezing
    logger.info("Training loop will be implemented in next phase")

    logger.info("Supervised fine-tuning script initialized successfully")
    logger.info("Full fine-tuning implementation coming in next phase")


if __name__ == "__main__":
    main()
