#!/usr/bin/env python3
"""
Quick start training script for TTM Accelerometry System
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_loader import AccelerometryDataLoader, StreamingDataset, create_stratified_splits
from models.ttm_classifier import TTMAccelerometryClassifier, FocalLoss, create_class_weights
from training.trainer import ThreeStageTrainer
from evaluation.evaluator import AccelerometryEvaluator, benchmark_capture24
from utils.config import Config
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train TTM Accelerometry Classifier')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training (reduced epochs for testing)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # Override checkpoint dir
    config.training.checkpoint_dir = args.checkpoint_dir

    # Quick mode
    if args.quick:
        logger.info("Quick mode: reducing epochs for testing")
        config.training.stage1_epochs = 2
        config.training.stage2_epochs = 2
        config.training.stage3_epochs = 2

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # ====================================================================
    # 1. Load Data
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 70)

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    # Create splits
    train_idx, val_idx, test_idx = create_stratified_splits(
        args.data,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.seed,
    )

    # Create datasets
    train_dataset = StreamingDataset(args.data, indices=train_idx)
    val_dataset = StreamingDataset(args.data, indices=val_idx)
    test_dataset = StreamingDataset(args.data, indices=test_idx)

    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val:   {len(val_dataset)} samples")
    logger.info(f"Test:  {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # ====================================================================
    # 2. Initialize Model
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STEP 2: Initializing Model")
    logger.info("=" * 70)

    model = TTMAccelerometryClassifier(
        model_name=config.model.model_name,
        n_classes=config.model.n_classes,
        n_channels=config.model.n_channels,
        context_length=config.model.context_length,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        freeze_encoder=config.model.freeze_encoder,
    )

    model = model.to(device)

    stats = model.get_parameter_stats()
    logger.info(f"Total parameters:     {stats['total']:,}")
    logger.info(f"Trainable parameters: {stats['trainable']:,}")

    # ====================================================================
    # 3. Setup Loss Function
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STEP 3: Setting up Loss Function")
    logger.info("=" * 70)

    # Calculate class weights from training data
    import h5py
    with h5py.File(args.data, 'r') as f:
        labels = f['labels'][train_idx]

    class_counts = {i: (labels == i).sum() for i in range(config.model.n_classes)}
    class_weights = create_class_weights(class_counts, mode='inverse')
    class_weights = class_weights.to(device)

    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # ====================================================================
    # 4. Training
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STEP 4: Training")
    logger.info("=" * 70)

    trainer = ThreeStageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        checkpoint_dir=config.training.checkpoint_dir,
        use_amp=config.training.use_amp,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
    )

    history = trainer.run_full_pipeline(
        stage1_epochs=config.training.stage1_epochs,
        stage2_epochs=config.training.stage2_epochs,
        stage3_epochs=config.training.stage3_epochs,
        stage1_lr=config.training.stage1_lr,
        stage2_lr=config.training.stage2_lr,
        stage3_lr=config.training.stage3_lr,
    )

    # ====================================================================
    # 5. Evaluation
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STEP 5: Evaluation")
    logger.info("=" * 70)

    evaluator = AccelerometryEvaluator(
        class_names=['Sleep', 'Sedentary', 'Light', 'MVPA'],
    )

    metrics = evaluator.evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    evaluator.print_results(metrics)

    # CAPTURE-24 Benchmark
    benchmark_results = benchmark_capture24(
        model=model,
        test_loader=test_loader,
        device=device,
        target_f1=0.85,
    )

    # ====================================================================
    # 6. Save Model
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STEP 6: Saving Model")
    logger.info("=" * 70)

    model_path = Path(config.training.checkpoint_dir) / 'ttm_accelerometry_final.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'metrics': metrics,
        'history': history,
    }, model_path)

    logger.info(f"Saved model to {model_path}")

    # ====================================================================
    # Summary
    # ====================================================================
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final F1 Score: {metrics['f1_macro']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
