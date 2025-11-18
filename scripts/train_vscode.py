#!/usr/bin/env python3
"""
VS Code friendly training script
Works locally and on Colab GPU
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import argparse
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def detect_environment():
    """Detect if running in Colab, local, or remote."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        if 'SSH_CONNECTION' in os.environ or 'SSH_CLIENT' in os.environ:
            return "remote"
        return "local"


def setup_environment(env_type, args):
    """Setup environment based on type."""
    logger.info(f"Environment: {env_type}")

    paths = {
        'data_dir': args.data_dir or './data',
        'checkpoint_dir': args.checkpoint_dir or './checkpoints',
        'drive_checkpoint_dir': None,
    }

    if env_type == "colab":
        logger.info("Setting up Colab environment...")

        # Mount Drive if requested
        if args.mount_drive:
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                drive_dir = '/content/drive/MyDrive/ttm_accelerometry'
                os.makedirs(f"{drive_dir}/checkpoints", exist_ok=True)
                paths['drive_checkpoint_dir'] = f"{drive_dir}/checkpoints"
                logger.info(f"✓ Mounted Drive: {drive_dir}")
            except Exception as e:
                logger.warning(f"Could not mount Drive: {e}")

    # Create directories
    for key, path in paths.items():
        if path and not key.endswith('_dir'):
            continue
        if path:
            os.makedirs(path, exist_ok=True)

    return paths


def main():
    parser = argparse.ArgumentParser(description='Train TTM Accelerometry Classifier (VS Code)')

    # Data arguments
    parser.add_argument('--data', type=str, help='Path to HDF5 dataset')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')

    # Training arguments
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training (reduced epochs)')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Total epochs (overrides stages)')

    # Environment arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--mount-drive', action='store_true',
                       help='Mount Google Drive (Colab only)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    # VS Code specific
    parser.add_argument('--vscode', action='store_true',
                       help='Running from VS Code')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode (show plots)')

    args = parser.parse_args()

    # Detect environment
    env_type = detect_environment()
    logger.info("=" * 70)
    logger.info("TTM ACCELEROMETRY TRAINING (VS Code)")
    logger.info("=" * 70)

    # Setup environment
    paths = setup_environment(env_type, args)

    # Load configuration
    from utils.config import Config

    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # Override with CLI arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir

    # Quick mode
    if args.quick:
        logger.info("Quick mode: reducing epochs")
        config.training.stage1_epochs = 2
        config.training.stage2_epochs = 2
        config.training.stage3_epochs = 2

    # Debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check if data exists
    if args.data:
        data_path = args.data
    else:
        data_path = 'data/synthetic_data.h5'
        if not Path(data_path).exists():
            logger.warning(f"Dataset not found: {data_path}")
            logger.info("Creating synthetic dataset for demo...")
            create_synthetic_dataset(data_path)

    # Import training modules
    from data.data_loader import StreamingDataset, create_stratified_splits
    from models.ttm_classifier import TTMAccelerometryClassifier, FocalLoss, create_class_weights
    from training.trainer import ThreeStageTrainer
    from evaluation.evaluator import AccelerometryEvaluator
    from torch.utils.data import DataLoader
    import h5py

    # Load data
    logger.info("=" * 70)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 70)

    train_idx, val_idx, test_idx = create_stratified_splits(
        data_path,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.seed,
    )

    train_dataset = StreamingDataset(data_path, indices=train_idx)
    val_dataset = StreamingDataset(data_path, indices=val_idx)
    test_dataset = StreamingDataset(data_path, indices=test_idx)

    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val:   {len(val_dataset)} samples")
    logger.info(f"Test:  {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Initialize model
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

    # Setup loss
    logger.info("=" * 70)
    logger.info("STEP 3: Setting up Loss Function")
    logger.info("=" * 70)

    with h5py.File(data_path, 'r') as f:
        labels = f['labels'][train_idx]

    class_counts = {i: (labels == i).sum() for i in range(config.model.n_classes)}
    class_weights = create_class_weights(class_counts, mode='inverse')
    class_weights = class_weights.to(device)

    logger.info(f"Class distribution: {class_counts}")
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # Training
    logger.info("=" * 70)
    logger.info("STEP 4: Training")
    logger.info("=" * 70)

    trainer = ThreeStageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        checkpoint_dir=paths['drive_checkpoint_dir'] or config.training.checkpoint_dir,
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

    # Evaluation
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

    # Save results
    results_dir = Path(config.training.checkpoint_dir) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(results_dir / 'metrics.json', 'w') as f:
        # Convert to JSON-serializable format
        metrics_json = {
            'f1_macro': metrics['f1_macro'],
            'accuracy': metrics['accuracy'],
            'per_class': metrics['per_class'],
        }
        json.dump(metrics_json, f, indent=2)

    # Save model
    model_path = Path(config.training.checkpoint_dir) / 'ttm_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'metrics': metrics_json,
        'history': history,
    }, model_path)

    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved metrics to {results_dir / 'metrics.json'}")

    # Plot if interactive
    if args.interactive:
        plot_results(history, metrics, results_dir)

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final F1 Score: {metrics['f1_macro']:.4f}")
    logger.info(f"Results saved to: {results_dir}")


def create_synthetic_dataset(output_path):
    """Create synthetic dataset for demo."""
    import h5py

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    n_samples = 10000
    windows = np.random.randn(n_samples, 820, 3).astype(np.float32)
    labels = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('windows', data=windows, compression='gzip')
        f.create_dataset('labels', data=labels)

    logger.info(f"Created synthetic dataset: {output_path}")


def plot_results(history, metrics, save_dir):
    """Plot training results."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved training curves to {save_dir / 'training_curves.png'}")

    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=['Sleep', 'Sedentary', 'Light', 'MVPA'],
        yticklabels=['Sleep', 'Sedentary', 'Light', 'MVPA'],
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to {save_dir / 'confusion_matrix.png'}")


if __name__ == '__main__':
    main()
