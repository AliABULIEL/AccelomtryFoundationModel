"""
Colab-Optimized Training Script
Features:
- Automatic batch size reduction on OOM
- Google Drive checkpointing
- Gradient checkpointing for memory efficiency
- Automatic resume on disconnect
- Memory monitoring
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
import logging
import gc
from pathlib import Path
import json
import psutil
import GPUtil

logger = logging.getLogger(__name__)


class ColabOptimizedTrainer:
    """
    Trainer optimized for Google Colab constraints:
    - 12GB RAM
    - T4 GPU (16GB VRAM)
    - 12-hour session limit
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = './checkpoints',
        drive_checkpoint_dir: Optional[str] = None,
        initial_batch_size: int = 64,
        min_batch_size: int = 8,
        use_gradient_checkpointing: bool = True,
        target_gpu_memory_gb: float = 3.0,
    ):
        """
        Initialize Colab trainer.

        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            criterion: Loss function
            device: Device
            checkpoint_dir: Local checkpoint directory
            drive_checkpoint_dir: Google Drive checkpoint directory
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            use_gradient_checkpointing: Use gradient checkpointing
            target_gpu_memory_gb: Target GPU memory usage in GB
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.drive_checkpoint_dir = Path(drive_checkpoint_dir) if drive_checkpoint_dir else None
        if self.drive_checkpoint_dir:
            self.drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.target_gpu_memory_gb = target_gpu_memory_gb

        # Enable gradient checkpointing
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Initialize data loaders with automatic batch size
        self._initialize_dataloaders()

        logger.info(f"Initialized Colab trainer with batch_size={self.batch_size}")

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory by ~40%."""
        try:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            elif hasattr(self.model.ttm_model, 'gradient_checkpointing_enable'):
                self.model.ttm_model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing on TTM encoder")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    def _initialize_dataloaders(self):
        """Initialize data loaders with optimal batch size."""
        success = False
        current_batch_size = self.batch_size

        while not success and current_batch_size >= self.min_batch_size:
            try:
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=current_batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                )

                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=current_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                )

                # Test with one batch
                test_batch = next(iter(self.train_loader))
                x, y = test_batch
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(x)
                    loss = self.criterion(output['logits'], y)
                    loss.backward()

                # Clear test batch
                del x, y, output, loss
                torch.cuda.empty_cache()
                gc.collect()

                self.batch_size = current_batch_size
                success = True
                logger.info(f"Successfully initialized with batch_size={self.batch_size}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM with batch_size={current_batch_size}, reducing...")
                    current_batch_size //= 2
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e

        if not success:
            raise RuntimeError(f"Could not initialize with min_batch_size={self.min_batch_size}")

    def monitor_memory(self) -> Dict[str, float]:
        """
        Monitor system and GPU memory usage.

        Returns:
            Dictionary with memory statistics
        """
        # CPU memory
        cpu_percent = psutil.virtual_memory().percent
        cpu_gb = psutil.virtual_memory().used / 1e9

        # GPU memory
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.memoryUtil * 100
                gpu_gb = gpu.memoryUsed / 1024  # Convert MB to GB
            else:
                gpu_percent = 0
                gpu_gb = 0
        except:
            # Fallback to torch
            if torch.cuda.is_available():
                gpu_gb = torch.cuda.memory_allocated() / 1e9
                gpu_percent = (gpu_gb / torch.cuda.get_device_properties(0).total_memory) * 100
            else:
                gpu_percent = 0
                gpu_gb = 0

        return {
            'cpu_percent': cpu_percent,
            'cpu_gb': cpu_gb,
            'gpu_percent': gpu_percent,
            'gpu_gb': gpu_gb,
        }

    def check_gpu_memory(self) -> bool:
        """
        Check if GPU memory usage is within target.

        Returns:
            True if within target, False otherwise
        """
        memory = self.monitor_memory()
        return memory['gpu_gb'] <= self.target_gpu_memory_gb

    def save_checkpoint_to_drive(
        self,
        checkpoint_name: str,
        model_state: dict,
        metadata: dict
    ):
        """
        Save checkpoint to Google Drive for persistence.

        Args:
            checkpoint_name: Name of checkpoint file
            model_state: Model state dict
            metadata: Additional metadata
        """
        if self.drive_checkpoint_dir is None:
            logger.warning("Drive checkpoint directory not configured")
            return

        checkpoint_path = self.drive_checkpoint_dir / checkpoint_name

        checkpoint = {
            'model_state_dict': model_state,
            'metadata': metadata,
        }

        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to Drive: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save to Drive: {e}")

    def auto_resume(self) -> bool:
        """
        Automatically resume from latest checkpoint.

        Returns:
            True if resumed, False otherwise
        """
        # Check Drive first
        if self.drive_checkpoint_dir and self.drive_checkpoint_dir.exists():
            checkpoints = list(self.drive_checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"Resuming from Drive checkpoint: {latest_checkpoint}")
                self.load_checkpoint(str(latest_checkpoint))
                return True

        # Check local
        if self.checkpoint_dir.exists():
            checkpoints = list(self.checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"Resuming from local checkpoint: {latest_checkpoint}")
                self.load_checkpoint(str(latest_checkpoint))
                return True

        logger.info("No checkpoint found, starting from scratch")
        return False

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def optimize_for_colab(self):
        """
        Apply all Colab optimizations.
        """
        logger.info("Applying Colab optimizations...")

        # 1. Empty cache
        torch.cuda.empty_cache()
        gc.collect()

        # 2. Enable TF32 for faster computation
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for faster computation")

        # 3. Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN autotuner")

        # 4. Check memory
        memory = self.monitor_memory()
        logger.info(f"Memory: CPU={memory['cpu_gb']:.2f}GB, GPU={memory['gpu_gb']:.2f}GB")

        # 5. Reduce batch size if needed
        if memory['gpu_gb'] > self.target_gpu_memory_gb:
            logger.warning(f"GPU memory ({memory['gpu_gb']:.2f}GB) exceeds target ({self.target_gpu_memory_gb}GB)")
            logger.warning("Consider reducing batch size")

    def get_training_config(self) -> Dict:
        """Get current training configuration."""
        return {
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': max(1, 64 // self.batch_size),  # Effective batch size = 64
            'num_workers': 2,
            'pin_memory': True,
            'mixed_precision': True,
        }

    def estimate_training_time(
        self,
        epochs: int,
        samples_per_epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate training time.

        Args:
            epochs: Number of epochs
            samples_per_epoch: Samples per epoch (defaults to dataset size)

        Returns:
            Dictionary with time estimates
        """
        if samples_per_epoch is None:
            samples_per_epoch = len(self.train_dataset)

        # Benchmark one batch
        import time
        self.model.train()

        batch = next(iter(self.train_loader))
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        torch.cuda.synchronize()
        start = time.time()

        with torch.cuda.amp.autocast():
            output = self.model(x)
            loss = self.criterion(output['logits'], y)
            loss.backward()

        torch.cuda.synchronize()
        batch_time = time.time() - start

        # Extrapolate
        batches_per_epoch = samples_per_epoch // self.batch_size
        epoch_time = batch_time * batches_per_epoch
        total_time = epoch_time * epochs

        return {
            'batch_time_sec': batch_time,
            'batches_per_epoch': batches_per_epoch,
            'epoch_time_min': epoch_time / 60,
            'total_time_hours': total_time / 3600,
            'fits_in_colab_12h': total_time < 12 * 3600,
        }


def setup_colab_environment(
    mount_drive: bool = True,
    drive_path: str = '/content/drive',
) -> Dict[str, str]:
    """
    Set up Google Colab environment.

    Args:
        mount_drive: Whether to mount Google Drive
        drive_path: Path to mount Drive

    Returns:
        Dictionary with paths
    """
    logger.info("Setting up Colab environment...")

    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False
        logger.warning("Not running in Colab")

    paths = {
        'data_dir': './data',
        'checkpoint_dir': './checkpoints',
        'drive_checkpoint_dir': None,
    }

    if in_colab and mount_drive:
        try:
            from google.colab import drive
            drive.mount(drive_path)
            logger.info(f"Mounted Google Drive at {drive_path}")

            # Create checkpoint directory in Drive
            drive_checkpoint_dir = f"{drive_path}/MyDrive/ttm_accelerometry/checkpoints"
            os.makedirs(drive_checkpoint_dir, exist_ok=True)
            paths['drive_checkpoint_dir'] = drive_checkpoint_dir

        except Exception as e:
            logger.error(f"Failed to mount Drive: {e}")

    # Create local directories
    os.makedirs(paths['data_dir'], exist_ok=True)
    os.makedirs(paths['checkpoint_dir'], exist_ok=True)

    logger.info(f"Paths configured: {paths}")
    return paths
