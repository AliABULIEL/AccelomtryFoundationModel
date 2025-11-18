"""
3-Stage Training Pipeline for TTM Accelerometry Classifier
Optimized for Google Colab (12GB RAM, T4 GPU)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, List, Tuple
import logging
from tqdm import tqdm
import time
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreeStageTrainer:
    """
    Three-stage training pipeline:
    1. Linear Probe: Freeze encoder, train classifier only
    2. LoRA Fine-tuning: Add LoRA adapters, train with low rank
    3. Full Fine-tuning: Unfreeze all, full fine-tuning
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = './checkpoints',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        log_every: int = 100,
        save_every: int = 1000,
    ):
        """
        Initialize trainer.

        Args:
            model: TTM classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            device: Device (cuda/cpu)
            checkpoint_dir: Directory for checkpoints
            use_amp: Use automatic mixed precision (FP16)
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            log_every: Log frequency (steps)
            save_every: Save frequency (steps)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.save_every = save_every

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.current_stage = None
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)

        logger.info(f"Initialized trainer with AMP={use_amp}")

    def stage1_linear_probe(
        self,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> Dict:
        """
        Stage 1: Linear Probe
        Freeze encoder, train classifier head only.

        Args:
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay

        Returns:
            Training history
        """
        logger.info("=" * 50)
        logger.info("Stage 1: Linear Probe (Frozen Encoder)")
        logger.info("=" * 50)

        self.current_stage = 'stage1'

        # Freeze encoder
        self.model.freeze_encoder()

        # Optimizer (only classifier parameters)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * epochs // self.gradient_accumulation_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        # Training loop
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")

            train_metrics = self._train_epoch(optimizer, scheduler)
            val_metrics = self._validate()

            # Log
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint('stage1_best.pt', epoch)

        logger.info(f"Stage 1 complete. Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def stage2_lora_finetuning(
        self,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ) -> Dict:
        """
        Stage 2: LoRA Fine-tuning
        Enable LoRA adapters, train with parameter-efficient fine-tuning.

        Args:
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha

        Returns:
            Training history
        """
        logger.info("=" * 50)
        logger.info("Stage 2: LoRA Fine-tuning")
        logger.info("=" * 50)

        self.current_stage = 'stage2'

        # Enable LoRA
        if not self.model.use_lora:
            self.model.enable_lora(rank=lora_rank, alpha=lora_alpha)

        # Optimizer (LoRA parameters + classifier)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler with warmup
        total_steps = len(self.train_loader) * epochs // self.gradient_accumulation_steps
        warmup_steps = total_steps // 10

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")

            train_metrics = self._train_epoch(optimizer, scheduler)
            val_metrics = self._validate()

            # Log
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint('stage2_best.pt', epoch)

        logger.info(f"Stage 2 complete. Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def stage3_full_finetuning(
        self,
        epochs: int = 10,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> Dict:
        """
        Stage 3: Full Fine-tuning
        Unfreeze all parameters, full fine-tuning with low learning rate.

        Args:
            epochs: Number of epochs
            lr: Learning rate (low)
            weight_decay: Weight decay

        Returns:
            Training history
        """
        logger.info("=" * 50)
        logger.info("Stage 3: Full Fine-tuning")
        logger.info("=" * 50)

        self.current_stage = 'stage3'

        # Unfreeze encoder
        self.model.unfreeze_encoder()

        # Optimizer (all parameters)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler with warmup
        total_steps = len(self.train_loader) * epochs // self.gradient_accumulation_steps
        warmup_steps = total_steps // 10

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")

            train_metrics = self._train_epoch(optimizer, scheduler)
            val_metrics = self._validate()

            # Log
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint('stage3_best.pt', epoch)

        logger.info(f"Stage 3 complete. Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> Dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                output = self.model(x)
                logits = output['logits']
                loss = self.criterion(logits, y)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                self.global_step += 1

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(
                        f'{self.current_stage}_step{self.global_step}.pt'
                    )

            # Metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': total_correct / total_samples,
            })

        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': total_correct / total_samples,
        }

        return metrics

    def _validate(self) -> Dict:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation"):
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                with autocast(enabled=self.use_amp):
                    output = self.model(x)
                    logits = output['logits']
                    loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': total_correct / total_samples,
        }

        return metrics

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics."""
        logger.info(
            f"Epoch {epoch+1} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Store history
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])

    def _save_checkpoint(self, filename: str, epoch: Optional[int] = None):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history),
            'current_stage': self.current_stage,
        }

        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = defaultdict(list, checkpoint.get('history', {}))
        self.current_stage = checkpoint.get('current_stage', None)

        logger.info(f"Loaded checkpoint from step {self.global_step}")

    def run_full_pipeline(
        self,
        stage1_epochs: int = 10,
        stage2_epochs: int = 20,
        stage3_epochs: int = 10,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 1e-4,
        stage3_lr: float = 1e-5,
    ):
        """
        Run complete 3-stage training pipeline.

        Args:
            stage1_epochs: Epochs for stage 1
            stage2_epochs: Epochs for stage 2
            stage3_epochs: Epochs for stage 3
            stage1_lr: Learning rate for stage 1
            stage2_lr: Learning rate for stage 2
            stage3_lr: Learning rate for stage 3
        """
        logger.info("Starting 3-stage training pipeline")
        logger.info("=" * 50)

        start_time = time.time()

        # Stage 1
        self.stage1_linear_probe(epochs=stage1_epochs, lr=stage1_lr)

        # Stage 2
        self.stage2_lora_finetuning(epochs=stage2_epochs, lr=stage2_lr)

        # Stage 3
        self.stage3_full_finetuning(epochs=stage3_epochs, lr=stage3_lr)

        total_time = time.time() - start_time

        logger.info("=" * 50)
        logger.info("Training pipeline complete!")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 50)

        return self.history
