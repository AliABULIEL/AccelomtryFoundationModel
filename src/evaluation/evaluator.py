"""
Evaluation Module for CAPTURE-24 Benchmark
Implements comprehensive metrics with confidence intervals
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from scipy import stats
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class AccelerometryEvaluator:
    """
    Comprehensive evaluator for accelerometry classification.

    Metrics:
    - F1 score (macro, weighted, per-class)
    - Precision and recall
    - Confusion matrix
    - Confidence intervals via bootstrapping
    - Inference speed
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize evaluator.

        Args:
            class_names: Names of activity classes
            n_bootstrap: Number of bootstrap samples for CI
            confidence_level: Confidence level for intervals
        """
        if class_names is None:
            class_names = ['Sleep', 'Sedentary', 'Light', 'MVPA']

        self.class_names = class_names
        self.n_classes = len(class_names)
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        return_predictions: bool = False,
        use_amp: bool = True,
    ) -> Dict:
        """
        Comprehensive evaluation on test set.

        Args:
            model: Trained model
            dataloader: Test data loader
            device: Device
            return_predictions: Whether to return predictions
            use_amp: Use automatic mixed precision

        Returns:
            Dictionary with metrics and statistics
        """
        logger.info("Running evaluation...")

        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        inference_times = []

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Evaluating"):
                x = x.to(device)
                y = y.to(device)

                # Measure inference time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(x)
                    logits = output['logits']

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = time.time() - start_time

                # Predictions
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                inference_times.append(inference_time)

        # Concatenate
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels, all_probs)

        # Add inference speed
        metrics['inference_speed'] = {
            'mean_batch_time_ms': np.mean(inference_times) * 1000,
            'std_batch_time_ms': np.std(inference_times) * 1000,
            'mean_per_sample_ms': np.mean(inference_times) / dataloader.batch_size * 1000,
        }

        # Add classification report
        metrics['classification_report'] = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            digits=4,
        )

        if return_predictions:
            metrics['predictions'] = all_preds
            metrics['labels'] = all_labels
            metrics['probabilities'] = all_probs

        return metrics

    def _calculate_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
    ) -> Dict:
        """Calculate comprehensive metrics."""

        # Basic metrics
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        f1_per_class = f1_score(labels, preds, average=None)

        precision_macro = precision_score(labels, preds, average='macro')
        recall_macro = recall_score(labels, preds, average='macro')

        accuracy = (preds == labels).mean()

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        # Per-class metrics with confidence intervals
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            # Get samples for this class
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_preds = preds[class_mask]
                class_labels = labels[class_mask]

                # F1 score with CI
                f1, f1_ci = self._bootstrap_metric(
                    class_labels, class_preds, f1_score, average='binary'
                )

                per_class_metrics[class_name] = {
                    'f1': float(f1_per_class[i]),
                    'f1_ci': f1_ci,
                    'precision': float(precision_score(labels, preds, labels=[i], average='micro')),
                    'recall': float(recall_score(labels, preds, labels=[i], average='micro')),
                    'support': int(class_mask.sum()),
                }

        # Overall metrics with confidence intervals
        f1_macro_ci = self._bootstrap_metric(
            labels, preds, f1_score, average='macro'
        )[1]

        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_macro_ci': f1_macro_ci,
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'confusion_matrix': cm.tolist(),
            'per_class': per_class_metrics,
            'n_samples': len(labels),
        }

        return metrics

    def _bootstrap_metric(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        metric_fn,
        **metric_kwargs
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate metric with bootstrap confidence interval.

        Args:
            labels: True labels
            preds: Predictions
            metric_fn: Metric function (e.g., f1_score)
            **metric_kwargs: Additional arguments for metric function

        Returns:
            metric_value: Point estimate
            ci: Confidence interval (lower, upper)
        """
        n_samples = len(labels)

        # Point estimate
        metric_value = metric_fn(labels, preds, **metric_kwargs)

        # Bootstrap
        bootstrap_scores = []
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            labels_boot = labels[indices]
            preds_boot = preds[indices]

            # Skip if only one class in bootstrap sample
            if len(np.unique(labels_boot)) < 2:
                continue

            try:
                score = metric_fn(labels_boot, preds_boot, **metric_kwargs)
                bootstrap_scores.append(score)
            except:
                continue

        # Calculate confidence interval
        if len(bootstrap_scores) > 0:
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
            ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
            ci = (float(ci_lower), float(ci_upper))
        else:
            ci = (float(metric_value), float(metric_value))

        return float(metric_value), ci

    def evaluate_with_uncertainty(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        n_mc_samples: int = 10,
    ) -> Dict:
        """
        Evaluate with uncertainty estimation using Monte Carlo dropout.

        Args:
            model: Model with dropout layers
            dataloader: Test data loader
            device: Device
            n_mc_samples: Number of MC samples

        Returns:
            Metrics including uncertainty estimates
        """
        logger.info(f"Evaluating with uncertainty ({n_mc_samples} MC samples)...")

        all_preds = []
        all_labels = []
        all_uncertainties = []

        for x, y in tqdm(dataloader, desc="MC Evaluation"):
            x = x.to(device)
            y = y.to(device)

            # MC dropout predictions
            mean_probs, uncertainty = model.predict_with_uncertainty(
                x, n_samples=n_mc_samples
            )

            preds = mean_probs.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_uncertainties = np.concatenate(all_uncertainties)

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_preds, all_labels, np.zeros((len(all_preds), self.n_classes))
        )

        # Add uncertainty statistics
        metrics['uncertainty'] = {
            'mean': float(all_uncertainties.mean()),
            'std': float(all_uncertainties.std()),
            'min': float(all_uncertainties.min()),
            'max': float(all_uncertainties.max()),
        }

        # Uncertainty by correctness
        correct_mask = all_preds == all_labels
        metrics['uncertainty']['correct'] = float(all_uncertainties[correct_mask].mean())
        metrics['uncertainty']['incorrect'] = float(all_uncertainties[~correct_mask].mean())

        return metrics

    def print_results(self, metrics: Dict):
        """
        Print formatted results.

        Args:
            metrics: Metrics dictionary from evaluate()
        """
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):      {metrics['f1_macro']:.4f} "
              f"[{metrics['f1_macro_ci'][0]:.4f}, {metrics['f1_macro_ci'][1]:.4f}]")
        print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
        print(f"  Precision:       {metrics['precision_macro']:.4f}")
        print(f"  Recall:          {metrics['recall_macro']:.4f}")

        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name:12s}: F1={class_metrics['f1']:.4f} "
                  f"[{class_metrics['f1_ci'][0]:.4f}, {class_metrics['f1_ci'][1]:.4f}], "
                  f"Precision={class_metrics['precision']:.4f}, "
                  f"Recall={class_metrics['recall']:.4f}, "
                  f"Support={class_metrics['support']}")

        # Inference speed
        if 'inference_speed' in metrics:
            speed = metrics['inference_speed']
            print(f"\nInference Speed:")
            print(f"  Per sample: {speed['mean_per_sample_ms']:.2f} ± "
                  f"{speed['std_batch_time_ms'] / np.sqrt(metrics['n_samples']):.2f} ms")
            print(f"  Per batch:  {speed['mean_batch_time_ms']:.2f} ± "
                  f"{speed['std_batch_time_ms']:.2f} ms")

        # Classification report
        if 'classification_report' in metrics:
            print(f"\nDetailed Classification Report:")
            print(metrics['classification_report'])

        print("=" * 70 + "\n")

    def plot_confusion_matrix(
        self,
        metrics: Dict,
        save_path: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Plot confusion matrix.

        Args:
            metrics: Metrics dictionary
            save_path: Path to save figure
            normalize: Normalize by true labels
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(metrics['confusion_matrix'])

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.show()


def benchmark_capture24(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_f1: float = 0.85,
) -> Dict:
    """
    Benchmark model on CAPTURE-24 standard.

    Target: >0.85 F1 without smoothing, >0.90 with HMM smoothing.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        target_f1: Target F1 score

    Returns:
        Benchmark results
    """
    logger.info("Running CAPTURE-24 benchmark...")

    evaluator = AccelerometryEvaluator()

    # Evaluate
    metrics = evaluator.evaluate(model, test_loader, device)

    # Check if meets target
    meets_target = metrics['f1_macro'] >= target_f1

    results = {
        'metrics': metrics,
        'target_f1': target_f1,
        'achieved_f1': metrics['f1_macro'],
        'meets_target': meets_target,
        'gap_to_target': target_f1 - metrics['f1_macro'],
    }

    # Print results
    print("\n" + "=" * 70)
    print("CAPTURE-24 BENCHMARK")
    print("=" * 70)
    print(f"Target F1:    {target_f1:.4f}")
    print(f"Achieved F1:  {metrics['f1_macro']:.4f} "
          f"[{metrics['f1_macro_ci'][0]:.4f}, {metrics['f1_macro_ci'][1]:.4f}]")
    print(f"Status:       {'✓ PASS' if meets_target else '✗ FAIL'}")
    print("=" * 70 + "\n")

    return results
