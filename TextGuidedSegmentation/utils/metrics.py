"""
Evaluation metrics for semantic segmentation.
Includes IoU, Dice, Precision, Recall, F1, PQ (Panoptic Quality).
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

class ConfusionMatrix:
    """
    Confusion matrix for semantic segmentation evaluation.
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset the confusion matrix."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(
        self, 
        pred: Union[np.ndarray, torch.Tensor], 
        target: Union[np.ndarray, torch.Tensor]
    ):
        """
        Update confusion matrix with predictions and targets.
        
        Args:
            pred: Predicted labels (B, H, W) or (H, W)
            target: Ground truth labels (B, H, W) or (H, W)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.flatten()
        target = target.flatten()
        
        # Filter out ignored indices
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Clip to valid range
        pred = np.clip(pred, 0, self.num_classes - 1)
        target = np.clip(target, 0, self.num_classes - 1)
        
        # Update matrix
        for p, t in zip(pred, target):
            self.matrix[t, p] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Compute all metrics from confusion matrix."""
        # Per-class metrics
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-10)
        
        # Dice
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Overall accuracy
        accuracy = tp.sum() / self.matrix.sum() if self.matrix.sum() > 0 else 0
        
        # Mean metrics
        # Only consider classes with at least one ground truth sample
        valid_classes = self.matrix.sum(axis=1) > 0
        
        metrics = {
            'mIoU': iou[valid_classes].mean() if valid_classes.any() else 0,
            'mDice': dice[valid_classes].mean() if valid_classes.any() else 0,
            'mPrecision': precision[valid_classes].mean() if valid_classes.any() else 0,
            'mRecall': recall[valid_classes].mean() if valid_classes.any() else 0,
            'mF1': f1[valid_classes].mean() if valid_classes.any() else 0,
            'accuracy': accuracy,
            'per_class_iou': iou.tolist(),
            'per_class_dice': dice.tolist(),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
        }
        
        return metrics


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def compute_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    ignore_index: int = -100,
) -> Tuple[float, List[float]]:
    """
    Compute mean IoU and per-class IoU.
    
    Args:
        pred: Predicted labels (B, H, W) or (H, W)
        target: Ground truth labels (B, H, W) or (H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        Tuple of (mean IoU, list of per-class IoU)
    """
    cm = ConfusionMatrix(num_classes, ignore_index)
    cm.update(pred, target)
    metrics = cm.get_metrics()
    return metrics['mIoU'], metrics['per_class_iou']


def compute_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    ignore_index: int = -100,
) -> Tuple[float, List[float]]:
    """
    Compute mean Dice and per-class Dice.
    """
    cm = ConfusionMatrix(num_classes, ignore_index)
    cm.update(pred, target)
    metrics = cm.get_metrics()
    return metrics['mDice'], metrics['per_class_dice']


def compute_f1(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    ignore_index: int = -100,
) -> Tuple[float, List[float]]:
    """
    Compute mean F1 and per-class F1.
    """
    cm = ConfusionMatrix(num_classes, ignore_index)
    cm.update(pred, target)
    metrics = cm.get_metrics()
    return metrics['mF1'], metrics['per_class_f1']


# =============================================================================
# PANOPTIC QUALITY (PQ) FOR INSTANCE SEGMENTATION
# =============================================================================

def compute_pq(
    pred_masks: np.ndarray,      # (H, W) instance IDs
    pred_classes: np.ndarray,    # (N,) class for each instance
    gt_masks: np.ndarray,        # (H, W) instance IDs
    gt_classes: np.ndarray,      # (M,) class for each instance
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) metric.
    PQ = (sum of IoU of matched pairs) / (|TP| + 0.5*|FP| + 0.5*|FN|)
       = SQ * RQ
    where SQ = mean IoU of matched pairs, RQ = F1 of instance matching.
    
    Args:
        pred_masks: Predicted instance segmentation
        pred_classes: Predicted class for each instance
        gt_masks: Ground truth instance segmentation
        gt_classes: Ground truth class for each instance
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching (default 0.5)
        
    Returns:
        Dictionary with PQ, SQ, RQ metrics
    """
    # Get unique instance IDs
    pred_ids = np.unique(pred_masks)
    gt_ids = np.unique(gt_masks)
    
    # Remove background (0)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = gt_ids[gt_ids > 0]
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    
    for i, gt_id in enumerate(gt_ids):
        gt_mask = pred_masks == gt_id
        for j, pred_id in enumerate(pred_ids):
            pred_mask = pred_masks == pred_id
            intersection = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            iou_matrix[i, j] = intersection / (union + 1e-10)
    
    # Match instances (greedy matching)
    tp_iou_sum = 0
    tp_count = 0
    matched_gt = set()
    matched_pred = set()
    
    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        max_iou = iou_matrix[max_idx]
        
        if max_iou < iou_threshold:
            break
        
        gt_idx, pred_idx = max_idx
        
        # Check class match
        if pred_classes[pred_idx] == gt_classes[gt_idx]:
            tp_iou_sum += max_iou
            tp_count += 1
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
        
        # Remove matched pair from consideration
        iou_matrix[gt_idx, :] = 0
        iou_matrix[:, pred_idx] = 0
    
    # Count FP and FN
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    
    # Compute metrics
    sq = tp_iou_sum / tp_count if tp_count > 0 else 0
    rq = tp_count / (tp_count + 0.5 * fp + 0.5 * fn) if (tp_count + fp + fn) > 0 else 0
    pq = sq * rq
    
    return {
        'PQ': pq,
        'SQ': sq,
        'RQ': rq,
        'TP': tp_count,
        'FP': fp,
        'FN': fn,
    }


# =============================================================================
# METRIC TRACKER
# =============================================================================

class MetricTracker:
    """
    Track metrics during training and evaluation.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = ConfusionMatrix(self.num_classes, self.ignore_index)
        self.losses = []
        self.batch_count = 0
    
    def update(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        loss: Optional[float] = None,
    ):
        """Update metrics with a batch."""
        # Handle logits (B, C, H, W) -> (B, H, W)
        if isinstance(pred, torch.Tensor) and pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        self.confusion_matrix.update(pred, target)
        
        if loss is not None:
            self.losses.append(loss)
        
        self.batch_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all computed metrics."""
        metrics = self.confusion_matrix.get_metrics()
        
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        return metrics
    
    def get_summary(self) -> str:
        """Get a formatted summary string."""
        metrics = self.get_metrics()
        
        lines = [
            f"{'='*50}",
            f"Evaluation Results",
            f"{'='*50}",
            f"Mean IoU:       {metrics['mIoU']:.4f}",
            f"Mean Dice:      {metrics['mDice']:.4f}",
            f"Mean F1:        {metrics['mF1']:.4f}",
            f"Mean Precision: {metrics['mPrecision']:.4f}",
            f"Mean Recall:    {metrics['mRecall']:.4f}",
            f"Accuracy:       {metrics['accuracy']:.4f}",
        ]
        
        if 'loss' in metrics:
            lines.append(f"Loss:           {metrics['loss']:.4f}")
        
        lines.append(f"\n{'Per-Class Results':^50}")
        lines.append(f"{'-'*50}")
        
        for i, name in enumerate(self.class_names):
            lines.append(
                f"{name:20s} | IoU: {metrics['per_class_iou'][i]:.4f} | "
                f"Dice: {metrics['per_class_dice'][i]:.4f} | "
                f"F1: {metrics['per_class_f1'][i]:.4f}"
            )
        
        return "\n".join(lines)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute pixel accuracy."""
    valid = target != ignore_index
    correct = (pred == target) & valid
    return correct.sum().item() / valid.sum().item() if valid.sum() > 0 else 0


def mean_accuracy(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    num_classes: int,
    ignore_index: int = -100
) -> float:
    """Compute mean class accuracy."""
    accuracies = []
    for c in range(num_classes):
        mask = target == c
        if mask.sum() > 0:
            correct = ((pred == c) & mask).sum().item()
            accuracies.append(correct / mask.sum().item())
    return np.mean(accuracies) if accuracies else 0


def class_wise_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Get detailed per-class metrics.
    
    Returns:
        Dictionary mapping class names to their metrics
    """
    cm = ConfusionMatrix(num_classes)
    cm.update(pred, target)
    metrics = cm.get_metrics()
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    result = {}
    for i, name in enumerate(class_names):
        result[name] = {
            'iou': metrics['per_class_iou'][i],
            'dice': metrics['per_class_dice'][i],
            'precision': metrics['per_class_precision'][i],
            'recall': metrics['per_class_recall'][i],
            'f1': metrics['per_class_f1'][i],
        }
    
    return result
