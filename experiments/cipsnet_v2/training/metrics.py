"""
Performance Metrics for CIPS-Net V2

Fast metrics computed every epoch during validation:
- NP: Dice Score, IoU, Precision, Recall, F1
- Type: Accuracy, F1 (macro), F1 (per-class), Balanced Accuracy

These metrics are computed on the pixel level and are fast to calculate.
mPQ and bPQ are computed separately after training (require watershed post-processing).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class SegmentationMetrics:
    """
    Compute segmentation metrics for NP (nuclei presence) head.
    
    Metrics:
    - Dice Score: 2 * |A ∩ B| / (|A| + |B|)
    - IoU: |A ∩ B| / |A ∪ B|
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1: 2 * Precision * Recall / (Precision + Recall)
    """
    
    def __init__(self, smooth: float = 1e-6):
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.tn = 0  # True Negatives
        self.intersection = 0
        self.union = 0
        self.pred_sum = 0
        self.target_sum = 0
        self.num_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: NP predictions [B, 2, H, W] (logits) or [B, H, W] (binary)
            target: NP targets [B, H, W] or [B, 1, H, W] (binary 0/1)
        """
        # Handle logits -> binary
        if pred.dim() == 4 and pred.size(1) == 2:
            pred = torch.argmax(pred, dim=1)  # [B, H, W]
        
        # Handle target with channel dimension
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, H, W]
        
        # Ensure binary
        pred = (pred > 0).float()
        target = (target > 0).float()
        
        # Flatten for calculation
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute TP, FP, FN, TN
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
        
        # Accumulate
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        
        # For Dice/IoU
        intersection = tp
        union = tp + fp + fn
        
        self.intersection += intersection
        self.union += union
        self.pred_sum += pred_flat.sum().item()
        self.target_sum += target_flat.sum().item()
        self.num_samples += pred.size(0)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        # Dice Score
        dice = (2 * self.intersection + self.smooth) / (self.pred_sum + self.target_sum + self.smooth)
        
        # IoU
        iou = (self.intersection + self.smooth) / (self.union + self.smooth)
        
        # Precision, Recall, F1
        precision = (self.tp + self.smooth) / (self.tp + self.fp + self.smooth)
        recall = (self.tp + self.smooth) / (self.tp + self.fn + self.smooth)
        f1 = 2 * precision * recall / (precision + recall + self.smooth)
        
        return {
            'np_dice': dice,
            'np_iou': iou,
            'np_precision': precision,
            'np_recall': recall,
            'np_f1': f1,
        }


class ClassificationMetrics:
    """
    Compute classification metrics for Type head.
    
    Metrics:
    - Accuracy: Overall pixel-wise accuracy
    - F1 (macro): Average F1 across all classes
    - F1 (per-class): F1 for each class
    - Balanced Accuracy: Average recall per class
    
    Classes (PanNuke):
    0: Background
    1: Neoplastic
    2: Inflammatory
    3: Connective/Soft tissue
    4: Dead
    5: Epithelial
    """
    
    CLASS_NAMES = [
        'Background',
        'Neoplastic', 
        'Inflammatory',
        'Connective',
        'Dead',
        'Epithelial'
    ]
    
    def __init__(self, num_classes: int = 6, ignore_background: bool = False, smooth: float = 1e-6):
        """
        Args:
            num_classes: Number of classes (including background)
            ignore_background: Whether to ignore background (class 0) in metrics
            smooth: Smoothing factor
        """
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        # Confusion matrix: rows = true, cols = pred
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.num_samples = 0
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Type predictions [B, C, H, W] (logits) or [B, H, W] (class indices)
            target: Type targets [B, H, W] (class indices)
            mask: Optional mask to only consider certain pixels [B, H, W]
        """
        # Handle logits -> class indices
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)  # [B, H, W]
        
        # Handle target with channel dimension
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Flatten
        pred_flat = pred.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1).cpu().numpy().astype(bool)
            pred_flat = pred_flat[mask_flat]
            target_flat = target_flat[mask_flat]
        
        # Update confusion matrix
        for t, p in zip(target_flat, pred_flat):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[int(t), int(p)] += 1
        
        self.num_samples += pred.size(0)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {}
        
        # Start class index (1 if ignoring background, 0 otherwise)
        start_idx = 1 if self.ignore_background else 0
        
        # Per-class metrics
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        
        for i in range(start_idx, self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            precision = (tp + self.smooth) / (tp + fp + self.smooth)
            recall = (tp + self.smooth) / (tp + fn + self.smooth)
            f1 = 2 * precision * recall / (precision + recall + self.smooth)
            
            per_class_precision.append(precision)
            per_class_recall.append(recall)
            per_class_f1.append(f1)
            
            # Add per-class F1
            class_name = self.CLASS_NAMES[i] if i < len(self.CLASS_NAMES) else f'class_{i}'
            metrics[f'type_f1_{class_name.lower()}'] = f1
        
        # Overall accuracy
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        accuracy = correct / (total + self.smooth)
        
        # Macro F1 (average across classes)
        macro_f1 = np.mean(per_class_f1)
        
        # Balanced accuracy (average recall)
        balanced_acc = np.mean(per_class_recall)
        
        # Macro Precision and Recall
        macro_precision = np.mean(per_class_precision)
        macro_recall = np.mean(per_class_recall)
        
        metrics.update({
            'type_accuracy': accuracy,
            'type_f1_macro': macro_f1,
            'type_precision_macro': macro_precision,
            'type_recall_macro': macro_recall,
            'type_balanced_accuracy': balanced_acc,
        })
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return self.confusion_matrix.copy()


class ValidationMetrics:
    """
    Combined metrics for validation.
    
    Computes all fast metrics:
    - NP: Dice, IoU, Precision, Recall, F1
    - Type: Accuracy, F1 (macro), F1 (per-class), Balanced Accuracy
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        ignore_background_for_type: bool = False,
    ):
        """
        Args:
            num_classes: Number of type classes
            ignore_background_for_type: Whether to ignore background in type metrics
        """
        self.segmentation_metrics = SegmentationMetrics()
        self.classification_metrics = ClassificationMetrics(
            num_classes=num_classes,
            ignore_background=ignore_background_for_type,
        )
    
    def reset(self):
        """Reset all metrics."""
        self.segmentation_metrics.reset()
        self.classification_metrics.reset()
    
    def update(
        self,
        pred_np: torch.Tensor,
        pred_type: torch.Tensor,
        target_np: torch.Tensor,
        target_type: torch.Tensor,
        focus_mask: Optional[torch.Tensor] = None,
    ):
        """
        Update all metrics with a batch.
        
        Args:
            pred_np: NP predictions [B, 2, H, W]
            pred_type: Type predictions [B, C, H, W]
            target_np: NP targets [B, H, W] or [B, 1, H, W]
            target_type: Type targets [B, H, W]
            focus_mask: Optional mask for type metrics (nuclei only)
        """
        # Update NP metrics
        self.segmentation_metrics.update(pred_np, target_np)
        
        # Update Type metrics (optionally only on nuclei pixels)
        if focus_mask is not None:
            self.classification_metrics.update(pred_type, target_type, mask=focus_mask)
        else:
            # Use NP target as mask (only evaluate on nuclei pixels)
            np_mask = (target_np > 0).squeeze(1) if target_np.dim() == 4 else (target_np > 0)
            self.classification_metrics.update(pred_type, target_type, mask=np_mask)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        metrics.update(self.segmentation_metrics.compute())
        metrics.update(self.classification_metrics.compute())
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get type classification confusion matrix."""
        return self.classification_metrics.get_confusion_matrix()


def compute_batch_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute metrics for a single batch (quick computation).
    
    Args:
        outputs: Model outputs with 'np', 'hv', 'type' keys
        targets: Targets with 'np', 'hv', 'type' keys
    
    Returns:
        Dict of metric values
    """
    metrics = {}
    
    # NP metrics (quick)
    pred_np = torch.argmax(outputs['np'], dim=1)  # [B, H, W]
    target_np = targets['np']
    if target_np.dim() == 4:
        target_np = target_np.squeeze(1)
    
    # Dice for batch
    pred_flat = pred_np.view(-1).float()
    target_flat = (target_np > 0).view(-1).float()
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    metrics['batch_np_dice'] = dice.item()
    
    # Type accuracy for batch (on nuclei pixels only)
    pred_type = torch.argmax(outputs['type'], dim=1)  # [B, H, W]
    target_type = targets['type']
    if target_type.dim() == 4:
        target_type = target_type.squeeze(1)
    
    nuclei_mask = target_np > 0
    if nuclei_mask.sum() > 0:
        correct = (pred_type[nuclei_mask] == target_type[nuclei_mask]).float().mean()
        metrics['batch_type_acc'] = correct.item()
    else:
        metrics['batch_type_acc'] = 0.0
    
    return metrics


# ==========================================================================
# Testing
# ==========================================================================

def test_metrics():
    """Test metrics computation."""
    print("Testing Performance Metrics...")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = 4, 256, 256
    num_classes = 6
    
    # Create dummy predictions and targets
    print("\n1. Creating dummy data...")
    pred_np = torch.randn(B, 2, H, W, device=device)
    pred_type = torch.randn(B, num_classes, H, W, device=device)
    
    target_np = torch.randint(0, 2, (B, H, W), device=device)
    target_type = torch.randint(0, num_classes, (B, H, W), device=device)
    
    print(f"   pred_np: {pred_np.shape}")
    print(f"   pred_type: {pred_type.shape}")
    print(f"   target_np: {target_np.shape}")
    print(f"   target_type: {target_type.shape}")
    
    # Test SegmentationMetrics
    print("\n2. Testing SegmentationMetrics...")
    seg_metrics = SegmentationMetrics()
    seg_metrics.update(pred_np, target_np)
    seg_results = seg_metrics.compute()
    print(f"   Results:")
    for name, value in seg_results.items():
        print(f"     {name}: {value:.4f}")
    
    # Test ClassificationMetrics
    print("\n3. Testing ClassificationMetrics...")
    cls_metrics = ClassificationMetrics(num_classes=num_classes)
    nuclei_mask = target_np > 0
    cls_metrics.update(pred_type, target_type, mask=nuclei_mask)
    cls_results = cls_metrics.compute()
    print(f"   Results:")
    for name, value in cls_results.items():
        print(f"     {name}: {value:.4f}")
    
    # Test confusion matrix
    print("\n4. Testing Confusion Matrix...")
    cm = cls_metrics.get_confusion_matrix()
    print(f"   Shape: {cm.shape}")
    print(f"   Total pixels: {cm.sum()}")
    
    # Test ValidationMetrics (combined)
    print("\n5. Testing ValidationMetrics (combined)...")
    val_metrics = ValidationMetrics(num_classes=num_classes)
    
    # Simulate multiple batches
    for i in range(3):
        pred_np_i = torch.randn(B, 2, H, W, device=device)
        pred_type_i = torch.randn(B, num_classes, H, W, device=device)
        target_np_i = torch.randint(0, 2, (B, H, W), device=device)
        target_type_i = torch.randint(0, num_classes, (B, H, W), device=device)
        
        val_metrics.update(pred_np_i, pred_type_i, target_np_i, target_type_i)
    
    all_results = val_metrics.compute()
    print(f"   Results (3 batches accumulated):")
    for name, value in all_results.items():
        print(f"     {name}: {value:.4f}")
    
    # Test batch metrics
    print("\n6. Testing quick batch metrics...")
    outputs = {'np': pred_np, 'hv': torch.randn(B, 2, H, W, device=device), 'type': pred_type}
    targets = {'np': target_np, 'hv': torch.randn(B, 2, H, W, device=device), 'type': target_type}
    batch_metrics = compute_batch_metrics(outputs, targets)
    print(f"   Batch NP Dice: {batch_metrics['batch_np_dice']:.4f}")
    print(f"   Batch Type Acc: {batch_metrics['batch_type_acc']:.4f}")
    
    print("\n" + "=" * 60)
    print("All metrics tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics()
