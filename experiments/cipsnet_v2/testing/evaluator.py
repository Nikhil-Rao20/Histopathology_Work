"""
PanNuke Evaluator for CIPS-Net V2
==================================

Comprehensive evaluation using official HoVer-Net metrics.

Metrics Computed:
-----------------
1. Instance Segmentation Metrics (bPQ):
   - Dice (binary foreground)
   - AJI (Aggregated Jaccard Index)
   - AJI+ (with unique pairing)
   - DQ (Detection Quality)
   - SQ (Segmentation Quality)  
   - bPQ (binary Panoptic Quality = DQ * SQ)

2. Multi-class Metrics (mPQ):
   - Per-class PQ for each nucleus type
   - mPQ (mean PQ across classes)

3. Detection Metrics:
   - Per-class Precision, Recall, F1
   - Overall Detection F1
   - Overall Accuracy

4. Tissue-wise Breakdown:
   - All metrics computed per tissue type

Output:
-------
- JSON results file with all metrics
- CSV tables for paper (tissue-wise, class-wise)
- Summary statistics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from our metrics module (official HoVer-Net metrics)
from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_pq,
    remap_label,
    pair_coordinates,
)

# Import post-processing
from testing.post_processing import PostProcessor, PostProcessConfig

# Import dataset info
from datasets.pannuke import PANNUKE_CLASSES, PANNUKE_CLASS_NAMES, PANNUKE_TISSUES


# ==========================================================================
# Class-wise PQ Computation
# ==========================================================================

def get_class_pq(
    true_inst: np.ndarray,
    true_type: np.ndarray,
    pred_inst: np.ndarray,
    pred_type: np.ndarray,
    num_classes: int = 6,
    match_iou: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Panoptic Quality per class (for mPQ).
    
    For each class:
    1. Extract instances of that class from pred and true
    2. Compute PQ for those instances only
    
    Args:
        true_inst: Ground truth instance map [H, W]
        true_type: Ground truth type map [H, W]
        pred_inst: Predicted instance map [H, W]
        pred_type: Predicted type map [H, W]
        num_classes: Number of classes (including background)
        match_iou: IoU threshold for matching
        
    Returns:
        Dict with per-class PQ scores and mPQ
    """
    results = {}
    pq_list = []
    
    # Skip background (class 0)
    for class_id in range(1, num_classes):
        class_name = PANNUKE_CLASSES.get(class_id, f'class_{class_id}')
        
        # Get instances of this class
        true_class_mask = (true_type == class_id) & (true_inst > 0)
        pred_class_mask = (pred_type == class_id) & (pred_inst > 0)
        
        # Create class-specific instance maps
        true_class_inst = np.zeros_like(true_inst)
        pred_class_inst = np.zeros_like(pred_inst)
        
        # Remap instances for this class
        true_class_inst[true_class_mask] = true_inst[true_class_mask]
        pred_class_inst[pred_class_mask] = pred_inst[pred_class_mask]
        
        # Remap to contiguous labels
        true_class_inst = remap_label(true_class_inst)
        pred_class_inst = remap_label(pred_class_inst)
        
        # Check if any instances exist
        true_count = len(np.unique(true_class_inst)) - 1  # Exclude 0
        pred_count = len(np.unique(pred_class_inst)) - 1
        
        if true_count == 0 and pred_count == 0:
            # No instances of this class - skip (don't include in mPQ)
            results[f'pq_{class_name}'] = np.nan
            continue
        elif true_count == 0:
            # Only FP - PQ = 0
            results[f'pq_{class_name}'] = 0.0
            pq_list.append(0.0)
            continue
        elif pred_count == 0:
            # Only FN - PQ = 0
            results[f'pq_{class_name}'] = 0.0
            pq_list.append(0.0)
            continue
        
        # Compute PQ for this class
        pq_info, _ = get_fast_pq(true_class_inst, pred_class_inst, match_iou=match_iou)
        dq, sq, pq = pq_info
        
        results[f'pq_{class_name}'] = pq
        results[f'dq_{class_name}'] = dq
        results[f'sq_{class_name}'] = sq
        pq_list.append(pq)
    
    # Compute mPQ (mean over classes with instances)
    if pq_list:
        results['mPQ'] = np.nanmean(pq_list)
    else:
        results['mPQ'] = 0.0
    
    return results


# ==========================================================================
# Detection Metrics
# ==========================================================================

def compute_detection_metrics(
    true_inst: np.ndarray,
    true_type: np.ndarray,
    pred_inst: np.ndarray,
    pred_type: np.ndarray,
    num_classes: int = 6,
    distance_threshold: float = 12.0,
) -> Dict[str, float]:
    """
    Compute detection metrics (Precision, Recall, F1) per class and overall.
    
    Uses centroid-based matching with Hungarian algorithm.
    
    Args:
        true_inst: Ground truth instance map
        true_type: Ground truth type map
        pred_inst: Predicted instance map
        pred_type: Predicted type map
        num_classes: Number of classes
        distance_threshold: Max distance for centroid matching
        
    Returns:
        Dict with per-class and overall detection metrics
    """
    from scipy.ndimage import center_of_mass, label as scipy_label
    
    # Extract centroids and types for each instance
    def get_instance_info(inst_map, type_map):
        """Get list of (centroid, type) for each instance."""
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        centroids = []
        types = []
        
        for inst_id in unique_ids:
            mask = (inst_map == inst_id)
            
            # Centroid
            y_coords, x_coords = np.where(mask)
            cx, cy = np.mean(x_coords), np.mean(y_coords)
            centroids.append([cx, cy])
            
            # Type (majority vote)
            inst_types = type_map[mask]
            type_counts = np.bincount(inst_types, minlength=num_classes)
            if type_counts[1:].sum() > 0:
                type_counts[0] = 0
            assigned_type = np.argmax(type_counts)
            types.append(assigned_type)
        
        return np.array(centroids) if centroids else np.zeros((0, 2)), np.array(types)
    
    true_centroids, true_types = get_instance_info(true_inst, true_type)
    pred_centroids, pred_types = get_instance_info(pred_inst, pred_type)
    
    results = {}
    
    # Overall detection (regardless of type)
    if len(true_centroids) == 0 and len(pred_centroids) == 0:
        # No nuclei at all
        results['detection_f1'] = 1.0
        results['detection_precision'] = 1.0
        results['detection_recall'] = 1.0
    elif len(true_centroids) == 0:
        # Only FP
        results['detection_f1'] = 0.0
        results['detection_precision'] = 0.0
        results['detection_recall'] = 1.0
    elif len(pred_centroids) == 0:
        # Only FN
        results['detection_f1'] = 0.0
        results['detection_precision'] = 1.0
        results['detection_recall'] = 0.0
    else:
        # Match using Hungarian algorithm
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroids.astype(np.float32),
            pred_centroids.astype(np.float32),
            distance_threshold
        )
        
        tp = len(paired)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['detection_precision'] = precision
        results['detection_recall'] = recall
        results['detection_f1'] = f1
        
        # Classification accuracy on matched pairs
        if tp > 0:
            matched_true_types = true_types[paired[:, 0]]
            matched_pred_types = pred_types[paired[:, 1]]
            correct = (matched_true_types == matched_pred_types).sum()
            results['classification_accuracy'] = correct / tp
        else:
            results['classification_accuracy'] = 0.0
    
    # Per-class detection
    for class_id in range(1, num_classes):
        class_name = PANNUKE_CLASSES.get(class_id, f'class_{class_id}')
        
        # Filter by class
        true_class_mask = (true_types == class_id)
        pred_class_mask = (pred_types == class_id)
        
        true_class_centroids = true_centroids[true_class_mask] if len(true_centroids) > 0 else np.zeros((0, 2))
        pred_class_centroids = pred_centroids[pred_class_mask] if len(pred_centroids) > 0 else np.zeros((0, 2))
        
        n_true = len(true_class_centroids)
        n_pred = len(pred_class_centroids)
        
        if n_true == 0 and n_pred == 0:
            results[f'precision_{class_name}'] = np.nan
            results[f'recall_{class_name}'] = np.nan
            results[f'f1_{class_name}'] = np.nan
            continue
        elif n_true == 0:
            results[f'precision_{class_name}'] = 0.0
            results[f'recall_{class_name}'] = 1.0
            results[f'f1_{class_name}'] = 0.0
            continue
        elif n_pred == 0:
            results[f'precision_{class_name}'] = 1.0
            results[f'recall_{class_name}'] = 0.0
            results[f'f1_{class_name}'] = 0.0
            continue
        
        # Match for this class
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_class_centroids.astype(np.float32),
            pred_class_centroids.astype(np.float32),
            distance_threshold
        )
        
        tp = len(paired)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[f'precision_{class_name}'] = precision
        results[f'recall_{class_name}'] = recall
        results[f'f1_{class_name}'] = f1
    
    return results


# ==========================================================================
# Main Evaluator Class
# ==========================================================================

@dataclass
class EvaluationConfig:
    """Evaluation configuration.
    
    Tuned defaults from post_processing_tuning.ipynb (Feb 2026).
    """
    # Post-processing (tuned values — see post_processing_tuning.ipynb)
    np_threshold: float = 0.525
    min_instance_size: int = 70
    marker_erosion_size: int = 3
    
    # Metrics
    match_iou: float = 0.5  # IoU threshold for PQ matching
    distance_threshold: float = 12.0  # Distance threshold for detection
    
    # Output
    save_predictions: bool = False  # Save predictions as .mat files
    output_dir: str = "evaluation_results"


class PanNukeEvaluator:
    """
    Comprehensive evaluator for PanNuke dataset.
    
    Computes all official metrics + tissue-wise breakdown.
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
            device: Device for model inference
        """
        self.config = config or EvaluationConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize post-processor
        pp_config = PostProcessConfig(
            np_threshold=self.config.np_threshold,
            min_instance_size=self.config.min_instance_size,
            marker_erosion_size=self.config.marker_erosion_size,
        )
        self.post_processor = PostProcessor(pp_config)
        
        # Storage for accumulating results
        self.reset()
    
    def reset(self):
        """Reset accumulated results."""
        # Per-image results
        self.all_results = []
        
        # Tissue-wise accumulation
        self.tissue_results = defaultdict(list)
        
        # Overall accumulation
        self.overall_metrics = {
            'dice': [],
            'aji': [],
            'aji_plus': [],
            'dq': [],
            'sq': [],
            'pq': [],  # bPQ
        }
        
        # Per-class PQ accumulation
        self.class_pq = defaultdict(list)
        
        # Detection accumulation
        self.detection_results = []
    
    def add_batch(
        self,
        pred_np: torch.Tensor,
        pred_hv: torch.Tensor,
        pred_type: torch.Tensor,
        gt_instances: np.ndarray,
        gt_types: np.ndarray,
        tissues: List[str],
        indices: List[int],
    ):
        """
        Add a batch of predictions for evaluation.
        
        This is the main interface for the ExperimentRunner.
        
        Args:
            pred_np: NP predictions [B, 2, H, W]
            pred_hv: HV predictions [B, 2, H, W]
            pred_type: Type predictions [B, C, H, W]
            gt_instances: Ground truth instance maps [B, H, W]
            gt_types: Ground truth type maps [B, H, W]
            tissues: List of tissue names
            indices: List of sample indices
        """
        # Convert tensors to numpy
        if isinstance(pred_np, torch.Tensor):
            pred_np = pred_np.detach().cpu().numpy()
        if isinstance(pred_hv, torch.Tensor):
            pred_hv = pred_hv.detach().cpu().numpy()
        if isinstance(pred_type, torch.Tensor):
            pred_type = pred_type.detach().cpu().numpy()
        
        batch_size = pred_np.shape[0]
        
        for i in range(batch_size):
            self.evaluate_single(
                pred_np=pred_np[i],
                pred_hv=pred_hv[i],
                pred_type=pred_type[i],
                true_inst=gt_instances[i],
                true_type=gt_types[i],
                tissue=tissues[i] if isinstance(tissues[i], str) else str(tissues[i]),
                image_id=str(indices[i]),
            )
    
    def add_batch_from_postprocessed(
        self,
        postprocessed_results: list,
        gt_instances: np.ndarray,
        gt_types: np.ndarray,
        tissues: list,
        indices: list,
    ):
        """
        Add a batch of already-post-processed predictions for evaluation.
        
        Used by decoders with custom post-processing (e.g., LVIT_IE).
        
        Args:
            postprocessed_results: List of dicts from custom post-processor,
                each with 'inst_map' [H,W] and 'type_map' [H,W]
            gt_instances: Ground truth instance maps [B, H, W]
            gt_types: Ground truth type maps [B, H, W]
            tissues: List of tissue names
            indices: List of sample indices
        """
        for i, result in enumerate(postprocessed_results):
            pred_inst = result['inst_map']
            pred_type_map = result['type_map']
            true_inst = gt_instances[i]
            true_type = gt_types[i]
            tissue = tissues[i] if isinstance(tissues[i], str) else str(tissues[i])
            image_id = str(indices[i])
            
            self._evaluate_from_maps(
                pred_inst=pred_inst,
                pred_type_map=pred_type_map,
                true_inst=true_inst,
                true_type=true_type,
                tissue=tissue,
                image_id=image_id,
            )
    
    def _evaluate_from_maps(
        self,
        pred_inst: np.ndarray,
        pred_type_map: np.ndarray,
        true_inst: np.ndarray,
        true_type: np.ndarray,
        tissue: str = "unknown",
        image_id: str = "unknown",
    ) -> Dict[str, float]:
        """
        Evaluate from pre-computed instance and type maps (skips post-processing).
        
        Args:
            pred_inst: Predicted instance map [H, W]
            pred_type_map: Predicted type map [H, W]
            true_inst: Ground truth instance map [H, W]
            true_type: Ground truth type map [H, W]
            tissue: Tissue type name
            image_id: Image identifier
        """
        # Remap labels for metric computation
        true_inst_remap = remap_label(true_inst.astype(np.int32))
        pred_inst_remap = remap_label(pred_inst.astype(np.int32))
        
        results = {
            'image_id': image_id,
            'tissue': tissue,
        }
        
        # Binary Dice
        results['dice'] = get_dice_1(true_inst_remap, pred_inst_remap)
        
        # AJI
        if true_inst_remap.max() > 0 and pred_inst_remap.max() > 0:
            results['aji'] = get_fast_aji(true_inst_remap, pred_inst_remap)
            results['aji_plus'] = get_fast_aji_plus(true_inst_remap, pred_inst_remap)
        else:
            results['aji'] = 0.0
            results['aji_plus'] = 0.0
        
        # PQ (DQ, SQ, bPQ)
        if true_inst_remap.max() > 0 or pred_inst_remap.max() > 0:
            pq_info, _ = get_fast_pq(
                true_inst_remap, pred_inst_remap,
                match_iou=self.config.match_iou
            )
            results['dq'] = pq_info[0]
            results['sq'] = pq_info[1]
            results['bpq'] = pq_info[2]
        else:
            results['dq'] = 1.0
            results['sq'] = 1.0
            results['bpq'] = 1.0
        
        # Multi-class PQ (mPQ)
        class_pq_results = get_class_pq(
            true_inst_remap, true_type.astype(np.int32),
            pred_inst_remap, pred_type_map.astype(np.int32),
            num_classes=6,
            match_iou=self.config.match_iou,
        )
        results.update(class_pq_results)
        
        # Detection metrics
        detection_results = compute_detection_metrics(
            true_inst_remap, true_type.astype(np.int32),
            pred_inst_remap, pred_type_map.astype(np.int32),
            num_classes=6,
            distance_threshold=self.config.distance_threshold,
        )
        results.update(detection_results)
        
        # Accumulate
        self._accumulate(results, tissue)
        
        return results
    
    def compute_and_save(self, output_dir: str) -> Dict[str, Any]:
        """
        Compute final metrics and save results.
        
        This is the main finalization interface for ExperimentRunner.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Results dictionary suitable for aggregation
        """
        summary = self.save_results(output_dir)
        
        # Format results for the aggregator
        results = {
            'overall': {
                'dice': summary['overall'].get('dice_mean', float('nan')),
                'aji': summary['overall'].get('aji_mean', float('nan')),
                'aji_plus': summary['overall'].get('aji_plus_mean', float('nan')),
                'dq': summary['overall'].get('dq_mean', float('nan')),
                'sq': summary['overall'].get('sq_mean', float('nan')),
                'bpq': summary['overall'].get('pq_mean', float('nan')),
                'mpq': summary['overall'].get('mPQ_mean', float('nan')),
            },
            'class_wise': {},
            'tissue_wise': {},
            'detection': {
                'overall_precision': summary['detection'].get('detection_precision_mean', float('nan')),
                'overall_recall': summary['detection'].get('detection_recall_mean', float('nan')),
                'overall_f1': summary['detection'].get('detection_f1_mean', float('nan')),
            },
        }
        
        # Add class-wise
        for class_name, metrics in summary['class_wise'].items():
            results['class_wise'][class_name] = {
                'pq_mean': metrics.get('pq_mean', float('nan')),
                'pq_std': metrics.get('pq_std', float('nan')),
            }
        
        # Add tissue-wise
        for tissue, metrics in summary['tissue_wise'].items():
            results['tissue_wise'][tissue] = {
                'dice': metrics.get('dice_mean', float('nan')),
                'aji': metrics.get('aji_mean', float('nan')),
                'bpq': metrics.get('bpq_mean', float('nan')),
                'mpq': metrics.get('mPQ_mean', float('nan')),
                'n_images': metrics.get('n_images', 0),
            }
        
        return results
    
    def evaluate_single(
        self,
        pred_np: np.ndarray,
        pred_hv: np.ndarray,
        pred_type: np.ndarray,
        true_inst: np.ndarray,
        true_type: np.ndarray,
        tissue: str = "unknown",
        image_id: str = "unknown",
    ) -> Dict[str, float]:
        """
        Evaluate a single image.
        
        Args:
            pred_np: NP prediction [2, H, W] or [H, W]
            pred_hv: HV prediction [2, H, W]
            pred_type: Type prediction [C, H, W] or [H, W]
            true_inst: Ground truth instance map [H, W]
            true_type: Ground truth type map [H, W]
            tissue: Tissue type name
            image_id: Image identifier
            
        Returns:
            Dict with all computed metrics
        """
        # Post-process predictions
        pred_result = self.post_processor.process(pred_np, pred_hv, pred_type)
        pred_inst = pred_result['inst_map']
        pred_type_map = pred_result['type_map']
        
        # Remap labels for metric computation
        true_inst_remap = remap_label(true_inst.astype(np.int32))
        pred_inst_remap = remap_label(pred_inst.astype(np.int32))
        
        results = {
            'image_id': image_id,
            'tissue': tissue,
        }
        
        # === Instance Segmentation Metrics (bPQ) ===
        
        # Binary Dice
        results['dice'] = get_dice_1(true_inst_remap, pred_inst_remap)
        
        # AJI
        if true_inst_remap.max() > 0 and pred_inst_remap.max() > 0:
            results['aji'] = get_fast_aji(true_inst_remap, pred_inst_remap)
            results['aji_plus'] = get_fast_aji_plus(true_inst_remap, pred_inst_remap)
        else:
            results['aji'] = 0.0
            results['aji_plus'] = 0.0
        
        # PQ (DQ, SQ, bPQ)
        if true_inst_remap.max() > 0 or pred_inst_remap.max() > 0:
            pq_info, _ = get_fast_pq(
                true_inst_remap, pred_inst_remap, 
                match_iou=self.config.match_iou
            )
            results['dq'] = pq_info[0]
            results['sq'] = pq_info[1]
            results['bpq'] = pq_info[2]
        else:
            results['dq'] = 1.0
            results['sq'] = 1.0
            results['bpq'] = 1.0
        
        # === Multi-class Metrics (mPQ) ===
        class_pq_results = get_class_pq(
            true_inst_remap, true_type.astype(np.int32),
            pred_inst_remap, pred_type_map.astype(np.int32),
            num_classes=6,
            match_iou=self.config.match_iou,
        )
        results.update(class_pq_results)
        
        # === Detection Metrics ===
        detection_results = compute_detection_metrics(
            true_inst_remap, true_type.astype(np.int32),
            pred_inst_remap, pred_type_map.astype(np.int32),
            num_classes=6,
            distance_threshold=self.config.distance_threshold,
        )
        results.update(detection_results)
        
        # Accumulate
        self._accumulate(results, tissue)
        
        return results
    
    def _accumulate(self, results: Dict[str, float], tissue: str):
        """Accumulate results for summary statistics."""
        # Store full results
        self.all_results.append(results)
        
        # Overall metrics
        for metric in ['dice', 'aji', 'aji_plus', 'dq', 'sq']:
            if metric in results:
                self.overall_metrics[metric].append(results[metric])
        if 'bpq' in results:
            self.overall_metrics['pq'].append(results['bpq'])
        
        # Tissue-wise
        self.tissue_results[tissue].append(results)
        
        # Per-class PQ
        for class_name in PANNUKE_CLASS_NAMES:
            key = f'pq_{class_name}'
            if key in results and not np.isnan(results[key]):
                self.class_pq[class_name].append(results[key])
        
        # mPQ
        if 'mPQ' in results:
            self.class_pq['mPQ'].append(results['mPQ'])
    
    def evaluate_batch(
        self,
        pred_np: torch.Tensor,
        pred_hv: torch.Tensor,
        pred_type: torch.Tensor,
        true_inst: torch.Tensor,
        true_type: torch.Tensor,
        tissues: List[str],
        image_ids: List[str],
    ) -> List[Dict[str, float]]:
        """
        Evaluate a batch of images.
        
        Args:
            pred_np: [B, 2, H, W]
            pred_hv: [B, 2, H, W]
            pred_type: [B, C, H, W]
            true_inst: [B, H, W]
            true_type: [B, H, W]
            tissues: List of tissue names
            image_ids: List of image IDs
            
        Returns:
            List of result dicts
        """
        # Convert to numpy
        pred_np = pred_np.detach().cpu().numpy()
        pred_hv = pred_hv.detach().cpu().numpy()
        pred_type = pred_type.detach().cpu().numpy()
        true_inst = true_inst.detach().cpu().numpy()
        true_type = true_type.detach().cpu().numpy()
        
        results = []
        batch_size = pred_np.shape[0]
        
        for i in range(batch_size):
            result = self.evaluate_single(
                pred_np=pred_np[i],
                pred_hv=pred_hv[i],
                pred_type=pred_type[i],
                true_inst=true_inst[i],
                true_type=true_type[i],
                tissue=tissues[i],
                image_id=image_ids[i],
            )
            results.append(result)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dict with:
                - overall: Overall metrics (mean ± std)
                - tissue_wise: Metrics per tissue
                - class_wise: Per-class PQ
                - detection: Detection metrics summary
        """
        summary = {}
        
        # === Overall Metrics ===
        overall = {}
        for metric, values in self.overall_metrics.items():
            if values:
                overall[f'{metric}_mean'] = np.mean(values)
                overall[f'{metric}_std'] = np.std(values)
        
        # mPQ
        if self.class_pq['mPQ']:
            overall['mPQ_mean'] = np.mean(self.class_pq['mPQ'])
            overall['mPQ_std'] = np.std(self.class_pq['mPQ'])
        
        summary['overall'] = overall
        
        # === Tissue-wise Metrics ===
        tissue_wise = {}
        for tissue, results_list in self.tissue_results.items():
            tissue_metrics = {}
            
            # Aggregate metrics
            for metric in ['dice', 'aji', 'aji_plus', 'dq', 'sq', 'bpq', 'mPQ']:
                values = [r.get(metric, np.nan) for r in results_list]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    tissue_metrics[f'{metric}_mean'] = np.mean(values)
                    tissue_metrics[f'{metric}_std'] = np.std(values)
                    tissue_metrics[f'{metric}_count'] = len(values)
            
            tissue_metrics['n_images'] = len(results_list)
            tissue_wise[tissue] = tissue_metrics
        
        summary['tissue_wise'] = tissue_wise
        
        # === Class-wise PQ ===
        class_wise = {}
        for class_name in PANNUKE_CLASS_NAMES:
            if self.class_pq[class_name]:
                class_wise[class_name] = {
                    'pq_mean': np.mean(self.class_pq[class_name]),
                    'pq_std': np.std(self.class_pq[class_name]),
                    'n_samples': len(self.class_pq[class_name]),
                }
        
        summary['class_wise'] = class_wise
        
        # === Detection Summary ===
        detection = {}
        det_metrics = ['detection_f1', 'detection_precision', 'detection_recall', 
                       'classification_accuracy']
        for metric in det_metrics:
            values = [r.get(metric, np.nan) for r in self.all_results]
            values = [v for v in values if not np.isnan(v)]
            if values:
                detection[f'{metric}_mean'] = np.mean(values)
                detection[f'{metric}_std'] = np.std(values)
        
        # Per-class detection
        for class_name in PANNUKE_CLASS_NAMES:
            for metric_type in ['precision', 'recall', 'f1']:
                key = f'{metric_type}_{class_name}'
                values = [r.get(key, np.nan) for r in self.all_results]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    detection[f'{key}_mean'] = np.mean(values)
        
        summary['detection'] = detection
        
        return summary
    
    def save_results(self, output_dir: str):
        """
        Save all results to files.
        
        Creates:
            - results.json: Full results
            - summary.json: Summary statistics
            - tissue_wise.csv: Tissue-wise table for paper
            - class_wise.csv: Class-wise PQ table
            - detection.csv: Detection metrics table
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get summary
        summary = self.get_summary()
        
        # Save full results
        with open(output_path / 'results.json', 'w') as f:
            # Convert numpy types
            json_results = []
            for r in self.all_results:
                json_r = {}
                for k, v in r.items():
                    if isinstance(v, (np.float32, np.float64)):
                        json_r[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        json_r[k] = v.tolist()
                    else:
                        json_r[k] = v
                json_results.append(json_r)
            json.dump(json_results, f, indent=2)
        
        # Save summary
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        
        # === Tissue-wise CSV ===
        tissue_data = []
        for tissue, metrics in summary['tissue_wise'].items():
            row = {'Tissue': tissue}
            for key in ['dice_mean', 'aji_mean', 'bpq_mean', 'mPQ_mean', 'n_images']:
                row[key.replace('_mean', '').upper() if '_mean' in key else key] = metrics.get(key, np.nan)
            tissue_data.append(row)
        
        if tissue_data:
            tissue_df = pd.DataFrame(tissue_data)
            tissue_df.to_csv(output_path / 'tissue_wise.csv', index=False)
        
        # === Class-wise CSV ===
        class_data = []
        for class_name, metrics in summary['class_wise'].items():
            row = {
                'Class': class_name,
                'PQ_mean': metrics.get('pq_mean', np.nan),
                'PQ_std': metrics.get('pq_std', np.nan),
                'n_samples': metrics.get('n_samples', 0),
            }
            class_data.append(row)
        
        if class_data:
            class_df = pd.DataFrame(class_data)
            class_df.to_csv(output_path / 'class_wise.csv', index=False)
        
        # === Detection CSV ===
        detection_data = []
        for class_name in PANNUKE_CLASS_NAMES:
            row = {
                'Class': class_name,
                'Precision': summary['detection'].get(f'precision_{class_name}_mean', np.nan),
                'Recall': summary['detection'].get(f'recall_{class_name}_mean', np.nan),
                'F1': summary['detection'].get(f'f1_{class_name}_mean', np.nan),
            }
            detection_data.append(row)
        
        # Add overall
        detection_data.append({
            'Class': 'OVERALL',
            'Precision': summary['detection'].get('detection_precision_mean', np.nan),
            'Recall': summary['detection'].get('detection_recall_mean', np.nan),
            'F1': summary['detection'].get('detection_f1_mean', np.nan),
        })
        
        detection_df = pd.DataFrame(detection_data)
        detection_df.to_csv(output_path / 'detection.csv', index=False)
        
        print(f"\n[Evaluator] Results saved to: {output_path}")
        
        return summary
    
    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        # Overall
        print("\n[Overall Metrics]")
        overall = summary['overall']
        print(f"  Dice:    {overall.get('dice_mean', 0):.4f} ± {overall.get('dice_std', 0):.4f}")
        print(f"  AJI:     {overall.get('aji_mean', 0):.4f} ± {overall.get('aji_std', 0):.4f}")
        print(f"  AJI+:    {overall.get('aji_plus_mean', 0):.4f} ± {overall.get('aji_plus_std', 0):.4f}")
        print(f"  DQ:      {overall.get('dq_mean', 0):.4f} ± {overall.get('dq_std', 0):.4f}")
        print(f"  SQ:      {overall.get('sq_mean', 0):.4f} ± {overall.get('sq_std', 0):.4f}")
        print(f"  bPQ:     {overall.get('pq_mean', 0):.4f} ± {overall.get('pq_std', 0):.4f}")
        print(f"  mPQ:     {overall.get('mPQ_mean', 0):.4f} ± {overall.get('mPQ_std', 0):.4f}")
        
        # Class-wise PQ
        print("\n[Class-wise PQ]")
        for class_name, metrics in summary['class_wise'].items():
            print(f"  {class_name:15s}: {metrics.get('pq_mean', 0):.4f} ± {metrics.get('pq_std', 0):.4f}")
        
        # Detection
        print("\n[Detection Metrics]")
        detection = summary['detection']
        print(f"  Overall F1:     {detection.get('detection_f1_mean', 0):.4f}")
        print(f"  Overall Prec:   {detection.get('detection_precision_mean', 0):.4f}")
        print(f"  Overall Recall: {detection.get('detection_recall_mean', 0):.4f}")
        print(f"  Class Accuracy: {detection.get('classification_accuracy_mean', 0):.4f}")
        
        print("\n  Per-class F1:")
        for class_name in PANNUKE_CLASS_NAMES:
            f1 = detection.get(f'f1_{class_name}_mean', np.nan)
            if not np.isnan(f1):
                print(f"    {class_name:15s}: {f1:.4f}")
        
        # Tissue-wise (first 5)
        print("\n[Tissue-wise bPQ (top 5)]")
        tissue_bpq = [(t, m.get('bpq_mean', 0)) for t, m in summary['tissue_wise'].items()]
        tissue_bpq.sort(key=lambda x: x[1], reverse=True)
        for tissue, bpq in tissue_bpq[:5]:
            n = summary['tissue_wise'][tissue].get('n_images', 0)
            print(f"  {tissue:15s}: {bpq:.4f} (n={n})")
        
        print("\n" + "=" * 70)


# ==========================================================================
# Testing
# ==========================================================================

def test_evaluator():
    """Test evaluator with dummy data."""
    print("Testing PanNukeEvaluator...")
    print("=" * 60)
    
    evaluator = PanNukeEvaluator()
    
    # Create dummy data
    H, W = 256, 256
    
    # Ground truth
    true_inst = np.zeros((H, W), dtype=np.int32)
    true_type = np.zeros((H, W), dtype=np.int32)
    
    # Add some instances
    true_inst[50:70, 50:70] = 1
    true_type[50:70, 50:70] = 1  # Neoplastic
    
    true_inst[100:130, 100:130] = 2
    true_type[100:130, 100:130] = 2  # Inflammatory
    
    true_inst[80:100, 180:200] = 3
    true_type[80:100, 180:200] = 5  # Epithelial
    
    # Predictions (with some errors)
    pred_np = np.zeros((2, H, W), dtype=np.float32)
    pred_np[1, 48:72, 48:72] = 3.0  # Slightly shifted
    pred_np[1, 105:125, 105:125] = 3.0  # Slightly smaller
    pred_np[1, 78:102, 178:202] = 3.0  # Slightly larger
    pred_np[0] = -pred_np[1]
    
    pred_hv = np.zeros((2, H, W), dtype=np.float32)
    for (y1, y2, x1, x2) in [(48, 72, 48, 72), (105, 125, 105, 125), (78, 102, 178, 202)]:
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        yy, xx = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')
        pred_hv[0, y1:y2, x1:x2] = (xx - cx) / 20
        pred_hv[1, y1:y2, x1:x2] = (yy - cy) / 20
    
    pred_type = np.random.randn(6, H, W).astype(np.float32) * 0.1
    pred_type[1, 48:72, 48:72] = 3.0  # Neoplastic
    pred_type[2, 105:125, 105:125] = 3.0  # Inflammatory
    pred_type[5, 78:102, 178:202] = 3.0  # Epithelial
    
    # Evaluate
    result = evaluator.evaluate_single(
        pred_np=pred_np,
        pred_hv=pred_hv,
        pred_type=pred_type,
        true_inst=true_inst,
        true_type=true_type,
        tissue="Breast",
        image_id="test_001",
    )
    
    print("\nSingle image results:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Add more samples
    for i in range(5):
        evaluator.evaluate_single(
            pred_np=pred_np,
            pred_hv=pred_hv,
            pred_type=pred_type,
            true_inst=true_inst,
            true_type=true_type,
            tissue=["Breast", "Colon", "Lung", "Liver", "Kidney"][i % 5],
            image_id=f"test_{i:03d}",
        )
    
    # Print summary
    evaluator.print_summary()
    
    print("\n" + "=" * 60)
    print("Evaluator tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_evaluator()
