"""
Result Aggregator for CIPS-Net V2
==================================

Aggregates results from multiple folds into paper-ready format:
- Mean ± std across folds
- Per-class and per-tissue breakdowns
- LaTeX table generation
- Summary JSON for quick reference

Output Files:
-------------
- summary.json: Overall metrics with mean±std
- class_wise.csv: Per-class PQ with fold values
- tissue_wise.csv: Per-tissue metrics with fold values  
- detection.csv: Detection metrics with fold values
- latex_tables.tex: Ready-to-use LaTeX tables
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# PanNuke class names
PANNUKE_CLASSES = {
    0: 'background',
    1: 'neoplastic',
    2: 'inflammatory', 
    3: 'connective',
    4: 'dead',
    5: 'epithelial',
}


@dataclass
class FoldResults:
    """Container for results from a single fold."""
    fold: int
    metrics_json_path: str
    metrics: Dict[str, Any] = None
    
    def load(self):
        """Load metrics from JSON file."""
        if os.path.exists(self.metrics_json_path):
            with open(self.metrics_json_path, 'r') as f:
                self.metrics = json.load(f)
        else:
            raise FileNotFoundError(f"Results not found: {self.metrics_json_path}")
        return self


class ResultAggregator:
    """
    Aggregates results from multiple folds.
    
    Features:
    - Computes mean ± std across folds
    - Generates paper-ready CSV tables
    - Creates LaTeX tables
    - Saves comprehensive summary
    
    Usage:
        aggregator = ResultAggregator(experiment_dir="/path/to/experiment")
        aggregator.add_fold_results(1, metrics_dict)
        aggregator.add_fold_results(2, metrics_dict)
        aggregator.add_fold_results(3, metrics_dict)
        aggregator.aggregate_and_save()
    """
    
    def __init__(self, experiment_dir: str, variant: str = "UNKNOWN"):
        """
        Initialize aggregator.
        
        Args:
            experiment_dir: Root directory of the experiment
            variant: Model variant name
        """
        self.experiment_dir = experiment_dir
        self.variant = variant
        self.aggregate_dir = os.path.join(experiment_dir, "aggregate")
        
        # Storage for fold results
        self.fold_results: Dict[int, Dict[str, Any]] = {}
        
        # Create aggregate directory
        os.makedirs(self.aggregate_dir, exist_ok=True)
    
    def add_fold_results(self, fold: int, metrics: Dict[str, Any]):
        """
        Add results from a fold.
        
        Args:
            fold: Fold number (1, 2, or 3)
            metrics: Dictionary of metrics from evaluator
        """
        self.fold_results[fold] = metrics
    
    def load_fold_results(self, fold: int, results_dir: str):
        """
        Load fold results from saved JSON.
        
        Args:
            fold: Fold number
            results_dir: Directory containing results.json
        """
        results_path = os.path.join(results_dir, "results.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                self.fold_results[fold] = json.load(f)
        else:
            raise FileNotFoundError(f"Results not found: {results_path}")
    
    def aggregate_and_save(self) -> Dict[str, Any]:
        """
        Aggregate all fold results and save to aggregate directory.
        
        Returns:
            Aggregated results dictionary
        """
        if len(self.fold_results) == 0:
            raise ValueError("No fold results to aggregate")
        
        print(f"\n[Aggregating Results] {len(self.fold_results)} folds")
        
        # Aggregate different metric types
        overall_agg = self._aggregate_overall_metrics()
        class_wise_agg = self._aggregate_class_wise_metrics()
        tissue_wise_agg = self._aggregate_tissue_wise_metrics()
        detection_agg = self._aggregate_detection_metrics()
        
        # Create summary
        summary = {
            "variant": self.variant,
            "num_folds": len(self.fold_results),
            "folds": list(self.fold_results.keys()),
            "overall": overall_agg,
            "class_wise": class_wise_agg,
            "detection": detection_agg,
        }
        
        # Save all outputs
        self._save_summary_json(summary)
        self._save_class_wise_csv(class_wise_agg)
        self._save_tissue_wise_csv(tissue_wise_agg)
        self._save_detection_csv(detection_agg)
        self._save_latex_tables(summary)
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _aggregate_overall_metrics(self) -> Dict[str, Dict[str, float]]:
        """Aggregate overall metrics (Dice, AJI, bPQ, mPQ, etc.)."""
        metrics_to_aggregate = ['dice', 'aji', 'aji_plus', 'dq', 'sq', 'bpq', 'mpq']
        
        aggregated = {}
        
        for metric_name in metrics_to_aggregate:
            values = []
            for fold, results in self.fold_results.items():
                # Handle different possible key formats
                value = None
                if 'overall' in results:
                    overall = results['overall']
                    # Try different key formats
                    for key in [metric_name, metric_name.upper(), metric_name.lower()]:
                        if key in overall:
                            value = overall[key]
                            break
                    # Also check mean key
                    for key in [f'{metric_name}_mean', f'{metric_name.upper()}_mean']:
                        if key in overall:
                            value = overall[key]
                            break
                
                if value is not None and not np.isnan(value):
                    values.append(value)
            
            if values:
                aggregated[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "fold_values": {f"fold_{i}": v for i, v in zip(self.fold_results.keys(), values)},
                }
            else:
                aggregated[metric_name] = {
                    "mean": float('nan'),
                    "std": float('nan'),
                    "fold_values": {},
                }
        
        return aggregated
    
    def _aggregate_class_wise_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-class PQ metrics."""
        classes = list(PANNUKE_CLASSES.values())[1:]  # Skip background
        aggregated = {}
        
        for class_name in classes:
            values = []
            for fold, results in self.fold_results.items():
                value = None
                if 'class_wise' in results:
                    class_data = results['class_wise']
                    if class_name in class_data:
                        if isinstance(class_data[class_name], dict):
                            value = class_data[class_name].get('pq_mean', class_data[class_name].get('pq'))
                        else:
                            value = class_data[class_name]
                
                if value is not None and not np.isnan(value):
                    values.append(value)
            
            if values:
                aggregated[class_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "fold_values": list(values),
                }
            else:
                aggregated[class_name] = {
                    "mean": float('nan'),
                    "std": float('nan'),
                    "fold_values": [],
                }
        
        return aggregated
    
    def _aggregate_tissue_wise_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-tissue metrics."""
        # First, collect all tissue names from all folds
        all_tissues = set()
        for fold, results in self.fold_results.items():
            if 'tissue_wise' in results:
                all_tissues.update(results['tissue_wise'].keys())
        
        metrics_to_aggregate = ['dice', 'aji', 'bpq', 'mpq']
        aggregated = {}
        
        for tissue in all_tissues:
            tissue_data = {}
            
            for metric_name in metrics_to_aggregate:
                values = []
                for fold, results in self.fold_results.items():
                    if 'tissue_wise' in results and tissue in results['tissue_wise']:
                        tissue_results = results['tissue_wise'][tissue]
                        # Try different key formats
                        for key in [metric_name, metric_name.upper(), metric_name.lower()]:
                            if key in tissue_results:
                                val = tissue_results[key]
                                if val is not None and not np.isnan(val):
                                    values.append(val)
                                break
                
                if values:
                    tissue_data[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                else:
                    tissue_data[metric_name] = {
                        "mean": float('nan'),
                        "std": float('nan'),
                    }
            
            aggregated[tissue] = tissue_data
        
        return aggregated
    
    def _aggregate_detection_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate detection metrics (Precision, Recall, F1)."""
        metrics_to_aggregate = ['precision', 'recall', 'f1']
        classes = ['overall'] + list(PANNUKE_CLASSES.values())[1:]  # Overall + classes
        
        aggregated = {}
        
        for class_name in classes:
            class_data = {}
            
            for metric_name in metrics_to_aggregate:
                values = []
                for fold, results in self.fold_results.items():
                    if 'detection' in results:
                        det = results['detection']
                        key = f"{class_name}_{metric_name}" if class_name != 'overall' else f"overall_{metric_name}"
                        alt_key = metric_name if class_name == 'overall' else f"{class_name}_{metric_name}"
                        
                        for k in [key, alt_key, metric_name]:
                            if k in det:
                                val = det[k]
                                if val is not None and not np.isnan(val):
                                    values.append(val)
                                break
                
                if values:
                    class_data[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                else:
                    class_data[metric_name] = {
                        "mean": float('nan'),
                        "std": float('nan'),
                    }
            
            aggregated[class_name] = class_data
        
        return aggregated
    
    def _save_summary_json(self, summary: Dict[str, Any]):
        """Save comprehensive summary as JSON."""
        path = os.path.join(self.aggregate_dir, "summary.json")
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        
        print(f"  Saved: {path}")
    
    def _save_class_wise_csv(self, class_wise: Dict[str, Dict]):
        """Save class-wise metrics as CSV."""
        rows = []
        for class_name, data in class_wise.items():
            row = {
                "Class": class_name,
                "PQ_mean": data.get("mean", float('nan')),
                "PQ_std": data.get("std", float('nan')),
            }
            # Add individual fold values
            for i, val in enumerate(data.get("fold_values", []), 1):
                row[f"Fold_{i}"] = val
            rows.append(row)
        
        df = pd.DataFrame(rows)
        path = os.path.join(self.aggregate_dir, "class_wise.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")
    
    def _save_tissue_wise_csv(self, tissue_wise: Dict[str, Dict]):
        """Save tissue-wise metrics as CSV."""
        rows = []
        for tissue, metrics in tissue_wise.items():
            row = {"Tissue": tissue}
            for metric_name, data in metrics.items():
                row[f"{metric_name.upper()}_mean"] = data.get("mean", float('nan'))
                row[f"{metric_name.upper()}_std"] = data.get("std", float('nan'))
            rows.append(row)
        
        df = pd.DataFrame(rows)
        # Sort by bPQ mean if available
        if 'BPQ_mean' in df.columns:
            df = df.sort_values('BPQ_mean', ascending=False)
        
        path = os.path.join(self.aggregate_dir, "tissue_wise.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")
    
    def _save_detection_csv(self, detection: Dict[str, Dict]):
        """Save detection metrics as CSV."""
        rows = []
        for class_name, metrics in detection.items():
            row = {"Class": class_name}
            for metric_name, data in metrics.items():
                row[f"{metric_name.capitalize()}_mean"] = data.get("mean", float('nan'))
                row[f"{metric_name.capitalize()}_std"] = data.get("std", float('nan'))
            rows.append(row)
        
        df = pd.DataFrame(rows)
        path = os.path.join(self.aggregate_dir, "detection.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")
    
    def _save_latex_tables(self, summary: Dict[str, Any]):
        """Generate LaTeX tables for paper."""
        latex = []
        
        # Header
        latex.append(f"% CIPS-Net V2 Results - {self.variant}")
        latex.append(f"% Generated automatically - 3-fold cross-validation")
        latex.append("")
        
        # Overall metrics table
        latex.append("% Overall Metrics")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Instance Segmentation Results - {self.variant}}}")
        latex.append("\\begin{tabular}{lcc}")
        latex.append("\\toprule")
        latex.append("Metric & Mean & Std \\\\")
        latex.append("\\midrule")
        
        overall = summary.get('overall', {})
        for metric in ['bpq', 'mpq', 'aji', 'dice']:
            if metric in overall:
                mean = overall[metric].get('mean', float('nan'))
                std = overall[metric].get('std', float('nan'))
                if not np.isnan(mean):
                    latex.append(f"{metric.upper()} & {mean*100:.2f} & {std*100:.2f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")
        
        # Class-wise PQ table
        latex.append("% Per-Class PQ")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Per-Class Panoptic Quality - {self.variant}}}")
        latex.append("\\begin{tabular}{lcc}")
        latex.append("\\toprule")
        latex.append("Class & PQ (\\%) & Std \\\\")
        latex.append("\\midrule")
        
        class_wise = summary.get('class_wise', {})
        for class_name in ['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']:
            if class_name in class_wise:
                mean = class_wise[class_name].get('mean', float('nan'))
                std = class_wise[class_name].get('std', float('nan'))
                if not np.isnan(mean):
                    latex.append(f"{class_name.capitalize()} & {mean*100:.2f} & {std*100:.2f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        # Save
        path = os.path.join(self.aggregate_dir, "latex_tables.tex")
        with open(path, 'w') as f:
            f.write('\n'.join(latex))
        print(f"  Saved: {path}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "=" * 70)
        print(f"AGGREGATED RESULTS: {self.variant}")
        print(f"Folds: {summary['folds']}")
        print("=" * 70)
        
        print("\n[Overall Metrics] (Mean ± Std)")
        overall = summary.get('overall', {})
        for metric in ['bpq', 'mpq', 'aji', 'aji_plus', 'dice', 'dq', 'sq']:
            if metric in overall:
                mean = overall[metric].get('mean', float('nan'))
                std = overall[metric].get('std', float('nan'))
                if not np.isnan(mean):
                    print(f"  {metric.upper():10s}: {mean*100:6.2f} ± {std*100:.2f}")
        
        print("\n[Per-Class PQ]")
        class_wise = summary.get('class_wise', {})
        for class_name in ['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']:
            if class_name in class_wise:
                mean = class_wise[class_name].get('mean', float('nan'))
                std = class_wise[class_name].get('std', float('nan'))
                if not np.isnan(mean):
                    print(f"  {class_name:15s}: {mean*100:6.2f} ± {std*100:.2f}")
        
        print("\n[Detection Metrics]")
        detection = summary.get('detection', {})
        if 'overall' in detection:
            for metric in ['precision', 'recall', 'f1']:
                if metric in detection['overall']:
                    mean = detection['overall'][metric].get('mean', float('nan'))
                    std = detection['overall'][metric].get('std', float('nan'))
                    if not np.isnan(mean):
                        print(f"  {metric.capitalize():10s}: {mean*100:6.2f} ± {std*100:.2f}")
        
        print("\n" + "=" * 70)
        print(f"Results saved to: {self.aggregate_dir}")
        print("=" * 70 + "\n")


# ==========================================================================
# Testing
# ==========================================================================

if __name__ == "__main__":
    print("Testing ResultAggregator...")
    
    # Create mock results
    mock_results_fold1 = {
        "overall": {
            "dice": 0.75,
            "aji": 0.45,
            "bpq": 0.48,
            "mpq": 0.35,
        },
        "class_wise": {
            "neoplastic": {"pq_mean": 0.42},
            "inflammatory": {"pq_mean": 0.38},
            "connective": {"pq_mean": 0.35},
            "dead": {"pq_mean": 0.25},
            "epithelial": {"pq_mean": 0.40},
        },
        "detection": {
            "overall_precision": 0.65,
            "overall_recall": 0.58,
            "overall_f1": 0.61,
        },
    }
    
    mock_results_fold2 = {
        "overall": {
            "dice": 0.77,
            "aji": 0.47,
            "bpq": 0.50,
            "mpq": 0.37,
        },
        "class_wise": {
            "neoplastic": {"pq_mean": 0.44},
            "inflammatory": {"pq_mean": 0.40},
            "connective": {"pq_mean": 0.37},
            "dead": {"pq_mean": 0.27},
            "epithelial": {"pq_mean": 0.42},
        },
        "detection": {
            "overall_precision": 0.67,
            "overall_recall": 0.60,
            "overall_f1": 0.63,
        },
    }
    
    # Test aggregation
    aggregator = ResultAggregator("/tmp/test_experiment", "TEST")
    aggregator.add_fold_results(1, mock_results_fold1)
    aggregator.add_fold_results(2, mock_results_fold2)
    
    summary = aggregator.aggregate_and_save()
    
    print("\n✓ ResultAggregator tests passed!")
