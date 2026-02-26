"""
Testing/Evaluation Module for CIPS-Net V2
==========================================

This module provides comprehensive evaluation of trained models on the test fold.

Components:
-----------
1. post_processing.py - HoVer-Net style watershed to get instance maps
2. evaluator.py - Main evaluation class with all official metrics
3. evaluate.py - CLI script to run evaluation

Metrics Computed:
-----------------
Instance Segmentation:
    - Dice (binary)
    - AJI (Aggregated Jaccard Index)
    - AJI+ (with unique pairing)
    - DQ (Detection Quality)
    - SQ (Segmentation Quality)
    - bPQ (binary Panoptic Quality = DQ * SQ)

Multi-class (per nucleus type):
    - mPQ (multi-class PQ)
    - Per-class PQ

Detection:
    - Per-class Precision, Recall, F1
    - Overall Detection F1, Accuracy

Tissue-wise:
    - All above metrics computed per tissue type
"""

from .post_processing import PostProcessor, process_batch, PostProcessConfig
from .evaluator import PanNukeEvaluator, EvaluationConfig

__all__ = [
    'PostProcessor',
    'PostProcessConfig',
    'process_batch',
    'PanNukeEvaluator',
    'EvaluationConfig',
]
