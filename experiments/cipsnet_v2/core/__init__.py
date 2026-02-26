"""
Core Experiment Infrastructure for CIPS-Net V2
==============================================

This module provides the infrastructure for running reproducible experiments
with 3-fold cross-validation, comprehensive result tracking, and paper-ready
output generation.

Components:
-----------
- ExperimentConfig: Comprehensive experiment configuration
- ExperimentRunner: Orchestrates training + testing across folds
- ResultAggregator: Aggregates results across folds with mean±std

Usage:
------
    from experiments.cipsnet_v2.core import ExperimentRunner, ExperimentConfig
    
    config = ExperimentConfig(
        variant='BASELINE',
        epochs=50,
        early_stopping_patience=8,
    )
    
    runner = ExperimentRunner(config)
    results = runner.run()  # Runs all 3 folds, tests, and aggregates
"""

from .experiment_config import ExperimentConfig
from .experiment_runner import ExperimentRunner
from .result_aggregator import ResultAggregator

__all__ = [
    'ExperimentConfig',
    'ExperimentRunner', 
    'ResultAggregator',
]
