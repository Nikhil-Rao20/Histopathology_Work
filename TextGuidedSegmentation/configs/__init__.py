"""
TextGuidedSegmentation Configuration Module
============================================

Provides flexible configuration system for training and evaluation.
"""

from .config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig,
    get_config,
    get_all_models,
    MODEL_DEFAULTS,
)

__all__ = [
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'ExperimentConfig',
    'get_config',
    'get_all_models',
    'MODEL_DEFAULTS',
]
