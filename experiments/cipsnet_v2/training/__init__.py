"""
Training Module for CIPS-Net V2

This module provides training utilities for CIPS-Net V2:
- Configuration management
- Trainer class with mixed precision
- Experiment management (logging, plotting, checkpoints)
- Training utilities

Usage:
    from experiments.cipsnet_v2.training import (
        TrainingConfig,
        Trainer,
        ExperimentManager,
        train_model,
    )
"""

from .config import TrainingConfig
from .trainer import Trainer, create_trainer
from .experiment_manager import ExperimentManager
from .utils import (
    AverageMeter,
    get_optimizer,
    get_scheduler,
    seed_everything,
    count_parameters,
)

__all__ = [
    'TrainingConfig',
    'Trainer',
    'create_trainer',
    'ExperimentManager',
    'AverageMeter',
    'get_optimizer',
    'get_scheduler',
    'seed_everything',
    'count_parameters',
]
