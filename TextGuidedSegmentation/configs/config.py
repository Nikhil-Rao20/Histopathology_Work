"""
Configuration System for TextGuidedSegmentation
================================================

This module provides a flexible configuration system for training
and evaluating text-guided segmentation models.

Usage:
------
>>> from TextGuidedSegmentation.configs import get_config
>>> 
>>> # Get default config
>>> cfg = get_config()
>>> 
>>> # Get config for specific model
>>> cfg = get_config("clipseg")
>>> 
>>> # Override specific values
>>> cfg = get_config("clipseg", batch_size=16, learning_rate=1e-4)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Dataset paths
    pannuke_root: str = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Dataset/multi_images"
    pannuke_mask_root: str = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Dataset/multi_masks"
    consep_root: str = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Dataset/CoNSeP"
    monusac_root: str = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Dataset/MoNuSAC"
    
    # CSV files
    pannuke_csv: str = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Dataset/Images_With_Unique_Labels_Refer_Segmentation_Task.csv"
    
    # Image settings
    image_size: int = 256
    num_classes: int = 5
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    random_flip: bool = True
    random_rotation: bool = True
    color_jitter: bool = True
    
    # Class names and prompts
    class_names: List[str] = field(default_factory=lambda: [
        "neoplastic",
        "inflammatory", 
        "connective",
        "dead",
        "epithelial",
    ])
    
    text_prompts: List[str] = field(default_factory=lambda: [
        "neoplastic cells",
        "inflammatory cells",
        "connective tissue cells",
        "dead cells",
        "epithelial cells",
    ])


@dataclass  
class ModelConfig:
    """Model configuration."""
    
    # Model selection
    model_name: str = "clipseg"
    
    # CLIP backbone
    clip_model: str = "ViT-B/16"
    freeze_clip: bool = True
    
    # Architecture
    hidden_dim: int = 256
    num_decoder_layers: int = 3
    
    # Dropout
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training settings
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    
    # Learning rate scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss function
    loss_type: str = "combined"  # "ce", "dice", "focal", "combined"
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    focal_gamma: float = 2.0
    
    # Gradient clipping
    clip_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every: int = 5
    eval_every: int = 1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "iou",
        "dice", 
        "f1",
        "pq",
    ])
    
    # Evaluation settings
    eval_on_train: bool = False
    visualize_predictions: bool = True
    num_visualizations: int = 10


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    
    # Experiment name
    experiment_name: str = "text_guided_seg"
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    # Cross-validation
    num_folds: int = 3
    current_fold: int = 0
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = "cuda"
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "text_guided_histopath"


@dataclass
class Config:
    """Complete configuration."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Ensure directories exist."""
        for dir_path in [
            self.experiment.checkpoint_dir,
            self.experiment.log_dir,
            self.experiment.results_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
        )


# Model-specific default configurations
MODEL_DEFAULTS = {
    'clipseg': {
        'hidden_dim': 256,
        'num_decoder_layers': 3,
        'learning_rate': 1e-4,
    },
    'lseg': {
        'hidden_dim': 256,
        'num_decoder_layers': 4,
        'learning_rate': 1e-4,
    },
    'groupvit': {
        'hidden_dim': 384,
        'learning_rate': 5e-5,
        'warmup_epochs': 10,
    },
    'san': {
        'hidden_dim': 256,
        'learning_rate': 2e-4,
    },
    'fc_clip': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
    'ovseg': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
    'cat_seg': {
        'hidden_dim': 256,
        'learning_rate': 2e-4,
    },
    'sed': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
    'maft_plus': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
    'x_decoder': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
    'openseed': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
    'odise': {
        'hidden_dim': 256,
        'learning_rate': 5e-5,
    },
    'tagalign': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
    'semantic_sam': {
        'hidden_dim': 256,
        'learning_rate': 1e-4,
    },
}


def get_config(
    model_name: Optional[str] = None,
    **overrides
) -> Config:
    """
    Get configuration with model-specific defaults.
    
    Args:
        model_name: Name of the model (uses model-specific defaults if provided)
        **overrides: Override any config value
        
    Returns:
        Config object
    """
    cfg = Config()
    
    # Apply model-specific defaults
    if model_name is not None:
        cfg.model.model_name = model_name
        
        if model_name.lower() in MODEL_DEFAULTS:
            defaults = MODEL_DEFAULTS[model_name.lower()]
            
            if 'hidden_dim' in defaults:
                cfg.model.hidden_dim = defaults['hidden_dim']
            if 'num_decoder_layers' in defaults:
                cfg.model.num_decoder_layers = defaults['num_decoder_layers']
            if 'learning_rate' in defaults:
                cfg.training.learning_rate = defaults['learning_rate']
            if 'warmup_epochs' in defaults:
                cfg.training.warmup_epochs = defaults['warmup_epochs']
    
    # Apply overrides
    for key, value in overrides.items():
        # Check in each config section
        for section_name in ['data', 'model', 'training', 'evaluation', 'experiment']:
            section = getattr(cfg, section_name)
            if hasattr(section, key):
                setattr(section, key, value)
                break
    
    return cfg


def get_all_models() -> List[str]:
    """Get list of all supported model names."""
    return list(MODEL_DEFAULTS.keys())
