"""
Experiment Configuration for CIPS-Net V2
=========================================

Comprehensive configuration for running experiments with:
- All hyperparameters
- 3-fold cross-validation setup
- Early stopping configuration
- Result organization paths

The configuration is designed to be:
1. Reproducible (full config saved with each experiment)
2. Self-documenting (clear parameter organization)
3. Extensible (easy to add new parameters)
"""

import os
import yaml
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


# ==========================================================================
# Valid Values
# ==========================================================================

VALID_VARIANTS = [
    'BASELINE',             # ViT encoder only (no text)
    'WITH_TEXT',            # + Text encoder
    'WITH_CGR',             # + Cross-modal Gated Refinement
    'WITH_TEXT_CONDITIONED_TYPE',  # + Text-conditioned type prediction
    'FULL',                 # All components
    'LAVT',                 # Language-Aware ViT (early fusion at every layer)
    'CRIS',                 # CLIP-Driven Referring Image Segmentation
    'LVIT',                 # Language-guided Vision Transformer (U-Net style)
    'LVIT2',                # LViT2: Enhanced with Deep Supervision + Aux Classification
    'LVIT3',                # LViT3: Instance Normalization + Contrastive Loss Support
    'LVIT4',                # LViT4: Multi-stage Fusion + PWAM (Phase 2)
    'LVIT5',                # LViT5: Ultimate - Cross-Modal Decoder + Grounding + All improvements
    'LVIT_IE',              # LViT-IE: Instance Embedding decoder (novel — Phase 3)
    'GROUNDING_DINO',       # Detection-based with text-guided queries
]

VALID_OPTIMIZERS = ['adam', 'adamw', 'sgd']
VALID_SCHEDULERS = ['cosine', 'onecycle', 'step', 'none']
VALID_IMAGE_ENCODERS = ['vit_b_16', 'vit_b_32', 'resnet50', 'resnet101']
VALID_TEXT_ENCODERS = ['distilbert-base-uncased', 'bert-base-uncased']

# Valid backbone options for LViT-style models
VALID_BACKBONES = [
    'vit',              # torchvision ViT-B/16 (ImageNet supervised)
    'convnext_base',    # ConvNeXt-Base (ImageNet supervised)
    'dinov2_vit_b_14',  # DINOv2 ViT-B/14 (self-supervised, patch14)
    'dinov2_vit_l_14',  # DINOv2 ViT-L/14 (self-supervised, patch14)
    'dinov2_vit_s_14',  # DINOv2 ViT-S/14 (self-supervised, patch14)
    'dinov2_vit_g_14',  # DINOv2 ViT-g/14 (self-supervised, giant)
    'swin_b',           # Swin Transformer V2 Base (ImageNet pretrained)
    'swin_l',           # Swin Transformer V2 Large (ImageNet-22k pretrained)
]


# ==========================================================================
# Experiment Configuration
# ==========================================================================

@dataclass
class ExperimentConfig:
    """
    Comprehensive experiment configuration.
    
    This configuration covers all aspects of an experiment:
    - Model architecture (variant, encoders)
    - Training hyperparameters
    - Data loading
    - Loss function settings
    - Early stopping
    - Output organization
    
    The experiment runs 3-fold cross-validation automatically.
    """
    
    # ==========================================================================
    # Experiment Identification
    # ==========================================================================
    variant: str = "BASELINE"
    experiment_name: Optional[str] = None  # Auto-generated if None
    description: str = ""  # Optional description for logs
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    num_classes: int = 6  # 5 nucleus types + background
    image_size: int = 256
    
    # Encoders
    image_encoder: str = "vit_b_16"
    text_encoder: str = "distilbert-base-uncased"
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = True
    
    # Backbone for text-guided models (LVIT, LAVT, etc.)
    backbone: str = "vit"  # 'vit', 'convnext_base', 'dinov2_vit_b_14', 'dinov2_vit_l_14'
    
    # DINOv2-specific settings
    freeze_dinov2_backbone: bool = False  # Freeze DINOv2 weights (use pretrained features only)
    dinov2_pretrained_path: str = ""  # Path to supervised pretrained DINOv2 checkpoint (optional)
    use_gradient_checkpointing: bool = False  # Gradient checkpointing for DINOv2 (saves ~50% activation memory)
    
    # ==========================================================================
    # Dataset Configuration
    # ==========================================================================
    pannuke_path: str = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Histopathology_Datasets_Official/PanNuke"
    num_workers: int = 0  # 0 = main process only (most memory efficient)
    pin_memory: bool = False  # Disable to save GPU memory
    
    # Dataloader Configuration
    use_permutation_dataloader: bool = True  # Use the new dataloader (with CSV support)
    use_permutations_csv: bool = False  # If True: permutations CSV (partial masks), If False: unique labels CSV (full masks)
    
    # ==========================================================================
    # Memory Optimization
    # ==========================================================================
    low_memory_mode: bool = True  # Enable memory optimizations
    gc_every_n_batches: int = 50  # Run garbage collection every N batches
    
    # ==========================================================================
    # Training Configuration
    # ==========================================================================
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Gradient
    gradient_clip: float = 1.0
    
    # Mixed Precision
    use_amp: bool = True
    
    # ==========================================================================
    # Early Stopping
    # ==========================================================================
    early_stopping: bool = True
    early_stopping_patience: int = 8  # Stop if no improvement for N epochs
    early_stopping_min_delta: float = 0.0001  # Minimum improvement to count
    
    # ==========================================================================
    # Loss Configuration
    # ==========================================================================
    np_weight: float = 1.0
    hv_weight: float = 2.0
    type_weight: float = 2.0  # Increased for better type classification
    
    # Loss Function Type (for ablation studies)
    # Options: 'ce' (Cross-Entropy), 'weighted_ce', 'focal', 'weighted_focal', 'ldam'
    loss_type: str = 'weighted_focal'  # Default: Weighted Focal CE
    
    # Focal Loss Parameters
    focal_gamma: float = 2.0  # Focusing parameter (higher = more focus on hard examples)
    
    # Class Weights for Type Loss (inverse sqrt frequency, normalized)
    # Background=0, Neoplastic=1.0, Inflammatory=1.29, Connective=1.41, Dead=3.16, Epithelial=1.05
    use_class_weights: bool = True
    
    # LDAM Parameters (only used when loss_type='ldam')
    use_ldam: bool = False  # Deprecated: use loss_type='ldam' instead
    ldam_max_m: float = 0.5
    ldam_s: float = 30.0
    
    # DRW (Deferred Re-Weighting) Schedule
    use_drw: bool = True  # Enable DRW schedule for class weights
    drw_start_epoch: int = 15  # Start DRW at 30% of 50 epochs (earlier rebalancing)
    drw_start_ratio: float = 0.3  # Alternative: specify as ratio of total epochs
    
    # ==========================================================================
    # Contrastive Loss Configuration (for LViT3/LViT4)
    # ==========================================================================
    enable_contrastive: bool = True  # Enable contrastive loss features in model
    contrastive_weight: float = 0.5  # Weight for contrastive loss term
    contrastive_temperature: float = 0.07  # Temperature for similarity scaling
    
    # LViT4 Multi-scale contrastive weights (deep, mid, out)
    contrastive_scale_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # ==========================================================================
    # Evaluation Configuration (tuned — see post_processing_tuning.ipynb)
    # ==========================================================================
    np_threshold: float = 0.525
    min_instance_size: int = 70
    marker_erosion_size: int = 3
    
    # ==========================================================================
    # Data Augmentation — Dataset Expansion (Phase 2)
    # ==========================================================================
    # When enabled, training dataset is expanded by creating augmented copies
    # of each image. Each type adds N new samples (N = original dataset size).
    # Total = N × (1 + num_enabled_types).  Default 3 types ON = 4× expansion.
    # Only affects TRAINING data.  Val/test are never augmented.
    use_data_augmentation: bool = False          # Master toggle
    augmentation_stain_light: bool = True        # +N (mild stain variation)
    augmentation_stain_strong: bool = True       # +N (aggressive stain variation)
    augmentation_elastic: bool = True            # +N (spatial deformation)
    augmentation_gaussian_noise: bool = False    # +N (scanner noise) — OFF by default
    
    # ==========================================================================
    # Output Configuration
    # ==========================================================================
    base_output_dir: str = "experiments/cipsnet_v2/experiments"
    save_predictions: bool = False  # Save prediction maps (uses disk space)
    
    # Logging
    log_every_n_batches: int = 50
    
    # ==========================================================================
    # Debug Configuration  
    # ==========================================================================
    debug: bool = False  # Enable debug mode
    debug_batches: int = 0  # Limit batches per epoch (0 = all)
    seed: int = 42
    
    # ==========================================================================
    # Fold Selection
    # ==========================================================================
    folds: List[int] = field(default_factory=lambda: [1, 2, 3])  # Which folds to run
    
    # ==========================================================================
    # Internal (Auto-computed)
    # ==========================================================================
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def __post_init__(self):
        """Validate and process configuration."""
        # Validate variant
        if self.variant not in VALID_VARIANTS:
            raise ValueError(
                f"Invalid variant: {self.variant}. "
                f"Must be one of {VALID_VARIANTS}"
            )
        
        # Validate backbone
        if self.backbone not in VALID_BACKBONES:
            raise ValueError(
                f"Invalid backbone: {self.backbone}. "
                f"Must be one of {VALID_BACKBONES}"
            )
        
        # Validate optimizer
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}. "
                f"Must be one of {VALID_OPTIMIZERS}"
            )
        
        # Validate scheduler
        if self.scheduler not in VALID_SCHEDULERS:
            raise ValueError(
                f"Invalid scheduler: {self.scheduler}. "
                f"Must be one of {VALID_SCHEDULERS}"
            )
        
        # Auto-generate experiment name
        if self.experiment_name is None:
            self.experiment_name = f"{self.variant}_{self.timestamp}"
        
        # Ensure output directory exists
        os.makedirs(self.base_output_dir, exist_ok=True)
    
    # ==========================================================================
    # Directory Paths
    # ==========================================================================
    
    @property
    def experiment_dir(self) -> str:
        """Root directory for this experiment."""
        return os.path.join(self.base_output_dir, self.experiment_name)
    
    def get_fold_dir(self, fold: int) -> str:
        """Get directory for a specific fold."""
        return os.path.join(self.experiment_dir, f"fold_{fold}")
    
    def get_checkpoint_dir(self, fold: int) -> str:
        """Get checkpoint directory for a fold."""
        return os.path.join(self.get_fold_dir(fold), "checkpoints")
    
    def get_checkpoint_path(self, fold: int, checkpoint_type: str = "best") -> str:
        """Get checkpoint path for a fold."""
        return os.path.join(self.get_checkpoint_dir(fold), f"{checkpoint_type}.pth")
    
    def get_training_log_path(self, fold: int) -> str:
        """Get training log CSV path for a fold."""
        return os.path.join(self.get_fold_dir(fold), "training_log.csv")
    
    def get_results_dir(self, fold: int) -> str:
        """Get results directory for a fold."""
        return os.path.join(self.get_fold_dir(fold), "results")
    
    def get_aggregate_dir(self) -> str:
        """Get aggregate results directory."""
        return os.path.join(self.experiment_dir, "aggregate")
    
    @property
    def config_path(self) -> str:
        """Path to save config YAML."""
        return os.path.join(self.experiment_dir, "config.yaml")
    
    @property
    def experiment_log_path(self) -> str:
        """Path to experiment log."""
        return os.path.join(self.experiment_dir, "experiment.log")
    
    # ==========================================================================
    # Dataloader Mode
    # ==========================================================================
    
    @property
    def dataloader_mode(self) -> str:
        """Get dataloader mode based on variant."""
        if self.variant == 'BASELINE':
            return 'baseline'
        else:
            return 'single_text'
    
    # ==========================================================================
    # Serialization
    # ==========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        # Convert tuples to lists for YAML compatibility
        d['betas'] = list(d['betas'])
        return d
    
    def save(self, path: Optional[str] = None):
        """Save configuration to YAML file."""
        path = path or self.config_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert lists back to tuples
        if 'betas' in config_dict:
            config_dict['betas'] = tuple(config_dict['betas'])
        
        return cls(**config_dict)
    
    # ==========================================================================
    # Display
    # ==========================================================================
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("\n" + "=" * 70)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 70)
        
        sections = {
            "Experiment": ["variant", "experiment_name", "description", "timestamp"],
            "Model": ["num_classes", "image_size", "image_encoder", "text_encoder",
                     "freeze_image_encoder", "freeze_text_encoder"],
            "Dataset": ["pannuke_path", "num_workers", "pin_memory"],
            "Training": ["epochs", "batch_size", "learning_rate", "weight_decay",
                        "optimizer", "scheduler", "warmup_epochs", "gradient_clip", "use_amp"],
            "Early Stopping": ["early_stopping", "early_stopping_patience", "early_stopping_min_delta"],
            "Loss": ["np_weight", "hv_weight", "type_weight", 
                    "use_ldam", "drw_start_epoch"],
            "Evaluation": ["np_threshold", "min_instance_size"],
            "Output": ["base_output_dir", "experiment_dir"],
            "Debug": ["debug", "debug_batches", "seed"],
        }
        
        for section_name, keys in sections.items():
            print(f"\n[{section_name}]")
            for key in keys:
                value = getattr(self, key, "N/A")
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 70 + "\n")
    
    def get_summary(self) -> str:
        """Get one-line summary of experiment."""
        return (
            f"{self.variant} | "
            f"epochs={self.epochs} | "
            f"lr={self.learning_rate} | "
            f"batch_size={self.batch_size} | "
            f"early_stop={self.early_stopping_patience}"
        )


# ==========================================================================
# Preset Configurations
# ==========================================================================

def get_debug_config(variant: str = "BASELINE") -> ExperimentConfig:
    """Get configuration for debugging (quick test run)."""
    return ExperimentConfig(
        variant=variant,
        epochs=3,
        batch_size=4,
        debug=True,
        debug_batches=5,  # Only 5 batches per epoch
        early_stopping=False,
    )


def get_quick_test_config(variant: str = "BASELINE") -> ExperimentConfig:
    """Get configuration for quick testing (10 epochs)."""
    return ExperimentConfig(
        variant=variant,
        epochs=10,
        batch_size=8,
        debug=True,
        debug_batches=0,  # All batches
        early_stopping=True,
        early_stopping_patience=5,
    )


def get_production_config(variant: str = "BASELINE") -> ExperimentConfig:
    """Get configuration for production training (50 epochs)."""
    return ExperimentConfig(
        variant=variant,
        epochs=50,
        batch_size=8,
        debug=False,
        debug_batches=0,
        early_stopping=True,
        early_stopping_patience=8,
    )


# ==========================================================================
# Testing
# ==========================================================================

if __name__ == "__main__":
    print("Testing ExperimentConfig...")
    
    # Test basic config
    config = ExperimentConfig(variant="BASELINE")
    config.print_config()
    
    # Test paths
    print(f"Experiment dir: {config.experiment_dir}")
    print(f"Fold 1 dir: {config.get_fold_dir(1)}")
    print(f"Fold 1 checkpoint: {config.get_checkpoint_path(1, 'best')}")
    print(f"Aggregate dir: {config.get_aggregate_dir()}")
    
    # Test serialization
    config.save("/tmp/test_config.yaml")
    loaded = ExperimentConfig.load("/tmp/test_config.yaml")
    print(f"\nSaved and loaded config: {loaded.variant}")
    
    print("\n✓ ExperimentConfig tests passed!")
