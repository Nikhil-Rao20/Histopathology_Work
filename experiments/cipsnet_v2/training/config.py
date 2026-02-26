"""
Training Configuration for CIPS-Net V2

Centralized configuration for all training parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for CIPS-Net V2 training."""
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    variant: str = "FULL"  # BASELINE, WITH_TEXT, WITH_CGR, WITH_TEXT_CONDITIONED_TYPE, FULL
    num_classes: int = 6  # 5 nucleus types + background
    image_size: int = 256
    
    # Encoder settings
    image_encoder: str = "vit_b_16"
    text_encoder: str = "distilbert-base-uncased"
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = True  # Usually freeze text encoder
    
    # ==========================================================================
    # Dataset Configuration
    # ==========================================================================
    pannuke_path: str = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Histopathology_Datasets_Official/PanNuke"
    fold: int = 1  # Which fold to use for validation (1, 2, or 3)
    dataloader_mode: str = "single_text"  # baseline, single_text, multi_text
    num_workers: int = 4
    pin_memory: bool = True
    
    # ==========================================================================
    # Training Configuration
    # ==========================================================================
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"  # adam, adamw, sgd
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, onecycle, step, none
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Gradient
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # Mixed Precision
    use_amp: bool = True
    
    # ==========================================================================
    # Loss Configuration
    # ==========================================================================
    np_weight: float = 1.0
    hv_weight: float = 2.0
    type_weight: float = 1.0
    
    # LDAM + DRW
    use_ldam: bool = True
    drw_start_epoch: int = 30  # 60% of 50 epochs
    ldam_max_m: float = 0.5
    ldam_s: float = 30.0
    
    # ==========================================================================
    # Checkpoint & Logging
    # ==========================================================================
    runs_dir: str = "experiments/cipsnet_v2/runs"  # Base directory for all experiments
    experiment_name: Optional[str] = None  # Auto-generated if None
    
    save_every: int = 10  # Save checkpoint every N epochs
    log_every: int = 10  # Log every N batches
    
    # Resume training
    resume_from: Optional[str] = None
    
    # ==========================================================================
    # Debugging
    # ==========================================================================
    debug: bool = True  # Enable debugging prints
    debug_batches: int = 2  # Number of batches to run in debug mode (0 = all)
    seed: int = 42
    
    def __post_init__(self):
        """Validate and process configuration."""
        # Validate variant
        valid_variants = ['BASELINE', 'WITH_TEXT', 'WITH_CGR', 'WITH_TEXT_CONDITIONED_TYPE', 'FULL']
        if self.variant not in valid_variants:
            raise ValueError(f"Invalid variant: {self.variant}. Must be one of {valid_variants}")
        
        # Validate fold
        if self.fold not in [1, 2, 3]:
            raise ValueError(f"Invalid fold: {self.fold}. Must be 1, 2, or 3")
        
        # Set dataloader mode based on variant
        if self.variant == 'BASELINE':
            self.dataloader_mode = 'baseline'
        else:
            self.dataloader_mode = 'single_text'
        
        # Auto-generate experiment name
        if self.experiment_name is None:
            self.experiment_name = f"{self.variant}_fold{self.fold}"
        
        # Create runs directory
        os.makedirs(self.runs_dir, exist_ok=True)
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path."""
        return os.path.join(self.runs_dir, self.experiment_name)
    
    def get_checkpoint_path(self, suffix: str = "best") -> str:
        """Get checkpoint file path."""
        return os.path.join(self.runs_dir, self.experiment_name, "checkpoints", f"{suffix}.pth")
    
    def get_log_path(self) -> str:
        """Get log directory path."""
        return os.path.join(self.runs_dir, self.experiment_name, "logs")
    
    def print_config(self):
        """Print configuration for debugging."""
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        
        sections = {
            "Model": ["variant", "num_classes", "image_size", "image_encoder", "text_encoder", 
                      "freeze_image_encoder", "freeze_text_encoder"],
            "Dataset": ["pannuke_path", "fold", "dataloader_mode", "num_workers", "pin_memory"],
            "Training": ["epochs", "batch_size", "learning_rate", "weight_decay", "optimizer",
                         "scheduler", "warmup_epochs", "gradient_clip", "use_amp"],
            "Loss": ["np_weight", "hv_weight", "type_weight", "use_ldam", "drw_start_epoch"],
            "Output": ["runs_dir", "experiment_name", "save_every"],
            "Debug": ["debug", "debug_batches", "seed"],
        }
        
        for section_name, keys in sections.items():
            print(f"\n[{section_name}]")
            for key in keys:
                value = getattr(self, key, "N/A")
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 70 + "\n")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# ==========================================================================
# Preset Configurations
# ==========================================================================

def get_debug_config(variant: str = "BASELINE", fold: int = 1) -> TrainingConfig:
    """Get configuration for debugging (small batches, few epochs)."""
    return TrainingConfig(
        variant=variant,
        fold=fold,
        epochs=10,  # 10 epochs for testing output saving
        batch_size=8,
        debug=True,
        debug_batches=0,  # 0 = run all batches
        log_every=50,
        save_every=10,
    )


def get_training_config(variant: str, fold: int) -> TrainingConfig:
    """Get configuration for full training."""
    return TrainingConfig(
        variant=variant,
        fold=fold,
        epochs=50,
        batch_size=8,
        debug=False,
        debug_batches=0,
    )


def get_all_experiment_configs() -> List[TrainingConfig]:
    """Get configurations for all experiments (5 variants x 3 folds)."""
    configs = []
    variants = ['BASELINE', 'WITH_TEXT', 'WITH_CGR', 'WITH_TEXT_CONDITIONED_TYPE', 'FULL']
    
    for variant in variants:
        for fold in [1, 2, 3]:
            configs.append(get_training_config(variant, fold))
    
    return configs


# ==========================================================================
# Testing
# ==========================================================================

def test_config():
    """Test configuration."""
    print("Testing TrainingConfig...")
    print("=" * 60)
    
    # Test default config
    print("\n1. Testing default config...")
    config = TrainingConfig()
    config.print_config()
    print("   ✓ Default config created")
    
    # Test debug config
    print("\n2. Testing debug config...")
    debug_config = get_debug_config("BASELINE", 1)
    print(f"   Variant: {debug_config.variant}")
    print(f"   Fold: {debug_config.fold}")
    print(f"   Epochs: {debug_config.epochs}")
    print(f"   Debug batches: {debug_config.debug_batches}")
    print("   ✓ Debug config created")
    
    # Test all variants
    print("\n3. Testing all variants...")
    for variant in ['BASELINE', 'WITH_TEXT', 'WITH_CGR', 'WITH_TEXT_CONDITIONED_TYPE', 'FULL']:
        config = TrainingConfig(variant=variant, fold=1)
        print(f"   {variant}: dataloader_mode={config.dataloader_mode}")
    print("   ✓ All variants valid")
    
    # Test paths
    print("\n4. Testing paths...")
    config = TrainingConfig(variant="FULL", fold=2)
    print(f"   Experiment dir: {config.get_experiment_dir()}")
    print(f"   Checkpoint path: {config.get_checkpoint_path('best')}")
    print(f"   Log path: {config.get_log_path()}")
    print("   ✓ Paths generated")
    
    # Test to_dict
    print("\n5. Testing serialization...")
    config_dict = config.to_dict()
    restored = TrainingConfig.from_dict(config_dict)
    print(f"   Original variant: {config.variant}")
    print(f"   Restored variant: {restored.variant}")
    print("   ✓ Serialization works")
    
    print("\n" + "=" * 60)
    print("All config tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_config()
