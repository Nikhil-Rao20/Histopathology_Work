"""
Configuration file for CIPS-Net training and evaluation
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Image encoder
    img_encoder_name: str = 'vit_b_16'  # 'vit_b_16' or 'vit_l_16'
    img_encoder_pretrained: bool = True
    
    # Text encoder
    text_encoder_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    # Alternatives: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    text_encoder_freeze: bool = False
    
    # Shared embedding
    embed_dim: int = 768
    
    # Image processing
    img_size: int = 224
    patch_size: int = 16
    
    # Graph reasoning
    num_graph_layers: int = 3
    num_relation_types: int = 3
    
    # Decoder
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    
    # Classes
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        'Neoplastic',
        'Inflammatory',
        'Connective_Soft_tissue',
        'Epithelial',
        'Dead'
    ])
    
    # Dropout
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    # Paths
    data_root: str = "c:/Users/nikhi/Desktop/Histopathology_Work/Dataset"
    images_dir: str = "multi_images"
    masks_dir: str = "multi_masks"
    
    # CSV files
    unique_labels_csv: str = "Images_With_Unique_Labels_Refer_Segmentation_Task.csv"
    permutations_csv: str = "Images_With_Permutations_Labels_Refer_Segmentation_Task.csv"
    
    # Use permutations for data augmentation
    use_permutations: bool = True
    
    # Train/val split
    train_folds: List[int] = field(default_factory=lambda: [1, 2])
    val_folds: List[int] = field(default_factory=lambda: [3])
    
    # Data augmentation
    augmentation: bool = True
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    accumulation_steps: int = 4  # Gradient accumulation
    
    # Schedule
    num_epochs: int = 100
    warmup_epochs: int = 5
    scheduler: str = 'cosine'  # 'cosine' or 'step'
    
    # Loss weights
    seg_loss_weight: float = 1.0
    presence_loss_weight: float = 0.1
    attention_reg_weight: float = 0.01
    
    # Class weights (for handling imbalance)
    use_class_weights: bool = True
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_freq: int = 5  # Save every N epochs
    save_best_only: bool = True
    
    # Logging
    log_dir: str = "logs"
    log_freq: int = 10  # Log every N iterations
    use_wandb: bool = False
    wandb_project: str = "cips-net"
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Reproducibility
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    
    # Metrics
    threshold: float = 0.5
    compute_per_class: bool = True
    compute_per_organ: bool = True
    
    # Visualization
    visualize_predictions: bool = True
    num_vis_samples: int = 10
    
    # Test-time augmentation
    use_tta: bool = False
    tta_transforms: List[str] = field(default_factory=lambda: ['hflip', 'vflip', 'rotate90'])


@dataclass
class Config:
    """Complete configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Experiment
    experiment_name: str = "cips_net_v1"
    device: str = "cuda"  # 'cuda' or 'cpu'
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'eval': self.eval.__dict__,
            'experiment_name': self.experiment_name,
            'device': self.device
        }


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_quick_test_config() -> Config:
    """Get configuration for quick testing."""
    config = Config()
    
    # Use smaller model
    config.model.img_encoder_name = 'vit_b_16'
    config.model.embed_dim = 384
    config.model.num_graph_layers = 2
    config.model.decoder_channels = [256, 128, 64]
    
    # Smaller batch and fewer epochs
    config.training.batch_size = 4
    config.training.num_epochs = 10
    config.data.num_workers = 2
    
    config.experiment_name = "cips_net_quick_test"
    
    return config


if __name__ == "__main__":
    # Print default config
    config = get_default_config()
    
    print("Default CIPS-Net Configuration:")
    print("=" * 80)
    
    print("\n[Model]")
    for key, value in config.model.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Data]")
    for key, value in config.data.__dict__.items():
        if not isinstance(value, list) or len(value) < 10:
            print(f"  {key}: {value}")
    
    print("\n[Training]")
    for key, value in config.training.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Evaluation]")
    for key, value in config.eval.__dict__.items():
        if not isinstance(value, list):
            print(f"  {key}: {value}")
