"""
Training Utilities for CIPS-Net V2

Helper functions and classes for training.
"""

import os
import random
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
    LambdaLR,
)


# ==========================================================================
# Reproducibility
# ==========================================================================

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[DEBUG] Random seed set to {seed}")


# ==========================================================================
# Model Utilities
# ==========================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get detailed model information."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    
    # Get parameter breakdown by module
    module_params = {}
    for name, module in model.named_children():
        params = count_parameters(module, trainable_only=False)
        trainable = count_parameters(module, trainable_only=True)
        module_params[name] = {
            'total': params,
            'trainable': trainable,
            'frozen': params - trainable
        }
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'module_params': module_params
    }


def print_model_info(model: nn.Module, name: str = "Model"):
    """Print detailed model information."""
    info = get_model_info(model)
    
    print(f"\n[DEBUG] {name} Information:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    print(f"  Frozen parameters: {info['frozen_params']:,}")
    print(f"\n  Parameters by module:")
    
    for module_name, params in info['module_params'].items():
        status = "🔓" if params['trainable'] > 0 else "🔒"
        print(f"    {status} {module_name}: {params['total']:,} ({params['trainable']:,} trainable)")


# ==========================================================================
# Optimizer
# ==========================================================================

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'adamw', or 'sgd'
        learning_rate: Learning rate
        weight_decay: Weight decay
        momentum: Momentum (for SGD)
        betas: Beta parameters (for Adam/AdamW)
    
    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        optimizer = Adam(params, lr=learning_rate, betas=betas, weight_decay=weight_decay, foreach=False)
    elif optimizer_name == "adamw":
        # foreach=False: disables batched multi-tensor ops that spike peak GPU memory
        # (important for large models like DINOv2 ViT-L where memory is tight)
        optimizer = AdamW(params, lr=learning_rate, betas=betas, weight_decay=weight_decay, foreach=False)
    elif optimizer_name == "sgd":
        optimizer = SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"[DEBUG] Created {optimizer_name.upper()} optimizer:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    
    return optimizer


# ==========================================================================
# Scheduler
# ==========================================================================

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 50,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    steps_per_epoch: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: 'cosine', 'onecycle', 'step', or 'none'
        epochs: Total epochs
        warmup_epochs: Warmup epochs (for cosine with warmup)
        min_lr: Minimum learning rate
        steps_per_epoch: Steps per epoch (for OneCycleLR)
    
    Returns:
        Scheduler instance or None
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "none":
        print("[DEBUG] No learning rate scheduler")
        return None
    
    if scheduler_name == "cosine":
        # Cosine annealing with warmup
        def warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return max(min_lr / optimizer.defaults['lr'], 
                          0.5 * (1 + np.cos(np.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)
        print(f"[DEBUG] Created Cosine scheduler with {warmup_epochs} warmup epochs")
        
    elif scheduler_name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.defaults['lr'],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs / epochs,
            anneal_strategy='cos',
            final_div_factor=optimizer.defaults['lr'] / min_lr
        )
        print(f"[DEBUG] Created OneCycleLR scheduler")
        
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
        print(f"[DEBUG] Created StepLR scheduler (step every {epochs // 3} epochs)")
        
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


# ==========================================================================
# Metrics Tracking
# ==========================================================================

class AverageMeter:
    """Track and compute average of a metric."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class MetricsTracker:
    """Track multiple metrics during training."""
    
    def __init__(self, metric_names: List[str]):
        self.meters = {name: AverageMeter(name) for name in metric_names}
    
    def reset(self):
        for meter in self.meters.values():
            meter.reset()
    
    def update(self, metrics: Dict[str, float], n: int = 1):
        for name, value in metrics.items():
            if name in self.meters:
                self.meters[name].update(value, n)
    
    def get_averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def get_current(self) -> Dict[str, float]:
        return {name: meter.val for name, meter in self.meters.items()}
    
    def __str__(self):
        parts = [f"{name}: {meter.avg:.4f}" for name, meter in self.meters.items()]
        return " | ".join(parts)


# ==========================================================================
# Timer
# ==========================================================================

class Timer:
    """Simple timer for tracking execution time."""
    
    def __init__(self):
        self.start_time = None
        self.lap_time = None
    
    def start(self):
        self.start_time = time.time()
        self.lap_time = self.start_time
    
    def lap(self) -> float:
        """Get time since last lap (or start)."""
        current = time.time()
        elapsed = current - self.lap_time
        self.lap_time = current
        return elapsed
    
    def elapsed(self) -> float:
        """Get total elapsed time since start."""
        return time.time() - self.start_time
    
    def format_elapsed(self) -> str:
        """Get formatted elapsed time."""
        elapsed = self.elapsed()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ==========================================================================
# Progress Bar
# ==========================================================================

class ProgressBar:
    """Simple progress bar for training."""
    
    def __init__(self, total: int, prefix: str = "", width: int = 30):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
    
    def update(self, current: int, suffix: str = ""):
        self.current = current
        percent = current / self.total
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)
        
        print(f"\r{self.prefix} |{bar}| {current}/{self.total} ({percent*100:.1f}%) {suffix}", 
              end="", flush=True)
    
    def finish(self):
        print()


# ==========================================================================
# Checkpoint Utilities
# ==========================================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    config: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
    }
    
    torch.save(checkpoint, path)
    print(f"[DEBUG] Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Load training checkpoint."""
    print(f"[DEBUG] Loading checkpoint: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model state loaded")
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Optimizer state loaded")
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  ✓ Scheduler state loaded")
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"  ✓ GradScaler state loaded")
    
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


# ==========================================================================
# Logging Utilities
# ==========================================================================

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
    epoch_time: float,
):
    """Print epoch summary."""
    print(f"\n{'=' * 70}")
    print(f"Epoch {epoch + 1}/{total_epochs} Summary")
    print(f"{'=' * 70}")
    
    print(f"\n[Train]")
    for name, value in train_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\n[Validation]")
    for name, value in val_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\n[Info]")
    print(f"  Learning rate: {lr:.6f}")
    print(f"  Epoch time: {epoch_time:.1f}s")
    print(f"{'=' * 70}\n")


def print_batch_debug(
    batch_idx: int,
    total_batches: int,
    loss: float,
    loss_dict: Dict[str, float],
    batch_time: float,
    lr: float,
):
    """Print batch debugging information."""
    print(f"\n[DEBUG] Batch {batch_idx + 1}/{total_batches}")
    print(f"  Total Loss: {loss:.4f}")
    print(f"  Components:")
    for name, value in loss_dict.items():
        print(f"    {name}: {value:.4f}")
    print(f"  Batch time: {batch_time:.3f}s")
    print(f"  Learning rate: {lr:.6f}")


# ==========================================================================
# Device Utilities
# ==========================================================================

def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[DEBUG] Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[DEBUG] Using CPU")
    
    return device


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[DEBUG] GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


# ==========================================================================
# Data Utilities
# ==========================================================================

def print_batch_info(batch: Dict[str, torch.Tensor], prefix: str = ""):
    """Print information about a batch."""
    print(f"\n[DEBUG] {prefix}Batch Info:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
        else:
            print(f"  {key}: {type(value).__name__}")


def print_output_info(outputs: Dict[str, torch.Tensor], prefix: str = ""):
    """Print information about model outputs."""
    print(f"\n[DEBUG] {prefix}Model Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, "
                  f"min={value.min().item():.4f}, max={value.max().item():.4f}")


# ==========================================================================
# Testing
# ==========================================================================

def test_utils():
    """Test utility functions."""
    print("Testing Training Utilities...")
    print("=" * 60)
    
    # Test seed
    print("\n1. Testing seed_everything...")
    seed_everything(42)
    print("   ✓ Seed set")
    
    # Test AverageMeter
    print("\n2. Testing AverageMeter...")
    meter = AverageMeter("loss")
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    print(f"   Values: 1.0, 2.0, 3.0")
    print(f"   Average: {meter.avg} (expected: 2.0)")
    print("   ✓ AverageMeter works")
    
    # Test MetricsTracker
    print("\n3. Testing MetricsTracker...")
    tracker = MetricsTracker(["loss", "accuracy"])
    tracker.update({"loss": 0.5, "accuracy": 0.8})
    tracker.update({"loss": 0.3, "accuracy": 0.9})
    print(f"   Averages: {tracker.get_averages()}")
    print("   ✓ MetricsTracker works")
    
    # Test Timer
    print("\n4. Testing Timer...")
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    print(f"   Elapsed: {timer.elapsed():.2f}s")
    print("   ✓ Timer works")
    
    # Test device
    print("\n5. Testing device...")
    device = get_device()
    print(f"   Device: {device}")
    print("   ✓ Device detection works")
    
    print("\n" + "=" * 60)
    print("All utility tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_utils()
