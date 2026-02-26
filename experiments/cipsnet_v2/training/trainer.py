"""
Trainer Class for CIPS-Net V2

Main trainer class with:
- Mixed precision training (AMP)
- Gradient clipping
- DRW schedule integration
- Comprehensive debugging output
- Checkpoint management
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .config import TrainingConfig
from .experiment_manager import ExperimentManager
from .metrics import ValidationMetrics
from .utils import (
    seed_everything,
    get_device,
    get_optimizer,
    get_scheduler,
    print_model_info,
    print_gpu_memory,
    print_batch_info,
    print_output_info,
    print_batch_debug,
    print_epoch_summary,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    MetricsTracker,
    Timer,
    ProgressBar,
)


class Trainer:
    """
    Trainer class for CIPS-Net V2.
    
    Handles:
    - Training loop with mixed precision
    - Validation loop
    - DRW schedule for class imbalance
    - Checkpoint saving/loading
    - Comprehensive debugging output
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        drw_scheduler: Any,
        config: TrainingConfig,
    ):
        """
        Initialize trainer.
        
        Args:
            model: CIPS-Net V2 model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            loss_fn: HoVerNetLoss or CIPSNetV2Loss
            drw_scheduler: DRW scheduler for class weights
            config: Training configuration
        """
        self.config = config
        self.debug = config.debug
        
        # Device
        self.device = get_device()
        
        # Model
        self.model = model.to(self.device)
        if self.debug:
            print_model_info(self.model, f"CIPS-Net V2 ({config.variant})")
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if self.debug:
            print(f"\n[DEBUG] DataLoaders:")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
            print(f"  Batch size: {config.batch_size}")
        
        # Loss
        self.loss_fn = loss_fn.to(self.device)
        self.drw_scheduler = drw_scheduler
        
        if self.debug:
            print(f"\n[DEBUG] Loss Function:")
            print(f"  Type: {type(loss_fn).__name__}")
            print(f"  LDAM: {config.use_ldam}")
            print(f"  DRW start epoch: {config.drw_start_epoch}")
        
        # Optimizer
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_name=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        steps_per_epoch = len(train_loader) if config.scheduler == 'onecycle' else None
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_name=config.scheduler,
            epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            min_lr=config.min_lr,
            steps_per_epoch=steps_per_epoch,
        )
        
        # Mixed precision
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.debug:
            print(f"\n[DEBUG] Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        
        # Metrics
        self.metric_names = [
            'total', 'np_total', 'hv_total', 'type_total',
            'np_bce', 'np_dice', 'hv_mse', 'hv_msge', 'type_ce', 'type_dice'
        ]
        self.train_metrics = MetricsTracker(self.metric_names)
        self.val_metrics = MetricsTracker(self.metric_names)
        
        # Performance metrics (Dice, IoU, F1, etc.)
        self.val_performance_metrics = ValidationMetrics(
            num_classes=config.num_classes,
            ignore_background_for_type=True,  # Only compute type metrics on nuclei
        )
        
        # Experiment manager for saving outputs
        self.experiment_manager = ExperimentManager(
            experiment_name=config.experiment_name,
            base_dir=config.runs_dir,
            resume=config.resume_from is not None,
        )
        self.experiment_manager.save_config(config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.timer = Timer()
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Prepare batch for training.
        
        Returns:
            inputs: Dict with model inputs
            targets: Dict with target tensors
        """
        # Debug: print batch keys
        if self.debug and hasattr(self, '_batch_keys_printed') is False:
            print(f"\n[DEBUG] Batch keys: {list(batch.keys())}")
            self._batch_keys_printed = True
        
        # Move tensors to device
        # Note: DataLoader collate function pluralizes keys (image -> images)
        image = batch['images'].to(self.device)
        np_target = batch['np_maps'].to(self.device)
        hv_target = batch['hv_maps'].to(self.device)
        type_target = batch['type_maps'].to(self.device)
        
        # Prepare inputs based on variant
        inputs = {'image': image}
        
        # Add text inputs if present (plural keys from collate)
        if 'input_ids' in batch:
            inputs['input_ids'] = batch['input_ids'].to(self.device)
            inputs['attention_mask'] = batch['attention_mask'].to(self.device)
        
        # Add class presence if present
        if 'class_presences' in batch:
            inputs['class_presence'] = batch['class_presences'].to(self.device)
        
        # Prepare targets
        targets = {
            'np': np_target,
            'hv': hv_target,
            'type': type_target,
        }
        
        return inputs, targets
    
    def _forward_pass(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model.
        
        Args:
            inputs: Dict with model inputs
        
        Returns:
            Model outputs dict
        """
        # Check variant and call model accordingly
        if self.config.variant == 'BASELINE':
            outputs = self.model(inputs['image'])
        else:
            # Text-based variants
            outputs = self.model(
                image=inputs['image'],
                input_ids=inputs.get('input_ids'),
                attention_mask=inputs.get('attention_mask'),
                class_presence=inputs.get('class_presence'),
            )
        
        return outputs
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        # Update DRW weights
        if self.drw_scheduler is not None:
            drw_weights = self.drw_scheduler.get_weights(epoch)
            self.loss_fn.update_type_weights(drw_weights)
            
            if self.debug:
                drw_active = self.drw_scheduler.is_drw_active(epoch)
                print(f"\n[DEBUG] DRW: active={drw_active}, weights[:3]={drw_weights[:3].tolist()}")
        
        num_batches = len(self.train_loader)
        debug_batches = self.config.debug_batches if self.config.debug_batches > 0 else num_batches
        
        epoch_timer = Timer()
        epoch_timer.start()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Debug mode: limit batches
            if self.debug and batch_idx >= debug_batches:
                print(f"\n[DEBUG] Stopping after {debug_batches} batches (debug mode)")
                break
            
            batch_timer = Timer()
            batch_timer.start()
            
            # Prepare batch
            inputs, targets = self._prepare_batch(batch)
            
            if self.debug and batch_idx == 0:
                print_batch_info(batch, "Input ")
            
            # Forward pass with AMP
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self._forward_pass(inputs)
                
                if self.debug and batch_idx == 0:
                    print_output_info(outputs, "")
                
                loss, loss_dict = self.loss_fn(outputs, targets)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            batch_size = inputs['image'].size(0)
            self.train_metrics.update(loss_dict, batch_size)
            
            # Debug logging
            if self.debug and (batch_idx + 1) % self.config.log_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print_batch_debug(
                    batch_idx=batch_idx,
                    total_batches=min(num_batches, debug_batches),
                    loss=loss.item(),
                    loss_dict=loss_dict,
                    batch_time=batch_timer.elapsed(),
                    lr=lr,
                )
                print_gpu_memory()
        
        # Step scheduler (epoch-based)
        if self.scheduler is not None and self.config.scheduler != 'onecycle':
            self.scheduler.step()
        
        if self.debug:
            print(f"\n[DEBUG] Epoch {epoch + 1} training completed in {epoch_timer.format_elapsed()}")
        
        return self.train_metrics.get_averages()
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (loss_metrics, performance_metrics)
        """
        self.model.eval()
        self.val_metrics.reset()
        self.val_performance_metrics.reset()
        
        num_batches = len(self.val_loader)
        debug_batches = self.config.debug_batches if self.config.debug_batches > 0 else num_batches
        
        if self.debug:
            print(f"\n[DEBUG] Starting validation...")
        
        for batch_idx, batch in enumerate(self.val_loader):
            # Debug mode: limit batches
            if self.debug and debug_batches > 0 and batch_idx >= debug_batches:
                break
            
            # Prepare batch
            inputs, targets = self._prepare_batch(batch)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self._forward_pass(inputs)
                loss, loss_dict = self.loss_fn(outputs, targets)
            
            # Update loss metrics
            batch_size = inputs['image'].size(0)
            self.val_metrics.update(loss_dict, batch_size)
            
            # Update performance metrics (Dice, IoU, F1, etc.)
            self.val_performance_metrics.update(
                pred_np=outputs['np'],
                pred_type=outputs['type'],
                target_np=targets['np'],
                target_type=targets['type'],
            )
        
        # Get results
        val_loss_averages = self.val_metrics.get_averages()
        val_perf_metrics = self.val_performance_metrics.compute()
        
        if self.debug:
            print(f"\n[DEBUG] Validation Loss Results:")
            for name, value in val_loss_averages.items():
                print(f"  {name}: {value:.4f}")
            
            print(f"\n[DEBUG] Validation Performance Metrics:")
            for name, value in val_perf_metrics.items():
                print(f"  {name}: {value:.4f}")
        
        return val_loss_averages, val_perf_metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training history dict
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        if self.debug:
            self.config.print_config()
        
        seed_everything(self.config.seed)
        
        # Start experiment logging
        self.experiment_manager.start_logging()
        
        self.timer.start()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_num = epoch + 1  # 1-indexed for display
            
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch_num}/{self.config.epochs}")
            print(f"{'=' * 70}")
            
            epoch_timer = Timer()
            epoch_timer.start()
            
            # Train
            train_metrics = self._train_epoch(epoch)
            
            # Validate (returns loss metrics and performance metrics)
            val_loss_metrics, val_perf_metrics = self._validate_epoch(epoch)
            
            # Combine all validation metrics for logging
            val_metrics = {**val_loss_metrics, **val_perf_metrics}
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check if best (based on loss)
            is_best = val_loss_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss_metrics['total']
                print(f"[DEBUG] New best validation loss: {self.best_val_loss:.4f}")
            
            # Log epoch to experiment manager
            self.experiment_manager.log_epoch(
                epoch=epoch_num,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                epoch_time=epoch_timer.elapsed(),
                is_best=is_best,
            )
            
            # Print epoch summary
            print_epoch_summary(
                epoch=epoch,
                total_epochs=self.config.epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                epoch_time=epoch_timer.elapsed(),
            )
            
            # Save checkpoints (only best and last)
            if is_best:
                self._save_checkpoint("best", val_metrics)
            
            # Always save last checkpoint
            self._save_checkpoint("last", val_metrics)
        
        # Training complete
        total_time = self.timer.format_elapsed()
        
        # Stop logging and generate plots
        self.experiment_manager.stop_logging()
        self.experiment_manager.plot_all()
        self.experiment_manager.print_summary()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Outputs saved to: {self.experiment_manager.experiment_dir}")
        print("=" * 70 + "\n")
        
        return self.experiment_manager.history
    
    def _save_checkpoint(self, suffix: str, metrics: Dict[str, float]):
        """Save checkpoint."""
        path = self.experiment_manager.get_checkpoint_path(suffix)
        
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics=metrics,
            config=self.config,
            scaler=self.scaler,
        )
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=str(self.device),
        )
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['metrics'].get('total', float('inf'))
        
        print(f"[DEBUG] Resuming from epoch {self.current_epoch}")


# ==========================================================================
# Factory Function
# ==========================================================================

def create_trainer(config: TrainingConfig) -> Trainer:
    """
    Create trainer from configuration.
    
    This is the main entry point for training setup.
    
    Args:
        config: Training configuration
    
    Returns:
        Trainer instance ready for training
    """
    print("\n" + "=" * 70)
    print("CREATING TRAINER")
    print("=" * 70)
    
    # Import model and dataset modules
    from experiments.cipsnet_v2.models import create_model
    from experiments.cipsnet_v2.datasets import create_fold_dataloaders
    from experiments.cipsnet_v2.losses import (
        CIPSNetV2Loss,
        DRWScheduler,
        get_class_frequencies_pannuke,
    )
    
    print(f"\n[DEBUG] Creating model: {config.variant}")
    
    # Create model (pass variant as string)
    model = create_model(
        variant=config.variant,  # String like 'BASELINE', 'FULL', etc.
        num_classes=config.num_classes,
        backbone=config.image_encoder,
        freeze_backbone=config.freeze_image_encoder,
    )
    
    print(f"\n[DEBUG] Creating dataloaders for fold {config.fold}")
    
    # Create dataloaders
    train_loader, val_loader, _ = create_fold_dataloaders(
        data_root=config.pannuke_path,
        mode=config.dataloader_mode,
        test_fold=config.fold,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    
    print(f"\n[DEBUG] Creating loss function")
    
    # Get class frequencies
    cls_num_list = get_class_frequencies_pannuke()
    
    # Create loss function
    loss_fn = CIPSNetV2Loss(
        num_classes=config.num_classes,
        cls_num_list=cls_num_list,
        use_ldam=config.use_ldam,
        np_weight=config.np_weight,
        hv_weight=config.hv_weight,
        type_weight=config.type_weight,
    )
    
    print(f"\n[DEBUG] Creating DRW scheduler")
    
    # Create DRW scheduler
    drw_scheduler = DRWScheduler(
        cls_num_list=cls_num_list,
        total_epochs=config.epochs,
        drw_start_epoch=config.drw_start_epoch,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    print(f"\n[DEBUG] Creating trainer")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        drw_scheduler=drw_scheduler,
        config=config,
    )
    
    print("\n" + "=" * 70)
    print("TRAINER READY")
    print("=" * 70)
    
    return trainer


# ==========================================================================
# Testing
# ==========================================================================

def test_trainer():
    """Test trainer creation (without actual training)."""
    print("Testing Trainer Setup...")
    print("=" * 60)
    
    # This is a quick test that doesn't require data
    print("\n1. Testing TrainingConfig...")
    config = TrainingConfig(
        variant="BASELINE",
        fold=1,
        epochs=2,
        batch_size=4,
        debug=True,
        debug_batches=1,
    )
    print(f"   Variant: {config.variant}")
    print(f"   Fold: {config.fold}")
    print("   ✓ Config created")
    
    print("\n2. Testing Timer...")
    timer = Timer()
    timer.start()
    print(f"   Elapsed: {timer.elapsed():.4f}s")
    print("   ✓ Timer works")
    
    print("\n3. Testing MetricsTracker...")
    tracker = MetricsTracker(['loss', 'accuracy'])
    tracker.update({'loss': 0.5, 'accuracy': 0.8})
    print(f"   Averages: {tracker.get_averages()}")
    print("   ✓ MetricsTracker works")
    
    print("\n" + "=" * 60)
    print("Trainer component tests passed!")
    print("=" * 60)
    print("\nTo test full trainer, run:")
    print("  python train.py --variant BASELINE --fold 1 --debug")


if __name__ == "__main__":
    test_trainer()
