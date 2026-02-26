"""
Experiment Runner for CIPS-Net V2
==================================

Orchestrates complete experiment workflow:
1. Setup experiment directory structure
2. Train model on all 3 folds (with early stopping)
3. Test best models on respective test folds
4. Aggregate results with mean±std
5. Generate paper-ready outputs

This is the main entry point for running experiments.

Usage:
------
    from experiments.cipsnet_v2.core import ExperimentRunner, ExperimentConfig
    
    config = ExperimentConfig(variant='BASELINE', epochs=50)
    runner = ExperimentRunner(config)
    results = runner.run()  # Runs everything
"""

import os
import sys
import gc
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .experiment_config import ExperimentConfig
from .result_aggregator import ResultAggregator


# ==========================================================================
# Early Stopping
# ==========================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to count as improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
    """
    
    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 0.0001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current validation value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset for new fold."""
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.should_stop = False
        self.best_epoch = 0


# ==========================================================================
# Fold Trainer
# ==========================================================================

class FoldTrainer:
    """
    Trainer for a single fold.
    
    Handles:
    - Training loop with early stopping
    - Validation
    - Checkpoint saving (best + last)
    - Training log CSV
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        drw_scheduler: Optional[Any],
        config: ExperimentConfig,
        fold: int,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.drw_scheduler = drw_scheduler
        self.config = config
        self.fold = fold
        self.device = device
        
        # AMP
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min'  # Minimize validation loss
        ) if config.early_stopping else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Create directories
        os.makedirs(config.get_checkpoint_dir(fold), exist_ok=True)
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Prepare batch for training."""
        image = batch['images'].to(self.device)
        np_target = batch['np_maps'].to(self.device)
        hv_target = batch['hv_maps'].to(self.device)
        type_target = batch['type_maps'].to(self.device)
        
        inputs = {'image': image}
        
        # Add text inputs if present
        if 'input_ids' in batch:
            inputs['input_ids'] = batch['input_ids'].to(self.device)
            inputs['attention_mask'] = batch['attention_mask'].to(self.device)
        
        if 'class_presences' in batch:
            inputs['class_presence'] = batch['class_presences'].to(self.device)
        
        # Add instructions if available (for text variants)
        if 'instructions' in batch:
            inputs['instructions'] = batch['instructions']
        
        targets = {
            'np': np_target,
            'hv': hv_target,
            'type': type_target,
        }
        
        # Add IE-specific targets (distance transform + instance map)
        variant_upper = self.config.variant.upper()
        if variant_upper == 'LVIT_IE':
            if 'dist_maps' in batch:
                targets['dist'] = batch['dist_maps'].to(self.device)
            if 'instance_maps' in batch:
                targets['instance'] = batch['instance_maps'].to(self.device)
                # Also pass to inputs for forward pass (needed by inst-pooled type head)
                inputs['instance_maps'] = targets['instance']
        
        return inputs, targets
    
    def _forward(self, inputs: Dict) -> Dict:
        """Forward pass."""
        variant_upper = self.config.variant.upper()
        
        if variant_upper == 'BASELINE':
            return self.model(inputs['image'])
        
        # LViT3: needs return_contrastive_features=True for training
        elif variant_upper == 'LVIT3':
            return self.model(
                images=inputs['image'],
                texts=inputs.get('instructions'),
                return_contrastive_features=True,  # Enable contrastive features
            )
        
        # LViT4: Multi-stage fusion + PWAM, always returns multi-scale contrastive features
        elif variant_upper == 'LVIT4':
            return self.model(
                images=inputs['image'],
                texts=inputs.get('instructions'),
                return_contrastive_features=True,  # Enable multi-scale contrastive
            )
        
        # LViT5: Ultimate - Cross-Modal Decoder + Grounding + All features
        elif variant_upper == 'LVIT5':
            return self.model(
                images=inputs['image'],
                texts=inputs.get('instructions'),
                return_contrastive_features=True,  # Enable all contrastive features
                return_grounding=True,  # Enable grounding head
            )
        
        # LViT-IE: Instance Embedding decoder — pass instance_maps during training
        elif variant_upper == 'LVIT_IE':
            return self.model(
                images=inputs['image'],
                texts=inputs.get('instructions'),
                instance_maps=inputs.get('instance_maps'),
            )
        
        # New models use 'texts' parameter (LVIT, LVIT2, CRIS, GROUNDING_DINO)
        elif variant_upper in ['CRIS', 'LVIT', 'LVIT2', 'LVIT3', 'LVIT4', 'LVIT5', 'GROUNDING_DINO']:
            return self.model(
                images=inputs['image'],
                texts=inputs.get('instructions'),
            )
        
        # Original variants (LAVT, WITH_TEXT, etc.) use 'instructions' parameter
        else:
            return self.model(
                images=inputs['image'],
                instructions=inputs.get('instructions'),
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Update DRW weights
        if self.drw_scheduler is not None:
            drw_weights = self.drw_scheduler.get_weights(epoch)
            self.loss_fn.update_type_weights(drw_weights)
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        debug_batches = self.config.debug_batches if self.config.debug_batches > 0 else num_batches
        
        for batch_idx, batch in enumerate(self.train_loader):
            if self.config.debug and batch_idx >= debug_batches:
                break
            
            inputs, targets = self._prepare_batch(batch)
            
            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            with autocast('cuda', enabled=self.use_amp):
                outputs = self._forward(inputs)
                loss, loss_dict = self.loss_fn(outputs, targets)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Delete intermediates to free memory
            del inputs, targets, outputs, loss
            
            # Periodic garbage collection
            if self.config.low_memory_mode and (batch_idx + 1) % self.config.gc_every_n_batches == 0:
                clear_memory()
            
            # Progress logging
            if (batch_idx + 1) % self.config.log_every_n_batches == 0:
                print(f"    Batch {batch_idx + 1}/{min(num_batches, debug_batches)}, Loss: {total_loss / (batch_idx + 1):.4f}")
        
        # Step scheduler
        if self.scheduler is not None and self.config.scheduler != 'onecycle':
            self.scheduler.step()
        
        # Clear memory after epoch
        if self.config.low_memory_mode:
            clear_memory()
        
        avg_loss = total_loss / min(num_batches, debug_batches)
        return {'total': avg_loss}
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        debug_batches = self.config.debug_batches if self.config.debug_batches > 0 else num_batches
        
        for batch_idx, batch in enumerate(self.val_loader):
            if self.config.debug and batch_idx >= debug_batches:
                break
            
            inputs, targets = self._prepare_batch(batch)
            
            with autocast('cuda', enabled=self.use_amp):
                outputs = self._forward(inputs)
                loss, _ = self.loss_fn(outputs, targets)
            
            total_loss += loss.item()
            
            # Free memory
            del inputs, targets, outputs, loss
        
        # Clear memory after validation
        if self.config.low_memory_mode:
            clear_memory()
        
        avg_loss = total_loss / min(num_batches, debug_batches)
        return {'total': avg_loss}
    
    def save_checkpoint(self, checkpoint_type: str, epoch: int, metrics: Dict):
        """Save checkpoint."""
        path = self.config.get_checkpoint_path(self.fold, checkpoint_type)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config.to_dict(),
            'variant': self.config.variant,
            'fold': self.fold,
        }
        
        torch.save(checkpoint, path)
    
    def save_training_log(self):
        """Save training log to CSV."""
        if self.training_history:
            df = pd.DataFrame(self.training_history)
            df.to_csv(self.config.get_training_log_path(self.fold), index=False)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training summary with best metrics
        """
        print(f"\n  [Fold {self.fold}] Starting training for {self.config.epochs} epochs")
        
        start_time = time.time()
        stopped_early = False
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch()
            
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']
                self.save_checkpoint('best', epoch, val_metrics)
            
            # Always save last
            self.save_checkpoint('last', epoch, val_metrics)
            
            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['total'],
                'val_loss': val_metrics['total'],
                'lr': lr,
                'is_best': is_best,
                'time': epoch_time,
            })
            
            # Print progress
            best_marker = " *" if is_best else ""
            print(f"    Epoch {epoch + 1}/{self.config.epochs} | "
                  f"Train: {train_metrics['total']:.4f} | "
                  f"Val: {val_metrics['total']:.4f}{best_marker} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['total'], epoch):
                    print(f"\n    [Early Stopping] No improvement for {self.config.early_stopping_patience} epochs")
                    print(f"    Best epoch: {self.early_stopping.best_epoch + 1}")
                    stopped_early = True
                    break
        
        # Save training log
        self.save_training_log()
        
        total_time = time.time() - start_time
        
        return {
            'fold': self.fold,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.early_stopping.best_epoch + 1 if self.early_stopping else len(self.training_history),
            'total_epochs': len(self.training_history),
            'stopped_early': stopped_early,
            'total_time': total_time,
        }


# ==========================================================================
# Experiment Runner
# ==========================================================================

class ExperimentRunner:
    """
    Main experiment orchestrator.
    
    Runs complete 3-fold cross-validation experiment:
    1. Creates experiment directory structure
    2. Trains on all 3 folds
    3. Tests best models
    4. Aggregates results
    
    Usage:
        config = ExperimentConfig(variant='BASELINE', epochs=50)
        runner = ExperimentRunner(config)
        results = runner.run()
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()
        
        # Results storage
        self.training_summaries: Dict[int, Dict] = {}
        self.test_results: Dict[int, Dict] = {}
        
        # Setup directories
        self._setup_directories()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(f'CIPS-Net-{self.config.variant}')
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def _setup_directories(self):
        """Create experiment directory structure."""
        os.makedirs(self.config.experiment_dir, exist_ok=True)
        
        for fold in [1, 2, 3]:
            os.makedirs(self.config.get_fold_dir(fold), exist_ok=True)
            os.makedirs(self.config.get_checkpoint_dir(fold), exist_ok=True)
            os.makedirs(self.config.get_results_dir(fold), exist_ok=True)
        
        os.makedirs(self.config.get_aggregate_dir(), exist_ok=True)
        
        # Save configuration
        self.config.save()
        
        print(f"\n[Experiment Setup]")
        print(f"  Variant: {self.config.variant}")
        print(f"  Experiment: {self.config.experiment_name}")
        print(f"  Directory: {self.config.experiment_dir}")
    
    def _create_model(self):
        """Create model for variant."""
        variant_upper = self.config.variant.upper()
        
        # Check if LAVT variant
        if variant_upper == 'LAVT':
            from experiments.cipsnet_v2.models import create_lavt_model
            
            model = create_lavt_model(
                num_classes=self.config.num_classes,
                img_size=256,
                pretrained=True,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
            )
        
        # CRIS Model
        elif variant_upper == 'CRIS':
            from experiments.cipsnet_v2.models import create_cris_model
            
            model = create_cris_model(
                num_classes=self.config.num_classes,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
                freeze_image_encoder=self.config.freeze_image_encoder if hasattr(self.config, 'freeze_image_encoder') else False,
            )
        
        # LViT Model
        elif variant_upper == 'LVIT':
            from experiments.cipsnet_v2.models import create_lvit_model
            
            # Get backbone option (default to 'vit' if not specified)
            backbone = getattr(self.config, 'backbone', 'vit')
            freeze_dinov2 = getattr(self.config, 'freeze_dinov2_backbone', False)
            dinov2_path = getattr(self.config, 'dinov2_pretrained_path', '')
            use_grad_ckpt = getattr(self.config, 'use_gradient_checkpointing', False)
            
            model = create_lvit_model(
                num_classes=self.config.num_classes,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
                img_size=256,
                backbone=backbone,
                freeze_dinov2_backbone=freeze_dinov2,
                dinov2_pretrained_path=dinov2_path,
                use_gradient_checkpointing=use_grad_ckpt,
            )
        
        # LViT2 Model (Enhanced with Deep Supervision + Aux Classification)
        elif variant_upper == 'LVIT2':
            from experiments.cipsnet_v2.models import create_lvit2_model
            
            # Get backbone option (default to 'vit' if not specified)
            backbone = getattr(self.config, 'backbone', 'vit')
            
            model = create_lvit2_model(
                num_classes=self.config.num_classes,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
                img_size=256,
                backbone=backbone,
                deep_supervision=True,
                aux_classification=True,
            )
        
        # LViT3 Model (Instance Normalization + Contrastive Loss Support)
        elif variant_upper == 'LVIT3':
            from experiments.cipsnet_v2.models import create_lvit3_model
            
            # Check if contrastive loss is enabled via config (default: True)
            enable_contrastive = getattr(self.config, 'enable_contrastive', True)
            
            model = create_lvit3_model(
                num_classes=self.config.num_classes,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
                img_size=256,
                enable_contrastive=enable_contrastive,
            )
        
        # LViT4 Model (Phase 2: Multi-stage Fusion + PWAM)
        elif variant_upper == 'LVIT4':
            from experiments.cipsnet_v2.models import create_lvit4_model
            
            model = create_lvit4_model(
                num_classes=self.config.num_classes,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
                img_size=256,
            )
        
        # LViT5 Model (Phase 3: Ultimate - Cross-Modal Decoder + Grounding)
        elif variant_upper == 'LVIT5':
            from experiments.cipsnet_v2.models import create_lvit5_model
            
            model = create_lvit5_model(
                num_classes=self.config.num_classes,
                img_size=256,
                pretrained=True,
            )
        
        # Grounding DINO Model
        elif variant_upper == 'GROUNDING_DINO':
            from experiments.cipsnet_v2.models import create_grounding_dino_model
            
            model = create_grounding_dino_model(
                num_classes=self.config.num_classes,
                num_queries=100,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
                img_size=256,
            )
        
        # LViT-IE Model (Instance Embedding decoder — novel, Phase 3)
        elif variant_upper == 'LVIT_IE':
            from experiments.cipsnet_v2.models import create_lvit_ie_model
            
            backbone = getattr(self.config, 'backbone', 'vit')
            freeze_dinov2 = getattr(self.config, 'freeze_dinov2_backbone', False)
            dinov2_path = getattr(self.config, 'dinov2_pretrained_path', '')
            use_grad_ckpt = getattr(self.config, 'use_gradient_checkpointing', False)
            
            model = create_lvit_ie_model(
                num_classes=self.config.num_classes,
                freeze_text_encoder=self.config.freeze_text_encoder if hasattr(self.config, 'freeze_text_encoder') else True,
                img_size=256,
                backbone=backbone,
                instance_embed_dim=16,
                freeze_dinov2_backbone=freeze_dinov2,
                dinov2_pretrained_path=dinov2_path,
                use_gradient_checkpointing=use_grad_ckpt,
            )
        
        # Original CIPSNet variants
        else:
            from experiments.cipsnet_v2.models import create_model
            
            model = create_model(
                variant=self.config.variant,
                num_classes=self.config.num_classes,
                backbone=self.config.image_encoder,
                freeze_backbone=self.config.freeze_image_encoder,
            )
        
        return model.to(self.device)

    def _create_dataloaders(self, fold: int):
        """Create dataloaders for a fold using memory-optimized loader."""
        # Build augmentation config if dataset expansion is enabled
        aug_config = None
        if getattr(self.config, 'use_data_augmentation', False):
            from experiments.cipsnet_v2.datasets.augmentation import AugmentationConfig
            aug_config = AugmentationConfig(
                enabled=True,
                stain_jitter_light=getattr(self.config, 'augmentation_stain_light', True),
                stain_jitter_strong=getattr(self.config, 'augmentation_stain_strong', True),
                elastic_deformation=getattr(self.config, 'augmentation_elastic', True),
                gaussian_noise=getattr(self.config, 'augmentation_gaussian_noise', False),
                seed=self.config.seed,
            )
            print(f"\n[Augmentation] Dataset expansion enabled: {aug_config.expansion_factor}×")
            print(f"  Types: {aug_config.augmentation_names}")
        
        if self.config.use_permutation_dataloader:
            # Use the new dataloader with CSV support
            from experiments.cipsnet_v2.datasets import create_permutation_fold_dataloaders
            
            train_loader, val_loader, test_loader = create_permutation_fold_dataloaders(
                data_root=self.config.pannuke_path,
                test_fold=fold,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                variant=self.config.variant,
                use_permutations=self.config.use_permutations_csv,
                augmentation_config=aug_config,
            )
        else:
            # Use original optimized dataloader
            from experiments.cipsnet_v2.datasets import create_optimized_fold_dataloaders
            
            train_loader, val_loader, test_loader = create_optimized_fold_dataloaders(
                data_root=self.config.pannuke_path,
                mode=self.config.dataloader_mode,
                test_fold=fold,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                augmentation_config=aug_config,
            )
        
        return train_loader, val_loader, test_loader
    
    def _create_loss_and_schedulers(self):
        """Create loss function and DRW scheduler."""
        from experiments.cipsnet_v2.losses import (
            CIPSNetV2Loss,
            DRWScheduler,
            get_class_frequencies_pannuke,
        )
        
        cls_num_list = get_class_frequencies_pannuke()
        
        # Get loss configuration from config
        loss_type = getattr(self.config, 'loss_type', 'weighted_focal')
        focal_gamma = getattr(self.config, 'focal_gamma', 2.0)
        use_class_weights = getattr(self.config, 'use_class_weights', True)
        use_drw = getattr(self.config, 'use_drw', True)
        drw_start_epoch = getattr(self.config, 'drw_start_epoch', 15)
        drw_start_ratio = getattr(self.config, 'drw_start_ratio', 0.3)
        
        # Handle legacy use_ldam parameter
        use_ldam = getattr(self.config, 'use_ldam', False)
        if use_ldam and loss_type not in ['ldam']:
            loss_type = 'ldam'
        
        # Use LViT2Loss for LVIT2 variant (includes Deep Supervision + Aux Classification)
        variant_upper = self.config.variant.upper()
        if variant_upper == 'LVIT2':
            from experiments.cipsnet_v2.losses.lvit2_loss import LViT2Loss
            
            loss_fn = LViT2Loss(
                num_classes=self.config.num_classes,
                np_weight=self.config.np_weight,
                hv_weight=self.config.hv_weight,
                type_weight=self.config.type_weight,
                type_dice_weight=0.5,  # NEW: Dice loss
                label_smoothing=0.1,   # NEW: Label smoothing
                focal_gamma=focal_gamma,
                deep_supervision=True,
                ds_weight=0.5,
                aux_classification=True,
                aux_weight=0.3,
                use_drw=use_drw,
            ).to(self.device)
        
        # Use LViT3Loss for LVIT3 variant (includes Contrastive Loss)
        elif variant_upper == 'LVIT3':
            from experiments.cipsnet_v2.losses.lvit3_loss import LViT3Loss
            
            # Get contrastive loss settings from config
            contrastive_weight = getattr(self.config, 'contrastive_weight', 0.5)
            contrastive_temperature = getattr(self.config, 'contrastive_temperature', 0.07)
            
            loss_fn = LViT3Loss(
                num_classes=self.config.num_classes,
                np_weight=self.config.np_weight,
                hv_weight=self.config.hv_weight,
                type_weight=self.config.type_weight,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                use_class_weights=use_class_weights,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
                use_drw=use_drw,
                cls_num_list=cls_num_list,
            ).to(self.device)
        
        # Use LViT4Loss for LVIT4 variant (Multi-scale Contrastive + PWAM)
        elif variant_upper == 'LVIT4':
            from experiments.cipsnet_v2.losses.lvit4_loss import LViT4Loss
            
            # Get contrastive loss settings from config
            contrastive_weight = getattr(self.config, 'contrastive_weight', 0.5)
            contrastive_scale_weights = getattr(self.config, 'contrastive_scale_weights', [0.5, 0.3, 0.2])
            
            loss_fn = LViT4Loss(
                num_classes=self.config.num_classes,
                np_weight=self.config.np_weight,
                hv_weight=self.config.hv_weight,
                type_weight=self.config.type_weight,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                use_class_weights=use_class_weights,
                contrastive_weight=contrastive_weight,
                contrastive_scale_weights=contrastive_scale_weights,
                use_drw=use_drw,
                cls_num_list=cls_num_list,
            ).to(self.device)
        
        # Use LViT5Loss for LVIT5 variant (Ultimate - All improvements)
        elif variant_upper == 'LVIT5':
            from experiments.cipsnet_v2.losses.lvit5_loss import LViT5Loss
            
            # Get loss weights from config (with stronger defaults)
            pixel_contrastive_weight = getattr(self.config, 'pixel_contrastive_weight', 1.0)
            batch_contrastive_weight = getattr(self.config, 'batch_contrastive_weight', 0.5)
            grounding_weight = getattr(self.config, 'grounding_weight', 1.0)
            attention_reg_weight = getattr(self.config, 'attention_reg_weight', 0.1)
            
            loss_fn = LViT5Loss(
                num_classes=self.config.num_classes,
                np_weight=self.config.np_weight,
                hv_weight=self.config.hv_weight,
                type_weight=self.config.type_weight,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                use_class_weights=use_class_weights,
                pixel_contrastive_weight=pixel_contrastive_weight,
                batch_contrastive_weight=batch_contrastive_weight,
                grounding_weight=grounding_weight,
                attention_reg_weight=attention_reg_weight,
                use_drw=use_drw,
                cls_num_list=cls_num_list,
            ).to(self.device)
        
        # Use LViTIELoss for LVIT_IE variant (Instance Embedding decoder)
        elif variant_upper == 'LVIT_IE':
            from experiments.cipsnet_v2.losses import create_lvit_ie_loss
            
            loss_fn = create_lvit_ie_loss(
                num_classes=self.config.num_classes,
                focal_gamma=focal_gamma,
                type_weight=self.config.type_weight,
                use_class_weights=use_class_weights,
            ).to(self.device)
        
        else:
            loss_fn = CIPSNetV2Loss(
                num_classes=self.config.num_classes,
                cls_num_list=cls_num_list,
                np_weight=self.config.np_weight,
                hv_weight=self.config.hv_weight,
                type_weight=self.config.type_weight,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                use_class_weights=use_class_weights,
                ldam_max_m=getattr(self.config, 'ldam_max_m', 0.5),
                ldam_s=getattr(self.config, 'ldam_s', 30.0),
            ).to(self.device)
        
        # Create DRW scheduler
        drw_scheduler = DRWScheduler(
            cls_num_list=cls_num_list,
            total_epochs=self.config.epochs,
            drw_start_epoch=drw_start_epoch if use_drw else self.config.epochs + 1,  # Never start if disabled
            drw_start_ratio=drw_start_ratio,
            device=str(self.device),
        )
        
        return loss_fn, drw_scheduler
    
    def _create_optimizer_and_scheduler(self, model: nn.Module):
        """Create optimizer and LR scheduler."""
        from experiments.cipsnet_v2.training.utils import get_optimizer, get_scheduler
        
        optimizer = get_optimizer(
            model=model,
            optimizer_name=self.config.optimizer,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name=self.config.scheduler,
            epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            min_lr=self.config.min_lr,
        )
        
        return optimizer, scheduler
    
    def train_fold(self, fold: int) -> Dict[str, Any]:
        """
        Train a single fold.
        
        Args:
            fold: Fold number (1, 2, or 3)
            
        Returns:
            Training summary
        """
        print(f"\n{'=' * 70}")
        print(f"TRAINING FOLD {fold}/3")
        print(f"{'=' * 70}")
        print(f"  Train folds: {[f for f in [1,2,3] if f != fold]}")
        print(f"  Test fold: {fold}")
        
        # Clear memory before starting fold
        clear_memory()
        
        # Create components (fresh for each fold)
        model = self._create_model()
        train_loader, val_loader, _ = self._create_dataloaders(fold)
        loss_fn, drw_scheduler = self._create_loss_and_schedulers()
        optimizer, scheduler = self._create_optimizer_and_scheduler(model)
        
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        
        # Create trainer
        trainer = FoldTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            drw_scheduler=drw_scheduler,
            config=self.config,
            fold=fold,
            device=self.device,
        )
        
        # Train
        summary = trainer.train()
        self.training_summaries[fold] = summary
        
        print(f"\n  [Fold {fold}] Training complete")
        print(f"    Best val loss: {summary['best_val_loss']:.4f}")
        print(f"    Best epoch: {summary['best_epoch']}")
        print(f"    Total time: {summary['total_time']/60:.1f} min")
        
        # Aggressive memory cleanup after fold
        del model, trainer, train_loader, val_loader, loss_fn, drw_scheduler, optimizer, scheduler
        clear_memory()
        
        return summary
    
    def test_fold(self, fold: int) -> Dict[str, Any]:
        """
        Test a single fold using its best model.
        
        Args:
            fold: Fold number
            
        Returns:
            Test results
        """
        print(f"\n{'=' * 70}")
        print(f"TESTING FOLD {fold}/3")
        print(f"{'=' * 70}")
        
        # Clear memory before testing
        clear_memory()
        
        from experiments.cipsnet_v2.testing import PanNukeEvaluator, EvaluationConfig
        from experiments.cipsnet_v2.testing.post_processing import PostProcessConfig
        
        # Load best model
        checkpoint_path = self.config.get_checkpoint_path(fold, 'best')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = self._create_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint  # Free checkpoint memory
        model.eval()
        
        print(f"  Loaded checkpoint: {checkpoint_path}")
        
        # Get test loader
        _, _, test_loader = self._create_dataloaders(fold)
        print(f"  Test samples: {len(test_loader.dataset)}")
        
        # Create evaluator
        eval_config = EvaluationConfig(
            np_threshold=self.config.np_threshold,
            min_instance_size=self.config.min_instance_size,
            marker_erosion_size=self.config.marker_erosion_size,
            save_predictions=self.config.save_predictions,
        )
        
        evaluator = PanNukeEvaluator(config=eval_config, device=str(self.device))
        
        # Run evaluation
        print(f"  Evaluating...")
        
        from tqdm import tqdm
        
        variant_upper = self.config.variant.upper()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"  Testing Fold {fold}")):
                images = batch['images'].to(self.device)
                gt_instances = batch['instance_maps'].numpy()
                gt_types = batch['type_maps'].numpy()
                tissues = batch['tissues']
                indices = batch['indices']
                
                # Forward pass
                if variant_upper == 'BASELINE':
                    outputs = model(images)
                elif variant_upper == 'LVIT_IE':
                    # IE model: no instance_maps at test time → pixel fallback
                    outputs = model(
                        images=images,
                        texts=batch.get('instructions', None),
                        instance_maps=None,
                    )
                elif variant_upper in ['CRIS', 'LVIT', 'LVIT3', 'LVIT4', 'LVIT5', 'GROUNDING_DINO']:
                    # New models use 'texts' parameter
                    outputs = model(
                        images=images,
                        texts=batch.get('instructions', None),
                    )
                else:
                    # Original variants use 'instructions' parameter
                    outputs = model(
                        images=images,
                        instructions=batch.get('instructions', None),
                    )
                
                # Add batch to evaluator — IE uses different post-processing
                if variant_upper == 'LVIT_IE':
                    from experiments.cipsnet_v2.testing.instance_embed_postprocess import (
                        IEPostProcessConfig, process_batch_ie
                    )
                    
                    ie_config = IEPostProcessConfig(
                        np_threshold=self.config.np_threshold,
                        min_instance_size=self.config.min_instance_size,
                    )
                    
                    ie_results = process_batch_ie(
                        pred_np=outputs['np'],
                        pred_dist=outputs['dist'],
                        pred_type=outputs['type'],
                        pred_embed=outputs.get('embed'),
                        config=ie_config,
                    )
                    
                    # Feed IE post-processed results to evaluator
                    evaluator.add_batch_from_postprocessed(
                        ie_results,
                        gt_instances=gt_instances,
                        gt_types=gt_types,
                        tissues=tissues,
                        indices=indices,
                    )
                else:
                    evaluator.add_batch(
                        pred_np=outputs['np'],
                        pred_hv=outputs['hv'],
                        pred_type=outputs['type'],
                        gt_instances=gt_instances,
                        gt_types=gt_types,
                        tissues=tissues,
                        indices=indices,
                    )
                
                # Free batch memory
                del images, outputs
        
        # Compute and save results
        results_dir = self.config.get_results_dir(fold)
        results = evaluator.compute_and_save(results_dir)
        
        self.test_results[fold] = results
        
        # Print summary
        evaluator.print_summary()
        
        # Aggressive cleanup
        del model, evaluator, test_loader
        clear_memory()
        
        return results
    
    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results from all folds.
        
        Returns:
            Aggregated results
        """
        print(f"\n{'=' * 70}")
        print("AGGREGATING RESULTS")
        print(f"{'=' * 70}")
        
        aggregator = ResultAggregator(
            experiment_dir=self.config.experiment_dir,
            variant=self.config.variant,
        )
        
        for fold, results in self.test_results.items():
            aggregator.add_fold_results(fold, results)
        
        summary = aggregator.aggregate_and_save()
        
        return summary
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete experiment.
        
        This runs:
        1. Training on specified folds (default: all 3)
        2. Testing on specified folds
        3. Result aggregation
        
        Returns:
            Complete experiment results
        """
        print("\n" + "=" * 70)
        print(f"CIPS-Net V2 EXPERIMENT: {self.config.variant}")
        print("=" * 70)
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Early Stopping: {self.config.early_stopping} (patience={self.config.early_stopping_patience})")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Folds: {self.config.folds}")
        
        start_time = time.time()
        
        # ===== PHASE 1: TRAINING =====
        num_folds = len(self.config.folds)
        print("\n" + "=" * 70)
        print(f"PHASE 1: TRAINING ({num_folds} Fold{'s' if num_folds > 1 else ''})")
        print("=" * 70)
        
        for fold in self.config.folds:
            self.train_fold(fold)
        
        # ===== PHASE 2: TESTING =====
        print("\n" + "=" * 70)
        print(f"PHASE 2: TESTING ({num_folds} Fold{'s' if num_folds > 1 else ''})")
        print("=" * 70)
        
        for fold in self.config.folds:
            self.test_fold(fold)
        
        # ===== PHASE 3: AGGREGATION =====
        aggregated = self.aggregate_results()
        
        # ===== FINAL SUMMARY =====
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"  Variant: {self.config.variant}")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Results: {self.config.get_aggregate_dir()}")
        print("=" * 70 + "\n")
        
        return {
            'config': self.config.to_dict(),
            'training_summaries': self.training_summaries,
            'test_results': self.test_results,
            'aggregated': aggregated,
            'total_time': total_time,
        }
    
    def run_training_only(self) -> Dict[str, Any]:
        """Run training only (skip testing)."""
        print("\n" + "=" * 70)
        print(f"TRAINING ONLY: {self.config.variant}")
        print("=" * 70)
        
        for fold in [1, 2, 3]:
            self.train_fold(fold)
        
        return {'training_summaries': self.training_summaries}
    
    def run_testing_only(self) -> Dict[str, Any]:
        """Run testing only (assumes training is complete)."""
        print("\n" + "=" * 70)
        print(f"TESTING ONLY: {self.config.variant}")
        print("=" * 70)
        
        # Use config folds instead of hardcoded [1, 2, 3]
        for fold in self.config.folds:
            self.test_fold(fold)
        
        aggregated = self.aggregate_results()
        
        return {
            'test_results': self.test_results,
            'aggregated': aggregated,
        }


# ==========================================================================
# Convenience Functions
# ==========================================================================

def run_experiment(
    variant: str,
    epochs: int = 50,
    early_stopping_patience: int = 8,
    debug: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run an experiment.
    
    Args:
        variant: Model variant
        epochs: Number of epochs
        early_stopping_patience: Early stopping patience
        debug: Enable debug mode
        **kwargs: Additional config parameters
        
    Returns:
        Experiment results
    """
    config = ExperimentConfig(
        variant=variant,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        debug=debug,
        **kwargs
    )
    
    runner = ExperimentRunner(config)
    return runner.run()


# ==========================================================================
# Testing
# ==========================================================================

if __name__ == "__main__":
    print("Testing ExperimentRunner...")
    print("Note: This is a structural test only, not a full run")
    
    # Test config creation
    config = ExperimentConfig(
        variant="BASELINE",
        epochs=2,
        debug=True,
        debug_batches=2,
    )
    
    print(f"\n[Config Test]")
    print(f"  Variant: {config.variant}")
    print(f"  Experiment dir: {config.experiment_dir}")
    print(f"  Fold 1 checkpoint: {config.get_checkpoint_path(1, 'best')}")
    
    print("\n[Early Stopping Test]")
    es = EarlyStopping(patience=3, min_delta=0.001)
    
    # Simulate improving then stagnating
    losses = [1.0, 0.8, 0.7, 0.65, 0.64, 0.63, 0.63, 0.63, 0.63]
    for epoch, loss in enumerate(losses):
        should_stop = es(loss, epoch)
        print(f"  Epoch {epoch}: loss={loss:.2f}, counter={es.counter}, stop={should_stop}")
        if should_stop:
            break
    
    print("\n✓ ExperimentRunner component tests passed!")
    print("\nTo run a full experiment:")
    print("  python run_experiment.py --variant BASELINE --epochs 50")
