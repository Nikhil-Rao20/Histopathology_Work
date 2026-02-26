"""
Experiment Manager for CIPS-Net V2

Handles:
- Experiment folder structure
- Saving/loading results
- Plotting training curves
- Logging to CSV and text files
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class ExperimentManager:
    """
    Manages experiment outputs: checkpoints, logs, plots.
    
    Folder Structure:
    runs/
    └── {experiment_name}/
        ├── checkpoints/
        │   ├── best.pth
        │   └── last.pth
        ├── logs/
        │   ├── train_losses.csv
        │   ├── val_losses.csv
        │   ├── metrics.json
        │   └── training_log.txt
        ├── plots/
        │   ├── loss_curves.png
        │   ├── lr_curve.png
        │   └── component_losses.png
        └── config.json
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments/cipsnet_v2/runs",
        resume: bool = False,
    ):
        """
        Initialize experiment manager.
        
        Args:
            experiment_name: Name of experiment (e.g., "BASELINE_fold1")
            base_dir: Base directory for all experiments
            resume: Whether to resume existing experiment
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        
        # Sub-directories
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        self.plot_dir = self.experiment_dir / "plots"
        
        # Create directories
        self._create_directories()
        
        # History tracking
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': [],
            # Performance metrics
            'np_dice': [],
            'np_iou': [],
            'np_f1': [],
            'type_f1_macro': [],
            'type_accuracy': [],
        }
        
        # Log file handle
        self.log_file = None
        
        # Resume existing experiment
        if resume and self._experiment_exists():
            self._load_history()
            print(f"[ExperimentManager] Resuming experiment: {experiment_name}")
        else:
            print(f"[ExperimentManager] New experiment: {experiment_name}")
            print(f"[ExperimentManager] Output directory: {self.experiment_dir}")
    
    def _create_directories(self):
        """Create all required directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def _experiment_exists(self) -> bool:
        """Check if experiment directory exists with history."""
        metrics_file = self.log_dir / "metrics.json"
        return metrics_file.exists()
    
    def _load_history(self):
        """Load existing history from files."""
        metrics_file = self.log_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.history = json.load(f)
            print(f"[ExperimentManager] Loaded history: {len(self.history.get('epochs', []))} epochs")
    
    # ==========================================================================
    # Configuration
    # ==========================================================================
    
    def save_config(self, config: Any):
        """Save training configuration to JSON."""
        config_path = self.experiment_dir / "config.json"
        
        if hasattr(config, '__dict__'):
            config_dict = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith('_')}
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {'config': str(config)}
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"[ExperimentManager] Config saved: {config_path}")
    
    # ==========================================================================
    # Logging
    # ==========================================================================
    
    def start_logging(self):
        """Start text logging."""
        log_path = self.log_dir / "training_log.txt"
        self.log_file = open(log_path, 'a')
        self._log(f"\n{'='*70}")
        self._log(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'='*70}")
    
    def stop_logging(self):
        """Stop text logging."""
        if self.log_file:
            self._log(f"{'='*70}")
            self._log(f"Training ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log(f"{'='*70}\n")
            self.log_file.close()
            self.log_file = None
    
    def _log(self, message: str):
        """Write to log file."""
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time: float,
        is_best: bool = False,
    ):
        """
        Log epoch results.
        
        Args:
            epoch: Current epoch (1-indexed)
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict (includes both loss and performance metrics)
            lr: Current learning rate
            epoch_time: Epoch duration in seconds
            is_best: Whether this is the best epoch so far
        """
        # Update history
        self.history['epochs'].append(epoch)
        self.history['train_losses'].append(train_metrics.get('total', 0))
        self.history['val_losses'].append(val_metrics.get('total', 0))
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rates'].append(lr)
        
        # Extract and save performance metrics
        self.history['np_dice'].append(val_metrics.get('np_dice', 0))
        self.history['np_iou'].append(val_metrics.get('np_iou', 0))
        self.history['np_f1'].append(val_metrics.get('np_f1', 0))
        self.history['type_f1_macro'].append(val_metrics.get('type_f1_macro', 0))
        self.history['type_accuracy'].append(val_metrics.get('type_accuracy', 0))
        
        # Log to text file
        self._log(f"\nEpoch {epoch}")
        self._log(f"  Train Loss: {train_metrics.get('total', 0):.4f}")
        self._log(f"  Val Loss: {val_metrics.get('total', 0):.4f}")
        self._log(f"  LR: {lr:.6f}")
        self._log(f"  Time: {epoch_time:.1f}s")
        
        # Log performance metrics
        self._log(f"  Performance Metrics:")
        self._log(f"    NP Dice: {val_metrics.get('np_dice', 0):.4f}")
        self._log(f"    NP IoU: {val_metrics.get('np_iou', 0):.4f}")
        self._log(f"    NP F1: {val_metrics.get('np_f1', 0):.4f}")
        self._log(f"    Type F1 (macro): {val_metrics.get('type_f1_macro', 0):.4f}")
        self._log(f"    Type Accuracy: {val_metrics.get('type_accuracy', 0):.4f}")
        
        if is_best:
            self._log(f"  *** NEW BEST ***")
        
        # Save to CSV
        self._save_epoch_csv(epoch, train_metrics, val_metrics, lr)
        
        # Save history to JSON
        self._save_history()
    
    def _save_epoch_csv(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ):
        """Save epoch metrics to CSV files."""
        # Train losses CSV
        train_csv = self.log_dir / "train_losses.csv"
        self._append_to_csv(train_csv, epoch, train_metrics, lr)
        
        # Val losses CSV
        val_csv = self.log_dir / "val_losses.csv"
        self._append_to_csv(val_csv, epoch, val_metrics, lr)
    
    def _append_to_csv(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float],
        lr: float,
    ):
        """Append metrics to CSV file."""
        file_exists = path.exists()
        
        # Prepare row
        row = {'epoch': epoch, 'lr': lr, **metrics}
        
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
    
    def _save_history(self):
        """Save complete history to JSON."""
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    # ==========================================================================
    # Checkpoints
    # ==========================================================================
    
    def get_checkpoint_path(self, name: str = "best") -> str:
        """Get checkpoint file path."""
        return str(self.checkpoint_dir / f"{name}.pth")
    
    def checkpoint_exists(self, name: str = "best") -> bool:
        """Check if checkpoint exists."""
        return (self.checkpoint_dir / f"{name}.pth").exists()
    
    # ==========================================================================
    # Plotting
    # ==========================================================================
    
    def plot_all(self):
        """Generate all plots."""
        print("\n[ExperimentManager] Generating plots...")
        
        if len(self.history['epochs']) == 0:
            print("  No data to plot")
            return
        
        self._plot_loss_curves()
        self._plot_lr_curve()
        self._plot_component_losses()
        self._plot_performance_metrics()
        
        print(f"  Plots saved to: {self.plot_dir}")
    
    def _plot_loss_curves(self):
        """Plot train/val loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = self.history['epochs']
        train_losses = self.history['train_losses']
        val_losses = self.history['val_losses']
        
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        
        # Mark best epoch
        if val_losses:
            best_epoch_idx = np.argmin(val_losses)
            best_epoch = epochs[best_epoch_idx]
            best_loss = val_losses[best_epoch_idx]
            ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, 
                       label=f'Best (epoch {best_epoch})')
            ax.scatter([best_epoch], [best_loss], color='g', s=100, zorder=5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{self.experiment_name} - Loss Curves', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'loss_curves.png', dpi=150)
        plt.close()
    
    def _plot_lr_curve(self):
        """Plot learning rate curve."""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        epochs = self.history['epochs']
        lrs = self.history['learning_rates']
        
        ax.plot(epochs, lrs, 'purple', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(f'{self.experiment_name} - Learning Rate Schedule', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'lr_curve.png', dpi=150)
        plt.close()
    
    def _plot_component_losses(self):
        """Plot individual loss components."""
        if not self.history['train_metrics']:
            return
        
        # Get loss component names
        sample_metrics = self.history['train_metrics'][0]
        component_names = [k for k in sample_metrics.keys() if k != 'total']
        
        if not component_names:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        epochs = self.history['epochs']
        
        for idx, name in enumerate(component_names[:6]):
            ax = axes[idx]
            
            train_values = [m.get(name, 0) for m in self.history['train_metrics']]
            val_values = [m.get(name, 0) for m in self.history['val_metrics']]
            
            ax.plot(epochs, train_values, 'b-', label='Train', linewidth=1.5)
            ax.plot(epochs, val_values, 'r-', label='Val', linewidth=1.5)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(name)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(component_names), 6):
            axes[idx].axis('off')
        
        plt.suptitle(f'{self.experiment_name} - Component Losses', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'component_losses.png', dpi=150)
        plt.close()
    
    def _plot_performance_metrics(self):
        """Plot performance metrics (NP and Type head metrics)."""
        if not self.history['np_dice']:
            return
        
        epochs = self.history['epochs']
        
        # =====================================================================
        # Plot 1: NP Head Metrics (Dice, IoU, F1)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, self.history['np_dice'], 'b-', label='NP Dice', linewidth=2)
        ax.plot(epochs, self.history['np_iou'], 'g-', label='NP IoU', linewidth=2)
        ax.plot(epochs, self.history['np_f1'], 'r-', label='NP F1', linewidth=2)
        
        # Mark best epoch (based on NP Dice)
        if self.history['np_dice']:
            best_idx = np.argmax(self.history['np_dice'])
            best_epoch = epochs[best_idx]
            best_dice = self.history['np_dice'][best_idx]
            ax.axvline(x=best_epoch, color='purple', linestyle='--', alpha=0.7,
                       label=f'Best Dice (epoch {best_epoch})')
            ax.scatter([best_epoch], [best_dice], color='purple', s=100, zorder=5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.experiment_name} - NP Head Metrics', fontsize=14)
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'np_metrics.png', dpi=150)
        plt.close()
        
        # =====================================================================
        # Plot 2: Type Head Metrics (F1 macro, Accuracy)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, self.history['type_f1_macro'], 'b-', 
                label='Type F1 (macro)', linewidth=2)
        ax.plot(epochs, self.history['type_accuracy'], 'g-', 
                label='Type Accuracy', linewidth=2)
        
        # Mark best epoch (based on Type F1 macro)
        if self.history['type_f1_macro']:
            best_idx = np.argmax(self.history['type_f1_macro'])
            best_epoch = epochs[best_idx]
            best_f1 = self.history['type_f1_macro'][best_idx]
            ax.axvline(x=best_epoch, color='purple', linestyle='--', alpha=0.7,
                       label=f'Best F1 (epoch {best_epoch})')
            ax.scatter([best_epoch], [best_f1], color='purple', s=100, zorder=5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.experiment_name} - Type Head Metrics', fontsize=14)
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'type_metrics.png', dpi=150)
        plt.close()
        
        # =====================================================================
        # Plot 3: Combined Performance Summary
        # =====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # NP Metrics subplot
        ax1 = axes[0]
        ax1.plot(epochs, self.history['np_dice'], 'b-', label='Dice', linewidth=2)
        ax1.plot(epochs, self.history['np_iou'], 'g-', label='IoU', linewidth=2)
        ax1.plot(epochs, self.history['np_f1'], 'r-', label='F1', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('NP Head (Binary Segmentation)', fontsize=12)
        ax1.set_ylim([0, 1])
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Type Metrics subplot
        ax2 = axes[1]
        ax2.plot(epochs, self.history['type_f1_macro'], 'b-', 
                 label='F1 (macro)', linewidth=2)
        ax2.plot(epochs, self.history['type_accuracy'], 'g-', 
                 label='Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Type Head (6-class Classification)', fontsize=12)
        ax2.set_ylim([0, 1])
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.experiment_name} - Performance Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'performance_summary.png', dpi=150)
        plt.close()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    def print_summary(self):
        """Print experiment summary."""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*70}")
        
        if not self.history['epochs']:
            print("  No training data available")
            return
        
        total_epochs = len(self.history['epochs'])
        
        # Best epoch (by val loss)
        val_losses = self.history['val_losses']
        best_idx = np.argmin(val_losses)
        best_epoch = self.history['epochs'][best_idx]
        best_loss = val_losses[best_idx]
        
        print(f"\n[Training]")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Best epoch (by loss): {best_epoch}")
        print(f"  Best val loss: {best_loss:.4f}")
        
        # Final losses
        print(f"\n[Final Losses (Epoch {total_epochs})]")
        print(f"  Train: {self.history['train_losses'][-1]:.4f}")
        print(f"  Val: {self.history['val_losses'][-1]:.4f}")
        
        # Performance metrics at best epoch
        if self.history['np_dice']:
            print(f"\n[Performance at Best Epoch ({best_epoch})]")
            print(f"  NP Dice: {self.history['np_dice'][best_idx]:.4f}")
            print(f"  NP IoU: {self.history['np_iou'][best_idx]:.4f}")
            print(f"  NP F1: {self.history['np_f1'][best_idx]:.4f}")
            print(f"  Type F1 (macro): {self.history['type_f1_macro'][best_idx]:.4f}")
            print(f"  Type Accuracy: {self.history['type_accuracy'][best_idx]:.4f}")
            
            # Best performance metrics (may differ from best loss epoch)
            best_dice_idx = np.argmax(self.history['np_dice'])
            best_type_f1_idx = np.argmax(self.history['type_f1_macro'])
            
            print(f"\n[Best Performance Metrics]")
            print(f"  Best NP Dice: {max(self.history['np_dice']):.4f} "
                  f"(epoch {self.history['epochs'][best_dice_idx]})")
            print(f"  Best Type F1: {max(self.history['type_f1_macro']):.4f} "
                  f"(epoch {self.history['epochs'][best_type_f1_idx]})")
        
        # Component breakdown for best epoch
        if self.history['val_metrics']:
            print(f"\n[Loss Components at Best Epoch]")
            best_metrics = self.history['val_metrics'][best_idx]
            loss_keys = ['np_total', 'hv_total', 'type_total', 'np_bce', 'np_dice',
                         'hv_mse', 'hv_msge', 'type_ldam', 'type_dice']
            for name in loss_keys:
                if name in best_metrics:
                    print(f"  {name}: {best_metrics[name]:.4f}")
        
        print(f"\n[Output Files]")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Plots: {self.plot_dir}")
        print(f"{'='*70}\n")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get metrics from best epoch."""
        if not self.history['val_losses']:
            return {}
        
        best_idx = np.argmin(self.history['val_losses'])
        return self.history['val_metrics'][best_idx]
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get metrics from final epoch."""
        if not self.history['val_metrics']:
            return {}
        return self.history['val_metrics'][-1]


# ==========================================================================
# Testing
# ==========================================================================

def test_experiment_manager():
    """Test experiment manager."""
    import tempfile
    import shutil
    
    print("Testing ExperimentManager...")
    print("=" * 60)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create manager
        print("\n1. Creating ExperimentManager...")
        em = ExperimentManager(
            experiment_name="TEST_fold1",
            base_dir=temp_dir,
        )
        print(f"   ✓ Created at: {em.experiment_dir}")
        
        # Save config
        print("\n2. Saving config...")
        config = {'variant': 'TEST', 'epochs': 10, 'lr': 0.001}
        em.save_config(config)
        print("   ✓ Config saved")
        
        # Start logging
        print("\n3. Testing logging...")
        em.start_logging()
        
        # Log some epochs
        for epoch in range(1, 6):
            train_metrics = {
                'total': 2.0 - epoch * 0.2,
                'np_total': 0.5,
                'hv_total': 0.8,
                'type_total': 0.7 - epoch * 0.05,
            }
            val_metrics = {
                'total': 2.1 - epoch * 0.15,
                'np_total': 0.55,
                'hv_total': 0.85,
                'type_total': 0.7 - epoch * 0.04,
                # Performance metrics
                'np_dice': 0.70 + epoch * 0.03,
                'np_iou': 0.55 + epoch * 0.03,
                'np_f1': 0.72 + epoch * 0.02,
                'type_f1_macro': 0.40 + epoch * 0.05,
                'type_accuracy': 0.60 + epoch * 0.04,
            }
            
            is_best = val_metrics['total'] < min(em.history['val_losses'] or [float('inf')])
            
            em.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=0.001 * (0.9 ** epoch),
                epoch_time=10.5,
                is_best=is_best,
            )
        
        em.stop_logging()
        print(f"   ✓ Logged {len(em.history['epochs'])} epochs")
        
        # Generate plots
        print("\n4. Generating plots...")
        em.plot_all()
        print("   ✓ Plots generated")
        
        # Print summary
        print("\n5. Summary:")
        em.print_summary()
        
        # Check files exist
        print("6. Checking output files...")
        assert (em.log_dir / "train_losses.csv").exists(), "train_losses.csv missing"
        assert (em.log_dir / "val_losses.csv").exists(), "val_losses.csv missing"
        assert (em.log_dir / "metrics.json").exists(), "metrics.json missing"
        assert (em.plot_dir / "loss_curves.png").exists(), "loss_curves.png missing"
        assert (em.plot_dir / "lr_curve.png").exists(), "lr_curve.png missing"
        assert (em.plot_dir / "np_metrics.png").exists(), "np_metrics.png missing"
        assert (em.plot_dir / "type_metrics.png").exists(), "type_metrics.png missing"
        assert (em.plot_dir / "performance_summary.png").exists(), "performance_summary.png missing"
        assert (em.experiment_dir / "config.json").exists(), "config.json missing"
        print("   ✓ All files created")
        
        print("\n" + "=" * 60)
        print("All ExperimentManager tests passed!")
        print("=" * 60)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_experiment_manager()
