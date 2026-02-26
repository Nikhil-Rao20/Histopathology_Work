#!/usr/bin/env python3
"""
Main Training Script for CIPS-Net V2

Usage:
    # Debug mode (2 batches, 2 epochs)
    python train.py --variant BASELINE --fold 1 --debug

    # Full training
    python train.py --variant FULL --fold 1 --epochs 50

    # Resume training
    python train.py --variant FULL --fold 1 --resume checkpoints/FULL_fold1_last.pth

    # Train all variants on all folds
    python train.py --train_all

Examples:
    # Quick debug run
    python train.py --variant BASELINE --fold 1 --debug --debug_batches 2

    # Full training with custom settings
    python train.py --variant FULL --fold 2 --epochs 50 --batch_size 8 --lr 1e-4
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CIPS-Net V2 for nucleus instance segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model
    parser.add_argument(
        "--variant",
        type=str,
        default="BASELINE",
        choices=["BASELINE", "WITH_TEXT", "WITH_CGR", "WITH_TEXT_CONDITIONED_TYPE", "FULL"],
        help="Model variant to train"
    )
    
    # Data
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Validation fold (1, 2, or 3)"
    )
    parser.add_argument(
        "--pannuke_path",
        type=str,
        default="/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Histopathology_Datasets_Official/PanNuke",
        help="Path to PanNuke dataset"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Loss
    parser.add_argument("--np_weight", type=float, default=1.0, help="NP loss weight")
    parser.add_argument("--hv_weight", type=float, default=2.0, help="HV loss weight")
    parser.add_argument("--type_weight", type=float, default=1.0, help="Type loss weight")
    parser.add_argument("--drw_start", type=int, default=30, help="DRW start epoch")
    
    # Checkpoints
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="experiments/cipsnet_v2/runs",
        help="Output directory for experiments"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug_batches", type=int, default=2, help="Batches in debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Multi-experiment
    parser.add_argument(
        "--train_all",
        action="store_true",
        help="Train all variants on all folds"
    )
    
    # AMP
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    
    return parser.parse_args()


def train_single(args):
    """Train a single model variant on a single fold."""
    from experiments.cipsnet_v2.training.config import TrainingConfig
    from experiments.cipsnet_v2.training.trainer import create_trainer
    
    # Create configuration
    config = TrainingConfig(
        # Model
        variant=args.variant,
        
        # Data
        pannuke_path=args.pannuke_path,
        fold=args.fold,
        num_workers=args.num_workers,
        
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        
        # Loss
        np_weight=args.np_weight,
        hv_weight=args.hv_weight,
        type_weight=args.type_weight,
        drw_start_epoch=args.drw_start,
        
        # Output
        runs_dir=args.runs_dir,
        resume_from=args.resume,
        
        # Debug
        debug=args.debug,
        debug_batches=args.debug_batches,
        seed=args.seed,
        
        # AMP
        use_amp=not args.no_amp,
    )
    
    # Print configuration
    print("\n" + "=" * 70)
    print("CIPS-Net V2 Training")
    print("=" * 70)
    print(f"Variant: {config.variant}")
    print(f"Fold: {config.fold}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Debug mode: {config.debug}")
    print("=" * 70)
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Train
    history = trainer.train()
    
    return history


def train_all(args):
    """Train all variants on all folds."""
    variants = ["BASELINE", "WITH_TEXT", "WITH_CGR", "WITH_TEXT_CONDITIONED_TYPE", "FULL"]
    folds = [1, 2, 3]
    
    total_experiments = len(variants) * len(folds)
    current = 0
    
    print("\n" + "=" * 70)
    print(f"TRAINING ALL EXPERIMENTS ({total_experiments} total)")
    print("=" * 70)
    
    results = {}
    
    for variant in variants:
        for fold in folds:
            current += 1
            experiment_name = f"{variant}_fold{fold}"
            
            print(f"\n{'#' * 70}")
            print(f"# Experiment {current}/{total_experiments}: {experiment_name}")
            print(f"{'#' * 70}")
            
            # Update args for this experiment
            args.variant = variant
            args.fold = fold
            
            try:
                history = train_single(args)
                results[experiment_name] = {
                    'status': 'success',
                    'best_val_loss': min(history['val_loss']),
                }
                print(f"\n✓ {experiment_name} completed successfully")
            except Exception as e:
                results[experiment_name] = {
                    'status': 'failed',
                    'error': str(e),
                }
                print(f"\n✗ {experiment_name} failed: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    for name, result in results.items():
        if result['status'] == 'success':
            print(f"  ✓ {name}: val_loss={result['best_val_loss']:.4f}")
        else:
            print(f"  ✗ {name}: {result['error']}")
    
    print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"\n[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    else:
        print("\n[WARNING] CUDA not available, using CPU")
    
    if args.train_all:
        # Train all experiments
        train_all(args)
    else:
        # Train single experiment
        train_single(args)


if __name__ == "__main__":
    main()
