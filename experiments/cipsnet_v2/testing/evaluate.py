"""
Evaluation Script for CIPS-Net V2
==================================

Run evaluation on the test fold using the best trained model.

Usage:
------
# Evaluate a trained model
python evaluate.py --checkpoint runs/BASELINE_fold1/checkpoints/best.pth --fold 1

# Evaluate with custom output directory
python evaluate.py --checkpoint runs/BASELINE_fold1/checkpoints/best.pth --fold 1 --output results/BASELINE_fold1

# Full evaluation command
python evaluate.py \
    --checkpoint experiments/cipsnet_v2/runs/BASELINE_fold1/checkpoints/best.pth \
    --fold 1 \
    --variant BASELINE \
    --batch_size 4 \
    --output experiments/cipsnet_v2/results/BASELINE_fold1
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.cipsnet_v2.training.config import TrainingConfig
from experiments.cipsnet_v2.training.utils import load_checkpoint
from experiments.cipsnet_v2.models import create_model
from experiments.cipsnet_v2.datasets.pannuke import (
    create_fold_dataloaders,
    PANNUKE_CLASSES,
    PANNUKE_TISSUES,
)
from experiments.cipsnet_v2.testing.evaluator import PanNukeEvaluator, EvaluationConfig
from experiments.cipsnet_v2.testing.post_processing import PostProcessConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate CIPS-Net V2 on PanNuke test fold'
    )
    
    # Required
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (best.pth)'
    )
    parser.add_argument(
        '--fold', type=int, required=True, choices=[1, 2, 3],
        help='Test fold (1, 2, or 3)'
    )
    
    # Model
    parser.add_argument(
        '--variant', type=str, default='BASELINE',
        choices=['BASELINE', 'WITH_TEXT', 'WITH_CGR', 'WITH_TEXT_CONDITIONED_TYPE', 'FULL'],
        help='Model variant'
    )
    
    # Data
    parser.add_argument(
        '--data_root', type=str,
        default='/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work/Histopathology_Datasets_Official/PanNuke',
        help='Path to PanNuke dataset'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    # Post-processing (tuned defaults from post_processing_tuning.ipynb)
    parser.add_argument(
        '--np_threshold', type=float, default=0.525,
        help='Threshold for NP prediction (tuned default: 0.525)'
    )
    parser.add_argument(
        '--min_instance_size', type=int, default=70,
        help='Minimum instance size in pixels (tuned default: 70)'
    )
    parser.add_argument(
        '--marker_erosion_size', type=int, default=3,
        help='Marker erosion size for watershed (tuned default: 3)'
    )
    
    # Output
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results (default: auto-generated)'
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save predictions as .mat files'
    )
    
    # Hardware
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("CIPS-Net V2 Evaluation")
    print("=" * 70)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    
    # === Load Checkpoint ===
    print(f"\n[Loading checkpoint] {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        print(f"  Loaded config from checkpoint")
        print(f"  Variant: {saved_config.get('variant', args.variant)}")
        print(f"  Fold: {saved_config.get('fold', args.fold)}")
    else:
        saved_config = {}
    
    # === Create Model ===
    print(f"\n[Creating model] Variant: {args.variant}")
    
    model = create_model(variant=args.variant)
    model = model.to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model weights")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"  Loaded model weights")
    else:
        raise KeyError("No model weights found in checkpoint")
    
    model.eval()
    
    # === Create Test DataLoader ===
    print(f"\n[Loading data] Test fold: {args.fold}")
    
    _, _, test_loader = create_fold_dataloaders(
        data_root=args.data_root,
        test_fold=args.fold,
        mode='baseline',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batches: {len(test_loader)}")
    
    # === Setup Evaluator ===
    eval_config = EvaluationConfig(
        np_threshold=args.np_threshold,
        min_instance_size=args.min_instance_size,
        marker_erosion_size=args.marker_erosion_size,
        save_predictions=args.save_predictions,
    )
    
    evaluator = PanNukeEvaluator(config=eval_config, device=str(device))
    
    # === Run Evaluation ===
    print(f"\n[Evaluating]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Debug: print batch keys on first iteration
            if batch_idx == 0:
                print(f"\n[DEBUG] Batch keys: {batch.keys()}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: type={type(v)}")
            
            # Get inputs - handle different possible key names
            if 'image' in batch:
                images = batch['image'].to(device)
            elif 'images' in batch:
                images = batch['images'].to(device)
            else:
                # Try first tensor key that looks like image
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[1] == 3:
                        images = v.to(device)
                        print(f"[DEBUG] Using '{k}' as image tensor")
                        break
                else:
                    raise KeyError(f"Could not find image tensor in batch. Keys: {batch.keys()}")
            
            # Get ground truth (keys from debug: instance_maps, type_maps, tissues, indices)
            true_inst = batch['instance_maps'].numpy()  # [B, H, W]
            true_type = batch['type_maps'].numpy()  # [B, H, W]
            tissues = batch['tissues']  # List of tissue names
            indices = batch['indices']  # List of image indices
            
            # Forward pass
            outputs = model(images)
            
            pred_np = outputs['np']  # [B, 2, H, W]
            pred_hv = outputs['hv']  # [B, 2, H, W]
            pred_type = outputs['type']  # [B, C, H, W]
            
            # Convert to numpy
            pred_np = pred_np.cpu().numpy()
            pred_hv = pred_hv.cpu().numpy()
            pred_type = pred_type.cpu().numpy()
            
            # Evaluate each sample
            batch_size = images.shape[0]
            for i in range(batch_size):
                image_id = f"fold{args.fold}_{indices[i]}"
                
                evaluator.evaluate_single(
                    pred_np=pred_np[i],
                    pred_hv=pred_hv[i],
                    pred_type=pred_type[i],
                    true_inst=true_inst[i],
                    true_type=true_type[i],
                    tissue=tissues[i],
                    image_id=image_id,
                )
    
    # === Save Results ===
    if args.output:
        output_dir = args.output
    else:
        # Auto-generate output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"experiments/cipsnet_v2/results/{args.variant}_fold{args.fold}_{timestamp}"
    
    print(f"\n[Saving results] {output_dir}")
    
    # Save evaluation config
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'eval_config.json'), 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'fold': args.fold,
            'variant': args.variant,
            'np_threshold': args.np_threshold,
            'min_instance_size': args.min_instance_size,
            'marker_erosion_size': args.marker_erosion_size,
            'device': str(device),
            'n_test_samples': len(test_loader.dataset),
        }, f, indent=2)
    
    # Save all results
    summary = evaluator.save_results(output_dir)
    
    # Print summary
    evaluator.print_summary()
    
    # === Print Final Summary for Paper ===
    print("\n" + "=" * 70)
    print("RESULTS FOR PAPER")
    print("=" * 70)
    
    overall = summary['overall']
    print(f"\n📊 Instance Segmentation:")
    print(f"   Dice: {overall.get('dice_mean', 0)*100:.2f}")
    print(f"   AJI:  {overall.get('aji_mean', 0)*100:.2f}")
    print(f"   AJI+: {overall.get('aji_plus_mean', 0)*100:.2f}")
    print(f"   bPQ:  {overall.get('pq_mean', 0)*100:.2f}")
    
    print(f"\n📊 Multi-class (mPQ): {overall.get('mPQ_mean', 0)*100:.2f}")
    
    print(f"\n📊 Per-class PQ:")
    for class_name, metrics in summary['class_wise'].items():
        print(f"   {class_name:15s}: {metrics.get('pq_mean', 0)*100:.2f}")
    
    detection = summary['detection']
    print(f"\n📊 Detection:")
    print(f"   F1:        {detection.get('detection_f1_mean', 0)*100:.2f}")
    print(f"   Precision: {detection.get('detection_precision_mean', 0)*100:.2f}")
    print(f"   Recall:    {detection.get('detection_recall_mean', 0)*100:.2f}")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70 + "\n")
    
    return summary


if __name__ == "__main__":
    main()
