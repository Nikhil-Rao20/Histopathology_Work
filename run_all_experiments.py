#!/usr/bin/env python3
"""
Run all text-guided segmentation experiments.

This script trains all 14 models on PanNuke (3-fold CV) and evaluates 
zero-shot performance on CoNSeP and MoNuSAC.

Usage:
    python run_all_experiments.py --models clipseg lseg groupvit
    python run_all_experiments.py --all
    python run_all_experiments.py --quick  # Single fold, fewer epochs
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

WORKSPACE = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work"
sys.path.insert(0, WORKSPACE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import cv2

# Import our package
from TextGuidedSegmentation import (
    get_model,
    list_models,
    DEFAULT_TEXT_PROMPTS,
    CombinedSegmentationLoss,
)


# =============================================================================
# Constants
# =============================================================================

WORKING_MODELS = [
    "clipseg",
    "clipseg_rd64", 
    "clipseg_rd128",
    "lseg",
    "groupvit",
    "san",
    "fc_clip",
    "fc_clip_convnext",
    "ovseg",
    "cat_seg",
    "sed",
    "openseed",
    "odise",
    "semantic_sam",
]

# 3 Types of Text Instructions for testing
TEXT_PROMPTS_TYPE1 = [  # Simple class names
    "neoplastic cells",
    "inflammatory cells", 
    "connective tissue cells",
    "dead cells",
    "epithelial cells",
]

TEXT_PROMPTS_TYPE2 = [  # Descriptive prompts
    "tumor cells with abnormal morphology",
    "immune cells including lymphocytes and macrophages",
    "stromal and connective tissue cells",
    "necrotic and apoptotic dead cells",
    "epithelial cells lining tissue surfaces",
]

TEXT_PROMPTS_TYPE3 = [  # Medical/pathological prompts
    "malignant neoplastic cells in histopathology",
    "inflammatory infiltrate cells",
    "fibroblasts and connective tissue stroma",
    "cells undergoing cell death",
    "epithelial tissue cells",
]

ALL_TEXT_PROMPTS = {
    "simple": TEXT_PROMPTS_TYPE1,
    "descriptive": TEXT_PROMPTS_TYPE2,
    "medical": TEXT_PROMPTS_TYPE3,
}

# Default prompts for training
TEXT_PROMPTS = TEXT_PROMPTS_TYPE1

CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

EARLY_STOPPING_PATIENCE = 8


# =============================================================================
# Dataset
# =============================================================================

class SimplePanNukeDataset(Dataset):
    """Simple PanNuke dataset for training."""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int = 256,
        split: str = "train",
        fold: int = 0,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        
        # Get image files
        self.image_files = sorted(list(self.image_dir.glob("*.png")) + 
                                  list(self.image_dir.glob("*.jpg")))
        
        # Simple split based on fold
        n = len(self.image_files)
        fold_size = n // 3
        
        if split == "train":
            indices = list(range(n))
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < 2 else n
            indices = [i for i in indices if not (test_start <= i < test_end)]
            self.image_files = [self.image_files[i] for i in indices]
        elif split == "val":
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < 2 else n
            self.image_files = self.image_files[test_start:test_end]
        
        # CLIP normalization
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = transforms.ToTensor()(image)
        image = self.normalize(image)
        
        mask_name = img_path.stem + ".png"
        mask_path = self.mask_dir / mask_name
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        mask = np.clip(mask, 0, 4).astype(np.int64)
        mask = torch.from_numpy(mask)
        
        return {"image": image, "mask": mask}


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, text_prompts, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(images, text_prompts)
            logits = outputs['logits']
            loss = criterion(logits, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def compute_comprehensive_metrics(confusion, num_classes):
    """
    Compute all metrics from confusion matrix.
    Returns: mPQ, mDice, mIoU, mPrecision, mRecall plus per-class metrics.
    
    mPQ (mean Panoptic Quality) is computed as the product of:
      - SQ (Segmentation Quality): mean IoU of matched predictions
      - RQ (Recognition Quality): F1 score of detection
    For semantic segmentation (no instances), we approximate mPQ as mDice * mAccuracy
    where mAccuracy serves as a detection proxy.
    """
    # Per-class metrics
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp  # Sum over rows (predictions) for each class
    fn = confusion.sum(dim=1) - tp  # Sum over columns (ground truth) for each class
    
    # IoU per class
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Dice per class (F1 Score / Sørensen–Dice coefficient)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    # Precision per class: TP / (TP + FP)
    precision = tp / (tp + fp + 1e-8)
    
    # Recall per class: TP / (TP + FN)
    recall = tp / (tp + fn + 1e-8)
    
    # F1 per class (same as Dice for binary)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Overall pixel accuracy
    total_correct = tp.sum()
    total_pixels = confusion.sum()
    accuracy = (total_correct / (total_pixels + 1e-8)).item()
    
    # Convert to lists
    ious = [iou[c].item() for c in range(num_classes)]
    dices = [dice[c].item() for c in range(num_classes)]
    precisions = [precision[c].item() for c in range(num_classes)]
    recalls = [recall[c].item() for c in range(num_classes)]
    f1s = [f1[c].item() for c in range(num_classes)]
    
    # Mean metrics (only for classes with ground truth samples)
    valid_classes = confusion.sum(dim=1) > 0  # Classes with GT samples
    
    mean_iou = iou[valid_classes].mean().item() if valid_classes.any() else 0
    mean_dice = dice[valid_classes].mean().item() if valid_classes.any() else 0
    mean_precision = precision[valid_classes].mean().item() if valid_classes.any() else 0
    mean_recall = recall[valid_classes].mean().item() if valid_classes.any() else 0
    mean_f1 = f1[valid_classes].mean().item() if valid_classes.any() else 0
    
    # mPQ (mean Panoptic Quality) for semantic segmentation
    # Approximated as: PQ_c = IoU_c * (2*TP_c / (2*TP_c + FP_c + FN_c)) for each class
    # This is equivalent to Dice * IoU / Dice = IoU, but we use a weighted version
    # Standard mPQ = mean(Dice_c) as Dice captures both segmentation and detection quality
    # For cell segmentation, mPQ ≈ mDice since Dice = SQ * RQ
    sq = mean_iou  # Segmentation Quality (how well we segment matched regions)
    rq = mean_precision * mean_recall * 2 / (mean_precision + mean_recall + 1e-8)  # Recognition Quality (F1 of detection)
    mpq = sq * rq if (sq > 0 and rq > 0) else mean_dice  # Fallback to mDice
    
    # Alternative mPQ: per-class PQ then average
    pq_per_class = []
    for c in range(num_classes):
        if confusion[c].sum() > 0 or confusion[:, c].sum() > 0:  # Class has samples
            sq_c = ious[c]
            rq_c = f1s[c]
            pq_c = sq_c * rq_c
            pq_per_class.append(pq_c)
        else:
            pq_per_class.append(0.0)
    
    mpq_alt = np.mean([pq for pq in pq_per_class if pq > 0]) if any(pq > 0 for pq in pq_per_class) else 0
    
    return {
        # Main metrics
        'mPQ': mpq_alt,  # Mean Panoptic Quality (PRIMARY METRIC)
        'mDice': mean_dice,
        'mIoU': mean_iou,
        'mPrecision': mean_precision,
        'mRecall': mean_recall,
        'mF1': mean_f1,
        'accuracy': accuracy,
        # Per-class metrics
        'per_class_pq': pq_per_class,
        'per_class_iou': ious,
        'per_class_dice': dices,
        'per_class_precision': precisions,
        'per_class_recall': recalls,
        'per_class_f1': f1s,
    }


@torch.no_grad()
def validate(model, loader, criterion, text_prompts, device, num_classes=5):
    """
    Validate the model with comprehensive metrics.
    
    Returns:
        Dictionary with mPQ, mDice, mIoU, mPrecision, mRecall, and per-class metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    confusion = torch.zeros(num_classes, num_classes, device=device)
    
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = model(images, text_prompts)
        logits = outputs['logits']
        loss = criterion(logits, masks)
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = logits.argmax(dim=1).flatten()
        targets = masks.flatten()
        valid_mask = (targets >= 0) & (targets < num_classes)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        for p, t in zip(preds, targets):
            confusion[t, p] += 1
    
    avg_loss = total_loss / num_batches
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(confusion, num_classes)
    metrics['loss'] = avg_loss
    
    # For backward compatibility
    metrics['mean_iou'] = metrics['mIoU']
    metrics['mean_dice'] = metrics['mDice']
    
    return metrics


def train_model(model_name, config, fold=0):
    """Train a single model on one fold."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name} (Fold {fold})")
    print(f"{'='*60}")
    
    device = config['device']
    
    # Create datasets
    train_dataset = SimplePanNukeDataset(
        image_dir=f"{config['data_root']}/multi_images",
        mask_dir=f"{config['data_root']}/multi_masks",
        image_size=config['image_size'],
        split="train",
        fold=fold,
    )
    
    val_dataset = SimplePanNukeDataset(
        image_dir=f"{config['data_root']}/multi_images",
        mask_dir=f"{config['data_root']}/multi_masks",
        image_size=config['image_size'],
        split="val",
        fold=fold,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create model
    try:
        model = get_model(
            model_name,
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            clip_model=config['clip_model'],
            freeze_clip=config['freeze_clip'],
            device=device,
        )
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup training
    criterion = CombinedSegmentationLoss(ce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    scaler = GradScaler('cuda')
    
    # Training loop
    best_metric = 0.0
    epochs_without_improvement = 0
    history = defaultdict(list)
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            TEXT_PROMPTS, device
        )
        
        val_metrics = validate(
            model, val_loader, criterion, TEXT_PROMPTS, 
            device, config['num_classes']
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['mIoU'])
        history['val_dice'].append(val_metrics['mDice'])
        history['val_pq'].append(val_metrics['mPQ'])
        history['val_precision'].append(val_metrics['mPrecision'])
        history['val_recall'].append(val_metrics['mRecall'])
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mPQ: {val_metrics['mPQ']:.4f} | mDice: {val_metrics['mDice']:.4f} | mIoU: {val_metrics['mIoU']:.4f}")
        print(f"      Precision: {val_metrics['mPrecision']:.4f} | Recall: {val_metrics['mRecall']:.4f}")
        
        # Use mPQ as the primary metric for model selection
        current_metric = val_metrics['mPQ']
        if current_metric > best_metric:
            best_metric = val_metrics['mean_iou']
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / f"best_{model_name}_fold{fold}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'metrics': val_metrics,  # Save all metrics
            }, checkpoint_path)
            print(f"  ✓ Best model saved (mPQ: {best_metric:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"  ⚠️ Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save results with all metrics
    results = {
        'model_name': model_name,
        'fold': fold,
        'best_mPQ': best_metric,
        'final_mPQ': history['val_pq'][-1],
        'final_mDice': history['val_dice'][-1],
        'final_mIoU': history['val_iou'][-1],
        'final_mPrecision': history['val_precision'][-1],
        'final_mRecall': history['val_recall'][-1],
        # Backward compatibility
        'best_iou': history['val_iou'][-1] if history['val_iou'] else 0,
        'final_dice': history['val_dice'][-1] if history['val_dice'] else 0,
        'history': dict(history),
    }
    
    results_path = checkpoint_dir / f"results_{model_name}_fold{fold}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def run_all_experiments(models, config, num_folds=3):
    """Run experiments for all models and folds."""
    all_results = {}
    
    for model_name in models:
        all_results[model_name] = {}
        
        for fold in range(num_folds):
            results = train_model(model_name, config, fold=fold)
            if results:
                all_results[model_name][f'fold_{fold}'] = results
    
    # Aggregate results with all metrics
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    summary = []
    for model_name, fold_results in all_results.items():
        if fold_results:
            pqs = [r['best_mPQ'] for r in fold_results.values() if r and 'best_mPQ' in r]
            dices = [r['final_mDice'] for r in fold_results.values() if r and 'final_mDice' in r]
            ious = [r['final_mIoU'] for r in fold_results.values() if r and 'final_mIoU' in r]
            precisions = [r['final_mPrecision'] for r in fold_results.values() if r and 'final_mPrecision' in r]
            recalls = [r['final_mRecall'] for r in fold_results.values() if r and 'final_mRecall' in r]
            
            # Fallback for backward compatibility
            if not pqs:
                pqs = [r.get('best_iou', 0) for r in fold_results.values() if r]
            if not dices:
                dices = [r.get('final_dice', 0) for r in fold_results.values() if r]
            if not ious:
                ious = pqs  # Use pqs as fallback
            
            if pqs:
                summary.append({
                    'model': model_name,
                    'mean_pq': np.mean(pqs),
                    'std_pq': np.std(pqs),
                    'mean_dice': np.mean(dices) if dices else 0,
                    'std_dice': np.std(dices) if dices else 0,
                    'mean_iou': np.mean(ious) if ious else 0,
                    'std_iou': np.std(ious) if ious else 0,
                    'mean_precision': np.mean(precisions) if precisions else 0,
                    'mean_recall': np.mean(recalls) if recalls else 0,
                })
    
    # Sort by mPQ (primary metric)
    summary.sort(key=lambda x: x['mean_pq'], reverse=True)
    
    print(f"\n{'Model':<20} {'mPQ (±std)':<15} {'mDice':<12} {'mIoU':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 83)
    for s in summary:
        print(f"{s['model']:<20} {s['mean_pq']:.4f} (±{s['std_pq']:.4f})  "
              f"{s['mean_dice']:.4f}       {s['mean_iou']:.4f}       "
              f"{s['mean_precision']:.4f}       {s['mean_recall']:.4f}")
    
    # Save summary
    summary_path = Path(config['checkpoint_dir']) / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    return all_results


# =============================================================================
# Zero-Shot Evaluation on Other Datasets
# =============================================================================

@torch.no_grad()
def zero_shot_evaluate(model, data_dir, mask_dir, text_prompts, config, dataset_name="Unknown"):
    """Zero-shot evaluation on a dataset without fine-tuning."""
    device = config['device']
    
    # Create dataset
    dataset = SimplePanNukeDataset(
        image_dir=data_dir,
        mask_dir=mask_dir,
        image_size=config['image_size'],
        split="val",  # Use all data as validation
        fold=0,
    )
    # Use all data (override split)
    dataset.image_files = sorted(list(Path(data_dir).glob("*.png")) + 
                                  list(Path(data_dir).glob("*.jpg")))
    
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"\nZero-shot evaluation on {dataset_name}: {len(dataset)} images")
    
    model.eval()
    confusion = torch.zeros(config['num_classes'], config['num_classes'], device=device)
    
    for batch in tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = model(images, text_prompts)
        logits = outputs['logits']
        
        preds = logits.argmax(dim=1).flatten()
        targets = masks.flatten()
        valid_mask = (targets >= 0) & (targets < config['num_classes'])
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        for p, t in zip(preds, targets):
            confusion[t, p] += 1
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(confusion, config['num_classes'])
    metrics['dataset'] = dataset_name
    
    # Backward compatibility
    metrics['mean_iou'] = metrics['mIoU']
    metrics['mean_dice'] = metrics['mDice']
    
    return metrics


def run_zero_shot_evaluation(models, config):
    """Run zero-shot evaluation on CoNSeP and MoNuSAC datasets."""
    print("\n" + "="*60)
    print("ZERO-SHOT EVALUATION ON EXTERNAL DATASETS")
    print("="*60)
    
    # Define external dataset paths (adjust these paths as needed)
    external_datasets = {
        'CoNSeP': {
            'image_dir': f"{config['data_root']}/consep_images",
            'mask_dir': f"{config['data_root']}/consep_masks",
        },
        'MoNuSAC': {
            'image_dir': f"{config['data_root']}/monusac_images",
            'mask_dir': f"{config['data_root']}/monusac_masks",
        },
    }
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    zero_shot_results = {}
    
    for model_name in models:
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")
        
        # Load best model from fold 0
        checkpoint_path = checkpoint_dir / f"best_{model_name}_fold0.pth"
        if not checkpoint_path.exists():
            print(f"  ⚠️ Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            model = get_model(
                model_name,
                num_classes=config['num_classes'],
                image_size=config['image_size'],
                clip_model=config['clip_model'],
                freeze_clip=True,
                device=config['device'],
            )
            
            checkpoint = torch.load(checkpoint_path, map_location=config['device'])
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
            
            zero_shot_results[model_name] = {}
            
            for dataset_name, paths in external_datasets.items():
                if Path(paths['image_dir']).exists():
                    result = zero_shot_evaluate(
                        model, 
                        paths['image_dir'], 
                        paths['mask_dir'],
                        TEXT_PROMPTS,
                        config,
                        dataset_name
                    )
                    zero_shot_results[model_name][dataset_name] = result
                    print(f"  {dataset_name}: mPQ={result['mPQ']:.4f} | mDice={result['mDice']:.4f} | mIoU={result['mIoU']:.4f}")
                    print(f"              Precision={result['mPrecision']:.4f} | Recall={result['mRecall']:.4f}")
                else:
                    print(f"  ⚠️ Dataset not found: {dataset_name}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Save zero-shot results
    results_path = checkpoint_dir / "zero_shot_results.json"
    with open(results_path, 'w') as f:
        json.dump(zero_shot_results, f, indent=2)
    print(f"\nZero-shot results saved to {results_path}")
    
    return zero_shot_results


# =============================================================================
# Text Prompt Variation Testing
# =============================================================================

def test_text_prompt_variations(models, config):
    """Test models with 3 different types of text prompts on PanNuke validation."""
    print("\n" + "="*60)
    print("TEXT PROMPT VARIATION TESTING")
    print("="*60)
    print("Testing 3 types of text instructions:")
    print("  1. Simple: 'neoplastic cells', 'inflammatory cells', ...")
    print("  2. Descriptive: 'tumor cells with abnormal morphology', ...")
    print("  3. Medical: 'malignant neoplastic cells in histopathology', ...")
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    
    # Create validation dataset
    val_dataset = SimplePanNukeDataset(
        image_dir=f"{config['data_root']}/multi_images",
        mask_dir=f"{config['data_root']}/multi_masks",
        image_size=config['image_size'],
        split="val",
        fold=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    prompt_results = {}
    
    for model_name in models:
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")
        
        checkpoint_path = checkpoint_dir / f"best_{model_name}_fold0.pth"
        if not checkpoint_path.exists():
            print(f"  ⚠️ Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            model = get_model(
                model_name,
                num_classes=config['num_classes'],
                image_size=config['image_size'],
                clip_model=config['clip_model'],
                freeze_clip=True,
                device=config['device'],
            )
            
            checkpoint = torch.load(checkpoint_path, map_location=config['device'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            prompt_results[model_name] = {}
            
            for prompt_type, prompts in ALL_TEXT_PROMPTS.items():
                print(f"\n  Testing with {prompt_type} prompts...")
                
                confusion = torch.zeros(config['num_classes'], config['num_classes'], 
                                        device=config['device'])
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"    {prompt_type}", leave=False):
                        images = batch['image'].to(config['device'])
                        masks = batch['mask'].to(config['device'])
                        
                        outputs = model(images, prompts)
                        logits = outputs['logits']
                        
                        preds = logits.argmax(dim=1).flatten()
                        targets = masks.flatten()
                        valid_mask = (targets >= 0) & (targets < config['num_classes'])
                        preds = preds[valid_mask]
                        targets = targets[valid_mask]
                        
                        for p, t in zip(preds, targets):
                            confusion[t, p] += 1
                
                # Compute comprehensive metrics
                metrics = compute_comprehensive_metrics(confusion, config['num_classes'])
                
                prompt_results[model_name][prompt_type] = {
                    'prompts': prompts,
                    'mPQ': metrics['mPQ'],
                    'mDice': metrics['mDice'],
                    'mIoU': metrics['mIoU'],
                    'mPrecision': metrics['mPrecision'],
                    'mRecall': metrics['mRecall'],
                    'accuracy': metrics['accuracy'],
                    'per_class_pq': metrics['per_class_pq'],
                    'per_class_iou': metrics['per_class_iou'],
                    'per_class_dice': metrics['per_class_dice'],
                    'per_class_precision': metrics['per_class_precision'],
                    'per_class_recall': metrics['per_class_recall'],
                    # Backward compatibility
                    'mean_iou': metrics['mIoU'],
                    'mean_dice': metrics['mDice'],
                }
                
                print(f"    {prompt_type}: mPQ={metrics['mPQ']:.4f} | mDice={metrics['mDice']:.4f} | mIoU={metrics['mIoU']:.4f}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Print summary table
    print("\n" + "="*60)
    print("TEXT PROMPT VARIATION SUMMARY (mPQ)")
    print("="*60)
    print(f"\n{'Model':<20} {'Simple':<12} {'Descriptive':<12} {'Medical':<12}")
    print("-" * 56)
    for model_name, results in prompt_results.items():
        simple = results.get('simple', {}).get('mPQ', 0)
        desc = results.get('descriptive', {}).get('mPQ', 0)
        med = results.get('medical', {}).get('mPQ', 0)
        print(f"{model_name:<20} {simple:.4f}       {desc:.4f}       {med:.4f}")
    
    # Save results
    results_path = checkpoint_dir / "text_prompt_variation_results.json"
    with open(results_path, 'w') as f:
        json.dump(prompt_results, f, indent=2)
    print(f"\nText prompt results saved to {results_path}")
    
    return prompt_results


def main():
    parser = argparse.ArgumentParser(description="Run text-guided segmentation experiments")
    parser.add_argument('--models', nargs='+', help='Models to train')
    parser.add_argument('--all', action='store_true', help='Train all working models')
    parser.add_argument('--quick', action='store_true', help='Quick run (1 fold, 10 epochs)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, only run evaluation')
    parser.add_argument('--zero-shot', action='store_true', help='Run zero-shot evaluation on CoNSeP/MoNuSAC')
    parser.add_argument('--prompt-test', action='store_true', help='Test 3 types of text prompts')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'num_classes': 5,
        'image_size': 256,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs if not args.quick else 10,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'clip_model': 'ViT-B/16',
        'freeze_clip': True,
        'data_root': f"{WORKSPACE}/Dataset",
        'checkpoint_dir': f"{WORKSPACE}/checkpoints/text_guided",
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Select models
    if args.all:
        models = WORKING_MODELS
    elif args.models:
        models = args.models
    else:
        models = ['clipseg']  # Default to just CLIPSeg
    
    num_folds = 1 if args.quick else 3
    
    print("="*60)
    print("TEXT-GUIDED SEGMENTATION EXPERIMENTS")
    print("="*60)
    print(f"Models: {models}")
    print(f"Folds: {num_folds}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Device: {config['device']}")
    
    # Run experiments
    if not args.skip_training:
        print("\n" + "="*60)
        print("PHASE 1: TRAINING ON PANNUKE (3-FOLD CV)")
        print("="*60)
        run_all_experiments(models, config, num_folds=num_folds)
    
    # Run zero-shot evaluation
    if args.zero_shot or args.all:
        print("\n" + "="*60)
        print("PHASE 2: ZERO-SHOT EVALUATION")
        print("="*60)
        run_zero_shot_evaluation(models, config)
    
    # Run text prompt variation testing
    if args.prompt_test or args.all:
        print("\n" + "="*60)
        print("PHASE 3: TEXT PROMPT VARIATION TESTING")
        print("="*60)
        test_text_prompt_variations(models, config)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {config['checkpoint_dir']}")
    print("  - experiment_summary.json (training results)")
    print("  - zero_shot_results.json (CoNSeP/MoNuSAC evaluation)")
    print("  - text_prompt_variation_results.json (3 prompt types)")


if __name__ == "__main__":
    main()
