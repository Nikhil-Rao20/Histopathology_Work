"""
PanNuke Dataset Preprocessing Script

Converts the original PanNuke .npy format to our unified format:
- PNG images
- NPZ masks (binary semantic masks per class)
- annotations.csv with metadata

Usage:
    python preprocess_pannuke.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml

# ============================================================================
# Configuration
# ============================================================================

# Paths
SOURCE_PATH = "Histopathology_Datasets_Official/PanNuke"
TARGET_PATH = "PanNuke_Preprocess"

# Class configuration (from PanNuke README)
# Channel mapping: 0: Neoplastic, 1: Inflammatory, 2: Connective/Soft tissue, 3: Dead, 4: Epithelial, 5: Background
CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective_Soft_tissue', 'Dead', 'Epithelial']
NUM_CLASSES = len(CLASS_NAMES)

# Folds to process
FOLDS = [1, 2, 3]


def create_directory_structure(target_path):
    """Create the unified directory structure."""
    for fold in FOLDS:
        os.makedirs(os.path.join(target_path, 'images', f'fold{fold}'), exist_ok=True)
        os.makedirs(os.path.join(target_path, 'masks', f'fold{fold}'), exist_ok=True)
    print(f"Created directory structure at {target_path}")


def instance_to_semantic(instance_masks):
    """
    Convert instance-level masks to binary semantic masks.
    
    Args:
        instance_masks: [H, W, 6] array where each channel contains instance IDs
        
    Returns:
        semantic_masks: [H, W, 5] binary array (excluding background channel)
    """
    # Convert instance masks to binary (instance ID > 0 means class is present)
    semantic_masks = (instance_masks[:, :, :5] > 0).astype(np.uint8)
    return semantic_masks


def get_classes_present(semantic_masks):
    """Get list of classes present in the image."""
    classes = []
    for i, class_name in enumerate(CLASS_NAMES):
        if np.any(semantic_masks[:, :, i] > 0):
            classes.append(class_name)
    return classes


def process_fold(source_path, target_path, fold_num):
    """Process a single fold."""
    print(f"\n{'='*60}")
    print(f"Processing Fold {fold_num}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading images...")
    images = np.load(os.path.join(source_path, f'Fold {fold_num}/images/fold{fold_num}/images.npy'))
    print("Loading masks...")
    masks = np.load(os.path.join(source_path, f'Fold {fold_num}/masks/fold{fold_num}/masks.npy'))
    print("Loading types...")
    types = np.load(os.path.join(source_path, f'Fold {fold_num}/images/fold{fold_num}/types.npy'))
    
    print(f"Loaded {len(images)} images")
    
    annotations = []
    
    # Process each image
    for idx in tqdm(range(len(images)), desc=f"Fold {fold_num}"):
        # Generate unique image ID
        image_id = f"pannuke_f{fold_num}_{idx:04d}"
        tissue_type = str(types[idx])
        
        # Get image (convert from float64 to uint8)
        image = images[idx].astype(np.uint8)
        
        # Get semantic masks
        semantic_masks = instance_to_semantic(masks[idx])
        
        # Get classes present
        classes_present = get_classes_present(semantic_masks)
        
        # Save image as PNG
        img_path = os.path.join(target_path, 'images', f'fold{fold_num}', f'{image_id}.png')
        Image.fromarray(image).save(img_path)
        
        # Save masks as NPZ
        mask_path = os.path.join(target_path, 'masks', f'fold{fold_num}', f'{image_id}.npz')
        np.savez_compressed(
            mask_path,
            masks=semantic_masks,
            class_names=CLASS_NAMES
        )
        
        # Add to annotations
        annotations.append({
            'image_id': image_id,
            'fold': fold_num,
            'tissue_type': tissue_type,
            'classes_present': ';'.join(classes_present) if classes_present else '',
            'instruction': '',  # Will be filled later with text generation
            'original_index': idx
        })
    
    return annotations


def create_config(target_path):
    """Create dataset configuration file."""
    config = {
        'dataset_name': 'PanNuke',
        'version': '1.0',
        'description': 'Pan-cancer histopathology nuclei dataset',
        'classes': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
        'folds': FOLDS,
        'original_resolution': [256, 256],
        'preprocessing': {
            'instance_to_semantic': True,
            'normalized': False
        },
        'stats': {
            'mean': [0.485, 0.456, 0.406],  # ImageNet default, can compute dataset-specific later
            'std': [0.229, 0.224, 0.225]
        },
        'tissue_types': [
            'Adrenal_gland', 'Bile-duct', 'Bladder', 'Breast', 'Cervix', 'Colon',
            'Esophagus', 'HeadNeck', 'Kidney', 'Liver', 'Lung', 'Ovarian',
            'Pancreatic', 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus'
        ]
    }
    
    config_path = os.path.join(target_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved config to {config_path}")


def main():
    print("="*60)
    print("PanNuke Dataset Preprocessing")
    print("="*60)
    print(f"Source: {SOURCE_PATH}")
    print(f"Target: {TARGET_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print("="*60)
    
    # Create directory structure
    create_directory_structure(TARGET_PATH)
    
    # Process all folds
    all_annotations = []
    for fold_num in FOLDS:
        fold_annotations = process_fold(SOURCE_PATH, TARGET_PATH, fold_num)
        all_annotations.extend(fold_annotations)
    
    # Save annotations CSV
    df = pd.DataFrame(all_annotations)
    csv_path = os.path.join(TARGET_PATH, 'annotations.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved annotations to {csv_path}")
    print(f"Total samples: {len(df)}")
    
    # Show fold distribution
    print("\nFold distribution:")
    for fold in FOLDS:
        count = len(df[df['fold'] == fold])
        print(f"  Fold {fold}: {count} samples")
    
    # Show class distribution
    print("\nClass distribution:")
    for class_name in CLASS_NAMES:
        count = df['classes_present'].str.contains(class_name, na=False).sum()
        pct = 100 * count / len(df)
        print(f"  {class_name}: {count} samples ({pct:.1f}%)")
    
    # Create config
    create_config(TARGET_PATH)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
