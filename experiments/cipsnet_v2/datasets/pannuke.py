"""
PanNuke Dataset for CIPS-Net V2
================================

Comprehensive dataloader for PanNuke dataset supporting:
1. 3-fold cross-validation
2. Multiple text conditioning modes
3. HoVer-Net style outputs (NP, HV, Type maps)
4. Data augmentation

Dataset Structure (Original PanNuke):
-------------------------------------
PanNuke/
├── Fold 1/
│   ├── images/fold1/images.npy  # (N, 256, 256, 3)
│   ├── images/fold1/types.npy   # (N,) tissue types
│   └── masks/fold1/masks.npy    # (N, 256, 256, 6) instance masks per class
├── Fold 2/
│   └── ...
└── Fold 3/
    └── ...

Mask Format:
------------
Channel 0: Neoplastic
Channel 1: Inflammatory
Channel 2: Connective/Soft tissue
Channel 3: Dead
Channel 4: Epithelial
Channel 5: Background (all zeros, not used)

Classes:
--------
0: Background
1: Neoplastic
2: Inflammatory
3: Connective/Soft tissue
4: Dead
5: Epithelial
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import cv2
from scipy.ndimage import distance_transform_edt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


# ============================================================
# Constants
# ============================================================

PANNUKE_CLASSES = {
    0: 'background',
    1: 'neoplastic',
    2: 'inflammatory',
    3: 'connective',  # Connective/Soft tissue
    4: 'dead',
    5: 'epithelial'
}

PANNUKE_CLASS_NAMES = [
    'neoplastic',
    'inflammatory',
    'connective',
    'dead',
    'epithelial'
]

PANNUKE_TISSUES = [
    'Adrenal_gland', 'Bile-duct', 'Bladder', 'Breast', 'Cervix',
    'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'Liver',
    'Lung', 'Ovarian', 'Pancreatic', 'Prostate', 'Skin',
    'Stomach', 'Testis', 'Thyroid', 'Uterus'
]

# Default text prompts for baseline mode (segment all)
DEFAULT_INSTRUCTIONS = [
    "Segment all nuclei in this histopathology image.",
    "Identify and segment all cell nuclei present.",
    "Detect all nuclear instances in this tissue sample.",
    "Mark all visible nuclei in the image.",
    "Perform instance segmentation of all nuclei."
]


# ============================================================
# HV Map Generation
# ============================================================

def gen_instance_hv_map(instance_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate horizontal and vertical distance maps from instance mask.
    
    For each instance, compute normalized distance from each pixel to the 
    instance center. H ranges from -1 (left) to 1 (right), V ranges from 
    -1 (top) to 1 (bottom).
    
    Args:
        instance_map: Instance segmentation map [H, W] with unique IDs per instance
        
    Returns:
        h_map: Horizontal distance map [H, W]
        v_map: Vertical distance map [H, W]
    """
    H, W = instance_map.shape
    h_map = np.zeros((H, W), dtype=np.float32)
    v_map = np.zeros((H, W), dtype=np.float32)
    
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]  # Remove background
    
    for inst_id in instance_ids:
        inst_mask = (instance_map == inst_id)
        
        # Get bounding box
        coords = np.where(inst_mask)
        if len(coords[0]) == 0:
            continue
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Create coordinate grids for the instance
        y_coords, x_coords = np.meshgrid(
            np.arange(y_min, y_max + 1),
            np.arange(x_min, x_max + 1),
            indexing='ij'
        )
        
        # Normalize to [-1, 1]
        if y_max > y_min:
            v_norm = 2 * (y_coords - y_min) / (y_max - y_min) - 1
        else:
            v_norm = np.zeros_like(y_coords, dtype=np.float32)
            
        if x_max > x_min:
            h_norm = 2 * (x_coords - x_min) / (x_max - x_min) - 1
        else:
            h_norm = np.zeros_like(x_coords, dtype=np.float32)
        
        # Apply to the instance region
        inst_crop = inst_mask[y_min:y_max+1, x_min:x_max+1]
        h_map[y_min:y_max+1, x_min:x_max+1][inst_crop] = h_norm[inst_crop]
        v_map[y_min:y_max+1, x_min:x_max+1][inst_crop] = v_norm[inst_crop]
    
    return h_map, v_map


def masks_to_instance_and_type(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert PanNuke 6-channel masks to instance map and type map.
    
    Args:
        masks: PanNuke mask array [H, W, 6]
        
    Returns:
        instance_map: Instance IDs [H, W]
        type_map: Class labels [H, W] (0=bg, 1-5=classes)
    """
    H, W = masks.shape[:2]
    instance_map = np.zeros((H, W), dtype=np.int32)
    type_map = np.zeros((H, W), dtype=np.int32)
    
    inst_id = 1
    
    # Process each class channel (0-4, skip 5 which is background)
    for class_idx in range(5):
        class_mask = masks[:, :, class_idx]
        
        # Get unique instance IDs in this channel
        unique_ids = np.unique(class_mask)
        unique_ids = unique_ids[unique_ids != 0]
        
        for uid in unique_ids:
            inst_mask = (class_mask == uid)
            instance_map[inst_mask] = inst_id
            type_map[inst_mask] = class_idx + 1  # 1-indexed class
            inst_id += 1
    
    return instance_map, type_map


def prepare_hover_targets(masks: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Prepare all HoVer-Net style targets from PanNuke masks.
    
    Args:
        masks: PanNuke mask array [H, W, 6]
        
    Returns:
        Dictionary with:
            - np_map: Binary nuclei presence [H, W]
            - hv_map: Horizontal-Vertical maps [H, W, 2]
            - type_map: Per-pixel class labels [H, W]
            - instance_map: Instance IDs [H, W]
    """
    # Convert to instance and type maps
    instance_map, type_map = masks_to_instance_and_type(masks)
    
    # Binary nuclei presence
    np_map = (instance_map > 0).astype(np.float32)
    
    # Generate HV maps
    h_map, v_map = gen_instance_hv_map(instance_map)
    hv_map = np.stack([h_map, v_map], axis=-1)
    
    return {
        'np_map': np_map,
        'hv_map': hv_map,
        'type_map': type_map,
        'instance_map': instance_map
    }


# ============================================================
# Data Augmentation
# ============================================================

def get_train_transforms(img_size: int = 256) -> A.Compose:
    """Get training augmentations."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            rotate=(-45, 45),
            border_mode=cv2.BORDER_REFLECT,
            p=0.5,
            fit_output=False
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        ], p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={
        'mask': 'mask',
        'hv_map': 'mask',
        'type_map': 'mask',
        'instance_map': 'mask'
    })


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """Get validation/test augmentations (only normalization)."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={
        'mask': 'mask',
        'hv_map': 'mask',
        'type_map': 'mask',
        'instance_map': 'mask'
    })


# ============================================================
# Text Generation Utilities
# ============================================================

def get_classes_from_mask(masks: np.ndarray) -> List[str]:
    """
    Get list of classes present in the mask.
    
    Args:
        masks: PanNuke mask array [H, W, 6]
        
    Returns:
        List of class names present
    """
    classes = []
    for i, class_name in enumerate(PANNUKE_CLASS_NAMES):
        if np.any(masks[:, :, i] > 0):
            classes.append(class_name)
    return classes


def generate_instruction_from_classes(
    classes: List[str],
    tissue_type: Optional[str] = None
) -> str:
    """
    Generate a natural language instruction from present classes.
    
    Args:
        classes: List of class names present
        tissue_type: Optional tissue type for context
        
    Returns:
        Generated instruction string
    """
    if not classes:
        return "Segment all nuclei in this image."
    
    # Templates for different numbers of classes
    templates_single = [
        "Segment {classes} nuclei in this image.",
        "Identify all {classes} cells.",
        "Detect the {classes} nuclear instances.",
        "Mark {classes} regions.",
    ]
    
    templates_multi = [
        "Segment {classes} nuclei in this image.",
        "Identify all {classes} cells.",
        "Detect {classes} nuclear instances.",
        "Mark the {classes} regions.",
    ]
    
    templates_tissue = [
        "In this {tissue} tissue, segment {classes} nuclei.",
        "Analyze this {tissue} sample and identify {classes} cells.",
        "{tissue} tissue: detect {classes} regions.",
    ]
    
    # Format class names
    if len(classes) == 1:
        class_str = classes[0]
        templates = templates_single
    elif len(classes) == 2:
        class_str = f"{classes[0]} and {classes[1]}"
        templates = templates_multi
    else:
        class_str = ", ".join(classes[:-1]) + f", and {classes[-1]}"
        templates = templates_multi
    
    # Choose template
    if tissue_type and random.random() > 0.5:
        template = random.choice(templates_tissue)
        return template.format(tissue=tissue_type, classes=class_str)
    else:
        template = random.choice(templates)
        return template.format(classes=class_str)


def generate_all_class_instructions() -> Dict[str, List[str]]:
    """
    Generate instruction templates for all class combinations.
    
    Returns:
        Dictionary mapping class combination to list of instructions
    """
    from itertools import combinations
    
    instructions = {}
    
    # Single classes
    for cls in PANNUKE_CLASS_NAMES:
        key = cls
        instructions[key] = [
            f"Segment all {cls} nuclei.",
            f"Identify {cls} cells in this image.",
            f"Detect {cls} nuclear instances.",
            f"Mark only {cls} regions.",
            f"Focus on {cls} cells.",
        ]
    
    # All classes
    instructions['all'] = DEFAULT_INSTRUCTIONS
    
    return instructions


# ============================================================
# Base Dataset Class
# ============================================================

class PanNukeDataset(Dataset):
    """
    Base PanNuke dataset class.
    
    Loads data from original PanNuke numpy files and prepares
    HoVer-Net style targets (NP, HV, Type maps).
    """
    
    def __init__(
        self,
        data_root: str,
        folds: List[int] = [1, 2, 3],
        transform: Optional[Callable] = None,
        cache_data: bool = True,
        precompute_targets: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Path to PanNuke root directory
            folds: List of fold numbers to include [1, 2, 3]
            transform: Albumentations transform
            cache_data: Whether to cache data in memory
            precompute_targets: Whether to precompute HV maps (slower init, faster training)
        """
        self.data_root = Path(data_root)
        self.folds = folds
        self.transform = transform
        self.cache_data = cache_data
        self.precompute_targets = precompute_targets
        
        # Load data from all specified folds
        self.images = []
        self.masks = []
        self.types = []
        self.fold_indices = []  # Track which fold each sample belongs to
        self.local_indices = []  # Track local index within fold
        
        for fold in folds:
            fold_path = self.data_root / f"Fold {fold}"
            
            images = np.load(fold_path / f"images/fold{fold}/images.npy")
            masks = np.load(fold_path / f"masks/fold{fold}/masks.npy")
            types = np.load(fold_path / f"images/fold{fold}/types.npy")
            
            n_samples = len(images)
            
            if cache_data:
                self.images.append(images)
                self.masks.append(masks)
                self.types.extend(types.tolist())
            else:
                # Store paths for lazy loading
                self.images.append(fold_path / f"images/fold{fold}/images.npy")
                self.masks.append(fold_path / f"masks/fold{fold}/masks.npy")
                self.types.extend(types.tolist())
            
            self.fold_indices.extend([fold] * n_samples)
            self.local_indices.extend(list(range(n_samples)))
        
        if cache_data:
            self.images = np.concatenate(self.images, axis=0)
            self.masks = np.concatenate(self.masks, axis=0)
        
        # Precompute targets if requested
        self.cached_targets = None
        if precompute_targets and cache_data:
            print(f"Precomputing HV targets for {len(self)} samples...")
            self.cached_targets = []
            for i in range(len(self)):
                targets = prepare_hover_targets(self.masks[i])
                self.cached_targets.append(targets)
            print("Done!")
        
        print(f"Loaded PanNuke dataset: {len(self)} samples from folds {folds}")
    
    def __len__(self) -> int:
        if self.cache_data:
            return len(self.images)
        else:
            return len(self.fold_indices)
    
    def _get_raw_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
        """Get raw image, mask, tissue type, fold, and local index."""
        if self.cache_data:
            image = self.images[idx]
            mask = self.masks[idx]
        else:
            # Lazy loading
            fold = self.fold_indices[idx]
            local_idx = self.local_indices[idx]
            
            # Find which fold's data to load
            fold_offset = 0
            for i, f in enumerate(self.folds):
                if f == fold:
                    images = np.load(self.images[i], mmap_mode='r')
                    masks = np.load(self.masks[i], mmap_mode='r')
                    image = images[local_idx].copy()
                    mask = masks[local_idx].copy()
                    break
        
        tissue = self.types[idx]
        fold = self.fold_indices[idx]
        local_idx = self.local_indices[idx]
        
        return image, mask, tissue, fold, local_idx
    
    def _prepare_targets(self, mask: np.ndarray, idx: int) -> Dict[str, np.ndarray]:
        """Prepare HoVer-Net targets."""
        if self.cached_targets is not None:
            return self.cached_targets[idx]
        return prepare_hover_targets(mask)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        image, mask, tissue, fold, local_idx = self._get_raw_data(idx)
        
        # Convert image to uint8 if needed
        if image.dtype == np.float64:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Prepare targets
        targets = self._prepare_targets(mask, idx)
        np_map = targets['np_map']
        hv_map = targets['hv_map']
        type_map = targets['type_map']
        instance_map = targets['instance_map']
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=np_map,
                hv_map=hv_map,
                type_map=type_map,
                instance_map=instance_map
            )
            image = transformed['image']
            np_map = transformed['mask']
            hv_map = transformed['hv_map']
            type_map = transformed['type_map']
            instance_map = transformed['instance_map']
        else:
            # Manual conversion to tensors
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            np_map = torch.from_numpy(np_map).float()
            hv_map = torch.from_numpy(hv_map).float()
            type_map = torch.from_numpy(type_map).long()
            instance_map = torch.from_numpy(instance_map).long()
        
        # Ensure correct tensor formats
        if isinstance(hv_map, np.ndarray):
            hv_map = torch.from_numpy(hv_map).float()
        if hv_map.dim() == 3 and hv_map.shape[-1] == 2:
            hv_map = hv_map.permute(2, 0, 1)  # [H, W, 2] -> [2, H, W]
        
        if isinstance(type_map, np.ndarray):
            type_map = torch.from_numpy(type_map).long()
        
        if isinstance(instance_map, np.ndarray):
            instance_map = torch.from_numpy(instance_map).long()
        
        return {
            'image': image,
            'np_map': np_map.unsqueeze(0) if np_map.dim() == 2 else np_map,
            'hv_map': hv_map,
            'type_map': type_map,
            'instance_map': instance_map,
            'tissue': tissue,
            'fold': fold,
            'idx': idx,
            'local_idx': local_idx
        }


# ============================================================
# Baseline Dataset (No Text)
# ============================================================

class PanNukeBaselineDataset(PanNukeDataset):
    """
    Baseline dataset without text instructions.
    
    For training baseline model (BASELINE variant).
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        # No text instruction for baseline
        return sample


# ============================================================
# Single Text Dataset
# ============================================================

class PanNukeSingleTextDataset(PanNukeDataset):
    """
    Dataset with a single text instruction per image.
    
    Uses annotations.csv for text instructions or generates them from mask.
    """
    
    def __init__(
        self,
        data_root: str,
        annotations_csv: Optional[str] = None,
        folds: List[int] = [1, 2, 3],
        transform: Optional[Callable] = None,
        use_generated_instructions: bool = False,
        cache_data: bool = True,
        precompute_targets: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Path to PanNuke root directory
            annotations_csv: Path to annotations.csv (optional)
            folds: List of fold numbers to include
            transform: Albumentations transform
            use_generated_instructions: Generate instructions from masks instead of CSV
            cache_data: Whether to cache data in memory
            precompute_targets: Whether to precompute HV maps
        """
        super().__init__(data_root, folds, transform, cache_data, precompute_targets)
        
        self.use_generated = use_generated_instructions
        self.annotations = None
        
        if annotations_csv and not use_generated_instructions:
            self.annotations = pd.read_csv(annotations_csv)
            # Filter to specified folds
            self.annotations = self.annotations[
                self.annotations['fold'].isin(folds)
            ].reset_index(drop=True)
            print(f"Loaded {len(self.annotations)} annotations from CSV")
    
    def _get_instruction(self, idx: int) -> str:
        """Get text instruction for sample."""
        if self.use_generated or self.annotations is None:
            # Generate instruction from mask
            _, mask, tissue, _, _ = self._get_raw_data(idx)
            classes = get_classes_from_mask(mask)
            return generate_instruction_from_classes(classes, tissue)
        else:
            # Use CSV annotation
            row = self.annotations.iloc[idx]
            return row['instruction']
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        sample['instruction'] = self._get_instruction(idx)
        return sample


# ============================================================
# Multi-Text Dataset
# ============================================================

class PanNukeMultiTextDataset(PanNukeDataset):
    """
    Dataset with multiple possible text instructions per image.
    
    During training, randomly samples one instruction from available options.
    Useful for data augmentation through text variation.
    """
    
    def __init__(
        self,
        data_root: str,
        annotations_csv: Optional[str] = None,
        folds: List[int] = [1, 2, 3],
        transform: Optional[Callable] = None,
        num_variations: int = 5,
        include_tissue_context: bool = True,
        cache_data: bool = True,
        precompute_targets: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Path to PanNuke root directory
            annotations_csv: Path to annotations.csv (optional)
            folds: List of fold numbers to include
            transform: Albumentations transform
            num_variations: Number of instruction variations to generate
            include_tissue_context: Include tissue type in some instructions
            cache_data: Whether to cache data in memory
            precompute_targets: Whether to precompute HV maps
        """
        super().__init__(data_root, folds, transform, cache_data, precompute_targets)
        
        self.num_variations = num_variations
        self.include_tissue_context = include_tissue_context
        
        # Load annotations if provided (for reference)
        self.annotations = None
        if annotations_csv:
            self.annotations = pd.read_csv(annotations_csv)
            self.annotations = self.annotations[
                self.annotations['fold'].isin(folds)
            ].reset_index(drop=True)
    
    def _generate_instructions(self, idx: int) -> List[str]:
        """Generate multiple instruction variations for a sample."""
        _, mask, tissue, _, _ = self._get_raw_data(idx)
        classes = get_classes_from_mask(mask)
        
        instructions = []
        
        # Generate variations
        for _ in range(self.num_variations):
            if self.include_tissue_context:
                inst = generate_instruction_from_classes(classes, tissue)
            else:
                inst = generate_instruction_from_classes(classes, None)
            instructions.append(inst)
        
        # Add CSV instruction if available
        if self.annotations is not None and idx < len(self.annotations):
            instructions.append(self.annotations.iloc[idx]['instruction'])
        
        return list(set(instructions))  # Remove duplicates
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        
        # Get all possible instructions
        instructions = self._generate_instructions(idx)
        
        # Randomly select one for this iteration
        sample['instruction'] = random.choice(instructions)
        sample['all_instructions'] = instructions
        
        return sample


# ============================================================
# Collate Functions
# ============================================================

def baseline_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for baseline dataset (no text)."""
    images = torch.stack([b['image'] for b in batch])
    np_maps = torch.stack([b['np_map'] for b in batch])
    hv_maps = torch.stack([b['hv_map'] for b in batch])
    type_maps = torch.stack([b['type_map'] for b in batch])
    instance_maps = torch.stack([b['instance_map'] for b in batch])
    
    return {
        'images': images,
        'np_maps': np_maps,
        'hv_maps': hv_maps,
        'type_maps': type_maps,
        'instance_maps': instance_maps,
        'tissues': [b['tissue'] for b in batch],
        'folds': [b['fold'] for b in batch],
        'indices': [b['idx'] for b in batch]
    }


def text_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Collate function for text-conditioned datasets."""
    images = torch.stack([b['image'] for b in batch])
    np_maps = torch.stack([b['np_map'] for b in batch])
    hv_maps = torch.stack([b['hv_map'] for b in batch])
    type_maps = torch.stack([b['type_map'] for b in batch])
    instance_maps = torch.stack([b['instance_map'] for b in batch])
    
    return {
        'images': images,
        'np_maps': np_maps,
        'hv_maps': hv_maps,
        'type_maps': type_maps,
        'instance_maps': instance_maps,
        'instructions': [b['instruction'] for b in batch],
        'tissues': [b['tissue'] for b in batch],
        'folds': [b['fold'] for b in batch],
        'indices': [b['idx'] for b in batch]
    }


# ============================================================
# DataLoader Factory Functions
# ============================================================

def create_pannuke_dataloaders(
    data_root: str,
    annotations_csv: Optional[str] = None,
    mode: str = 'baseline',
    train_folds: List[int] = [1, 2],
    val_folds: List[int] = [3],
    batch_size: int = 8,
    num_workers: int = 4,
    cache_data: bool = True,
    precompute_targets: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Path to PanNuke root directory
        annotations_csv: Path to annotations.csv (for text modes)
        mode: Dataset mode - 'baseline', 'single_text', 'multi_text'
        train_folds: Folds to use for training
        val_folds: Folds to use for validation
        batch_size: Batch size
        num_workers: Number of data loading workers
        cache_data: Whether to cache data in memory
        precompute_targets: Whether to precompute HV maps
        **kwargs: Additional arguments for dataset
        
    Returns:
        train_loader, val_loader
    """
    # Select dataset class based on mode
    if mode == 'baseline':
        DatasetClass = PanNukeBaselineDataset
        collate_fn = baseline_collate_fn
    elif mode == 'single_text':
        DatasetClass = PanNukeSingleTextDataset
        collate_fn = text_collate_fn
    elif mode == 'multi_text':
        DatasetClass = PanNukeMultiTextDataset
        collate_fn = text_collate_fn
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'baseline', 'single_text', 'multi_text'")
    
    # Create datasets
    common_args = {
        'data_root': data_root,
        'cache_data': cache_data,
        'precompute_targets': precompute_targets
    }
    
    if mode != 'baseline':
        common_args['annotations_csv'] = annotations_csv
    
    train_dataset = DatasetClass(
        folds=train_folds,
        transform=get_train_transforms(),
        **common_args,
        **kwargs
    )
    
    val_dataset = DatasetClass(
        folds=val_folds,
        transform=get_val_transforms(),
        **common_args,
        **kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_fold_dataloaders(
    data_root: str,
    annotations_csv: Optional[str] = None,
    mode: str = 'baseline',
    test_fold: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for 3-fold cross-validation.
    
    Following PanNuke convention:
    - test_fold=1: Train on 2,3 | Val split from train | Test on 1
    - test_fold=2: Train on 1,3 | Val split from train | Test on 2
    - test_fold=3: Train on 1,2 | Val split from train | Test on 3
    
    Args:
        data_root: Path to PanNuke root directory
        annotations_csv: Path to annotations.csv
        mode: Dataset mode
        test_fold: Which fold to use as test set
        batch_size: Batch size
        num_workers: Number of workers
        **kwargs: Additional arguments
        
    Returns:
        train_loader, val_loader, test_loader
    """
    all_folds = [1, 2, 3]
    train_folds = [f for f in all_folds if f != test_fold]
    
    # Use one training fold for validation (simple split)
    # In practice, you might want to do a more sophisticated split
    train_loader, val_loader = create_pannuke_dataloaders(
        data_root=data_root,
        annotations_csv=annotations_csv,
        mode=mode,
        train_folds=train_folds,
        val_folds=[test_fold],  # Use test fold as validation for monitoring
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
    
    # Create test loader (same as val but labeled as test)
    test_loader = val_loader
    
    print(f"\n3-Fold CV Setup (test_fold={test_fold}):")
    print(f"  Train folds: {train_folds} ({len(train_loader.dataset)} samples)")
    print(f"  Test fold: {test_fold} ({len(test_loader.dataset)} samples)")
    
    return train_loader, val_loader, test_loader


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PanNuke dataloader')
    parser.add_argument('--data_root', type=str, required=True, help='Path to PanNuke root')
    parser.add_argument('--annotations', type=str, default=None, help='Path to annotations.csv')
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'single_text', 'multi_text'])
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Testing PanNuke DataLoader")
    print("="*60)
    
    # Test dataloader creation
    train_loader, val_loader = create_pannuke_dataloaders(
        data_root=args.data_root,
        annotations_csv=args.annotations,
        mode=args.mode,
        train_folds=[1, 2],
        val_folds=[3],
        batch_size=4,
        num_workers=0,
        cache_data=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  images: {batch['images'].shape}")
    print(f"  np_maps: {batch['np_maps'].shape}")
    print(f"  hv_maps: {batch['hv_maps'].shape}")
    print(f"  type_maps: {batch['type_maps'].shape}")
    print(f"  instance_maps: {batch['instance_maps'].shape}")
    print(f"  tissues: {batch['tissues']}")
    
    if args.mode != 'baseline':
        print(f"  instructions: {batch['instructions']}")
    
    print("\n✓ DataLoader test passed!")
