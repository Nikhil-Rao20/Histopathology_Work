"""
PanNuke Permutation Dataset for Referring Expression Segmentation
==================================================================

This dataset uses the user-prepared permutation CSV that contains all 2^N-1 
class combinations for each image. This enables PROPER text-conditioned
segmentation training where:

    TEXT ←→ MASK correspondence is EXPLICIT

Training Protocol:
    - Use ALL permutations from CSV (diverse text prompts)
    - Return PARTIAL mask (only classes mentioned in text)
    - Model learns: "segment neoplastic" → only neoplastic mask

Testing Protocol:
    - Use max-class entries per image (all classes present)
    - Return FULL mask for fair SOTA comparison
    - Evaluates complete segmentation capability

Why this matters:
    - Previous dataloader used generic templates + always full mask
    - Text was noise → model learned to ignore it
    - This dataloader creates proper text-mask alignment

Author: Created for CIPS-Net V2 Ablation Study
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import cv2
import random
from collections import defaultdict
from scipy.ndimage import distance_transform_edt as scipy_edt

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Constants
# ============================================================

# Mapping between CSV class names and channel indices
CLASS_TO_CHANNEL = {
    'Neoplastic': 0,
    'Inflammatory': 1, 
    'Connective_Soft_tissue': 2,
    'Dead': 3,
    'Epithelial': 4
}

CHANNEL_TO_CLASS = {v: k for k, v in CLASS_TO_CHANNEL.items()}

# Standard 5 classes (lowercase for model compatibility)
PANNUKE_CLASS_NAMES = [
    'neoplastic',
    'inflammatory', 
    'connective',
    'dead',
    'epithelial'
]


# ============================================================
# HV Map Generation (Same as before)
# ============================================================

def gen_instance_hv_map(instance_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate horizontal and vertical distance maps from instance mask."""
    H, W = instance_map.shape
    h_map = np.zeros((H, W), dtype=np.float32)
    v_map = np.zeros((H, W), dtype=np.float32)
    
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]
    
    for inst_id in instance_ids:
        inst_mask = (instance_map == inst_id)
        coords = np.where(inst_mask)
        if len(coords[0]) == 0:
            continue
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        y_coords, x_coords = np.meshgrid(
            np.arange(y_min, y_max + 1),
            np.arange(x_min, x_max + 1),
            indexing='ij'
        )
        
        if y_max > y_min:
            v_norm = 2 * (y_coords - y_min) / (y_max - y_min) - 1
        else:
            v_norm = np.zeros_like(y_coords, dtype=np.float32)
            
        if x_max > x_min:
            h_norm = 2 * (x_coords - x_min) / (x_max - x_min) - 1
        else:
            h_norm = np.zeros_like(x_coords, dtype=np.float32)
        
        inst_crop = inst_mask[y_min:y_max+1, x_min:x_max+1]
        h_map[y_min:y_max+1, x_min:x_max+1][inst_crop] = h_norm[inst_crop]
        v_map[y_min:y_max+1, x_min:x_max+1][inst_crop] = v_norm[inst_crop]
    
    return h_map, v_map


def masks_to_instance_and_type(masks: np.ndarray, requested_channels: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert multi-channel masks to instance and type maps.
    
    Args:
        masks: (H, W, 5 or 6) numpy array with per-class instance masks
        requested_channels: List of channels to include. If None, include all.
    
    Returns:
        instance_map: (H, W) unique instance IDs
        type_map: (H, W) class labels (1-5)
    """
    H, W = masks.shape[:2]
    instance_map = np.zeros((H, W), dtype=np.int32)
    type_map = np.zeros((H, W), dtype=np.int32)
    
    # Default to all 5 classes
    if requested_channels is None:
        requested_channels = list(range(5))
    
    inst_id = 1
    for class_idx in requested_channels:
        if class_idx >= masks.shape[2]:
            continue
            
        class_mask = masks[:, :, class_idx]
        unique_ids = np.unique(class_mask)
        unique_ids = unique_ids[unique_ids != 0]
        
        for uid in unique_ids:
            inst_mask = (class_mask == uid)
            instance_map[inst_mask] = inst_id
            type_map[inst_mask] = class_idx + 1  # 1-indexed
            inst_id += 1
    
    return instance_map, type_map


def gen_normalized_distance_transform(instance_map: np.ndarray) -> np.ndarray:
    """Generate normalized EDT from instance map. Peaks at 1.0 at centers, 0 at boundaries."""
    H, W = instance_map.shape
    dist_map = np.zeros((H, W), dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        inst_mask = (instance_map == inst_id)
        edt = scipy_edt(inst_mask)
        max_val = edt.max()
        if max_val > 0:
            dist_map[inst_mask] = edt[inst_mask] / max_val
        else:
            dist_map[inst_mask] = 1.0
    return dist_map


def prepare_hover_targets(masks: np.ndarray, requested_channels: List[int] = None) -> Dict[str, np.ndarray]:
    """Prepare HoVer-Net targets from masks. Also computes normalized EDT for IE decoder."""
    instance_map, type_map = masks_to_instance_and_type(masks, requested_channels)
    np_map = (instance_map > 0).astype(np.float32)
    h_map, v_map = gen_instance_hv_map(instance_map)
    hv_map = np.stack([h_map, v_map], axis=-1)
    dist_map = gen_normalized_distance_transform(instance_map)
    
    return {
        'np_map': np_map,
        'hv_map': hv_map,
        'type_map': type_map,
        'instance_map': instance_map,
        'dist_map': dist_map,
    }


# ============================================================
# Transforms
# ============================================================

def get_train_transforms(img_size: int = 256) -> A.Compose:
    """Training augmentations."""
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
        ], p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={
        'mask': 'mask',
        'hv_map': 'mask',
        'type_map': 'mask',
        'instance_map': 'mask',
        'dist_map': 'mask',
    })


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """Validation transforms."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={
        'mask': 'mask',
        'hv_map': 'mask',
        'type_map': 'mask',
        'instance_map': 'mask',
        'dist_map': 'mask',
    })


# ============================================================
# Permutation Dataset
# ============================================================

class PanNukePermutationDataset(Dataset):
    """
    Dataset that uses the permutation CSV for proper text-mask alignment.
    
    Key features:
    - Training: Uses all permutations with PARTIAL masks
    - Testing: Uses max-class entries with FULL masks
    - Proper text ↔ mask correspondence for referring segmentation
    
    Args:
        csv_path: Path to Images_With_Permutations_Labels_Refer_Segmentation_Task.csv
        images_dir: Path to multi_images folder
        masks_dir: Path to multi_masks folder
        folds: List of fold numbers to use [1, 2, 3]
        transform: Albumentations transform
        mode: 'train' (all permutations + partial masks) or 'test' (max-class only + full mask)
        variant: Model variant ('BASELINE', 'WITH_TEXT', 'WITH_CGR', etc.)
        augmentation_config: AugmentationConfig for dataset expansion (train only, None = no expansion)
    """
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        masks_dir: str,
        folds: List[int] = [1, 2, 3],
        transform: Optional[Callable] = None,
        mode: str = 'train',
        variant: str = 'BASELINE',
        augmentation_config=None,
    ):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.folds = sorted(folds)
        self.transform = transform
        self.mode = mode
        self.variant = variant
        
        # Load and filter CSV
        print(f"Loading permutation CSV from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Filter by folds
        self._filter_by_folds()
        
        # For test mode: keep only max-class entries per image
        if mode == 'test':
            self._filter_max_class_entries()
        
        # Build index
        self.samples = self.df.to_dict('records')
        
        # Data augmentation expansion (only affects training)
        self.aug_config = augmentation_config
        self._original_len = len(self.samples)
        
        if self.aug_config and self.aug_config.enabled:
            factor = self.aug_config.expansion_factor
            print(f"PanNukePermutation [{mode}]: {self._original_len} original × {factor} = "
                  f"{len(self)} samples from folds {folds} "
                  f"(augmentations: {self.aug_config.augmentation_names})")
        else:
            print(f"PanNukePermutation [{mode}]: {len(self)} samples from folds {folds}")
        print(f"  Variant: {variant}")
        if mode == 'train':
            print(f"  Using ALL permutations with PARTIAL masks")
        else:
            print(f"  Using max-class entries with FULL masks")
    
    def _filter_by_folds(self):
        """Filter DataFrame to only include samples from specified folds."""
        # Extract fold number from image_path (e.g., "1_Breast_fold_1_0000_img.png")
        def get_fold(image_path):
            # Format: {FoldNum}_{Organ}_fold_{FoldNum}_{ID}_img.png
            parts = image_path.split('_')
            # Find "fold" and get the next number
            for i, part in enumerate(parts):
                if part == 'fold' and i + 1 < len(parts):
                    return int(parts[i + 1])
            return None
        
        self.df['fold'] = self.df['image_path'].apply(get_fold)
        self.df = self.df[self.df['fold'].isin(self.folds)].reset_index(drop=True)
    
    def _filter_max_class_entries(self):
        """For test mode: keep only the entry with most classes per image."""
        def count_classes(classes_str):
            return len(classes_str.split(';'))
        
        self.df['num_classes'] = self.df['classes'].apply(count_classes)
        
        # For each image_id, keep only the row with max classes
        idx_max = self.df.groupby('image_id')['num_classes'].idxmax()
        self.df = self.df.loc[idx_max].reset_index(drop=True)
        self.df = self.df.drop(columns=['num_classes'])
    
    def __len__(self) -> int:
        if self.aug_config and self.aug_config.enabled:
            return self._original_len * self.aug_config.expansion_factor
        return self._original_len
    
    def _parse_classes(self, classes_str: str) -> List[int]:
        """Parse semicolon-separated classes to channel indices."""
        classes = classes_str.split(';')
        channels = []
        for cls in classes:
            cls = cls.strip()
            if cls in CLASS_TO_CHANNEL:
                channels.append(CLASS_TO_CHANNEL[cls])
        return sorted(channels)
    
    def _get_all_channels_for_image(self, image_path: str) -> List[int]:
        """Get all channels that have content for this image."""
        # Parse image path to construct mask paths
        # Image: 1_Breast_fold_1_0000_img.png
        # Mask: 1_Breast_fold_1_0000_channel_0_Neoplastic.png
        
        base_name = image_path.replace('_img.png', '')
        
        channels = []
        for class_name, channel_idx in CLASS_TO_CHANNEL.items():
            mask_name = f"{base_name}_channel_{channel_idx}_{class_name}.png"
            mask_path = self.masks_dir / mask_name
            
            if mask_path.exists():
                # Check if mask has any content
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.any(mask > 0):
                    channels.append(channel_idx)
        
        return sorted(channels)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from multi_images folder."""
        img_path = self.images_dir / image_path
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_masks(self, image_path: str, requested_channels: List[int] = None) -> np.ndarray:
        """
        Load masks from multi_masks folder.
        
        Args:
            image_path: Image filename (e.g., "1_Breast_fold_1_0000_img.png")
            requested_channels: Channels to load. If None, load all 5 channels.
        
        Returns:
            masks: (H, W, 5) numpy array with per-class instance masks
        """
        base_name = image_path.replace('_img.png', '')
        
        # First, load any mask to get dimensions
        sample_mask = None
        for class_name, channel_idx in CLASS_TO_CHANNEL.items():
            mask_name = f"{base_name}_channel_{channel_idx}_{class_name}.png"
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                sample_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if sample_mask is not None:
                    break
        
        if sample_mask is None:
            # Try loading from a default size (256x256)
            H, W = 256, 256
        else:
            H, W = sample_mask.shape
        
        # Initialize masks array
        masks = np.zeros((H, W, 5), dtype=np.int32)
        
        # Load each channel
        for class_name, channel_idx in CLASS_TO_CHANNEL.items():
            # For test mode: load all channels
            # For train mode: only load requested channels
            if requested_channels is not None and channel_idx not in requested_channels:
                continue
            
            mask_name = f"{base_name}_channel_{channel_idx}_{class_name}.png"
            mask_path = self.masks_dir / mask_name
            
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks[:, :, channel_idx] = mask
        
        return masks
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        # Resolve expansion index (maps augmented idx → original idx + aug type)
        from .augmentation import resolve_expansion_index, apply_expansion_augmentation
        orig_idx, aug_type = resolve_expansion_index(idx, self._original_len, self.aug_config)
        
        sample = self.samples[orig_idx]
        
        image_path = sample['image_path']
        classes_str = sample['classes']
        instruction = sample['instruction']
        organ = sample['organ']
        fold = sample['fold']
        
        # Load image
        image = self._load_image(image_path)
        
        # Parse requested classes
        requested_channels = self._parse_classes(classes_str)
        
        # Determine which channels to use for mask
        if self.mode == 'test':
            # Test: use ALL channels present in the image (full mask)
            all_channels = self._get_all_channels_for_image(image_path)
            mask_channels = all_channels if all_channels else requested_channels
        else:
            # Train: use ONLY requested channels (partial mask)
            mask_channels = requested_channels
        
        # Load masks
        masks = self._load_masks(image_path, mask_channels)
        
        # Apply expansion augmentation BEFORE HoVer targets and transforms
        if aug_type is not None:
            image, masks = apply_expansion_augmentation(
                image, masks, aug_type, self.aug_config, orig_idx,
            )
        
        # Prepare HoVer targets
        targets = prepare_hover_targets(masks, mask_channels)
        np_map = targets['np_map']
        hv_map = targets['hv_map']
        type_map = targets['type_map']
        instance_map = targets['instance_map']
        dist_map = targets['dist_map']
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=np_map,
                hv_map=hv_map,
                type_map=type_map,
                instance_map=instance_map,
                dist_map=dist_map,
            )
            image = transformed['image']
            np_map = transformed['mask']
            hv_map = transformed['hv_map']
            type_map = transformed['type_map']
            instance_map = transformed['instance_map']
            dist_map = transformed['dist_map']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            np_map = torch.from_numpy(np_map).float()
            hv_map = torch.from_numpy(hv_map).float()
            type_map = torch.from_numpy(type_map).long()
            instance_map = torch.from_numpy(instance_map).long()
            dist_map = torch.from_numpy(dist_map).float()
        
        # Ensure correct tensor formats
        if isinstance(hv_map, np.ndarray):
            hv_map = torch.from_numpy(hv_map).float()
        if hv_map.dim() == 3 and hv_map.shape[-1] == 2:
            hv_map = hv_map.permute(2, 0, 1)
        
        if isinstance(type_map, np.ndarray):
            type_map = torch.from_numpy(type_map).long()
        if isinstance(instance_map, np.ndarray):
            instance_map = torch.from_numpy(instance_map).long()
        if isinstance(dist_map, np.ndarray):
            dist_map = torch.from_numpy(dist_map).float()
        
        # Build result dict
        result = {
            'image': image,
            'np_map': np_map.unsqueeze(0) if np_map.dim() == 2 else np_map,
            'hv_map': hv_map,
            'type_map': type_map,
            'instance_map': instance_map,
            'dist_map': dist_map.unsqueeze(0) if dist_map.dim() == 2 else dist_map,
            'tissue': organ,
            'fold': fold,
            'idx': idx,
            'requested_classes': classes_str,
        }
        
        # Add instruction for text-conditioned variants
        if self.variant != 'BASELINE':
            result['instruction'] = instruction
        else:
            # For BASELINE: still include instruction but it won't be used
            result['instruction'] = "Segment all nuclei in this histopathology image."
        
        return result


# ============================================================
# Collate Function
# ============================================================

def permutation_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for permutation dataset.
    
    Keys are named to match experiment_runner expectations:
    - 'images' (not 'image')
    - 'np_maps' (not 'np_map')
    - 'hv_maps' (not 'hv_map')
    - 'type_maps' (not 'type_map')
    - 'instructions' (not 'instruction')
    - 'tissues' (plural, for testing)
    - 'indices' (plural, for testing)
    """
    result = {
        'images': torch.stack([b['image'] for b in batch]),
        'np_maps': torch.stack([b['np_map'] for b in batch]),
        'hv_maps': torch.stack([b['hv_map'] for b in batch]),
        'type_maps': torch.stack([b['type_map'] for b in batch]),
        'instance_maps': torch.stack([b['instance_map'] for b in batch]),
        'dist_maps': torch.stack([b['dist_map'] for b in batch]),
        'tissues': [b['tissue'] for b in batch],  # plural to match experiment_runner
        'fold': [b['fold'] for b in batch],
        'indices': [b['idx'] for b in batch],  # renamed to match experiment_runner
        'instructions': [b['instruction'] for b in batch],
        'requested_classes': [b['requested_classes'] for b in batch],
    }
    return result


# ============================================================
# Factory Functions
# ============================================================

def create_permutation_dataloaders(
    csv_path: str,
    images_dir: str,
    masks_dir: str,
    train_folds: List[int],
    val_folds: List[int],
    batch_size: int = 8,
    num_workers: int = 4,
    variant: str = 'BASELINE',
    img_size: int = 256,
    augmentation_config=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        csv_path: Path to permutation CSV
        images_dir: Path to multi_images folder
        masks_dir: Path to multi_masks folder
        train_folds: Folds for training [1, 2] or [1, 3] or [2, 3]
        val_folds: Fold for validation [3] or [2] or [1]
        batch_size: Batch size
        num_workers: Number of dataloader workers
        variant: Model variant
        img_size: Image size (256)
        augmentation_config: AugmentationConfig for train dataset expansion
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = PanNukePermutationDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        masks_dir=masks_dir,
        folds=train_folds,
        transform=get_train_transforms(img_size),
        mode='train',
        variant=variant,
        augmentation_config=augmentation_config,  # Expansion augmentation (train only)
    )
    
    val_dataset = PanNukePermutationDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        masks_dir=masks_dir,
        folds=val_folds,
        transform=get_val_transforms(img_size),
        mode='test',  # Validation uses test mode (full masks)
        variant=variant,
        # NOTE: No augmentation_config for val — never expand val/test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=permutation_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=permutation_collate_fn
    )
    
    return train_loader, val_loader


def create_permutation_fold_dataloaders(
    data_root: str,
    test_fold: int = 1,
    batch_size: int = 8,
    num_workers: int = 0,
    variant: str = 'BASELINE',
    use_permutations: bool = False,
    augmentation_config=None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for 3-fold cross-validation.
    
    This is the main factory function used by experiment_runner.py
    
    Args:
        data_root: Path to Histopathology_Work folder (or PanNuke - not used here)
        test_fold: Which fold to use as test set (1, 2, or 3)
        batch_size: Batch size
        num_workers: Number of workers
        variant: Model variant ('BASELINE', 'WITH_TEXT', etc.)
        use_permutations: If True, use permutations CSV (partial masks)
                         If False, use unique labels CSV (full masks)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Paths to dataset - derive Histopathology_Work from PanNuke path
    # PanNuke path: .../Histopathology_Work/Histopathology_Datasets_Official/PanNuke
    # We need:      .../Histopathology_Work/Dataset/...
    
    data_root_path = Path(data_root)
    
    # Find Histopathology_Work in the path
    path_parts = data_root_path.parts
    try:
        hw_idx = path_parts.index('Histopathology_Work')
        base_path = Path(*path_parts[:hw_idx + 1])
    except ValueError:
        # Fallback: assume data_root is directly Histopathology_Work or contains Dataset
        if (data_root_path / 'Dataset').exists():
            base_path = data_root_path
        else:
            base_path = data_root_path.parent.parent  # Go up from PanNuke
    
    # Choose CSV based on use_permutations flag
    if use_permutations:
        # Permutations CSV: 2^N-1 entries per image with partial masks
        csv_path = base_path / "Dataset/Images_With_Permutations_Labels_Refer_Segmentation_Task_FULL.csv"
        csv_type = "Permutations (partial masks)"
    else:
        # Unique labels CSV: 1 entry per image with full masks
        csv_path = base_path / "Dataset/Images_With_Unique_Labels_Refer_Segmentation_Task_FULL.csv"
        csv_type = "Unique Labels (full masks)"
    
    images_dir = base_path / "Dataset/multi_images"
    masks_dir = base_path / "Dataset/multi_masks"
    
    # Validate paths
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")
    
    all_folds = [1, 2, 3]
    train_folds = [f for f in all_folds if f != test_fold]
    
    train_loader, val_loader = create_permutation_dataloaders(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        train_folds=train_folds,
        val_folds=[test_fold],
        batch_size=batch_size,
        num_workers=num_workers,
        variant=variant,
        augmentation_config=augmentation_config,
        **kwargs
    )
    
    # Test loader is same as val loader for this fold
    test_loader = val_loader
    
    print(f"\n3-Fold CV Setup (test_fold={test_fold}):")
    print(f"  CSV Type: {csv_type}")
    print(f"  Train folds: {train_folds} ({len(train_loader.dataset)} samples)")
    print(f"  Test fold: {test_fold} ({len(test_loader.dataset)} samples)")
    print(f"  Variant: {variant}")
    print(f"  Workers: {num_workers}")
    
    return train_loader, val_loader, test_loader


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Test the dataset
    BASE_PATH = Path("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/Nikhil/Histopathology_Work")
    
    csv_path = BASE_PATH / "Dataset/Images_With_Permutations_Labels_Refer_Segmentation_Task.csv"
    images_dir = BASE_PATH / "Dataset/multi_images"
    masks_dir = BASE_PATH / "Dataset/multi_masks"
    
    # Test train mode
    print("\n=== Testing TRAIN mode ===")
    train_dataset = PanNukePermutationDataset(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        folds=[1, 2],
        transform=get_train_transforms(),
        mode='train',
        variant='FULL'
    )
    
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"np_map shape: {sample['np_map'].shape}")
    print(f"Instruction: {sample['instruction']}")
    print(f"Requested classes: {sample['requested_classes']}")
    
    # Test test mode
    print("\n=== Testing TEST mode ===")
    test_dataset = PanNukePermutationDataset(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        folds=[3],
        transform=get_val_transforms(),
        mode='test',
        variant='FULL'
    )
    
    sample = test_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"np_map shape: {sample['np_map'].shape}")
    print(f"Instruction: {sample['instruction']}")
    print(f"Requested classes: {sample['requested_classes']}")
