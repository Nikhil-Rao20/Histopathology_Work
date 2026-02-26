"""
PanNuke Dataset - Memory Optimized Version
============================================

Key optimizations:
1. Lazy loading with memory-mapped files
2. No caching of full arrays in RAM
3. Efficient on-the-fly target computation
4. Minimal memory footprint per worker
5. Proper cleanup and garbage collection

This version is designed to work on systems with limited RAM.
"""

import os
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import cv2
import random
from scipy.ndimage import distance_transform_edt as scipy_edt

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Constants
# ============================================================

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

DEFAULT_INSTRUCTIONS = [
    "Segment all nuclei in this histopathology image.",
    "Identify and segment all cell nuclei present.",
    "Detect all nuclear instances in this tissue sample.",
]


# ============================================================
# HV Map Generation (Optimized)
# ============================================================

def gen_instance_hv_map(instance_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate horizontal and vertical distance maps from instance mask.
    Memory efficient version.
    """
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


def gen_normalized_distance_transform(instance_map: np.ndarray) -> np.ndarray:
    """
    Generate normalized Euclidean distance transform for instance segmentation.
    
    For each instance, computes the EDT (distance from boundary) and normalizes
    to [0, 1] per-instance (1.0 at center, 0.0 at boundary).
    
    Used by the Instance Embedding (IE) decoder as replacement for HV maps.
    
    Args:
        instance_map: [H, W] integer instance IDs (0 = background)
    
    Returns:
        dist_map: [H, W] normalized EDT, float32 in [0, 1]
    """
    H, W = instance_map.shape
    dist_map = np.zeros((H, W), dtype=np.float32)
    
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]
    
    for inst_id in instance_ids:
        inst_mask = (instance_map == inst_id)
        
        # EDT: distance from each interior pixel to nearest boundary pixel
        edt = scipy_edt(inst_mask)
        
        # Normalize per instance to [0, 1]
        max_val = edt.max()
        if max_val > 0:
            dist_map[inst_mask] = edt[inst_mask] / max_val
        else:
            # Single-pixel instance: set to 1.0
            dist_map[inst_mask] = 1.0
    
    return dist_map


def masks_to_instance_and_type(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PanNuke 6-channel masks to instance and type maps."""
    H, W = masks.shape[:2]
    instance_map = np.zeros((H, W), dtype=np.int32)
    type_map = np.zeros((H, W), dtype=np.int32)
    
    inst_id = 1
    for class_idx in range(5):
        class_mask = masks[:, :, class_idx]
        unique_ids = np.unique(class_mask)
        unique_ids = unique_ids[unique_ids != 0]
        
        for uid in unique_ids:
            inst_mask = (class_mask == uid)
            instance_map[inst_mask] = inst_id
            type_map[inst_mask] = class_idx + 1
            inst_id += 1
    
    return instance_map, type_map


def prepare_hover_targets(masks: np.ndarray) -> Dict[str, np.ndarray]:
    """Prepare HoVer-Net targets - optimized to avoid unnecessary copies.
    
    Also computes normalized distance transform (used by IE decoder variant).
    """
    instance_map, type_map = masks_to_instance_and_type(masks)
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
# Memory-Efficient Dataset (Lazy Loading)
# ============================================================

class PanNukeOptimizedDataset(Dataset):
    """
    Memory-optimized PanNuke dataset using lazy loading.
    
    Key features:
    - Uses memory-mapped numpy arrays (no RAM caching)
    - Loads only what's needed per sample
    - Minimal metadata storage
    - Proper cleanup after each sample
    - Optional dataset expansion via augmentation (stain jitter, elastic, noise)
    """
    
    def __init__(
        self,
        data_root: str,
        folds: List[int] = [1, 2, 3],
        transform: Optional[Callable] = None,
        mode: str = 'baseline',  # 'baseline', 'single_text', 'multi_text'
        augmentation_config=None,  # AugmentationConfig for dataset expansion (train only)
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Path to PanNuke root directory
            folds: List of fold numbers [1, 2, 3]
            transform: Albumentations transform
            mode: Text conditioning mode
            augmentation_config: AugmentationConfig for dataset expansion (None = no expansion)
        """
        self.data_root = Path(data_root)
        self.folds = sorted(folds)
        self.transform = transform
        self.mode = mode
        
        # Build index: (fold, local_idx) for each sample
        # Only store metadata, not actual data
        self.index = []
        self.tissue_types = []
        
        # Store file paths for lazy loading
        self.fold_paths = {}
        
        for fold in self.folds:
            fold_path = self.data_root / f"Fold {fold}"
            
            # Store paths (not data)
            self.fold_paths[fold] = {
                'images': fold_path / f"images/fold{fold}/images.npy",
                'masks': fold_path / f"masks/fold{fold}/masks.npy",
                'types': fold_path / f"images/fold{fold}/types.npy"
            }
            
            # Load only types (small array) to know sample count
            types = np.load(self.fold_paths[fold]['types'])
            n_samples = len(types)
            
            # Build index
            for local_idx in range(n_samples):
                self.index.append((fold, local_idx))
                self.tissue_types.append(types[local_idx])
            
            # Immediately free types array
            del types
        
        # Text generation (minimal storage)
        if mode != 'baseline':
            self._setup_text_templates()
        
        # Data augmentation expansion (only affects training)
        self.aug_config = augmentation_config
        self._original_len = len(self.index)
        
        if self.aug_config and self.aug_config.enabled:
            factor = self.aug_config.expansion_factor
            print(f"PanNukeOptimized: {self._original_len} original × {factor} = "
                  f"{len(self)} samples from folds {folds} "
                  f"(augmentations: {self.aug_config.augmentation_names})")
        else:
            print(f"PanNukeOptimized: {len(self)} samples from folds {folds} (lazy loading)")
    
    def _setup_text_templates(self):
        """Setup text templates for instruction generation."""
        self.class_templates = {
            cls: [
                f"Segment all {cls} nuclei.",
                f"Identify {cls} cells in this image.",
                f"Detect {cls} nuclear instances.",
            ] for cls in PANNUKE_CLASS_NAMES
        }
        self.class_templates['all'] = DEFAULT_INSTRUCTIONS
    
    def __len__(self) -> int:
        if self.aug_config and self.aug_config.enabled:
            return self._original_len * self.aug_config.expansion_factor
        return self._original_len
    
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
        """
        Load a single sample using memory mapping.
        
        Memory-mapped loading means only the requested slice is loaded into RAM.
        """
        fold, local_idx = self.index[idx]
        
        # Memory-mapped loading (only loads requested indices)
        images_mmap = np.load(self.fold_paths[fold]['images'], mmap_mode='r')
        masks_mmap = np.load(self.fold_paths[fold]['masks'], mmap_mode='r')
        
        # Copy the specific sample we need (necessary to avoid issues with transforms)
        image = images_mmap[local_idx].copy()
        mask = masks_mmap[local_idx].copy()
        
        # Clean up memory maps immediately
        del images_mmap, masks_mmap
        
        tissue = self.tissue_types[idx]
        
        return image, mask, tissue, fold, local_idx
    
    def _get_classes_from_mask(self, mask: np.ndarray) -> List[str]:
        """Get present classes from mask."""
        classes = []
        for i, cls_name in enumerate(PANNUKE_CLASS_NAMES):
            if np.any(mask[:, :, i] > 0):
                classes.append(cls_name)
        return classes
    
    def _generate_instruction(self, classes: List[str]) -> str:
        """Generate instruction from present classes."""
        if not classes:
            return random.choice(DEFAULT_INSTRUCTIONS)
        
        if self.mode == 'single_text':
            # Pick one class randomly for single text mode
            cls = random.choice(classes)
            return random.choice(self.class_templates[cls])
        else:  # multi_text
            # All classes instruction
            return random.choice(self.class_templates['all'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with lazy loading."""
        # Resolve expansion index (maps augmented idx → original idx + aug type)
        from .augmentation import resolve_expansion_index, apply_expansion_augmentation
        orig_idx, aug_type = resolve_expansion_index(idx, self._original_len, self.aug_config)
        
        # Load raw data
        image, mask, tissue, fold, local_idx = self._load_sample(orig_idx)
        
        # Convert image dtype if needed
        if image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Apply expansion augmentation BEFORE HoVer targets and transforms
        if aug_type is not None:
            image, mask = apply_expansion_augmentation(
                image, mask, aug_type, self.aug_config, orig_idx,
            )
        
        # Prepare HoVer targets (computed on-the-fly)
        targets = prepare_hover_targets(mask)
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
            'tissue': tissue,
            'fold': fold,
            'idx': idx,
            'local_idx': local_idx
        }
        
        # Add instruction if not baseline
        if self.mode != 'baseline':
            classes = self._get_classes_from_mask(mask)
            result['instruction'] = self._generate_instruction(classes)
        
        # Clean up intermediate arrays
        del mask, targets
        
        return result


# ============================================================
# Collate Functions
# ============================================================

def optimized_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Memory-efficient collate function."""
    images = torch.stack([s['image'] for s in batch])
    np_maps = torch.stack([s['np_map'] for s in batch])
    hv_maps = torch.stack([s['hv_map'] for s in batch])
    type_maps = torch.stack([s['type_map'] for s in batch])
    instance_maps = torch.stack([s['instance_map'] for s in batch])
    dist_maps = torch.stack([s['dist_map'] for s in batch])
    
    result = {
        'images': images,
        'np_maps': np_maps,
        'hv_maps': hv_maps,
        'type_maps': type_maps,
        'instance_maps': instance_maps,
        'dist_maps': dist_maps,
        'tissues': [s['tissue'] for s in batch],
        'folds': [s['fold'] for s in batch],
        'indices': [s['idx'] for s in batch],
    }
    
    # Add instructions if present
    if 'instruction' in batch[0]:
        result['instructions'] = [s['instruction'] for s in batch]
    
    return result


# ============================================================
# Dataloader Factory
# ============================================================

def create_optimized_dataloaders(
    data_root: str,
    mode: str = 'baseline',
    train_folds: List[int] = [1, 2],
    val_folds: List[int] = [3],
    batch_size: int = 8,
    num_workers: int = 0,  # Default to 0 for memory safety
    pin_memory: bool = False,  # Disable by default to save memory
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    augmentation_config=None,  # AugmentationConfig for train dataset expansion
) -> Tuple[DataLoader, DataLoader]:
    """
    Create memory-optimized dataloaders.
    
    Args:
        data_root: Path to PanNuke
        mode: 'baseline', 'single_text', 'multi_text'
        train_folds: Training folds
        val_folds: Validation folds
        batch_size: Batch size
        num_workers: Number of workers (0 = main process only)
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Prefetch factor per worker
        persistent_workers: Keep workers alive between epochs
        augmentation_config: AugmentationConfig for dataset expansion (train only)
    """
    train_dataset = PanNukeOptimizedDataset(
        data_root=data_root,
        folds=train_folds,
        transform=get_train_transforms(),
        mode=mode,
        augmentation_config=augmentation_config,  # Expansion augmentation (train only)
    )
    
    val_dataset = PanNukeOptimizedDataset(
        data_root=data_root,
        folds=val_folds,
        transform=get_val_transforms(),
        mode=mode,
        # NOTE: No augmentation_config for val — never expand val/test
    )
    
    # Dataloader kwargs
    common_kwargs = {
        'collate_fn': optimized_collate_fn,
        'pin_memory': pin_memory,
    }
    
    # Only add these if using workers
    if num_workers > 0:
        common_kwargs['prefetch_factor'] = prefetch_factor
        common_kwargs['persistent_workers'] = persistent_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        **common_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **common_kwargs
    )
    
    return train_loader, val_loader


def create_optimized_fold_dataloaders(
    data_root: str,
    mode: str = 'baseline',
    test_fold: int = 1,
    batch_size: int = 8,
    num_workers: int = 0,
    augmentation_config=None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for 3-fold cross-validation (memory optimized).
    
    Args:
        data_root: Path to PanNuke
        mode: Dataset mode
        test_fold: Which fold to use as test set (1, 2, or 3)
        batch_size: Batch size
        num_workers: Number of workers
        augmentation_config: AugmentationConfig for train dataset expansion
        
    Returns:
        train_loader, val_loader, test_loader
    """
    all_folds = [1, 2, 3]
    train_folds = [f for f in all_folds if f != test_fold]
    
    train_loader, val_loader = create_optimized_dataloaders(
        data_root=data_root,
        mode=mode,
        train_folds=train_folds,
        val_folds=[test_fold],
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
        **kwargs
    )
    
    # Test loader is same as val loader for this fold
    test_loader = val_loader
    
    print(f"\n3-Fold CV Setup (test_fold={test_fold}):")
    print(f"  Train folds: {train_folds} ({len(train_loader.dataset)} samples)")
    print(f"  Test fold: {test_fold} ({len(test_loader.dataset)} samples)")
    print(f"  Workers: {num_workers}, Pin memory: {kwargs.get('pin_memory', False)}")
    
    return train_loader, val_loader, test_loader


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    import argparse
    import psutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--mode', type=str, default='baseline')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Testing Memory-Optimized PanNuke DataLoader")
    print("="*60)
    
    # Memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2
    print(f"Memory before loading: {mem_before:.1f} MB")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_optimized_fold_dataloaders(
        data_root=args.data_root,
        mode=args.mode,
        test_fold=1,
        batch_size=4,
        num_workers=0
    )
    
    mem_after_init = process.memory_info().rss / 1024**2
    print(f"Memory after init: {mem_after_init:.1f} MB (+{mem_after_init - mem_before:.1f} MB)")
    
    # Test batch loading
    print("\nLoading first batch...")
    batch = next(iter(train_loader))
    
    mem_after_batch = process.memory_info().rss / 1024**2
    print(f"Memory after batch: {mem_after_batch:.1f} MB (+{mem_after_batch - mem_after_init:.1f} MB)")
    
    print(f"\nBatch contents:")
    print(f"  images: {batch['images'].shape}")
    print(f"  np_maps: {batch['np_maps'].shape}")
    print(f"  hv_maps: {batch['hv_maps'].shape}")
    print(f"  type_maps: {batch['type_maps'].shape}")
    
    # Clean up
    del batch
    gc.collect()
    
    mem_after_cleanup = process.memory_info().rss / 1024**2
    print(f"\nMemory after cleanup: {mem_after_cleanup:.1f} MB")
    
    print("\n✓ Memory-optimized dataloader test passed!")
