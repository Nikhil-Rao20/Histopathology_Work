"""
Dataset utilities for text-guided segmentation models on histopathology data.
Supports PanNuke, CoNSeP, and MoNuSAC datasets with text prompt generation.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from typing import Dict, List, Tuple, Optional, Union
import cv2


# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

# PanNuke classes (5 classes + background)
PANNUKE_CLASSES = {
    0: "neoplastic",
    1: "inflammatory", 
    2: "connective",
    3: "dead",
    4: "epithelial"
}

PANNUKE_CLASS_NAMES = ["neoplastic cells", "inflammatory cells", "connective tissue cells", 
                       "dead cells", "epithelial cells"]

# CoNSeP classes mapping to PanNuke
CONSEP_TO_PANNUKE = {
    1: 4,  # epithelial -> epithelial
    2: 4,  # epithelial -> epithelial  
    3: 1,  # inflammatory -> inflammatory
    4: 1,  # inflammatory -> inflammatory
    5: 2,  # spindle/connective -> connective
    6: 2,  # spindle/connective -> connective
    7: 4,  # epithelial -> epithelial
}

# MoNuSAC classes mapping to PanNuke
MONUSAC_TO_PANNUKE = {
    1: 4,  # epithelial -> epithelial
    2: 1,  # lymphocyte -> inflammatory
    3: 1,  # macrophage -> inflammatory  
    4: 1,  # neutrophil -> inflammatory
}


# =============================================================================
# TEXT PROMPT TEMPLATES
# =============================================================================

def get_prompt_templates():
    """Returns various prompt templates for text-guided segmentation."""
    return {
        "simple": "a photo of {}",
        "detailed": "a histopathology image showing {} in tissue",
        "medical": "microscopy image of {} nuclei in pathology slide",
        "descriptive": "segmentation of {} in histopathology",
        "cell_specific": "{} visible in tissue sample",
        "plural": "multiple {} present in the image",
    }


def generate_text_prompts(class_names: List[str], template: str = "simple") -> List[str]:
    """
    Generate text prompts for each class using a template.
    
    Args:
        class_names: List of class names
        template: Template type from get_prompt_templates()
    
    Returns:
        List of formatted prompts
    """
    templates = get_prompt_templates()
    template_str = templates.get(template, templates["simple"])
    return [template_str.format(name) for name in class_names]


def get_class_presence_prompts(present_classes: List[int], 
                                class_names: List[str],
                                template: str = "simple") -> Tuple[List[str], List[int]]:
    """
    Generate prompts only for classes present in an image.
    
    Args:
        present_classes: List of class indices present in the image
        class_names: Full list of class names
        template: Template type
        
    Returns:
        Tuple of (prompts for present classes, indices of present classes)
    """
    templates = get_prompt_templates()
    template_str = templates.get(template, templates["simple"])
    prompts = [template_str.format(class_names[idx]) for idx in present_classes]
    return prompts, present_classes


# =============================================================================
# BASE DATASET CLASS
# =============================================================================

class BaseHistopathologyDataset(Dataset):
    """Base dataset class for histopathology segmentation datasets."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform = None,
        img_size: int = 256,
        num_classes: int = 5,
        class_names: List[str] = None,
        prompt_template: str = "simple",
        return_class_presence: bool = False,
        normalize: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names or PANNUKE_CLASS_NAMES
        self.prompt_template = prompt_template
        self.return_class_presence = return_class_presence
        self.normalize = normalize
        
        # Default transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        # Normalization (CLIP-compatible)
        self.normalize_transform = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        # Generate text prompts
        self.text_prompts = generate_text_prompts(self.class_names, self.prompt_template)
        
    def _get_default_transform(self):
        """Get default transformation pipeline."""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
    
    def _transform_pair(self, image: Image.Image, mask: np.ndarray):
        """Apply synchronized transforms to image and mask."""
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Data augmentation for training
        if self.split == "train":
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = np.fliplr(mask).copy()
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = np.flipud(mask).copy()
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = np.rot90(mask, k=angle // 90).copy()
        
        # Convert to tensor
        image = TF.to_tensor(image)
        
        # Normalize if specified
        if self.normalize:
            image = self.normalize_transform(image)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def get_class_presence(self, mask: np.ndarray) -> List[int]:
        """Get list of class indices present in the mask."""
        unique_classes = np.unique(mask)
        # Filter out background (if 0 or -1)
        present_classes = [int(c) for c in unique_classes if c >= 0 and c < self.num_classes]
        return present_classes
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError


# =============================================================================
# PANNUKE DATASET
# =============================================================================

class PanNukeDataset(BaseHistopathologyDataset):
    """
    PanNuke dataset for text-guided segmentation.
    
    Expected directory structure:
    root_dir/
        Images_With_Unique_Labels_Refer_Segmentation_Task.csv (or _Small.csv)
        multi_images/
            image_001.png
            ...
        multi_masks/
            mask_001.png
            ...
    """
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str = "Images_With_Unique_Labels_Refer_Segmentation_Task.csv",
        fold: int = 0,
        split: str = "train",
        use_small: bool = False,
        **kwargs
    ):
        super().__init__(root_dir, split, **kwargs)
        
        self.fold = fold
        
        # Load CSV
        if use_small:
            csv_file = csv_file.replace(".csv", "_Small.csv")
        csv_path = os.path.join(root_dir, csv_file)
        
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            # Try to find images directly
            self.df = self._create_df_from_images()
        
        # Split by fold (3-fold CV)
        self._setup_fold_split()
        
        # Paths
        self.image_dir = os.path.join(root_dir, "multi_images")
        self.mask_dir = os.path.join(root_dir, "multi_masks")
        
    def _create_df_from_images(self):
        """Create dataframe from image directory if CSV not found."""
        image_dir = os.path.join(self.root_dir, "multi_images")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        return pd.DataFrame({'image': images})
    
    def _setup_fold_split(self):
        """Setup 3-fold cross-validation split."""
        n_samples = len(self.df)
        indices = np.arange(n_samples)
        
        # 3-fold split
        fold_size = n_samples // 3
        folds = [
            indices[:fold_size],
            indices[fold_size:2*fold_size],
            indices[2*fold_size:]
        ]
        
        if self.split == "train":
            # Use 2 folds for training
            train_folds = [i for i in range(3) if i != self.fold]
            self.indices = np.concatenate([folds[f] for f in train_folds])
        else:  # val/test
            # Use 1 fold for validation
            self.indices = folds[self.fold]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        row = self.df.iloc[actual_idx]
        
        # Load image
        if 'image' in row:
            img_name = row['image']
        else:
            img_name = f"image_{actual_idx:04d}.png"
        
        img_path = os.path.join(self.image_dir, img_name)
        
        # Handle mask path
        mask_name = img_name.replace('image_', 'mask_').replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
        else:
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.int64)
        
        # Ensure mask has correct range (0-4 for 5 classes)
        if mask.max() > self.num_classes:
            # Remap if needed
            mask = mask % (self.num_classes + 1)
        
        # Apply transforms
        image, mask = self._transform_pair(image, mask)
        
        # Prepare output
        output = {
            'image': image,
            'mask': mask,
            'text_prompts': self.text_prompts,
            'image_path': img_path,
        }
        
        # Add class presence info if requested
        if self.return_class_presence:
            present_classes = self.get_class_presence(mask.numpy())
            output['present_classes'] = present_classes
            output['present_prompts'], _ = get_class_presence_prompts(
                present_classes, self.class_names, self.prompt_template
            )
        
        return output


# =============================================================================
# CONSEP DATASET
# =============================================================================

class CoNSePDataset(BaseHistopathologyDataset):
    """
    CoNSeP dataset for text-guided segmentation.
    Maps CoNSeP classes to PanNuke classes for zero-shot evaluation.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        class_mapping: Dict[int, int] = None,
        **kwargs
    ):
        super().__init__(root_dir, split, **kwargs)
        
        self.class_mapping = class_mapping or CONSEP_TO_PANNUKE
        
        # Find images
        self.image_dir = os.path.join(root_dir, "Images")
        self.label_dir = os.path.join(root_dir, "Labels")
        
        if os.path.exists(self.image_dir):
            self.images = sorted([f for f in os.listdir(self.image_dir) 
                                 if f.endswith(('.png', '.jpg', '.tif'))])
        else:
            self.images = []
    
    def _map_labels(self, mask: np.ndarray) -> np.ndarray:
        """Map CoNSeP labels to PanNuke labels."""
        mapped_mask = np.zeros_like(mask)
        for consep_class, pannuke_class in self.class_mapping.items():
            mapped_mask[mask == consep_class] = pannuke_class
        return mapped_mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (try different extensions)
        mask_name = img_name.replace('.png', '.mat').replace('.jpg', '.mat').replace('.tif', '.mat')
        mask_path = os.path.join(self.label_dir, mask_name)
        
        if os.path.exists(mask_path):
            from scipy.io import loadmat
            mat = loadmat(mask_path)
            # CoNSeP stores type map in 'type_map' or 'inst_type'
            if 'type_map' in mat:
                mask = mat['type_map']
            elif 'inst_type' in mat:
                mask = mat['inst_type']
            else:
                mask = np.zeros((image.size[1], image.size[0]), dtype=np.int64)
            mask = self._map_labels(mask)
        else:
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.int64)
        
        # Apply transforms
        image, mask = self._transform_pair(image, mask)
        
        output = {
            'image': image,
            'mask': mask,
            'text_prompts': self.text_prompts,
            'image_path': img_path,
        }
        
        if self.return_class_presence:
            present_classes = self.get_class_presence(mask.numpy())
            output['present_classes'] = present_classes
            output['present_prompts'], _ = get_class_presence_prompts(
                present_classes, self.class_names, self.prompt_template
            )
        
        return output


# =============================================================================
# MONUSAC DATASET  
# =============================================================================

class MoNuSACDataset(BaseHistopathologyDataset):
    """
    MoNuSAC dataset for text-guided segmentation.
    Maps MoNuSAC classes to PanNuke classes for zero-shot evaluation.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        class_mapping: Dict[int, int] = None,
        **kwargs
    ):
        super().__init__(root_dir, split, **kwargs)
        
        self.class_mapping = class_mapping or MONUSAC_TO_PANNUKE
        
        # Find images
        self.images = []
        self.masks = []
        
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith(('.png', '.jpg', '.tif', '.svs')):
                    if 'mask' not in f.lower() and 'label' not in f.lower():
                        self.images.append(os.path.join(root, f))
    
    def _map_labels(self, mask: np.ndarray) -> np.ndarray:
        """Map MoNuSAC labels to PanNuke labels."""
        mapped_mask = np.zeros_like(mask)
        for monusac_class, pannuke_class in self.class_mapping.items():
            mapped_mask[mask == monusac_class] = pannuke_class
        return mapped_mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Find corresponding mask
        mask_path = img_path.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
        if not os.path.exists(mask_path):
            mask_path = img_path.replace('/images/', '/masks/')
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            mask = self._map_labels(mask)
        else:
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.int64)
        
        # Apply transforms
        image, mask = self._transform_pair(image, mask)
        
        output = {
            'image': image,
            'mask': mask,
            'text_prompts': self.text_prompts,
            'image_path': img_path,
        }
        
        if self.return_class_presence:
            present_classes = self.get_class_presence(mask.numpy())
            output['present_classes'] = present_classes
            output['present_prompts'], _ = get_class_presence_prompts(
                present_classes, self.class_names, self.prompt_template
            )
        
        return output


# =============================================================================
# DATALOADER UTILITIES
# =============================================================================

def get_dataloader(
    dataset_name: str,
    root_dir: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Get a dataloader for the specified dataset.
    
    Args:
        dataset_name: One of 'pannuke', 'consep', 'monusac'
        root_dir: Path to dataset root directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader instance
    """
    datasets = {
        'pannuke': PanNukeDataset,
        'consep': CoNSePDataset,
        'monusac': MoNuSACDataset,
    }
    
    if dataset_name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(datasets.keys())}")
    
    dataset_class = datasets[dataset_name.lower()]
    dataset = dataset_class(root_dir=root_dir, split=split, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def collate_fn(batch):
    """Custom collate function for text-guided segmentation datasets."""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    # Text prompts are the same for all samples
    text_prompts = batch[0]['text_prompts']
    
    output = {
        'image': images,
        'mask': masks,
        'text_prompts': text_prompts,
    }
    
    # Handle class presence if present
    if 'present_classes' in batch[0]:
        output['present_classes'] = [item['present_classes'] for item in batch]
        output['present_prompts'] = [item['present_prompts'] for item in batch]
    
    return output


# =============================================================================
# 3-FOLD CROSS-VALIDATION UTILITIES
# =============================================================================

PANNUKE_CLASS_PROMPTS = [
    "neoplastic cells",
    "inflammatory cells",
    "connective tissue cells",
    "dead cells",
    "epithelial cells",
]


def get_pannuke_3fold_splits(
    csv_path: str,
    fold: int = 0,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get train/val/test splits for 3-fold cross-validation on PanNuke.
    
    PanNuke is naturally divided into 3 folds. This function provides
    consistent splits for reproducible experiments.
    
    Args:
        csv_path: Path to CSV file with image information
        fold: Fold number (0, 1, or 2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    if fold not in [0, 1, 2]:
        raise ValueError(f"Fold must be 0, 1, or 2. Got {fold}")
    
    # Read CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        all_ids = df['image_id'].tolist() if 'image_id' in df.columns else df.iloc[:, 0].tolist()
    else:
        # Generate dummy IDs if CSV doesn't exist
        all_ids = list(range(1000))
    
    # Shuffle with seed
    np.random.seed(seed)
    np.random.shuffle(all_ids)
    
    # Split into 3 folds
    n = len(all_ids)
    fold_size = n // 3
    
    folds = [
        all_ids[0:fold_size],
        all_ids[fold_size:2*fold_size],
        all_ids[2*fold_size:],
    ]
    
    # Assign folds based on fold number
    # Fold 0: train on folds 1,2, val on fold 0
    # Fold 1: train on folds 0,2, val on fold 1
    # Fold 2: train on folds 0,1, val on fold 2
    test_fold = folds[fold]
    train_folds = [folds[(fold + 1) % 3], folds[(fold + 2) % 3]]
    train_ids = train_folds[0] + train_folds[1]
    
    # Split train into train/val (80/20)
    val_size = len(train_ids) // 5
    val_ids = train_ids[:val_size]
    train_ids = train_ids[val_size:]
    
    return train_ids, val_ids, test_fold


def get_clip_transform(image_size: int = 256) -> transforms.Compose:
    """
    Get CLIP-compatible image transform.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    # CLIP normalization values
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])
