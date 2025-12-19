"""
Dataset loader for CIPS-Net
Handles multi-class histopathology segmentation with instructions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HistopathologySegmentationDataset(Dataset):
    """
    Dataset for instruction-conditioned histopathology segmentation.
    
    Args:
        data_root: Root directory of dataset
        csv_file: Path to CSV file (unique or permutations)
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        folds: List of folds to include
        img_size: Target image size
        augmentation: Whether to apply augmentations
        use_permutations: Whether using permutations CSV
    """
    
    def __init__(
        self,
        data_root: str,
        csv_file: str,
        images_dir: str = "multi_images",
        masks_dir: str = "multi_masks",
        folds: List[int] = [1],
        img_size: int = 224,
        augmentation: bool = False,
        use_permutations: bool = False
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.images_path = self.data_root / images_dir
        self.masks_path = self.data_root / masks_dir
        self.img_size = img_size
        self.augmentation = augmentation
        self.use_permutations = use_permutations
        
        # Load CSV
        self.df = pd.read_csv(self.data_root / csv_file)
        
        # Extract fold information
        self.df['fold'] = self.df['image_path'].str.extract(r'fold_(\d+)').astype(int)
        
        # Filter by folds
        self.df = self.df[self.df['fold'].isin(folds)].reset_index(drop=True)
        
        # Class names
        self.class_names = [
            'Neoplastic',
            'Inflammatory',
            'Connective_Soft_tissue',
            'Epithelial',
            'Dead'
        ]
        self.num_classes = len(self.class_names)
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.df)} samples from folds {folds}")
    
    def _get_transforms(self):
        """Get augmentation transforms."""
        if self.augmentation:
            return A.Compose([
                A.RandomResizedCrop(self.img_size, self.img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _load_masks(self, image_path: str, classes: str) -> np.ndarray:
        """
        Load all 5 class masks + background for an image.
        
        Args:
            image_path: Path to image file
            classes: Semicolon-separated class names
            
        Returns:
            masks: [H, W, num_classes+1] binary masks
        """
        # Parse image name
        img_name = Path(image_path).stem  # e.g., '1_Breast_fold_1_0000_img'
        parts = img_name.split('_')
        organ_num = parts[0]
        organ = parts[1]
        fold = parts[3]
        img_id = parts[4]
        
        # Parse present classes
        present_classes = set(classes.split(';')) if pd.notna(classes) else set()
        
        # Load masks for each class
        masks = []
        for i, class_name in enumerate(self.class_names):
            # Try channel naming convention
            mask_path_channel = self.masks_path / f"{organ_num}_{organ}_fold_{fold}_{img_id}_channel_{i}_{class_name}.png"
            
            # Try simple naming convention
            mask_path_simple = self.masks_path / f"{organ}_fold_{fold}_{img_id}_{class_name}.png"
            mask_path_empty = self.masks_path / f"{organ}_fold_{fold}_{img_id}_{class_name}_EMPTY.png"
            
            if mask_path_channel.exists():
                mask = cv2.imread(str(mask_path_channel), cv2.IMREAD_GRAYSCALE)
            elif mask_path_simple.exists():
                mask = cv2.imread(str(mask_path_simple), cv2.IMREAD_GRAYSCALE)
            elif mask_path_empty.exists():
                # Empty mask (all zeros)
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                # If no mask found, create empty mask
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # Binarize
            mask = (mask > 127).astype(np.float32)
            masks.append(mask)
        
        # Create background mask (inverse of all other masks)
        foreground = np.maximum.reduce(masks)
        background = 1.0 - foreground
        masks.append(background)
        
        # Stack masks: [num_classes+1, H, W]
        masks = np.stack(masks, axis=0)
        
        return masks
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - image: [3, H, W]
                - masks: [num_classes+1, H, W]
                - instruction: str
                - class_presence: [num_classes]
                - metadata: dict
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.images_path / row['image_path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load masks
        masks = self._load_masks(row['image_path'], row['classes'])
        # masks: [num_classes+1, H, W]
        
        # Apply transforms
        # Albumentations expects masks as [H, W, C], so transpose
        masks_hwc = np.transpose(masks, (1, 2, 0))  # [H, W, num_classes+1]
        
        transformed = self.transform(image=image, mask=masks_hwc)
        image = transformed['image']  # [3, H, W]
        masks_hwc = transformed['mask']  # [H, W, num_classes+1]
        
        # Transpose masks back to [C, H, W]
        masks = torch.from_numpy(masks_hwc).permute(2, 0, 1).float()
        
        # Get instruction
        instruction = row['instruction']
        
        # Get class presence
        present_classes = set(row['classes'].split(';')) if pd.notna(row['classes']) else set()
        class_presence = torch.zeros(self.num_classes)
        for i, class_name in enumerate(self.class_names):
            if class_name in present_classes:
                class_presence[i] = 1.0
        
        # Metadata
        metadata = {
            'image_id': row['image_id'],
            'image_path': row['image_path'],
            'organ': row['organ'],
            'fold': row['fold'],
            'classes': row['classes']
        }
        
        return {
            'image': image,
            'masks': masks,
            'instruction': instruction,
            'class_presence': class_presence,
            'metadata': metadata
        }


def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    train_folds: List[int] = [1, 2],
    val_folds: List[int] = [3],
    use_permutations: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
        train_folds: Folds for training
        val_folds: Folds for validation
        use_permutations: Whether to use permutations CSV
        
    Returns:
        train_loader, val_loader
    """
    # CSV file
    csv_file = "Images_With_Permutations_Labels_Refer_Segmentation_Task.csv" if use_permutations \
               else "Images_With_Unique_Labels_Refer_Segmentation_Task.csv"
    
    # Training dataset
    train_dataset = HistopathologySegmentationDataset(
        data_root=data_root,
        csv_file=csv_file,
        folds=train_folds,
        img_size=img_size,
        augmentation=True,
        use_permutations=use_permutations
    )
    
    # Validation dataset
    val_dataset = HistopathologySegmentationDataset(
        data_root=data_root,
        csv_file=csv_file,
        folds=val_folds,
        img_size=img_size,
        augmentation=False,
        use_permutations=False  # Always use unique for validation
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    dataset = HistopathologySegmentationDataset(
        data_root="c:/Users/nikhi/Desktop/Histopathology_Work/Dataset",
        csv_file="Images_With_Unique_Labels_Refer_Segmentation_Task.csv",
        folds=[1],
        img_size=224,
        augmentation=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Masks shape: {sample['masks'].shape}")
    print(f"Instruction: {sample['instruction']}")
    print(f"Class presence: {sample['class_presence']}")
    print(f"Metadata: {sample['metadata']}")
    
    # Test dataloader
    train_loader, val_loader = get_dataloaders(
        data_root="c:/Users/nikhi/Desktop/Histopathology_Work/Dataset",
        batch_size=4,
        num_workers=0,
        train_folds=[1],
        val_folds=[2]
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch image shape: {batch['image'].shape}")
    print(f"Batch masks shape: {batch['masks'].shape}")
    print(f"Batch instructions: {len(batch['instruction'])}")
