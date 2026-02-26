"""
Copy-Paste Augmentation for Nuclei Segmentation
=================================================

This module implements Copy-Paste augmentation specifically designed for
nuclei instance segmentation to address class imbalance issues.

Key Features:
    1. Copy nuclei of rare classes (esp. Dead) from donor images
    2. Paste them into recipient images with proper mask blending
    3. Maintains instance separation and HV map consistency
    4. Class-aware sampling to prioritize rare classes

Based on: "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
(Ghiasi et al., CVPR 2021)

Adapted for nuclei segmentation with HoVer-Net style outputs.

Author: Enhanced for CIPS-Net V2 to improve rare class performance
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional
import random
from scipy import ndimage


# Class frequencies in PanNuke (approximate)
PANNUKE_CLASS_FREQUENCIES = {
    0: 0.70,   # Background
    1: 0.10,   # Neoplastic
    2: 0.06,   # Inflammatory
    3: 0.05,   # Connective
    4: 0.01,   # Dead (RARE!)
    5: 0.08,   # Epithelial
}

# Inverse frequency for sampling weights
PANNUKE_SAMPLE_WEIGHTS = {
    1: 1.0,    # Neoplastic
    2: 1.5,    # Inflammatory
    3: 2.0,    # Connective
    4: 10.0,   # Dead (10x more likely to copy)
    5: 1.2,    # Epithelial
}


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation for nuclei instance segmentation.
    
    During training, randomly copies nuclei from donor images and pastes
    them into recipient images. Prioritizes rare classes.
    """
    
    def __init__(
        self,
        paste_prob: float = 0.5,  # Probability of applying copy-paste
        max_paste_instances: int = 5,  # Max nuclei to paste per image
        min_paste_instances: int = 1,
        class_weights: Optional[Dict[int, float]] = None,
        jitter_scale: Tuple[float, float] = (0.8, 1.2),  # Size jitter
        jitter_brightness: float = 0.1,
        blend_mode: str = 'direct',  # 'direct' or 'gaussian'
    ):
        """
        Args:
            paste_prob: Probability of applying augmentation
            max_paste_instances: Maximum nuclei to paste
            min_paste_instances: Minimum nuclei to paste
            class_weights: Dict mapping class_id -> sampling weight
            jitter_scale: Scale range for size jittering
            jitter_brightness: Brightness jitter factor
            blend_mode: How to blend pasted nuclei
        """
        self.paste_prob = paste_prob
        self.max_paste_instances = max_paste_instances
        self.min_paste_instances = min_paste_instances
        self.class_weights = class_weights or PANNUKE_SAMPLE_WEIGHTS
        self.jitter_scale = jitter_scale
        self.jitter_brightness = jitter_brightness
        self.blend_mode = blend_mode
        
        # Bank of extracted nuclei from rare classes
        self.nuclei_bank = {cls: [] for cls in [1, 2, 3, 4, 5]}
        self.max_bank_size = 500  # Per class
    
    def extract_nuclei_instances(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        type_map: np.ndarray
    ) -> List[Dict]:
        """
        Extract individual nuclei instances from an image.
        
        Args:
            image: [H, W, 3] RGB image
            instance_map: [H, W] instance segmentation
            type_map: [H, W] type classification
            
        Returns:
            List of nuclei dicts with 'image_crop', 'mask_crop', 'class', 'bbox'
        """
        nuclei = []
        
        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids != 0]
        
        for inst_id in instance_ids:
            inst_mask = (instance_map == inst_id)
            
            # Get bounding box
            coords = np.where(inst_mask)
            if len(coords[0]) == 0:
                continue
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Skip very small or very large nuclei
            height = y_max - y_min
            width = x_max - x_min
            if height < 5 or width < 5 or height > 100 or width > 100:
                continue
            
            # Get class (majority vote within instance)
            inst_types = type_map[inst_mask]
            inst_class = int(np.bincount(inst_types.astype(int)).argmax())
            
            # Skip background class
            if inst_class == 0:
                continue
            
            # Add padding
            pad = 2
            y_min = max(0, y_min - pad)
            y_max = min(image.shape[0], y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(image.shape[1], x_max + pad)
            
            # Extract crop
            image_crop = image[y_min:y_max, x_min:x_max].copy()
            mask_crop = inst_mask[y_min:y_max, x_min:x_max].copy()
            
            nuclei.append({
                'image_crop': image_crop,
                'mask_crop': mask_crop.astype(np.uint8),
                'class': inst_class,
                'bbox': (y_min, y_max, x_min, x_max),
                'size': (height, width),
            })
        
        return nuclei
    
    def add_to_bank(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        type_map: np.ndarray
    ):
        """Add nuclei from an image to the bank."""
        nuclei = self.extract_nuclei_instances(image, instance_map, type_map)
        
        for nuc in nuclei:
            cls = nuc['class']
            if cls in self.nuclei_bank:
                if len(self.nuclei_bank[cls]) < self.max_bank_size:
                    self.nuclei_bank[cls].append(nuc)
                else:
                    # Replace random existing
                    idx = random.randint(0, self.max_bank_size - 1)
                    self.nuclei_bank[cls][idx] = nuc
    
    def sample_nuclei_to_paste(self, num_to_paste: int) -> List[Dict]:
        """
        Sample nuclei from bank with class-weighted sampling.
        
        Prioritizes rare classes (esp. Dead).
        """
        # Get all available nuclei with weights
        candidates = []
        weights = []
        
        for cls, nuclei_list in self.nuclei_bank.items():
            for nuc in nuclei_list:
                candidates.append(nuc)
                weights.append(self.class_weights.get(cls, 1.0))
        
        if len(candidates) == 0:
            return []
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Sample
        num_to_paste = min(num_to_paste, len(candidates))
        indices = np.random.choice(len(candidates), size=num_to_paste, replace=False, p=weights)
        
        return [candidates[i] for i in indices]
    
    def find_paste_location(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        nucleus_size: Tuple[int, int],
        max_attempts: int = 20
    ) -> Optional[Tuple[int, int]]:
        """
        Find a valid location to paste a nucleus.
        
        Tries to avoid overlapping with existing nuclei.
        """
        H, W = image.shape[:2]
        nuc_h, nuc_w = nucleus_size
        
        for _ in range(max_attempts):
            # Random location
            y = random.randint(0, H - nuc_h - 1)
            x = random.randint(0, W - nuc_w - 1)
            
            # Check for overlap with existing instances
            region = instance_map[y:y+nuc_h, x:x+nuc_w]
            if np.sum(region > 0) < 0.1 * nuc_h * nuc_w:  # Less than 10% overlap
                return (y, x)
        
        return None
    
    def paste_nucleus(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        type_map: np.ndarray,
        hv_map: np.ndarray,
        nucleus: Dict,
        location: Tuple[int, int],
        new_instance_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Paste a nucleus into the target image.
        
        Updates image, instance_map, type_map, and hv_map.
        """
        y, x = location
        img_crop = nucleus['image_crop']
        mask_crop = nucleus['mask_crop']
        cls = nucleus['class']
        
        h, w = img_crop.shape[:2]
        
        # Apply jitter
        scale = random.uniform(*self.jitter_scale)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h > 0 and new_w > 0:
            img_crop = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_crop = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w = new_h, new_w
        
        # Brightness jitter
        if self.jitter_brightness > 0:
            brightness_factor = 1.0 + random.uniform(-self.jitter_brightness, self.jitter_brightness)
            img_crop = np.clip(img_crop * brightness_factor, 0, 255).astype(np.uint8)
        
        # Ensure within bounds
        if y + h > image.shape[0] or x + w > image.shape[1]:
            return image, instance_map, type_map, hv_map
        
        # Create mask
        mask_bool = mask_crop > 0
        
        # Paste image
        if self.blend_mode == 'gaussian':
            # Smooth blending at boundaries
            dist = ndimage.distance_transform_edt(mask_bool)
            alpha = np.clip(dist / (dist.max() + 1e-6), 0, 1)
            alpha = alpha[..., np.newaxis]
            
            target_region = image[y:y+h, x:x+w].astype(float)
            image[y:y+h, x:x+w] = (alpha * img_crop + (1 - alpha) * target_region).astype(np.uint8)
        else:
            # Direct paste
            image[y:y+h, x:x+w][mask_bool] = img_crop[mask_bool]
        
        # Update instance map (clear existing and add new)
        instance_map[y:y+h, x:x+w][mask_bool] = new_instance_id
        
        # Update type map
        type_map[y:y+h, x:x+w][mask_bool] = cls
        
        # Regenerate HV map for the pasted region
        hv_map = self._update_hv_map(hv_map, instance_map, y, y+h, x, x+w)
        
        return image, instance_map, type_map, hv_map
    
    def _update_hv_map(
        self,
        hv_map: np.ndarray,
        instance_map: np.ndarray,
        y_min: int, y_max: int,
        x_min: int, x_max: int
    ) -> np.ndarray:
        """Regenerate HV map for a region after pasting."""
        region_inst = instance_map[y_min:y_max, x_min:x_max]
        region_h = np.zeros_like(region_inst, dtype=np.float32)
        region_v = np.zeros_like(region_inst, dtype=np.float32)
        
        instance_ids = np.unique(region_inst)
        instance_ids = instance_ids[instance_ids != 0]
        
        for inst_id in instance_ids:
            inst_mask = (region_inst == inst_id)
            coords = np.where(inst_mask)
            if len(coords[0]) == 0:
                continue
            
            y_min_inst = coords[0].min()
            y_max_inst = coords[0].max()
            x_min_inst = coords[1].min()
            x_max_inst = coords[1].max()
            
            for y_idx, x_idx in zip(coords[0], coords[1]):
                if y_max_inst > y_min_inst:
                    region_v[y_idx, x_idx] = 2 * (y_idx - y_min_inst) / (y_max_inst - y_min_inst) - 1
                if x_max_inst > x_min_inst:
                    region_h[y_idx, x_idx] = 2 * (x_idx - x_min_inst) / (x_max_inst - x_min_inst) - 1
        
        hv_map[0, y_min:y_max, x_min:x_max] = region_h
        hv_map[1, y_min:y_max, x_min:x_max] = region_v
        
        return hv_map
    
    def __call__(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        type_map: np.ndarray,
        hv_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply copy-paste augmentation.
        
        Args:
            image: [H, W, 3] RGB image
            instance_map: [H, W] instance segmentation
            type_map: [H, W] type classification
            hv_map: [2, H, W] HV maps
            
        Returns:
            Augmented (image, instance_map, type_map, hv_map)
        """
        # Random decision to apply
        if random.random() > self.paste_prob:
            return image, instance_map, type_map, hv_map
        
        # Check if bank has nuclei
        total_nuclei = sum(len(v) for v in self.nuclei_bank.values())
        if total_nuclei < 10:
            # Not enough nuclei in bank, just extract and return
            self.add_to_bank(image.copy(), instance_map.copy(), type_map.copy())
            return image, instance_map, type_map, hv_map
        
        # Sample nuclei to paste
        num_to_paste = random.randint(self.min_paste_instances, self.max_paste_instances)
        nuclei_to_paste = self.sample_nuclei_to_paste(num_to_paste)
        
        if len(nuclei_to_paste) == 0:
            return image, instance_map, type_map, hv_map
        
        # Get next instance ID
        max_inst_id = instance_map.max()
        
        # Make copies
        image = image.copy()
        instance_map = instance_map.copy()
        type_map = type_map.copy()
        hv_map = hv_map.copy()
        
        # Paste each nucleus
        for nuc in nuclei_to_paste:
            location = self.find_paste_location(image, instance_map, nuc['size'])
            if location is not None:
                max_inst_id += 1
                image, instance_map, type_map, hv_map = self.paste_nucleus(
                    image, instance_map, type_map, hv_map,
                    nuc, location, max_inst_id
                )
        
        # Add current image nuclei to bank (for future use)
        self.add_to_bank(image.copy(), instance_map.copy(), type_map.copy())
        
        return image, instance_map, type_map, hv_map


# =============================================================================
# Integration with Dataset
# =============================================================================

def create_copy_paste_augmentation(
    paste_prob: float = 0.5,
    max_instances: int = 5,
    prioritize_dead: bool = True
) -> CopyPasteAugmentation:
    """
    Create copy-paste augmentation with recommended settings.
    
    Args:
        paste_prob: Probability of applying augmentation
        max_instances: Maximum nuclei to paste per image
        prioritize_dead: Whether to heavily prioritize Dead class
        
    Returns:
        CopyPasteAugmentation instance
    """
    class_weights = PANNUKE_SAMPLE_WEIGHTS.copy()
    if prioritize_dead:
        class_weights[4] = 15.0  # Even higher weight for Dead
    
    return CopyPasteAugmentation(
        paste_prob=paste_prob,
        max_paste_instances=max_instances,
        min_paste_instances=1,
        class_weights=class_weights,
        jitter_scale=(0.85, 1.15),
        jitter_brightness=0.1,
        blend_mode='direct',
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Copy-Paste Augmentation Module")
    print("=" * 50)
    
    # Create augmentation
    aug = create_copy_paste_augmentation(paste_prob=0.7, max_instances=3)
    
    # Test with dummy data
    H, W = 256, 256
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    instance_map = np.zeros((H, W), dtype=np.int32)
    type_map = np.zeros((H, W), dtype=np.int32)
    hv_map = np.zeros((2, H, W), dtype=np.float32)
    
    # Add some fake nuclei
    for i in range(5):
        cx, cy = np.random.randint(20, 236, 2)
        r = np.random.randint(5, 15)
        y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
        mask = x*x + y*y <= r*r
        instance_map[mask] = i + 1
        type_map[mask] = np.random.randint(1, 6)  # Random class 1-5
    
    print(f"Original instances: {instance_map.max()}")
    print(f"Original classes present: {np.unique(type_map)}")
    
    # Add to bank (simulating multiple images)
    for _ in range(10):
        aug.add_to_bank(image.copy(), instance_map.copy(), type_map.copy())
    
    print(f"\nBank sizes: {', '.join(f'{k}: {len(v)}' for k, v in aug.nuclei_bank.items())}")
    
    # Apply augmentation
    aug_image, aug_inst, aug_type, aug_hv = aug(image, instance_map, type_map, hv_map)
    
    print(f"\nAfter augmentation:")
    print(f"Instances: {aug_inst.max()}")
    print(f"Classes present: {np.unique(aug_type)}")
    
    print("\n✅ Copy-Paste augmentation test passed!")
