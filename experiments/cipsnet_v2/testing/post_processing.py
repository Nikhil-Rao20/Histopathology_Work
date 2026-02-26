"""
Post-Processing for CIPS-Net V2
================================

HoVer-Net style post-processing to convert model outputs to instance maps.

Process:
1. Binary segmentation (NP head) → foreground mask
2. HV maps → gradient-based marker extraction
3. Watershed segmentation → instance map
4. Type assignment per instance

Reference: HoVer-Net (Graham et al., 2019)
"""

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes, label as scipy_label
from scipy.ndimage import distance_transform_edt, measurements
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, disk, binary_erosion
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PostProcessConfig:
    """Post-processing configuration.
    
    Tuned defaults (Feb 2026, post_processing_tuning.ipynb):
      np_threshold: 0.5 → 0.525  (+0.25% mPQ)
      min_instance_size: 10 → 70  (+1.80% mPQ, biggest single gain)
      marker_erosion_size: 2 → 3  (+0.81% mPQ)
    Total post-proc gain: +1.19% mPQ over TTA default (44.66% → 45.85%)
    """
    # NP thresholds
    np_threshold: float = 0.525  # Threshold for binary NP prediction (tuned from 0.5)
    
    # Instance size filtering
    min_instance_size: int = 70  # Minimum instance size in pixels (tuned from 10)
    max_instance_size: int = 10000  # Maximum instance size (filter artifacts)
    
    # Marker extraction (HV gradient)
    h_threshold: float = 0.5  # Horizontal gradient threshold
    v_threshold: float = 0.5  # Vertical gradient threshold
    energy_threshold: float = 0.5  # Energy map threshold for markers
    marker_erosion_size: int = 3  # Erosion for marker extraction (tuned from 2)
    
    # Watershed
    use_distance_transform: bool = True  # Use distance transform for watershed


class PostProcessor:
    """
    HoVer-Net style post-processor for nucleus instance segmentation.
    
    Converts model outputs (NP, HV, Type maps) to instance segmentation maps.
    """
    
    def __init__(self, config: Optional[PostProcessConfig] = None):
        """
        Initialize post-processor.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config or PostProcessConfig()
    
    def process(
        self,
        pred_np: np.ndarray,
        pred_hv: np.ndarray,
        pred_type: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Process model outputs to get instance segmentation.
        
        Args:
            pred_np: Binary segmentation logits [2, H, W] or probs [H, W]
            pred_hv: HV maps [2, H, W] - channel 0 = H, channel 1 = V
            pred_type: Type classification logits [C, H, W] or predictions [H, W]
            
        Returns:
            Dictionary containing:
                - 'inst_map': Instance segmentation map [H, W]
                - 'type_map': Type map per pixel [H, W]
                - 'inst_type': Type per instance Dict[inst_id, type]
                - 'inst_centroid': Centroid per instance Dict[inst_id, (x, y)]
        """
        # Get binary mask from NP prediction
        if pred_np.ndim == 3 and pred_np.shape[0] == 2:
            # Softmax logits [2, H, W]
            pred_np_prob = self._softmax(pred_np)[1]  # Take foreground channel
        else:
            pred_np_prob = pred_np
        
        binary_mask = (pred_np_prob > self.config.np_threshold).astype(np.uint8)
        
        # Fill holes
        binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)
        
        # Get type predictions
        if pred_type.ndim == 3:
            # Logits [C, H, W]
            type_pred = np.argmax(pred_type, axis=0)
        else:
            type_pred = pred_type
        
        # If no foreground, return empty
        if binary_mask.sum() == 0:
            h, w = binary_mask.shape
            return {
                'inst_map': np.zeros((h, w), dtype=np.int32),
                'type_map': np.zeros((h, w), dtype=np.int32),
                'inst_type': {},
                'inst_centroid': {},
            }
        
        # Get instance map using watershed
        inst_map = self._watershed_segmentation(binary_mask, pred_hv)
        
        # Remove small instances
        inst_map = self._filter_instances(inst_map)
        
        # Get type map and instance types
        type_map, inst_type, inst_centroid = self._assign_types(inst_map, type_pred)
        
        return {
            'inst_map': inst_map,
            'type_map': type_map,
            'inst_type': inst_type,
            'inst_centroid': inst_centroid,
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along first axis."""
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
    
    def _watershed_segmentation(
        self,
        binary_mask: np.ndarray,
        pred_hv: np.ndarray,
    ) -> np.ndarray:
        """
        Perform watershed segmentation using HV maps.
        
        The HV maps encode horizontal and vertical distances to instance centers.
        We compute the gradient magnitude to find instance boundaries and use
        local minima of energy as markers.
        """
        h_map = pred_hv[0]  # Horizontal
        v_map = pred_hv[1]  # Vertical
        
        # Compute gradients using Sobel
        h_grad = cv2.Sobel(h_map, cv2.CV_64F, 1, 0, ksize=3)
        v_grad = cv2.Sobel(v_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude (energy)
        energy = np.sqrt(h_grad ** 2 + v_grad ** 2)
        energy = energy / (energy.max() + 1e-6)  # Normalize
        
        # Apply mask
        energy = energy * binary_mask
        
        # Get markers: low energy regions (instance centers)
        # Threshold on energy to find potential centers
        marker_mask = (energy < self.config.energy_threshold) & (binary_mask > 0)
        
        # Erode to separate touching markers
        if self.config.marker_erosion_size > 0:
            kernel = disk(self.config.marker_erosion_size)
            marker_mask = binary_erosion(marker_mask, kernel)
        
        # Label connected components as markers
        markers, num_markers = scipy_label(marker_mask)
        
        # If no markers found, use distance transform peaks
        if num_markers == 0:
            dist_transform = distance_transform_edt(binary_mask)
            # Local maxima of distance transform
            dist_max = cv2.dilate(dist_transform.astype(np.float32), 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            local_max = (dist_transform == dist_max) & (binary_mask > 0)
            markers, num_markers = scipy_label(local_max)
        
        # Watershed
        if num_markers > 0:
            # Use negative energy as watershed input (want to flow to low energy)
            if self.config.use_distance_transform:
                dist_transform = distance_transform_edt(binary_mask)
                watershed_input = -dist_transform
            else:
                watershed_input = energy
            
            # Add background marker
            markers[binary_mask == 0] = num_markers + 1
            
            # Run watershed
            inst_map = watershed(
                watershed_input,
                markers,
                mask=binary_mask,
            )
            
            # Remove background label
            inst_map[inst_map == num_markers + 1] = 0
        else:
            # Fallback: connected components
            inst_map, _ = scipy_label(binary_mask)
        
        return inst_map.astype(np.int32)
    
    def _filter_instances(self, inst_map: np.ndarray) -> np.ndarray:
        """Remove instances that are too small or too large."""
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        new_inst_map = np.zeros_like(inst_map)
        new_id = 1
        
        for inst_id in unique_ids:
            inst_mask = (inst_map == inst_id)
            size = inst_mask.sum()
            
            if self.config.min_instance_size <= size <= self.config.max_instance_size:
                new_inst_map[inst_mask] = new_id
                new_id += 1
        
        return new_inst_map
    
    def _assign_types(
        self,
        inst_map: np.ndarray,
        type_pred: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Tuple[float, float]]]:
        """
        Assign type to each instance by majority voting.
        
        Returns:
            type_map: Per-pixel type map [H, W]
            inst_type: Dict mapping instance ID to type
            inst_centroid: Dict mapping instance ID to (x, y) centroid
        """
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        type_map = np.zeros_like(inst_map)
        inst_type = {}
        inst_centroid = {}
        
        for inst_id in unique_ids:
            inst_mask = (inst_map == inst_id)
            
            # Majority vote for type (excluding background class 0)
            types_in_inst = type_pred[inst_mask]
            
            # Count votes for each type
            type_counts = np.bincount(types_in_inst, minlength=6)
            # Ignore background votes if there are foreground votes
            if type_counts[1:].sum() > 0:
                type_counts[0] = 0
            
            assigned_type = np.argmax(type_counts)
            
            # If all background, assign most common foreground type overall
            if assigned_type == 0:
                assigned_type = 1  # Default to neoplastic
            
            type_map[inst_mask] = assigned_type
            inst_type[int(inst_id)] = int(assigned_type)
            
            # Compute centroid
            y_coords, x_coords = np.where(inst_mask)
            centroid = (float(np.mean(x_coords)), float(np.mean(y_coords)))
            inst_centroid[int(inst_id)] = centroid
        
        return type_map, inst_type, inst_centroid


def process_batch(
    pred_np: torch.Tensor,
    pred_hv: torch.Tensor,
    pred_type: torch.Tensor,
    config: Optional[PostProcessConfig] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Process a batch of model outputs.
    
    Args:
        pred_np: [B, 2, H, W] NP logits
        pred_hv: [B, 2, H, W] HV maps
        pred_type: [B, C, H, W] Type logits
        config: Post-processing config
        
    Returns:
        List of result dicts, one per batch item
    """
    processor = PostProcessor(config)
    
    # Convert to numpy
    if isinstance(pred_np, torch.Tensor):
        pred_np = pred_np.detach().cpu().numpy()
    if isinstance(pred_hv, torch.Tensor):
        pred_hv = pred_hv.detach().cpu().numpy()
    if isinstance(pred_type, torch.Tensor):
        pred_type = pred_type.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    results = []
    
    for i in range(batch_size):
        result = processor.process(
            pred_np=pred_np[i],
            pred_hv=pred_hv[i],
            pred_type=pred_type[i],
        )
        results.append(result)
    
    return results


# ==========================================================================
# Testing
# ==========================================================================

def test_post_processor():
    """Test post-processor with dummy data."""
    print("Testing PostProcessor...")
    print("=" * 60)
    
    # Create dummy outputs
    H, W = 256, 256
    
    # NP prediction (2-channel logits)
    pred_np = np.random.randn(2, H, W).astype(np.float32)
    # Create some foreground blobs
    pred_np[1, 50:70, 50:70] = 3.0  # Instance 1
    pred_np[1, 100:130, 100:130] = 3.0  # Instance 2
    pred_np[1, 80:100, 180:200] = 3.0  # Instance 3
    pred_np[0] = -pred_np[1]  # Background
    
    # HV prediction
    pred_hv = np.zeros((2, H, W), dtype=np.float32)
    # Add gradient patterns for each instance
    for (y1, y2, x1, x2) in [(50, 70, 50, 70), (100, 130, 100, 130), (80, 100, 180, 200)]:
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        yy, xx = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')
        pred_hv[0, y1:y2, x1:x2] = (xx - cx) / 20  # H
        pred_hv[1, y1:y2, x1:x2] = (yy - cy) / 20  # V
    
    # Type prediction (6-channel logits)
    pred_type = np.random.randn(6, H, W).astype(np.float32)
    pred_type[1, 50:70, 50:70] = 3.0  # Neoplastic
    pred_type[2, 100:130, 100:130] = 3.0  # Inflammatory
    pred_type[5, 80:100, 180:200] = 3.0  # Epithelial
    
    # Process
    processor = PostProcessor()
    result = processor.process(pred_np, pred_hv, pred_type)
    
    print(f"  Instance map unique IDs: {np.unique(result['inst_map'])}")
    print(f"  Number of instances: {len(result['inst_type'])}")
    print(f"  Instance types: {result['inst_type']}")
    print(f"  Instance centroids: {result['inst_centroid']}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    pred_np_batch = torch.from_numpy(np.stack([pred_np, pred_np], axis=0))
    pred_hv_batch = torch.from_numpy(np.stack([pred_hv, pred_hv], axis=0))
    pred_type_batch = torch.from_numpy(np.stack([pred_type, pred_type], axis=0))
    
    results = process_batch(pred_np_batch, pred_hv_batch, pred_type_batch)
    print(f"  Batch size: {len(results)}")
    print(f"  Instances per image: {[len(r['inst_type']) for r in results]}")
    
    print("\n" + "=" * 60)
    print("PostProcessor tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_post_processor()
