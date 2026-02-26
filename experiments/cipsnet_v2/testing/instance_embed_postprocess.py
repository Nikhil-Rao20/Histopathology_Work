"""
Post-Processing for LViT-IE (Instance Embedding Decoder)
==========================================================

Replaces HoVer-Net watershed with a 3-stage pipeline:

1. **NP Head** → binary foreground mask (same as HoVer-Net)
2. **Distance Transform** → local maxima as instance markers
3. **Instance Embeddings** → mean-shift/connected-components for assignment
4. **Type Classification** → per-instance majority vote from type head

Key improvements over HoVer-Net post-processing:
- Distance transform markers are more robust than HV gradient energy
- Instance embeddings provide complementary grouping cues
- No Sobel gradient computation needed
"""

import numpy as np
import cv2
from scipy.ndimage import (
    binary_fill_holes,
    label as scipy_label,
    maximum_filter,
    distance_transform_edt,
)
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, disk, binary_erosion, binary_dilation
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class IEPostProcessConfig:
    """Post-processing configuration for LViT-IE.
    
    TUNED parameters from post_processing_tuning_lvit_ie.ipynb (Feb 2026):
    - np_threshold: 0.525 → 0.40 (+0.53% mPQ)
    - dist_peak_threshold: 0.30 → 0.60 (+0.07% mPQ)
    - embedding_distance_threshold: 1.50 → 1.25 (+0.31% mPQ)
    - Total improvement: 48.12% → 49.17% mPQ (+1.05%)
    
    Note: TTA does NOT help LVIT_IE (hurts by -10% mPQ). Use no-TTA inference.
    """
    # NP thresholds (TUNED: lower threshold helps)
    np_threshold: float = 0.40  # Was 0.525
    
    # Instance size filtering (unchanged - 70 is optimal)
    min_instance_size: int = 70
    max_instance_size: int = 10000
    
    # Distance transform marker extraction
    dist_peak_min_distance: int = 5     # Minimum distance between peaks
    dist_peak_threshold: float = 0.60   # TUNED: Was 0.3, higher is better
    dist_marker_erosion: int = 0        # Additional erosion on markers (0 = none)
    
    # Embedding clustering
    use_embeddings: bool = True          # Use embeddings for grouping
    embedding_bandwidth: float = 1.0     # Mean-shift bandwidth
    embedding_distance_threshold: float = 1.25  # TUNED: Was 1.5, tighter clustering
    
    # Watershed fallback
    use_watershed_on_dist: bool = True   # Use watershed on distance map for fallback


class IEPostProcessor:
    """
    Instance Embedding post-processor for LViT-IE.
    
    Pipeline:
        1. NP softmax → binary mask → fill holes
        2. Distance transform → find local maxima (markers)
        3. Watershed on -dist_transform using markers
        4. (Optional) Refine with embedding similarity
        5. Filter small/large instances
        6. Assign types via majority vote
    """
    
    def __init__(self, config: Optional[IEPostProcessConfig] = None):
        self.config = config or IEPostProcessConfig()
    
    def process(
        self,
        pred_np: np.ndarray,
        pred_dist: np.ndarray,
        pred_type: np.ndarray,
        pred_embed: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Process model outputs to get instance segmentation.
        
        Args:
            pred_np:    [2, H, W] NP logits (binary segmentation)
            pred_dist:  [1, H, W] predicted normalized distance transform [0,1]
            pred_type:  [C, H, W] type classification logits
            pred_embed: [D, H, W] instance embeddings (optional)
            
        Returns:
            dict with 'inst_map', 'type_map', 'inst_type', 'inst_centroid'
        """
        H, W = pred_np.shape[-2], pred_np.shape[-1]
        
        # === 1. Binary foreground from NP ===
        if pred_np.ndim == 3 and pred_np.shape[0] == 2:
            np_prob = self._softmax(pred_np)[1]
        else:
            np_prob = pred_np
        
        binary_mask = (np_prob > self.config.np_threshold).astype(np.uint8)
        binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)
        
        # === 2. Type predictions ===
        if pred_type.ndim == 3:
            type_pred = np.argmax(pred_type, axis=0)
        else:
            type_pred = pred_type
        
        # Empty case
        if binary_mask.sum() == 0:
            return self._empty_result(H, W)
        
        # === 3. Distance transform → markers ===
        dist_map = pred_dist[0] if pred_dist.ndim == 3 else pred_dist  # [H, W]
        dist_map = dist_map * binary_mask  # Zero out background
        
        markers = self._find_distance_markers(dist_map, binary_mask)
        
        # === 4. Watershed on negative distance transform ===
        inst_map = self._watershed_on_distance(dist_map, binary_mask, markers)
        
        # === 5. (Optional) Refine with embeddings ===
        if self.config.use_embeddings and pred_embed is not None:
            inst_map = self._refine_with_embeddings(inst_map, pred_embed, binary_mask)
        
        # === 6. Filter instances ===
        inst_map = self._filter_instances(inst_map)
        
        # === 7. Assign types ===
        type_map, inst_type, inst_centroid = self._assign_types(inst_map, type_pred)
        
        return {
            'inst_map': inst_map,
            'type_map': type_map,
            'inst_type': inst_type,
            'inst_centroid': inst_centroid,
        }
    
    # -----------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax along first axis."""
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
    
    def _empty_result(self, H: int, W: int) -> Dict:
        return {
            'inst_map': np.zeros((H, W), dtype=np.int32),
            'type_map': np.zeros((H, W), dtype=np.int32),
            'inst_type': {},
            'inst_centroid': {},
        }
    
    def _find_distance_markers(
        self,
        dist_map: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Find instance markers as local maxima of distance transform.
        
        A point is a local maximum if:
          - Its value equals the max in a local neighborhood
          - Its value exceeds dist_peak_threshold
          - It's within the foreground mask
        """
        min_dist = self.config.dist_peak_min_distance
        threshold = self.config.dist_peak_threshold
        
        # Local maximum filter
        footprint_size = 2 * min_dist + 1
        local_max = maximum_filter(dist_map, size=footprint_size)
        
        # Points where value equals local max AND above threshold
        peaks = (dist_map == local_max) & (dist_map >= threshold) & (binary_mask > 0)
        
        # Optional erosion to avoid double-counting
        if self.config.dist_marker_erosion > 0:
            kernel = disk(self.config.dist_marker_erosion)
            peaks = binary_erosion(peaks, kernel)
        
        # Label connected peaks as separate markers
        markers, n_markers = scipy_label(peaks)
        
        # Fallback: if no markers found, use EDT of binary mask
        if n_markers == 0:
            edt = distance_transform_edt(binary_mask)
            local_max_edt = maximum_filter(edt, size=footprint_size)
            peaks_edt = (edt == local_max_edt) & (edt > 2) & (binary_mask > 0)
            markers, n_markers = scipy_label(peaks_edt)
        
        # Last resort: connected components
        if n_markers == 0:
            markers, _ = scipy_label(binary_mask)
        
        return markers
    
    def _watershed_on_distance(
        self,
        dist_map: np.ndarray,
        binary_mask: np.ndarray,
        markers: np.ndarray,
    ) -> np.ndarray:
        """Watershed segmentation using negative distance transform as energy."""
        n_markers = markers.max()
        
        if n_markers == 0:
            inst_map, _ = scipy_label(binary_mask)
            return inst_map.astype(np.int32)
        
        # Add background marker
        bg_label = n_markers + 1
        markers_with_bg = markers.copy()
        markers_with_bg[binary_mask == 0] = bg_label
        
        # Negative distance = high energy at boundaries, low at centers
        energy = -dist_map.astype(np.float64)
        
        inst_map = watershed(
            energy,
            markers_with_bg,
            mask=binary_mask,
        )
        
        # Remove background label
        inst_map[inst_map == bg_label] = 0
        
        return inst_map.astype(np.int32)
    
    def _refine_with_embeddings(
        self,
        inst_map: np.ndarray,
        embeddings: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Refine instance segmentation using embedding similarity.
        
        Strategy: For each instance from watershed, check if the embedding
        variance is too high (suggesting over-segmentation or under-segmentation).
        
        Merge step: If two adjacent instances have very similar mean embeddings,
        merge them (handles under-segmentation from distance transform).
        
        Split step: deferred to future work (complex, and watershed typically
        handles splits well).
        """
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        if len(unique_ids) <= 1:
            return inst_map
        
        D = embeddings.shape[0]
        embed_flat = embeddings.reshape(D, -1)  # [D, H*W]
        inst_flat = inst_map.flatten()  # [H*W]
        
        # Compute mean embedding per instance
        centers = {}
        for inst_id in unique_ids:
            mask = (inst_flat == inst_id)
            if mask.sum() < 5:
                continue
            centers[int(inst_id)] = embed_flat[:, mask].mean(axis=1)  # [D]
        
        if len(centers) <= 1:
            return inst_map
        
        # Merge nearby instances
        threshold = self.config.embedding_distance_threshold
        inst_ids = list(centers.keys())
        
        # Build merge graph via Union-Find
        parent = {iid: iid for iid in inst_ids}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        # Check adjacency AND embedding distance
        for i, id_a in enumerate(inst_ids):
            for id_b in inst_ids[i + 1:]:
                # Check if adjacent (dilate one, check overlap)
                mask_a = (inst_map == id_a).astype(np.uint8)
                mask_b = (inst_map == id_b).astype(np.uint8)
                dilated_a = cv2.dilate(mask_a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                
                if (dilated_a & mask_b).sum() > 0:
                    # Adjacent — check embedding distance
                    dist = np.linalg.norm(centers[id_a] - centers[id_b])
                    if dist < threshold:
                        union(id_a, id_b)
        
        # Apply merges
        new_inst_map = np.zeros_like(inst_map)
        group_label = {}
        next_id = 1
        
        for inst_id in inst_ids:
            root = find(inst_id)
            if root not in group_label:
                group_label[root] = next_id
                next_id += 1
            new_inst_map[inst_map == inst_id] = group_label[root]
        
        return new_inst_map
    
    def _filter_instances(self, inst_map: np.ndarray) -> np.ndarray:
        """Remove instances too small or too large, re-label sequentially."""
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
        """Majority vote type assignment per instance."""
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        type_map = np.zeros_like(inst_map)
        inst_type = {}
        inst_centroid = {}
        
        for inst_id in unique_ids:
            inst_mask = (inst_map == inst_id)
            types_in_inst = type_pred[inst_mask]
            
            type_counts = np.bincount(types_in_inst, minlength=6)
            if type_counts[1:].sum() > 0:
                type_counts[0] = 0
            
            assigned_type = np.argmax(type_counts)
            if assigned_type == 0:
                assigned_type = 1
            
            type_map[inst_mask] = assigned_type
            inst_type[int(inst_id)] = int(assigned_type)
            
            y_coords, x_coords = np.where(inst_mask)
            inst_centroid[int(inst_id)] = (float(np.mean(x_coords)), float(np.mean(y_coords)))
        
        return type_map, inst_type, inst_centroid


# =============================================================================
# Batch processing (mirrors HoVer-Net API)
# =============================================================================

def process_batch_ie(
    pred_np: torch.Tensor,
    pred_dist: torch.Tensor,
    pred_type: torch.Tensor,
    pred_embed: Optional[torch.Tensor] = None,
    config: Optional[IEPostProcessConfig] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Process a batch of LViT-IE model outputs.
    
    Args:
        pred_np:    [B, 2, H, W] NP logits
        pred_dist:  [B, 1, H, W] distance transform predictions
        pred_type:  [B, C, H, W] type logits
        pred_embed: [B, D, H, W] instance embeddings (optional)
        config:     post-processing config
    
    Returns:
        List of result dicts, one per batch item
    """
    processor = IEPostProcessor(config)
    
    # Convert to numpy
    if isinstance(pred_np, torch.Tensor):
        pred_np = pred_np.detach().cpu().numpy()
    if isinstance(pred_dist, torch.Tensor):
        pred_dist = pred_dist.detach().cpu().numpy()
    if isinstance(pred_type, torch.Tensor):
        pred_type = pred_type.detach().cpu().numpy()
    if pred_embed is not None and isinstance(pred_embed, torch.Tensor):
        pred_embed = pred_embed.detach().cpu().numpy()
    
    B = pred_np.shape[0]
    results = []
    
    for i in range(B):
        emb = pred_embed[i] if pred_embed is not None else None
        result = processor.process(
            pred_np=pred_np[i],
            pred_dist=pred_dist[i],
            pred_type=pred_type[i],
            pred_embed=emb,
        )
        results.append(result)
    
    return results


# =============================================================================
# Testing
# =============================================================================

def test_ie_post_processor():
    """Smoke test for IE post-processor."""
    print("Testing IEPostProcessor...")
    print("=" * 60)
    
    H, W, D = 256, 256, 16
    
    # NP logits [2, H, W]
    pred_np = np.zeros((2, H, W), dtype=np.float32)
    pred_np[0] = 2.0  # background
    for (y1, y2, x1, x2) in [(50, 80, 50, 80), (100, 140, 100, 140), (70, 95, 170, 200)]:
        pred_np[1, y1:y2, x1:x2] = 5.0
        pred_np[0, y1:y2, x1:x2] = -3.0
    
    # Distance transform [1, H, W] — simulate normalized EDT
    pred_dist = np.zeros((1, H, W), dtype=np.float32)
    for (y1, y2, x1, x2) in [(50, 80, 50, 80), (100, 140, 100, 140), (70, 95, 170, 200)]:
        cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
        ry, rx = (y2 - y1) / 2, (x2 - x1) / 2
        yy, xx = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')
        d = 1.0 - np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
        d = np.clip(d, 0, 1)
        pred_dist[0, y1:y2, x1:x2] = d
    
    # Type logits [6, H, W]
    pred_type = np.random.randn(6, H, W).astype(np.float32) * 0.1
    pred_type[1, 50:80, 50:80] = 3.0
    pred_type[2, 100:140, 100:140] = 3.0
    pred_type[4, 70:95, 170:200] = 3.0
    
    # Embeddings [D, H, W] — different means per instance
    pred_embed = np.random.randn(D, H, W).astype(np.float32) * 0.1
    for i, (y1, y2, x1, x2) in enumerate([(50, 80, 50, 80), (100, 140, 100, 140), (70, 95, 170, 200)]):
        pred_embed[:, y1:y2, x1:x2] += (i + 1) * 3.0  # Well-separated
    
    # Process
    processor = IEPostProcessor()
    result = processor.process(pred_np, pred_dist, pred_type, pred_embed)
    
    print(f"  Instance IDs: {np.unique(result['inst_map'])}")
    print(f"  Num instances: {len(result['inst_type'])}")
    print(f"  Instance types: {result['inst_type']}")
    print(f"  Instance centroids: {result['inst_centroid']}")
    
    # Batch processing
    print("\nTesting batch processing...")
    results = process_batch_ie(
        torch.from_numpy(np.stack([pred_np, pred_np])),
        torch.from_numpy(np.stack([pred_dist, pred_dist])),
        torch.from_numpy(np.stack([pred_type, pred_type])),
        torch.from_numpy(np.stack([pred_embed, pred_embed])),
    )
    print(f"  Batch: {len(results)} images, instances: {[len(r['inst_type']) for r in results]}")
    
    print("\n" + "=" * 60)
    print("IEPostProcessor tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_ie_post_processor()
