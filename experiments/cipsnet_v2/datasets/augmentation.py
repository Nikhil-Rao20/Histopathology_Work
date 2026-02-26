"""
Data Augmentation for PanNuke Dataset Expansion
=================================================

Provides histopathology-specific augmentations that EXPAND the training dataset
by creating deterministic augmented copies of each image-mask pair.

This is NOT on-the-fly random transforms (those are in get_train_transforms()).
This creates NEW training samples that are:
  1. Deterministic (same augmented copy every epoch for reproducibility)
  2. Still subject to on-the-fly random transforms on top
  3. Meaningful for histopathology (stain variations, tissue deformations)

Augmentation Types:
  1. Stain Jitter (Light)  — H&E stain deconvolution + mild perturbation     [image only]
  2. Stain Jitter (Strong) — H&E stain deconvolution + aggressive perturbation [image only]
  3. Elastic Deformation   — Smooth random spatial deformation                 [image + mask]
  4. Gaussian Noise        — Additive noise to simulate scanner variation      [image only]

Dataset Expansion Math:
  Original: N training samples
  With K augmentation types enabled: N × (1 + K) total training samples

  Default (3 types ON, noise OFF): N × 4 = 4× expansion
  Max (all 4 ON):                  N × 5 = 5× expansion

  Example (test_fold=1, static dataloader, folds 2+3):
      Original:     2523 + 2722 = 5245
      4× expanded:  5245 × 4    = 20,980 training samples

Usage:
  This module is consumed by PanNukeOptimizedDataset and PanNukePermutationDataset.
  Enable via ExperimentConfig.use_data_augmentation = True.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter


# ============================================================
# Stain Deconvolution Matrices (Ruifrok & Johnston, 2001)
# ============================================================

# H&E stain vectors in optical density space (normalized rows)
_HE_STAIN_MATRIX = np.array([
    [0.6500286, 0.7041306, 0.2862040],   # Hematoxylin
    [0.0725258, 0.9909040, 0.1108150],   # Eosin
    [0.2680680, 0.5706310, 0.7783900],   # DAB (residual channel)
], dtype=np.float64)

_HE_STAIN_MATRIX_INV = np.linalg.inv(_HE_STAIN_MATRIX)


# ============================================================
# Augmentation Functions
# ============================================================

def stain_jitter(
    image: np.ndarray,
    alpha_range: Tuple[float, float] = (0.85, 1.15),
    beta_range: Tuple[float, float] = (-0.03, 0.03),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Augment H&E stained image via stain deconvolution + random perturbation.

    Method (based on Ruifrok & Johnston, 2001):
      1. Convert RGB → Optical Density (OD = -log10(I / 255))
      2. Deconvolve OD into Hematoxylin and Eosin concentration channels
      3. Randomly scale (alpha) and shift (beta) each stain channel
      4. Reconstruct OD and convert back to RGB

    This is biologically meaningful: it simulates variations in staining
    protocol, reagent concentrations, and section thickness across different
    labs and tissue preparation procedures.

    Args:
        image:       (H, W, 3) uint8 RGB image
        alpha_range: (min, max) multiplicative factor for stain channels
        beta_range:  (min, max) additive bias for stain channels
        rng:         Random state for reproducibility

    Returns:
        Augmented (H, W, 3) uint8 RGB image
    """
    if rng is None:
        rng = np.random.RandomState()

    H, W, C = image.shape
    img = image.astype(np.float64) / 255.0
    img = np.clip(img, 1e-6, 1.0)          # Avoid log(0)

    # RGB → Optical Density
    od = -np.log10(img).reshape(-1, 3)      # (H*W, 3)

    # Deconvolve into stain concentrations
    concentrations = od @ _HE_STAIN_MATRIX_INV.T  # (H*W, 3)

    # Perturb Hematoxylin (ch 0) and Eosin (ch 1); leave DAB residual alone
    for ch in range(2):
        alpha = rng.uniform(*alpha_range)
        beta = rng.uniform(*beta_range)
        concentrations[:, ch] = concentrations[:, ch] * alpha + beta

    # Reconstruct
    od_aug = np.clip(concentrations @ _HE_STAIN_MATRIX.T, 0, None)
    img_aug = np.clip(10.0 ** (-od_aug), 0, 1).reshape(H, W, 3)

    return (img_aug * 255).astype(np.uint8)


def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 30.0,
    sigma: float = 10.0,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply smooth elastic deformation to both image and mask.

    Generates random displacement fields, smooths them with a Gaussian
    kernel, and warps the image (bilinear) and mask (nearest-neighbor).
    Simulates tissue sectioning, mounting, and dehydration artifacts
    that are common in histopathology slide preparation.

    Args:
        image: (H, W, 3) uint8 RGB image
        mask:  (H, W, C) integer mask array (any number of channels)
        alpha: Deformation magnitude in pixels. Higher = more distortion.
        sigma: Gaussian smoothing sigma. Higher = smoother / more global.
        rng:   Random state for reproducibility

    Returns:
        (deformed_image, deformed_mask) tuple with same dtypes as input
    """
    if rng is None:
        rng = np.random.RandomState()

    H, W = image.shape[:2]

    # Random displacement fields, smoothed for spatial coherence
    dx = gaussian_filter(rng.rand(H, W) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(rng.rand(H, W) * 2 - 1, sigma) * alpha

    # Coordinate grids
    x, y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Warp image (bilinear interpolation for smooth result)
    image_aug = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Warp mask channels (nearest-neighbor to preserve integer instance IDs)
    mask_aug = np.zeros_like(mask)
    if mask.ndim == 3:
        for c in range(mask.shape[2]):
            ch = mask[:, :, c].astype(np.float32)
            mask_aug[:, :, c] = cv2.remap(
                ch, map_x, map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101,
            ).astype(mask.dtype)
    else:
        ch = mask.astype(np.float32)
        mask_aug = cv2.remap(
            ch, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(mask.dtype)

    return image_aug, mask_aug


def gaussian_noise(
    image: np.ndarray,
    sigma: float = 10.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to simulate scanner/imaging noise.

    Different scanners and imaging conditions produce different noise
    levels. This augmentation creates a copy with realistic sensor noise.

    Args:
        image: (H, W, 3) uint8 RGB image
        sigma: Noise standard deviation (0-255 scale). 10 = subtle, 25 = visible.
        rng:   Random state for reproducibility

    Returns:
        Noisy (H, W, 3) uint8 RGB image
    """
    if rng is None:
        rng = np.random.RandomState()

    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ============================================================
# Configuration
# ============================================================

@dataclass
class AugmentationConfig:
    """
    Configuration for dataset expansion via augmentation.

    When enabled, the training dataset is expanded by creating deterministic
    augmented copies of each image. Each augmentation type adds N new samples
    (where N = original dataset size), so the total becomes:

        total = N × (1 + num_enabled_augmentation_types)

    Only affects TRAINING data. Validation/test data is never augmented.

    The augmented copies are generated on-the-fly in __getitem__ (not saved
    to disk), but they are deterministic — the same copy looks the same
    every epoch via fixed seeding per (sample_index, augmentation_type).
    On top of this, standard on-the-fly transforms (flip, rotate, color
    jitter, etc.) still apply randomly each epoch.
    """

    # Master toggle
    enabled: bool = False

    # Individual augmentation toggles
    stain_jitter_light: bool = True       # +N samples (mild stain variation)
    stain_jitter_strong: bool = True      # +N samples (aggressive stain variation)
    elastic_deformation: bool = True      # +N samples (spatial deformation)
    gaussian_noise: bool = False          # +N samples (scanner noise) — OFF by default

    # Stain jitter parameters
    stain_light_alpha: Tuple[float, float] = (0.85, 1.15)    # Scale range
    stain_light_beta: Tuple[float, float] = (-0.03, 0.03)    # Shift range
    stain_strong_alpha: Tuple[float, float] = (0.7, 1.3)     # Scale range
    stain_strong_beta: Tuple[float, float] = (-0.07, 0.07)   # Shift range

    # Elastic deformation parameters
    elastic_alpha: float = 30.0     # Deformation magnitude (pixels)
    elastic_sigma: float = 10.0     # Smoothness (higher = more global)

    # Gaussian noise parameters
    noise_sigma: float = 10.0       # Noise std-dev (0-255 scale)

    # Reproducibility
    seed: int = 42

    # ------------------------------------------------------------------
    # Derived Properties
    # ------------------------------------------------------------------

    @property
    def augmentation_names(self) -> List[str]:
        """Ordered list of enabled augmentation type names."""
        names = []
        if self.stain_jitter_light:
            names.append('stain_light')
        if self.stain_jitter_strong:
            names.append('stain_strong')
        if self.elastic_deformation:
            names.append('elastic')
        if self.gaussian_noise:
            names.append('gaussian_noise')
        return names

    @property
    def num_augmentation_types(self) -> int:
        """Number of enabled augmentation types."""
        return len(self.augmentation_names)

    @property
    def expansion_factor(self) -> int:
        """Total expansion factor.  1 = no expansion, 4 = 4× data."""
        if not self.enabled:
            return 1
        return 1 + self.num_augmentation_types

    def get_augmentation_for_group(self, group: int) -> str:
        """Get augmentation name for a given group index (0-based)."""
        return self.augmentation_names[group]


# ============================================================
# Index Resolution
# ============================================================

def resolve_expansion_index(
    idx: int,
    original_len: int,
    config: Optional[AugmentationConfig],
) -> Tuple[int, Optional[str]]:
    """
    Map an expanded dataset index to (original_index, augmentation_type).

    Index layout (N = original_len, K = num_augmentation_types):
        [0,   N)    → original samples             → aug_type = None
        [N,  2N)    → augmentation_names[0] copies  → aug_type = names[0]
        [2N, 3N)    → augmentation_names[1] copies  → aug_type = names[1]
        ...
        [K*N, (K+1)*N) → augmentation_names[K-1]    → aug_type = names[K-1]

    Args:
        idx:          Index into the expanded dataset
        original_len: Length of the un-expanded dataset
        config:       AugmentationConfig (or None / disabled)

    Returns:
        (original_index, augmentation_type_string_or_None)
    """
    if config is None or not config.enabled or idx < original_len:
        # Original sample (or augmentation disabled)
        return min(idx, original_len - 1), None

    group = (idx - original_len) // original_len
    orig = (idx - original_len) % original_len
    aug_type = config.get_augmentation_for_group(group)
    return orig, aug_type


# ============================================================
# Apply Augmentation
# ============================================================

def apply_expansion_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    aug_type: str,
    config: AugmentationConfig,
    orig_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a specific expansion augmentation to an image-mask pair.

    Uses deterministic seeding based on (orig_idx, aug_type) so that the
    same augmented copy is produced every epoch (reproducibility), while
    different images and different augmentation types get different seeds.

    Args:
        image:    (H, W, 3) uint8 RGB image
        mask:     (H, W, C) mask array (channels depend on dataset)
        aug_type: One of 'stain_light', 'stain_strong', 'elastic', 'gaussian_noise'
        config:   AugmentationConfig with parameters
        orig_idx: Original sample index (for deterministic seeding)

    Returns:
        (augmented_image, augmented_mask)
        Mask is unchanged for image-only augmentations (stain, noise).
    """
    # Deterministic seed per (sample, augmentation type)
    type_hash = hash(aug_type) & 0x7FFFFFFF
    rng = np.random.RandomState((config.seed + orig_idx * 7 + type_hash) % (2 ** 31))

    if aug_type == 'stain_light':
        image = stain_jitter(image, config.stain_light_alpha, config.stain_light_beta, rng)

    elif aug_type == 'stain_strong':
        image = stain_jitter(image, config.stain_strong_alpha, config.stain_strong_beta, rng)

    elif aug_type == 'elastic':
        image, mask = elastic_deformation(
            image, mask, config.elastic_alpha, config.elastic_sigma, rng,
        )

    elif aug_type == 'gaussian_noise':
        image = gaussian_noise(image, config.noise_sigma, rng)

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    return image, mask
