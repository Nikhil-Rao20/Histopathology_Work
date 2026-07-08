"""Native-caption raw training for external histopathology datasets.

This script trains the CIPS-Net V2 family on the native class captions of
CoNSeP, MoNuSAC, Lizard, and CoNIC without cross-validation.

Design choices:
- Native class captions only
- Fixed split per dataset
- Shared HoVer-Net style supervision for all standard models
- Dedicated instance-embedding supervision for LViT-IE
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .losses import CIPSNetV2Loss, LViTIELoss
    from .models import (
        create_cris_model,
        create_grounding_dino_model,
        create_lavt_model,
        create_lvit_ie_model,
        create_lvit_model,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from losses import CIPSNetV2Loss, LViTIELoss
    from models import (
        create_cris_model,
        create_grounding_dino_model,
        create_lavt_model,
        create_lvit_ie_model,
        create_lvit_model,
    )


PANNUKE_CLASS_NAMES = [
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
]

CONSEP_NATIVE_NAMES = {
    0: "background",
    1: "other",
    2: "inflammatory",
    3: "healthy epithelial",
    4: "dysplastic/malignant epithelial",
    5: "fibroblast",
    6: "muscle",
    7: "endothelial",
}

CONSEP_TO_PANNUKE = {2: 1, 3: 4, 4: 0, 5: 2}
MONUSAC_TO_PANNUKE = {
    "Epithelial": 4,
    "Lymphocyte": 1,
    "Macrophage": 1,
    "Neutrophil": 1,
}
LIZARD_TO_PANNUKE = {
    1: 1,
    2: 4,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
}
CONIC_TO_PANNUKE = LIZARD_TO_PANNUKE

DEFAULT_TEMPLATES = [
    "Segment all nuclei of {classes} in this histopathology image.",
    "Identify {classes} nuclei in this tissue sample.",
    "Mark every {classes} nucleus visible in the slide.",
    "Locate and segment {classes} cell nuclei.",
    "Find all nuclei belonging to {classes}.",
    "Outline the {classes} nuclei present in the image.",
    "Perform instance segmentation for {classes} nuclei.",
    "Detect {classes} cells and their boundaries.",
]


@dataclass
class SampleRecord:
    dataset: str
    split: str
    image_id: str
    image_path: Path
    label_path: Optional[Path]
    instruction: str
    tissue: str = "unknown"


def _stable_bucket(key: str, modulo: int = 100) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _ensure_3ch(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image


def _resize_if_needed(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[:2] == (size, size):
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def remap_label(inst_map: np.ndarray) -> np.ndarray:
    inst_map = inst_map.astype(np.int32)
    out = np.zeros_like(inst_map, dtype=np.int32)
    inst_ids = [int(x) for x in np.unique(inst_map) if x > 0]
    for new_id, old_id in enumerate(inst_ids, start=1):
        out[inst_map == old_id] = new_id
    return out


def gen_instance_hv_map(instance_map: np.ndarray) -> np.ndarray:
    instance_map = instance_map.astype(np.int32)
    h, w = instance_map.shape
    hv_map = np.zeros((h, w, 2), dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if inst_id <= 0:
            continue
        mask = instance_map == inst_id
        if not mask.any():
            continue
        dist = distance_transform_edt(mask)
        if dist.max() <= 0:
            continue
        dist = dist / dist.max()
        coords = np.where(mask)
        ys, xs = coords
        cy = float(ys.mean())
        cx = float(xs.mean())
        y_norm = (ys.astype(np.float32) - cy) / max(float(mask.shape[0]), 1.0)
        x_norm = (xs.astype(np.float32) - cx) / max(float(mask.shape[1]), 1.0)
        hv_map[ys, xs, 0] = np.clip(x_norm, -1.0, 1.0)
        hv_map[ys, xs, 1] = np.clip(y_norm, -1.0, 1.0)
    return hv_map


def gen_normalized_distance_transform(instance_map: np.ndarray) -> np.ndarray:
    instance_map = instance_map.astype(np.int32)
    dist = np.zeros_like(instance_map, dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if inst_id <= 0:
            continue
        mask = instance_map == inst_id
        if not mask.any():
            continue
        d = distance_transform_edt(mask)
        if d.max() > 0:
            d = d / d.max()
        dist[mask] = d[mask]
    return dist


def parse_consep_mask(mask_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    mat = sio.loadmat(str(mask_path))
    inst_map = mat.get("inst_map")
    if inst_map is None:
        raise ValueError(f"Missing inst_map in {mask_path}")
    type_map = mat.get("type_map")
    if type_map is None:
        type_map = mat.get("inst_type")
    if type_map is None:
        type_map = np.zeros_like(inst_map)
    inst_map = inst_map.astype(np.int32)
    type_map = type_map.astype(np.int32)
    native = []
    for cls_id in sorted(int(x) for x in np.unique(type_map) if x > 0):
        native.append(CONSEP_NATIVE_NAMES.get(cls_id, f"class_{cls_id}"))
    return inst_map, type_map, native


def parse_monusac_xml(xml_path: Path, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    h, w = image_shape
    inst_map = np.zeros((h, w), dtype=np.int32)
    type_map = np.zeros((h, w), dtype=np.int32)
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    class_to_id = {"Epithelial": 4, "Lymphocyte": 1, "Macrophage": 1, "Neutrophil": 1}
    native: List[str] = []
    inst_id = 1
    for annotation in root.findall(".//Annotation"):
        class_name = None
        for attr in annotation.findall(".//Attribute"):
            name = attr.get("Name")
            if name in class_to_id:
                class_name = name
                break
        if class_name is None:
            continue
        regions = annotation.findall(".//Region")
        for region in regions:
            vertices = region.findall(".//Vertex")
            if len(vertices) < 3:
                continue
            pts = np.array([[int(float(v.get("X"))), int(float(v.get("Y")))] for v in vertices], dtype=np.int32)
            if pts.size == 0:
                continue
            cv2.fillPoly(inst_map, [pts], inst_id)
            cv2.fillPoly(type_map, [pts], class_to_id[class_name])
            native.append(class_name)
            inst_id += 1
    return inst_map, type_map, sorted(set(native))


def parse_lizard_mask(mask_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    mat = sio.loadmat(str(mask_path))
    inst_map = mat.get("inst_map")
    if inst_map is None:
        raise ValueError(f"Missing inst_map in {mask_path}")
    inst_map = inst_map.astype(np.int32)
    type_map = np.zeros_like(inst_map, dtype=np.int32)
    cls = mat.get("class")
    ids = mat.get("id")
    native = []
    if cls is not None and ids is not None:
        cls = np.array(cls).reshape(-1)
        ids = np.array(ids).reshape(-1)
        for inst_id, cls_id in zip(ids, cls):
            inst_id = int(inst_id)
            cls_id = int(cls_id)
            mapped_cls = LIZARD_TO_PANNUKE.get(cls_id, 0)
            type_map[inst_map == inst_id] = mapped_cls
            if cls_id == 1:
                native.append("Neutrophil")
            elif cls_id == 2:
                native.append("Epithelial")
            elif cls_id == 3:
                native.append("Lymphocyte")
            elif cls_id == 4:
                native.append("Plasma")
            elif cls_id == 5:
                native.append("Eosinophil")
            elif cls_id == 6:
                native.append("Connective")
    return inst_map, type_map, sorted(set(native))


def parse_conic_sample(image_path: Path, labels: np.ndarray, index: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    inst_map = labels[index, :, :, 0].astype(np.int32)
    raw_type_map = labels[index, :, :, 1].astype(np.int32)
    
    # Remap to PanNuke class IDs
    type_map = np.zeros_like(raw_type_map, dtype=np.int32)
    native = []
    unique_classes = sorted(int(x) for x in np.unique(raw_type_map) if x > 0)
    
    for cls_id in unique_classes:
        mapped_cls = CONIC_TO_PANNUKE.get(cls_id, 0)
        type_map[raw_type_map == cls_id] = mapped_cls
        
        if cls_id == 1:
            native.append("Neutrophil")
        elif cls_id == 2:
            native.append("Epithelial")
        elif cls_id == 3:
            native.append("Lymphocyte")
        elif cls_id == 4:
            native.append("Plasma")
        elif cls_id == 5:
            native.append("Eosinophil")
        elif cls_id == 6:
            native.append("Connective")
    
    return inst_map, type_map, sorted(set(native))


def make_instruction(native_classes: Sequence[str], dataset: str, rng: random.Random) -> str:
    classes = " and ".join(native_classes) if native_classes else "nuclei"
    template = rng.choice(DEFAULT_TEMPLATES)
    return template.format(classes=classes)


def load_caption_table(caption_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if caption_csv is None or not caption_csv.exists():
        return None
    df = pd.read_csv(caption_csv)
    cols = {c.lower(): c for c in df.columns}
    if "instruction_native" not in cols and "instruction" not in cols:
        return None
    return df


class ExternalRawDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        root: Path,
        split: str,
        image_size: int = 256,
        caption_csv: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.root = root
        self.split = split.lower()
        self.image_size = image_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.caption_df = load_caption_table(caption_csv)
        self.records = self._build_records()
        
        # Load CoNIC data if needed
        self._conic_images = None
        self._conic_labels = None
        if dataset.lower() == "conic":
            images_path = root / "images.npy"
            labels_path = root / "labels.npy"
            if images_path.exists() and labels_path.exists():
                self._conic_images = np.load(str(images_path), mmap_mode="r")
                self._conic_labels = np.load(str(labels_path), mmap_mode="r")
        self._conic_patch_info = None
        if self.dataset == "conic":
            self._conic_images = np.load(str(self.root / "images.npy"), mmap_mode="r")
            self._conic_labels = np.load(str(self.root / "labels.npy"), mmap_mode="r")
            patch_info = self.root / "patch_info.csv"
            if patch_info.exists():
                self._conic_patch_info = pd.read_csv(patch_info)

    def _build_records(self) -> List[SampleRecord]:
        ds = self.dataset.lower()
        if ds == "consep":
            return self._build_consep_records()
        if ds == "monusac":
            return self._build_monusac_records()
        if ds == "lizard":
            return self._build_lizard_records()
        if ds == "conic":
            return self._build_conic_records()
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _split_keep(self, key: str) -> bool:
        bucket = _stable_bucket(key)
        if self.split == "train":
            return bucket < 70
        if self.split == "val":
            return 70 <= bucket < 85
        return bucket >= 85

    def _build_consep_records(self) -> List[SampleRecord]:
        base = self.root / ("Train" if self.split in {"train", "val"} else "Test")
        image_dir = base / "Images"
        label_dir = base / "Labels"
        records: List[SampleRecord] = []
        if not image_dir.exists():
            return records
        for img_path in sorted(image_dir.glob("*.png")):
            image_id = img_path.stem
            if self.split in {"train", "val"}:
                if self.split == "train" and not self._split_keep(f"consep::{image_id}"):
                    continue
                if self.split == "val" and self._split_keep(f"consep::{image_id}"):
                    continue
            mask_path = label_dir / f"{image_id}.mat"
            instr = self._caption_for(image_id, ["inflammatory", "epithelial", "neoplastic", "connective"])
            records.append(SampleRecord("consep", self.split, image_id, img_path, mask_path, instr, tissue="CoNSeP"))
        return records

    def _build_monusac_records(self) -> List[SampleRecord]:
        base = self.root / ("Train" if self.split in {"train", "val"} else "Test")
        records: List[SampleRecord] = []
        if not base.exists():
            return records
        for patient_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            for img_path in sorted(list(patient_dir.glob("*.tif")) + list(patient_dir.glob("*.png")) + list(patient_dir.glob("*.jpg"))):
                image_id = img_path.stem
                if self.split in {"train", "val"}:
                    if self.split == "train" and not self._split_keep(f"monusac::{image_id}"):
                        continue
                    if self.split == "val" and self._split_keep(f"monusac::{image_id}"):
                        continue
                label_path = img_path.with_suffix(".xml")
                instr = self._caption_for(image_id, ["epithelial", "inflammatory"])
                records.append(SampleRecord("monusac", self.split, image_id, img_path, label_path, instr, tissue="MoNuSAC"))
        return records

    def _build_lizard_records(self) -> List[SampleRecord]:
        image_dirs = [self.root / "lizard_images1", self.root / "lizard_images2"]
        label_root = self.root / "lizard_labels"
        label_files = list(label_root.rglob("*.mat"))
        label_by_stem = {p.stem: p for p in label_files}
        records: List[SampleRecord] = []
        for img_dir in image_dirs:
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.rglob("*.png")):
                image_id = img_path.stem
                if self.split in {"train", "val"}:
                    if self.split == "train" and not self._split_keep(f"lizard::{image_id}"):
                        continue
                    if self.split == "val" and self._split_keep(f"lizard::{image_id}"):
                        continue
                label_path = label_by_stem.get(image_id)
                instr = self._caption_for(image_id, ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"])
                records.append(SampleRecord("lizard", self.split, image_id, img_path, label_path, instr, tissue="Lizard"))
        return records

    def _build_conic_records(self) -> List[SampleRecord]:
        image_dir = self.root / "images.npy"
        label_file = self.root / "labels.npy"
        if not image_dir.exists() or not label_file.exists():
            return []
        labels = np.load(str(label_file), mmap_mode="r")
        records: List[SampleRecord] = []
        for idx in range(labels.shape[0]):
            image_id = f"conic_{idx:05d}"
            if self.split in {"train", "val"}:
                if self.split == "train" and not self._split_keep(f"conic::{idx}"):
                    continue
                if self.split == "val" and self._split_keep(f"conic::{idx}"):
                    continue
            img_path = self.root / f"{image_id}.png"
            instr = self._caption_for(image_id, ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"])
            records.append(SampleRecord("conic", self.split, image_id, img_path, label_file, instr, tissue="CoNIC"))
        return records

    def _caption_for(self, image_id: str, fallback_classes: Sequence[str]) -> str:
        if self.caption_df is None:
            return make_instruction(fallback_classes, self.dataset, self.rng)
        df = self.caption_df
        cols = {c.lower(): c for c in df.columns}
        image_col = cols.get("image_id") or cols.get("image_path") or cols.get("image")
        instr_col = cols.get("instruction_native") or cols.get("instruction")
        if image_col is None or instr_col is None:
            return make_instruction(fallback_classes, self.dataset, self.rng)
        matches = df[df[image_col].astype(str).str.contains(image_id, regex=False, na=False)]
        if len(matches) == 0:
            return make_instruction(fallback_classes, self.dataset, self.rng)
        row = matches.iloc[0]
        return str(row[instr_col])

    def __len__(self) -> int:
        return len(self.records)

    def _load_sample(self, record: SampleRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        ds = record.dataset
        if ds == "consep":
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(record.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inst_map, type_map, native = parse_consep_mask(record.label_path)
            return image, inst_map, type_map, native
        if ds == "monusac":
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(record.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inst_map, type_map, native = parse_monusac_xml(record.label_path, image.shape[:2])
            return image, inst_map, type_map, native
        if ds == "lizard":
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(record.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if record.label_path is None:
                raise FileNotFoundError(f"Missing Lizard label for {record.image_id}")
            inst_map, type_map, native = parse_lizard_mask(record.label_path)
            return image, inst_map, type_map, native
        if ds == "conic":
            idx = int(record.image_id.split("_")[-1])
            image = np.asarray(self._conic_images[idx]).copy()
            image = _ensure_3ch(image)
            inst_map, type_map, native = parse_conic_sample(record.image_path, self._conic_labels, idx)
            return image, inst_map, type_map, native
        raise ValueError(ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image, inst_map, type_map, native = self._load_sample(record)

        image = _resize_if_needed(image, self.image_size)
        inst_map = _resize_if_needed(inst_map, self.image_size)
        type_map = _resize_if_needed(type_map, self.image_size)

        inst_map = remap_label(inst_map)
        np_map = (inst_map > 0).astype(np.int64)
        hv_map = gen_instance_hv_map(inst_map)
        dist_map = gen_normalized_distance_transform(inst_map)[..., None]

        image = image.astype(np.float32) / 255.0
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        else:
            image = np.repeat(image[None, ...], 3, axis=0)

        return {
            "image": torch.from_numpy(image).float(),
            "np_map": torch.from_numpy(np_map).long(),
            "hv_map": torch.from_numpy(hv_map.transpose(2, 0, 1)).float(),
            "type_map": torch.from_numpy(type_map.astype(np.int64)).long(),
            "instance_map": torch.from_numpy(inst_map.astype(np.int64)).long(),
            "dist_map": torch.from_numpy(dist_map.transpose(2, 0, 1)).float(),
            "instruction": record.instruction,
            "image_id": record.image_id,
            "tissue": record.tissue,
            "native_classes": native,
        }


class RawCollate:
    def __call__(self, batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        keys = ["image", "np_map", "hv_map", "type_map", "instance_map", "dist_map"]
        for key in keys:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
        out["instruction"] = [b["instruction"] for b in batch]
        out["image_id"] = [b["image_id"] for b in batch]
        out["tissue"] = [b["tissue"] for b in batch]
        out["native_classes"] = [b["native_classes"] for b in batch]
        return out


def compute_class_counts(dataset: ExternalRawDataset) -> List[int]:
    counts = np.zeros(6, dtype=np.int64)
    for rec in dataset.records:
        _, _, type_map, _ = dataset._load_sample(rec)
        for cls_id in range(6):
            counts[cls_id] += int((type_map == cls_id).sum() > 0)
    counts[0] = max(int(counts[0]), 1)
    return counts.tolist()


def build_model(model_name: str, num_classes: int, image_size: int, backbone: str) -> nn.Module:
    name = model_name.upper()
    if name == "LAVT":
        return create_lavt_model(num_classes=num_classes, img_size=image_size, pretrained=True, freeze_text_encoder=True)
    if name == "CRIS":
        return create_cris_model(num_classes=num_classes, freeze_text_encoder=True, freeze_image_encoder=False)
    if name == "LVIT":
        return create_lvit_model(num_classes=num_classes, freeze_text_encoder=True, img_size=image_size, backbone=backbone)
    if name == "GROUNDING_DINO":
        return create_grounding_dino_model(num_classes=num_classes, freeze_text_encoder=True, img_size=image_size)
    if name == "LVIT_IE":
        return create_lvit_ie_model(num_classes=num_classes, freeze_text_encoder=True, img_size=image_size, backbone=backbone)
    raise ValueError(f"Unsupported model: {model_name}")


class Trainer:
    def __init__(self, model_name: str, model: nn.Module, loss_fn: nn.Module, device: torch.device, lr: float, weight_decay: float):
        self.model_name = model_name.upper()
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        texts = batch["instruction"]
       
           # Model-specific forward calls (different parameter names)
           if self.model_name == "LAVT":
               return self.model(images=images, instructions=texts)
           if self.model_name == "LVIT_IE":
            return self.model(images=images, texts=texts, instance_maps=batch["instance_map"].to(self.device))
           if self.model_name == "LVIT5":
            return self.model(images=images, texts=texts, return_contrastive_features=True, return_grounding=True)
           if self.model_name == "LVIT4":
            return self.model(images=images, texts=texts, return_contrastive_features=True)
           # Default for CRIS, LVIT, GROUNDING_DINO
           return self.model(images=images, texts=texts)

    def make_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        targets = {
            "np": batch["np_map"].to(self.device),
            "type": batch["type_map"].to(self.device),
            "instance": batch["instance_map"].to(self.device),
        }
        if self.model_name == "LVIT_IE":
            targets["dist"] = batch["dist_map"].to(self.device)
        else:
            targets["hv"] = batch["hv_map"].to(self.device)
            targets["focus_mask"] = (batch["np_map"].to(self.device) > 0).float()
        return targets

    def step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        outputs = self.forward_batch(batch)
        targets = self.make_targets(batch)
        loss, loss_dict = self.loss_fn(outputs, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item()), loss_dict


def run_epoch(trainer: Trainer, loader: DataLoader, train: bool = True) -> Dict[str, float]:
    trainer.model.train(mode=train)
    agg: Dict[str, List[float]] = {}
    for batch in loader:
        if train:
            loss, loss_dict = trainer.step(batch)
        else:
            with torch.no_grad():
                outputs = trainer.forward_batch(batch)
                targets = trainer.make_targets(batch)
                loss, loss_dict = trainer.loss_fn(outputs, targets)
                loss = float(loss.detach().cpu().item())
        agg.setdefault("loss", []).append(loss)
        for k, v in loss_dict.items():
            if isinstance(v, (float, int)):
                agg.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in agg.items() if len(v) > 0}


def train_one_dataset(args: argparse.Namespace, dataset_name: str, root: Path) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_ds = ExternalRawDataset(dataset_name, root, "train", image_size=args.image_size, caption_csv=args.caption_csv, seed=args.seed)
    val_ds = ExternalRawDataset(dataset_name, root, "val", image_size=args.image_size, caption_csv=args.caption_csv, seed=args.seed)
    test_ds = ExternalRawDataset(dataset_name, root, "test", image_size=args.image_size, caption_csv=args.caption_csv, seed=args.seed)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found for {dataset_name} at {root}")

    class_counts = compute_class_counts(train_ds)
    model = build_model(args.model, num_classes=args.num_classes, image_size=args.image_size, backbone=args.backbone)
    if args.model.upper() == "LVIT_IE":
        loss_fn = LViTIELoss(num_classes=args.num_classes)
    else:
        loss_fn = CIPSNetV2Loss(num_classes=args.num_classes, cls_num_list=class_counts, use_ldam=True)

    trainer = Trainer(args.model, model, loss_fn, device=device, lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, collate_fn=RawCollate())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, collate_fn=RawCollate())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, collate_fn=RawCollate())

    out_dir = args.output_dir / dataset_name.lower() / args.model.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(trainer, train_loader, train=True)
        val_metrics = run_epoch(trainer, val_loader, train=False) if len(val_ds) > 0 else {"loss": float("nan")}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if not math.isnan(val_metrics.get("loss", float("nan"))) and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "dataset": dataset_name,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "class_counts": class_counts,
                },
                out_dir / "best.pth",
            )
        print(f"[{dataset_name}] epoch {epoch}/{args.epochs} train_loss={train_metrics.get('loss', float('nan')):.4f} val_loss={val_metrics.get('loss', float('nan')):.4f}")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    final_test = run_epoch(trainer, test_loader, train=False) if len(test_ds) > 0 else {}
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test, f, indent=2)

    return {
        "dataset": dataset_name,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "best_val_loss": best_val,
        "test_metrics": final_test,
        "class_counts": class_counts,
        "output_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native-caption raw training for external histopathology datasets")
    parser.add_argument("--model", required=True, choices=["LAVT", "CRIS", "LVIT", "LVIT_IE", "GROUNDING_DINO"])
    parser.add_argument("--datasets", nargs="+", default=["CoNSeP", "MoNuSAC", "Lizard", "CoNIC"])
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing CoNSeP/MoNuSAC/Lizard/CoNIC")
    parser.add_argument("--caption-csv", type=Path, default=None, help="Optional native caption CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cipsnet_v2/experiments/raw_training"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="vit")
    parser.add_argument("--num-classes", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}
    for dataset_name in args.datasets:
        root = args.data_root / dataset_name
        print(f"\n=== Training {args.model} on {dataset_name} ===")
        summary[dataset_name] = train_one_dataset(args, dataset_name, root)

    with open(args.output_dir / f"{args.model.lower()}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
