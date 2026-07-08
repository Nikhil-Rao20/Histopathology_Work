#!/usr/bin/env python3
"""
Zero-shot external evaluation for VL models compared in the main table.

Key features:
- Evaluates trained VL models (LAVT, CRIS, LVIT, LVIT_IE, GROUNDING_DINO by default)
- Uses external datasets: CoNIC, CoNSeP, Lizard, MoNuSAC
- Test split handling is strict (official test for CoNSeP/MoNuSAC, held-out split for CoNIC/Lizard)
- Prompt mode switch: --instruction-type native|canonical
- Saves detailed results:
  - per-model/per-dataset evaluator outputs (JSON + CSV)
  - global aggregate tables (overall + per-class)
  - semantic confusion-based metrics (mIoU, mDice, precision, recall, F1)

Usage examples:
  python experiments/cipsnet_v2/zero_shot_external_eval.py --instruction-type native
  python experiments/cipsnet_v2/zero_shot_external_eval.py --instruction-type canonical --models LVIT_IE LAVT
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Ensure local imports work when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cipsnet_v2.testing.evaluator import EvaluationConfig, PanNukeEvaluator
from experiments.cipsnet_v2.testing.instance_embed_postprocess import IEPostProcessConfig, process_batch_ie
from experiments.cipsnet_v2.testing.post_processing import PostProcessConfig, PostProcessor


# ============================================================
# Constants
# ============================================================

PANNUKE_ID_TO_NAME = {
    0: "background",
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "dead",
    5: "epithelial",
}

PANNUKE_EVAL_CLASS_IDS = [1, 2, 3, 4, 5]  # exclude background from macro summary

# Default model set aligned with your compared VL models
DEFAULT_MODEL_SPECS = {
    "LAVT": {
        "variant": "LAVT",
        "checkpoint": "experiments/cipsnet_v2/experiments/LAVT_20260213_000906_Static_Dataloader_Weighted_Focal_100epoch/fold_1/checkpoints/best.pth",
        "backbone": "vit",
    },
    "CRIS": {
        "variant": "CRIS",
        "checkpoint": "experiments/cipsnet_v2/experiments/CRIS_20260211_174530_Static_Dataloader/fold_1/checkpoints/best.pth",
        "backbone": "vit",
    },
    "LVIT": {
        "variant": "LVIT",
        "checkpoint": "experiments/cipsnet_v2/experiments/LVIT_20260213_000653_Permutation_Dataloader_Weighted_Focal_100epoch/fold_1/checkpoints/best.pth",
        "backbone": "vit",
    },
    "LVIT_IE": {
        "variant": "LVIT_IE",
        "checkpoint": "experiments/cipsnet_v2/experiments/LVIT_IE_DINOv2_ViTB14_Perm_Actual_100ep/fold_1/checkpoints/best.pth",
        "backbone": "dinov2_vit_b_14",
    },
    "GROUNDING_DINO": {
        "variant": "GROUNDING_DINO",
        "checkpoint": "experiments/cipsnet_v2/experiments/GROUNDING_DINO_20260213_001029_Static_Dataloader_Weighted_Focal_100epochs/fold_1/checkpoints/best.pth",
        "backbone": "vit",
    },
}

# Mappings from external datasets to PanNuke class IDs
CONIC_TO_PANNUKE = {
    1: 2,  # Neutrophil -> inflammatory
    2: 5,  # Epithelial -> epithelial
    3: 2,  # Lymphocyte -> inflammatory
    4: 2,  # Plasma -> inflammatory
    5: 2,  # Eosinophil -> inflammatory
    6: 3,  # Connective -> connective
}

CONSEP_TO_PANNUKE = {
    1: 0,  # Other -> ignore/background in canonical eval
    2: 2,  # Inflammatory
    3: 5,  # Healthy epithelial
    4: 1,  # Dysplastic/malignant epithelial -> neoplastic
    5: 3,  # Fibroblast -> connective
    6: 3,  # Muscle -> connective
    7: 3,  # Endothelial -> connective
}

LIZARD_TO_PANNUKE = CONIC_TO_PANNUKE.copy()

MONUSAC_TO_PANNUKE = {
    "Epithelial": 5,
    "Lymphocyte": 2,
    "Macrophage": 2,
    "Neutrophil": 2,
    "Ambiguous": 0,
}


# ============================================================
# Utility
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunningConfusion:
    """Accumulates confusion matrix and computes semantic metrics."""

    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        p = pred.flatten().astype(np.int64)
        t = true.flatten().astype(np.int64)
        valid = (t >= 0) & (t < self.num_classes)
        p = np.clip(p[valid], 0, self.num_classes - 1)
        t = np.clip(t[valid], 0, self.num_classes - 1)
        np.add.at(self.mat, (t, p), 1)

    def summarize(self) -> Dict[str, Any]:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + 1e-10)
        dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-10)

        eval_ids = PANNUKE_EVAL_CLASS_IDS

        return {
            "accuracy": float(tp.sum() / max(self.mat.sum(), 1)),
            "mIoU": float(np.nanmean(iou[eval_ids])),
            "mDice": float(np.nanmean(dice[eval_ids])),
            "mPrecision": float(np.nanmean(precision[eval_ids])),
            "mRecall": float(np.nanmean(recall[eval_ids])),
            "mF1": float(np.nanmean(f1[eval_ids])),
            "per_class_iou": {PANNUKE_ID_TO_NAME[i]: float(iou[i]) for i in range(self.num_classes)},
            "per_class_dice": {PANNUKE_ID_TO_NAME[i]: float(dice[i]) for i in range(self.num_classes)},
            "per_class_precision": {PANNUKE_ID_TO_NAME[i]: float(precision[i]) for i in range(self.num_classes)},
            "per_class_recall": {PANNUKE_ID_TO_NAME[i]: float(recall[i]) for i in range(self.num_classes)},
            "per_class_f1": {PANNUKE_ID_TO_NAME[i]: float(f1[i]) for i in range(self.num_classes)},
            "confusion_matrix": self.mat.tolist(),
        }


# ============================================================
# Dataset records
# ============================================================


@dataclass
class EvalRecord:
    dataset: str
    image_id: str
    image_path: str
    prompt: str
    tissue: str


class ExternalZeroShotDataset(Dataset):
    """Unified external dataset that returns image + GT maps + text instruction."""

    def __init__(
        self,
        dataset_name: str,
        instruction_type: str,
        root_dir: Path,
        captions_csv: Path,
        image_size: int = 256,
        lizard_holdout_split: int = 1,
        max_samples: int = 0,
    ):
        self.dataset_name = dataset_name
        self.instruction_type = instruction_type
        self.root_dir = root_dir
        self.captions_csv = captions_csv
        self.image_size = image_size
        self.lizard_holdout_split = lizard_holdout_split

        self._caption_lookup = self._build_caption_lookup()
        self._default_prompt = "Segment nuclei in this image."

        self._conic_images = None
        self._conic_labels = None
        if self.dataset_name == "CoNIC":
            # Keep memory-mapped arrays open for fast indexed access.
            self._conic_images = np.load(self.root_dir / "images.npy", mmap_mode="r")
            self._conic_labels = np.load(self.root_dir / "labels.npy", mmap_mode="r")

        self.records = self._build_records()
        if max_samples > 0:
            self.records = self.records[:max_samples]

        self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _build_caption_lookup(self) -> Dict[str, Dict[str, str]]:
        df = pd.read_csv(self.captions_csv)
        lookup: Dict[str, Dict[str, str]] = {}
        for _, row in df.iterrows():
            key = str(row.get("image_id", ""))
            if not key:
                continue
            lookup[key] = {
                "instruction": str(row.get("instruction", "")).strip(),
                "instruction_native": str(row.get("instruction_native", "")).strip(),
                "instruction_canonical": str(row.get("instruction_canonical", "")).strip(),
            }
        return lookup

    def _get_prompt(self, image_id: str) -> str:
        row = self._caption_lookup.get(str(image_id), None)
        if row is None:
            return self._default_prompt

        col = "instruction_native" if self.instruction_type == "native" else "instruction_canonical"
        txt = row.get(col, "")
        if txt:
            return txt
        txt = row.get("instruction", "")
        return txt if txt else self._default_prompt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        image, gt_inst, gt_type = self._load_sample(rec)

        # Resize image to model input size
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        gt_inst = cv2.resize(gt_inst, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt_type = cv2.resize(gt_type, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = (image - self.normalize_mean) / self.normalize_std
        image = np.transpose(image, (2, 0, 1))  # CHW

        return {
            "images": torch.from_numpy(image).float(),
            "instance_maps": torch.from_numpy(gt_inst).long(),
            "type_maps": torch.from_numpy(gt_type).long(),
            "instructions": rec.prompt,
            "tissues": rec.tissue,
            "indices": rec.image_id,
            "dataset": rec.dataset,
        }

    def _build_records(self) -> List[EvalRecord]:
        if self.dataset_name == "CoNSeP":
            return self._build_records_consep()
        if self.dataset_name == "MoNuSAC":
            return self._build_records_monusac()
        if self.dataset_name == "Lizard":
            return self._build_records_lizard()
        if self.dataset_name == "CoNIC":
            return self._build_records_conic()
        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _build_records_consep(self) -> List[EvalRecord]:
        records: List[EvalRecord] = []
        test_img_dir = self.root_dir / "Test" / "Images"

        for img_path in sorted(test_img_dir.glob("*.png")):
            img_id = img_path.stem
            prompt = self._get_prompt(img_id)
            records.append(
                EvalRecord(
                    dataset="CoNSeP",
                    image_id=img_id,
                    image_path=str(img_path),
                    prompt=prompt,
                    tissue="Colon",
                )
            )
        return records

    def _build_records_monusac(self) -> List[EvalRecord]:
        records: List[EvalRecord] = []
        test_dir = self.root_dir / "Test"

        for xml_path in sorted(test_dir.glob("**/*.xml")):
            img_path = xml_path.with_suffix(".tif")
            if not img_path.exists():
                continue
            img_id = xml_path.stem
            prompt = self._get_prompt(img_id)
            records.append(
                EvalRecord(
                    dataset="MoNuSAC",
                    image_id=img_id,
                    image_path=str(img_path),
                    prompt=prompt,
                    tissue="Multi_organ",
                )
            )
        return records

    def _build_records_lizard(self) -> List[EvalRecord]:
        records: List[EvalRecord] = []
        info_path = self.root_dir / "lizard_labels" / "Lizard_Labels" / "info.csv"
        info_df = pd.read_csv(info_path)
        holdout_df = info_df[info_df["Split"] == self.lizard_holdout_split]

        img1 = self.root_dir / "lizard_images1" / "Lizard_Images1"
        img2 = self.root_dir / "lizard_images2" / "Lizard_Images2"

        for _, row in holdout_df.iterrows():
            img_id = str(row["Filename"])
            p1 = img1 / f"{img_id}.png"
            p2 = img2 / f"{img_id}.png"
            if p1.exists():
                img_path = p1
            elif p2.exists():
                img_path = p2
            else:
                continue

            prompt = self._get_prompt(img_id)
            records.append(
                EvalRecord(
                    dataset="Lizard",
                    image_id=img_id,
                    image_path=str(img_path),
                    prompt=prompt,
                    tissue="Colon",
                )
            )
        return records

    def _build_records_conic(self) -> List[EvalRecord]:
        records: List[EvalRecord] = []

        # Build held-out source IDs from Lizard split mapping.
        info_path = self.root_dir.parent / "Lizard" / "lizard_labels" / "Lizard_Labels" / "info.csv"
        info_df = pd.read_csv(info_path)
        holdout_ids = set(info_df[info_df["Split"] == self.lizard_holdout_split]["Filename"].astype(str).tolist())

        patch_info = pd.read_csv(self.root_dir / "patch_info.csv")
        for idx, row in patch_info.iterrows():
            source = str(row["patch_info"]).split("-")[0]
            if source not in holdout_ids:
                continue
            img_id = str(idx)
            prompt = self._get_prompt(img_id)
            records.append(
                EvalRecord(
                    dataset="CoNIC",
                    image_id=img_id,
                    image_path=f"CoNIC/images.npy[{idx}]",
                    prompt=prompt,
                    tissue="Colon",
                )
            )
        return records

    def _load_sample(self, rec: EvalRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rec.dataset == "CoNSeP":
            return self._load_consep(rec)
        if rec.dataset == "MoNuSAC":
            return self._load_monusac(rec)
        if rec.dataset == "Lizard":
            return self._load_lizard(rec)
        if rec.dataset == "CoNIC":
            return self._load_conic(rec)
        raise ValueError(rec.dataset)

    def _load_consep(self, rec: EvalRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        mat_path = self.root_dir / "Test" / "Labels" / f"{rec.image_id}.mat"
        mat = sio.loadmat(mat_path)

        inst = mat["inst_map"].astype(np.int32)
        if "type_map" in mat:
            native_type = mat["type_map"].astype(np.int32)
        else:
            native_type = np.zeros_like(inst, dtype=np.int32)

        gt_type = np.zeros_like(native_type, dtype=np.int32)
        for k, v in CONSEP_TO_PANNUKE.items():
            gt_type[native_type == k] = v

        return img, inst, gt_type

    def _load_conic(self, rec: EvalRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = int(rec.image_id)
        img = self._conic_images[idx].astype(np.uint8)
        inst = self._conic_labels[idx, :, :, 0].astype(np.int32)
        native_type = self._conic_labels[idx, :, :, 1].astype(np.int32)

        gt_type = np.zeros_like(native_type, dtype=np.int32)
        for k, v in CONIC_TO_PANNUKE.items():
            gt_type[native_type == k] = v

        return img, inst, gt_type

    def _load_lizard(self, rec: EvalRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = np.array(Image.open(rec.image_path).convert("RGB"))

        mat_path = (
            self.root_dir
            / "lizard_labels"
            / "Lizard_Labels"
            / "Labels"
            / f"{rec.image_id}.mat"
        )
        mat = sio.loadmat(mat_path)

        inst = mat["inst_map"].astype(np.int32)
        id_arr = np.squeeze(mat["id"]).astype(np.int32)
        cls_arr = np.squeeze(mat["class"]).astype(np.int32)

        gt_type = np.zeros_like(inst, dtype=np.int32)
        id_to_cls = {int(i): int(c) for i, c in zip(id_arr.tolist(), cls_arr.tolist())}

        for inst_id, native_cls in id_to_cls.items():
            mapped = LIZARD_TO_PANNUKE.get(native_cls, 0)
            gt_type[inst == inst_id] = mapped

        return img, inst, gt_type

    def _load_monusac(self, rec: EvalRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        h, w = img.shape[:2]

        xml_path = Path(rec.image_path).with_suffix(".xml")
        root = ET.parse(xml_path).getroot()

        gt_inst = np.zeros((h, w), dtype=np.int32)
        gt_type = np.zeros((h, w), dtype=np.int32)
        inst_id = 1

        # Annotation-level class assignment
        for ann in root.findall(".//Annotation"):
            cls_name = None
            for attr in ann.findall(".//Attribute"):
                name = attr.attrib.get("Name", "")
                if name in MONUSAC_TO_PANNUKE:
                    cls_name = name
                    break
            if cls_name is None:
                continue

            class_id = MONUSAC_TO_PANNUKE.get(cls_name, 0)
            if class_id == 0:
                continue

            for region in ann.findall(".//Region"):
                verts = region.findall(".//Vertex")
                if len(verts) < 3:
                    continue

                pts = []
                for v in verts:
                    x = int(round(float(v.attrib.get("X", 0))))
                    y = int(round(float(v.attrib.get("Y", 0))))
                    pts.append([x, y])

                poly = np.array([pts], dtype=np.int32)

                temp = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(temp, poly, 1)

                gt_inst[temp == 1] = inst_id
                gt_type[temp == 1] = class_id
                inst_id += 1

        return img, gt_inst, gt_type


# ============================================================
# Model loading
# ============================================================


def load_model(variant: str, checkpoint_path: Path, device: str, backbone: str = "vit") -> torch.nn.Module:
    from experiments.cipsnet_v2.models import (
        create_cris_model,
        create_grounding_dino_model,
        create_lavt_model,
        create_lvit_ie_model,
        create_lvit_model,
    )

    variant_upper = variant.upper()
    num_classes = 6

    if variant_upper == "LAVT":
        model = create_lavt_model(
            num_classes=num_classes,
            img_size=256,
            pretrained=True,
            freeze_text_encoder=True,
        )
    elif variant_upper == "CRIS":
        model = create_cris_model(
            num_classes=num_classes,
            freeze_text_encoder=True,
            freeze_image_encoder=False,
        )
    elif variant_upper in ("LVIT", "LVIT2"):
        model = create_lvit_model(
            num_classes=num_classes,
            freeze_text_encoder=True,
            img_size=256,
            backbone=backbone,
            use_gradient_checkpointing=False,
        )
    elif variant_upper == "GROUNDING_DINO":
        model = create_grounding_dino_model(
            num_classes=num_classes,
            num_queries=100,
            freeze_text_encoder=True,
            img_size=256,
        )
    elif variant_upper == "LVIT_IE":
        model = create_lvit_ie_model(
            num_classes=num_classes,
            freeze_text_encoder=True,
            img_size=256,
            backbone=backbone,
            instance_embed_dim=16,
            use_gradient_checkpointing=False,
        )
    else:
        raise ValueError(f"Unsupported variant for this script: {variant}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def forward_variant(model: torch.nn.Module, variant: str, images: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
    variant_upper = variant.upper()

    if variant_upper == "LVIT_IE":
        return model(images=images, texts=texts, instance_maps=None)
    if variant_upper in ["CRIS", "LVIT", "LVIT2", "GROUNDING_DINO"]:
        return model(images=images, texts=texts)
    if variant_upper in ["LVIT3", "LVIT4"]:
        return model(images=images, texts=texts, return_contrastive_features=False)
    if variant_upper == "LVIT5":
        return model(images=images, texts=texts, return_contrastive_features=False, return_grounding=False)

    # LAVT and others use instructions keyword
    return model(images=images, instructions=texts)


# ============================================================
# Evaluation loop
# ============================================================


def evaluate_one_model_dataset(
    model: torch.nn.Module,
    variant: str,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    eval_config: EvaluationConfig,
) -> Dict[str, Any]:
    evaluator = PanNukeEvaluator(config=eval_config, device=str(device))
    confusion = RunningConfusion(num_classes=6)

    post = PostProcessor(
        PostProcessConfig(
            np_threshold=eval_config.np_threshold,
            min_instance_size=eval_config.min_instance_size,
            marker_erosion_size=eval_config.marker_erosion_size,
        )
    )
    ie_cfg = IEPostProcessConfig()

    for batch in tqdm(loader, desc=f"{variant}:{loader.dataset.dataset_name}", leave=False):
        images = batch["images"].to(device)
        gt_inst = batch["instance_maps"].cpu().numpy()
        gt_type = batch["type_maps"].cpu().numpy()
        texts = batch["instructions"]
        tissues = batch["tissues"]
        indices = batch["indices"]

        with torch.no_grad():
            outputs = forward_variant(model, variant, images, list(texts))

        if variant.upper() == "LVIT_IE":
            batch_results = process_batch_ie(
                pred_np=outputs["np"],
                pred_dist=outputs["dist"],
                pred_embed=outputs["embed"],
                pred_type=outputs["type"],
                config=ie_cfg,
            )
            evaluator.add_batch_from_postprocessed(
                postprocessed_results=batch_results,
                gt_instances=gt_inst,
                gt_types=gt_type,
                tissues=list(tissues),
                indices=list(indices),
            )
            for i, r in enumerate(batch_results):
                confusion.update(r["type_map"].astype(np.int32), gt_type[i].astype(np.int32))
        else:
            pred_np = outputs["np"].detach().cpu().numpy()
            pred_hv = outputs["hv"].detach().cpu().numpy()
            pred_type = outputs["type"].detach().cpu().numpy()

            post_batch = []
            for i in range(pred_np.shape[0]):
                res = post.process(pred_np[i], pred_hv[i], pred_type[i])
                post_batch.append(res)
                confusion.update(res["type_map"].astype(np.int32), gt_type[i].astype(np.int32))

            evaluator.add_batch_from_postprocessed(
                postprocessed_results=post_batch,
                gt_instances=gt_inst,
                gt_types=gt_type,
                tissues=list(tissues),
                indices=list(indices),
            )

    results = evaluator.compute_and_save(str(out_dir))
    semantic = confusion.summarize()

    with open(out_dir / "semantic_metrics.json", "w") as f:
        json.dump(semantic, f, indent=2)

    results["semantic"] = semantic
    return results


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot external evaluation for VL models")
    parser.add_argument("--instruction-type", type=str, default="native", choices=["native", "canonical"],
                        help="Which prompt column to use from caption CSV")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_SPECS.keys()),
                        help="Model keys to evaluate")
    parser.add_argument("--datasets", nargs="+", default=["CoNSeP", "MoNuSAC", "Lizard", "CoNIC"],
                        help="Datasets to evaluate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Debug cap per dataset (0 means full)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lizard-holdout-split", type=int, default=1,
                        help="Split value in Lizard info.csv treated as held-out test")
    parser.add_argument("--output-root", type=str,
                        default=str(PROJECT_ROOT / "results" / "zero_shot_external"),
                        help="Root directory for outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / args.instruction_type / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = PROJECT_ROOT / "Histopathology_Datasets_Official"
    captions_root = PROJECT_ROOT / "Dataset"

    dataset_cfg = {
        "CoNSeP": {
            "root": dataset_root / "CoNSeP",
            "captions": captions_root / "CoNSeP_Images_With_Unique_Labels_Refer_Segmentation_Task_FULL.csv",
        },
        "MoNuSAC": {
            "root": dataset_root / "MoNuSAC",
            "captions": captions_root / "MoNuSAC_Images_With_Unique_Labels_Refer_Segmentation_Task_FULL.csv",
        },
        "Lizard": {
            "root": dataset_root / "Lizard",
            "captions": captions_root / "Lizard_Images_With_Unique_Labels_Refer_Segmentation_Task_FULL.csv",
        },
        "CoNIC": {
            "root": dataset_root / "CoNIC",
            "captions": captions_root / "CoNIC_Images_With_Unique_Labels_Refer_Segmentation_Task_FULL.csv",
        },
    }

    # Save run config
    with open(output_root / "run_config.json", "w") as f:
        json.dump(
            {
                "instruction_type": args.instruction_type,
                "models": args.models,
                "datasets": args.datasets,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "lizard_holdout_split": args.lizard_holdout_split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "device": str(device),
            },
            f,
            indent=2,
        )

    overall_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    eval_config = EvaluationConfig()

    for model_key in args.models:
        if model_key not in DEFAULT_MODEL_SPECS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(DEFAULT_MODEL_SPECS.keys())}")

        spec = DEFAULT_MODEL_SPECS[model_key]
        ckpt = PROJECT_ROOT / spec["checkpoint"]
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        print(f"\n[Model] {model_key} -> {ckpt}")
        model = load_model(spec["variant"], ckpt, str(device), backbone=spec.get("backbone", "vit"))

        for ds_name in args.datasets:
            if ds_name not in dataset_cfg:
                raise ValueError(f"Unknown dataset: {ds_name}")

            cfg = dataset_cfg[ds_name]
            if not cfg["root"].exists():
                print(f"  [Skip] {ds_name} root missing: {cfg['root']}")
                continue
            if not cfg["captions"].exists():
                print(f"  [Skip] {ds_name} captions missing: {cfg['captions']}")
                continue

            dataset = ExternalZeroShotDataset(
                dataset_name=ds_name,
                instruction_type=args.instruction_type,
                root_dir=cfg["root"],
                captions_csv=cfg["captions"],
                image_size=256,
                lizard_holdout_split=args.lizard_holdout_split,
                max_samples=args.max_samples,
            )

            if len(dataset) == 0:
                print(f"  [Skip] {ds_name} has zero samples after split filtering")
                continue

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            print(f"  [Dataset] {ds_name}: {len(dataset)} samples")

            out_dir = output_root / model_key / ds_name
            out_dir.mkdir(parents=True, exist_ok=True)

            res = evaluate_one_model_dataset(
                model=model,
                variant=spec["variant"],
                loader=loader,
                out_dir=out_dir,
                device=device,
                eval_config=eval_config,
            )

            overall = res.get("overall", {})
            semantic = res.get("semantic", {})
            detection = res.get("detection", {})

            overall_rows.append(
                {
                    "model": model_key,
                    "variant": spec["variant"],
                    "dataset": ds_name,
                    "instruction_type": args.instruction_type,
                    "n_samples": len(dataset),
                    "dice": overall.get("dice", np.nan),
                    "aji": overall.get("aji", np.nan),
                    "aji_plus": overall.get("aji_plus", np.nan),
                    "dq": overall.get("dq", np.nan),
                    "sq": overall.get("sq", np.nan),
                    "bpq": overall.get("bpq", np.nan),
                    "mpq": overall.get("mpq", np.nan),
                    "det_precision": detection.get("overall_precision", np.nan),
                    "det_recall": detection.get("overall_recall", np.nan),
                    "det_f1": detection.get("overall_f1", np.nan),
                    "accuracy": semantic.get("accuracy", np.nan),
                    "mIoU": semantic.get("mIoU", np.nan),
                    "mDice": semantic.get("mDice", np.nan),
                    "mPrecision": semantic.get("mPrecision", np.nan),
                    "mRecall": semantic.get("mRecall", np.nan),
                    "mF1": semantic.get("mF1", np.nan),
                }
            )

            class_wise = res.get("class_wise", {})
            per_cls_iou = semantic.get("per_class_iou", {})
            per_cls_dice = semantic.get("per_class_dice", {})
            per_cls_prec = semantic.get("per_class_precision", {})
            per_cls_rec = semantic.get("per_class_recall", {})
            per_cls_f1 = semantic.get("per_class_f1", {})

            for class_name in ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]:
                cw = class_wise.get(class_name, {})
                per_class_rows.append(
                    {
                        "model": model_key,
                        "variant": spec["variant"],
                        "dataset": ds_name,
                        "instruction_type": args.instruction_type,
                        "class": class_name,
                        "pq_mean": cw.get("pq_mean", np.nan),
                        "pq_std": cw.get("pq_std", np.nan),
                        "iou": per_cls_iou.get(class_name, np.nan),
                        "dice": per_cls_dice.get(class_name, np.nan),
                        "precision": per_cls_prec.get(class_name, np.nan),
                        "recall": per_cls_rec.get(class_name, np.nan),
                        "f1": per_cls_f1.get(class_name, np.nan),
                    }
                )

        del model
        torch.cuda.empty_cache()

    overall_df = pd.DataFrame(overall_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    overall_csv = output_root / "overall_metrics_table.csv"
    per_class_csv = output_root / "per_class_metrics_table.csv"

    overall_df.to_csv(overall_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    # Also export grouped pivots for quick paper table drafting
    if len(overall_df) > 0:
        pivot_mpq = overall_df.pivot_table(index="model", columns="dataset", values="mpq", aggfunc="mean")
        pivot_mpq.to_csv(output_root / "pivot_mpq.csv")

        pivot_miou = overall_df.pivot_table(index="model", columns="dataset", values="mIoU", aggfunc="mean")
        pivot_miou.to_csv(output_root / "pivot_miou.csv")

    print("\n" + "=" * 80)
    print("ZERO-SHOT EXTERNAL EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results root: {output_root}")
    print(f"Overall table: {overall_csv}")
    print(f"Per-class table: {per_class_csv}")


if __name__ == "__main__":
    main()
