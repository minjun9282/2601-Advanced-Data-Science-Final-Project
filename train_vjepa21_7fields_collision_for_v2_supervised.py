#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/evaluate V-JEPA2 with multi-task heads:
1) 7-field semantic classification
2) collision risk prediction (binary)
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

import train_test_lstm_ver4_1 as eval_ref


SEED = 42
DEFAULT_MODEL = "vjepa2_1_vitl_dist_vitG_384"
DEFAULT_LABELS = "output2/labels_for_vjepa_v2_supervised.jsonl"
DEFAULT_SPLITS = "data/train_test_clips_for_vjepa2.jsonl"
DEFAULT_FRAMES_ROOT = "data/frames"
DEFAULT_OFFICIAL_REPO_ROOT = "external/vjepa2_official"

FIELDS = ["PHT", "HPOS", "HMOT", "HPROX", "PATH", "GAP", "LATORIG"]
FIELD_CLASS_NAMES = {
    "PHT": ["car", "truck", "bus", "motorcyclist", "pedestrian", "cyclist", "roadside_object", "none"],
    "HPOS": ["ego_lane_front", "adjacent_left", "adjacent_right", "crossing_ahead", "roadside", "none"],
    "HMOT": ["stationary", "slowing", "moving_steady", "accelerating", "entering_lane", "crossing", "parked", "none"],
    "HPROX": ["very_close", "close", "medium", "far", "none"],
    "PATH": ["in_path", "entering_path", "crossing_path", "parallel_adjacent", "none"],
    "GAP": ["closing", "stable_gap", "none"],
    "LATORIG": ["left", "right", "none"],
}
FIELD_NUM_CLASSES = {k: len(v) for k, v in FIELD_CLASS_NAMES.items()}
ANALYSIS_FIELD_KEYS = {
    "PHT": "primary_hazard_type",
    "HPOS": "hazard_position",
    "HMOT": "hazard_motion_state",
    "HPROX": "hazard_proximity",
    "PATH": "path_relation",
    "GAP": "gap_trend",
    "LATORIG": "lateral_origin",
}
PATH_REL_FOR_LATERAL = {"entering_path", "crossing_path"}

FIELD_VALUE_ALIASES = {
    "PHT": {"motorcycle": "motorcyclist", "bicycle": "cyclist", "obstacle": "roadside_object", "unknown": "none"},
    "HPOS": {"ego_lane_front_left": "ego_lane_front", "ego_lane_front_right": "ego_lane_front", "roadside_left": "roadside", "roadside_right": "roadside", "unknown": "none"},
    "HMOT": {"movin_steady": "moving_steady", "unknown": "none"},
    "HPROX": {"contact_imminent": "very_close", "unknown": "none"},
    "PATH": {"outside_path": "parallel_adjacent", "unknown": "none"},
    "GAP": {"closing_fast": "closing", "opening": "stable_gap", "unclear": "none", "unknown": "none"},
    "LATORIG": {"center": "none", "unknown": "none"},
}

DEFAULT_SEMANTIC_FIELD_WEIGHTS = {
    "PHT": 0.05,
    "HPOS": 0.30,
    "HMOT": 0.15,
    "HPROX": 0.25,
    "PATH": 0.20,
    "GAP": 0.03,
    "LATORIG": 0.02,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_splits(path: str) -> Dict[str, set]:
    out = {"train": set(), "val": set(), "test": set()}
    for row in read_jsonl(path):
        split = (row.get("split") or "").strip().lower()
        category = (row.get("category") or "").strip().lower()
        clip_name = str(row.get("clip_name") or "").strip()
        if split not in out:
            continue
        if category not in ("crash", "normal"):
            continue
        if not clip_name:
            continue
        out[split].add((category, clip_name))
    return out


def parse_thresholds(csv_text: str) -> List[float]:
    out = []
    for tok in (csv_text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        return [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    return out


def parse_semantic_field_weights(spec: str) -> Dict[str, float]:
    out = {f: 1.0 for f in FIELDS}
    txt = (spec or "").strip()
    if not txt:
        return out

    for tok in txt.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Invalid --semantic_field_weights token='{tok}'. Expected FIELD:weight")
        field, w_txt = tok.split(":", 1)
        field = field.strip().upper()
        if field not in out:
            raise ValueError(f"Unknown field in --semantic_field_weights: '{field}'")
        w = float(w_txt.strip())
        if w < 0:
            raise ValueError(f"Field weight must be >= 0, got {field}:{w}")
        out[field] = w
    return out


def count_trainable_parameters(module: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def parse_lora_target_modules(spec: str) -> List[str]:
    mods = parse_csv_strs(spec)
    return mods if mods else ["qkv", "proj"]


def get_field_class_idx(field: str, value: str) -> int:
    options = FIELD_CLASS_NAMES[field]
    v = (value or "").strip().lower()
    if v not in options:
        if "unknown" in options:
            v = "unknown"
        elif "none" in options:
            v = "none"
        else:
            v = options[0]
    return options.index(v)


def normalize_field_value(field: str, value: str) -> str:
    v = (value or "").strip().lower()
    v = FIELD_VALUE_ALIASES.get(field, {}).get(v, v)
    options = FIELD_CLASS_NAMES[field]
    if v in options:
        return v
    if "none" in options:
        return "none"
    return options[0]


def parse_analysis_fields(window: Dict[str, Any]) -> Dict[str, str]:
    analysis = window.get("analysis") or {}
    fields = analysis.get("fields") if isinstance(analysis, dict) else None
    if not isinstance(fields, dict):
        return {f: "none" for f in FIELDS}

    out: Dict[str, str] = {}
    for f in FIELDS:
        key = ANALYSIS_FIELD_KEYS[f]
        raw_val = str(fields.get(key, "none") or "none")
        out[f] = normalize_field_value(f, raw_val)

    # lateral_origin is meaningful only when path_relation indicates entering/crossing.
    if out["PATH"] not in PATH_REL_FOR_LATERAL:
        out["LATORIG"] = "none"
    return out


def build_frame_window_indices(target_frame_idx: int, window_frames: int) -> List[int]:
    start = target_frame_idx - window_frames + 1
    idxs = list(range(start, target_frame_idx + 1))
    if not idxs:
        return []
    # Left-pad with first valid index for robustness at sequence start.
    if len(idxs) < window_frames:
        idxs = [idxs[0]] * (window_frames - len(idxs)) + idxs
    return idxs


def parse_triplet_indices(window: Dict[str, Any]) -> Optional[List[int]]:
    fi = window.get("frame_indices")
    if not isinstance(fi, list) or len(fi) != 3:
        return None
    try:
        return [int(fi[0]), int(fi[1]), int(fi[2])]
    except Exception:
        return None


def build_samples(
    windows: List[Dict[str, Any]],
    allowed_clips: set,
    frames_root: str,
    window_frames: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = Counter()
    samples: List[Dict[str, Any]] = []

    for w in windows:
        stats["total"] += 1
        category = (w.get("category") or "").strip().lower()
        clip_name = str(w.get("clip_name") or "").strip()
        key = (category, clip_name)
        if key not in allowed_clips:
            stats["skip_split"] += 1
            continue

        try:
            end_seg = int(w.get("target_frame_idx"))
        except Exception:
            stats["skip_bad_target_frame"] += 1
            continue

        # Keep frame selection aligned with label definitions:
        # - wf=3  : use explicit triplet (e.g., 0-5-10 / 5-10-15)
        # - wf=11 : use full range between triplet start/end (e.g., 0..10 / 5..15)
        triplet = parse_triplet_indices(w)
        idxs: List[int]
        if triplet is not None and window_frames == 3:
            idxs = triplet
        elif triplet is not None and window_frames == (triplet[-1] - triplet[0] + 1):
            idxs = list(range(triplet[0], triplet[-1] + 1))
        else:
            idxs = build_frame_window_indices(end_seg, window_frames)

        if len(idxs) != window_frames:
            stats["skip_bad_window_indices"] += 1
            continue

        paths = [os.path.join(frames_root, category, clip_name, f"{idx:05d}.jpg") for idx in idxs]
        if not all(os.path.exists(p) for p in paths):
            stats["skip_missing_frames"] += 1
            continue

        fvals = parse_analysis_fields(w)
        labels = {
            f: get_field_class_idx(f, fvals[f]) for f in FIELDS
        }
        alert_value = str(w.get("alert", "no-alert")).strip().lower()
        collision_label = 1 if alert_value == "alert" else 0

        samples.append(
            {
                "category": category,
                "clip_name": clip_name,
                "end_seg": end_seg,
                "frame_paths": paths,
                "field_labels": labels,
                "collision_label": collision_label,
            }
        )
        stats["kept"] += 1

    return samples, dict(stats)


class WindowSemanticCollisionDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], processor: Any):
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        frames = [np.array(Image.open(p).convert("RGB")) for p in s["frame_paths"]]
        inp = self.processor(np.array(frames), return_tensors="pt")
        if "pixel_values_videos" in inp:
            pixels = inp["pixel_values_videos"].squeeze(0)
        elif "pixel_values" in inp:
            pixels = inp["pixel_values"].squeeze(0)
        else:
            pixels = list(inp.values())[0].squeeze(0)

        field_labels = torch.tensor([int(s["field_labels"][f]) for f in FIELDS], dtype=torch.long)
        collision_label = torch.tensor(float(s["collision_label"]), dtype=torch.float32)
        return {
            "pixel_values": pixels,
            "field_labels": field_labels,
            "collision_label": collision_label,
            "clip_name": s["clip_name"],
            "end_seg": int(s["end_seg"]),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "field_labels": torch.stack([b["field_labels"] for b in batch], dim=0),
        "collision_label": torch.stack([b["collision_label"] for b in batch], dim=0),
        "clip_name": [b["clip_name"] for b in batch],
        "end_seg": torch.tensor([int(b["end_seg"]) for b in batch], dtype=torch.long),
    }


def _resolve_vjepa21_student_variant(model_name: str) -> str:
    raw = str(model_name or "")
    lower = raw.lower()
    token = lower
    if "vjepa2_1_" in lower:
        token = lower.split("vjepa2_1_", 1)[1]
    token = token.split("_dist", 1)[0]
    token = token.split("_", 1)[0]

    if token in ("vitb", "vit_base", "base"):
        return "base"
    if token in ("vitl", "vit_large", "large"):
        return "large"
    if token in ("vith", "vit_huge", "huge"):
        return "huge"
    if token in ("vitg", "vit_giant", "giant"):
        return "giant"
    if token in ("vitgiga", "gigantic", "vit_gigantic"):
        return "gigantic"

    if "vitb" in lower:
        return "base"
    if "vitl" in lower:
        return "large"
    if "vith" in lower:
        return "huge"
    if "gigantic" in lower:
        return "gigantic"
    if "vitg" in lower:
        return "giant"
    return "large"


def infer_hidden_dim(model_name: str) -> int:
    variant = _resolve_vjepa21_student_variant(model_name)
    if variant == "base":
        return 768
    if variant == "large":
        return 1024
    if variant == "huge":
        return 1280
    if variant == "giant":
        return 1408
    if variant == "gigantic":
        return 1664
    return 1024


def parse_csv_strs(csv_text: str) -> List[str]:
    out: List[str] = []
    for tok in (csv_text or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def ensure_official_repo_on_path(repo_root: str) -> None:
    repo_abs = os.path.abspath(repo_root)
    if not os.path.isdir(repo_abs):
        raise FileNotFoundError(f"official repo root not found: {repo_root}")
    if repo_abs not in sys.path:
        sys.path.insert(0, repo_abs)


class OfficialVJepa21VideoProcessor:
    """
    Lightweight wrapper so dataset call-site stays unchanged:
      processor(np.array(frames), return_tensors="pt") -> {"pixel_values_videos": [1,C,T,H,W]}
    """

    def __init__(self, repo_root: str, crop_size: int = 384):
        ensure_official_repo_on_path(repo_root)
        from evals.video_classification_frozen.utils import make_transforms

        self.crop = int(crop_size)
        self.shortest_edge = int(round(self.crop * 256.0 / 224.0))
        self.transform = make_transforms(training=False, crop_size=self.crop)
        # Keep these fields for run_meta compatibility.
        self.do_resize = True
        self.size = {"shortest_edge": self.shortest_edge}
        self.do_center_crop = True
        self.crop_size = {"height": self.crop, "width": self.crop}

    def __call__(self, frames: np.ndarray, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        views = self.transform(frames)
        if not isinstance(views, list) or len(views) == 0:
            raise RuntimeError("official transform returned no views")
        x = views[0]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        # expected: C,T,H,W
        if x.ndim != 4:
            raise RuntimeError(f"unexpected transformed tensor shape: {tuple(x.shape)}")
        if return_tensors != "pt":
            raise ValueError("OfficialVJepa21VideoProcessor supports return_tensors='pt' only")
        return {"pixel_values_videos": x.unsqueeze(0)}


def _strip_prefixes_from_state_dict(
    state: Dict[str, torch.Tensor],
    prefixes: List[str],
) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = str(k)
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        cleaned[nk] = v
    return cleaned


def resolve_vjepa21_backbone_family(model_name: str) -> str:
    variant = _resolve_vjepa21_student_variant(model_name)
    if variant == "gigantic":
        return "gigantic"
    if variant == "giant":
        return "giant"
    if variant == "large":
        return "large"
    if variant == "base":
        return "base"
    return "large"


def build_official_vjepa21_encoder(repo_root: str, model_name: str) -> nn.Module:
    ensure_official_repo_on_path(repo_root)
    from src.hub.backbones import (
        vjepa2_1_vit_base_384,
        vjepa2_1_vit_giant_384,
        vjepa2_1_vit_gigantic_384,
        vjepa2_1_vit_large_384,
    )

    family = resolve_vjepa21_backbone_family(model_name)
    if family == "base":
        encoder, _ = vjepa2_1_vit_base_384(pretrained=False)
    elif family == "large":
        encoder, _ = vjepa2_1_vit_large_384(pretrained=False)
    elif family == "giant":
        encoder, _ = vjepa2_1_vit_giant_384(pretrained=False)
    elif family == "gigantic":
        encoder, _ = vjepa2_1_vit_gigantic_384(pretrained=False)
    else:
        raise ValueError(f"Unsupported V-JEPA2.1 model family for model_name={model_name}")
    return encoder


def load_backbone_weights_from_vjepa_ckpt(
    backbone: nn.Module,
    ckpt_path: str,
    ckpt_key: str = "ema_encoder",
    strip_prefixes_csv: str = "module.backbone.,backbone.,module.",
    ckpt_format: str = "raw",
    strict: bool = False,
) -> Dict[str, Any]:
    if not ckpt_path:
        return {"loaded": False}
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"backbone_ckpt not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint type at {ckpt_path}: {type(raw)}")

    state = raw
    if ckpt_key:
        if ckpt_key not in raw:
            raise KeyError(f"Checkpoint key '{ckpt_key}' not found in {ckpt_path}; top-level keys={list(raw.keys())}")
        state = raw[ckpt_key]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint subkey '{ckpt_key}' is not a state-dict: {type(state)}")

    prefixes = parse_csv_strs(strip_prefixes_csv)
    cleaned = _strip_prefixes_from_state_dict(state, prefixes)
    incompatible = backbone.load_state_dict(cleaned, strict=bool(strict))
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    return {
        "loaded": True,
        "source_ckpt": ckpt_path,
        "source_key": ckpt_key,
        "ckpt_format": str(ckpt_format),
        "strict": bool(strict),
        "num_source_keys": len(state),
        "num_loaded_keys": len(cleaned),
        "num_missing_keys": len(missing),
        "num_unexpected_keys": len(unexpected),
        "missing_keys_head": missing[:20],
        "unexpected_keys_head": unexpected[:20],
    }


class VJEPA2SemanticCollision(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, official_repo_root: str = DEFAULT_OFFICIAL_REPO_ROOT):
        super().__init__()
        self.backbone = build_official_vjepa21_encoder(official_repo_root, model_name=model_name)
        hidden = int(getattr(self.backbone, "embed_dim", 1024))
        self.dropout = nn.Dropout(dropout)
        self.semantic_heads = nn.ModuleDict({f: nn.Linear(hidden, FIELD_NUM_CLASSES[f]) for f in FIELDS})
        self.collision_head = nn.Linear(hidden, 1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # official encoder expects [B,C,T,H,W]
        feats = self.backbone(pixel_values, training=False)

        if feats.dim() == 3:
            pooled = feats.mean(dim=1)
        elif feats.dim() == 2:
            pooled = feats
        else:
            pooled = feats.view(feats.size(0), -1)

        pooled = self.dropout(pooled)
        semantic_logits = {f: self.semantic_heads[f](pooled) for f in FIELDS}
        collision_logit = self.collision_head(pooled).squeeze(-1)
        if return_features:
            return semantic_logits, collision_logit, pooled
        return semantic_logits, collision_logit


def compute_field_class_weights(
    samples: List[Dict[str, Any]],
    power: float = 1.0,
    min_w: float = 0.0,
    max_w: float = 0.0,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for field in FIELDS:
        counts = [0] * FIELD_NUM_CLASSES[field]
        for s in samples:
            counts[int(s["field_labels"][field])] += 1
        c = np.array(counts, dtype=np.float32)
        c = np.maximum(c, 1.0)
        w = c.sum() / (len(c) * c)
        w = np.power(w, float(power))
        w = w / w.mean()
        if float(min_w) > 0.0 or float(max_w) > 0.0:
            lo = float(min_w) if float(min_w) > 0.0 else float(w.min())
            hi = float(max_w) if float(max_w) > 0.0 else float(w.max())
            if hi < lo:
                hi = lo
            w = np.clip(w, lo, hi)
            w = w / max(1e-9, float(w.mean()))
        out[field] = torch.tensor(w, dtype=torch.float32)
    return out


def compute_collision_pos_weight(samples: List[Dict[str, Any]]) -> torch.Tensor:
    pos = sum(int(s["collision_label"]) == 1 for s in samples)
    neg = sum(int(s["collision_label"]) == 0 for s in samples)
    pos = max(pos, 1)
    neg = max(neg, 1)
    return torch.tensor([float(neg) / float(pos)], dtype=torch.float32)


def parse_contrastive_fields(spec: str) -> List[str]:
    vals = []
    for tok in parse_csv_strs(spec):
        name = tok.strip().upper()
        if name == "COL":
            name = "COLLISION"
        if name != "COLLISION" and name not in FIELDS:
            raise ValueError(f"Unknown contrastive field: {tok}")
        vals.append(name)
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def compute_balanced_class_weights_from_counts(
    counts: List[int],
    power: float = 0.5,
    max_w: float = 3.0,
) -> torch.Tensor:
    c = np.array(counts, dtype=np.float32)
    c = np.maximum(c, 1.0)
    w = c.sum() / (len(c) * c)
    w = np.power(w, float(power))
    if float(max_w) > 0.0:
        w = np.clip(w, 0.0, float(max_w))
    w = w / max(1e-9, float(w.mean()))
    return torch.tensor(w, dtype=torch.float32)


def compute_contrastive_class_weights(
    samples: List[Dict[str, Any]],
    active_fields: List[str],
    power: float = 0.5,
    max_w: float = 3.0,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for field in active_fields:
        if field == "COLLISION":
            counts = [0, 0]
            for s in samples:
                counts[int(s["collision_label"])] += 1
        else:
            counts = [0] * FIELD_NUM_CLASSES[field]
            for s in samples:
                counts[int(s["field_labels"][field])] += 1
        out[field] = compute_balanced_class_weights_from_counts(counts, power=power, max_w=max_w)
    return out


def weighted_supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.new_zeros(())
    out_dtype = z.dtype
    z = F.normalize(z.float(), dim=-1)
    logits = ((z @ z.t()) / float(max(temperature, 1e-6))).float()
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    same = labels[:, None].eq(labels[None, :]) & (~eye)
    valid = same.any(dim=1)
    if not bool(valid.any()):
        return z.new_zeros((), dtype=out_dtype)
    exp_mask = ~eye
    log_prob = logits - torch.logsumexp(logits.masked_fill(~exp_mask, -1e9), dim=1, keepdim=True)
    pos_log_prob = (log_prob * same.to(log_prob.dtype)).sum(dim=1) / same.sum(dim=1).clamp_min(1)
    if class_weights is None:
        anchor_w = torch.ones_like(pos_log_prob, dtype=log_prob.dtype, device=log_prob.device)
    else:
        anchor_w = class_weights.to(device=labels.device, dtype=log_prob.dtype)[labels]
    anchor_w = anchor_w[valid]
    pos_log_prob = pos_log_prob[valid]
    return (-(anchor_w * pos_log_prob).sum() / anchor_w.sum().clamp_min(1e-6)).to(dtype=out_dtype)


def safe_ap_auc(y_true: List[int], y_score: List[float]) -> Tuple[float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan")
    if len(set(y_true)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y_true, y_score)), float(roc_auc_score(y_true, y_score))


def weighted_field_macro_f1(metrics: Dict[str, Any], field_weights: Dict[str, float]) -> float:
    per_field = metrics["semantic"]["per_field"]
    num = 0.0
    den = 0.0
    for f in FIELDS:
        w = float(field_weights.get(f, 1.0))
        if w <= 0.0:
            continue
        num += w * float(per_field[f]["macro_f1"])
        den += w
    if den <= 0.0:
        return float(metrics["semantic"]["overall"]["mean_field_macro_f1"])
    return float(num / den)


def score_for_model_selection(
    metrics: Dict[str, Any],
    metric_name: str,
    semantic_weight: float,
    semantic_field_weights: Dict[str, float],
    semantic_priority_hprox_weight: float,
) -> float:
    def _safe_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    sem_f1 = float(metrics["semantic"]["overall"]["mean_field_macro_f1"])
    sem_f1_weighted = weighted_field_macro_f1(metrics, semantic_field_weights)
    hprox_soft = float(metrics["semantic"]["overall"].get("hprox_soft_accuracy", 0.0))
    col_ap = _safe_float(metrics["collision"]["clip_ap_precrash_max"])
    col_auc = _safe_float(metrics["collision"]["clip_auc_precrash_max"])
    if np.isnan(col_ap):
        col_ap = -1e9
    if np.isnan(col_auc):
        col_auc = -1e9

    if metric_name == "collision_clip_ap":
        return col_ap
    if metric_name == "collision_clip_auc":
        return col_auc
    if metric_name == "semantic_macro_f1":
        return sem_f1
    if metric_name == "semantic_weighted_macro_f1":
        return sem_f1_weighted
    if metric_name == "semantic_priority":
        return sem_f1_weighted + float(semantic_priority_hprox_weight) * hprox_soft
    if metric_name == "joint":
        return col_ap + semantic_weight * sem_f1
    raise ValueError(f"Unknown model_select_metric={metric_name}")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    field_class_weights: Dict[str, torch.Tensor],
    collision_pos_weight: torch.Tensor,
    lambda_semantic: float,
    lambda_collision: float,
    collision_seg: int,
    time_stride: float,
    thresholds: List[float],
    hprox_near_miss_credit: float,
    collect_window_outputs: bool = False,
) -> Dict[str, Any]:
    model.eval()

    all_true_fields = {f: [] for f in FIELDS}
    all_pred_fields = {f: [] for f in FIELDS}
    all_true_collision: List[int] = []
    all_score_collision: List[float] = []
    clip_probs: Dict[str, Dict[str, Any]] = {}
    window_records: List[Dict[str, Any]] = []

    total_loss = 0.0
    total_sem_loss = 0.0
    total_col_loss = 0.0
    n = 0

    precrash_end = int(collision_seg) - 1

    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            labels_field = batch["field_labels"].to(device)
            labels_col = batch["collision_label"].to(device)
            clip_names = batch["clip_name"]
            end_segs = batch["end_seg"]

            logits_field, logits_col = model(pixels)

            loss_sem = 0.0
            pred_field_idx: Dict[str, torch.Tensor] = {}
            for i, f in enumerate(FIELDS):
                w = field_class_weights[f].to(device=device, dtype=logits_field[f].dtype)
                loss_sem = loss_sem + F.cross_entropy(logits_field[f], labels_field[:, i], weight=w)
                pred_field_idx[f] = torch.argmax(logits_field[f], dim=1)
                all_true_fields[f].extend(labels_field[:, i].detach().cpu().tolist())
                all_pred_fields[f].extend(pred_field_idx[f].detach().cpu().tolist())

            pos_w = collision_pos_weight.to(device=device, dtype=logits_col.dtype)
            loss_col = F.binary_cross_entropy_with_logits(
                logits_col,
                labels_col.to(dtype=logits_col.dtype),
                pos_weight=pos_w,
            )
            loss = lambda_semantic * loss_sem + lambda_collision * loss_col

            probs_col = torch.sigmoid(logits_col).detach().cpu().numpy()
            y_col = labels_col.detach().cpu().numpy().astype(np.int64)
            all_true_collision.extend(y_col.tolist())
            all_score_collision.extend(probs_col.tolist())

            for b in range(len(clip_names)):
                clip = str(clip_names[b])
                end_seg = int(end_segs[b].item())
                label = int(y_col[b])
                p = float(probs_col[b])
                if clip not in clip_probs:
                    clip_probs[clip] = {"label": label, "probs_by_end_seg": {}}
                else:
                    # For window-level labels, clip label must be "any alert in clip".
                    clip_probs[clip]["label"] = max(int(clip_probs[clip]["label"]), label)
                d = clip_probs[clip]["probs_by_end_seg"]
                d[end_seg] = max(float(d.get(end_seg, 0.0)), p)

                if collect_window_outputs:
                    rec = {
                        "clip_name": clip,
                        "end_seg": end_seg,
                        "label_collision": label,
                        "pred_collision_prob": p,
                        "pred_fields": {f: int(pred_field_idx[f][b].item()) for f in FIELDS},
                        "field_probs": {
                            f: [float(x) for x in torch.softmax(logits_field[f][b], dim=0).detach().cpu().tolist()]
                            for f in FIELDS
                        },
                        "gt_fields": {f: int(labels_field[b, i].item()) for i, f in enumerate(FIELDS)},
                    }
                    window_records.append(rec)

            bs = labels_field.size(0)
            total_loss += float(loss.item()) * bs
            total_sem_loss += float(loss_sem.item()) * bs
            total_col_loss += float(loss_col.item()) * bs
            n += bs

    semantic_per_field = {}
    sem_accs = []
    sem_macro_f1s = []
    for f in FIELDS:
        y_true = np.array(all_true_fields[f], dtype=np.int64)
        y_pred = np.array(all_pred_fields[f], dtype=np.int64)
        acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
        n_cls = FIELD_NUM_CLASSES[f]
        f1s = []
        for c in range(n_cls):
            tp = int(((y_true == c) & (y_pred == c)).sum())
            fp = int(((y_true != c) & (y_pred == c)).sum())
            fn = int(((y_true == c) & (y_pred != c)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        semantic_per_field[f] = {"accuracy": acc, "macro_f1": macro_f1}
        sem_accs.append(acc)
        sem_macro_f1s.append(macro_f1)

    hprox_soft_accuracy = 0.0
    if "HPROX" in all_true_fields and len(all_true_fields["HPROX"]) > 0:
        y_true_h = np.array(all_true_fields["HPROX"], dtype=np.int64)
        y_pred_h = np.array(all_pred_fields["HPROX"], dtype=np.int64)
        soft = np.zeros(len(y_true_h), dtype=np.float32)
        soft[y_true_h == y_pred_h] = 1.0
        hprox_classes = FIELD_CLASS_NAMES["HPROX"]
        if "very_close" in hprox_classes and "close" in hprox_classes:
            idx_vclose = hprox_classes.index("very_close")
            idx_close = hprox_classes.index("close")
            near = ((y_true_h == idx_vclose) & (y_pred_h == idx_close)) | (
                (y_true_h == idx_close) & (y_pred_h == idx_vclose)
            )
            soft[near] = float(hprox_near_miss_credit)
        hprox_soft_accuracy = float(soft.mean()) if len(soft) > 0 else 0.0

    window_ap, window_auc = safe_ap_auc(all_true_collision, all_score_collision)
    clip_ap, clip_auc = eval_ref.clip_level_precrash_max_ap_auc(clip_probs=clip_probs, precrash_end=precrash_end)
    op_df = eval_ref.operating_metrics_with_ttc(
        clip_probs=clip_probs,
        thresholds=thresholds,
        precrash_end=precrash_end,
        collision_seg=int(collision_seg),
        time_stride=float(time_stride),
    )

    if len(op_df) > 0:
        best_row = op_df.iloc[op_df["Recall"].idxmax()]
        best = {
            "best_recall_threshold": float(best_row["Threshold"]),
            "best_recall": float(best_row["Recall"]),
            "precision_at_best_recall": float(best_row["Precision"]),
            "mtta_at_best_recall": float(best_row["mTTA(sec)"]),
            "fp_at_best_recall": int(best_row["FP"]),
            "tp_at_best_recall": int(best_row["TP"]),
            "fn_at_best_recall": int(best_row["FN"]),
            "tn_at_best_recall": int(best_row["TN"]),
        }
    else:
        best = {
            "best_recall_threshold": float("nan"),
            "best_recall": float("nan"),
            "precision_at_best_recall": float("nan"),
            "mtta_at_best_recall": float("nan"),
            "fp_at_best_recall": 0,
            "tp_at_best_recall": 0,
            "fn_at_best_recall": 0,
            "tn_at_best_recall": 0,
        }

    result = {
        "loss": total_loss / max(1, n),
        "loss_semantic": total_sem_loss / max(1, n),
        "loss_collision": total_col_loss / max(1, n),
        "semantic": {
            "overall": {
                "mean_field_accuracy": float(mean(sem_accs)) if sem_accs else 0.0,
                "mean_field_macro_f1": float(mean(sem_macro_f1s)) if sem_macro_f1s else 0.0,
                "hprox_soft_accuracy": float(hprox_soft_accuracy),
            },
            "per_field": semantic_per_field,
        },
        "collision": {
            "window_ap": None if np.isnan(window_ap) else float(window_ap),
            "window_auc": None if np.isnan(window_auc) else float(window_auc),
            "clip_ap_precrash_max": None if np.isnan(clip_ap) else float(clip_ap),
            "clip_auc_precrash_max": None if np.isnan(clip_auc) else float(clip_auc),
            "num_windows": int(len(all_true_collision)),
            "num_clips": int(len(clip_probs)),
            **best,
        },
        "num_samples": int(n),
    }
    if collect_window_outputs:
        result["window_outputs"] = window_records
    result["operating_df"] = op_df
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    field_class_weights: Dict[str, torch.Tensor],
    collision_pos_weight: torch.Tensor,
    lambda_semantic: float,
    lambda_collision: float,
    use_amp: bool,
    accumulation_steps: int,
    lambda_contrastive: float,
    contrastive_temperature: float,
    contrastive_fields: List[str],
    contrastive_class_weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_sem = 0.0
    total_col = 0.0
    total_con = 0.0
    n = 0
    accumulation_steps = max(1, int(accumulation_steps))
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        pixels = batch["pixel_values"].to(device)
        labels_field = batch["field_labels"].to(device)
        labels_col = batch["collision_label"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_field, logits_col, pooled = model(pixels, return_features=True)
            loss_sem = 0.0
            for i, f in enumerate(FIELDS):
                w = field_class_weights[f].to(device=device, dtype=logits_field[f].dtype)
                loss_sem = loss_sem + F.cross_entropy(logits_field[f], labels_field[:, i], weight=w)
            pos_w = collision_pos_weight.to(device=device, dtype=logits_col.dtype)
            loss_col = F.binary_cross_entropy_with_logits(
                logits_col,
                labels_col.to(dtype=logits_col.dtype),
                pos_weight=pos_w,
            )
            loss_con = torch.zeros((), device=device, dtype=pooled.dtype)
            if contrastive_fields and float(lambda_contrastive) > 0.0:
                con_terms = []
                for field in contrastive_fields:
                    if field == "COLLISION":
                        con_labels = labels_col.to(torch.long)
                    else:
                        con_labels = labels_field[:, FIELDS.index(field)]
                    term = weighted_supervised_contrastive_loss(
                        pooled,
                        con_labels,
                        contrastive_class_weights.get(field),
                        contrastive_temperature,
                    )
                    con_terms.append(term)
                if con_terms:
                    loss_con = sum(con_terms) / float(len(con_terms))
            loss = lambda_semantic * loss_sem + lambda_collision * loss_col + float(lambda_contrastive) * loss_con

        loss_to_backward = loss / float(accumulation_steps)
        scaler.scale(loss_to_backward).backward()

        do_step = (batch_idx % accumulation_steps == 0) or (batch_idx == num_batches)
        if do_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = labels_field.size(0)
        total_loss += float(loss.item()) * bs
        total_sem += float(loss_sem.item()) * bs
        total_col += float(loss_col.item()) * bs
        total_con += float(loss_con.item()) * bs
        n += bs

    return {
        "loss": total_loss / max(1, n),
        "loss_semantic": total_sem / max(1, n),
        "loss_collision": total_col / max(1, n),
        "loss_contrastive": total_con / max(1, n),
    }


def maybe_slice(samples: List[Dict[str, Any]], max_n: int, seed: int) -> List[Dict[str, Any]]:
    if max_n <= 0 or len(samples) <= max_n:
        return samples
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    idx = idx[:max_n]
    idx.sort()
    return [samples[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--official_repo_root", type=str, default=DEFAULT_OFFICIAL_REPO_ROOT)
    ap.add_argument("--labels_path", default=DEFAULT_LABELS)
    ap.add_argument("--splits_path", default=DEFAULT_SPLITS)
    ap.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT)
    ap.add_argument("--output_dir", default="out_score/vjepa2_7fields_collision_for_v2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--window_frames", type=int, default=10)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--collision_seg", type=int, default=50)
    ap.add_argument("--time_stride", type=float, default=0.1)
    ap.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--lambda_semantic", type=float, default=1.0)
    ap.add_argument("--lambda_collision", type=float, default=1.0)
    ap.add_argument(
        "--model_select_metric",
        type=str,
        default="collision_clip_ap",
        choices=[
            "collision_clip_ap",
            "collision_clip_auc",
            "semantic_macro_f1",
            "semantic_weighted_macro_f1",
            "semantic_priority",
            "joint",
        ],
    )
    ap.add_argument("--joint_semantic_weight", type=float, default=0.2)
    ap.add_argument(
        "--semantic_field_weights",
        type=str,
        default="",
        help="Comma-separated FIELD:weight, e.g. HPOS:0.3,PATH:0.25,HPROX:0.25,PHT:0.05,LATORIG:0.02",
    )
    ap.add_argument("--hprox_near_miss_credit", type=float, default=0.5)
    ap.add_argument("--semantic_priority_hprox_weight", type=float, default=0.1)
    ap.add_argument("--early_stopping_patience", type=int, default=3)
    ap.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--freeze_backbone", type=int, default=0)
    ap.add_argument(
        "--backbone_ckpt",
        type=str,
        default="checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt",
        help="Local V-JEPA2.1 checkpoint path.",
    )
    ap.add_argument("--backbone_ckpt_key", type=str, default="ema_encoder", help="Top-level key in checkpoint that contains encoder state_dict.")
    ap.add_argument(
        "--backbone_strip_prefixes",
        type=str,
        default="module.backbone.,backbone.,module.",
        help="Comma-separated prefixes stripped from checkpoint keys before loading.",
    )
    ap.add_argument(
        "--backbone_ckpt_format",
        type=str,
        default="raw",
        help="Checkpoint mapping mode (kept for compatibility; official path uses raw).",
    )
    ap.add_argument("--backbone_ckpt_strict", type=int, default=1, help="If 1, strict=True for backbone load_state_dict.")
    ap.add_argument("--processor_crop_size", type=int, default=384, help="Official eval crop size for V-JEPA2.1.")
    ap.add_argument("--use_field_class_weights", type=int, default=1)
    ap.add_argument("--field_class_weight_power", type=float, default=1.0)
    ap.add_argument("--field_class_weight_min", type=float, default=0.0)
    ap.add_argument("--field_class_weight_max", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_only_ckpt", type=str, default="")
    ap.add_argument("--dump_val_window_preds", type=str, default="")
    ap.add_argument("--dump_test_window_preds", type=str, default="")
    ap.add_argument("--use_lora", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="qkv,proj",
        help="Comma-separated suffix names inside backbone, e.g. qkv,proj or qkv,proj,fc1,fc2",
    )
    ap.add_argument("--lambda_contrastive", type=float, default=0.0)
    ap.add_argument("--contrastive_temperature", type=float, default=0.12)
    ap.add_argument(
        "--contrastive_fields",
        type=str,
        default="",
        help="Comma-separated subset of HMOT,HPROX,PATH,PHT,HPOS,GAP,LATORIG,COLLISION",
    )
    ap.add_argument("--contrastive_class_weight_power", type=float, default=0.5)
    ap.add_argument("--contrastive_class_weight_max", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    use_amp = (device.type == "cuda")
    thresholds = parse_thresholds(args.thresholds)
    semantic_field_weights = parse_semantic_field_weights(args.semantic_field_weights)
    hprox_near_miss_credit = float(np.clip(float(args.hprox_near_miss_credit), 0.0, 1.0))
    contrastive_fields = parse_contrastive_fields(args.contrastive_fields)

    windows = read_jsonl(args.labels_path)
    splits = load_splits(args.splits_path)
    train_samples, train_stats = build_samples(windows, splits["train"], args.frames_root, args.window_frames)
    val_samples, val_stats = build_samples(windows, splits["val"], args.frames_root, args.window_frames)
    test_samples, test_stats = build_samples(windows, splits["test"], args.frames_root, args.window_frames)

    train_samples = maybe_slice(train_samples, args.max_train_samples, args.seed + 1)
    val_samples = maybe_slice(val_samples, args.max_val_samples, args.seed + 2)
    test_samples = maybe_slice(test_samples, args.max_test_samples, args.seed + 3)

    processor = OfficialVJepa21VideoProcessor(
        repo_root=args.official_repo_root,
        crop_size=int(args.processor_crop_size),
    )
    train_ds = WindowSemanticCollisionDataset(train_samples, processor)
    val_ds = WindowSemanticCollisionDataset(val_samples, processor)
    test_ds = WindowSemanticCollisionDataset(test_samples, processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = VJEPA2SemanticCollision(
        args.model_name,
        dropout=args.dropout,
        official_repo_root=args.official_repo_root,
    ).to(device)
    backbone_load_info = load_backbone_weights_from_vjepa_ckpt(
        model.backbone,
        ckpt_path=args.backbone_ckpt,
        ckpt_key=args.backbone_ckpt_key,
        strip_prefixes_csv=args.backbone_strip_prefixes,
        ckpt_format=args.backbone_ckpt_format,
        strict=bool(args.backbone_ckpt_strict),
    )
    if backbone_load_info.get("loaded"):
        print(
            "[BACKBONE_CKPT] loaded",
            f"missing={backbone_load_info['num_missing_keys']}",
            f"unexpected={backbone_load_info['num_unexpected_keys']}",
        )
    lora_info: Dict[str, Any] = {"enabled": False}
    if int(args.use_lora):
        target_modules = parse_lora_target_modules(args.lora_target_modules)
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            target_modules=target_modules,
            lora_dropout=float(args.lora_dropout),
            bias="none",
        )
        model.backbone = get_peft_model(model.backbone, lora_cfg)
        trainable_backbone, total_backbone = count_trainable_parameters(model.backbone)
        lora_info = {
            "enabled": True,
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": target_modules,
            "backbone_trainable_params": trainable_backbone,
            "backbone_total_params": total_backbone,
        }
        print(
            "[LORA]",
            f"targets={target_modules}",
            f"trainable_backbone={trainable_backbone}",
            f"total_backbone={total_backbone}",
        )
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    trainable_params, total_params = count_trainable_parameters(model)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if bool(args.use_field_class_weights):
        field_class_weights = compute_field_class_weights(
            train_samples,
            power=float(args.field_class_weight_power),
            min_w=float(args.field_class_weight_min),
            max_w=float(args.field_class_weight_max),
        )
    else:
        field_class_weights = {
            f: torch.ones(FIELD_NUM_CLASSES[f], dtype=torch.float32)
            for f in FIELDS
        }
    collision_pos_weight = compute_collision_pos_weight(train_samples)
    contrastive_class_weights = compute_contrastive_class_weights(
        train_samples,
        contrastive_fields,
        power=float(args.contrastive_class_weight_power),
        max_w=float(args.contrastive_class_weight_max),
    ) if contrastive_fields and float(args.lambda_contrastive) > 0.0 else {}

    run_meta = {
        "args": vars(args),
        "device": str(device),
        "thresholds": thresholds,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "build_stats": {"train": train_stats, "val": val_stats, "test": test_stats},
        "collision_pos_weight": float(collision_pos_weight.item()),
        "field_class_weight_ranges": {
            f: {
                "min": float(field_class_weights[f].min().item()),
                "max": float(field_class_weights[f].max().item()),
                "mean": float(field_class_weights[f].mean().item()),
            }
            for f in FIELDS
        },
        "semantic_field_weights_for_selection": semantic_field_weights,
        "semantic_field_weights_recommended_example": DEFAULT_SEMANTIC_FIELD_WEIGHTS,
        "hprox_near_miss_credit": float(hprox_near_miss_credit),
        "semantic_priority_hprox_weight": float(args.semantic_priority_hprox_weight),
        "backbone_load_info": backbone_load_info,
        "lora_info": lora_info,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "contrastive_fields": contrastive_fields,
        "contrastive_class_weight_ranges": {
            f: {
                "min": float(contrastive_class_weights[f].min().item()),
                "max": float(contrastive_class_weights[f].max().item()),
                "mean": float(contrastive_class_weights[f].mean().item()),
            }
            for f in contrastive_class_weights
        },
        "processor_config": {
            "do_resize": bool(getattr(processor, "do_resize", False)),
            "size": getattr(processor, "size", None),
            "do_center_crop": bool(getattr(processor, "do_center_crop", False)),
            "crop_size": getattr(processor, "crop_size", None),
            "use_vjepa21_recipe": True,
        },
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    if args.eval_only_ckpt:
        ckpt = torch.load(args.eval_only_ckpt, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[EVAL_ONLY_CKPT] loaded={args.eval_only_ckpt} missing={len(missing)} unexpected={len(unexpected)}")

        val = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            field_class_weights=field_class_weights,
            collision_pos_weight=collision_pos_weight,
            lambda_semantic=float(args.lambda_semantic),
            lambda_collision=float(args.lambda_collision),
            collision_seg=int(args.collision_seg),
            time_stride=float(args.time_stride),
            thresholds=thresholds,
            hprox_near_miss_credit=hprox_near_miss_credit,
            collect_window_outputs=bool(args.dump_val_window_preds),
        )
        val["semantic"]["overall"]["weighted_field_macro_f1"] = weighted_field_macro_f1(val, semantic_field_weights)
        test = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            field_class_weights=field_class_weights,
            collision_pos_weight=collision_pos_weight,
            lambda_semantic=float(args.lambda_semantic),
            lambda_collision=float(args.lambda_collision),
            collision_seg=int(args.collision_seg),
            time_stride=float(args.time_stride),
            thresholds=thresholds,
            hprox_near_miss_credit=hprox_near_miss_credit,
            collect_window_outputs=bool(args.dump_test_window_preds),
        )
        test["semantic"]["overall"]["weighted_field_macro_f1"] = weighted_field_macro_f1(test, semantic_field_weights)
        val_op_df = val.pop("operating_df")
        test_op_df = test.pop("operating_df")
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test, f, ensure_ascii=False, indent=2)
        val_op_df.to_csv(os.path.join(args.output_dir, "val_operating.csv"), index=False)
        test_op_df.to_csv(os.path.join(args.output_dir, "test_operating.csv"), index=False)
        for split_name, dump_arg, metrics in (
            ("val", args.dump_val_window_preds, val),
            ("test", args.dump_test_window_preds, test),
        ):
            if dump_arg:
                dump_path = dump_arg
                if not os.path.isabs(dump_path):
                    dump_path = os.path.join(args.output_dir, dump_path)
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                with open(dump_path, "w", encoding="utf-8") as f:
                    for rec in metrics.get("window_outputs", []):
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"Saved {split_name} window preds: {dump_path}")
        print("Val:", os.path.join(args.output_dir, "val_metrics.json"))
        print("Test:", os.path.join(args.output_dir, "test_metrics.json"))
        return

    best_score = -1e18
    best_path = os.path.join(args.output_dir, "best_model.pt")
    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            field_class_weights=field_class_weights,
            collision_pos_weight=collision_pos_weight,
            lambda_semantic=float(args.lambda_semantic),
            lambda_collision=float(args.lambda_collision),
            use_amp=use_amp,
            accumulation_steps=int(args.accumulation_steps),
            lambda_contrastive=float(args.lambda_contrastive),
            contrastive_temperature=float(args.contrastive_temperature),
            contrastive_fields=contrastive_fields,
            contrastive_class_weights=contrastive_class_weights,
        )
        val = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            field_class_weights=field_class_weights,
            collision_pos_weight=collision_pos_weight,
            lambda_semantic=float(args.lambda_semantic),
            lambda_collision=float(args.lambda_collision),
            collision_seg=int(args.collision_seg),
            time_stride=float(args.time_stride),
            thresholds=thresholds,
            hprox_near_miss_credit=hprox_near_miss_credit,
            collect_window_outputs=False,
        )
        val["semantic"]["overall"]["weighted_field_macro_f1"] = weighted_field_macro_f1(val, semantic_field_weights)
        score = score_for_model_selection(
            metrics=val,
            metric_name=args.model_select_metric,
            semantic_weight=float(args.joint_semantic_weight),
            semantic_field_weights=semantic_field_weights,
            semantic_priority_hprox_weight=float(args.semantic_priority_hprox_weight),
        )

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_loss_semantic": tr["loss_semantic"],
            "train_loss_collision": tr["loss_collision"],
            "train_loss_contrastive": tr["loss_contrastive"],
            "val_loss": val["loss"],
            "val_mean_field_macro_f1": val["semantic"]["overall"]["mean_field_macro_f1"],
            "val_weighted_field_macro_f1": val["semantic"]["overall"]["weighted_field_macro_f1"],
            "val_hprox_soft_accuracy": val["semantic"]["overall"]["hprox_soft_accuracy"],
            "val_collision_clip_ap": val["collision"]["clip_ap_precrash_max"],
            "val_collision_clip_auc": val["collision"]["clip_auc_precrash_max"],
            "val_collision_mtta_best_recall": val["collision"]["mtta_at_best_recall"],
            "select_score": score,
            "elapsed_sec": time.time() - t0,
        }
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))

        improved = (score > best_score + float(args.early_stopping_min_delta))
        if improved:
            best_score = score
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_score": float(score),
                    "args": vars(args),
                },
                best_path,
            )
        else:
            no_improve += 1
            if int(args.early_stopping_patience) > 0 and no_improve >= int(args.early_stopping_patience):
                print(f"[EARLY_STOP] epoch={epoch} no_improve={no_improve}")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        field_class_weights=field_class_weights,
        collision_pos_weight=collision_pos_weight,
        lambda_semantic=float(args.lambda_semantic),
        lambda_collision=float(args.lambda_collision),
        collision_seg=int(args.collision_seg),
        time_stride=float(args.time_stride),
        thresholds=thresholds,
        hprox_near_miss_credit=hprox_near_miss_credit,
        collect_window_outputs=False,
    )
    val["semantic"]["overall"]["weighted_field_macro_f1"] = weighted_field_macro_f1(val, semantic_field_weights)
    test = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        field_class_weights=field_class_weights,
        collision_pos_weight=collision_pos_weight,
        lambda_semantic=float(args.lambda_semantic),
        lambda_collision=float(args.lambda_collision),
        collision_seg=int(args.collision_seg),
        time_stride=float(args.time_stride),
        thresholds=thresholds,
        hprox_near_miss_credit=hprox_near_miss_credit,
        collect_window_outputs=bool(args.dump_test_window_preds),
    )
    test["semantic"]["overall"]["weighted_field_macro_f1"] = weighted_field_macro_f1(test, semantic_field_weights)

    best_epoch = int(ckpt.get("epoch", -1))
    best_score = float(ckpt.get("best_score", 0.0))
    val["best_epoch"] = best_epoch
    val["best_score"] = best_score
    test["best_epoch"] = best_epoch
    test["best_score"] = best_score

    val_op_df = val.pop("operating_df")
    test_op_df = test.pop("operating_df")

    with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    val_op_df.to_csv(os.path.join(args.output_dir, "val_operating.csv"), index=False)
    test_op_df.to_csv(os.path.join(args.output_dir, "test_operating.csv"), index=False)

    if args.dump_test_window_preds:
        dump_path = args.dump_test_window_preds
        if not os.path.isabs(dump_path):
            dump_path = os.path.join(args.output_dir, dump_path)
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            for rec in test.get("window_outputs", []):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved window preds: {dump_path}")

    print("Saved:", best_path)
    print("Val:", os.path.join(args.output_dir, "val_metrics.json"))
    print("Test:", os.path.join(args.output_dir, "test_metrics.json"))


if __name__ == "__main__":
    main()
