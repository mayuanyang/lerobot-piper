"""
Generic community-dataset training script supporting multiple model architectures.

Supported model types:
  - `interleaved`  : SmolVLA-style interleaved flow matching (SmolVLM2-500M, d_model=960)
  - `wilro`        : VLM KV-cache → DiT cross-attention (SmolVLM2-500M, d_model=960)
  - `wiltechs_vla` : Encoder-decoder MoT (Qwen3-VL-4B, d_model=2560)

Each model type has its own config/policy/processor classes but shares the same
training loop, dataset adapter, and canonical schema projection.

Usage:
    # Pretrain interleaved model on community data
    python src/train_community.py \
        --model_type interleaved \
        --output_dir outputs/train/community_interleaved \
        --batch_size 32 \
        --training_steps 300000

    # Pretrain wilro model with stride2 KV capture
    python src/train_community.py \
        --model_type wilro \
        --kv_capture_strategy stride2 \
        --output_dir outputs/train/community_wilro \
        --batch_size 32 \
        --training_steps 300000

    # Pretrain wiltechs_vla (Qwen3-VL-4B backbone)
    python src/train_community.py \
        --model_type wiltechs_vla \
        --output_dir outputs/train/community_wiltechs \
        --batch_size 16 \
        --training_steps 300000
"""

from __future__ import annotations

import json
import math
import random
import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
import huggingface_hub
from huggingface_hub import list_repo_files, snapshot_download
from safetensors.torch import load_file as load_safetensors

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

# ---------------------------------------------------------------------------
# Model registry — maps model_type string to (Config, Policy, processor_fn)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {}


def _register_model(model_type: str, config_cls, policy_cls, processor_fn, defaults: dict):
    MODEL_REGISTRY[model_type] = {
        "config_cls": config_cls,
        "policy_cls": policy_cls,
        "processor_fn": processor_fn,
        "defaults": defaults,
    }


def _lazy_imports():
    """Import model modules on demand to avoid loading all 3 backbones at startup."""
    from models.interleaved_flow_matching.interleaved_flow_matching_config import InterleavedFlowMatchingConfig
    from models.interleaved_flow_matching.interleaved_flow_matching_policy import InterleavedFlowMatchingPolicy
    from models.interleaved_flow_matching.processor_interleaved_flow_matching import make_pre_post_processors as proc_interleaved

    from models.wilro.wilro_config import WilroConfig
    from models.wilro.wilro_policy import WilroPolicy
    from models.wilro.processor_wilro import make_pre_post_processors as proc_wilro

    from models.wiltechs_vla.wiltechs_vla_config import WiltechsVLAConfig
    from models.wiltechs_vla.wiltechs_vla_policy import WiltechsVLAPolicy
    from models.wiltechs_vla.processor_wiltechs_vla import make_pre_post_processors as proc_wiltechs

    _register_model("interleaved", InterleavedFlowMatchingConfig, InterleavedFlowMatchingPolicy, proc_interleaved, {
        "d_model": 960,
        "vision_input_size": 384,
    })
    _register_model("wilro", WilroConfig, WilroPolicy, proc_wilro, {
        "d_model": 960,
        "vision_input_size": 384,
    })
    _register_model("wiltechs_vla", WiltechsVLAConfig, WiltechsVLAPolicy, proc_wiltechs, {
        "d_model": 2560,
        "vision_input_size": 448,
    })


def get_model_components(model_type: str):
    """Return (ConfigCls, PolicyCls, processor_fn, defaults) for the given model_type."""
    if not MODEL_REGISTRY:
        _lazy_imports()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    entry = MODEL_REGISTRY[model_type]
    return entry["config_cls"], entry["policy_cls"], entry["processor_fn"], entry["defaults"]


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Community dataset hub
# ---------------------------------------------------------------------------
COMMUNITY_DATASET_REPO = "ISdept/community_dataset_v3_part1"
INFO_MARKER = "meta/info.json"

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
CANONICAL_CAMERAS = [
    "observation.images.front",
    "observation.images.wrist",
    "observation.images.top",
]

CANONICAL_STATE_DIM = 8
CANONICAL_ACTION_DIM = 7

_ALLOWED_ITEM_KEYS = {
    "observation.state",
    "action",
    "action_is_pad",
    "action_dim_pad",
    "task",
    "task_description",
    "task_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "index",
}

CANONICAL_IMAGE_SIZE = 384  # base; wiltechs_vla overrides to 448 at model level


def _resize_camera_to_canonical(img: torch.Tensor, target_size: int = CANONICAL_IMAGE_SIZE) -> torch.Tensor:
    if not isinstance(img, torch.Tensor) or img.dim() != 4:
        return img
    h, w = img.shape[-2], img.shape[-1]
    if h == target_size and w == target_size:
        return img
    if h != w:
        max_dim = max(h, w)
        pad = (
            (max_dim - w) // 2, max_dim - w - (max_dim - w) // 2,
            (max_dim - h) // 2, max_dim - h - (max_dim - h) // 2,
        )
        img = F.pad(img.float(), pad, value=0.0)
    if img.shape[-2] != target_size or img.shape[-1] != target_size:
        img = F.interpolate(
            img.float(),
            size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
    return img

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def get_augmentations():
    spatial = v2.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0)
    color = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)
    blur = v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3)
    return v2.Compose([spatial, color, blur])


def apply_image_augmentations(batch: dict, camera_keys: list[str], transform) -> dict:
    present_keys = [k for k in camera_keys if k in batch and isinstance(batch[k], torch.Tensor)]
    if not present_keys:
        return batch
    B = batch[present_keys[0]].shape[0]
    for b in range(B):
        sample_img = batch[present_keys[0]][b]
        has_time_dim = sample_img.dim() == 4
        if has_time_dim:
            T = sample_img.shape[0]
            stacked = torch.cat([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i * T : (i + 1) * T]
        else:
            stacked = torch.stack([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i]
    return batch


def apply_joint_augmentations(batch: dict, state_key: str) -> dict:
    if torch.rand(1).item() > 0.5:
        if state_key in batch:
            noise = torch.randn_like(batch[state_key]) * 0.02
            batch[state_key] = batch[state_key] + noise
    return batch

# ---------------------------------------------------------------------------
# Discover sub-dataset list from the community hub
# ---------------------------------------------------------------------------
def discover_sub_datasets(repo_id: str = COMMUNITY_DATASET_REPO) -> list[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    roots: set[str] = set()
    for f in files:
        if f.endswith(INFO_MARKER):
            root = f[: -len(INFO_MARKER)].rstrip("/")
            roots.add(root)
    roots_sorted = sorted(roots)
    print(f"[discover] Found {len(roots_sorted)} LeRobot dataset roots in {repo_id} "
          f"(scanned {len(files)} files)")
    return roots_sorted


def classify_dataset_versions(repo_id: str = COMMUNITY_DATASET_REPO) -> dict[str, str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    fileset = set(files)

    episodes_dir_roots: set[str] = set()
    for f in files:
        idx = f.find("/meta/episodes/")
        if idx != -1:
            episodes_dir_roots.add(f[:idx])
        elif f.startswith("meta/episodes/"):
            episodes_dir_roots.add("")

    result: dict[str, str] = {}
    for f in files:
        if not f.endswith(INFO_MARKER):
            continue
        root = f[: -len(INFO_MARKER)].rstrip("/")
        prefix = f"{root}/meta/" if root else "meta/"
        v3 = (prefix + "tasks.parquet") in fileset or root in episodes_dir_roots
        v2 = (prefix + "tasks.jsonl") in fileset or (prefix + "episodes.jsonl") in fileset
        result[root] = "v3.0" if v3 else ("v2.1" if v2 else "unknown")
    return result


def print_version_report(repo_id: str = COMMUNITY_DATASET_REPO) -> dict[str, str]:
    versions = classify_dataset_versions(repo_id)
    by_ver: dict[str, list[str]] = {"v3.0": [], "v2.1": [], "unknown": []}
    for root, ver in sorted(versions.items()):
        by_ver.setdefault(ver, []).append(root)

    print(f"\n=== Version report for {repo_id} ===")
    print(f"  v3.0:    {len(by_ver['v3.0'])}")
    print(f"  v2.1:    {len(by_ver['v2.1'])}")
    print(f"  unknown: {len(by_ver['unknown'])}")
    print(f"  total:   {len(versions)}\n")
    for ver in ("v3.0", "v2.1", "unknown"):
        if by_ver[ver]:
            print(f"--- {ver} ({len(by_ver[ver])}) ---")
            for root in by_ver[ver]:
                print(f"  {root}")
            print()
    return versions


def _download_subdir(
    repo_id: str, subpath: str, workspace: Path, patterns: Optional[list[str]] = None,
) -> Path:
    if patterns is None:
        patterns = [f"{subpath}/**"] if subpath else None
    workspace.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        allow_patterns=patterns,
        local_dir=str(workspace),
    )
    return workspace / subpath if subpath else workspace


def load_sub_dataset_info(repo_id: str, subpath: str, workspace: Path) -> Optional[dict]:
    try:
        meta_patterns = [f"{subpath}/meta/info.json"] if subpath else ["meta/info.json"]
        root = _download_subdir(repo_id, subpath, workspace, patterns=meta_patterns)
        with open(root / "meta" / "info.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to read info.json for {repo_id}:{subpath or '<root>'}: {e}")
        return None


class _RawMeta:
    def __init__(self, info: dict):
        self.features = info.get("features", {})
        self.fps = info.get("fps", 30)
        self.codebase_version = str(info.get("codebase_version", "")).lstrip("v")


def _ensure_v30(repo_id: str, root: Path, subpath: str) -> Path:
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        version = str(json.load(f).get("codebase_version", "")).lstrip("v")
    if not version.startswith("2"):
        return root

    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    print(f"  [convert] {subpath}: v{version} → v3.0 (in place, no hub push)...")
    convert_dataset(
        repo_id=repo_id,
        root=root,
        push_to_hub=False,
        force_conversion=True,
    )
    return root

# ---------------------------------------------------------------------------
# Camera-name mapping discovery
# ---------------------------------------------------------------------------
def discover_all_camera_names(sub_metas: dict[str, "_RawMeta"]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for sub_dir, meta in sub_metas.items():
        features = dataset_to_policy_features(meta.features)
        cams = {k for k, ft in features.items() if ft.type == FeatureType.VISUAL}
        result[sub_dir] = cams
    return result


def discover_state_action_dims(
    sub_metas: dict[str, "_RawMeta"],
) -> tuple[dict[str, int], dict[str, int], dict[str, str], dict[str, str]]:
    state_dims: dict[str, int] = {}
    action_dims: dict[str, int] = {}
    state_keys: dict[str, str] = {}
    action_keys: dict[str, str] = {}
    for sub_dir, meta in sub_metas.items():
        features = dataset_to_policy_features(meta.features)
        for k, ft in features.items():
            if ft.type == FeatureType.STATE:
                state_dims[sub_dir] = ft.shape[-1]
                state_keys[sub_dir] = k
            elif ft.type == FeatureType.ACTION:
                action_dims[sub_dir] = ft.shape[-1]
                action_keys[sub_dir] = k
    return state_dims, action_dims, state_keys, action_keys

# ---------------------------------------------------------------------------
# Camera-mapping helper
# ---------------------------------------------------------------------------
def build_camera_mapping(
    sub_cameras: dict[str, set[str]],
    canonical: list[str],
) -> dict[str, dict[str, Optional[str]]]:
    mapping: dict[str, dict[str, Optional[str]]] = {}
    for sub_dir, cams in sub_cameras.items():
        sub_map: dict[str, Optional[str]] = {canon: None for canon in canonical}
        used_cams: set[str] = set()
        cams_lower = {c.lower(): c for c in cams}

        for canon in canonical:
            if canon in cams:
                sub_map[canon] = canon
                used_cams.add(canon)
                continue
            canon_suffix = canon.split(".")[-1].lower()
            for cam_lower, cam_orig in cams_lower.items():
                if cam_orig in used_cams:
                    continue
                if cam_lower.endswith(canon_suffix):
                    sub_map[canon] = cam_orig
                    used_cams.add(cam_orig)
                    break

        remaining_cams = sorted(c for c in cams if c not in used_cams)
        for canon in canonical:
            if sub_map[canon] is None and remaining_cams:
                sub_map[canon] = remaining_cams.pop(0)

        mapping[sub_dir] = sub_map
    return mapping

# ---------------------------------------------------------------------------
# Per-dataset normalization helpers
# ---------------------------------------------------------------------------
def native_feature_stats(
    dataset: LeRobotDataset, key: str, dim: int,
) -> Optional[dict[str, torch.Tensor]]:
    try:
        ms = getattr(getattr(dataset, "meta", None), "stats", None)
        if ms and key in ms and "mean" in ms[key] and "std" in ms[key]:
            mean = torch.as_tensor(np.asarray(ms[key]["mean"], dtype=np.float32)).reshape(-1)
            std = torch.as_tensor(np.asarray(ms[key]["std"], dtype=np.float32)).reshape(-1)
            if mean.numel() == dim:
                return {"mean": mean, "std": std.clamp_min(1e-6)}
    except Exception:
        pass
    try:
        arr = np.asarray(dataset.hf_dataset[key], dtype=np.float32)
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[-1] == dim:
            mean = torch.from_numpy(arr.mean(axis=0))
            std = torch.from_numpy(arr.std(axis=0)).clamp_min(1e-6)
            return {"mean": mean, "std": std}
    except Exception:
        pass
    return None


def identity_stats(
    camera_keys: list[str], state_dim: int, action_dim: int,
) -> dict[str, dict[str, torch.Tensor]]:
    stats = {
        "observation.state": {
            "mean": torch.zeros(state_dim), "std": torch.ones(state_dim),
            "min": -torch.ones(state_dim), "max": torch.ones(state_dim),
        },
        "action": {
            "mean": torch.zeros(action_dim), "std": torch.ones(action_dim),
            "min": -torch.ones(action_dim), "max": torch.ones(action_dim),
        },
    }
    for cam in camera_keys:
        stats[cam] = {
            "mean": torch.tensor([0.0]), "std": torch.tensor([1.0]),
            "min": torch.tensor([-1.0]), "max": torch.tensor([1.0]),
        }
    return stats

# ---------------------------------------------------------------------------
# DatasetAdapter
# ---------------------------------------------------------------------------
class DatasetAdapter(Dataset):
    def __init__(
        self,
        dataset: LeRobotDataset,
        sub_dir: str,
        camera_map: dict[str, Optional[str]],
        state_key: str,
        action_key: str,
        state_dim: int,
        action_dim: int,
        task_idx_to_desc: Optional[dict[int, str]] = None,
        canonical_state_dim: Optional[int] = None,
        canonical_action_dim: Optional[int] = None,
        state_stats: Optional[dict] = None,
        action_stats: Optional[dict] = None,
        normalize_in_adapter: bool = False,
        canonical_image_size: int = CANONICAL_IMAGE_SIZE,
    ):
        self.dataset = dataset
        self.sub_dir = sub_dir
        self.camera_map = camera_map
        self.state_key = state_key
        self.action_key = action_key
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.canonical_state_dim = (
            canonical_state_dim if canonical_state_dim is not None else CANONICAL_STATE_DIM
        )
        self.canonical_action_dim = (
            canonical_action_dim if canonical_action_dim is not None else CANONICAL_ACTION_DIM
        )
        self.task_idx_to_desc = task_idx_to_desc or {}
        self.normalize_in_adapter = normalize_in_adapter
        self._state_mean = state_stats["mean"].float() if state_stats else None
        self._state_std = state_stats["std"].float() if state_stats else None
        self._action_mean = action_stats["mean"].float() if action_stats else None
        self._action_std = action_stats["std"].float() if action_stats else None
        self._canonical_cams = list(camera_map.keys())
        self.canonical_image_size = canonical_image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        try:
            item = self.dataset[int(idx)]
        except Exception as e:
            item = None
            for _ in range(8):
                alt = random.randint(0, len(self.dataset) - 1)
                try:
                    item = self.dataset[alt]
                    break
                except Exception:
                    continue
            if item is None:
                raise RuntimeError(
                    f"{self.sub_dir}: 8 consecutive samples failed to decode "
                    f"(last error: {e})"
                ) from e

        for canon_cam in self._canonical_cams:
            actual_key = self.camera_map.get(canon_cam)
            if actual_key is not None and actual_key in item:
                if actual_key != canon_cam:
                    item[canon_cam] = item.pop(actual_key)
                val = item[canon_cam]
                if isinstance(val, torch.Tensor) and val.dim() == 3:
                    item[canon_cam] = val.unsqueeze(0)
            elif actual_key is None:
                ref_cam = next((k for k in self._canonical_cams if self.camera_map.get(k) and k in item), None)
                if ref_cam is not None:
                    ref_shape = item[ref_cam].shape
                    item[canon_cam] = torch.zeros(ref_shape, dtype=item[ref_cam].dtype)
                else:
                    item[canon_cam] = torch.zeros(1, 3, self.canonical_image_size, self.canonical_image_size)

        canonical_set = set(self._canonical_cams)
        for k in [k for k in item if k.startswith("observation.images.") and k not in canonical_set]:
            del item[k]

        for canon_cam in self._canonical_cams:
            if canon_cam in item:
                item[canon_cam] = _resize_camera_to_canonical(item[canon_cam], self.canonical_image_size)

        if self.state_key in item:
            state = item[self.state_key]
            if isinstance(state, torch.Tensor):
                if self.normalize_in_adapter and self._state_mean is not None:
                    state = (state.float() - self._state_mean) / self._state_std
                sd = state.shape[-1] if state.dim() >= 1 else 1
                if sd < self.canonical_state_dim:
                    pad = torch.zeros(*state.shape[:-1], self.canonical_state_dim - sd,
                                     dtype=state.dtype, device=state.device)
                    state = torch.cat([state, pad], dim=-1)
                elif sd > self.canonical_state_dim:
                    state = state[..., :self.canonical_state_dim]
                item[self.state_key] = state

        if self.action_key in item:
            action = item[self.action_key]
            if isinstance(action, torch.Tensor):
                if self.normalize_in_adapter and self._action_mean is not None:
                    action = (action.float() - self._action_mean) / self._action_std
                ad = action.shape[-1] if action.dim() >= 1 else 1
                if ad < self.canonical_action_dim:
                    pad = torch.zeros(*action.shape[:-1], self.canonical_action_dim - ad,
                                     dtype=action.dtype, device=action.device)
                    action = torch.cat([action, pad], dim=-1)
                elif ad > self.canonical_action_dim:
                    action = action[..., :self.canonical_action_dim]
                item[self.action_key] = action

        action_dim_pad = torch.zeros(self.canonical_action_dim, dtype=torch.bool)
        if self.action_dim < self.canonical_action_dim:
            action_dim_pad[self.action_dim:] = True
        item["action_dim_pad"] = action_dim_pad

        action_ref = item.get(self.action_key, item.get("action"))
        H_action = (
            action_ref.shape[0]
            if isinstance(action_ref, torch.Tensor) and action_ref.dim() >= 1
            else 0
        )
        pad_key = next(
            (k for k in item if "pad" in k.lower() and "action" in k.lower()),
            None,
        )
        if pad_key is None:
            if H_action > 0:
                item["action_is_pad"] = torch.zeros(H_action, dtype=torch.bool)
        else:
            ep = item[pad_key]
            if isinstance(ep, torch.Tensor):
                if ep.dim() == 2:
                    ep = ep.any(dim=-1)
                elif ep.dim() == 0 and H_action > 0:
                    ep = ep.unsqueeze(0).expand(H_action).clone()
                item[pad_key] = ep.bool()
            if pad_key != "action_is_pad":
                item["action_is_pad"] = item.pop(pad_key)

        if self.action_key != "action" and self.action_key in item:
            item["action"] = item.pop(self.action_key)
        if self.state_key != "observation.state" and self.state_key in item:
            item["observation.state"] = item.pop(self.state_key)

        if "task_description" not in item and "task" not in item:
            task_idx = item.get("task_index")
            if task_idx is not None:
                if isinstance(task_idx, torch.Tensor):
                    ti = int(task_idx.item()) if task_idx.numel() == 1 else int(task_idx[0].item())
                else:
                    ti = int(task_idx)
                desc = self.task_idx_to_desc.get(ti, "")
                item["task_description"] = desc

        def _is_blank(v) -> bool:
            return v is None or (isinstance(v, str) and not v.strip())

        if _is_blank(item.get("task_description")) and _is_blank(item.get("task")):
            for k in list(item.keys()):
                if not k.startswith("annotation."):
                    continue
                if "task_description" not in k:
                    continue
                v = item[k]
                if isinstance(v, (list, tuple)):
                    v = next((x for x in v if isinstance(x, str) and x.strip()), None)
                if isinstance(v, str) and v.strip():
                    item["task_description"] = v
                    break

        keep = _ALLOWED_ITEM_KEYS | set(self._canonical_cams)
        for k in list(item.keys()):
            if k not in keep:
                del item[k]

        return item


def load_task_descriptions(dataset: LeRobotDataset) -> dict[int, str]:
    task_map: dict[int, str] = {}
    try:
        tasks_path = dataset.root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return task_map
        df = pd.read_parquet(tasks_path)
        if "task_index" not in df.columns:
            return task_map
        if "task" in df.columns:
            for _, row in df.iterrows():
                task_map[int(row["task_index"])] = str(row["task"])
        else:
            for idx, row in df.iterrows():
                task_map[int(row["task_index"])] = str(idx)
    except Exception:
        pass
    return task_map

# ---------------------------------------------------------------------------
# StitchedDataset
# ---------------------------------------------------------------------------
class StitchedDataset(ConcatDataset):
    def __init__(self, datasets: list[DatasetAdapter], ep_boundaries: list[list[int]]):
        super().__init__(datasets)
        self.global_ep_from: list[int] = []
        self.global_ep_to: list[int] = []
        offset = 0
        for ds_idx, (ds, boundaries) in enumerate(zip(datasets, ep_boundaries)):
            for start, end in boundaries:
                self.global_ep_from.append(offset + start)
                self.global_ep_to.append(offset + end)
            offset += len(ds)
        print(f"StitchedDataset: {len(self.global_ep_from)} episodes, {offset} total frames")

    def get_episode_boundaries(self) -> tuple[list[int], list[int]]:
        return self.global_ep_from, self.global_ep_to


def get_sub_dataset_ep_boundaries(dataset: LeRobotDataset) -> list[tuple[int, int]]:
    ep_ids = np.array(dataset.hf_dataset["episode_index"])
    if len(ep_ids) == 0:
        return []
    boundaries: list[tuple[int, int]] = []
    ep_changes = np.where(np.diff(ep_ids) != 0)[0] + 1
    starts = np.concatenate([[0], ep_changes])
    ends = np.concatenate([ep_changes, [len(ep_ids)]])
    for s, e in zip(starts, ends):
        boundaries.append((int(s), int(e)))
    return boundaries

# ---------------------------------------------------------------------------
# Gradient analysis
# ---------------------------------------------------------------------------
def _log_gradient_analysis(policy, step: int) -> None:
    print(f"\n--- Gradient Analysis at Step {step} ---")

    def _grad_stats(prefix: str):
        total, count = 0.0, 0
        for name, param in policy.model.named_parameters():
            if param.requires_grad and prefix in name and param.grad is not None:
                total += param.grad.abs().mean().item() * param.numel()
                count += param.numel()
        return (total / count, count) if count > 0 else (None, 0)

    for label, prefix in [
        ("Vision",         "vision_model"),
        ("Vision LoRA",    "lora_"),
        ("Connector",      "connector"),
        ("State Enc",      "state_encoder"),
        ("Robot CNN",      "robot_visual_encoder"),
        ("Expert Layers",  "expert_layers"),
        ("DiT Layers",     "dit_layers"),
        ("Action In/Out",  "action_"),
        ("Final Norm",     "final_norm"),
        ("Latent Gen",     "latent_generator"),
        ("Lang Adaptor",   "lang_adaptor"),
    ]:
        grad, n = _grad_stats(prefix)
        if grad is not None:
            print(f"  {label:14s} - Avg Abs Grad: {grad:.6f} ({n} params)")
        else:
            print(f"  {label:14s} - no grad")

    if hasattr(policy.model, "latent_generator"):
        gen = policy.model.latent_generator
        w_norm_sq = 0.0
        g_norm_sq = 0.0
        for p in gen.parameters():
            w_norm_sq += p.detach().norm().item() ** 2
            if p.grad is not None:
                g_norm_sq += p.grad.norm().item() ** 2
        out_layer = gen[-1]
        out_w_norm = out_layer.weight.detach().norm().item()
        print(f"  Latent gen     - weight_norm: {w_norm_sq ** 0.5:.4e}   "
              f"grad_norm: {g_norm_sq ** 0.5:.4e}   out_layer_w: {out_w_norm:.4e}")

    if hasattr(policy.model, "lang_attn_bias"):
        bias_tensor = policy.model.lang_attn_bias.detach()
        softplus_vals = F.softplus(bias_tensor).cpu()
        grad = policy.model.lang_attn_bias.grad
        grad_norm_str = f"{grad.norm().item():.4e}" if grad is not None else "None"
        sp_str = "[" + " ".join(f"{v:.2f}" for v in softplus_vals.tolist()) + "]"
        print(f"  Lang attn bias - softplus per-layer: {sp_str}")
        print(f"                   min={softplus_vals.min().item():.3f}  "
              f"max={softplus_vals.max().item():.3f}  "
              f"mean={softplus_vals.mean().item():.3f}  grad_norm: {grad_norm_str}")

    if hasattr(policy.model, "lang_adaptor"):
        w_norm = sum(p.detach().norm().item() ** 2 for p in policy.model.lang_adaptor.parameters()) ** 0.5
        g_norm_sq = sum(p.grad.norm().item() ** 2 for p in policy.model.lang_adaptor.parameters() if p.grad is not None) ** 0.5
        print(f"  Lang adaptor   - weight_norm: {w_norm:.4e}   grad_norm: {g_norm_sq:.4e}")

    # Self-attn (interleaved: one joint softmax; wilro: DiT self-attn). "sink"
    # only exists for wilro; harmless for interleaved (key absent).
    stats = getattr(policy.model, "_last_attention_stats", None)
    xstats = getattr(policy.model, "_last_cross_attention_stats", None)
    if stats:
        order = ["sink", "vision", "language", "state", "robot", "latent", "action"]
        ordered = [(k, stats[k]) for k in order if k in stats]
        cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in ordered)
        label = "self-attn" if xstats else "attn     "
        print(f"  Action→ {label} : {cells}")
    # wilro only: cross-attn to the VLM KV cache (vision vs language).
    if xstats:
        cells = "  ".join(f"{k}={xstats[k]*100:5.1f}%" for k in ("vision", "language") if k in xstats)
        print(f"  Action→ x-attn     : {cells}    (cross-attn to VLM KV)")

    comps = getattr(policy.model, "_last_loss_components", None)
    cw = getattr(policy.model.config, "contrastive_loss_weight", 0.0)
    if comps is not None and cw > 0.0:
        margin = getattr(policy.model.config, "contrastive_margin", 0.05)
        main_v = comps.get("main", float("nan"))
        contr_v = comps.get("contrastive", float("nan"))
        pct = (contr_v / margin * 100.0) if margin > 0 else float("nan")
        print(f"  Contrastive    - main: {main_v:.4f}   contrastive: {contr_v:.4f} "
              f"({pct:.0f}% of margin {margin:.3f})   weight: {cw}")

    print("--- End Gradient Analysis ---\n")

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    model_type: str,
    output_dir: str,
    batch_size: int = 32,
    training_steps: int = 300000,
    resume_from_checkpoint: Optional[str] = None,
    reset_lang_params: bool = False,
    sub_datasets_allowlist: Optional[list[str]] = None,
    sub_datasets_denylist: Optional[list[str]] = None,
    source: str = COMMUNITY_DATASET_REPO,
    max_datasets: Optional[int] = None,
    seed: int = 42,
    version_filter: str = "v3",
    robot_encoder_tokens: int = 49,
    gripper_encoder_tokens: int = 100,
    kv_capture_strategy: str = "last",
    kv_capture_layers: Optional[list[int]] = None,
):
    # Resolve model components
    ConfigCls, PolicyCls, processor_fn, model_defaults = get_model_components(model_type)
    print(f"\nModel type: {model_type}")
    print(f"  Config: {ConfigCls.__name__}")
    print(f"  Policy: {PolicyCls.__name__}")
    print(f"  d_model: {model_defaults['d_model']}")
    print(f"  vision_input_size: {model_defaults['vision_input_size']}")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    workspace = output_directory / "_datasets"
    workspace.mkdir(parents=True, exist_ok=True)

    progress_update_freq = 200
    checkpoint_freq = 2000
    image_transforms = get_augmentations()

    # Canonical image size is model-dependent
    canonical_image_size = model_defaults["vision_input_size"]

    # ── Discover sub-datasets ───────────────────────────────────────────
    all_subs = discover_sub_datasets(source)

    if version_filter != "all":
        want = {"v3": "v3.0", "v2": "v2.1"}[version_filter]
        versions = classify_dataset_versions(source)
        kept = [s for s in all_subs if versions.get(s) == want]
        print(f"[version_filter={version_filter}] kept {len(kept)}/{len(all_subs)} "
              f"datasets matching {want}")
        all_subs = kept

    if sub_datasets_allowlist:
        all_subs = [s for s in all_subs if any(a in s for a in sub_datasets_allowlist)]
    if sub_datasets_denylist:
        all_subs = [s for s in all_subs if not any(d in s for d in sub_datasets_denylist)]

    if max_datasets is not None and len(all_subs) > max_datasets:
        rng = random.Random(seed)
        all_subs = sorted(rng.sample(all_subs, max_datasets))
        print(f"Sampled {max_datasets} of the discovered datasets (seed={seed})")

    print(f"Training on {len(all_subs)} dataset(s) after filtering")

    # ── Load metadata ──────────────────────────────────────────────────
    sub_metas: dict[str, _RawMeta] = {}
    for sub in all_subs:
        info = load_sub_dataset_info(source, sub, workspace)
        if info is not None:
            sub_metas[sub] = _RawMeta(info)
    print(f"Loaded metadata for {len(sub_metas)}/{len(all_subs)} datasets")

    if len(sub_metas) == 0:
        raise RuntimeError("No sub-dataset metadata could be loaded. Aborting.")

    # ── Discover cameras, dims ─────────────────────────────────────────
    sub_cameras = discover_all_camera_names(sub_metas)
    state_dims, action_dims, state_keys, action_keys = discover_state_action_dims(sub_metas)

    print("\nState dim distribution:")
    for sub, d in sorted(state_dims.items()):
        print(f"  {sub}: state_dim={d}  key='{state_keys[sub]}'")
    print("\nAction dim distribution:")
    for sub, d in sorted(action_dims.items()):
        print(f"  {sub}: action_dim={d}  key='{action_keys[sub]}'")
    print("\nCamera distribution:")
    all_cam_names = sorted(set().union(*sub_cameras.values()))
    print(f"  All unique camera names ({len(all_cam_names)}): {all_cam_names}")
    for sub, cams in sorted(sub_cameras.items()):
        print(f"  {sub}: {sorted(cams)}")

    # ── Build camera mapping ────────────────────────────────────────────
    camera_map = build_camera_mapping(sub_cameras, CANONICAL_CAMERAS)
    print("\nCamera mapping (canonical → sub-dataset actual):")
    for sub, cmap in sorted(camera_map.items()):
        mapped = {c: (a if a else "MISSING→ZERO") for c, a in cmap.items()}
        print(f"  {sub}: {mapped}")

    # ── Determine which canonical cameras are actually used ─────────────
    used_canonical: list[str] = []
    for canon in CANONICAL_CAMERAS:
        for sub in sub_metas:
            if camera_map.get(sub, {}).get(canon) is not None:
                used_canonical.append(canon)
                break
    used_canonical = [c for c in CANONICAL_CAMERAS if c in used_canonical]
    if not used_canonical:
        used_canonical = list(CANONICAL_CAMERAS)
    print(f"\nCanonical cameras used: {used_canonical}")

    # ── Load actual datasets ────────────────────────────────────────────
    fps = 10
    obs = 2
    horizon = 64
    n_action_steps = 64

    frame_time = 1.0 / fps
    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    action_temporal_window = [i * frame_time for i in range(horizon)]

    adapters: list[DatasetAdapter] = []
    all_ep_boundaries: list[list[tuple[int, int]]] = []
    dataset_stats_summary: list[tuple[str, int, int]] = []  # (sub, n_frames, n_episodes)

    for sub in sub_metas:
        meta = sub_metas[sub]
        features = dataset_to_policy_features(meta.features)

        delta_ts = {}
        for k, ft in features.items():
            if ft.type == FeatureType.VISUAL:
                delta_ts[k] = [0.0]
            elif ft.type == FeatureType.STATE:
                delta_ts[k] = obs_temporal_window
            elif ft.type == FeatureType.ACTION:
                delta_ts[k] = action_temporal_window

        try:
            full_root = _download_subdir(source, sub, workspace)
            full_root = _ensure_v30(source, full_root, sub)
            ds = LeRobotDataset(
                source,
                root=full_root,
                delta_timestamps=delta_ts,
                force_cache_sync=False,
                tolerance_s=0.04,
            )
        except Exception as e:
            print(f"  [SKIP] {sub}: failed to create/convert LeRobotDataset — {e}")
            continue

        task_map = load_task_descriptions(ds)
        sk = state_keys.get(sub, "observation.state")
        ak = action_keys.get(sub, "action")
        sd = state_dims.get(sub, CANONICAL_STATE_DIM)
        ad = action_dims.get(sub, CANONICAL_ACTION_DIM)
        cmap = camera_map.get(sub, {})

        sst = native_feature_stats(ds, sk, sd)
        ast = native_feature_stats(ds, ak, ad)
        if sst is None or ast is None:
            print(f"  [WARN] {sub}: missing native stats "
                  f"(state={sst is not None}, action={ast is not None}); "
                  f"that feature is left un-normalized for this dataset.")

        adapter = DatasetAdapter(
            dataset=ds,
            sub_dir=sub,
            camera_map=cmap,
            state_key=sk,
            action_key=ak,
            state_dim=sd,
            action_dim=ad,
            task_idx_to_desc=task_map,
            state_stats=sst,
            action_stats=ast,
            normalize_in_adapter=True,
            canonical_image_size=canonical_image_size,
        )
        adapters.append(adapter)

        ep_bounds = get_sub_dataset_ep_boundaries(ds)
        all_ep_boundaries.append(ep_bounds)
        dataset_stats_summary.append((sub, len(ds), len(ep_bounds)))
        print(f"  {sub}: {len(ds)} frames, {len(ep_bounds)} episodes")

    if len(adapters) == 0:
        raise RuntimeError("No datasets could be loaded. Aborting.")

    stitched = StitchedDataset(adapters, all_ep_boundaries)
    ep_from, ep_to = stitched.get_episode_boundaries()

    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from,
        dataset_to_indices=ep_to,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=True,
    )

    dataloader = DataLoader(
        stitched,
        num_workers=8,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    print(f"\nDataLoader: {len(dataloader)} batches/epoch, batch_size={batch_size}")

    # ── Dataset statistics summary (trajectories + frames) ──────────────
    total_frames = sum(n_f for _, n_f, _ in dataset_stats_summary)
    total_episodes = sum(n_e for _, _, n_e in dataset_stats_summary)
    steps_per_epoch = max(1, len(dataloader))
    print(f"\n{'='*64}")
    print(f"Dataset statistics ({len(dataset_stats_summary)} datasets loaded)")
    print(f"{'='*64}")
    print(f"  {'dataset':<34} {'frames':>10} {'episodes':>9}")
    print(f"  {'-'*34} {'-'*10} {'-'*9}")
    for sub, n_f, n_e in sorted(dataset_stats_summary, key=lambda x: -x[1]):
        name = sub if len(sub) <= 34 else "…" + sub[-33:]
        print(f"  {name:<34} {n_f:>10,} {n_e:>9,}")
    print(f"  {'-'*34} {'-'*10} {'-'*9}")
    print(f"  {'TOTAL':<34} {total_frames:>10,} {total_episodes:>9,}")
    avg_len = total_frames / total_episodes if total_episodes else 0.0
    print(f"\n  avg trajectory length: {avg_len:.1f} frames "
          f"({avg_len / fps:.1f}s @ {fps} fps)")
    print(f"  1 epoch = {steps_per_epoch:,} steps (batch_size={batch_size})")
    print(f"  budget {training_steps:,} steps ≈ {training_steps / steps_per_epoch:.2f} epochs")
    print(f"{'='*64}")

    # ── Build config ────────────────────────────────────────────────────
    from lerobot.configs.types import PolicyFeature

    input_feature_specs = {}
    for cam in used_canonical:
        input_feature_specs[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, canonical_image_size, canonical_image_size))
    input_feature_specs["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(CANONICAL_STATE_DIM,),
    )

    output_feature_specs = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(CANONICAL_ACTION_DIM,)),
    }

    # Build config kwargs — common fields for all model types
    cfg_kwargs = dict(
        input_features=input_feature_specs,
        output_features=output_feature_specs,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=CANONICAL_STATE_DIM,
        action_dim=CANONICAL_ACTION_DIM,
        num_vlm_layers=16,
        num_cameras=len(used_canonical),
        cameras_for_vision_state_concat=used_canonical,
        action_dim_weights=[1.0] * CANONICAL_ACTION_DIM,
        pos_decay_lambda=0.0,
        vision_lora_num_layers=0,
        num_latent_tokens=8,
        robot_encoder_tokens=robot_encoder_tokens,
    )

    # Model-specific fields
    if model_type == "interleaved":
        cfg_kwargs["vlm_attends_to_expert"] = True
        cfg_kwargs["gripper_encoder_tokens"] = gripper_encoder_tokens
    elif model_type == "wilro":
        cfg_kwargs["kv_capture_strategy"] = kv_capture_strategy
        if kv_capture_layers is not None:
            cfg_kwargs["kv_capture_layers"] = kv_capture_layers
        cfg_kwargs["gripper_encoder_tokens"] = gripper_encoder_tokens
        cfg_kwargs["use_robot_cnn"] = True
    elif model_type == "wiltechs_vla":
        cfg_kwargs["use_robot_cnn"] = True

    # The community canonical close-range view is "observation.images.wrist"
    # (canonical set is front/wrist/top), NOT the config default
    # "observation.images.gripper". Without this, gripper_camera matches no
    # camera and gripper_encoder_tokens is silently inert — every camera gets
    # robot_encoder_tokens. Point it at the real wrist cam so the dense grid
    # actually applies. (Saved into the checkpoint config → inherited at finetune.)
    if model_type in ("interleaved", "wilro"):
        cfg_kwargs["gripper_camera"] = "observation.images.wrist"

    cfg = ConfigCls(**cfg_kwargs)
    print(f"Robot CNN tokens: {robot_encoder_tokens} per cam "
          f"({int(robot_encoder_tokens ** 0.5)}x{int(robot_encoder_tokens ** 0.5)} grid)")
    if model_type in ("interleaved", "wilro"):
        print(f"Gripper cam '{cfg.gripper_camera}': {gripper_encoder_tokens} "
              f"({int(gripper_encoder_tokens ** 0.5)}x{int(gripper_encoder_tokens ** 0.5)} grid)")
    if model_type == "wilro":
        print(f"KV capture strategy: {kv_capture_strategy}")
        if kv_capture_strategy == "custom" and kv_capture_layers:
            print(f"KV capture layers: {kv_capture_layers}")

    # ── Dataset statistics ───────────────────────────────────────────────
    # Per-dataset normalization: each sub-dataset is z-scored by its own native
    # stats in the adapter, so the global preprocessor is identity. This is the
    # right scheme for heterogeneous multi-robot community data — a single pooled
    # mean/std across different robots would be wrong.
    print("\nPer-dataset normalization: each sub-dataset is z-scored by its own "
          "stats in the adapter; global preprocessor is identity.")
    dataset_stats = identity_stats(used_canonical, CANONICAL_STATE_DIM, CANONICAL_ACTION_DIM)

    # ── Model setup ─────────────────────────────────────────────────────
    if resume_from_checkpoint is not None:
        print(f"\nResuming training from checkpoint: {resume_from_checkpoint}")
        policy = PolicyCls(cfg)
        ckpt_path = Path(resume_from_checkpoint)
        local_ckpt = ckpt_path if ckpt_path.exists() else Path(
            huggingface_hub.snapshot_download(resume_from_checkpoint)
        )
        model_file = local_ckpt / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors found in {local_ckpt}")
            model_file = candidates[0]

        step, epoch = 0, 0
        for cfg_name in ("config.json", "pretrained_config.json"):
            cfg_file = local_ckpt / cfg_name
            if cfg_file.exists():
                with open(cfg_file) as f:
                    saved = json.load(f)
                step = saved.get("training_step", 0)
                epoch = saved.get("training_epoch", 0)
                saved_total = saved.get("training_steps_total", 0)
                if saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from {cfg_name}: step={step}, epoch={epoch}")
                break
        if step == 0 and local_ckpt.name.startswith("checkpoint-"):
            step = int(local_ckpt.name.split("-")[1])

        ckpt_state = load_safetensors(model_file, device=str(device))
        policy.train()
        policy.to(device)
        cur_state = policy.state_dict()
        filtered = {
            k: v for k, v in ckpt_state.items()
            if k in cur_state and cur_state[k].shape == v.shape
        }
        skipped = [k for k in ckpt_state if k not in filtered]
        missing = [k for k in cur_state if k not in ckpt_state]
        if skipped:
            print(f"Skipped {len(skipped)} keys (shape mismatch/removed): {skipped[:5]}")
        if missing:
            print(f"Missing {len(missing)} keys (will use init values): {missing[:5]}")
        policy.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(cur_state)} model keys")

        if reset_lang_params:
            with torch.no_grad():
                if hasattr(policy.model, "lang_attn_bias"):
                    policy.model.lang_attn_bias.zero_()
                    print("Reset lang_attn_bias to zero")
                if hasattr(policy.model, "lang_adaptor"):
                    policy.model.lang_adaptor[1].weight.fill_(1.0)
                    print("Reset lang_adaptor RMSNorm gamma to 1")

        preprocessor, postprocessor = processor_fn(policy.config, dataset_stats=dataset_stats)
        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
        opt_state_path = local_ckpt / "optimizer_state.pth"
        if opt_state_path.exists():
            try:
                optimizer.load_state_dict(torch.load(opt_state_path, map_location=device))
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.optimizer_lr
                print("Optimizer state loaded")
            except ValueError as e:
                print(f"Skipping optimizer state — {e}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.scheduler_warmup_steps, num_training_steps=training_steps,
        )
        for _ in range(step):
            scheduler.step()
        print(f"Scheduler fast-forwarded to step {step}, LR={optimizer.param_groups[0]['lr']:.2e}")
    else:
        policy = PolicyCls(cfg)
        policy.train()
        policy.to(device)
        preprocessor, postprocessor = processor_fn(cfg, dataset_stats=dataset_stats)
        step, epoch = 0, 0
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"Total trainable parameters: {n_trainable:,}  (frozen: {n_frozen:,})")
        optimizer = torch.optim.Adam(trainable_params, lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.scheduler_warmup_steps, num_training_steps=training_steps,
        )

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\nStarting training loop ({training_steps} steps, batch_size={batch_size})...")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)

    while not done:
        epoch += 1
        for batch in dataloader:
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                if "task_description" not in batch:
                    batch["task_description"] = batch["task"]

            present_cams = [c for c in used_canonical if c in batch]
            batch = apply_image_augmentations(batch, present_cams, image_transforms)

            if "observation.state" in batch:
                batch = apply_joint_augmentations(batch, "observation.state")

            if "action_is_pad" not in batch:
                batch["action_is_pad"] = torch.zeros(
                    batch["action"].shape[0], batch["action"].shape[1],
                    dtype=torch.bool, device=batch["action"].device,
                )
            if "action_dim_pad" not in batch:
                batch["action_dim_pad"] = torch.zeros(
                    batch["action"].shape[0], batch["action"].shape[2],
                    dtype=torch.bool, device=batch["action"].device,
                )

            batch = preprocessor(batch)

            if step % progress_update_freq == 0:
                policy.model._capture_attention_stats = True

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            loss.backward()

            if step % progress_update_freq == 0:
                _log_gradient_analysis(policy, step)

            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % progress_update_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")
                prog_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "lr": f"{lr:.2e}",
                    "grad_norm": f"{grad_norm:.2f}",
                })

            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.config.training_step = step
                policy.config.training_epoch = epoch
                policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
                policy.config.current_lr = optimizer.param_groups[0]["lr"]
                policy.config.training_steps_total = training_steps
                policy.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"\nCheckpoint saved at step {step}")

            step += 1
            if step % progress_update_freq == 0 or step >= training_steps:
                prog_bar.update(progress_update_freq)

            if step >= training_steps:
                done = True
                prog_bar.close()
                break

    prog_bar.close()

    policy.config.training_step = step
    policy.config.training_epoch = epoch
    policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
    policy.config.current_lr = optimizer.param_groups[0]["lr"]
    policy.config.training_steps_total = training_steps
    policy.save_pretrained(output_directory)
    torch.save(optimizer.state_dict(), output_directory / "optimizer_state.pth")
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"\nTraining complete. Model saved to {output_directory}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic community-dataset training script supporting multiple model architectures.",
    )
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["interleaved", "wilro", "wiltechs_vla"],
                        help="Model architecture to train. 'interleaved': SmolVLA-style "
                             "joint attention (SmolVLM2-500M). 'wilro': KV-cache → DiT "
                             "cross-attention (SmolVLM2-500M). 'wiltechs_vla': encoder-decoder "
                             "MoT (Qwen3-VL-4B).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (interleaved/wilro ~32, wiltechs_vla ~16 due to 4B VLM)")
    parser.add_argument("--training_steps", type=int, default=300000, help="Total training steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from a checkpoint")
    parser.add_argument("--reset_lang_params", action="store_true",
                        help="Reset language conditioning params after loading checkpoint")
    parser.add_argument("--source", type=str, default=COMMUNITY_DATASET_REPO,
                        help="Data source: HF collection slug, org/user name, or single dataset repo_id "
                             "(default: %(default)s)")
    parser.add_argument("--sub_datasets_allowlist", type=str, nargs="*", default=None,
                        help="Only train on datasets whose subpath CONTAINS any of these strings")
    parser.add_argument("--sub_datasets_denylist", type=str, nargs="*", default=None,
                        help="Exclude datasets whose subpath CONTAINS any of these strings")
    parser.add_argument("--max_datasets", type=int, default=None,
                        help="Cap the number of datasets (random sample across contributors)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --max_datasets sampling")
    parser.add_argument("--list_versions", action="store_true",
                        help="Classify all datasets as v3.0/v2.1/unknown and print the report, then exit")
    parser.add_argument("--version_filter", type=str, default="v3",
                        choices=["all", "v3", "v2"],
                        help="Train only on datasets of this format (default: v3)")
    parser.add_argument("--robot_encoder_tokens", type=int, default=49,
                        help="Robot CNN tokens per non-gripper camera (perfect square, default: 49=7x7)")
    parser.add_argument("--gripper_encoder_tokens", type=int, default=100,
                        help="Robot CNN tokens for gripper/wrist camera (perfect square, default: 100=10x10). "
                             "Used by interleaved and wilro models only.")
    # Wilro-specific options
    parser.add_argument("--kv_capture_strategy", type=str, default="last",
                        choices=["last", "stride2", "custom"],
                        help="[wilro only] Which VLM layers' KV the DiT sources from. "
                             "'last': trailing num_vlm_layers. 'stride2': evenly spaced every "
                             "other layer. 'custom': use --kv_capture_layers. (default: last)")
    parser.add_argument("--kv_capture_layers", type=int, nargs="*", default=None,
                        help="[wilro only] Explicit VLM layer indices for kv_capture_strategy=custom "
                             "(0-based). Example: --kv_capture_layers 3 7 11 15 19 23 27 31")
    args = parser.parse_args()

    if args.list_versions:
        print_version_report(args.source)
        raise SystemExit(0)

    for _name in ("robot_encoder_tokens", "gripper_encoder_tokens"):
        _v = getattr(args, _name)
        if int(_v ** 0.5) ** 2 != _v:
            parser.error(f"--{_name} must be a perfect square, got {_v}")

    kwargs = vars(args)
    kwargs.pop("list_versions", None)
    train(**kwargs)