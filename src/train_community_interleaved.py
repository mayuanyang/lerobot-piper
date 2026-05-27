"""
Training script for pretraining InterleavedFlowMatching on the HuggingFaceVLA
community_dataset_v3 — a collection of datasets contributed by many different
users with heterogeneous camera names, action dimensions, and state dimensions.

Key differences from `train_libero_interleaved.py` (single-dataset LIBERO train):
  - Loads multiple sub-datasets from `HuggingFaceVLA/community_dataset_v3`.
  - Discovers all unique camera names and state/action dimensions across sub-datasets.
  - Unifies heterogeneous data into a single canonical schema via:
      * Camera-name mapping → canonical camera set (with zero-padding for missing cams)
      * State-dim padding/truncation  → canonical state_dim
      * Action-dim padding/truncation → canonical action_dim (with action_is_pad masking)
  - A per-sub-dataset `DatasetAdapter` wraps a LeRobotDataset and projects its
    features into the canonical schema on-the-fly (cached to disk as a StitchedDataset).
  - Episode-aware sampling is done jointly across all sub-datasets.

Usage:
    python src/train_community_interleaved.py \
        --output_dir outputs/train/community_pretrain \
        --batch_size 32 \
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

from models.interleaved_flow_matching.interleaved_flow_matching_config import InterleavedFlowMatchingConfig
from models.interleaved_flow_matching.interleaved_flow_matching_policy import InterleavedFlowMatchingPolicy
from models.interleaved_flow_matching.processor_interleaved_flow_matching import make_pre_post_processors

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

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
#
# `HuggingFaceVLA/community_dataset_v3` is a SINGLE HF dataset repo that nests
# many LeRobot datasets inside it:
#
#   community_dataset_v3/                 ← the repo
#   ├── <contributor>/
#   │   ├── <dataset_name>/               ← a LeRobot dataset root
#   │   │   ├── data/   episode_*.parquet
#   │   │   ├── videos/ episode_*.mp4
#   │   │   └── meta/   info.json, stats, tasks, episodes
#   │   └── <dataset_name_2>/
#   └── <contributor_2>/ ...
#
# Each `<contributor>/<dataset_name>` directory is an independent dataset. We
# can NOT load them via separate repo_ids (HF repo_ids are exactly
# 'namespace/name', two segments). Instead we treat the whole thing as one
# repo, find every embedded dataset root by its `meta/info.json` marker, and
# load each with `LeRobotDataset(repo_id, root=<local subdir>)`.
# ---------------------------------------------------------------------------
COMMUNITY_DATASET_REPO = "hxma/RoboTwin-LeRobot-v3.0"
INFO_MARKER = "meta/info.json"

# Canonical camera names — these must be a subset of what the interleaved model
# was configured with. Missing cameras in a sub-dataset will be zero-padded.
CANONICAL_CAMERAS = [
    "observation.images.front",
    "observation.images.gripper",
    "observation.images.right",
    "observation.images.top",
    "observation.images.wrist",
]

# Canonical state & action dimensions. Sub-datasets with smaller dims are
# zero-padded; larger dims are truncated (with a warning).
CANONICAL_STATE_DIM = 7
CANONICAL_ACTION_DIM = 7

# ---------------------------------------------------------------------------
# Augmentation (same recipe as train_libero_interleaved.py)
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
        has_time_dim = sample_img.dim() == 4  # (T, C, H, W)
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
    """
    Walk a single HF dataset repo and return the relative paths of every
    embedded LeRobot dataset root, identified by a `meta/info.json` marker.

    For community_dataset_v3 these look like 'yangfengzzz/pick_place_v1'
    ('<contributor>/<dataset_name>'), but depth is not assumed — any directory
    containing 'meta/info.json' is a dataset root. '' means the repo root
    itself is a dataset.

    Only lists file names (no data download), so it is cheap.
    """
    files = list_repo_files(repo_id, repo_type="dataset")
    roots: set[str] = set()
    for f in files:
        if f.endswith(INFO_MARKER):
            root = f[: -len(INFO_MARKER)].rstrip("/")
            roots.add(root)            # '' if the marker sits at the repo root
    roots_sorted = sorted(roots)
    print(f"[discover] Found {len(roots_sorted)} LeRobot dataset roots in {repo_id} "
          f"(scanned {len(files)} files)")
    return roots_sorted


def classify_dataset_versions(repo_id: str = COMMUNITY_DATASET_REPO) -> dict[str, str]:
    """
    Classify every embedded dataset root as 'v3.0', 'v2.1', or 'unknown'
    using ONLY the repo file listing (one API call, no downloads).

    Markers (the meta-file layout differs between versions):
      v3.0 → meta/tasks.parquet exists, or a meta/episodes/ subdirectory exists
      v2.1 → meta/tasks.jsonl or meta/episodes.jsonl exists
      unknown → neither marker present (corrupt / partial upload)

    Note: the `codebase_version` string inside info.json can be stale/edited
    (some datasets here carry an info.json.bak), so we trust the actual file
    layout instead of that field.
    """
    files = list_repo_files(repo_id, repo_type="dataset")
    fileset = set(files)

    # Roots that have a meta/episodes/ subdir (a v3.0 marker) — precomputed.
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
    """Classify all datasets and print a grouped report. Returns the mapping."""
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
    """
    Download one embedded dataset's files into a WRITABLE workspace dir and
    return the local path to that dataset's root.

    Uses `local_dir=workspace` so files are materialised as real files (not
    HF-cache symlinks). This matters because v2.1→v3.0 conversion rewrites
    files in place — doing that inside the symlinked HF cache would corrupt
    the content-addressed blob store.

    `patterns` lets callers fetch only part of the subdir (e.g. just `meta/**`
    for cheap metadata discovery). When None, the whole subdir is fetched.
    """
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
    """
    Read a sub-dataset's raw `meta/info.json` (parsed dict) without going
    through LeRobotDatasetMetadata — which raises on v2.1 datasets. We only
    need `features` and `fps` for camera/dim discovery, both of which live in
    info.json regardless of codebase version.

    Returns the parsed dict, or None on failure.
    """
    try:
        meta_patterns = [f"{subpath}/meta/info.json"] if subpath else ["meta/info.json"]
        root = _download_subdir(repo_id, subpath, workspace, patterns=meta_patterns)
        with open(root / "meta" / "info.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to read info.json for {repo_id}:{subpath or '<root>'}: {e}")
        return None


class _RawMeta:
    """
    Minimal stand-in for LeRobotDatasetMetadata exposing just `.features` and
    `.fps`, parsed straight from info.json. Lets the camera/dim discovery
    helpers work on v2.1 datasets (which LeRobotDatasetMetadata refuses to
    load) since they only read `meta.features`.
    """

    def __init__(self, info: dict):
        self.features = info.get("features", {})
        self.fps = info.get("fps", 30)
        self.codebase_version = str(info.get("codebase_version", "")).lstrip("v")


def _ensure_v30(repo_id: str, root: Path, subpath: str) -> Path:
    """
    Ensure the dataset at `root` is in v3.0 format, converting in place from
    v2.1 if needed. Idempotent: a dataset already at v3.0 is left untouched,
    so re-runs skip the (expensive) conversion.

    Conversion runs fully local (`push_to_hub=False`); since `root` already
    holds the v2.1 files, convert_dataset skips any hub download.
    """
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        version = str(json.load(f).get("codebase_version", "")).lstrip("v")
    if not version.startswith("2"):
        return root  # already v3.0 (or newer) — nothing to do

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
    """
    For every sub-dataset, find the set of VISUAL feature keys.
    Returns {sub_dir: {cam_key, ...}}.
    """
    result: dict[str, set[str]] = {}
    for sub_dir, meta in sub_metas.items():
        features = dataset_to_policy_features(meta.features)
        cams = {k for k, ft in features.items() if ft.type == FeatureType.VISUAL}
        result[sub_dir] = cams
    return result


def discover_state_action_dims(
    sub_metas: dict[str, "_RawMeta"],
) -> tuple[dict[str, int], dict[str, int], dict[str, str], dict[str, str]]:
    """
    Returns:
      state_dims:  {sub_dir: state_dim}
      action_dims: {sub_dir: action_dim}
      state_keys:  {sub_dir: state_feature_key}
      action_keys: {sub_dir: action_feature_key}
    """
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
    """
    Build a mapping from canonical_camera → sub_dataset_camera for each sub-dataset.

    Strategy: exact name match first, then case-insensitive match on the last
    segment (e.g. 'front', 'gripper', 'right', 'top', 'wrist'). If no match,
    the canonical camera is mapped to None (will be zero-padded).

    Returns:
      {sub_dir: {canonical_cam: actual_cam_key_or_None}}
    """
    mapping: dict[str, dict[str, Optional[str]]] = {}
    for sub_dir, cams in sub_cameras.items():
        sub_map: dict[str, Optional[str]] = {}
        cams_lower = {c.lower(): c for c in cams}
        for canon in canonical:
            # Exact match
            if canon in cams:
                sub_map[canon] = canon
                continue
            # Match by last segment (e.g. "observation.images.front" → "front")
            canon_suffix = canon.split(".")[-1].lower()
            matched = False
            for cam_lower, cam_orig in cams_lower.items():
                if cam_lower.endswith(canon_suffix):
                    sub_map[canon] = cam_orig
                    matched = True
                    break
            if not matched:
                sub_map[canon] = None  # will be zero-padded
        mapping[sub_dir] = sub_map
    return mapping

# ---------------------------------------------------------------------------
# DatasetAdapter — wraps a LeRobotDataset, projects features into canonical schema
# ---------------------------------------------------------------------------
class DatasetAdapter(Dataset):
    """
    Wraps a single LeRobotDataset (from a sub-dataset) and projects its features
    into the canonical schema:
      - Camera keys are renamed/mapped; missing cameras are zero-tensors.
      - State is padded/truncated to CANONICAL_STATE_DIM.
      - Action is padded/truncated to CANONICAL_ACTION_DIM.
      - action_is_pad is created if not present, marking padded action dims.
      - task_description is populated if available.
    """

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
    ):
        self.dataset = dataset
        self.sub_dir = sub_dir
        self.camera_map = camera_map  # canonical → actual or None
        self.state_key = state_key
        self.action_key = action_key
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.canonical_state_dim = CANONICAL_STATE_DIM
        self.canonical_action_dim = CANONICAL_ACTION_DIM
        self.task_idx_to_desc = task_idx_to_desc or {}

        # Cache canonical camera list for iteration
        self._canonical_cams = list(camera_map.keys())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        # HF datasets reject numpy integer keys; coerce to a Python int so
        # this works whether the index comes from a sampler, ConcatDataset,
        # or np.random.choice.
        #
        # Community data has occasional corrupt/undecodable videos (AV1 packets
        # that torchcodec rejects). A single bad sample must not crash the whole
        # DataLoader worker — retry with random neighbours, then give up.
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

        # ── Camera remapping ──────────────────────────────────────────
        for canon_cam in self._canonical_cams:
            actual_key = self.camera_map.get(canon_cam)
            if actual_key is not None and actual_key in item:
                # Rename to canonical
                if actual_key != canon_cam:
                    item[canon_cam] = item.pop(actual_key)
                # Ensure proper shape: (T, C, H, W)
                val = item[canon_cam]
                if isinstance(val, torch.Tensor) and val.dim() == 3:
                    item[canon_cam] = val.unsqueeze(0)  # add T dimension
            elif actual_key is None:
                # Create a zero-tensor placeholder with the same shape as other cams
                ref_cam = next((k for k in self._canonical_cams if self.camera_map.get(k) and k in item), None)
                if ref_cam is not None:
                    ref_shape = item[ref_cam].shape
                    item[canon_cam] = torch.zeros(ref_shape, dtype=item[ref_cam].dtype)
                else:
                    # Fallback: (1, 3, 384, 384) placeholder
                    item[canon_cam] = torch.zeros(1, 3, 384, 384)

        # ── State padding / truncation ────────────────────────────────
        if self.state_key in item:
            state = item[self.state_key]
            if isinstance(state, torch.Tensor):
                sd = state.shape[-1] if state.dim() >= 1 else 1
                if sd < self.canonical_state_dim:
                    pad = torch.zeros(*state.shape[:-1], self.canonical_state_dim - sd,
                                     dtype=state.dtype, device=state.device)
                    state = torch.cat([state, pad], dim=-1)
                elif sd > self.canonical_state_dim:
                    state = state[..., :self.canonical_state_dim]
                item[self.state_key] = state

        # ── Action padding / truncation + pad mask ────────────────────
        if self.action_key in item:
            action = item[self.action_key]
            if isinstance(action, torch.Tensor):
                ad = action.shape[-1] if action.dim() >= 1 else 1
                if ad < self.canonical_action_dim:
                    pad = torch.zeros(*action.shape[:-1], self.canonical_action_dim - ad,
                                     dtype=action.dtype, device=action.device)
                    action = torch.cat([action, pad], dim=-1)
                    # Build action_is_pad if needed
                    existing_pad_key = None
                    for k in item:
                        if "pad" in k.lower() and "action" in k.lower():
                            existing_pad_key = k
                            break
                    if existing_pad_key and existing_pad_key in item:
                        ep = item[existing_pad_key]
                        if isinstance(ep, torch.Tensor):
                            # Broadcast existing pad mask to canonical dim
                            if ep.dim() == 1:
                                ep = ep.unsqueeze(-1).expand(-1, self.canonical_action_dim)
                            elif ep.shape[-1] == ad:
                                ep_pad = torch.ones(*ep.shape[:-1], self.canonical_action_dim - ad,
                                                   dtype=ep.dtype, device=ep.device)
                                ep = torch.cat([ep, ep_pad], dim=-1)
                            elif ep.shape[-1] == self.canonical_action_dim:
                                pass  # already correct
                            item[existing_pad_key] = ep
                    else:
                        # Create action_is_pad: real dims = 0 (not padded), extra dims = 1
                        is_pad = torch.zeros(*action.shape[:-1], self.canonical_action_dim,
                                            dtype=torch.bool)
                        is_pad[..., ad:] = True
                        item["action_is_pad"] = is_pad
                elif ad > self.canonical_action_dim:
                    action = action[..., :self.canonical_action_dim]
                item[self.action_key] = action

        # ── Rename action key to canonical "action" if needed ─────────
        if self.action_key != "action" and self.action_key in item:
            item["action"] = item.pop(self.action_key)
        # Also rename common state keys
        if self.state_key != "observation.state" and self.state_key in item:
            item["observation.state"] = item.pop(self.state_key)

        # ── Task description ──────────────────────────────────────────
        if "task_description" not in item and "task" not in item:
            task_idx = item.get("task_index")
            if task_idx is not None:
                if isinstance(task_idx, torch.Tensor):
                    ti = int(task_idx.item()) if task_idx.numel() == 1 else int(task_idx[0].item())
                else:
                    ti = int(task_idx)
                desc = self.task_idx_to_desc.get(ti, "")
                item["task_description"] = desc

        return item


def load_task_descriptions(dataset: LeRobotDataset) -> dict[int, str]:
    """Try to load task_index → task_description from tasks.parquet."""
    task_map: dict[int, str] = {}
    try:
        tasks_path = dataset.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            df = pd.read_parquet(tasks_path)
            if "task_index" in df.columns:
                task_map = {int(row["task_index"]): str(idx) for idx, row in df.iterrows()}
    except Exception:
        pass
    return task_map

# ---------------------------------------------------------------------------
# StitchedDataset — ConcatDataset that tracks per-dataset episode boundaries
# ---------------------------------------------------------------------------
class StitchedDataset(ConcatDataset):
    """
    Concatenates multiple DatasetAdapters and tracks cumulative lengths so we
    can build an EpisodeAwareSampler across all sub-datasets.

    Each sub-dataset's episode boundaries are offset by the cumulative frame
    count of all preceding sub-datasets.
    """

    def __init__(self, datasets: list[DatasetAdapter], ep_boundaries: list[list[int]]):
        super().__init__(datasets)
        # ep_boundaries[i] = list of (start_frame, end_frame) for sub-dataset i
        # We offset them to global frame indices.
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
    """Extract episode (start, end) frame indices from a LeRobotDataset."""
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
# Main training function
# ---------------------------------------------------------------------------
def train(
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
    version_filter: str = "all",
):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Writable workspace for downloaded + converted sub-datasets. Kept outside
    # the HF symlink cache so v2.1→v3.0 conversion can rewrite files in place.
    workspace = output_directory / "_datasets"
    workspace.mkdir(parents=True, exist_ok=True)

    progress_update_freq = 200
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

    # ── Discover sub-datasets ───────────────────────────────────────────
    # `all_subs` contains RELATIVE subpaths inside the `source` repo
    # (e.g. 'yangfengzzz/pick_place_v1'), each marked by meta/info.json.
    all_subs = discover_sub_datasets(source)

    # Version filter: optionally restrict to v3.0 (skip conversion) or v2.1.
    # Classification is cheap (one file-listing call, no downloads).
    if version_filter != "all":
        want = {"v3": "v3.0", "v2": "v2.1"}[version_filter]
        versions = classify_dataset_versions(source)
        kept = [s for s in all_subs if versions.get(s) == want]
        print(f"[version_filter={version_filter}] kept {len(kept)}/{len(all_subs)} "
              f"datasets matching {want}")
        all_subs = kept

    # Allowlist/denylist match by substring on the subpath.
    if sub_datasets_allowlist:
        all_subs = [s for s in all_subs if any(a in s for a in sub_datasets_allowlist)]
    if sub_datasets_denylist:
        all_subs = [s for s in all_subs if not any(d in s for d in sub_datasets_denylist)]

    # Cap the number of datasets (random sample for a representative subset
    # across contributors, not just the alphabetically-first ones). Seeded
    # for reproducibility — same seed → same subset.
    if max_datasets is not None and len(all_subs) > max_datasets:
        import random
        rng = random.Random(seed)
        all_subs = sorted(rng.sample(all_subs, max_datasets))
        print(f"Sampled {max_datasets} of the discovered datasets (seed={seed})")

    print(f"Training on {len(all_subs)} dataset(s) after filtering")

    # ── Load metadata for all sub-datasets ──────────────────────────────
    # Keys are relative subpaths inside `source`. Only meta/info.json is read
    # here (raw JSON, v2.1-safe) for camera/dim discovery; data/ + videos/ are
    # downloaded — and converted to v3.0 if needed — lazily in the build loop.
    sub_metas: dict[str, _RawMeta] = {}
    for sub in all_subs:
        info = load_sub_dataset_info(source, sub, workspace)
        if info is not None:
            sub_metas[sub] = _RawMeta(info)
    print(f"Loaded metadata for {len(sub_metas)}/{len(all_subs)} datasets")

    if len(sub_metas) == 0:
        raise RuntimeError("No sub-dataset metadata could be loaded. Aborting.")

    # ── Discover cameras, state/action dims ─────────────────────────────
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

    # ── Determine which canonical cameras are actually used by ≥1 dataset ─
    used_canonical: list[str] = []
    for canon in CANONICAL_CAMERAS:
        for sub in sub_metas:
            if camera_map.get(sub, {}).get(canon) is not None:
                used_canonical.append(canon)
                break
    # Keep the order from CANONICAL_CAMERAS
    used_canonical = [c for c in CANONICAL_CAMERAS if c in used_canonical]
    if not used_canonical:
        # Fallback: use all canonical if nothing found
        used_canonical = list(CANONICAL_CAMERAS)
    print(f"\nCanonical cameras used: {used_canonical}")

    # ── Load actual datasets ────────────────────────────────────────────
    fps = 10  # default; may be overridden per-dataset
    obs = 2
    horizon = 64
    n_action_steps = 64

    frame_time = 1.0 / fps
    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    action_temporal_window = [i * frame_time for i in range(horizon)]

    adapters: list[DatasetAdapter] = []
    all_ep_boundaries: list[list[tuple[int, int]]] = []

    for sub in sub_metas:
        meta = sub_metas[sub]
        features = dataset_to_policy_features(meta.features)

        # Build delta_timestamps for this sub-dataset
        delta_ts = {}
        for k, ft in features.items():
            if ft.type == FeatureType.VISUAL:
                delta_ts[k] = [0.0]
            elif ft.type == FeatureType.STATE:
                delta_ts[k] = obs_temporal_window
            elif ft.type == FeatureType.ACTION:
                delta_ts[k] = action_temporal_window

        try:
            # Download the full subdir (data/ + videos/), convert v2.1→v3.0 in
            # place if needed, then load locally via `root`. All sub-datasets
            # share the same repo_id (`source`); `root` disambiguates them.
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

        adapter = DatasetAdapter(
            dataset=ds,
            sub_dir=sub,
            camera_map=cmap,
            state_key=sk,
            action_key=ak,
            state_dim=sd,
            action_dim=ad,
            task_idx_to_desc=task_map,
        )
        adapters.append(adapter)

        # Collect episode boundaries
        ep_bounds = get_sub_dataset_ep_boundaries(ds)
        all_ep_boundaries.append(ep_bounds)
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

    # ── Build config ────────────────────────────────────────────────────
    # Construct synthetic input/output features matching the canonical schema.
    # lerobot 0.4.0 uses `PolicyFeature(type=..., shape=...)` — the same class
    # `dataset_to_policy_features` returns. (There is no `FeatureSpec`.)
    from lerobot.configs.types import PolicyFeature

    input_feature_specs = {}
    for cam in used_canonical:
        input_feature_specs[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 384, 384))
    input_feature_specs["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(CANONICAL_STATE_DIM,),
    )

    output_feature_specs = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(CANONICAL_ACTION_DIM,)),
    }

    cfg = InterleavedFlowMatchingConfig(
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
        vlm_attends_to_expert=True,
    )

    # ── Compute dataset statistics across all sub-datasets ───────────────
    # Since we're stitching many heterogeneous datasets, we compute unified
    # stats by sampling frames from each sub-dataset.
    dataset_stats = compute_unified_stats(adapters, used_canonical, CANONICAL_STATE_DIM, CANONICAL_ACTION_DIM)

    # ── Model setup ─────────────────────────────────────────────────────
    if resume_from_checkpoint is not None:
        print(f"\nResuming training from checkpoint: {resume_from_checkpoint}")
        policy = InterleavedFlowMatchingPolicy(cfg)
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

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, dataset_stats=dataset_stats,
        )
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
        policy = InterleavedFlowMatchingPolicy(cfg)
        policy.train()
        policy.to(device)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)
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

            # Task description handling
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                if "task_description" not in batch:
                    batch["task_description"] = batch["task"]

            # Image augmentations
            present_cams = [c for c in used_canonical if c in batch]
            batch = apply_image_augmentations(batch, present_cams, image_transforms)

            # State augmentation
            if "observation.state" in batch:
                batch = apply_joint_augmentations(batch, "observation.state")

            # Ensure action_is_pad exists
            if "action_is_pad" not in batch:
                batch["action_is_pad"] = torch.zeros(
                    batch["action"].shape[0], batch["action"].shape[1],
                    dtype=torch.bool, device=batch["action"].device,
                )

            batch = preprocessor(batch)

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            loss.backward()

            # Gradient analysis (periodic)
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

    # ── Final save ──────────────────────────────────────────────────────
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
# Unified statistics computation
# ---------------------------------------------------------------------------
def compute_unified_stats(
    adapters: list[DatasetAdapter],
    camera_keys: list[str],
    state_dim: int,
    action_dim: int,
    max_samples: int = 5000,
) -> dict:
    """Compute mean/std statistics across all sub-datasets by sampling frames."""
    print("\nComputing unified dataset statistics...")
    all_states = []
    all_actions = []

    total_frames = sum(len(a) for a in adapters)
    sample_ratio = min(1.0, max_samples / max(total_frames, 1))

    for adapter in adapters:
        n_samples = max(1, int(len(adapter) * sample_ratio))
        indices = np.random.choice(len(adapter), n_samples, replace=False)
        for idx in indices:
            item = adapter[int(idx)]   # HF dataset rejects numpy.int64 keys
            if "observation.state" in item:
                s = item["observation.state"]
                if isinstance(s, torch.Tensor):
                    s = s.numpy()
                all_states.append(np.asarray(s).reshape(-1)[:state_dim])
            if "action" in item:
                a = item["action"]
                if isinstance(a, torch.Tensor):
                    a = a.numpy()
                all_actions.append(np.asarray(a).reshape(-1)[:action_dim])

    if len(all_states) == 0:
        all_states = [np.zeros(state_dim)]
    if len(all_actions) == 0:
        all_actions = [np.zeros(action_dim)]

    all_states = np.stack(all_states).astype(np.float32)
    all_actions = np.stack(all_actions).astype(np.float32)

    stats = {
        "observation.state": {
            "mean": torch.from_numpy(all_states.mean(axis=0)),
            "std": torch.from_numpy(all_states.std(axis=0).clip(min=1e-6)),
            "min": torch.from_numpy(all_states.min(axis=0)),
            "max": torch.from_numpy(all_states.max(axis=0)),
        },
        "action": {
            "mean": torch.from_numpy(all_actions.mean(axis=0)),
            "std": torch.from_numpy(all_actions.std(axis=0).clip(min=1e-6)),
            "min": torch.from_numpy(all_actions.min(axis=0)),
            "max": torch.from_numpy(all_actions.max(axis=0)),
        },
    }
    # Vision features use identity normalization
    for cam in camera_keys:
        stats[cam] = {
            "mean": torch.tensor([0.0]),
            "std": torch.tensor([1.0]),
            "min": torch.tensor([-1.0]),
            "max": torch.tensor([1.0]),
        }
    print(f"  Sampled {len(all_states)} state frames, {len(all_actions)} action frames")
    print(f"  State mean: {stats['observation.state']['mean'].numpy()}")
    print(f"  State std:  {stats['observation.state']['std'].numpy()}")
    return stats

# ---------------------------------------------------------------------------
# Gradient analysis (same as train_libero_interleaved.py)
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

    # Latent generator (task-conditional MLP) — replaces static `latent_embs`.
    # `out_layer_w` is the cleanest health signal: zero-init at training start,
    # grows as the model learns to produce non-trivial task-conditional latents.
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
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain InterleavedFlowMatching on HuggingFaceVLA/community_dataset_v3",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (interleaved model is memory-heavy)")
    parser.add_argument("--training_steps", type=int, default=300000, help="Total training steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from a checkpoint")
    parser.add_argument("--reset_lang_params", action="store_true",
                        help="Reset language conditioning params after loading checkpoint")
    parser.add_argument("--source", type=str, default=COMMUNITY_DATASET_REPO,
                        help="Data source: HF collection slug, org/user name, or single dataset repo_id "
                             "(default: %(default)s)")
    parser.add_argument("--sub_datasets_allowlist", type=str, nargs="*", default=None,
                        help="Only train on datasets whose subpath CONTAINS any of these strings "
                             "(substring match, case-sensitive)")
    parser.add_argument("--sub_datasets_denylist", type=str, nargs="*", default=None,
                        help="Exclude datasets whose subpath CONTAINS any of these strings")
    parser.add_argument("--max_datasets", type=int, default=None,
                        help="Cap the number of datasets (random sample across contributors). "
                             "Useful to limit download size when testing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --max_datasets sampling (reproducible subset)")
    parser.add_argument("--list_versions", action="store_true",
                        help="Classify all datasets as v3.0/v2.1/unknown and print the report, "
                             "then exit (no download, no training).")
    parser.add_argument("--version_filter", type=str, default="all",
                        choices=["all", "v3", "v2"],
                        help="Train only on datasets of this format. 'v3' skips the slow "
                             "v2.1→v3.0 conversion entirely. (default: all)")
    args = parser.parse_args()

    # --list_versions is a standalone inspection mode.
    if args.list_versions:
        print_version_report(args.source)
        raise SystemExit(0)

    # SystemExit-only flag — drop it before calling train().
    kwargs = vars(args)
    kwargs.pop("list_versions", None)
    train(**kwargs)