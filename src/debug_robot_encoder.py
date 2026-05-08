"""
Debug script: visualise RobotVisualEncoder intermediate feature maps.

For each camera in a batch, saves:
  - layer1_<cam>.png  — 64-ch feature map averaged to a heatmap (56×56)
  - layer2_<cam>.png  — 128-ch feature map (28×28)
  - layer3_<cam>.png  — 256-ch feature map (14×14)
  - tokens_<cam>.pt   — raw (out_tokens, out_dim) token tensor

Usage:
  python src/debug_robot_encoder.py \
      --checkpoint outputs/my_run/checkpoint-22000 \
      --dataset_id ISdept/piper_arm \
      --output_dir debug_encoder \
      --num_samples 4
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_flow_matching.transformer_flow_matching_config import TransformerFlowMatchingConfig
from models.transformer_flow_matching.transformer_flow_matching_policy import TransformerFlowMatchingPolicy
from models.transformer_flow_matching.processor_transformer_flow_matching import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
import numpy as np


def save_heatmap(feat_map: torch.Tensor, orig_img: torch.Tensor, title: str, save_path: Path):
    """
    Args:
        feat_map: (C, H, W) feature map from one sample
        orig_img: (3, H, W) original image in [0, 1]
        title: plot title
        save_path: where to save
    """
    # Average across channels → (H, W)
    heatmap = feat_map.float().mean(0).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    img_np = orig_img.float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    # Resize heatmap to original image size
    h, w = img_np.shape[:2]
    heatmap_resized = F.interpolate(
        torch.from_numpy(heatmap)[None, None], size=(h, w), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"{title} heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay: blend heatmap over image
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap_resized)[:, :, :3]
    blend = 0.5 * img_np + 0.5 * heatmap_rgb
    axes[2].imshow(blend.clip(0, 1))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_id", default="ISdept/piper_arm")
    parser.add_argument("--output_dir", default="debug_encoder")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Fixed dataset index to use (random if not set)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = TransformerFlowMatchingPolicy.from_pretrained(args.checkpoint)
    policy.eval().to(device)
    encoder = policy.model.robot_visual_encoder

    # Register forward hooks to capture intermediate outputs
    captured = {}
    hooks = [
        encoder.layer1.register_forward_hook(lambda m, i, o: captured.update({"layer1": o.detach().cpu()})),
        encoder.layer2.register_forward_hook(lambda m, i, o: captured.update({"layer2": o.detach().cpu()})),
        encoder.layer3.register_forward_hook(lambda m, i, o: captured.update({"layer3": o.detach().cpu()})),
    ]

    # Load one batch from the dataset
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    EXCLUDED = {"observation.images.right"}
    camera_keys = sorted([k for k, ft in features.items() if ft.type is FeatureType.VISUAL and k not in EXCLUDED])
    print(f"Cameras: {camera_keys}")

    fps = 10
    delta_timestamps = {
        "observation.state": [-1 / fps, 0.0],
        "action": [i / fps for i in range(policy.config.horizon)],
        **{k: [0.0] for k in camera_keys},
    }
    dataset = LeRobotDataset(args.dataset_id, delta_timestamps=delta_timestamps,
                             force_cache_sync=True, revision="main", tolerance_s=0.005)

    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)

    # Pick sample indices
    rng = np.random.default_rng(42)
    if args.sample_idx is not None:
        indices = [args.sample_idx] * args.num_samples
    else:
        indices = rng.integers(0, len(dataset), size=args.num_samples).tolist()

    print(f"Inspecting {args.num_samples} samples: indices {indices}")

    with torch.no_grad():
        for sample_n, idx in enumerate(indices):
            sample = dataset[idx]
            # Add batch dimension
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in sample.items()}
            batch = preprocessor(batch)

            sample_dir = out_dir / f"sample_{sample_n:02d}_idx{idx}"
            sample_dir.mkdir(exist_ok=True)

            for cam_key in camera_keys:
                if cam_key not in batch:
                    continue
                cam_name = cam_key.split(".")[-1]

                # Run robot_visual_encoder for this camera only
                img = batch[cam_key]
                if img.dim() == 5:
                    img = img[:, -1]
                tokens = encoder(img.float())  # triggers hooks

                # Save raw tokens
                torch.save(tokens[0].cpu(), sample_dir / f"tokens_{cam_name}.pt")
                print(f"  [{cam_name}] tokens shape: {tokens[0].shape}")

                # Save heatmaps for each captured layer
                orig_img_raw = dataset[idx][cam_key]
                if orig_img_raw.dim() == 4:
                    orig_img_raw = orig_img_raw[-1]  # (C, H, W)

                for layer_name in ("layer1", "layer2", "layer3"):
                    feat = captured[layer_name][0]  # (C, H, W) for sample 0
                    channels, fh, fw = feat.shape
                    save_path = sample_dir / f"{layer_name}_{cam_name}.png"
                    save_heatmap(
                        feat, orig_img_raw,
                        title=f"{layer_name} [{channels}ch {fh}×{fw}] — {cam_name}",
                        save_path=save_path,
                    )
                    print(f"  [{cam_name}] {layer_name}: {feat.shape} → {save_path.name}")

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"\nDone. Results saved to: {out_dir.resolve()}")
    print("Each sample folder contains:")
    print("  layer1_<cam>.png  — 64-ch activation heatmap (56×56), overlaid on input")
    print("  layer2_<cam>.png  — 128-ch activation heatmap (28×28)")
    print("  layer3_<cam>.png  — 256-ch activation heatmap (14×14)")
    print("  tokens_<cam>.pt   — raw (16, 512) token tensor — load with torch.load()")


if __name__ == "__main__":
    main()
