"""
Trainable DINOv3 ViT-S/16 visual encoder for robot-specific features.

Replaces the ResNet-18 RobotVisualEncoder with a pretrained DINOv3 ViT-S/16
backbone, providing stronger visual representations through self-supervised
pretraining on large-scale image data.

Architecture:
  - DINOv3 ViT-S/16: 12 layers, 384 hidden dim, 16×16 patches
  - Output: spatial feature tokens projected to d_model
  - Supports freezing, LoRA fine-tuning, or full fine-tuning

~22M params (ViT-S/16 backbone), comparable to ResNet-18 (~11M) but with
significantly stronger pretrained features.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV3VisualEncoder(nn.Module):
    """
    DINOv3 ViT-S/16 backbone producing spatial feature tokens.

    Uses the pretrained DINOv3 ViT-S/16 model to extract visual features,
    then projects them to the target output dimension.

    Args:
        pretrained: Load pretrained DINOv3 weights (default: True).
        freeze: Freeze all DINOv3 parameters (default: True for initial training).
        lora_rank: If > 0, apply LoRA to attention layers (default: 0 = disabled).
        lora_alpha: LoRA scaling factor (default: 16).
        lora_dropout: LoRA dropout (default: 0.05).
        input_size: Images resized to this square resolution (default: 224).
        out_tokens: Number of output spatial tokens (default: 196 = 14×14 for 224/16).
        out_dim: Output feature dim per token — should match d_model.
    """

    # DINOv3 ViT-S/16 model IDs on HuggingFace
    # Pretrained on LVD-1689M dataset (1.6B images)
    MODEL_ID_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"  # ViT-S/16, 384 dim

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        input_size: int = 224,
        out_tokens: int = 196,
        out_dim: int = 960,
    ):
        super().__init__()
        self.input_size = input_size
        self.out_tokens = out_tokens
        self.freeze_backbone = freeze

        # Calculate expected token count from input size
        patch_size = 16
        expected_tokens = (input_size // patch_size) ** 2  # 224/16 = 14, 14² = 196

        # Load DINOv3 model
        if pretrained:
            print(f"[DinoV3] Loading pretrained {self.MODEL_ID_SMALL} ...")
            try:
                from transformers import AutoModel, AutoImageProcessor

                self.backbone = AutoModel.from_pretrained(
                    self.MODEL_ID_SMALL,
                    trust_remote_code=True,
                )
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.MODEL_ID_SMALL,
                    trust_remote_code=True,
                )
                # DINOv3 ViT-S hidden size is 384
                dinov3_hidden = self.backbone.config.hidden_size
                print(f"[DinoV3] Loaded: hidden_size={dinov3_hidden}, "
                      f"num_layers={self.backbone.config.num_hidden_layers}")
            except Exception as e:
                print(f"[DinoV3] Failed to load from HF: {e}")
                print("[DinoV3] Falling back to manual ViT construction")
                dinov3_hidden = 384
                self.backbone = self._build_vit_from_scratch()
                self.image_processor = None
        else:
            print("[DinoV3] Building ViT-S/16 from scratch (no pretrained weights)")
            dinov3_hidden = 384
            self.backbone = self._build_vit_from_scratch()
            self.image_processor = None

        # Freeze backbone if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("[DinoV3] Backbone frozen")

        # Apply LoRA if requested
        if lora_rank > 0 and not freeze:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)
            print(f"[DinoV3] LoRA applied: rank={lora_rank}, alpha={lora_alpha}")

        # Projection layer: DINOv3 hidden → target out_dim
        self.proj = nn.Linear(dinov3_hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # Adaptive pooling if out_tokens != expected_tokens
        self.use_adaptive_pool = out_tokens != expected_tokens
        if self.use_adaptive_pool:
            pool_side = int(out_tokens ** 0.5)
            assert pool_side * pool_side == out_tokens, "out_tokens must be a perfect square"
            self.pool_side = pool_side
            print(f"[DinoV3] Adaptive pooling: {expected_tokens} → {out_tokens} tokens")

        # DINOv3 image normalization
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _build_vit_from_scratch(self) -> nn.Module:
        """Build a ViT-S/16 from scratch when pretrained loading fails."""
        from transformers import ViTConfig, ViTModel

        config = ViTConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            patch_size=16,
            image_size=self.input_size,
        )
        return ViTModel(config)

    def _apply_lora(self, rank: int, alpha: int, dropout: float):
        """Apply LoRA to attention query and value projections."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=["query", "value"],
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
        except ImportError:
            print("[DinoV3] peft not installed, skipping LoRA. Install with: pip install peft")

    def forward(self, x: torch.Tensor, out_tokens: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) in [0, 1] — raw camera image, any resolution.
            out_tokens: override the token count for this call. If different from
                the construction default, adaptive pooling is applied.
                None → use the construction default.
        Returns:
            (B, out_tokens, out_dim) float32 feature tokens.
        """
        x = x.float()

        # Resize to fixed input size
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )

        # ImageNet normalization
        x = (x - self.img_mean) / self.img_std

        # DINOv3 forward pass
        # The backbone expects pixel_values in [0, 1] with ImageNet normalization
        if hasattr(self.backbone, 'embeddings'):
            # transformers ViTModel / DINOv3 model
            outputs = self.backbone(pixel_values=x, output_hidden_states=False)
            # last_hidden_state: (B, seq_len, hidden_size)
            # seq_len = 1 (CLS) + num_patches
            feat = outputs.last_hidden_state  # (B, 1 + num_patches, hidden)
            # Remove CLS token, keep only patch tokens
            feat = feat[:, 1:, :]  # (B, num_patches, hidden)
        else:
            raise RuntimeError("Unexpected backbone structure")

        # Adaptive pooling if needed
        target_tokens = out_tokens if out_tokens is not None else self.out_tokens
        current_tokens = feat.shape[1]

        if target_tokens != current_tokens:
            target_side = int(target_tokens ** 0.5)
            assert target_side * target_side == target_tokens, "target out_tokens must be a perfect square"

            # DINOv3 may include register tokens (e.g., 4 registers), so current_tokens
            # may not be a perfect square (e.g., 196 patches + 4 registers = 200).
            # We need to extract only the spatial patch tokens and reshape to 2D grid.
            # Try to infer the grid size from the input image size and patch size.
            patch_size = 16
            expected_patches = (self.input_size // patch_size) ** 2  # 196 for 224/16

            if current_tokens == expected_patches:
                # No extra tokens, direct reshape
                grid_side = int(expected_patches ** 0.5)
                feat_2d = feat.transpose(1, 2).view(feat.shape[0], -1, grid_side, grid_side)
            elif current_tokens > expected_patches:
                # Extra tokens present (registers or other). Take only the last
                # expected_patches tokens as spatial patches.
                feat_spatial = feat[:, -expected_patches:, :]
                grid_side = int(expected_patches ** 0.5)
                feat_2d = feat_spatial.transpose(1, 2).view(feat.shape[0], -1, grid_side, grid_side)
            else:
                # Fewer tokens than expected — use 1D adaptive pooling as fallback
                # (B, N, D) → (B, D, N) → pool → (B, D, target) → (B, target, D)
                feat_1d = feat.transpose(1, 2)  # (B, D, N)
                feat_pooled = F.adaptive_avg_pool1d(feat_1d, target_tokens)
                return self.norm(self.proj(feat_pooled.transpose(1, 2)))

            feat_2d = F.adaptive_avg_pool2d(feat_2d, (target_side, target_side))
            feat = feat_2d.flatten(2).transpose(1, 2)  # (B, target_tokens, hidden)

        # Project to output dimension
        return self.norm(self.proj(feat))  # (B, out_tokens, out_dim)

    def train(self, mode: bool = True):
        """Override to keep frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self