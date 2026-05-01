"""
Trainable ResNet-18 visual encoder for robot-specific features.

Runs in parallel with the frozen SigLIP ViT:
  SigLIP  — semantic, 14×14px patches, frozen
  ResNet  — spatial, pixel-level precision, fully trainable

ImageNet pretraining gives edge/texture/shape features for free.
Fine-tuning on robot data adapts these to gripper aperture, object
distance, contact state — features SigLIP misses because its patch
size is too coarse and its pretraining domain is internet images.

~11M params (ResNet-18 backbone), negligible vs VLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class RobotVisualEncoder(nn.Module):
    """
    Pretrained ResNet-18 backbone producing spatial feature tokens.

    Uses layers 1–3 of ResNet-18 (output stride 8, 256-channel feature map),
    then adaptive-pools to a fixed token grid and projects to d_model.
    Layer 4 is excluded to keep spatial resolution higher (better for
    precise localisation tasks like grasping).

    Args:
        input_size:  images resized to this square resolution before encoding.
        out_tokens:  spatial tokens per camera (must be a perfect square).
        out_dim:     output feature dim per token — should match d_model.
    """

    def __init__(self, input_size: int = 224, out_tokens: int = 16, out_dim: int = 512):
        super().__init__()
        assert int(out_tokens ** 0.5) ** 2 == out_tokens, "out_tokens must be a perfect square"
        self.input_size = input_size
        token_side = int(out_tokens ** 0.5)

        # Pretrained ResNet-18 backbone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Stem: conv1 + bn1 + relu + maxpool  (224 → 56)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # 56 → 56,  64 ch
        self.layer2 = backbone.layer2   # 56 → 28, 128 ch
        self.layer3 = backbone.layer3   # 28 → 14, 256 ch
        # layer4 excluded — keeps higher spatial resolution for precise localisation

        self.pool = nn.AdaptiveAvgPool2d((token_side, token_side))
        self.proj = nn.Linear(256, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # ImageNet normalisation constants
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) in [0, 1] — raw camera image, any resolution.
        Returns:
            (B, out_tokens, out_dim) float32 feature tokens.
        """
        x = x.float()

        # Resize to fixed input size
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)

        # ImageNet normalisation
        x = (x - self.img_mean) / self.img_std

        feat = self.stem(x)      # (B, 64,  56, 56)
        feat = self.layer1(feat) # (B, 64,  56, 56)
        feat = self.layer2(feat) # (B, 128, 28, 28)
        feat = self.layer3(feat) # (B, 256, 14, 14)

        feat = self.pool(feat)                          # (B, 256, token_side, token_side)
        feat = feat.flatten(2).transpose(1, 2)          # (B, out_tokens, 256)
        return self.norm(self.proj(feat))               # (B, out_tokens, out_dim)
