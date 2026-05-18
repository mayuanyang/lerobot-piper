"""
RobotFrameVQVAE: Discrete visual tokenizer for robot camera frames.

Converts (B, 3, 128, 128) images → (B, 256) integer token IDs from a codebook.
The 256 = 16×16 spatial grid comes from 3 stride-2 downsampling steps (128 / 8).

Why train on robot data instead of using a pretrained tokenizer:
  - Codebook learns manipulation-relevant patterns: gripper aperture, object
    contact, table surface, depth cues — things DALL-E / Emu3 tokenizers
    never saw in their internet image training.
  - Codebook size 1024 is deliberately small (vs. 131072 in Emu3) so that
    cross-entropy loss in the future-frame decoder stays tractable.
  - Training takes ~1-2 hours on robot data, not days.

Architecture:
  Encoder:  128x128 → 64x64 → 32x32 → 16x16 (3 stride-2 convs + residuals)
  VQ:       each 16x16 spatial position → nearest code in (1024, 256) codebook
  Decoder:  16x16 → 32x32 → 64x64 → 128x128 (mirror of encoder)

Training loss:
  L = ||x - x_hat||^2  +  β * ||z_e - sg(codebook_entry)||^2
  Codebook updated via EMA (more stable than straight-through estimator alone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Single residual block with GroupNorm (works at small batch sizes)."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    """
    128×128 → 16×16 spatial, 256-channel feature map.
    3 stride-2 convolutions give 2³ = 8× spatial downsampling.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 128 → 64
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            ResBlock(64),
            # 64 → 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            ResBlock(128),
            # 32 → 16
            nn.Conv2d(128, latent_dim, 4, stride=2, padding=1),
            ResBlock(latent_dim),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            # 1×1 conv to mix channels before VQ
            nn.Conv2d(latent_dim, latent_dim, 1),
        )

    def forward(self, x):
        return self.net(x)   # (B, latent_dim, 16, 16)


class Decoder(nn.Module):
    """
    16×16 → 128×128 RGB reconstruction.
    Mirror of Encoder using transposed convolutions.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1),
            ResBlock(latent_dim),
            # 16 → 32
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=2, padding=1),
            ResBlock(128),
            # 32 → 64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            ResBlock(64),
            # 64 → 128
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, z_q):
        return self.net(z_q)   # (B, 3, 128, 128)


# ---------------------------------------------------------------------------
# Vector Quantizer with EMA codebook updates
# ---------------------------------------------------------------------------

class VectorQuantizerEMA(nn.Module):
    """
    VQ bottleneck with Exponential Moving Average (EMA) codebook updates.

    EMA is more stable than the straight-through estimator alone because the
    codebook update is decoupled from the main gradient signal.

    Args:
        codebook_size:   number of discrete codes (1024 is enough for robot frames)
        codebook_dim:    dimension of each code vector (matches encoder output channels)
        commitment_beta: weight on the commitment loss term (default 0.25)
        ema_decay:       EMA decay for codebook updates (0.99 = slow, stable)
        ema_epsilon:     small constant for numerical stability in EMA division
    """
    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim  = codebook_dim
        self.commitment_beta = commitment_beta
        self.ema_decay  = ema_decay
        self.ema_epsilon = ema_epsilon

        # Codebook embedding table — NOT a parameter (updated via EMA, not gradient)
        embed = torch.randn(codebook_size, codebook_dim)
        self.register_buffer("embedding",  embed)

        # EMA accumulators: N_i = count of assignments, m_i = sum of z_e assigned to i
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed_avg",    embed.clone())

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: (B, C, H, W) continuous encoder output
        Returns:
            z_q:             (B, C, H, W) quantized (straight-through in backward)
            indices:         (B, H*W)     token IDs for each spatial position
            commitment_loss: scalar
        """
        B, C, H, W = z_e.shape

        # Flatten spatial dims: (B, C, H, W) → (B*H*W, C)
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)

        # L2 distance to each codebook entry: ||z - e||^2
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)           # (N, 1)
            - 2 * z_flat @ self.embedding.T               # (N, K)
            + self.embedding.pow(2).sum(1)                # (K,)
        )  # (N, K)

        # Nearest codebook entry
        indices_flat = dist.argmin(dim=1)                 # (N,)
        indices = indices_flat.reshape(B, H * W)          # (B, H*W)

        # Quantized vectors: (N, C) → (B, H, W, C) → (B, C, H, W)
        z_q_flat = self.embedding[indices_flat]           # (N, C)
        z_q = z_q_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # ---- EMA codebook update (training only) ----
        if self.training:
            with torch.no_grad():
                # One-hot assignment matrix: (N, K)
                one_hot = F.one_hot(indices_flat, self.codebook_size).float()

                # Update cluster size EMA: N_i ← γ*N_i + (1-γ)*n_i
                cluster_size = one_hot.sum(0)  # (K,)
                self.ema_cluster_size = (
                    self.ema_decay * self.ema_cluster_size
                    + (1 - self.ema_decay) * cluster_size
                )

                # Update embedding average EMA: m_i ← γ*m_i + (1-γ)*Σ_{z→i} z
                embed_sum = one_hot.T @ z_flat  # (K, C)
                self.ema_embed_avg = (
                    self.ema_decay * self.ema_embed_avg
                    + (1 - self.ema_decay) * embed_sum
                )

                # Laplace smoothing to avoid division by near-zero
                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.ema_epsilon)
                    / (n + self.codebook_size * self.ema_epsilon)
                    * n
                )
                self.embedding = self.ema_embed_avg / smoothed.unsqueeze(1)

        # Straight-through estimator: copy gradient from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss: encoder commits to nearest codebook entry
        commitment_loss = self.commitment_beta * F.mse_loss(z_e, z_q.detach())

        return z_q_st, indices, commitment_loss

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: (B, H*W) int64 token IDs
        Returns: (B, C, H, W) quantized feature maps
        """
        B, HW = indices.shape
        H = W = int(HW ** 0.5)
        z_q = self.embedding[indices.reshape(-1)]       # (B*HW, C)
        return z_q.reshape(B, H, W, self.codebook_dim).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Full VQ-VAE model
# ---------------------------------------------------------------------------

class RobotFrameVQVAE(nn.Module):
    """
    Full VQ-VAE for robot camera frames.

    Input/output: (B, 3, 128, 128) float32 images in [0, 1].
    Tokens:       (B, 256) int64 IDs from a codebook of size 1024.
                  256 = 16×16 spatial grid (one token per 8×8 pixel patch).

    Usage:
        # Training
        model = RobotFrameVQVAE()
        recon, loss, indices = model(frames)
        loss.backward()

        # Inference (encode only)
        indices = model.encode(frames)   # (B, 256) int64

        # Reconstruction from tokens
        recon = model.decode(indices)    # (B, 3, 128, 128)
    """
    INPUT_SIZE: int = 128
    N_TOKENS:   int = 256   # 16 × 16
    SPATIAL:    int = 16    # sqrt(N_TOKENS)

    def __init__(
        self,
        codebook_size: int = 1024,
        latent_dim:    int = 256,
        commitment_beta: float = 0.25,
    ):
        super().__init__()
        self.encoder  = Encoder(latent_dim)
        self.vq       = VectorQuantizerEMA(codebook_size, latent_dim, commitment_beta)
        self.decoder  = Decoder(latent_dim)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Resize any resolution image to INPUT_SIZE×INPUT_SIZE, ensure [0,1]."""
        x = x.float()
        if x.shape[-1] != self.INPUT_SIZE or x.shape[-2] != self.INPUT_SIZE:
            x = F.interpolate(x, (self.INPUT_SIZE, self.INPUT_SIZE),
                              mode="bilinear", align_corners=False)
        return x.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass for training.
        Returns: (reconstruction, total_loss, indices)
        """
        x = self.preprocess(x)
        z_e = self.encoder(x)
        z_q, indices, commitment_loss = self.vq(z_e)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        total_loss = recon_loss + commitment_loss
        return recon, total_loss, indices

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to discrete token IDs.
        Args:  x: (B, 3, H, W) float32 in [0, 1]
        Returns: (B, 256) int64 token IDs
        """
        x = self.preprocess(x)
        z_e = self.encoder(x)
        _, indices, _ = self.vq(z_e)
        return indices  # (B, 16*16)

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images from token IDs.
        Args:  indices: (B, 256) int64
        Returns: (B, 3, 128, 128) float32 in [0, 1]
        """
        z_q = self.vq.decode_indices(indices)
        return self.decoder(z_q)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":    self.state_dict(),
            "codebook_size": self.vq.codebook_size,
            "latent_dim":    self.vq.codebook_dim,
        }, path)
        print(f"[VQ-VAE] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "RobotFrameVQVAE":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            codebook_size=ckpt["codebook_size"],
            latent_dim=ckpt["latent_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[VQ-VAE] Loaded ← {path}  (codebook={ckpt['codebook_size']}, dim={ckpt['latent_dim']})")
        return model.to(device)
