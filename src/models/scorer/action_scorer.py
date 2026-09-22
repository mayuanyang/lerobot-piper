"""Q(o, a) -> normalized steps-to-success. A ranker for best-of-N, not a policy.

The measured fact this exists to exploit: this policy wins by RE-ROLLING NOISE
rather than by accuracy -- removing the per-chunk re-draw costs 25 points, and
roughly half the goal failures are episodes another noise stream won. Every
intervention tried so far has aimed at making the model know more. This one
leaves the policy frozen and only tries to pick a better ticket.

WHY THIS IS NOT AWR. The 2026-09-20 AWR run cost 28.5 points of goal because
the loss is `w * ||v - u||^2` with no negative weights, so a failed episode at
weight 0.26 still said "this way, but gently" and taught the policy to hover.
A scorer never imitates. It predicts a number, failures are ordinary negative
evidence, and the policy's weights are not touched at all.

WHY THE TARGET IS NOT `success`. Labelling every chunk of a successful episode
"good" is the same credit-assignment error that made AWR fail: the mediocre
actions inside a success get the same label as the decisive ones. Steps-to-
success is dense, per-frame, and well defined -- it IS the value function, and
a state inside a successful episode genuinely did lead to success.

Lower score = closer to succeeding = the ticket to play.
"""

import math

import torch
import torch.nn as nn

from ..transformer_flow_matching.robot_visual_encoder import RobotVisualEncoder


def _sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(d, dtype=torch.float32)[None, :]
    ang = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(ang[:, 0::2])
    pe[:, 1::2] = torch.cos(ang[:, 1::2])
    return pe


class ActionScorer(nn.Module):
    """Frozen-policy companion: scores a candidate action chunk in context.

    Standalone rather than a head on the policy's features. Caching the
    policy's tokens for the corpus would be 51,109 frames x 400 tokens x 960
    dims = 39 GB, and running the policy in the loop makes every training step
    pay a 639M-parameter forward for nothing. A ResNet-18 trunk on two cameras
    trains on the same corpus in minutes.

    The action chunk enters as `horizon` tokens rather than a pooled vector so
    that WHEN something happens inside the chunk is visible to the scorer --
    the failures this is aimed at are contact-phase, i.e. a question of timing.
    """

    def __init__(self, horizon: int = 50, action_dim: int = 7, state_dim: int = 8,
                 n_cams: int = 2, input_size: int = 224, vis_tokens: int = 36,
                 d_model: int = 256, n_layers: int = 3, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.horizon, self.action_dim, self.state_dim = horizon, action_dim, state_dim
        self.n_cams, self.vis_tokens, self.d_model = n_cams, vis_tokens, d_model

        # One trunk shared across cameras + a learned per-camera offset: the two
        # views are the same kind of image and the corpus is small.
        self.vis = RobotVisualEncoder(input_size=input_size, out_tokens=vis_tokens,
                                      out_dim=d_model, pool="avg")
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, 1, d_model))
        self.register_buffer("vis_pos", _sinusoid(vis_tokens, d_model)[None], persistent=False)

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.act_in = nn.Linear(action_dim, d_model)
        self.register_buffer("act_pos", _sinusoid(horizon, d_model)[None], persistent=False)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_emb = nn.Parameter(torch.zeros(3, 1, d_model))   # vision / state / action

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.cam_emb, std=0.02)
        nn.init.trunc_normal_(self.type_emb, std=0.02)

    def encode_obs(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """(B, n_cams, 3, H, W) in [0,1] and (B, state_dim) -> (B, L_obs, d).

        Split from `forward` because best-of-N scores K candidate chunks against
        ONE observation: the trunk runs once and the result is reused K times,
        which is what makes the scorer cheap at inference.
        """
        B, C = images.shape[0], images.shape[1]
        tok = self.vis(images.flatten(0, 1))                      # (B*C, V, d)
        tok = tok.view(B, C, self.vis_tokens, self.d_model)
        tok = tok + self.cam_emb[None, :C] + self.vis_pos[None]
        tok = tok.flatten(1, 2) + self.type_emb[0]                # (B, C*V, d)
        st = self.state_mlp(state)[:, None] + self.type_emb[1]    # (B, 1, d)
        return torch.cat([tok, st], dim=1)

    def score(self, obs_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """(B, L_obs, d) and (B, horizon, action_dim) -> (B,) in [0, 1]."""
        a = self.act_in(actions) + self.act_pos + self.type_emb[2]
        cls = self.cls.expand(actions.shape[0], -1, -1)
        h = self.enc(torch.cat([cls, obs_tokens, a], dim=1))
        return torch.sigmoid(self.head(h[:, 0]).squeeze(-1))

    def forward(self, images: torch.Tensor, state: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        return self.score(self.encode_obs(images, state), actions)

    @torch.no_grad()
    def score_candidates(self, images: torch.Tensor, state: torch.Tensor,
                         candidates: torch.Tensor) -> torch.Tensor:
        """(B, n_cams, 3, H, W), (B, state_dim), (B, K, horizon, action_dim) -> (B, K).

        The selection call. argmin over K is the ticket to play.
        """
        B, K = candidates.shape[0], candidates.shape[1]
        obs = self.encode_obs(images, state)
        obs = obs.repeat_interleave(K, dim=0)
        s = self.score(obs, candidates.flatten(0, 1))
        return s.view(B, K)
