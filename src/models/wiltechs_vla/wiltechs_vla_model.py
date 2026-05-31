"""
WiltechsVLATransformer — Qwen3-VL-based interleaved flow matching policy.

Architecture mirrors `interleaved_flow_matching_model.py` (SmolVLA-style joint
attention every layer) with the backbone swapped to Qwen/Qwen3-VL-4B-Instruct-FP8.

Key adaptations for Qwen3-VL:
  - Vision encoder path uses `Qwen3VLForConditionalGeneration.get_image_features`
    instead of raw SigLIP + connector tensors.
  - Text model is `language_model` (Qwen3VLTextModel) instead of `text_model`.
  - Image preprocessing goes through the Qwen3-VL image processor on-the-fly
    (PIL conversion). This is slower than the SmolVLM2 raw-tensor path.
  - hidden_size = 2560, num_heads = 32, num_kv_heads = 8, intermediate = 9728.

Mask semantics (unchanged):
                 VLM    Latent  Action
    VLM          full   ✓       ✓ (gated by `vlm_attends_to_expert`)
    Latent       ✓      ✓       ✗ (never see noisy actions)
    Action       ✓      ✓       causal (only earlier actions)
"""

import math
from contextlib import nullcontext
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .wiltechs_vla_config import WiltechsVLAConfig
from ..interleaved_flow_matching.expert_layer import ExpertProjections, RMSNorm
from ..transformer_flow_matching.robot_visual_encoder import RobotVisualEncoder


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension must be even, got {dimension}")
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period) * 2.0 * math.pi
    sin_input = scaling[None, :] * time[:, None].float()
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# ---------------------------------------------------------------------------
# M-RoPE helpers
#
# Qwen3-VL uses Multimodal RoPE with interleaved sections (config.rope_scaling
# = {mrope_interleaved: True, mrope_section: [24, 20, 20]}). The frozen
# q/k/v projections were trained against this positional scheme, so giving
# them plain 1D RoPE would corrupt their representations. Instead of
# re-implementing the interleaved M-RoPE math, we reuse the loaded VLM's own
# `language_model.rotary_emb` module — it already encapsulates the correct
# `(t, h, w) → cos/sin` mapping. We only need to construct the right
# position_ids of shape (3, B, L_total) for our joint sequence.
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding. `cos`/`sin` shape: (B, L, head_dim) — already
    in interleaved-M-RoPE layout when produced by Qwen3VLTextRotaryEmbedding.
    `q`/`k` shape: (B, num_heads, L, head_dim)."""
    cos = cos.unsqueeze(1)  # broadcast over heads
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _build_mrope_position_ids(
    image_grid_thw_list: list[torch.Tensor],
    L_non_vis: int,
    B: int,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct (3, B, L_total) position_ids for [vision … | non_vision].

    Mirrors HF Qwen3VL's get_vision_position_ids / get_rope_index:
      - Each camera's vision tokens get (t_idx, h_idx, w_idx) at the LLM grid
        resolution (post spatial merge by `spatial_merge_size`).
      - The temporal channel for camera n starts at the post-merge max
        position of camera n-1 + 1 (to keep all three channels collision-
        free across images).
      - After the last vision token, non-vision tokens (lang, state, robot,
        latent, action) receive a monotonic position with all three channels
        equal — this degenerates M-RoPE back to plain 1D RoPE on those
        tokens, matching how text tokens are positionally encoded by
        Qwen3-VL during pretraining.

    Assumes uniform batch (every batch element has the same image grid),
    which holds because DatasetAdapter resizes every camera to a fixed
    canonical resolution before collation.
    """
    pos_pieces: list[torch.Tensor] = []
    cur_start = 0

    for grid_thw in image_grid_thw_list:
        t = int(grid_thw[0].item())
        h = int(grid_thw[1].item()) // spatial_merge_size
        w = int(grid_thw[2].item()) // spatial_merge_size

        # Row-major (t, h, w) enumeration matching Qwen3-VL's vision encoder
        # output order. `position_height/width/temporal` follow the exact
        # repeat pattern from HF's get_vision_position_ids.
        pos_t = torch.arange(t, device=device).repeat_interleave(h * w) + cur_start
        pos_h = (
            torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            + cur_start
        )
        pos_w = torch.arange(w, device=device).repeat(t * h) + cur_start
        pos_pieces.append(torch.stack([pos_t, pos_h, pos_w], dim=0))  # (3, t*h*w)
        # Advance temporal channel by max(t, h, w) so subsequent cameras
        # don't share any of the three position channels.
        cur_start += max(t, h, w)

    if pos_pieces:
        vis_pos = torch.cat(pos_pieces, dim=1)              # (3, L_vis)
        next_pos = int(vis_pos.max().item()) + 1
    else:
        vis_pos = torch.zeros(3, 0, dtype=torch.long, device=device)
        next_pos = 0

    # Non-vision: monotonic, replicated across the three channels.
    non_vis = (
        torch.arange(next_pos, next_pos + L_non_vis, device=device)
        .unsqueeze(0)
        .expand(3, -1)
    )

    full = torch.cat([vis_pos, non_vis], dim=1)             # (3, L_total)
    return full.unsqueeze(1).expand(3, B, -1).contiguous()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class WiltechsVLATransformer(nn.Module):
    """Interleaved flow matching policy with Qwen3-VL backbone."""

    VLM_MODEL_ID: str = "Qwen/Qwen3-VL-4B-Instruct-FP8"

    def __init__(self, config: WiltechsVLAConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # 1. Load Qwen3-VL (frozen)
        # ------------------------------------------------------------------
        print(f"Loading {self.VLM_MODEL_ID} ...")
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.VLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.VLM_MODEL_ID)
        self.vlm_model = vlm.model
        self.visual = self.vlm_model.visual
        self.language_model = self.vlm_model.language_model

        # Truncate to first num_vlm_layers
        total_layers = len(self.language_model.layers)
        self.language_model.layers = self.language_model.layers[: config.num_vlm_layers]
        print(f"Using first {len(self.language_model.layers)}/{total_layers} text layers")

        # Read VLM attention shape from layer 0 (assume uniform across layers).
        first_attn = self.language_model.layers[0].self_attn
        text_cfg = self.language_model.config
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_heads = int(text_cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", self.num_heads))
        # IMPORTANT: read head_dim explicitly. Qwen3-VL-4B has head_dim=128
        # while hidden_size // num_heads = 2560/32 = 80 — they DON'T match.
        # Computing head_dim from hidden/num_heads silently produces mismatched
        # expert projections and the joint-attention cat fails. The expert
        # must mirror the VLM's actual q/k/v output widths.
        self.head_dim = int(
            getattr(text_cfg, "head_dim", None) or (self.hidden_size // self.num_heads)
        )
        self.intermediate_size = int(text_cfg.intermediate_size)
        self.rms_norm_eps = float(getattr(text_cfg, "rms_norm_eps", 1e-5))
        self.rope_theta = float(getattr(text_cfg, "rope_theta", 10000.0))
        print(f"VLM attn shape: hidden={self.hidden_size} heads={self.num_heads} "
              f"kv_heads={self.num_kv_heads} head_dim={self.head_dim} "
              f"intermediate={self.intermediate_size}")

        # The vision tower's spatial_merge_size determines the LLM-grid
        # resolution of vision tokens — needed when we build M-RoPE
        # position_ids. Default 2 matches Qwen3-VL configs we've inspected.
        vis_cfg = getattr(vlm.config, "vision_config", None)
        self.spatial_merge_size = int(getattr(vis_cfg, "spatial_merge_size", 2))

        # Sanity: M-RoPE requires the VLM's rotary embedding module. Reuse
        # it directly instead of re-implementing the interleaved math.
        if not hasattr(self.language_model, "rotary_emb"):
            raise RuntimeError(
                "language_model.rotary_emb not found — M-RoPE setup expects "
                "Qwen3VLTextRotaryEmbedding to live on language_model."
            )

        # Sanity: joint attention needs config's d_model to match VLM hidden.
        if config.d_model != self.hidden_size:
            print(f"[wiltechs_vla] forcing d_model {config.d_model} → {self.hidden_size} to match VLM")
            config.d_model = self.hidden_size

        # Freeze VLM
        for component in [self.visual, self.language_model]:
            for p in component.parameters():
                p.requires_grad = False
            component.eval()
        del vlm

        # ------------------------------------------------------------------
        # 2. Trainable expert layers (parallel to each VLM layer)
        # ------------------------------------------------------------------
        self.expert_layers = nn.ModuleList([
            ExpertProjections(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
                rms_norm_eps=self.rms_norm_eps,
                dropout=config.dropout,
            )
            for _ in range(len(self.language_model.layers))
        ])

        # ------------------------------------------------------------------
        # 3. Robot CNN (optional) and state encoder
        # ------------------------------------------------------------------
        if config.use_robot_cnn:
            self.robot_visual_encoder = RobotVisualEncoder(
                input_size=config.robot_encoder_input_size,
                out_tokens=config.robot_encoder_tokens,
                out_dim=self.hidden_size,
            )
        else:
            self.robot_visual_encoder = None
            print("[wiltechs_vla] use_robot_cnn=False — RobotVisualEncoder disabled (ablation mode)")
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, self.hidden_size),
            RMSNorm(self.hidden_size, eps=self.rms_norm_eps),
        )

        # ------------------------------------------------------------------
        # 4. Action expert input / output
        # ------------------------------------------------------------------
        self.action_in_proj = nn.Linear(config.action_dim, self.hidden_size)
        self.action_out_proj = nn.Linear(self.hidden_size, config.action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

        self.action_time_mlp_in = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size)

        self.action_pos_emb = nn.Parameter(torch.zeros(1, config.horizon, self.hidden_size))
        nn.init.normal_(self.action_pos_emb, std=0.02)

        # ------------------------------------------------------------------
        # 5. Latent "thought" tokens — task-conditional
        # ------------------------------------------------------------------
        self.num_latent_tokens = config.num_latent_tokens
        if self.num_latent_tokens > 0:
            hidden_mid = self.hidden_size * 2
            self.latent_generator = nn.Sequential(
                nn.Linear(self.hidden_size, hidden_mid),
                nn.SiLU(),
                nn.Linear(hidden_mid, self.num_latent_tokens * self.hidden_size),
            )
            nn.init.zeros_(self.latent_generator[-1].weight)
            nn.init.zeros_(self.latent_generator[-1].bias)

        # ------------------------------------------------------------------
        # 6. Final norm before action readout
        # ------------------------------------------------------------------
        self.final_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # ------------------------------------------------------------------
        # 7. Language adaptor — trainable residual projection
        # ------------------------------------------------------------------
        self.lang_adaptor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            RMSNorm(self.hidden_size, eps=self.rms_norm_eps),
        )
        nn.init.zeros_(self.lang_adaptor[0].weight)
        nn.init.zeros_(self.lang_adaptor[0].bias)

        num_layers = len(self.language_model.layers)
        self.lang_attn_bias = nn.Parameter(torch.zeros(num_layers))

        self._lang_max_len = 48

    # =====================================================================
    # Keep frozen components in eval mode
    # =====================================================================

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        self.language_model.eval()
        return self

    # =====================================================================
    # Encoding helpers
    # =====================================================================

    def _encode_images(
        self, batch: dict, B: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Vision tokens + per-camera grid_thw, frozen.

        Returns:
          vis_tokens: (B, sum_of_vision_tokens, hidden_size) bfloat16
          grid_thw_list: list of (3,) tensors — one per camera that produced
            real vision tokens, used downstream to build M-RoPE position_ids.

        Qwen3-VL uses its own image processor which returns patchified
        `pixel_values` plus `image_grid_thw`. We convert the dataset's
        [0,1] float tensors to PIL Images and feed them through the
        processor on-the-fly.

        TODO: Pre-compute Qwen3-VL visual features in the dataset pipeline
        to avoid the PIL conversion overhead at training time.
        """
        device = batch["observation.state"].device
        all_vis: list[torch.Tensor] = []
        grid_thw_list: list[torch.Tensor] = []
        for cam_key in self.config.cameras_for_vision_state_concat:
            if cam_key not in batch:
                continue
            imgs = batch[cam_key]
            img = imgs[:, -1] if imgs.dim() == 5 else imgs  # (B, C, H, W)

            # Dataset tensors are [0,1] float; Qwen3-VL processor expects
            # PIL/numpy in [0,255] range. Convert to PIL for reliability.
            img_np = (img.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_images = [Image.fromarray(img_np[i]) for i in range(B)]

            with torch.no_grad():
                processor_out = self.processor.image_processor(
                    images=pil_images, return_tensors="pt"
                )
                pixel_values = processor_out["pixel_values"].to(device=device)
                image_grid_thw = processor_out["image_grid_thw"].to(device=device)

                vis_features = self.vlm_model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                # HF Qwen3-VL `get_image_features` returns the merged vision
                # tokens directly as a tensor (or a `BaseModelOutput`-like
                # object on older versions). Accept both shapes.
                vis_tokens = getattr(vis_features, "last_hidden_state", vis_features)

            all_vis.append(vis_tokens)
            # Take grid_thw of the first image — for a uniform batch every
            # element has the same processed grid, which is what M-RoPE
            # construction below assumes.
            grid_thw_list.append(image_grid_thw[0].detach())

        if not all_vis:
            empty = torch.zeros(B, 0, self.hidden_size, device=device, dtype=torch.bfloat16)
            return empty, []
        return torch.cat(all_vis, dim=1), grid_thw_list

    def _encode_language(self, batch: dict, device: torch.device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Frozen embed_tokens + attention mask for the task description.

        Returns (lang_tokens, lang_mask) or None.
          lang_tokens: (B, L_max, hidden)
          lang_mask:   (B, L_max) — bool, True where a real (non-pad) token sits
        """
        descs = batch.get("task_description")
        if not descs:
            descs = batch.get("task")
        if not descs or not any(descs):
            return None
        inputs = self.processor.tokenizer(
            descs, return_tensors="pt", padding=True, truncation=True,
            max_length=self._lang_max_len, add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(device)
        lang_mask = inputs["attention_mask"].bool().to(device)  # (B, L)
        lang_tokens = self.language_model.get_input_embeddings()(input_ids)
        return lang_tokens, lang_mask

    # =====================================================================
    # Build attention mask for the joint sequence
    # =====================================================================

    def _build_joint_mask(
        self,
        L_vlm: int,
        R: int,
        K: int,
        H: int,
        device: torch.device,
        dtype: torch.dtype,
        lang_start: int = 0,
        lang_len: int = 0,
        lang_mask: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        L_total = L_vlm + R + K + H
        a_start = L_vlm + R + K
        l_start = L_vlm + R
        r_start = L_vlm

        mask = torch.zeros(L_total, L_total, device=device, dtype=dtype)

        if R > 0:
            mask[r_start:r_start + R, a_start:] = float("-inf")

        if K > 0:
            mask[l_start:l_start + K, a_start:] = float("-inf")

        if not self.config.vlm_attends_to_expert:
            mask[:L_vlm, L_vlm:] = float("-inf")

        if H > 1:
            causal = torch.triu(
                torch.full((H, H), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            mask[a_start:, a_start:] = causal

        if lang_len > 0 and lang_mask is not None:
            B = lang_mask.shape[0]
            mask = mask.unsqueeze(0).expand(B, -1, -1).clone()

            pad_cols = ~lang_mask.to(device)
            for b in range(B):
                pad_idx = pad_cols[b].nonzero(as_tuple=True)[0]
                if pad_idx.numel() == 0:
                    continue
                pad_pos = lang_start + pad_idx
                mask[b, :, pad_pos] = float("-inf")
                mask[b, pad_pos, :] = float("-inf")

            bias = F.softplus(self.lang_attn_bias[layer_idx])
            bias_per_col = bias * lang_mask.to(device, dtype)
            mask[:, L_vlm:, lang_start:lang_start + lang_len] += bias_per_col.unsqueeze(1)

            mask = mask.unsqueeze(1)

        elif lang_len > 0:
            bias = F.softplus(self.lang_attn_bias[layer_idx])
            mask[L_vlm:, lang_start:lang_start + lang_len] += bias

        return mask

    # =====================================================================
    # Joint attention layer — the core of interleaving
    # =====================================================================

    def _joint_layer(
        self,
        vlm_seq: torch.Tensor,
        exp_seq: torch.Tensor,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vlm_layer = self.language_model.layers[layer_idx]
        exp_layer = self.expert_layers[layer_idx]

        B, L_vlm, _ = vlm_seq.shape
        L_exp = exp_seq.size(1)
        L_total = L_vlm + L_exp
        H = self.num_heads
        Hk = self.num_kv_heads
        D = self.head_dim

        # Pre-norm (separate weights)
        vlm_norm = vlm_layer.input_layernorm(vlm_seq)
        exp_norm = exp_layer.input_layernorm(exp_seq)

        # Q/K/V from two parallel projection sets
        Q_vlm = vlm_layer.self_attn.q_proj(vlm_norm)
        K_vlm = vlm_layer.self_attn.k_proj(vlm_norm)
        V_vlm = vlm_layer.self_attn.v_proj(vlm_norm)

        Q_exp = exp_layer.q_proj(exp_norm)
        K_exp = exp_layer.k_proj(exp_norm)
        V_exp = exp_layer.v_proj(exp_norm)

        Q = torch.cat([Q_vlm, Q_exp], dim=1).view(B, L_total, H, D).transpose(1, 2)
        K = torch.cat([K_vlm, K_exp], dim=1).view(B, L_total, Hk, D).transpose(1, 2)
        V = torch.cat([V_vlm, V_exp], dim=1).view(B, L_total, Hk, D).transpose(1, 2)

        Q, K = _apply_rope(Q, K, cos, sin)

        if Hk != H:
            repeat = H // Hk
            K = K.repeat_interleave(repeat, dim=1)
            V = V.repeat_interleave(repeat, dim=1)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_total, H * D)

        attn_vlm = vlm_layer.self_attn.o_proj(attn_out[:, :L_vlm])
        attn_exp = exp_layer.o_proj(attn_out[:, L_vlm:])
        attn_exp = exp_layer.attn_dropout(attn_exp)

        vlm_seq = vlm_seq + attn_vlm
        exp_seq = exp_seq + attn_exp

        # FFN (pre-norm, parallel)
        vlm_ffn_in = vlm_layer.post_attention_layernorm(vlm_seq)
        exp_ffn_in = exp_layer.post_attention_layernorm(exp_seq)

        vlm_ffn_out = vlm_layer.mlp(vlm_ffn_in)
        exp_ffn_out = exp_layer.mlp(exp_ffn_in)
        exp_ffn_out = exp_layer.mlp_dropout(exp_ffn_out)

        vlm_seq = vlm_seq + vlm_ffn_out
        exp_seq = exp_seq + exp_ffn_out

        return vlm_seq, exp_seq

    # =====================================================================
    # Build expert sequence for a given timestep
    # =====================================================================

    def _generate_latents(
        self,
        lang_tokens: Optional[torch.Tensor],
        lang_mask: Optional[torch.Tensor],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        K = self.num_latent_tokens
        if K == 0:
            return None
        if lang_tokens is None or lang_mask is None:
            pooled = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        else:
            mask_f = lang_mask.float().unsqueeze(-1).to(lang_tokens.dtype)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = (lang_tokens * mask_f).sum(dim=1) / denom
        flat = self.latent_generator(pooled.float())
        return flat.view(B, K, self.hidden_size).to(dtype)

    def _build_expert_seq(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        robot_tokens: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H = noisy_actions.shape[1]

        action_emb = self.action_in_proj(noisy_actions)
        action_emb = action_emb + self.action_pos_emb[:, :H]

        t_emb = create_sinusoidal_pos_embedding(timesteps, self.hidden_size).to(noisy_actions.dtype)
        t_emb = t_emb.unsqueeze(1).expand(-1, H, -1)
        fused = torch.cat([action_emb, t_emb], dim=-1)
        fused = F.silu(self.action_time_mlp_in(fused))
        fused = self.action_time_mlp_out(fused)
        alpha = (1.0 - timesteps)[:, None, None]
        action_tgt = fused + alpha * action_emb

        parts = []
        if robot_tokens is not None:
            parts.append(robot_tokens.to(action_tgt.dtype))
        if self.num_latent_tokens > 0 and latents is not None:
            parts.append(latents.to(action_tgt.dtype))
        parts.append(action_tgt)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    # =====================================================================
    # Compute robot CNN tokens once
    # =====================================================================

    def _compute_robot_tokens(self, batch: dict) -> Optional[torch.Tensor]:
        if self.robot_visual_encoder is None:
            return None
        robot_tokens_list = []
        for cam_key in self.config.cameras_for_vision_state_concat:
            if cam_key not in batch:
                continue
            img = batch[cam_key]
            if img.dim() == 5:
                img = img[:, -1]
            robot_tokens_list.append(self.robot_visual_encoder(img.float()))
        if not robot_tokens_list:
            return None
        robot_tokens = torch.cat(robot_tokens_list, dim=1)

        vision_dropout_prob = float(getattr(self.config, "vision_dropout_prob", 0.0)) if self.training else 0.0
        if self.training and vision_dropout_prob > 0.0:
            B, R, _ = robot_tokens.shape
            keep = torch.rand(B, R, device=robot_tokens.device) > vision_dropout_prob
            robot_tokens = robot_tokens * keep.unsqueeze(-1).to(robot_tokens.dtype)

        return robot_tokens

    # =====================================================================
    # Build VLM-side sequence: frozen vision + language + (trainable) state
    # =====================================================================

    def _build_vlm_seq(
        self, batch: dict
    ) -> tuple[torch.Tensor, int, Optional[torch.Tensor], list[torch.Tensor]]:
        """Returns (vlm_seq, L_vis, lang_mask, grid_thw_list).

        `grid_thw_list` is passed to the M-RoPE position-id builder so
        vision tokens get (t, h, w) coordinates rather than 1D arange.
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        lang_result = self._encode_language(batch, device)
        lang_mask: Optional[torch.Tensor] = None

        with torch.no_grad():
            vis_tokens, grid_thw_list = self._encode_images(batch, B)

        L_vis = vis_tokens.shape[1]

        vision_dropout_prob = float(getattr(self.config, "vision_dropout_prob", 0.0)) if self.training else 0.0
        if self.training and L_vis > 0 and vision_dropout_prob > 0.0:
            keep = torch.rand(B, L_vis, device=vis_tokens.device) > vision_dropout_prob
            vis_tokens = vis_tokens * keep.unsqueeze(-1).to(vis_tokens.dtype)

        if lang_result is not None:
            lang_tokens, lang_mask = lang_result
            L_lang = lang_tokens.shape[1]

            lang_adapted = lang_tokens.to(vis_tokens.dtype) + self.lang_adaptor(lang_tokens.to(vis_tokens.dtype))
            lang_tokens = torch.where(
                lang_mask.unsqueeze(-1),
                lang_adapted,
                torch.zeros_like(lang_adapted),
            )
        else:
            lang_tokens = None
            L_lang = 0

        state = batch["observation.state"].float()
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state = state.nan_to_num(0.0).clamp(-10.0, 10.0)
        state_tokens = self.state_encoder(state).to(vis_tokens.dtype)

        parts = [vis_tokens]
        if lang_tokens is not None:
            parts.append(lang_tokens)
        parts.append(state_tokens)
        return torch.cat(parts, dim=1), L_vis, lang_mask, grid_thw_list

    # =====================================================================
    # Velocity field — runs the joint stack and reads off action positions
    # =====================================================================

    def velocity_field(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        vlm_seq: torch.Tensor,
        robot_tokens: Optional[torch.Tensor] = None,
        lang_start: int = 0,
        lang_len: int = 0,
        lang_mask: Optional[torch.Tensor] = None,
        grid_thw_list: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, H, _ = noisy_actions.shape
        K = self.num_latent_tokens
        R = robot_tokens.size(1) if robot_tokens is not None else 0
        L_vlm = vlm_seq.size(1)

        lang_part = (
            vlm_seq[:, lang_start : lang_start + lang_len]
            if (lang_len > 0 and lang_mask is not None) else None
        )
        latents = self._generate_latents(
            lang_part, lang_mask, B, vlm_seq.device, vlm_seq.dtype,
        )

        exp_seq = self._build_expert_seq(
            noisy_actions, timesteps, robot_tokens, latents,
        ).to(vlm_seq.dtype)

        # ── M-RoPE: build (3, B, L_total) position_ids, then defer cos/sin
        # synthesis to Qwen3-VL's own rotary module. L_non_vis covers the
        # lang + state + (robot + latent + action) tokens — every position
        # past L_vis. The lang_start the caller passed is the boundary
        # between vis and (lang+state) inside vlm_seq, so L_vis == lang_start.
        L_total = L_vlm + R + K + H
        L_vis_local = lang_start
        L_non_vis = (L_total - L_vis_local)
        position_ids = _build_mrope_position_ids(
            grid_thw_list or [],
            L_non_vis=L_non_vis,
            B=B,
            spatial_merge_size=self.spatial_merge_size,
            device=vlm_seq.device,
        )
        # `language_model.rotary_emb` is frozen but callable. It returns
        # cos/sin in `(B, L_total, head_dim)` shape with the interleaved
        # M-RoPE layout already applied — _apply_rope can consume directly.
        cos, sin = self.language_model.rotary_emb(vlm_seq, position_ids)

        for i in range(len(self.expert_layers)):
            attn_mask_i = self._build_joint_mask(
                L_vlm, R, K, H, vlm_seq.device, vlm_seq.dtype,
                lang_start=lang_start, lang_len=lang_len,
                lang_mask=lang_mask, layer_idx=i,
            )
            vlm_seq, exp_seq = self._joint_layer(
                vlm_seq, exp_seq, layer_idx=i,
                cos=cos, sin=sin, attn_mask=attn_mask_i,
            )

        action_out = self.final_norm(exp_seq[:, R + K:])
        return self.action_out_proj(action_out)

    # =====================================================================
    # Flow matching: training loss + inference sampling
    # =====================================================================

    def sample_noise(self, shape, device):
        rho = self.config.noise_temporal_correlation
        noise = torch.randn(shape, device=device)
        if rho == 0.0 or shape[1] == 1:
            return noise
        scale = math.sqrt(1.0 - rho * rho)
        for t in range(1, shape[1]):
            noise[:, t] = rho * noise[:, t - 1] + scale * noise[:, t]
        return noise

    def sample_time(self, B, device):
        t = torch.rand(B, device=device)
        return t * 0.998 + 0.001

    def compute_loss(self, batch: dict) -> torch.Tensor:
        actions = batch["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0)
        B = actions.shape[0]
        device = actions.device

        robot_tokens = self._compute_robot_tokens(batch)

        vlm_seq, L_vis, lang_mask, grid_thw_list = self._build_vlm_seq(batch)
        vlm_seq = vlm_seq.float()
        L_lang = lang_mask.shape[1] if lang_mask is not None else 0

        noise = self.sample_noise(actions.shape, actions.device)
        t = self.sample_time(B, actions.device)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions
        u_t = noise - actions

        v_t = self.velocity_field(x_t, t, vlm_seq.to(x_t.dtype), robot_tokens=robot_tokens,
                                  lang_start=L_vis, lang_len=L_lang, lang_mask=lang_mask,
                                  grid_thw_list=grid_thw_list)

        loss = F.mse_loss(v_t, u_t, reduction="none")
        if self.config.action_dim_weights:
            dim_w = torch.tensor(self.config.action_dim_weights, device=loss.device, dtype=loss.dtype)
            loss = loss * dim_w[None, None, :]

        H = loss.shape[1]
        n_exec = self.config.n_action_steps
        pos_w = torch.ones(H, device=loss.device, dtype=loss.dtype)
        pos_w[n_exec:] = self.config.future_steps_weight
        if self.config.pos_decay_lambda > 0.0:
            pos = torch.arange(H, device=loss.device, dtype=loss.dtype)
            pos_w = pos_w * torch.exp(-self.config.pos_decay_lambda * pos)
        loss = loss * pos_w[None, :, None]

        loss_dtype = loss.dtype
        B, H_, D_ = loss.shape

        is_pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
        if is_pad is not None:
            valid_t = (~is_pad.bool()).to(loss_dtype)
        else:
            valid_t = torch.ones(B, H_, device=loss.device, dtype=loss_dtype)

        dim_pad = batch.get("action_dim_pad")
        if dim_pad is not None:
            valid_d = (~dim_pad.bool()).to(loss_dtype)
        else:
            valid_d = torch.ones(B, D_, device=loss.device, dtype=loss_dtype)

        valid_cells = valid_t.unsqueeze(-1) * valid_d.unsqueeze(1)
        loss = loss * valid_cells

        denom = (pos_w[None, :, None] * valid_cells).sum().clamp(min=1e-6)
        main_loss = loss.sum() / denom

        contrastive_weight = getattr(self.config, "contrastive_loss_weight", 0.0)
        contrastive_loss_value = 0.0
        if (
            self.training
            and contrastive_weight > 0.0
            and lang_mask is not None
            and L_lang > 0
            and B >= 2
        ):
            perm = torch.randperm(B, device=device)
            if (perm == torch.arange(B, device=device)).any():
                perm = torch.roll(perm, shifts=1, dims=0)

            descs = batch.get("task")
            if descs is None:
                descs = batch.get("task_description")
            if descs is not None and len(descs) == B:
                perm_cpu = perm.detach().cpu().tolist()
                pair_diff = torch.tensor(
                    [descs[i] != descs[perm_cpu[i]] for i in range(B)],
                    device=device,
                    dtype=torch.bool,
                )
            else:
                pair_diff = torch.ones(B, device=device, dtype=torch.bool)

            if pair_diff.any():
                vis_part = vlm_seq[:, :L_vis]
                lang_part = vlm_seq[:, L_vis:L_vis + L_lang]
                state_part = vlm_seq[:, L_vis + L_lang:]
                vlm_seq_wrong = torch.cat(
                    [vis_part, lang_part[perm], state_part], dim=1,
                )
                lang_mask_wrong = lang_mask[perm]

                v_wrong = self.velocity_field(
                    x_t, t, vlm_seq_wrong.to(x_t.dtype),
                    robot_tokens=robot_tokens,
                    lang_start=L_vis, lang_len=L_lang, lang_mask=lang_mask_wrong,
                    grid_thw_list=grid_thw_list,
                )

                diff_sq = (v_t - v_wrong).pow(2).mean(dim=[1, 2])
                margin = float(getattr(self.config, "contrastive_margin", 0.05))
                hinge = F.relu(margin - diff_sq) * pair_diff.float()
                n_valid = pair_diff.float().sum().clamp(min=1.0)
                loss_contrastive = hinge.sum() / n_valid
                contrastive_loss_value = float(loss_contrastive.detach())
                main_loss = main_loss + contrastive_weight * loss_contrastive

        self._last_loss_components = {
            "main": float(main_loss.detach() - contrastive_weight * contrastive_loss_value),
            "contrastive": contrastive_loss_value,
        }
        return main_loss

    def forward(self, batch: dict) -> tuple:
        if self.training:
            return self.compute_loss(batch), {}
        return self.sample_actions(batch), {}

    @torch.no_grad()
    def sample_actions(self, batch: dict) -> torch.Tensor:
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )

        with autocast_ctx:
            robot_tokens = self._compute_robot_tokens(batch)
            vlm_seq, L_vis, lang_mask, grid_thw_list = self._build_vlm_seq(batch)
            vlm_seq = vlm_seq.float()
            L_lang = lang_mask.shape[1] if lang_mask is not None else 0

            x_t = self.sample_noise(
                (B, self.config.horizon, self.config.action_dim), device=device,
            )
            N = self.config.num_inference_steps
            dt = -1.0 / N
            t = torch.ones(B, device=device, dtype=torch.float32)

            for _ in range(N):
                v_t = self.velocity_field(x_t, t, vlm_seq.to(x_t.dtype), robot_tokens=robot_tokens,
                                          lang_start=L_vis, lang_len=L_lang, lang_mask=lang_mask,
                                          grid_thw_list=grid_thw_list)
                x_t = x_t + dt * v_t
                t = t + dt

        return x_t[:, : self.config.n_action_steps]

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
