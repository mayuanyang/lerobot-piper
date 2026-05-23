"""
InterleavedFlowMatchingTransformer — SmolVLA-style joint VLM + action expert.

How this differs from the encoder-decoder `transformer_flow_matching`:
  - Encoder-decoder: VLM runs to completion (frozen), action expert
    cross-attends to VLM outputs. VLM's attention pattern is fixed by the
    prefix alone; action context cannot influence VLM perception.
  - **Interleaved (this file)**: at every VLM depth, expert tokens join the
    VLM sequence in a single self-attention pass. VLM tokens still use their
    frozen Q/K/V/FFN; expert tokens use a parallel trainable Q/K/V/FFN. The
    softmax is computed over the *combined* (VLM + expert) sequence, so:
      • VLM tokens attend to expert tokens → VLM activations become
        task-aware even though VLM weights are frozen.
      • Expert tokens attend to VLM tokens at every layer (no separate
        cross-attention block needed).

Mask semantics:
                 VLM    Latent  Action
    VLM          full   ✓       ✓ (gated by `vlm_attends_to_expert`)
    Latent       ✓      ✓       ✗ (never see noisy actions)
    Action       ✓      ✓       causal (only earlier actions)

Caveats / things to verify when running:
  - The VLM layer attribute paths (e.g. `layer.self_attn.q_proj`) match
    SmolVLM2's actual structure but may need adjustment for transformers
    version drift. Failures will be visible immediately on first forward.
  - GQA: SmolLM2 may have num_kv_heads < num_heads; we replicate K/V
    heads for the expert side and use the same num_kv_heads.
  - RoPE: positions are assigned sequentially across [VLM, latent, action].
    Action token positions sit *after* VLM, which is unconventional but
    consistent with treating the whole thing as one sequence.
"""

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from .interleaved_flow_matching_config import InterleavedFlowMatchingConfig
from .expert_layer import ExpertProjections, RMSNorm
from ..transformer_flow_matching.robot_visual_encoder import RobotVisualEncoder


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (matches the encoder-decoder model — same math,
# duplicated here so this file is self-contained).
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
# RoPE helpers. Llama-style: rotate pairs of dims (i, i+1) by frequency-scaled
# angle. We re-implement here rather than import from transformers to avoid
# version-coupling.
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) each shaped (1, seq_len, head_dim) ready for broadcast."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)        # (L, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                       # (L, head_dim)
    return emb.cos().to(dtype).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k:  (B, H, L, D)
    cos, sin: (1, L, D) → unsqueeze head dim for broadcast.
    """
    cos = cos.unsqueeze(1)   # (1, 1, L, D)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class InterleavedFlowMatchingTransformer(nn.Module):
    """Interleaved (SmolVLA-style) flow matching policy."""

    VLM_MODEL_ID: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def __init__(self, config: InterleavedFlowMatchingConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # 1. Load SmolVLM2 (frozen)
        # ------------------------------------------------------------------
        print(f"Loading {self.VLM_MODEL_ID} ...")
        vlm = AutoModelForImageTextToText.from_pretrained(
            self.VLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.VLM_MODEL_ID)
        vlm_model = vlm.model
        self.vision_model = vlm_model.vision_model
        self.connector = vlm_model.connector
        self.text_model = vlm_model.text_model

        # Truncate to first num_vlm_layers
        total_layers = len(self.text_model.layers)
        self.text_model.layers = self.text_model.layers[: config.num_vlm_layers]
        print(f"Using first {len(self.text_model.layers)}/{total_layers} text layers")

        # Read VLM attention shape from layer 0 (assume uniform across layers).
        first_attn = self.text_model.layers[0].self_attn
        text_cfg = self.text_model.config
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_heads = int(text_cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", self.num_heads))
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = int(text_cfg.intermediate_size)
        self.rms_norm_eps = float(getattr(text_cfg, "rms_norm_eps", 1e-5))
        self.rope_theta = float(getattr(text_cfg, "rope_theta", 10000.0))

        # Sanity: the SmolVLA-style joint attention needs the config's d_model
        # to match VLM hidden. We override silently and warn rather than fail.
        if config.d_model != self.hidden_size:
            print(f"[interleaved] forcing d_model {config.d_model} → {self.hidden_size} to match VLM")
            config.d_model = self.hidden_size

        # Freeze VLM
        for component in [self.vision_model, self.connector, self.text_model]:
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
            for _ in range(len(self.text_model.layers))
        ])

        # ------------------------------------------------------------------
        # 3. Robot CNN (optional, ablation flag) and state encoder
        # ------------------------------------------------------------------
        if config.use_robot_cnn:
            self.robot_visual_encoder = RobotVisualEncoder(
                input_size=config.robot_encoder_input_size,
                out_tokens=config.robot_encoder_tokens,
                out_dim=self.hidden_size,
            )
        else:
            self.robot_visual_encoder = None
            print("[interleaved] use_robot_cnn=False — RobotVisualEncoder disabled (ablation mode)")
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

        # Time embedding fused with action emb (same recipe as encoder-decoder).
        self.action_time_mlp_in = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size)

        # Sinusoidal positional encoding within the action segment.
        # (RoPE is applied across the entire joint sequence too; this is a
        # small additional learned-friendly cue at the action positions.)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, config.horizon, self.hidden_size))
        nn.init.normal_(self.action_pos_emb, std=0.02)

        # ------------------------------------------------------------------
        # 5. Latent "thought" tokens
        # ------------------------------------------------------------------
        self.num_latent_tokens = config.num_latent_tokens
        if self.num_latent_tokens > 0:
            self.latent_embs = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.hidden_size))
            nn.init.normal_(self.latent_embs, std=0.02)

        # ------------------------------------------------------------------
        # 6. Final norm before action readout
        # ------------------------------------------------------------------
        self.final_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # ------------------------------------------------------------------
        # 7. Language adaptor — trainable residual projection that lets the
        #    model shift SmolVLM2's frozen language embeddings towards the
        #    robot instruction-following domain. Zero-init so training starts
        #    from the original frozen behaviour and learns gradually.
        # ------------------------------------------------------------------
        self.lang_adaptor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            RMSNorm(self.hidden_size, eps=self.rms_norm_eps),
        )
        nn.init.zeros_(self.lang_adaptor[0].weight)
        nn.init.zeros_(self.lang_adaptor[0].bias)

        # Learnable attention bias that boosts every expert query's attention
        # to language keys in the joint softmax. Initialised to 0 so the model
        # starts at "no extra bias" and learns how much language matters.
        self.lang_attn_bias = nn.Parameter(torch.tensor(0.0))

        self._lang_max_len = 48

    # =====================================================================
    # Keep frozen components in eval mode
    # =====================================================================

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        self.connector.eval()
        self.text_model.eval()
        return self

    # =====================================================================
    # Encoding helpers
    # =====================================================================

    def _encode_images(self, batch: dict, B: int) -> torch.Tensor:
        """Vision tokens, frozen. (B, V*num_cams, hidden_size) bfloat16."""
        vlm_dtype = next(self.vision_model.parameters()).dtype
        all_vis = []
        for cam_key in self.config.cameras_for_vision_state_concat:
            if cam_key not in batch:
                continue
            imgs = batch[cam_key]
            img = imgs[:, -1] if imgs.dim() == 5 else imgs
            img = img * 2.0 - 1.0     # SigLIP convention

            target = self.config.vision_input_size
            h, w = img.shape[-2], img.shape[-1]
            if h != w:
                max_dim = max(h, w)
                pad = (max_dim - w) // 2, max_dim - w - (max_dim - w) // 2, \
                      (max_dim - h) // 2, max_dim - h - (max_dim - h) // 2
                img = F.pad(img.float(), pad, value=-1.0)
            if img.shape[-2] != target or img.shape[-1] != target:
                img = F.interpolate(img.float(), size=(target, target),
                                     mode="bilinear", align_corners=False).to(vlm_dtype)
            else:
                img = img.to(vlm_dtype)

            vis_hidden = self.vision_model(pixel_values=img).last_hidden_state
            vis_tokens = self.connector(vis_hidden)
            all_vis.append(vis_tokens)
        if not all_vis:
            device = batch["observation.state"].device
            return torch.zeros(B, 0, self.hidden_size, device=device, dtype=torch.bfloat16)
        return torch.cat(all_vis, dim=1)

    def _encode_language(self, batch: dict, device: torch.device) -> Optional[torch.Tensor]:
        """Frozen embed_tokens of the task description. (B, L, hidden) or None.

        Accept either "task_description" (set by some training scripts) or
        "task" (the raw key produced by LeRobotDataset that survives most
        preprocessors). Falling back to "task" matters because some
        preprocessors strip unknown keys before the model sees the batch.
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
        return self.text_model.get_input_embeddings()(input_ids)

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
    ) -> torch.Tensor:
        """
        (L_total, L_total) additive mask. -inf blocks attention.

        Layout:
          rows/cols [0..L_vlm)                 = VLM tokens (vision + lang + state, + robot if VLM-side)
          rows/cols [L_vlm..L_vlm+R)           = robot tokens (if expert-side, else R=0 here)
          rows/cols [L_vlm+R..L_vlm+R+K)       = latent tokens
          rows/cols [L_vlm+R+K..L_total)       = noisy action tokens

        lang_start, lang_len: position range of language tokens within the
            VLM-side sequence, used to apply a learnable attention bias that
            counters vision's numerical dominance in the joint softmax.
        """
        L_total = L_vlm + R + K + H
        mask = torch.zeros(L_total, L_total, device=device, dtype=dtype)
        a_start = L_vlm + R + K
        l_start = L_vlm + R
        r_start = L_vlm

        # Robot on expert side must not see noisy actions (perception is about
        # the world, not about what action we plan to take next).
        if R > 0:
            mask[r_start:r_start + R, a_start:] = float("-inf")

        # Latents must never see noisy actions (keeps them as "pure thoughts").
        if K > 0:
            mask[l_start:l_start + K, a_start:] = float("-inf")

        # Optionally block VLM → expert (gate behind config flag).
        if not self.config.vlm_attends_to_expert:
            mask[:L_vlm, L_vlm:] = float("-inf")

        # Causal among actions.
        if H > 1:
            causal = torch.triu(
                torch.full((H, H), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            mask[a_start:, a_start:] = causal

        # Learnable positive bias on expert→language attention paths.
        # Vision tokens dominate the joint softmax (~546 vs ≤48 language keys);
        # a small learnable bias lets the model up-weight language when needed.
        if lang_len > 0:
            mask[L_vlm:, lang_start:lang_start + lang_len] += self.lang_attn_bias

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
        """
        One layer of joint VLM + expert processing. Returns updated (vlm_seq, exp_seq).

        cos, sin: precomputed RoPE for the *full* combined sequence length.
        attn_mask: precomputed additive mask for the full combined sequence.
        """
        vlm_layer = self.text_model.layers[layer_idx]
        exp_layer = self.expert_layers[layer_idx]

        B, L_vlm, _ = vlm_seq.shape
        L_exp = exp_seq.size(1)
        L_total = L_vlm + L_exp
        H = self.num_heads
        Hk = self.num_kv_heads
        D = self.head_dim

        # ---- Pre-norm (separate weights) ----
        vlm_norm = vlm_layer.input_layernorm(vlm_seq)
        exp_norm = exp_layer.input_layernorm(exp_seq)

        # ---- Q/K/V from two parallel projection sets ----
        # VLM side uses its frozen projections; expert side uses trainable.
        Q_vlm = vlm_layer.self_attn.q_proj(vlm_norm)
        K_vlm = vlm_layer.self_attn.k_proj(vlm_norm)
        V_vlm = vlm_layer.self_attn.v_proj(vlm_norm)

        Q_exp = exp_layer.q_proj(exp_norm)
        K_exp = exp_layer.k_proj(exp_norm)
        V_exp = exp_layer.v_proj(exp_norm)

        # Concat along sequence axis BEFORE reshape so positions match the
        # combined RoPE table.
        Q = torch.cat([Q_vlm, Q_exp], dim=1).view(B, L_total, H, D).transpose(1, 2)
        K = torch.cat([K_vlm, K_exp], dim=1).view(B, L_total, Hk, D).transpose(1, 2)
        V = torch.cat([V_vlm, V_exp], dim=1).view(B, L_total, Hk, D).transpose(1, 2)

        # ---- RoPE on Q, K ----
        Q, K = _apply_rope(Q, K, cos, sin)

        # ---- GQA: repeat K/V heads to match Q heads ----
        if Hk != H:
            repeat = H // Hk
            K = K.repeat_interleave(repeat, dim=1)
            V = V.repeat_interleave(repeat, dim=1)

        # ---- Scaled dot-product attention over the joint sequence ----
        # attn_mask: (L_total, L_total). SDPA broadcasts over batch + heads.
        attn_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_total, H * D)

        # Split + per-side output projection (separate weights)
        attn_vlm = vlm_layer.self_attn.o_proj(attn_out[:, :L_vlm])
        attn_exp = exp_layer.o_proj(attn_out[:, L_vlm:])
        attn_exp = exp_layer.attn_dropout(attn_exp)

        # Residual after attention
        vlm_seq = vlm_seq + attn_vlm
        exp_seq = exp_seq + attn_exp

        # ---- FFN (pre-norm, parallel) ----
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

    def _build_expert_seq(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        robot_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the expert-side sequence.
        Layout when robot_tokens given (expert-side mode):
            [robot, latent, action]   (B, R + K + H, hidden_size)
        Layout when robot_tokens is None (default VLM-side mode):
            [latent, action]          (B, K + H, hidden_size)
        """
        B, H, _ = noisy_actions.shape

        action_emb = self.action_in_proj(noisy_actions)            # (B, H, hidden)
        action_emb = action_emb + self.action_pos_emb[:, :H]       # add positional bias

        # Time conditioning (sinusoidal → fused MLP, residual on action_emb)
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
        if self.num_latent_tokens > 0:
            parts.append(self.latent_embs.expand(B, -1, -1).to(action_tgt.dtype))
        parts.append(action_tgt)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    # =====================================================================
    # Compute robot CNN tokens once (used by either VLM or expert side)
    # =====================================================================

    def _compute_robot_tokens(self, batch: dict) -> Optional[torch.Tensor]:
        """
        Run RobotVisualEncoder per camera, concat to (B, R, hidden_size).
        Returns None if the encoder is disabled (use_robot_cnn=False) or no
        cameras are present in the batch.
        """
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
        return torch.cat(robot_tokens_list, dim=1)

    # =====================================================================
    # Build VLM-side sequence: frozen vision + language + (trainable) state
    # Robot CNN tokens always go to the expert side, not here.
    # =====================================================================

    def _build_vlm_seq(self, batch: dict) -> tuple[torch.Tensor, int, int]:
        """Build [vision, language, state] and return (vlm_seq, L_vis, L_lang).

        Robot tokens belong on expert side.
        Language adaptor (trainable residual) is applied here so the model can
        learn to shift frozen language embeddings towards the robot domain.
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        with torch.no_grad():
            vis_tokens = self._encode_images(batch, B)
            lang_tokens = self._encode_language(batch, device)

        L_vis = vis_tokens.shape[1]
        L_lang = lang_tokens.shape[1] if lang_tokens is not None else 0

        # Apply trainable language adaptor (zero-init residual).
        if lang_tokens is not None and L_lang > 0:
            lang_tokens = lang_tokens + self.lang_adaptor(lang_tokens.to(vis_tokens.dtype))

        # State token
        state = batch["observation.state"].float()
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state = state.nan_to_num(0.0).clamp(-10.0, 10.0)
        state_tokens = self.state_encoder(state).to(vis_tokens.dtype)

        parts = [vis_tokens]
        if lang_tokens is not None:
            parts.append(lang_tokens)
        parts.append(state_tokens)
        return torch.cat(parts, dim=1), L_vis, L_lang

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
    ) -> torch.Tensor:
        """
        Run joint attention stack and read off action positions.

        If robot_tokens is passed (expert-side mode), expert sequence layout is
        [robot, latent, action] and we skip R+K positions for readout.
        Otherwise (default VLM-side mode), expert sequence is [latent, action]
        and we skip K positions for readout.

        lang_start, lang_len: position range of language tokens within vlm_seq,
            forwarded to _build_joint_mask for the language attention bias.
        """
        B, H, _ = noisy_actions.shape
        K = self.num_latent_tokens
        R = robot_tokens.size(1) if robot_tokens is not None else 0
        L_vlm = vlm_seq.size(1)

        exp_seq = self._build_expert_seq(noisy_actions, timesteps, robot_tokens).to(vlm_seq.dtype)

        L_total = L_vlm + R + K + H
        cos, sin = _build_rope_cache(L_total, self.head_dim, self.rope_theta,
                                       vlm_seq.device, vlm_seq.dtype)
        attn_mask = self._build_joint_mask(L_vlm, R, K, H, vlm_seq.device, vlm_seq.dtype,
                                           lang_start=lang_start, lang_len=lang_len)

        # Run all joint layers
        for i in range(len(self.expert_layers)):
            vlm_seq, exp_seq = self._joint_layer(
                vlm_seq, exp_seq, layer_idx=i,
                cos=cos, sin=sin, attn_mask=attn_mask,
            )

        # Read out the action positions (skip robot + latent prefix)
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

        # Robot tokens always go to the expert side (or are absent entirely
        # if use_robot_cnn=False). VLM side never carries robot tokens.
        robot_tokens = self._compute_robot_tokens(batch)

        vlm_seq, L_vis, L_lang = self._build_vlm_seq(batch)
        vlm_seq = vlm_seq.float()  # joint attention runs in fp32 for stability

        noise = self.sample_noise(actions.shape, actions.device)
        t = self.sample_time(B, actions.device)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions
        u_t = noise - actions

        v_t = self.velocity_field(x_t, t, vlm_seq.to(x_t.dtype), robot_tokens=robot_tokens,
                                  lang_start=L_vis, lang_len=L_lang)

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

        is_pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
        if is_pad is not None:
            valid = ~is_pad.bool()
            loss = loss * valid.unsqueeze(-1).float()
            pos_w_sum = (pos_w[None, :] * valid.float()).sum().clamp(min=1e-6)
            return loss.sum() / (pos_w_sum * self.config.action_dim)
        return loss.mean()

    def forward(self, batch: dict) -> tuple:
        if self.training:
            return self.compute_loss(batch), {}
        return self.sample_actions(batch), {}

    @torch.no_grad()
    def sample_actions(self, batch: dict) -> torch.Tensor:
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        # Wrap inference in bfloat16 autocast to match the training-time
        # context. The frozen VLM was loaded as bf16 (saves memory) and its
        # weights stay bf16; without autocast, fp32 inputs to bf16 linear
        # layers raise "expected mat1 and mat2 to have the same dtype".
        # Training works because the train script wraps policy.forward in
        # the same autocast block.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )

        with autocast_ctx:
            # Same routing as compute_loss: robot tokens (if any) go to expert side.
            robot_tokens = self._compute_robot_tokens(batch)
            vlm_seq, L_vis, L_lang = self._build_vlm_seq(batch)
            vlm_seq = vlm_seq.float()

            x_t = self.sample_noise(
                (B, self.config.horizon, self.config.action_dim), device=device,
            )
            N = self.config.num_inference_steps
            dt = -1.0 / N
            t = torch.ones(B, device=device, dtype=torch.float32)

            for _ in range(N):
                v_t = self.velocity_field(x_t, t, vlm_seq.to(x_t.dtype), robot_tokens=robot_tokens,
                                          lang_start=L_vis, lang_len=L_lang)
                x_t = x_t + dt * v_t
                t = t + dt

        return x_t[:, : self.config.n_action_steps]

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
