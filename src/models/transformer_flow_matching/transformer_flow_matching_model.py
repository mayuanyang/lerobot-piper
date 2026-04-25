"""
FlowMatchingTransformer: VLM-based flow matching policy for robot manipulation.

Architecture:
  - Frozen SmolVLM2-500M prefix encoder (ViT + connector + first N text layers)
    processes images + language into rich context tokens (960-dim).
  - Trainable context_proj (960 → d_model) and state_encoder (state_dim → d_model).
  - Trainable action expert: 8-layer TransformerDecoder cross-attending to context.
  - Flow matching loss with Beta(1.5, 1.0) time sampling (same as SmolVLA).

This design is similar to SmolVLA (encoder-decoder variant):
  SmolVLA uses interleaved VLM+expert layers.
  This model uses the VLM as a frozen context encoder then a separate action expert.
  Both leverage pretrained image-language representations from SmolVLM2.

Memory: ~2GB for batch=128 (VLM frozen → no activation storage through 32 layers).
Trainable params: ~26M (context_proj + state_encoder + action expert).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from .transformer_flow_matching_config import TransformerFlowMatchingConfig


# ---------------------------------------------------------------------------
# Utility: sinusoidal time embedding (SmolVLA-style, more expressive than
# a learned embedding for continuous flow-matching timesteps in [0, 1])
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """
    Sine-cosine positional embedding for scalar flow-matching timesteps.

    Args:
        time: (B,) timestep values in [0, 1].
        dimension: embedding dimension (must be even).
        min_period / max_period: frequency range.
    Returns:
        (B, dimension) float32 embeddings.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension must be even, got {dimension}")
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period) * 2.0 * math.pi
    sin_input = scaling[None, :] * time[:, None].float()
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# ---------------------------------------------------------------------------
# Positional encoding for action sequence
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for fixed-length sequences."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FlowMatchingTransformer(nn.Module):
    """
    Flow matching policy with frozen SmolVLM2 context encoder and trainable action expert.

    Forward (training):
      1. encode_prefix(batch) → (B, N, 960) [frozen VLM, no_grad]
      2. context_proj → (B, N, d_model)      [trainable]
      3. state_encoder(state) → (B, T, d_model)  [trainable]
      4. full_context = cat([context, state_tokens])
      5. velocity_field(x_t, t, full_context) → (B, H, action_dim)
      6. loss = MSE(velocity, noise - actions)
    """

    # SmolVLM2-500M text hidden size (fixed by pretrained model)
    VLM_HIDDEN: int = 960
    VLM_MODEL_ID: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def __init__(self, config: TransformerFlowMatchingConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # Load SmolVLM2-500M and extract frozen components
        # ------------------------------------------------------------------
        print(f"Loading {self.VLM_MODEL_ID} ...")
        vlm = AutoModelForImageTextToText.from_pretrained(
            self.VLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.VLM_MODEL_ID)

        vlm_model = vlm.model  # SmolVLMModel
        self.vision_model = vlm_model.vision_model   # SigLIP ViT (768-dim patches)
        self.connector = vlm_model.connector          # pixel-shuffle → 960-dim tokens
        self.text_model = vlm_model.text_model        # SmolLM2 (960-dim, 32 layers)

        # Truncate text model to first num_vlm_layers layers (saves compute, SmolVLA uses 16)
        num_vlm_layers = config.num_vlm_layers
        total_layers = len(self.text_model.layers)
        print(f"Using first {num_vlm_layers}/{total_layers} VLM text layers")
        self.text_model.layers = self.text_model.layers[:num_vlm_layers]

        # Freeze ALL VLM components — only the action expert trains
        for component in [self.vision_model, self.connector, self.text_model]:
            for p in component.parameters():
                p.requires_grad = False
            component.eval()

        # Free unused VLM weights (lm_head, etc.) to reduce memory footprint
        del vlm

        # ------------------------------------------------------------------
        # Trainable components (~26M params total)
        # ------------------------------------------------------------------

        # 1. Project VLM context to action expert dimension + LayerNorm for stability.
        #    LayerNorm is critical: VLM bfloat16 hidden states can have large per-token
        #    magnitudes (due to large RMSNorm γ values in the pretrained model).
        #    Without normalization, context_proj amplifies these and causes gradient explosion.
        self.context_proj = nn.Linear(self.VLM_HIDDEN, config.d_model)
        self.context_norm = nn.LayerNorm(config.d_model)

        # 2. State encoder: (B, T_obs, state_dim) → (B, T_obs, d_model)
        #    Uses n_obs_steps states (2 timesteps = position + velocity info).
        #    LayerNorm matches the treatment of context tokens (context_norm) so both
        #    inputs to the cross-attention memory have consistent scale.
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # 3. Action projections
        self.action_in_proj = nn.Linear(config.action_dim, config.d_model)
        # Zero-init action_out_proj so initial velocity predictions are ~0.
        # This makes initial loss ≈ E[u_t²] ≈ 2 instead of millions.
        self.action_out_proj = nn.Linear(config.d_model, config.action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)
        self.action_positional_encoding = PositionalEncoding(config.d_model, max_len=config.horizon)

        # 4. Time embedding MLP: [action_emb ‖ sinusoidal_time_emb] → d_model (SmolVLA-style)
        self.action_time_mlp_in = nn.Linear(config.d_model * 2, config.d_model)
        self.action_time_mlp_out = nn.Linear(config.d_model, config.d_model)

        # 5. Action expert: transformer decoder, cross-attends to VLM context + state
        expert_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
            dropout=0.0,
            norm_first=True,  # Pre-LayerNorm: more stable training
        )
        self.actions_expert = nn.TransformerDecoder(
            expert_layer, num_layers=config.num_decoder_layers
        )

        self._lang_max_len = 48

    # ------------------------------------------------------------------
    # Override train() to keep frozen VLM components in eval mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep frozen VLM in eval (prevents BN/dropout from activating)
        self.vision_model.eval()
        self.connector.eval()
        self.text_model.eval()
        return self

    # ------------------------------------------------------------------
    # Frozen prefix encoding (images + language → VLM hidden states)
    # ------------------------------------------------------------------

    def _encode_images(self, batch: dict, B: int, T_obs: int) -> torch.Tensor:
        """
        Encode last-obs-step images for all cameras through frozen ViT + connector.
        Returns: (B, num_cameras * tokens_per_image, VLM_HIDDEN) bfloat16
        """
        vlm_dtype = next(self.vision_model.parameters()).dtype
        all_vis_tokens = []

        for cam_key in self.config.cameras_for_vision_state_concat:
            if cam_key not in batch:
                continue
            imgs = batch[cam_key]
            # LeRobot omits the time dimension when T_obs=1, giving (B, C, H, W).
            # With T_obs>1 the shape is (B, T_obs, C, H, W); take the last step.
            if imgs.dim() == 5:
                img = imgs[:, -1]  # (B, C, H, W)
            else:
                img = imgs         # (B, C, H, W) — already squeezed by dataloader

            # Normalize [0, 1] → [-1, 1] for SigLIP
            img = img * 2.0 - 1.0

            # Resize to vision_input_size if needed
            target = self.config.vision_input_size
            if img.shape[-2] != target or img.shape[-1] != target:
                img = F.interpolate(
                    img.float(), size=(target, target),
                    mode="bilinear", align_corners=False,
                ).to(vlm_dtype)
            else:
                img = img.to(vlm_dtype)

            # ViT: (B, C, H, W) → (B, num_patches, 768)
            vis_hidden = self.vision_model(pixel_values=img).last_hidden_state
            # Connector (pixel-shuffle MLP): (B, num_patches, 768) → (B, tokens, 960)
            vis_tokens = self.connector(vis_hidden)

            all_vis_tokens.append(vis_tokens)

        if not all_vis_tokens:
            device = batch["observation.state"].device
            return torch.zeros(B, 0, self.VLM_HIDDEN, device=device, dtype=torch.bfloat16)

        return torch.cat(all_vis_tokens, dim=1)  # (B, V*num_cameras, 960)

    def _encode_language(self, batch: dict, B: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Tokenize task descriptions and embed through frozen embed_tokens.
        Returns: (B, L, VLM_HIDDEN) bfloat16, or None if no descriptions.
        """
        if "task_description" not in batch:
            return None
        descriptions = batch["task_description"]
        if not descriptions or not any(descriptions):
            return None

        inputs = self.processor.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._lang_max_len,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(device)  # (B, L)

        # Frozen embed_tokens: (B, L) → (B, L, 960)
        lang_embs = self.text_model.get_input_embeddings()(input_ids)
        return lang_embs  # bfloat16

    @torch.no_grad()
    def encode_prefix(self, batch: dict, B: int, T_obs: int) -> torch.Tensor:
        """
        Frozen VLM prefix encoding.
        images + language → [N VLM text layers] → (B, N, VLM_HIDDEN) bfloat16

        Uses @torch.no_grad() so VLM activations are NOT stored for backward.
        Gradient flows into context_proj (trainable) but NOT through VLM weights.
        This gives full VLM representational power at minimal memory cost.
        """
        device = batch["observation.state"].device

        # Encode images
        vis_tokens = self._encode_images(batch, B, T_obs)  # (B, V, 960)

        # Encode language
        lang_tokens = self._encode_language(batch, B, device)  # (B, L, 960) or None

        # NOTE: Do NOT scale by sqrt(960) here.
        # SmolVLA applies that scaling in their custom SmolVLMWithExpertModel which
        # interleaves expert layers inside the VLM. Here we only use the VLM as a
        # frozen feature extractor — the RMSNorm at each layer already controls scale.
        # Adding sqrt(960)≈31× scaling would push VLM attention into degenerate regimes.
        prefix_parts = [vis_tokens]
        if lang_tokens is not None:
            prefix_parts.append(lang_tokens)

        prefix_embs = torch.cat(prefix_parts, dim=1)  # (B, N, 960)

        # Run through truncated VLM text layers (frozen, causal attention).
        # Pass attention_mask=None → model uses its internal causal mask, which is
        # numerically cleaner than passing an explicit all-ones boolean mask.
        text_out = self.text_model(
            inputs_embeds=prefix_embs,
            attention_mask=None,
            use_cache=False,
        )
        return text_out.last_hidden_state  # (B, N, 960) bfloat16

    # ------------------------------------------------------------------
    # Context assembly (trainable projections on top of frozen VLM output)
    # ------------------------------------------------------------------

    def get_condition(self, batch: dict) -> torch.Tensor:
        """
        Build full context tensor for the action expert.

        Returns: (B, N + T_obs, d_model) float32
          - First N tokens: VLM context (images + language)
          - Last T_obs tokens: state tokens (current + prev timestep for velocity)
        """
        B = batch["observation.state"].shape[0]
        T_obs = self.config.n_obs_steps
        device = batch["observation.state"].device

        # Frozen VLM context — no gradient through this path
        context_hidden = self.encode_prefix(batch, B, T_obs)  # (B, N, 960) bfloat16

        # Guard: replace any NaN/Inf from bfloat16 VLM forward (rare but possible
        # with non-standard inputs like our connector tokens without image special tokens)
        context_hidden = torch.nan_to_num(context_hidden, nan=0.0, posinf=1.0, neginf=-1.0)

        # Trainable projection + LayerNorm.
        # LayerNorm normalizes away large-scale variation in VLM hidden states
        # (pretrained RMSNorm γ values can be large), preventing gradient explosion.
        context = self.context_norm(self.context_proj(context_hidden.float()))  # (B, N, d_model)

        # Trainable state encoder: direct gradient path through action_expert → state_encoder
        state = batch["observation.state"].float()      # (B, T_obs, state_dim) or (B, state_dim)
        if state.dim() == 2:
            state = state.unsqueeze(1)                  # (B, 1, state_dim)
        # Guard against large finite values from zero-variance state dimensions.
        # MEAN_STD normalization computes (x - mean) / (std + eps). For a dimension with
        # std=0 (e.g. a locked joint), this becomes x / eps giving values in the millions.
        # These are finite (not NaN/Inf), so nan_to_num won't catch them — use clamp.
        # Properly normalised dimensions stay within ±5; clamping to ±10 is a safe bound.
        state = state.nan_to_num(nan=0.0).clamp(-10.0, 10.0)
        state_tokens = self.state_encoder(state)        # (B, T_obs, d_model)

        # Combine: VLM context + state tokens
        return torch.cat([context, state_tokens], dim=1)  # (B, N+T_obs, d_model)

    # ------------------------------------------------------------------
    # Flow matching core
    # ------------------------------------------------------------------

    def sample_time(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Sample flow-matching timestep t ~ Beta(1.5, 1.0) clipped to (0.001, 1.0).
        Beta(1.5, 1.0) biases toward t=0 (near-clean), improving action accuracy
        for manipulation tasks (same distribution as SmolVLA).
        """
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=device),
            torch.tensor(1.0, device=device),
        )
        t = beta.sample((B,))
        return t * 0.999 + 0.001  # clamp away from exact 0/1

    def velocity_field(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the velocity field v(x_t, t) ≈ noise - actions.

        Args:
            noisy_actions: (B, H, action_dim) noisy action sequence at time t
            timesteps:     (B,) flow matching timesteps in [0, 1]
            context:       (B, N, d_model) VLM context + state tokens
        Returns:
            velocity: (B, H, action_dim)
        """
        B, T_act, _ = noisy_actions.shape

        # Project and positionally encode action tokens
        action_emb = self.action_in_proj(noisy_actions)      # (B, T, d_model)
        action_emb = self.action_positional_encoding(action_emb)

        # Sinusoidal time embedding (more expressive than learned for continuous t)
        time_emb = create_sinusoidal_pos_embedding(
            timesteps, self.config.d_model
        ).to(noisy_actions.dtype)                             # (B, d_model)
        time_emb = time_emb.unsqueeze(1).expand(-1, T_act, -1)  # (B, T, d_model)

        # MLP fusion: concat [action | time] → single token (SmolVLA-style)
        fused = torch.cat([action_emb, time_emb], dim=-1)    # (B, T, d_model*2)
        fused = F.silu(self.action_time_mlp_in(fused))
        fused = self.action_time_mlp_out(fused)               # (B, T, d_model)

        # Residual: preserve action information through MLP
        tgt = fused + action_emb

        # Action expert: decoder attends to VLM context + state.
        # Causal mask on SA (action token at step h only attends to steps 0..h) matches
        # SmolVLA's design: enforces temporal smoothness in the predicted action chunk.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T_act, device=noisy_actions.device, dtype=noisy_actions.dtype
        )
        output = self.actions_expert(tgt=tgt, memory=context, tgt_mask=causal_mask, tgt_is_causal=True)  # (B, T, d_model)

        return self.action_out_proj(output)                   # (B, T, action_dim)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Conditional flow matching loss with padded-step masking.

        Forward process: x_t = t * noise + (1 - t) * actions
        Target velocity: u_t = noise - actions
        Loss: MSE(v_theta(x_t, t, context), u_t) — MSE over non-padded steps only.

        Why masking matters:
          delta_timestamps requests future actions that may go past episode end.
          LeRobot fills those with zeros. After MEAN_STD normalization with non-zero
          action mean, padded zeros become (0 - mean)/std ≈ -5 to -8σ in normalized
          space. Without masking these appear as valid large targets → loss spikes of
          7M+ → gradient explosions (even though clipping prevents divergence, Adam's
          momentum accumulates bad signal from these batches).
        """
        actions = batch["action"].float().nan_to_num(nan=0.0).clamp(-10.0, 10.0)   # (B, H, action_dim)
        B = actions.shape[0]

        context = self.get_condition(batch)

        noise = torch.randn_like(actions)
        t = self.sample_time(B, actions.device)
        t_exp = t[:, None, None]

        x_t = t_exp * noise + (1.0 - t_exp) * actions   # noisy sample
        u_t = noise - actions                             # target velocity

        v_t = self.velocity_field(x_t, t, context)

        loss = F.mse_loss(v_t, u_t, reduction="none")  # (B, H, action_dim)

        # Mask out padding — LeRobot uses "action_is_pad" (bool, True = padded).
        # SmolVLA uses "actions_id_pad"; check both for compatibility.
        is_pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
        if is_pad is not None:
            valid = ~is_pad.bool()                           # (B, H)
            loss = loss * valid.unsqueeze(-1).float()        # zero out padded steps
            n_valid = valid.sum().clamp(min=1) * self.config.action_dim
            return loss.sum() / n_valid

        return loss.mean()

    def forward(self, batch: dict) -> tuple:
        """
        Training forward: returns (loss, {}).
        Inference forward: returns (actions, {}).
        """
        if self.training:
            loss = self.compute_loss(batch)
            return loss, {}
        else:
            actions = self.sample_actions(batch)
            return actions, {}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(self, batch: dict) -> torch.Tensor:
        """
        Sample actions via Euler ODE integration from t=1 (noise) to t=0 (actions).

        Returns: (B, n_action_steps, action_dim)
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        context = self.get_condition(batch)

        # Start from pure noise at t=1
        x_t = torch.randn(
            B, self.config.horizon, self.config.action_dim,
            device=device, dtype=torch.float32,
        )

        num_steps = self.config.num_inference_steps
        dt = torch.tensor(-1.0 / num_steps, device=device, dtype=torch.float32)
        t = torch.ones(B, device=device, dtype=torch.float32)

        for _ in range(num_steps):
            v_t = self.velocity_field(x_t, t, context)
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t[:, : self.config.n_action_steps]

    # ------------------------------------------------------------------
    # Debugging utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """Return trainable vs frozen parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
