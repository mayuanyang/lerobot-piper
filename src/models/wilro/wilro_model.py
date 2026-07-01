"""
WilroTransformer — SmolVLM2-based encoder-decoder flow matching policy.

Architecture (Xiaomi-Robotics-0 / pi0-style MoT, mirrors `wiltechs_vla` but
swaps the Qwen3-VL-4B backbone for HuggingFaceTB/SmolVLM2-500M-Video-Instruct):

  Stage A (run ONCE per inference): VLM encoder
    Input:   [vision tokens, language tokens]
    Run:     all SmolVLM2 text layers (frozen), with causal mask + RoPE
    Capture: post-RoPE K and V from the LAST `num_dit_layers` layers
             (these become the cross-attention memory for the DiT)

  Stage B (run `num_inference_steps` times during denoising): DiT decoder
    Input:   [SINK, state, (action_prefix), robot_cnn_tokens, latent_tokens,
              noisy_action_tokens(t)]
    Each layer:
       1. Self-attention with full causal mask (Λ-mask if a prefix is present)
       2. Cross-attention to ONE captured VLM KV pair (Q from DiT, K/V cached)
       3. SwiGLU FFN
       all three sublayers modulated by adaLN-Zero from the flow-matching time t

  Properties:
    - VLM never sees state / action / robot tokens — it stays in pure VL mode,
      preserving SmolVLM2's pretrained vision-language capabilities.
    - VLM runs once per inference (10× speedup vs interleaved at N=10 steps).
    - All VLM layers are used (not truncated) — DiT only reads from the last N
      as KV memory, but earlier layers still refine those features.
    - DiT inherits attention shape from the VLM text config so cross-attn GQA
      aligns naturally.
"""

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from .wilro_config import WilroConfig
from ..interleaved_flow_matching.expert_layer import RMSNorm, SwiGLU
from ..transformer_flow_matching.robot_visual_encoder import RobotVisualEncoder


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (flow matching)
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
# RoPE helpers — Llama-style 1D rotary positional embedding for the VLM.
# Used only inside the encoder pass; the DiT does not use RoPE (the VLM K
# already carries positional rotation, which is enough for cross-attention).
# ---------------------------------------------------------------------------

def _build_rope_cache(
    seq_len: int, head_dim: int, base: float, device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each (1, seq_len, head_dim), ready for broadcast."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)        # (L, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                       # (L, head_dim)
    return emb.cos().to(dtype).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, H, L, D); cos, sin: (1, L, D)."""
    cos = cos.unsqueeze(1)   # (1, 1, L, D)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# adaLN-Zero modulation
# ---------------------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: (B, L, D) — shift/scale: (B, D). Broadcasts over L."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _hard_negative_perm(
    descs: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a HARD-negative partner index for the contrastive language loss.

    For each sample i, pick the in-batch partner j (j != i, with a DIFFERENT
    instruction) that shares the most words with i — i.e. the most *confusable*
    negative available, not a random one. Random batch pairing almost never
    lands a same-template pair (e.g. two "put both ... in the basket" tasks that
    differ only in the object nouns), so the contrastive hinge gets satisfied by
    trivially-different instructions and never pressures fine-grained object
    grounding. Hard negatives put the gradient exactly where eval fails.

    Similarity is word-overlap (Jaccard). For LIBERO's templated strings the
    same-template tasks share the entire template and differ only in the object
    nouns, so the confusable minimal pair scores highest automatically; no
    object vocabulary or extra model is needed.

    Returns (perm, valid):
      - perm[i] = chosen partner index (perm[i]=i when no partner exists)
      - valid[i] = whether a different-instruction partner was found (False rows
        are skipped downstream via pair_diff)
    perm need NOT be a bijection — several samples may share the same hardest
    negative, which is fine for the gather-based shuffle. O(B^2) set ops on CPU;
    negligible next to the VLM forward.
    """
    B = len(descs)
    word_sets = [set(d.lower().split()) for d in descs]
    perm = list(range(B))
    valid = [False] * B
    for i in range(B):
        wi = word_sets[i]
        best_score, best = -1.0, []
        for j in range(B):
            if j == i or descs[j] == descs[i]:
                continue
            wj = word_sets[j]
            union = len(wi | wj)
            score = (len(wi & wj) / union) if union else 0.0
            if score > best_score + 1e-9:
                best_score, best = score, [j]
            elif score > best_score - 1e-9:
                best.append(j)
        if best:
            # Random pick among ties so the partner varies across steps.
            perm[i] = best[int(torch.randint(len(best), (1,)).item())]
            valid[i] = True
    return (
        torch.tensor(perm, device=device, dtype=torch.long),
        torch.tensor(valid, device=device, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# DiT layer: self-attn + cross-attn(to VLM KV) + FFN, modulated by adaLN-Zero
# ---------------------------------------------------------------------------

class DiTLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.1,
        use_robot_ca: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_robot_ca = use_robot_ca

        # ── Self-attention (over DiT sequence) ──────────────────────────
        self.sa_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.sa_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.sa_k = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.sa_v = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.sa_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.sa_drop = nn.Dropout(dropout)

        # ── Cross-attention (Q from DiT, K/V from VLM KV cache) ─────────
        self.ca_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ca_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.ca_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.ca_drop = nn.Dropout(dropout)

        # ── Robot CNN cross-attention (Q from DiT, K/V from Robot CNN) ──
        # This enables direct high-resolution spatial grounding: action
        # queries can attend to Robot CNN's fine-grained feature map
        # (14x14 @ 224x224) instead of only VLM's coarse SigLIP patches
        # (~729 patches @ 384x384). Critical for precise object localization
        # in spatial reasoning tasks (e.g., "bowl closer to plate").
        if use_robot_ca:
            self.robot_ca_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.robot_ca_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.robot_ca_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            self.robot_ca_drop = nn.Dropout(dropout)

        # ── FFN ─────────────────────────────────────────────────────────
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn = SwiGLU(hidden_size, intermediate_size)
        self.ffn_drop = nn.Dropout(dropout)

        # ── adaLN-Zero: 12 modulation vectors (shift/scale/gate × 4) ────
        # With robot_ca: 4 sublayers (sa, ca, robot_ca, ffn) × 3 = 12
        # Without robot_ca: 3 sublayers (sa, ca, ffn) × 3 = 9
        adaLN_dim = 12 * hidden_size if use_robot_ca else 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, adaLN_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
        vlm_kv_pad_mask: Optional[torch.Tensor],
        self_attn_mask: torch.Tensor,
        robot_k: Optional[torch.Tensor] = None,
        robot_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:               (B, L_dit, H)
        t_emb:           (B, H) — per-batch time conditioning
        vlm_k, vlm_v:    (B, num_kv_heads, L_vlm, head_dim) — frozen VLM cache
        vlm_kv_pad_mask: (B, L_vlm) bool, True at valid VLM positions
        self_attn_mask:  (1, 1, L_dit, L_dit) additive mask
        robot_k, robot_v: (B, num_kv_heads, R, head_dim) — Robot CNN K/V for
                         high-resolution spatial cross-attention (optional)
        """
        B, L_dit, _ = x.shape

        mod = self.adaLN_modulation(t_emb)
        if self.use_robot_ca and robot_k is not None:
            # 12 chunks: sa(3), ca(3), robot_ca(3), ffn(3)
            (
                s_sa, sc_sa, g_sa,
                s_ca, sc_ca, g_ca,
                s_rca, sc_rca, g_rca,
                s_ff, sc_ff, g_ff,
            ) = mod.chunk(12, dim=-1)
        else:
            # 9 chunks: sa(3), ca(3), ffn(3)
            (
                s_sa, sc_sa, g_sa,
                s_ca, sc_ca, g_ca,
                s_ff, sc_ff, g_ff,
            ) = mod.chunk(9, dim=-1)

        # ── Self-attention ───────────────────────────────────────────
        h = _modulate(self.sa_norm(x), s_sa, sc_sa)
        Q = self.sa_q(h).view(B, L_dit, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.sa_k(h).view(B, L_dit, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.sa_v(h).view(B, L_dit, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(r, dim=1)
            V = V.repeat_interleave(r, dim=1)
        sa = F.scaled_dot_product_attention(Q, K, V, attn_mask=self_attn_mask, is_causal=False)
        sa = sa.transpose(1, 2).contiguous().view(B, L_dit, self.num_heads * self.head_dim)
        sa = self.sa_drop(self.sa_o(sa))
        x = x + g_sa.unsqueeze(1) * sa

        # ── Cross-attention to frozen VLM cache ──────────────────────
        h = _modulate(self.ca_norm(x), s_ca, sc_ca)
        Q = self.ca_q(h).view(B, L_dit, self.num_heads, self.head_dim).transpose(1, 2)
        Kv, Vv = vlm_k, vlm_v
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            Kv = Kv.repeat_interleave(r, dim=1)
            Vv = Vv.repeat_interleave(r, dim=1)
        if vlm_kv_pad_mask is not None:
            kpad = ~vlm_kv_pad_mask                                 # True = pad
            ca_mask = torch.zeros(B, 1, 1, vlm_kv_pad_mask.shape[-1],
                                  device=x.device, dtype=Q.dtype)
            ca_mask.masked_fill_(kpad.unsqueeze(1).unsqueeze(1), float("-inf"))
        else:
            ca_mask = None
        ca = F.scaled_dot_product_attention(Q, Kv, Vv, attn_mask=ca_mask, is_causal=False)
        ca = ca.transpose(1, 2).contiguous().view(B, L_dit, self.num_heads * self.head_dim)
        ca = self.ca_drop(self.ca_o(ca))
        x = x + g_ca.unsqueeze(1) * ca

        # ── Robot CNN cross-attention (high-res spatial grounding) ───
        if self.use_robot_ca and robot_k is not None:
            h = _modulate(self.robot_ca_norm(x), s_rca, sc_rca)
            Q = self.robot_ca_q(h).view(B, L_dit, self.num_heads, self.head_dim).transpose(1, 2)
            Kr, Vr = robot_k, robot_v
            if self.num_kv_heads != self.num_heads:
                r = self.num_heads // self.num_kv_heads
                Kr = Kr.repeat_interleave(r, dim=1)
                Vr = Vr.repeat_interleave(r, dim=1)
            robot_ca = F.scaled_dot_product_attention(Q, Kr, Vr, is_causal=False)
            robot_ca = robot_ca.transpose(1, 2).contiguous().view(B, L_dit, self.num_heads * self.head_dim)
            robot_ca = self.robot_ca_drop(self.robot_ca_o(robot_ca))
            x = x + g_rca.unsqueeze(1) * robot_ca

        # ── FFN ──────────────────────────────────────────────────────
        h = _modulate(self.ffn_norm(x), s_ff, sc_ff)
        ff = self.ffn_drop(self.ffn(h))
        x = x + g_ff.unsqueeze(1) * ff

        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class WilroTransformer(nn.Module):
    """Encoder-decoder flow matching VLA built on frozen SmolVLM2-500M."""

    VLM_MODEL_ID: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def __init__(self, config: WilroConfig):
        super().__init__()
        self.config = config

        # ─────────────────────────────────────────────────────────────
        # 1. Load SmolVLM2 (frozen, ALL layers kept)
        # ─────────────────────────────────────────────────────────────
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

        # No layer truncation — encoder-decoder uses all text layers so that the
        # captured trailing KV caches benefit from the full upstream refinement.
        self.num_vlm_layers = len(self.text_model.layers)

        text_cfg = self.text_model.config
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_heads = int(text_cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(text_cfg, "head_dim", None) or (self.hidden_size // self.num_heads))
        self.intermediate_size = int(text_cfg.intermediate_size)
        self.rms_norm_eps = float(getattr(text_cfg, "rms_norm_eps", 1e-5))
        self.rope_theta = float(getattr(text_cfg, "rope_theta", 10000.0))
        print(f"VLM: {self.num_vlm_layers} layers  hidden={self.hidden_size}  "
              f"heads={self.num_heads}  kv_heads={self.num_kv_heads}  "
              f"head_dim={self.head_dim}  intermediate={self.intermediate_size}")

        if config.d_model != self.hidden_size:
            print(f"[wilro] forcing d_model {config.d_model} → {self.hidden_size} to match VLM")
            config.d_model = self.hidden_size

        # Freeze VLM
        for component in (self.vision_model, self.connector, self.text_model):
            for p in component.parameters():
                p.requires_grad = False
            component.eval()
        del vlm

        # ─────────────────────────────────────────────────────────────
        # 2. DiT (trainable) — N layers cross-attending to last N VLM KV pairs
        # ─────────────────────────────────────────────────────────────
        # `num_vlm_layers` field in config is reused as DiT depth (= number
        # of last VLM layers whose KV cache the DiT cross-attends to).
        self.num_dit_layers = int(config.num_vlm_layers)
        if self.num_dit_layers > self.num_vlm_layers:
            raise ValueError(
                f"num_dit_layers ({self.num_dit_layers}) > VLM layers "
                f"({self.num_vlm_layers}); not enough KV caches to source from."
            )

        # Resolve which VLM layers the DiT sources KV from. The resulting list is
        # ascending, length == num_dit_layers; DiT layer j cross-attends to the
        # KV cache captured at capture_indices[j].
        strategy = getattr(config, "kv_capture_strategy", "last")
        if strategy == "last":
            self.capture_indices = list(
                range(self.num_vlm_layers - self.num_dit_layers, self.num_vlm_layers)
            )
        elif strategy == "stride2":
            # End-anchored stride: always include the final (most refined) layer.
            stride = max(1, self.num_vlm_layers // self.num_dit_layers)
            idxs = list(range(self.num_vlm_layers - 1, -1, -stride))[: self.num_dit_layers]
            self.capture_indices = sorted(idxs)
        elif strategy == "custom":
            raw = list(getattr(config, "kv_capture_layers", []) or [])
            if not raw:
                raise ValueError(
                    "kv_capture_strategy='custom' requires a non-empty "
                    "config.kv_capture_layers list."
                )
            idxs = sorted({int(i) for i in raw})
            for i in idxs:
                if not (0 <= i < self.num_vlm_layers):
                    raise ValueError(
                        f"kv_capture_layers index {i} out of range "
                        f"[0, {self.num_vlm_layers})."
                    )
            self.capture_indices = idxs
            # DiT depth follows the explicit layer list, not num_vlm_layers.
            self.num_dit_layers = len(idxs)
        else:
            raise ValueError(f"unknown kv_capture_strategy: {strategy!r}")

        if len(self.capture_indices) != self.num_dit_layers:
            raise ValueError(
                f"kv_capture_strategy={strategy!r} produced {len(self.capture_indices)} "
                f"indices but num_dit_layers={self.num_dit_layers} "
                f"(VLM has {self.num_vlm_layers} layers)."
            )
        self._capture_set = set(self.capture_indices)
        print(f"DiT: {self.num_dit_layers} layers, kv_capture_strategy={strategy!r}, "
              f"sourcing KV from VLM layers {self.capture_indices}")

        # Robot CNN cross-attention config
        self.use_robot_ca = getattr(config, "use_robot_ca", False)
        if self.use_robot_ca:
            print(f"[wilro] Robot CNN cross-attention ENABLED — action queries will "
                  f"directly attend to high-res Robot CNN features for spatial grounding")

        self.dit_layers = nn.ModuleList([
            DiTLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
                rms_norm_eps=self.rms_norm_eps,
                dropout=config.dropout,
                use_robot_ca=self.use_robot_ca,
            ) for _ in range(self.num_dit_layers)
        ])

        # ── Robot CNN K/V projections for cross-attention ────────────────
        # Projects Robot CNN tokens into K/V format matching the DiT's
        # cross-attention heads. This allows action queries to directly
        # attend to high-resolution spatial features (14x14 grid) instead
        # of only VLM's coarse SigLIP patches (~729 patches).
        if self.use_robot_ca:
            self.robot_ca_k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.robot_ca_v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.robot_ca_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # ─────────────────────────────────────────────────────────────
        # 3. DiT-side embeddings: SINK, state, action, time MLP
        # ─────────────────────────────────────────────────────────────
        self.sink_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.normal_(self.sink_token, std=0.02)

        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, self.hidden_size),
            RMSNorm(self.hidden_size, eps=self.rms_norm_eps),
        )

        self.action_in_proj = nn.Linear(config.action_dim, self.hidden_size)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, config.horizon, self.hidden_size))
        nn.init.normal_(self.action_pos_emb, std=0.02)

        self.final_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.action_out_proj = nn.Linear(self.hidden_size, config.action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

        # Time MLP: sinusoidal → MLP → fed to every DiT layer's adaLN
        self.time_embedder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # ─────────────────────────────────────────────────────────────
        # 4. Robot CNN (optional parallel visual path)
        # ─────────────────────────────────────────────────────────────
        if config.use_robot_cnn:
            self.robot_visual_encoder = RobotVisualEncoder(
                input_size=config.robot_encoder_input_size,
                out_tokens=config.robot_encoder_tokens,
                out_dim=self.hidden_size,
            )
        else:
            self.robot_visual_encoder = None
            print("[wilro] use_robot_cnn=False — RobotVisualEncoder disabled")

        # ─────────────────────────────────────────────────────────────
        # 5. Latent "thought" tokens — task-conditional, zero-init output
        # ─────────────────────────────────────────────────────────────
        self.num_latent_tokens = config.num_latent_tokens
        if self.num_latent_tokens > 0:
            hidden_mid = self.hidden_size * 2
            self.latent_generator = nn.Sequential(
                nn.Linear(self.hidden_size, hidden_mid),
                nn.SiLU(),
                nn.Linear(hidden_mid, self.num_latent_tokens * self.hidden_size),
            )
            # Small std init on the output projection. Zero-init here causes
            # latents to stay exactly 0, with K/V from latent positions also
            # exactly 0, which gives them uniform-share but zero-value
            # attention — gradient back to latent_generator becomes near-zero
            # and the module never wakes up. A small random start breaks the
            # deadlock without destabilising training.
            nn.init.normal_(self.latent_generator[-1].weight, std=0.01)
            nn.init.zeros_(self.latent_generator[-1].bias)

        
        self._lang_max_len = 48
        self.gradient_checkpointing = False

        # Attention-mass diagnostic. When set to True, the next _run_dit call
        # will compute action-query attention mass in the LAST DiT layer for
        # both self-attn (per DiT region) and cross-attn (vision vs language
        # of the VLM KV cache), then flip itself back off and stash both
        # dicts on the model.
        self._capture_attention_stats = False
        self._last_attention_stats: Optional[dict] = None
        self._last_cross_attention_stats: Optional[dict] = None

    # =========================================================================
    # Keep frozen components in eval mode
    # =========================================================================
    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        self.connector.eval()
        self.text_model.eval()
        return self

    def gradient_checkpointing_enable(self):
        """Recompute DiT layer activations during backward instead of storing
        them. Frozen VLM runs in no_grad and is unaffected."""
        self.gradient_checkpointing = True
        print(f"[wilro] DiT gradient checkpointing ENABLED "
              f"({self.num_dit_layers} layers will be recomputed in backward)")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    # =========================================================================
    # Vision / language encoding (no gradient, frozen VLM components)
    # =========================================================================
    def _encode_images(self, batch: dict, B: int) -> torch.Tensor:
        vlm_dtype = next(self.vision_model.parameters()).dtype
        all_vis: list[torch.Tensor] = []
        for cam_key in self.config.cameras_for_vision_state_concat:
            if cam_key not in batch:
                continue
            imgs = batch[cam_key]
            img = imgs[:, -1] if imgs.dim() == 5 else imgs
            img = img * 2.0 - 1.0

            target = self.config.vision_input_size
            h, w = img.shape[-2], img.shape[-1]
            if h != w:
                max_dim = max(h, w)
                pad = (
                    (max_dim - w) // 2, max_dim - w - (max_dim - w) // 2,
                    (max_dim - h) // 2, max_dim - h - (max_dim - h) // 2,
                )
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

    def _encode_language(self, batch: dict, device: torch.device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
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
        lang_mask = inputs["attention_mask"].bool().to(device)
        lang_tokens = self.text_model.get_input_embeddings()(input_ids)
        return lang_tokens, lang_mask

    # =========================================================================
    # VLM encoder: run all layers, cache K/V from the trailing num_dit_layers
    # =========================================================================
    @torch.no_grad()
    def _run_vlm_and_cache_kv(
        self, batch: dict,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, int, int]:
        """
        Returns:
          kv_cache:        list of length num_dit_layers, each entry is (K, V)
                           with shape (B, num_kv_heads, L_vlm, head_dim).
                           K is post-RoPE rotation.
          vlm_kv_pad_mask: (B, L_vlm) bool — True at non-padded positions.
          L_vis:           number of vision tokens.
          L_lang:          number of language tokens.
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        vis_tokens = self._encode_images(batch, B)
        L_vis = vis_tokens.shape[1]

        # Vision token dropout (regularizer). Disabled in eval / sampling.
        vp = float(getattr(self.config, "vision_dropout_prob", 0.0)) if self.training else 0.0
        if self.training and L_vis > 0 and vp > 0.0:
            keep = torch.rand(B, L_vis, device=vis_tokens.device) > vp
            vis_tokens = vis_tokens * keep.unsqueeze(-1).to(vis_tokens.dtype)

        lang_result = self._encode_language(batch, device)
        if lang_result is not None:
            lang_tokens, lang_mask = lang_result
            lang_tokens = lang_tokens.to(vis_tokens.dtype)
            # Zero out pad slots so they contribute no signal to the VLM.
            lang_tokens = torch.where(
                lang_mask.unsqueeze(-1), lang_tokens, torch.zeros_like(lang_tokens),
            )
            L_lang = lang_tokens.shape[1]
        else:
            lang_tokens = None
            lang_mask = None
            L_lang = 0

        parts = [vis_tokens]
        if lang_tokens is not None:
            parts.append(lang_tokens)
        vlm_seq = torch.cat(parts, dim=1).to(torch.bfloat16)
        L_vlm = vlm_seq.shape[1]

        # Key-padding mask: vision always valid; language follows lang_mask.
        if lang_mask is not None:
            vis_mask = torch.ones(B, L_vis, device=device, dtype=torch.bool)
            vlm_kv_pad_mask = torch.cat([vis_mask, lang_mask], dim=1)
        else:
            vlm_kv_pad_mask = torch.ones(B, L_vlm, device=device, dtype=torch.bool)

        # Causal + key-padding mask for VLM self-attention. Vision and language
        # are concatenated, monotonically positioned; SmolVLM2 was pretrained
        # causal so we keep that.
        causal = torch.triu(
            torch.full((L_vlm, L_vlm), float("-inf"), device=device, dtype=vlm_seq.dtype),
            diagonal=1,
        )
        full_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L_vlm, L_vlm).clone()
        key_pad = ~vlm_kv_pad_mask
        full_mask.masked_fill_(key_pad.unsqueeze(1).unsqueeze(1), float("-inf"))

        # RoPE cache for the full VLM sequence (positions 0..L_vlm-1).
        cos, sin = _build_rope_cache(
            L_vlm, self.head_dim, self.rope_theta, device, vlm_seq.dtype,
        )

        # Layer-by-layer forward, capturing K/V from the selected layers.
        hidden = vlm_seq
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.text_model.layers):
            residual = hidden
            h_in = layer.input_layernorm(hidden)

            Q = layer.self_attn.q_proj(h_in)
            K = layer.self_attn.k_proj(h_in)
            V = layer.self_attn.v_proj(h_in)

            Bn, Ln, _ = Q.shape
            Q = Q.view(Bn, Ln, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(Bn, Ln, self.num_kv_heads, self.head_dim).transpose(1, 2)
            V = V.view(Bn, Ln, self.num_kv_heads, self.head_dim).transpose(1, 2)

            Q, K = _apply_rope(Q, K, cos, sin)

            # Capture post-RoPE K and V for the DiT cross-attn memory.
            if i in self._capture_set:
                kv_cache.append((K.detach(), V.detach()))

            if self.num_kv_heads != self.num_heads:
                r = self.num_heads // self.num_kv_heads
                K_x = K.repeat_interleave(r, dim=1)
                V_x = V.repeat_interleave(r, dim=1)
            else:
                K_x, V_x = K, V

            attn = F.scaled_dot_product_attention(Q, K_x, V_x, attn_mask=full_mask, is_causal=False)
            attn = attn.transpose(1, 2).contiguous().view(Bn, Ln, self.num_heads * self.head_dim)
            attn = layer.self_attn.o_proj(attn)
            hidden = residual + attn

            residual = hidden
            h_in = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(h_in)

        # Extract VLM-processed language embeddings from the final hidden state.
        # These are used as DiT sequence tokens so robot/action can self-attend
        # to language directly (language grounding for Robot CNN features).
        lang_embeddings = None
        if L_lang > 0:
            lang_embeddings = hidden[:, L_vis:L_vis + L_lang].detach()

        return kv_cache, vlm_kv_pad_mask, L_vis, L_lang, lang_embeddings

    # =========================================================================
    # DiT-side helpers: robot CNN, latents, time, input assembly
    # =========================================================================
    def _compute_robot_tokens(self, batch: dict) -> Optional[torch.Tensor]:
        if self.robot_visual_encoder is None:
            return None
        toks_list = []
        for cam_key in self.config.cameras_for_vision_state_concat:
            if cam_key not in batch:
                continue
            img = batch[cam_key]
            if img.dim() == 5:
                img = img[:, -1]
            n_tok = (
                self.config.gripper_encoder_tokens
                if cam_key == self.config.gripper_camera
                else self.config.robot_encoder_tokens
            )
            toks_list.append(self.robot_visual_encoder(img.float(), out_tokens=n_tok))
        if not toks_list:
            return None
        toks = torch.cat(toks_list, dim=1)
        vp = float(getattr(self.config, "vision_dropout_prob", 0.0)) if self.training else 0.0
        if vp > 0:
            B, R, _ = toks.shape
            keep = torch.rand(B, R, device=toks.device) > vp
            toks = toks * keep.unsqueeze(-1).to(toks.dtype)
        return toks

    def _generate_latents(
        self, batch: dict, B: int, device: torch.device, dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.num_latent_tokens == 0:
            return None
        lang_result = self._encode_language(batch, device)
        if lang_result is not None:
            lang_tokens, lang_mask = lang_result
            mask_f = lang_mask.float().unsqueeze(-1).to(lang_tokens.dtype)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = (lang_tokens * mask_f).sum(dim=1) / denom
        else:
            pooled = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        flat = self.latent_generator(pooled.float())
        return flat.view(B, self.num_latent_tokens, self.hidden_size).to(dtype)

    def _build_dit_input(
        self,
        batch: dict,
        noisy_actions: torch.Tensor,
        robot_tokens: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
        action_prefix: Optional[torch.Tensor],
        lang_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int, int, int, int]:
        """
        Layout: [SINK, (latent?), state, (language?), (action_prefix?), (robot?), action]

        Language tokens (from VLM's final hidden state) are inserted after state
        so that robot and action tokens can self-attend to language directly.
        This provides language grounding for Robot CNN features — robot tokens
        learn to condition on the task instruction through self-attention.

        Latents sit right after SINK so that every downstream module (state
        encoding, prefix, robot CNN, action) can self-attend to them under the
        causal mask. This broadcasts the pooled-language task representation
        to the visual / motor path, enabling top-down (language → vision)
        conditioning rather than the previous layout where only `action`
        could read latents.

        Returns:
          dit_seq:           (B, L_dit, H)
          action_start_idx:  index where noisy action tokens begin (for readout)
          prefix_start_idx:  index where the clean action prefix begins (-1 if none)
          latent_start_idx:  index where latent tokens begin (-1 if none)
          lang_start_idx:    index where language tokens begin (-1 if none)
        """
        B, H, _ = noisy_actions.shape
        dtype = noisy_actions.dtype

        sink = self.sink_token.expand(B, -1, -1).to(dtype)

        state = batch["observation.state"].float()
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state = state.nan_to_num(0.0).clamp(-10.0, 10.0)
        state_tok = self.state_encoder(state).to(dtype)
        if state_tok.shape[1] > 1:
            state_tok = state_tok[:, -1:]

        action_emb = self.action_in_proj(noisy_actions) + self.action_pos_emb[:, :H]
        action_emb = action_emb.to(dtype)

        # Layout: [SINK, (latent?), state, (language?), (prefix?), (robot?), action]
        parts = [sink]
        latent_start_idx = -1
        if latents is not None:
            latent_start_idx = sum(p.size(1) for p in parts)
            parts.append(latents.to(dtype))
        parts.append(state_tok)

        # Language tokens from VLM (after all layers) — enables robot/action
        # to self-attend to language for direct language grounding.
        lang_start_idx = -1
        if lang_tokens is not None and lang_tokens.shape[1] > 0:
            lang_start_idx = sum(p.size(1) for p in parts)
            parts.append(lang_tokens.to(dtype))

        prefix_start_idx = -1
        if action_prefix is not None and action_prefix.shape[1] > 0:
            prefix_start_idx = sum(p.size(1) for p in parts)
            # Detach: prefix is treated as conditioning, not a target.
            prefix_emb = self.action_in_proj(action_prefix.detach()).to(dtype)
            parts.append(prefix_emb)
        if robot_tokens is not None:
            parts.append(robot_tokens.to(dtype))

        action_start_idx = sum(p.size(1) for p in parts)
        parts.append(action_emb)
        seq = torch.cat(parts, dim=1)
        return seq, action_start_idx, prefix_start_idx, latent_start_idx, lang_start_idx

    def _build_dit_self_attn_mask(
        self,
        L_dit: int,
        action_start_idx: int,
        action_prefix_len: int,
        prefix_start_idx: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Full left-to-right causal mask, with optional Λ-shape suppression:

        When a clean action prefix is present, only the first
        `lambda_mask_window` noisy actions may attend to the prefix slot. Later
        noisy actions are blocked from the prefix, forcing them to rely on
        vision/language via cross-attn rather than copying nearby clean steps.
        """
        mask = torch.triu(
            torch.full((L_dit, L_dit), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        window = int(getattr(self.config, "lambda_mask_window", 0))
        if action_prefix_len > 0 and window > 0 and prefix_start_idx >= 0:
            H = self.config.horizon
            prefix_e = prefix_start_idx + action_prefix_len
            block_from = action_start_idx + window
            block_to = action_start_idx + H
            if block_from < block_to:
                mask[block_from:block_to, prefix_start_idx:prefix_e] = float("-inf")
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

    # =========================================================================
    # Diagnostic: per-region attention mass from action queries
    # =========================================================================
    @torch.no_grad()
    def _compute_cross_attention_mass(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        kv_last: tuple[torch.Tensor, torch.Tensor],
        vlm_kv_pad_mask: torch.Tensor,
        action_start: int,
        action_len: int,
        L_vis: int,
        L_lang: int,
    ) -> dict[str, float]:
        """Re-run the LAST DiT layer's cross-attn Q·K^T softmax manually and
        report mass that action queries place on vision vs language portions
        of the VLM KV cache.

        x is the input to the last DiT layer (pre-self-attn). The "real"
        cross-attn input is x + sa_gate · sa_out, but at training-init the
        gate is small and x dominates — close enough for a diagnostic.
        """
        if action_len <= 0:
            return {}
        layer = self.dit_layers[-1]
        mod = layer.adaLN_modulation(t_emb)
        # Chunk count depends on use_robot_ca: 12 (4 sublayers × 3) or 9 (3 × 3)
        n_chunks = 12 if layer.use_robot_ca else 9
        chunks = mod.chunk(n_chunks, dim=-1)
        s_ca, sc_ca = chunks[3], chunks[4]
        h = _modulate(layer.ca_norm(x), s_ca, sc_ca)

        B, _, _ = h.shape
        H, Hk, D = layer.num_heads, layer.num_kv_heads, layer.head_dim
        Q_full = layer.ca_q(h).view(B, -1, H, D).transpose(1, 2).float()
        Q = Q_full[:, :, action_start:action_start + action_len, :]   # only action rows

        K = kv_last[0].float()                                          # (B, Hk, L_vlm, D)
        if Hk != H:
            K = K.repeat_interleave(H // Hk, dim=1)

        scale = 1.0 / math.sqrt(D)
        scores = (Q @ K.transpose(-1, -2)) * scale                      # (B, H, a_len, L_vlm)

        if vlm_kv_pad_mask is not None:
            kpad = ~vlm_kv_pad_mask
            ca_mask = torch.zeros(
                B, 1, 1, vlm_kv_pad_mask.shape[-1],
                device=scores.device, dtype=scores.dtype,
            )
            ca_mask.masked_fill_(kpad.unsqueeze(1).unsqueeze(1), float("-inf"))
            scores = scores + ca_mask

        weights = torch.softmax(scores, dim=-1)

        stats: dict[str, float] = {}
        if L_vis > 0:
            stats["vision"] = weights[:, :, :, :L_vis].sum(dim=-1).mean().item()
        if L_lang > 0:
            stats["language"] = weights[:, :, :, L_vis:L_vis + L_lang].sum(dim=-1).mean().item()
        return stats

    @torch.no_grad()
    def _compute_attention_mass(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        regions: dict[str, tuple[int, int]],
    ) -> dict[str, float]:
        """Re-run the LAST DiT layer's self-attn Q·K^T softmax manually and
        report, for each region (sink/state/prefix/robot/latent/prev_actions),
        the average attention mass that the action queries place on it.

        SDPA doesn't expose softmax weights, so this re-projects Q/K once;
        cost ≈ one extra layer's self-attn projection.
        """
        layer = self.dit_layers[-1]
        mod = layer.adaLN_modulation(t_emb)
        # Chunk count depends on use_robot_ca: 12 (4 sublayers × 3) or 9 (3 × 3)
        n_chunks = 12 if layer.use_robot_ca else 9
        chunks = mod.chunk(n_chunks, dim=-1)
        s_sa, sc_sa = chunks[0], chunks[1]
        h = _modulate(layer.sa_norm(x), s_sa, sc_sa)

        B, L, _ = h.shape
        H, Hk, D = layer.num_heads, layer.num_kv_heads, layer.head_dim
        Q = layer.sa_q(h).view(B, L, H, D).transpose(1, 2).float()
        K = layer.sa_k(h).view(B, L, Hk, D).transpose(1, 2).float()
        if Hk != H:
            K = K.repeat_interleave(H // Hk, dim=1)

        scale = 1.0 / math.sqrt(D)
        scores = (Q @ K.transpose(-1, -2)) * scale          # (B, H, L, L)
        if attn_mask is not None:
            scores = scores + attn_mask.float()
        weights = torch.softmax(scores, dim=-1)

        a_start, a_len = regions["action"]
        if a_len <= 0:
            return {}
        action_w = weights[:, :, a_start:a_start + a_len, :]  # (B, H, a_len, L)

        stats: dict[str, float] = {}
        for name, span in regions.items():
            if span is None:
                continue
            start, length = span
            if length <= 0:
                continue
            mass = action_w[:, :, :, start:start + length].sum(dim=-1).mean().item()
            stats[name] = mass
        return stats

    # =========================================================================
    # DiT decoder pass — one denoising step given pre-computed VLM KV cache
    # =========================================================================
    def _run_dit(
        self,
        batch: dict,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_kv_pad_mask: torch.Tensor,
        robot_tokens: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
        action_prefix: Optional[torch.Tensor],
        lang_tokens: Optional[torch.Tensor] = None,
        L_vis: int = 0,
        L_lang: int = 0,
    ) -> torch.Tensor:
        """One DiT denoising step. Returns velocity prediction (B, H, action_dim)."""
        device = noisy_actions.device
        dtype = noisy_actions.dtype

        # Time embedding
        t_emb_raw = create_sinusoidal_pos_embedding(timesteps, self.hidden_size).to(dtype)
        t_emb = self.time_embedder(t_emb_raw.float()).to(dtype)

        # Build sequence (lang_tokens injected into DiT for language grounding)
        dit_seq, action_start_idx, prefix_start_idx, latent_start_idx, lang_start_idx = self._build_dit_input(
            batch, noisy_actions, robot_tokens, latents, action_prefix, lang_tokens,
        )
        L_dit = dit_seq.shape[1]
        prefix_len = action_prefix.shape[1] if action_prefix is not None else 0
        attn_mask = self._build_dit_self_attn_mask(
            L_dit, action_start_idx, prefix_len, prefix_start_idx, device, dtype,
        )

        # Region boundaries (used by attention-mass diagnostic). Layout:
        #   [SINK(1), (latent(K))?, state(1), (language(L))?, (prefix(P))?, robot(R), action(H)]
        robot_len = robot_tokens.shape[1] if robot_tokens is not None else 0
        latent_len = latents.shape[1] if latents is not None else 0
        lang_len = lang_tokens.shape[1] if lang_tokens is not None else 0
        H_horizon = self.config.horizon
        # state always sits right after sink + optional latent block
        state_idx = 1 + latent_len
        # robot sits right before action
        robot_idx = action_start_idx - robot_len
        regions: dict[str, Optional[tuple[int, int]]] = {
            "sink":         (0, 1),
            "latent":       (latent_start_idx, latent_len) if latent_len > 0 else None,
            "state":        (state_idx, 1),
            "language":     (lang_start_idx, lang_len) if lang_len > 0 else None,
            "prefix":       (prefix_start_idx, prefix_len) if prefix_len > 0 else None,
            "robot":        (robot_idx, robot_len),
            "action":       (action_start_idx, H_horizon),
        }

        # ── Robot CNN K/V projections for cross-attention ────────────────
        # Project robot tokens into K/V format matching DiT's cross-attention
        # heads. This enables action queries to directly attend to high-res
        # spatial features (14x14 grid) for precise object localization.
        robot_k, robot_v = None, None
        if self.use_robot_ca and robot_tokens is not None:
            robot_normed = self.robot_ca_norm(robot_tokens)
            B_r, R, _ = robot_normed.shape
            robot_k = self.robot_ca_k_proj(robot_normed).view(
                B_r, R, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)  # (B, num_kv_heads, R, head_dim)
            robot_v = self.robot_ca_v_proj(robot_normed).view(
                B_r, R, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)

        x = dit_seq
        use_ckpt = self.gradient_checkpointing and self.training
        for i, layer in enumerate(self.dit_layers):
            # Capture attention mass at the LAST DiT layer's input. This is
            # the "final say" before the action readout — what action queries
            # decide to attend to here is what shapes the velocity output.
            if (
                i == len(self.dit_layers) - 1
                and self._capture_attention_stats
            ):
                self._last_attention_stats = self._compute_attention_mass(
                    x, t_emb, attn_mask, regions,
                )
                self._last_cross_attention_stats = self._compute_cross_attention_mass(
                    x, t_emb, kv_cache[i], vlm_kv_pad_mask,
                    action_start_idx, H_horizon, L_vis, L_lang,
                )
                # one-shot: don't capture again if _run_dit is called twice
                # (e.g. for the contrastive-language v_wrong forward).
                self._capture_attention_stats = False
                # The no_grad capture above populated the autocast weight
                # cache with GRAD-LESS bf16 casts of this layer's sa_q/sa_k/
                # ca_q/adaLN weights. Clear it so the real forward below
                # re-casts them WITH grad tracking — otherwise those weights
                # silently receive no gradient on every capture step.
                torch.clear_autocast_cache()

            vlm_k, vlm_v = kv_cache[i]
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, t_emb, vlm_k, vlm_v, vlm_kv_pad_mask, attn_mask,
                    robot_k, robot_v,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x, t_emb=t_emb,
                    vlm_k=vlm_k, vlm_v=vlm_v,
                    vlm_kv_pad_mask=vlm_kv_pad_mask,
                    self_attn_mask=attn_mask,
                    robot_k=robot_k, robot_v=robot_v,
                )

        H = self.config.horizon
        action_out = self.final_norm(x[:, action_start_idx:action_start_idx + H])
        return self.action_out_proj(action_out)

    # =========================================================================
    # Flow matching training / sampling
    # =========================================================================
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
        # "lognormal": SD3-style logit-normal t = sigmoid(N(mean, std)). A negative
        # mean biases toward LOW t (= x_t≈actions), spending more capacity on the
        # fine-detail denoising that sets placement precision. Default "uniform".
        if getattr(self.config, "time_sampling", "uniform") == "lognormal":
            z = (torch.randn(B, device=device) * self.config.time_lognormal_std
                 + self.config.time_lognormal_mean)
            return torch.sigmoid(z).clamp(0.001, 0.999)
        t = torch.rand(B, device=device)
        return t * 0.998 + 0.001

    def compute_loss(self, batch: dict) -> torch.Tensor:
        actions = batch["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0)
        B = actions.shape[0]
        device = actions.device

        # ── Encoder: run VLM once, cache KV + extract lang embeddings ──
        kv_cache, vlm_kv_pad_mask, L_vis, L_lang, lang_embeddings = self._run_vlm_and_cache_kv(batch)

        # ── DiT-side conditioning that does NOT depend on noise ─────
        robot_tokens = self._compute_robot_tokens(batch)
        latents = self._generate_latents(batch, B, device, torch.bfloat16)

        # ── Action prefix for async execution training ──────────────
        action_prefix = None
        max_prefix = int(getattr(self.config, "max_action_prefix_steps", 0))
        if self.training and max_prefix > 0:
            prefix_len = torch.randint(0, max_prefix + 1, (1,), device=device).item()
            if prefix_len > 0:
                action_prefix = actions[:, :prefix_len]

        # ── Flow matching: build noisy actions, predict velocity ────
        noise = self.sample_noise(actions.shape, device)
        t = self.sample_time(B, device)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions
        u_t = noise - actions

        v_t = self._run_dit(
            batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
            robot_tokens, latents, action_prefix, lang_embeddings,
            L_vis=L_vis, L_lang=L_lang,
        ).float()

        # ── Per-position / per-dim weighting ────────────────────────
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
        # Combined per-(B, H, 1) weight: chunk-position weight × optional gripper
        # phase weight. Built once and reused in the denominator so the loss stays
        # a weighted MEAN (reweight, not rescale) — effective LR is unchanged.
        w_pos = pos_w[None, :, None].expand(loss.shape[0], H, 1).clone()

        gpw = float(getattr(self.config, "gripper_phase_weight", 1.0))
        if gpw != 1.0:
            # Up-weight frames near a gripper open<->close transition (grasp /
            # release): |Δgripper| over the chunk, dilated to a ±window, scaled to
            # gpw. Off-window frames keep weight 1.0.
            gidx = getattr(self.config, "gripper_action_index", -1)
            g = actions[:, :, gidx]                                      # (B, H)
            dg = torch.zeros_like(g)
            dg[:, 1:] = (g[:, 1:] - g[:, :-1]).abs()
            trans = (dg > self.config.gripper_transition_thresh).to(loss.dtype)
            win = int(getattr(self.config, "gripper_transition_window", 2))
            if win > 0:
                trans = F.max_pool1d(trans.unsqueeze(1), 2 * win + 1,
                                     stride=1, padding=win).squeeze(1)
            phase_w = 1.0 + (gpw - 1.0) * trans                          # (B, H)
            w_pos = w_pos * phase_w[:, :, None]
        loss = loss * w_pos

        loss_dtype = loss.dtype
        Bn, Hn, Dn = loss.shape

        is_pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
        valid_t = (~is_pad.bool()).to(loss_dtype) if is_pad is not None \
                  else torch.ones(Bn, Hn, device=loss.device, dtype=loss_dtype)

        dim_pad = batch.get("action_dim_pad")
        valid_d = (~dim_pad.bool()).to(loss_dtype) if dim_pad is not None \
                  else torch.ones(Bn, Dn, device=loss.device, dtype=loss_dtype)

        valid_cells = valid_t.unsqueeze(-1) * valid_d.unsqueeze(1)
        loss = loss * valid_cells
        denom = (w_pos * valid_cells).sum().clamp(min=1e-6)
        main_loss = loss.sum() / denom

        # ── Contrastive language loss: permute the LANGUAGE portion of
        # the cached KV across batch and re-run only the DiT. Avoids a
        # second full VLM forward.
        contrastive_w = float(getattr(self.config, "contrastive_loss_weight", 0.0))
        contrastive_v = 0.0
        if (
            self.training and contrastive_w > 0.0
            and L_lang > 0 and B >= 2
        ):
            # Prefer task_description (may be rewritten for spatial grounding)
            # over raw task string from dataset.
            descs = batch.get("task_description") or batch.get("task")
            use_hard_neg = getattr(self.config, "contrastive_hard_negatives", False)

            if (
                use_hard_neg
                and descs is not None and len(descs) == B
            ):
                perm, pair_diff = _hard_negative_perm(descs, device)
            else:
                perm = torch.randperm(B, device=device)
                if (perm == torch.arange(B, device=device)).any():
                    perm = torch.roll(perm, shifts=1, dims=0)

                if descs is not None and len(descs) == B:
                    perm_cpu = perm.detach().cpu().tolist()
                    pair_diff = torch.tensor(
                        [descs[i] != descs[perm_cpu[i]] for i in range(B)],
                        device=device, dtype=torch.bool,
                    )
                else:
                    pair_diff = torch.ones(B, device=device, dtype=torch.bool)

            if pair_diff.any():
                # Treat the wrong-language prediction as a FIXED negative
                # target (stop-gradient). Building the shuffled KV cache and
                # running the second DiT forward under no_grad frees the
                # activations immediately instead of storing a full 309M-param
                # backward graph — this removes the ~2x memory blow-up. The
                # contrastive gradient still flows through v_t, pushing the
                # correct-language prediction away from the (detached) wrong one.
                with torch.no_grad():
                    shuffled_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
                    for K, V in kv_cache:
                        K_shuf = K.clone()
                        V_shuf = V.clone()
                        K_shuf[:, :, L_vis:L_vis + L_lang, :] = K[perm, :, L_vis:L_vis + L_lang, :]
                        V_shuf[:, :, L_vis:L_vis + L_lang, :] = V[perm, :, L_vis:L_vis + L_lang, :]
                        shuffled_cache.append((K_shuf, V_shuf))
                    shuffled_pad_mask = vlm_kv_pad_mask.clone()
                    shuffled_pad_mask[:, L_vis:L_vis + L_lang] = vlm_kv_pad_mask[perm][:, L_vis:L_vis + L_lang]
                    # Shuffle lang_embeddings for the contrastive forward
                    shuffled_lang = lang_embeddings[perm] if lang_embeddings is not None else None

                    v_wrong = self._run_dit(
                        batch, x_t.to(torch.bfloat16), t,
                        shuffled_cache, shuffled_pad_mask,
                        robot_tokens, latents, action_prefix, shuffled_lang,
                        L_vis=L_vis, L_lang=L_lang,
                    ).float()

                diff_sq = (v_t - v_wrong).pow(2).mean(dim=[1, 2])
                margin = float(getattr(self.config, "contrastive_margin", 0.05))
                hinge = F.relu(margin - diff_sq) * pair_diff.float()
                n_valid = pair_diff.float().sum().clamp(min=1.0)
                loss_contrastive = hinge.sum() / n_valid
                contrastive_v = float(loss_contrastive.detach())
                main_loss = main_loss + contrastive_w * loss_contrastive

        self._last_loss_components = {
            "main": float(main_loss.detach() - contrastive_w * contrastive_v),
            "contrastive": contrastive_v,
        }
        return main_loss

    def forward(self, batch: dict) -> tuple:
        if self.training:
            return self.compute_loss(batch), {}
        return self.sample_actions(batch), {}

    def flow_actions_from_noise(self, batch: dict, x_init: torch.Tensor) -> torch.Tensor:
        """Deterministic flow ODE solution from a GIVEN initial noise x_init
        (B, horizon, action_dim), returned for the FULL horizon in normalized
        action space. Differentiable through the DiT / robot CNN / latent
        generator (the frozen VLM encoder still runs under no_grad inside
        _run_vlm_and_cache_kv).

        Used by RL (GRPO): the policy is N(flow_actions_from_noise(s, x1), sigma^2)
        conditioned on the stored noise latent x1, so action log-probs are exact
        and importance ratios are computable. Mirrors sample_actions' integration
        exactly.

        NOTE: Wraps the DiT loop in autocast so the DiT linear layers (stored in
        fp32) accept bf16 noisy_actions without dtype mismatch. Caller controls
        grad/no_grad context.
        """
        B = x_init.shape[0]
        device = x_init.device

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )

        with autocast_ctx:
            kv_cache, vlm_kv_pad_mask, L_vis, L_lang, lang_embeddings = self._run_vlm_and_cache_kv(batch)
            robot_tokens = self._compute_robot_tokens(batch)
            latents = self._generate_latents(batch, B, device, torch.bfloat16)

            N = int(getattr(self.config, "num_inference_steps", 10))
            x_t = x_init.float()
            dt = -1.0 / N
            t = torch.ones(B, device=device, dtype=torch.float32)
            for _ in range(N):
                v_t = self._run_dit(
                    batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
                    robot_tokens, latents, action_prefix=None, lang_tokens=lang_embeddings,
                    L_vis=L_vis, L_lang=L_lang,
                ).float()
                x_t = x_t + dt * v_t
                t = t + dt
        return x_t

    @torch.no_grad()
    def sample_actions(self, batch: dict) -> torch.Tensor:
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )

        with autocast_ctx:
            kv_cache, vlm_kv_pad_mask, L_vis, L_lang, lang_embeddings = self._run_vlm_and_cache_kv(batch)
            robot_tokens = self._compute_robot_tokens(batch)
            latents = self._generate_latents(batch, B, device, torch.bfloat16)

            N = int(getattr(self.config, "num_inference_steps", 10))
            x_t = self.sample_noise(
                (B, self.config.horizon, self.config.action_dim), device=device,
            )
            dt = -1.0 / N
            t = torch.ones(B, device=device, dtype=torch.float32)

            for _ in range(N):
                v_t = self._run_dit(
                    batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
                    robot_tokens, latents, action_prefix=None, lang_tokens=lang_embeddings,
                    L_vis=L_vis, L_lang=L_lang,
                ).float()
                x_t = x_t + dt * v_t
                t = t + dt

        return x_t[:, : self.config.n_action_steps]

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
