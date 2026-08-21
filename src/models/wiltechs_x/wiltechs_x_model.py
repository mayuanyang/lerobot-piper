"""WiltechsX — joint-attention (Mixture-of-Transformers) VLA.

See ARCHITECTURE.md. The short version of the layout:

    [ PREFIX: lang | vision | wrist | motion ]  [ SUFFIX: state | reg | x_t ]
      Qwen3-VL weights + LoRA, bidirectional      expert weights, causal
      trained by the discrete action head         trained by flow matching
                        ^                                   |
                        +--------- stop-grad ---------------+

One attention per layer over the concatenation, with per-segment weights. The
prefix->suffix block is masked, so the prefix is a function of the observation
ONLY -- which is what lets sampling compute it once and then run the denoising
loop against a cached K/V (`_run_prefix` / `forward_cached`).

INTERFACES BORROWED FROM WORKING CODE. `_apply_rope`,
`_build_mrope_position_ids`, `preprocess_camera_to_pixels` and the vlm_*_key
helpers are imported from wiltechs_vla_model rather than reimplemented: that
module runs, so its Qwen3-VL interface assumptions are the verified ones. The
dependency is deliberate and one-way.

DEVIATIONS FROM THE DESIGN DOC, stated up front:
  - The discrete head on the VLM side uses uniform per-dimension BINNING
    (RT-2/OpenVLA style) predicted in parallel, not FAST's DCT+BPE tokens with
    autoregressive decoding. Knowledge insulation needs *a* discrete token
    objective on the VLM side; it does not specifically need FAST. This version
    has no extra dependency and no download. `fast_tokenizer_id` is unused.
  - `flow_objective="meanflow"` raises. "flow" and "shortcut" are implemented.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# `transformers` is imported lazily inside WiltechsXModel.__init__ and
# WristTokenizer, so the pure-torch components (and test_components.py) can be
# used without it.
from ..interleaved_flow_matching.expert_layer import RMSNorm, SwiGLU
from ..wiltechs_vla.wiltechs_vla_model import (
    _apply_rope,
    _build_mrope_position_ids,
    preprocess_camera_to_pixels,
    vlm_grid_key,
    vlm_pixels_key,
)
from .wiltechs_x_config import WiltechsXConfig

try:                                     # optional, only for the rewrites
    from ..wiltechs_vla.task_rewrites import rewrite_instruction
except Exception:                        # pragma: no cover
    def rewrite_instruction(s):
        return s


# =========================================================================
# LoRA (hand-rolled: no peft dependency, and the wrap must be transparent
# to JointExpertLayer, which calls vlm_layer.self_attn.q_proj directly)
# =========================================================================
class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update. B is zero-init, so the
    wrapped module is an exact identity at step 0."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.dropout(x).to(self.lora_a.dtype)
        delta = F.linear(F.linear(h, self.lora_a), self.lora_b) * self.scaling
        return out + delta.to(out.dtype)


def attach_lora(module: nn.Module, targets: list[str], rank: int, alpha: int,
                dropout: float = 0.0) -> int:
    """Replace every child Linear whose attribute name is in `targets`.
    Returns the number of wrapped modules."""
    n = 0
    for child in module.modules():
        for name in targets:
            sub = getattr(child, name, None)
            if isinstance(sub, nn.Linear):
                setattr(child, name, LoRALinear(sub, rank, alpha, dropout))
                n += 1
    return n


# =========================================================================
# Attention mask geometry
# =========================================================================
def build_joint_attn_mask(
    L_prefix: int,
    L_suffix: int,
    prefix_pad_mask: torch.Tensor,
    bidirectional_prefix: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive (B, 1, L, L) mask for the joint [prefix | suffix] sequence.

    The REFERENCE definition of the architecture's mask geometry. Production
    builds the two halves separately (`_run_prefix` and
    `build_suffix_attn_mask`) to avoid materialising an (L_p+L_s)^2 tensor per
    layer; test_components.py asserts the halves agree with this.

        prefix -> prefix   full if bidirectional_prefix else causal
        prefix -> suffix   MASKED. The prefix must not depend on the noise
                           level, or it would have to be recomputed at every
                           denoising step. This block is what makes the VLM a
                           once-per-chunk cost -- breaking it does not raise,
                           it just makes rollouts N times slower, so
                           test_components.py checks it mechanically.
        suffix -> prefix   full
        suffix -> suffix   causal
    """
    B = prefix_pad_mask.shape[0]
    device = prefix_pad_mask.device
    L = L_prefix + L_suffix
    neg = torch.finfo(dtype).min

    mask = torch.full((L, L), neg, device=device, dtype=dtype)
    if bidirectional_prefix:
        mask[:L_prefix, :L_prefix] = 0.0
    else:
        mask[:L_prefix, :L_prefix] = torch.triu(
            torch.full((L_prefix, L_prefix), neg, device=device, dtype=dtype), diagonal=1)
    mask[L_prefix:, :L_prefix] = 0.0
    mask[L_prefix:, L_prefix:] = torch.triu(
        torch.full((L_suffix, L_suffix), neg, device=device, dtype=dtype), diagonal=1)

    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).clone()
    key_pad = torch.zeros(B, L, device=device, dtype=torch.bool)
    key_pad[:, :L_prefix] = ~prefix_pad_mask
    mask.masked_fill_(key_pad.unsqueeze(1).unsqueeze(1), neg)
    return mask


def build_suffix_attn_mask(
    L_prefix: int, L_suffix: int, prefix_pad_mask: torch.Tensor, dtype: torch.dtype,
) -> torch.Tensor:
    """(B, 1, L_suffix, L_prefix + L_suffix) — the suffix rows of the joint
    mask. Used by the cached sampling path."""
    B = prefix_pad_mask.shape[0]
    device = prefix_pad_mask.device
    neg = torch.finfo(dtype).min
    m = torch.zeros(B, 1, L_suffix, L_prefix + L_suffix, device=device, dtype=dtype)
    m[..., L_prefix:] = torch.triu(
        torch.full((L_suffix, L_suffix), neg, device=device, dtype=dtype), diagonal=1)
    m[..., :L_prefix].masked_fill_(
        (~prefix_pad_mask).unsqueeze(1).unsqueeze(1), neg)
    return m


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, L, D) -> (B, n_kv * n_rep, L, D) for GQA."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


# =========================================================================
# Joint expert layer — the architectural core
# =========================================================================
class JointExpertLayer(nn.Module):
    """One transformer layer over [prefix | suffix] with per-segment weights.

    The prefix half reuses the VLM's own layer (passed at forward time, so the
    weights are not duplicated); the suffix half uses this module's own,
    fully-trainable projections. Q/K/V are concatenated and a SINGLE
    scaled-dot-product attention runs over the whole sequence.

    That single attention is the point. WiltechsVLA spent ~31% of its decoder
    parameters on a cross-attention module bridging into the frozen VLM's head
    geometry, and still had to answer "which layer's KV does the decoder read"
    -- a question `vlm_capture_mode="spread"` answered badly (39.8 -> 13.3).
    Here there is no bridge and no choice.

    Head geometry is shared with the VLM (Q and K meet in one dot product).
    The expert's WIDTH is free: it projects into the shared head space and back.
    """

    def __init__(self, expert_hidden: int, expert_intermediate: int,
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 rms_norm_eps: float = 1e-6, ada_rank: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads

        self.attn_norm = RMSNorm(expert_hidden, eps=rms_norm_eps)
        self.q_proj = nn.Linear(expert_hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(expert_hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(expert_hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, expert_hidden, bias=False)

        self.ffn_norm = RMSNorm(expert_hidden, eps=rms_norm_eps)
        self.ffn = SwiGLU(expert_hidden, expert_intermediate)

        # adaLN-Zero on the suffix only: the flow time modulates the action
        # tokens, never the prefix. Zero-init on the LAST layer makes each
        # block an identity at step 0, which keeps a fresh expert from
        # wrecking the pretrained prefix representation before it has learned
        # anything.
        #
        # LOW RANK, and not for elegance. A plain Linear(d, 6d) is d*6d
        # parameters: at d=1024 that is 6.3M per layer, 226M over 36 layers,
        # 32% of the entire expert -- spent on six vectors per layer. The
        # factorisation is 7*d*r, i.e. 465K at r=64, and it was the difference
        # between OOM and fitting on a 22 GiB card. ada_rank<=0 restores the
        # full-rank form.
        if ada_rank and ada_rank > 0:
            self.ada = nn.Sequential(
                nn.SiLU(),
                nn.Linear(expert_hidden, min(ada_rank, expert_hidden), bias=False),
                nn.Linear(min(ada_rank, expert_hidden), 6 * expert_hidden))
        else:
            self.ada = nn.Sequential(nn.SiLU(),
                                     nn.Linear(expert_hidden, 6 * expert_hidden))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _split(self, x, n):
        return x.view(x.shape[0], x.shape[1], n, self.head_dim).transpose(1, 2)

    def prefix_qkv(self, prefix, vlm_layer):
        p_in = vlm_layer.input_layernorm(prefix)
        return (self._split(vlm_layer.self_attn.q_proj(p_in), self.num_heads),
                self._split(vlm_layer.self_attn.k_proj(p_in), self.num_kv_heads),
                self._split(vlm_layer.self_attn.v_proj(p_in), self.num_kv_heads))

    def suffix_qkv(self, suffix, shift, scale):
        s_in = self._modulate(self.attn_norm(suffix), shift, scale)
        return (self._split(self.q_proj(s_in), self.num_heads),
                self._split(self.k_proj(s_in), self.num_kv_heads),
                self._split(self.v_proj(s_in), self.num_kv_heads))

    def _suffix_out(self, suffix, s_attn, gate_a, shift_f, scale_f, gate_f):
        B, L_s = suffix.shape[0], suffix.shape[1]
        # The attention ran in the VLM's dtype (bf16); the expert's own weights
        # may be fp32. Reconcile on the way back into the residual stream.
        s_attn = s_attn.transpose(1, 2).contiguous().view(B, L_s, -1)
        s_attn = s_attn.to(self.o_proj.weight.dtype)
        suffix = suffix + gate_a.unsqueeze(1) * self.o_proj(s_attn).to(suffix.dtype)
        s_ff = self._modulate(self.ffn_norm(suffix), shift_f, scale_f)
        return suffix + gate_f.unsqueeze(1) * self.ffn(s_ff).to(suffix.dtype)

    def forward(self, prefix, suffix, vlm_layer, attn_mask, rope, t_emb,
                stop_grad_to_prefix):
        """REFERENCE implementation of the layer: one attention over the
        concatenated sequence. Returns (prefix_out, suffix_out).

        Not the hot path. Production runs `_run_prefix` + `forward_cached`,
        which computes the same thing without recomputing the prefix at every
        denoising step. This exists as the readable definition of what the
        architecture IS, and test_components.py asserts the fast path agrees
        with it -- if the two ever diverge, the fast path is wrong.
        """
        B, L_p, _ = prefix.shape
        L_s = suffix.shape[1]
        cos, sin = rope
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada(t_emb).chunk(6, -1)

        pq, pk, pv = self.prefix_qkv(prefix, vlm_layer)
        sq, sk, sv = self.suffix_qkv(suffix, shift_a, scale_a)
        # The expert may run in a different dtype from the bf16 VLM; the
        # concatenation is where that has to be reconciled.
        sq, sk, sv = sq.to(pq.dtype), sk.to(pk.dtype), sv.to(pv.dtype)

        q = torch.cat([pq, sq], dim=2)
        k = torch.cat([pk, sk], dim=2)
        v = torch.cat([pv, sv], dim=2)
        if cos is not None:
            q, k = _apply_rope(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep),
            attn_mask=attn_mask.to(q.dtype), is_causal=False)
        p_attn, s_attn = attn[:, :, :L_p], attn[:, :, L_p:]

        # Knowledge insulation: the expert reads the VLM, it does not rewrite
        # it. Cutting the gradient here rather than freezing the VLM is what
        # lets the discrete action head still train the backbone.
        if stop_grad_to_prefix:
            p_attn = p_attn.detach()

        p_attn = p_attn.transpose(1, 2).contiguous().view(B, L_p, -1)
        prefix = prefix + vlm_layer.self_attn.o_proj(p_attn)
        prefix = prefix + vlm_layer.mlp(vlm_layer.post_attention_layernorm(prefix))

        suffix = self._suffix_out(suffix, s_attn, gate_a, shift_f, scale_f, gate_f)
        return prefix, suffix

    def forward_cached(self, suffix, k_pre, v_pre, attn_mask, rope_suffix, t_emb):
        """Sampling path: suffix only, against a prefix K/V computed once.

        Mathematically identical to `forward`'s suffix half -- the prefix
        cannot see the suffix, so its per-layer K/V do not change across
        denoising steps. test_components.py asserts the two agree.
        """
        cos, sin = rope_suffix
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada(t_emb).chunk(6, -1)
        sq, sk, sv = self.suffix_qkv(suffix, shift_a, scale_a)
        sq, sk, sv = sq.to(k_pre.dtype), sk.to(k_pre.dtype), sv.to(v_pre.dtype)
        if cos is not None:
            sq, sk = _apply_rope(sq, sk, cos, sin)

        k = torch.cat([k_pre, sk], dim=2)
        v = torch.cat([v_pre, sv], dim=2)
        s_attn = F.scaled_dot_product_attention(
            sq, _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep),
            attn_mask=attn_mask.to(sq.dtype), is_causal=False)
        return self._suffix_out(suffix, s_attn, gate_a, shift_f, scale_f, gate_f)


# =========================================================================
# Long-horizon helpers
# =========================================================================
class MotionVectorEncoder(nn.Module):
    """Hindsight as low-dimensional motion vectors, not stacked frames.

    HiF-VLA's mechanism, currently the LIBERO-Long SOTA (96.4) at 58% lower
    latency than frame stacking.

    RISK (ARCHITECTURE.md 8.2): this can leak the demonstrator's action and
    reintroduce causal confusion -- the exact failure frame stacking has. The
    control is a motion-vector-ONLY model, which must not score above chance.
    """

    def __init__(self, state_dim: int, history_len: int, n_tokens: int, d_out: int,
                 d_hid: int = 256):
        super().__init__()
        self.history_len = history_len
        self.n_tokens = n_tokens
        self.d_hid = d_hid
        # The mixing Linear runs at d_hid, NOT at the VLM width. Mixing at
        # d_out=2560 makes this one layer (2560*8) x (8*2560) = 419M
        # parameters -- 80% of everything trainable in the model, to encode
        # eight frames of an eight-dimensional proprioceptive vector. Measured
        # on the first real run; the bottleneck brings it to ~5M.
        self.proj = nn.Sequential(nn.Linear(state_dim * 2, d_hid), nn.SiLU(),
                                  nn.Linear(d_hid, d_hid))
        self.to_tokens = nn.Linear(d_hid * history_len, n_tokens * d_hid)
        self.out = nn.Linear(d_hid, d_out)
        self.norm = RMSNorm(d_out)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """(B, T, state_dim) -> (B, n_tokens, d_out). Both the absolute state
        and its first difference go in: velocity alone loses the pose the
        motion happened from."""
        h = state_history[:, -self.history_len:]
        if h.shape[1] < self.history_len:
            pad = h[:, :1].expand(-1, self.history_len - h.shape[1], -1)
            h = torch.cat([pad, h], dim=1)
        delta = torch.cat([h[:, :1] * 0.0, h[:, 1:] - h[:, :-1]], dim=1)
        z = self.proj(torch.cat([h, delta], dim=-1))
        z = self.to_tokens(z.flatten(1)).view(-1, self.n_tokens, self.d_hid)
        return self.norm(self.out(z))


class ProgressHead(nn.Module):
    """Normalized time-to-completion in [0, 1].

    The payoff is in stage B, not stage A: a binary terminal reward on a
    10-stage LIBERO-Long task is the hardest credit-assignment problem in the
    pipeline, and an explicit phase estimate is the cheapest handle on it.
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(d_in), nn.Linear(d_in, max(d_in // 4, 8)),
                                 nn.SiLU(), nn.Linear(max(d_in // 4, 8), 1))

    def forward(self, x):
        return self.net(x).squeeze(-1).sigmoid()


class WristTokenizer(nn.Module):
    """Self-supervised wrist features -> prefix tokens.

    Two departures from the RobotCNN it replaces:
      1. DINO-family self-supervised features rather than a from-scratch CNN.
         Dense spatial correspondence is what the 34-point RobotCNN result was
         buying, and that is the axis self-supervised features win on.
      2. Output goes into the SHARED prefix. The previously observed "reliance
         migrated to the RobotCNN" is a consequence of side-channel placement;
         in the shared sequence these tokens face the same attention
         competition, and the same language conditioning, as everything else.
    """

    # ImageNet statistics -- what every DINO checkpoint was trained under.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: str, n_tokens: int, d_out: int, freeze: bool,
                 input_size: int = 256):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_id)
        self.n_tokens = n_tokens
        self.grid = int(round(n_tokens ** 0.5))
        self.input_size = input_size
        self.proj = nn.Linear(self.backbone.config.hidden_size, d_out)
        self.norm = RMSNorm(d_out)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1), False)
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1), False)
        self.frozen = freeze
        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) float in [0, 1] -> (B, n_tokens, d_out)."""
        if img.shape[-1] != self.input_size or img.shape[-2] != self.input_size:
            img = F.interpolate(img, size=(self.input_size, self.input_size),
                                mode="bilinear", align_corners=False)
        x = (img - self.mean) / self.std
        ctx = torch.no_grad() if self.frozen else nullcontext()
        with ctx:
            out = self.backbone(pixel_values=x).last_hidden_state
        patches = out[:, 1:]                                  # drop CLS
        B, N, D = patches.shape
        side = int(round(N ** 0.5))
        fm = patches.transpose(1, 2).reshape(B, D, side, side)
        fm = F.adaptive_avg_pool2d(fm, (self.grid, self.grid))
        return self.norm(self.proj(fm.flatten(2).transpose(1, 2)))


# =========================================================================
# Discrete action head (VLM side) — knowledge insulation
# =========================================================================
class DiscreteActionHead(nn.Module):
    """Uniform per-dimension binning, predicted in parallel from the prefix.

    Purpose is knowledge insulation, not decoding: the VLM never generates
    from this head. It exists so the backbone receives a CROSS-ENTROPY
    gradient it was pretrained for, instead of the flow-matching regression
    gradient that degrades language grounding.

    NOT FAST. FAST's DCT+BPE tokenization buys sequence-length efficiency for
    autoregressive decoding, which nothing here does. Swapping it in later
    only changes this class.
    """

    def __init__(self, d_in: int, horizon: int, action_dim: int, n_bins: int = 256,
                 clip: float = 3.0):
        super().__init__()
        self.horizon, self.action_dim, self.n_bins, self.clip = \
            horizon, action_dim, n_bins, clip
        # RMSNorm FIRST. Without it the head reads a 36-layer Qwen hidden
        # state whose norm is large, and a default-initialised Linear turns
        # that into logits far off uniform: measured CE 25.5 at init against
        # ln(256) = 5.545, i.e. 20 nats of pure initialisation error on the
        # ONLY gradient path into the VLM.
        self.head = nn.Sequential(
            RMSNorm(d_in),
            nn.Linear(d_in, d_in // 2), nn.SiLU(),
            nn.Linear(d_in // 2, horizon * action_dim * n_bins),
        )
        # Start NEAR uniform, not AT it. Exact zero-init would give
        # dL/d(input) = W^T . dL/d(logits) = 0, so no gradient would reach the
        # VLM on the first step -- and this head is the only path that ever
        # does. Small-but-nonzero keeps the initial CE at ~ln(n_bins) while
        # leaving the backward path open.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, H, A) normalized actions -> (B, H, A) long bin indices."""
        a = actions.clamp(-self.clip, self.clip)
        u = (a + self.clip) / (2 * self.clip)                  # -> [0, 1]
        return (u * (self.n_bins - 1)).round().long().clamp(0, self.n_bins - 1)

    def forward(self, readout: torch.Tensor) -> torch.Tensor:
        """(B, d_in) -> (B, H, A, n_bins)."""
        B = readout.shape[0]
        return self.head(readout).view(B, self.horizon, self.action_dim, self.n_bins)


# =========================================================================
# Top-level model
# =========================================================================
class WiltechsXModel(nn.Module):
    """Joint-attention VLA. See ARCHITECTURE.md."""

    def __init__(self, config: WiltechsXConfig):
        super().__init__()
        self.config = config

        if config.flow_objective == "meanflow":
            raise NotImplementedError(
                "flow_objective='meanflow' is not implemented. 'shortcut' also "
                "gives few-step inference and is implemented and tested; "
                "shipping a plausible-but-unverified mean-flow identity would "
                "be worse than this error."
            )

        # ---- 1. VLM ------------------------------------------------------
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"[wiltechs_x] loading {config.vlm_model_id} ...")
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_id)
        self.vlm_model = vlm.model
        self.visual = self.vlm_model.visual
        self.language_model = self.vlm_model.language_model
        del vlm

        tcfg = self.language_model.config
        self.hidden_size = int(tcfg.hidden_size)
        self.num_heads = int(tcfg.num_attention_heads)
        self.num_kv_heads = int(getattr(tcfg, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(tcfg, "head_dim", None)
                            or (self.hidden_size // self.num_heads))
        self.rms_norm_eps = float(getattr(tcfg, "rms_norm_eps", 1e-5))
        self.num_vlm_layers = len(self.language_model.layers)
        self.spatial_merge_size = int(getattr(
            getattr(self.vlm_model.config, "vision_config", None),
            "spatial_merge_size", 2))
        if not hasattr(self.language_model, "rotary_emb"):
            raise RuntimeError("language_model.rotary_emb not found — the joint "
                               "forward needs Qwen3VLTextRotaryEmbedding there.")
        print(f"[wiltechs_x] VLM: {self.num_vlm_layers}L hidden={self.hidden_size} "
              f"heads={self.num_heads} kv={self.num_kv_heads} hd={self.head_dim}")

        # Freeze the base weights; LoRA re-opens exactly what we choose.
        for p in self.visual.parameters():
            p.requires_grad_(False)
        for p in self.language_model.parameters():
            p.requires_grad_(False)
        self.visual.eval()

        if config.freeze_vlm:
            print("[wiltechs_x] freeze_vlm=True — ABLATION ONLY. This is the "
                  "configuration that produced this repo's vision collapse.")
            self.n_lora = 0
        else:
            self.n_lora = attach_lora(self.language_model, config.lora_target_modules,
                                      config.lora_rank, config.lora_alpha,
                                      config.lora_dropout)
            if config.lora_on_vision_tower:
                self.n_lora += attach_lora(self.visual, config.lora_target_modules,
                                           config.lora_rank, config.lora_alpha,
                                           config.lora_dropout)
            print(f"[wiltechs_x] LoRA r={config.lora_rank} a={config.lora_alpha} "
                  f"on {self.n_lora} projections")

        # ---- 2. Action expert -------------------------------------------
        self.d_exp = int(config.expert_hidden_size)
        if self.d_exp % 2:
            # _time_embedding builds cat([cos, sin]) of width 2*(d//2); an odd
            # width silently produces d-1 features and the first Linear fails
            # with a shape error nowhere near the cause.
            raise ValueError(f"expert_hidden_size must be even, got {self.d_exp}")
        d_ffn = int(config.expert_intermediate_size or self.d_exp)
        n_exp = int(config.expert_num_layers or self.num_vlm_layers)
        if n_exp > self.num_vlm_layers:
            raise ValueError(f"expert_num_layers ({n_exp}) > VLM layers "
                             f"({self.num_vlm_layers})")
        self.first_joint_layer = self.num_vlm_layers - n_exp
        self.expert_layers = nn.ModuleList([
            JointExpertLayer(self.d_exp, d_ffn, self.num_heads, self.num_kv_heads,
                             self.head_dim, self.rms_norm_eps,
                             ada_rank=int(config.ada_rank))
            for _ in range(n_exp)
        ])
        n_e = sum(p.numel() for p in self.expert_layers.parameters())
        print(f"[wiltechs_x] expert: {n_exp} layers @ {self.d_exp} "
              f"(joined at VLM layer {self.first_joint_layer}) — "
              f"{n_e / 1e6:.0f}M params, {n_e / max(n_exp, 1) / 1e6:.1f}M/layer")

        # ---- 3. Suffix embeddings ---------------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, self.d_exp),
            RMSNorm(self.d_exp, eps=self.rms_norm_eps))
        self.action_in_proj = nn.Linear(config.action_dim, self.d_exp)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, config.horizon, self.d_exp))
        nn.init.normal_(self.action_pos_emb, std=0.02)
        self.num_register_tokens = int(config.num_register_tokens or 0)
        if self.num_register_tokens:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, self.d_exp))
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        self.final_norm = RMSNorm(self.d_exp, eps=self.rms_norm_eps)
        self.action_out_proj = nn.Linear(self.d_exp, config.action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

        # Time embedder. Under "shortcut" it also takes the step size, so the
        # network knows how far the velocity it emits has to be valid for.
        t_in = self.d_exp * (2 if config.flow_objective == "shortcut" else 1)
        self.time_embedder = nn.Sequential(
            nn.Linear(t_in, self.d_exp), nn.SiLU(),
            nn.Linear(self.d_exp, self.d_exp))

        # ---- 4. Prefix extras -------------------------------------------
        self.wrist_encoder = None
        if config.use_wrist_encoder:
            self.wrist_encoder = WristTokenizer(
                config.wrist_encoder_id, config.wrist_tokens, self.hidden_size,
                config.freeze_wrist_encoder, config.wrist_input_size)
            g = int(round(config.wrist_tokens ** 0.5))
            print(f"[wiltechs_x] wrist: {config.wrist_encoder_id}, "
                  f"{config.wrist_tokens} tok = {g}x{g} grid @ "
                  f"{config.wrist_input_size}px input")
            # The verdict needs the VLM's own per-camera grid, which is only
            # known once an image has been through the processor. Printed by
            # _build_prefix instead.

        self.motion_encoder = None
        if config.use_motion_vectors:
            self.motion_encoder = MotionVectorEncoder(
                config.state_dim, config.motion_history_len,
                config.motion_vector_tokens, self.hidden_size)

        self.progress_head = ProgressHead(self.d_exp) if config.progress_head else None

        self.discrete_head = None
        if config.fast_token_head and not config.freeze_vlm:
            self.discrete_head = DiscreteActionHead(
                self.hidden_size, config.horizon, config.action_dim)
            print("[wiltechs_x] discrete action head ON. "
                  "ABLATE THIS FIRST — it may be dead weight at LIBERO's scale.")
        # Printed unconditionally and on its own line: this flag changes WHICH
        # LOSSES CAN TRAIN THE BACKBONE, and until 2026-08-19 nothing in the
        # log distinguished the two runs. The discrete-head banner above used
        # to say "(knowledge insulation)" whether or not it was on, which is
        # worse than silence.
        _ki = ("ON  — K/V cache DETACHED: flow/shortcut/gripper/progress "
               "cannot reach the prefix; the discrete head is the only path"
               if config.knowledge_insulation else
               "OFF — flow gradients reach the prefix (LoRA, wrist, motion) "
               "as well as the discrete head")
        print(f"[wiltechs_x] knowledge insulation {_ki}")

        # ---- 5. Consistency checks --------------------------------------
        # With knowledge insulation the K/V cache is detached, so the flow
        # loss cannot reach anything upstream of it. The discrete head is then
        # the ONLY gradient path into the prefix -- and the wrist and motion
        # encoders live in the prefix. Turning the head off therefore does not
        # just ablate the head: it silently freezes the wrist path this repo
        # measured at 34 points.
        #
        # Refused rather than warned, because the failure looks exactly like a
        # bad learning rate.
        if (config.knowledge_insulation and self.discrete_head is None
                and (self.wrist_encoder is not None or self.motion_encoder is not None)
                and not config.freeze_vlm):
            raise ValueError(
                "knowledge_insulation=True with fast_token_head=False leaves the "
                "wrist/motion encoders (and LoRA) with NO gradient path: the "
                "detached K/V cache blocks the flow loss and there is no "
                "discrete head to replace it.\n"
                "Pick one:\n"
                "  --no_discrete_head --no_knowledge_insulation   ablate the head, "
                "let the flow loss train the prefix\n"
                "  --no_discrete_head --no_wrist_encoder --no_motion_vectors   "
                "ablate the head AND the prefix-side trainables\n"
                "  keep the discrete head")

        # ---- 6. Misc -----------------------------------------------------
        self._lang_max_len = int(config.lang_max_len)
        self._template_ids_cpu = None
        self._printed = set()
        self._motion_grace = 0
        self._suite_tokens = {}       # instruction -> token set  (hinge buckets)
        self._suite_id = {}           # instruction -> suite representative
        self.gradient_checkpointing = False

    # =====================================================================
    # Frozen components stay in eval
    # =====================================================================
    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        if self.config.freeze_vlm:
            self.language_model.eval()
        if self.wrist_encoder is not None and self.wrist_encoder.frozen:
            self.wrist_encoder.backbone.eval()
        return self

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        print(f"[wiltechs_x] gradient checkpointing ON — {self.num_vlm_layers} "
              f"prefix layers + {len(self.expert_layers)} expert layers "
              f"recomputed in backward. The prefix is the expensive half.")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def count_parameters(self) -> dict:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fr = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": tr, "frozen": fr, "total": tr + fr}

    # =====================================================================
    # Vision / language encoding
    # =====================================================================
    def _find_visual_merger(self):
        for owner in (self.visual, self.vlm_model):
            for attr in ("merger", "patch_merger", "visual_merger", "merger_module"):
                cand = getattr(owner, attr, None)
                if cand is not None:
                    return cand
        return None

    def _once(self, key: str, msg: str):
        if key not in self._printed:
            self._printed.add(key)
            print(msg)

    def _check_motion_history(self, hist: torch.Tensor, src: str) -> None:
        """Measure the motion window ONCE instead of trusting the plumbing.

        A single frame is not an error on this path, which is the problem:
        MotionVectorEncoder left-pads it, the first difference comes out
        identically zero, and the motion tokens quietly degrade into a fixed
        function of the current state. Nothing raises, the run just loses the
        LIBERO-Long mechanism (ARCHITECTURE.md 3.5) and keeps paying
        `motion_vector_tokens` prefix positions for it. This repo has already
        paid once for a feature that was not actually in the forward batch.

        Training REFUSES rather than warns: the symptom there is a slightly
        worse long-horizon score, which reads as a hyperparameter problem for
        weeks. At inference a short window is legitimate (an env hands over one
        frame at a time), so that path warns and continues.
        """
        if "motion" in self._printed:
            return
        t_req = int(self.config.motion_history_len)
        # At inference the FIRST call is always t=0, where the window is one
        # frame repeated -- that is what a correct left-pad looks like at
        # episode start (StateHistory.reset), not a fault. Judging there
        # reported every healthy rollout as dead. Training has no such phase:
        # every batch is a random mid-episode frame, so a degenerate window on
        # call 1 is a real error there.
        if not self.training:
            self._motion_grace += 1
            if self._motion_grace <= t_req:
                return
        w = hist[:, -t_req:].float()
        t_got = int(w.shape[1])
        d = float((w[:, 1:] - w[:, :-1]).abs().mean()) if t_got > 1 else 0.0
        self._once("motion",
                   f"[wiltechs_x] motion: {t_got}/{t_req} frames from {src}, "
                   f"mean |delta| = {d:.4f} per step (normalized units; "
                   f"0 = this path is a no-op)")
        if t_got >= 2 and d > 0.0:
            return

        why = (f"only {t_got} frame(s) reached the model" if t_got < 2
               else "the frames are identical, so every delta is zero")
        msg = (f"[wiltechs_x] motion vectors are DEAD: {why}. The encoder "
               f"left-pads, so this does not raise on its own -- it just "
               f"removes the LIBERO-Long mechanism (ARCHITECTURE.md 3.5) and "
               f"costs {self.config.motion_vector_tokens} prefix tokens for "
               f"nothing.\n"
               f"  training: n_obs_steps must be >= motion_history_len "
               f"({t_req}) so delta_timestamps stacks the window into "
               f"observation.state.\n"
               f"  inference: keep a rolling window yourself -- see "
               f"StateHistory in src/eval_wiltechs_x.py -- or pass "
               f"observation.state_history as (B, T, D).")
        if self.training and t_req > 1:
            raise ValueError(msg)
        self._once("motion_dead", msg)

    def _encode_images(self, batch: dict, B: int):
        """-> (vis_tokens (B, N, hidden), [grid_thw per camera])."""
        device = batch["observation.state"].device
        cams = self.config.cameras_for_vlm or []
        all_vis, grids = [], []
        # An unconditional no_grad here would silently disconnect the vision
        # tower when lora_on_vision_tower is set -- the adapters would exist,
        # report as trainable, and never receive a gradient.
        vis_trainable = any(p.requires_grad for p in self.visual.parameters())
        for cam_key in cams:
            pvk, thwk = vlm_pixels_key(cam_key), vlm_grid_key(cam_key)
            with (nullcontext() if vis_trainable else torch.no_grad()):
                if pvk in batch:
                    pv, thw = batch[pvk], batch[thwk]
                    if pv.dim() == 3:
                        pv = pv.reshape(-1, pv.shape[-1])
                    if thw.dim() == 1:
                        thw = thw.unsqueeze(0)
                    pixel_values, grid = pv.to(device), thw.to(device)
                elif cam_key in batch:
                    imgs = batch[cam_key]
                    img = imgs[:, -1] if imgs.dim() == 5 else imgs
                    pixel_values, grid = preprocess_camera_to_pixels(
                        self.processor.image_processor, img,
                        target_size=int(self.config.vision_input_size or 0))
                    pixel_values, grid = pixel_values.to(device), grid.to(device)
                else:
                    continue
                self._once(f"grid::{cam_key}",
                           f"[wiltechs_x] vision grid {cam_key}: {grid[0].tolist()}")
                try:
                    vis = self.visual(pixel_values, grid_thw=grid)
                except TypeError:
                    vis = self.visual(pixel_values, image_grid_thw=grid)
                vis = getattr(vis, "last_hidden_state", vis)
                if vis.shape[-1] != self.hidden_size:
                    merger = self._find_visual_merger()
                    if merger is None:
                        raise RuntimeError(
                            f"vision dim {vis.shape[-1]} != text hidden "
                            f"{self.hidden_size} and no merger submodule found.")
                    try:
                        vis = merger(vis)
                    except TypeError:
                        vis = merger(vis, grid)
                    vis = getattr(vis, "last_hidden_state", vis)
            if vis.dim() == 2:
                if vis.shape[0] % B:
                    raise RuntimeError(
                        f"cannot unpack vis tokens {tuple(vis.shape)} into B={B}; "
                        f"mixed resolutions?")
                vis = vis.reshape(B, -1, self.hidden_size)
            all_vis.append(vis)
            grids.append(grid[0].detach())
        if not all_vis:
            return torch.zeros(B, 0, self.hidden_size, device=device,
                               dtype=torch.bfloat16), []
        return torch.cat(all_vis, dim=1), grids

    def _resolve_descs(self, batch: dict):
        v = batch.get("task_description", batch.get("task"))
        if v is None:
            return None
        descs = [v] if isinstance(v, str) else list(v)
        if self.config.use_descriptive_objects:
            descs = [rewrite_instruction(d) for d in descs]
        return descs

    def _suite_map(self, descs):
        """Group instructions into "suites" by token overlap.

        Returns instruction -> suite representative. Built by union-find over
        EVERY instruction seen so far, not just this batch, so the grouping
        does not flicker as batches sample different tasks; LIBERO has 40
        unique strings, so the O(n^2) rebuild is trivial and only runs on the
        steps that introduce a new one.

        Clustering the strings avoids depending on task_index -> suite, which
        would assume the merged dataset orders tasks by suite. libero_spatial's
        ten instructions differ only in a prepositional phrase, so they fall
        out as one component on their own.
        """
        thr = float(getattr(self.config, "contrastive_suite_jaccard", 0.0) or 0.0)
        if thr <= 0.0:
            return None
        fresh = [d for d in set(descs) if d not in self._suite_tokens]
        if fresh:
            for d in fresh:
                self._suite_tokens[d] = frozenset(str(d).lower().split())
            keys = list(self._suite_tokens)
            parent = {k: k for k in keys}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a in range(len(keys)):
                ta = self._suite_tokens[keys[a]]
                for b in range(a + 1, len(keys)):
                    tb = self._suite_tokens[keys[b]]
                    u = len(ta | tb)
                    if u and len(ta & tb) / u >= thr:
                        ra, rb = find(keys[a]), find(keys[b])
                        if ra != rb:
                            parent[ra] = rb
            self._suite_id = {k: find(k) for k in keys}
            n = len(set(self._suite_id.values()))
            self._once(f"suites::{len(keys)}::{n}",
                       f"[wiltechs_x] hinge negatives: {len(keys)} instructions "
                       f"seen -> {n} suite(s) at Jaccard {thr}")
        return self._suite_id

    def _hinge_pairs(self, descs, order, draw):
        """Pick a wrong-instruction partner for each index in `order`.

        Returns (keep, other, n_same_suite, n_dropped). `keep` can be shorter
        than `order`: a sample with no differently-worded partner anywhere in
        the batch is dropped rather than paired, because every candidate would
        be a CORRECT instruction and the hinge would then punish the model for
        agreeing with itself.

        `draw` is a list of floats in [0, 1), one per entry of `order`, so the
        caller owns the randomness and the test can make it deterministic.
        """
        suites = self._suite_map(descs)
        B = len(descs)
        buckets = {}
        if suites is not None:
            for p, d in enumerate(descs):
                buckets.setdefault(suites[d], []).append(p)
        keep, other, hard, dropped = [], [], 0, 0
        for j, i in enumerate(order):
            pool = ()
            if suites is not None:
                pool = tuple(p for p in buckets[suites[descs[i]]]
                             if descs[p] != descs[i])
            if pool:
                hard += 1
            else:
                # Suite has no second task in THIS batch (or buckets are off).
                # Any different instruction still trains language sensitivity,
                # just more cheaply than a same-suite one would.
                pool = tuple(p for p in range(B) if descs[p] != descs[i])
            if not pool:
                dropped += 1
                continue
            keep.append(i)
            other.append(pool[min(int(draw[j] * len(pool)), len(pool) - 1)])
        return keep, other, hard, dropped

    def _format_instruction(self, descs):
        tmpl = str(self.config.instruction_template or "").strip()
        if tmpl:
            return [tmpl.replace("{instruction}", str(d)) for d in descs]
        return [str(d) for d in descs]

    def _template_ids(self, device):
        if self._template_ids_cpu is None:
            tok = self.processor.tokenizer
            head = tok("<|im_start|>user\n", add_special_tokens=False,
                       return_tensors="pt")["input_ids"][0].long()
            tail = tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False,
                       return_tensors="pt")["input_ids"][0].long()
            vs = tok.convert_tokens_to_ids("<|vision_start|>")
            ve = tok.convert_tokens_to_ids("<|vision_end|>")
            if vs is None or ve is None:
                raise RuntimeError("tokenizer lacks <|vision_start|>/<|vision_end|>")
            self._template_ids_cpu = (head, torch.tensor([vs]).long(),
                                      torch.tensor([ve]).long(), tail)
        return tuple(t.to(device) for t in self._template_ids_cpu)

    # =====================================================================
    # Prefix
    # =====================================================================
    def _build_prefix(self, batch: dict):
        """-> (prefix_emb, prefix_pad_mask, segments, spans)

        Layout:  <|im_start|>user\\n {instruction}
                 (<|vision_start|> [cam] <|vision_end|>) x cams
                 [wrist tokens] [motion tokens]
                 <|im_end|>\\n<|im_start|>assistant\\n

        The instruction is padded to a FIXED length (`padding="max_length"`),
        not to the batch max. That costs a few dead tokens and buys a prefix
        whose length -- and therefore whose M-RoPE phase for every later
        segment -- does not move from batch to batch. WiltechsVLA's "known
        asymmetries" note records exactly this as an unquantified noise source
        there; here it costs nothing to remove.

        Independent of the flow time and the noise, by construction.
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device
        embed = self.language_model.get_input_embeddings()
        head_ids, vs_id, ve_id, tail_ids = self._template_ids(device)

        parts, segments, masks = [], [], []
        spans: dict = {}
        cursor = 0

        def add(emb, seg, mask=None):
            """Append a segment and return its (start, end) index range.

            The ranges are what the attention-mass diagnostics slice, so they
            have to be tracked here rather than recomputed later -- `segments`
            holds grid tensors for image blocks and cannot be summed."""
            nonlocal cursor
            n = emb.shape[1]
            parts.append(emb)
            segments.append(seg)
            masks.append(mask if mask is not None else torch.ones(
                B, n, device=device, dtype=torch.bool))
            cursor += n
            return (cursor - n, cursor)

        # header
        h = embed(head_ids).unsqueeze(0).expand(B, -1, -1)
        add(h, ("text", head_ids.shape[0]))

        # instruction (fixed length)
        descs = self._resolve_descs(batch)
        if descs:
            texts = self._format_instruction(descs)
            enc = self.processor.tokenizer(
                texts, return_tensors="pt", padding="max_length", truncation=True,
                max_length=self._lang_max_len, add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            lmask = enc["attention_mask"].bool().to(device)
            self._once("lang", f"[wiltechs_x] lang fixed at {self._lang_max_len} tok, "
                               f"longest in first batch={int(lmask.sum(1).max())}")
            lang = embed(ids)
            lang = torch.where(lmask.unsqueeze(-1), lang, torch.zeros_like(lang))
            spans["lang"] = add(lang, ("text", ids.shape[1]), lmask)

        # vision, one bracketed block per camera
        vis, grids = self._encode_images(batch, B)
        m = self.spatial_merge_size
        sizes = [int(g[0]) * (int(g[1]) // m) * (int(g[2]) // m) for g in grids]
        cam_tokens = list(vis.split(sizes, dim=1)) if sizes else []
        vs_e = embed(vs_id).unsqueeze(0).expand(B, -1, -1)
        ve_e = embed(ve_id).unsqueeze(0).expand(B, -1, -1)
        vis_spans = []
        for ct, g in zip(cam_tokens, grids):
            add(vs_e, ("text", 1))
            vis_spans.append(add(ct.to(vs_e.dtype), ("image", g)))
            add(ve_e, ("text", 1))
        spans["vision"] = vis_spans

        # wrist (self-supervised, trainable) -- in the SHARED sequence, NOT a
        # side channel. See WristTokenizer.
        wrist_spans = []
        if self.wrist_encoder is not None:
            for cam_key in (self.config.wrist_cameras or []):
                if cam_key not in batch:
                    continue
                imgs = batch[cam_key]
                img = imgs[:, -1] if imgs.dim() == 5 else imgs
                w = self.wrist_encoder(img.float())
                wrist_spans.append(add(w.to(vs_e.dtype), ("text", w.shape[1])))
        spans["wrist"] = wrist_spans

        # motion vectors (hindsight)
        if self.motion_encoder is not None:
            # `observation.state_history` is an OPTIONAL override. The normal
            # path is the fallback: the trainer sets n_obs_steps =
            # motion_history_len and gives `observation.state` matching
            # delta_timestamps, so it arrives as (B, T, D) already.
            src = "observation.state_history"
            hist = batch.get(src)
            if hist is None:
                src = "observation.state"
                st = batch[src]
                hist = st if st.dim() == 3 else st.unsqueeze(1)

            self._check_motion_history(hist, src)
            mv = self.motion_encoder(hist.float())
            spans["motion"] = add(mv.to(vs_e.dtype), ("text", mv.shape[1]))

        # assistant header -- the discrete head's readout position
        tail = embed(tail_ids).unsqueeze(0).expand(B, -1, -1)
        add(tail, ("text", tail_ids.shape[0]))

        prefix = torch.cat(parts, dim=1).to(torch.bfloat16)
        pad_mask = torch.cat(masks, dim=1)
        spans["readout"] = prefix.shape[1] - 1
        n_vis = sum(e - s for s, e in vis_spans)
        n_wri = sum(e - s for s, e in wrist_spans)
        self._once("prefix",
                   f"[wiltechs_x] L_prefix={prefix.shape[1]} "
                   f"(lang {self._lang_max_len} | vision {n_vis} over "
                   f"{len(cam_tokens)} cams | wrist {n_wri} | "
                   f"motion {self.config.motion_vector_tokens if self.motion_encoder else 0})")

        # Wrist resolution verdict, now that the VLM's own grid is known.
        #
        # The comparison is TOKENS PER CAMERA, not pixels per token. An
        # earlier version of this check divided wrist_input_size by
        # sqrt(wrist_tokens) and compared against 32, which is wrong twice
        # over: it ignores the source frame's real resolution, and raising
        # wrist_input_size only upsamples before a pool that throws the extra
        # patches away. What decides whether this path resolves anything the
        # VLM cannot is simply how many cells the image is cut into.
        if self.wrist_encoder is not None and cam_tokens:
            vlm_per_cam = sizes[0]
            wt = int(self.config.wrist_tokens)
            g_v, g_w = int(round(vlm_per_cam ** 0.5)), int(round(wt ** 0.5))
            verdict = ("FINER" if wt > vlm_per_cam else
                       "IDENTICAL" if wt == vlm_per_cam else "COARSER")
            self._once("wrist_grid",
                       f"[wiltechs_x] wrist grid {g_w}x{g_w} vs VLM "
                       f"{g_v}x{g_v} per camera — {verdict}")
            if wt <= vlm_per_cam:
                self._once("wrist_warn",
                           f"[wiltechs_x]   ^ this path exists to resolve detail "
                           f"the VLM tokens cannot. At {wt} <= {vlm_per_cam} "
                           f"tokens it resolves nothing extra. Raise "
                           f"--wrist_tokens above {vlm_per_cam} "
                           f"(--wrist_input_size does NOT help: it upsamples "
                           f"before a pool that discards the extra patches).")
        return prefix, pad_mask, segments, spans

    # =====================================================================
    # Suffix
    # =====================================================================
    def _time_embedding(self, t: torch.Tensor, d: Optional[torch.Tensor] = None):
        """Sinusoidal(t) [and (d)] -> (B, d_exp)."""
        def sinusoid(x):
            half = self.d_exp // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=x.device, dtype=torch.float32) / half)
            a = x.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
            return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)

        e = sinusoid(t)
        if self.config.flow_objective == "shortcut":
            e = torch.cat([e, sinusoid(d if d is not None else torch.zeros_like(t))], -1)
        return self.time_embedder(e)

    def _build_suffix(self, state: torch.Tensor, x_t: torch.Tensor):
        """[state(1), register(R), action(H)] -> (B, L_suffix, d_exp).

        Ordered by role, and the order is load-bearing under the causal mask:
        the state carries observation and comes first; the registers are a
        scratchpad that must read it; every action reads all registers.

        Takes the state tensor rather than the batch so the shortcut branch can
        run on a subset without rebuilding a dict.
        """
        st = state.float()
        if st.dim() == 3:
            st = st[:, -1]
        parts = [self.state_encoder(st).unsqueeze(1)]
        if self.register_tokens is not None:
            parts.append(self.register_tokens.expand(st.shape[0], -1, -1))
        a = self.action_in_proj(x_t.to(self.action_in_proj.weight.dtype))
        parts.append(a + self.action_pos_emb[:, : a.shape[1]])
        return torch.cat(parts, dim=1)

    # =====================================================================
    # The stack
    # =====================================================================
    def _prefix_only_layer(self, layer, hidden, mask, cos, sin):
        """A VLM layer run on the prefix alone (depths below first_joint_layer)."""
        res = hidden
        h = layer.input_layernorm(hidden)
        B, L, _ = h.shape
        q = layer.self_attn.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(h).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(h).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        n_rep = self.num_heads // self.num_kv_heads
        o = F.scaled_dot_product_attention(q, _repeat_kv(k, n_rep), _repeat_kv(v, n_rep),
                                           attn_mask=mask.to(q.dtype), is_causal=False)
        o = o.transpose(1, 2).contiguous().view(B, L, -1)
        hidden = res + layer.self_attn.o_proj(o)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden)), (k, v)

    def _rope(self, segments, B, L_suffix, device, ref):
        """M-RoPE cos/sin over prefix + suffix.

        The suffix is appended as one monotonic text segment. Because the
        instruction is padded to a fixed length, the prefix length is a
        constant, so the suffix's phase does not drift between batches.
        """
        segs = list(segments) + ([("text", L_suffix)] if L_suffix else [])
        pos = _build_mrope_position_ids(segs, B=B,
                                        spatial_merge_size=self.spatial_merge_size,
                                        device=device)
        return self.language_model.rotary_emb(ref, pos)

    @property
    def needs_prefix_grad(self) -> bool:
        """Does anything actually consume the prefix's computation graph?

        With knowledge insulation on, the K/V cache is detached, so the flow
        loss stops at the cache and the discrete head is the ONLY consumer of
        `prefix_out`. No head -> no consumer -> the 36-layer prefix can run
        under no_grad, which is most of this model's training memory.

        The same fact is the landmine __init__ refuses: with KI on and no
        discrete head, the wrist and motion encoders sit in the prefix with no
        gradient path at all.
        """
        if not self.training:
            return False
        if not self.config.knowledge_insulation:
            return True                       # flow loss flows through the cache
        return self.discrete_head is not None

    def _run_prefix(self, prefix, pad_mask, segments, L_suffix, needs_grad=None):
        """Phase 1: prefix alone, caching per-layer K/V.

        Used by TRAINING AND SAMPLING ALIKE. The prefix cannot see the suffix,
        so its per-layer K/V are the same tensors the joint forward would have
        produced -- running it once and reusing it is not an inference-only
        approximation, it is the same computation. Sharing the path is
        deliberate: a separate training path is where train/inference skew
        comes from, and this model has enough moving parts already.

        Correct ONLY because prefix->suffix is masked. If that block is ever
        opened this cache silently becomes wrong rather than raising, which is
        why test_components.py checks the mask and checks
        forward_cached == forward.

        MEMORY. This is where it goes, not the expert: 36 layers over a
        ~420-token prefix stores roughly 9 GiB of activations at B=8, on top
        of 8 GiB of bf16 weights. Two mitigations, both automatic:

          * `needs_prefix_grad` -- with knowledge insulation ON the only
            consumer of the prefix graph is the discrete head, so without that
            head the whole stack runs under no_grad.
          * gradient checkpointing covers these layers too. It used to wrap
            only the expert, which is the cheap half.
        """
        needs_grad = self.needs_prefix_grad if needs_grad is None else needs_grad
        B, L_p = prefix.shape[0], prefix.shape[1]
        dtype = prefix.dtype
        cos, sin = self._rope(segments, B, L_suffix, prefix.device, prefix)
        cos_p, sin_p = cos[:, :L_p], sin[:, :L_p]
        neg = torch.finfo(dtype).min
        pre_mask = torch.zeros(B, 1, L_p, L_p, device=prefix.device, dtype=dtype)
        if not self.config.bidirectional_prefix:
            pre_mask = pre_mask + torch.triu(
                torch.full((L_p, L_p), neg, device=prefix.device, dtype=dtype), 1)
        pre_mask.masked_fill_((~pad_mask).unsqueeze(1).unsqueeze(1), neg)

        n_rep = self.num_heads // self.num_kv_heads

        def joint_prefix_step(layer, exp, hidden):
            pq, pk, pv = exp.prefix_qkv(hidden, layer)
            pq, pk = _apply_rope(pq, pk, cos_p, sin_p)
            o = F.scaled_dot_product_attention(
                pq, _repeat_kv(pk, n_rep), _repeat_kv(pv, n_rep),
                attn_mask=pre_mask.to(pq.dtype), is_causal=False)
            o = o.transpose(1, 2).contiguous().view(B, L_p, -1)
            hidden = hidden + layer.self_attn.o_proj(o)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            return hidden, pk, pv

        ckpt = self.gradient_checkpointing and self.training and needs_grad
        ctx = nullcontext() if needs_grad else torch.no_grad()
        cache = []
        with ctx:
            for i, layer in enumerate(self.language_model.layers):
                if i < self.first_joint_layer:
                    if ckpt:
                        prefix, _ = torch.utils.checkpoint.checkpoint(
                            self._prefix_only_layer, layer, prefix, pre_mask,
                            cos_p, sin_p, use_reentrant=False)
                    else:
                        prefix, _ = self._prefix_only_layer(
                            layer, prefix, pre_mask, cos_p, sin_p)
                    continue
                exp = self.expert_layers[i - self.first_joint_layer]
                if ckpt:
                    prefix, pk, pv = torch.utils.checkpoint.checkpoint(
                        joint_prefix_step, layer, exp, prefix, use_reentrant=False)
                else:
                    prefix, pk, pv = joint_prefix_step(layer, exp, prefix)
                cache.append((pk, pv))
        return prefix, cache, (cos, sin)

    def _run_suffix_cached(self, suffix, cache, rope, pad_mask, t_emb):
        """Phase 2: one pass of the suffix against the cached prefix K/V."""
        cos, sin = rope
        L_p = cache[0][0].shape[2]
        L_s = suffix.shape[1]
        mask = build_suffix_attn_mask(L_p, L_s, pad_mask, cache[0][0].dtype)
        rope_s = (cos[:, L_p:L_p + L_s], sin[:, L_p:L_p + L_s])
        for exp, (pk, pv) in zip(self.expert_layers, cache):
            if self.gradient_checkpointing and self.training:
                suffix = torch.utils.checkpoint.checkpoint(
                    exp.forward_cached, suffix, pk, pv, mask, rope_s, t_emb,
                    use_reentrant=False)
            else:
                suffix = exp.forward_cached(suffix, pk, pv, mask, rope_s, t_emb)
        return suffix

    def _suffix_pass(self, state, x, t, d, cache, rope, pad_mask):
        """-> (velocity (B, H, A) float, suffix_out). One denoising evaluation."""
        t_emb = self._time_embedding(t, d)
        suffix = self._build_suffix(state, x)
        out = self._run_suffix_cached(suffix, cache, rope, pad_mask, t_emb)
        return self._velocity(out).float(), out

    @staticmethod
    def _subset(cache, rope, pad_mask, idx):
        """Slice the cached prefix for the shortcut branch."""
        c = [(k[idx], v[idx]) for k, v in cache]
        return c, (rope[0][idx], rope[1][idx]), pad_mask[idx]

    def _velocity(self, suffix_out) -> torch.Tensor:
        H = self.config.horizon
        return self.action_out_proj(self.final_norm(suffix_out[:, -H:]))

    # =====================================================================
    # Flow matching
    # =====================================================================
    def sample_noise(self, shape, device):
        rho = float(self.config.noise_temporal_correlation)
        noise = torch.randn(shape, device=device) * float(self.config.sample_noise_scale)
        if rho == 0.0 or shape[1] == 1:
            return noise
        scale = math.sqrt(1.0 - rho * rho)
        for t in range(1, shape[1]):
            noise[:, t] = rho * noise[:, t - 1] + scale * noise[:, t]
        return noise

    def sample_time(self, B, device):
        return torch.rand(B, device=device) * 0.998 + 0.001

    # =====================================================================
    # Loss
    # =====================================================================
    def compute_loss(self, batch: dict, return_parts: bool = False):
        cfg = self.config
        actions = batch["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0)
        actions = actions[:, : cfg.horizon]
        B, H, A = actions.shape
        device = actions.device

        prefix, pad_mask, segments, spans = self._build_prefix(batch)
        L_s = 1 + self.num_register_tokens + H
        prefix_out, cache, rope = self._run_prefix(prefix, pad_mask, segments, L_s)

        # Knowledge insulation. Detaching the CACHE is the same cut as the
        # stop-grad inside JointExpertLayer, applied once instead of per layer:
        # the expert reads the VLM, the flow loss never rewrites it. The
        # discrete head below still trains the backbone, through prefix_out.
        # `raw_cache` keeps the undetached graph for the contrastive hinge
        # below. Knowledge insulation exists to stop the flow REGRESSION
        # gradient from rewriting the VLM; the hinge is the term KI replaced,
        # and its whole purpose is to reach the language pathway. Insulating it
        # too would leave it able to train the expert and nothing else, which
        # cannot move the 4% the probe measured on the VLM side.
        raw_cache = cache
        if cfg.knowledge_insulation:
            cache = [(k.detach(), v.detach()) for k, v in cache]

        noise = self.sample_noise((B, H, A), device)
        t = self.sample_time(B, device)
        t_e = t[:, None, None]
        x_t = t_e * noise + (1.0 - t_e) * actions
        u_t = noise - actions                                  # velocity target

        state = batch["observation.state"]
        d0 = torch.zeros(B, device=device) if cfg.flow_objective == "shortcut" else None
        v_t, suffix_out = self._suffix_pass(state, x_t, t, d0, cache, rope, pad_mask)

        # ---- main flow loss with padding / position weighting -----------
        loss = F.mse_loss(v_t, u_t, reduction="none")
        is_pad = batch.get("action_is_pad")
        valid_t = (~is_pad.bool()).float()[:, :H] if is_pad is not None \
            else torch.ones(B, H, device=device)
        pos_w = torch.ones(H, device=device)
        n_exec = min(max(int(cfg.loss_exec_steps or H), 1), H)
        pos_w[n_exec:] = float(cfg.future_steps_weight)
        cells = valid_t.unsqueeze(-1) * pos_w[None, :, None]
        # expand_as, NOT a bare cells.sum(). `cells` is (B, H, 1) and `loss` is
        # (B, H, A), so the numerator sums B*H*A terms while cells.sum() counts
        # only B*H -- the mean was short by exactly action_dim, making `flow`
        # 7x too large on a 7-DOF arm.
        #
        # Not a display bug. `shortcut`, `gripper` and `progress` are added to
        # the same `main` and share the expert with the flow term, so all three
        # had their effective weights divided by 7 -- which is why the shortcut
        # consistency term, the thing that makes 1-4 NFE inference valid at
        # all, sat at 0.003 while flow was at 1.55.
        #
        # The discrete CE below already expands correctly, and that is what
        # dated this: it starts at exactly ln(n_bins), while flow started at
        # 7x its own baseline of (sample_noise_scale^2 + E[a^2]).
        denom = cells.expand_as(loss).sum().clamp(min=1e-6)
        main = (loss * cells).sum() / denom * cfg.action_loss_weight
        parts = {"flow": float(main.detach())}

        # ---- shortcut self-consistency ----------------------------------
        # s(x, t, 2d) should equal the average of two consecutive d-steps.
        # That identity is what makes 1-4 NFE inference valid; without it,
        # num_inference_steps=4 is just an under-integrated Euler solve.
        #
        # Cost is three extra SUFFIX passes on a fraction of the batch -- the
        # prefix cache is reused, so the VLM is not re-run. That is the whole
        # reason training goes through the cached path.
        if cfg.flow_objective == "shortcut" and cfg.shortcut_consistency_frac > 0:
            k = max(1, int(B * float(cfg.shortcut_consistency_frac)))
            idx = torch.randperm(B, device=device)[:k]
            sub_cache, sub_rope, sub_pad = self._subset(cache, rope, pad_mask, idx)
            sub_state, x_i, t_i = state[idx], x_t[idx], t[idx]
            half = torch.pow(2.0, -torch.randint(2, 6, (k,), device=device).float())
            # Both half-steps and the combined step must stay inside [0, 1]:
            # t is sampled uniformly, so an unclamped d=1/4 at t=0.01 would
            # evaluate the field at t=-0.24, where nothing was ever trained.
            half = torch.minimum(half, t_i / 2)

            with torch.no_grad():
                s1, _ = self._suffix_pass(sub_state, x_i, t_i, half,
                                          sub_cache, sub_rope, sub_pad)
                x_mid = x_i - half[:, None, None] * s1
                s2, _ = self._suffix_pass(sub_state, x_mid, t_i - half, half,
                                          sub_cache, sub_rope, sub_pad)
                target = 0.5 * (s1 + s2)
            s_full, _ = self._suffix_pass(sub_state, x_i, t_i, 2 * half,
                                          sub_cache, sub_rope, sub_pad)
            sc = F.mse_loss(s_full, target)
            main = main + sc
            parts["shortcut"] = float(sc.detach())

        # ---- contrastive language hinge ---------------------------------
        # Permutes the LANGUAGE SPAN of the cached per-layer K/V across the
        # batch: every sample keeps its own vision, wrist and motion and gets
        # another sample's instruction. That is why this costs one extra
        # SUFFIX pass (25 tokens) rather than a second prefix pass (452 tokens
        # through 36 VLM layers) -- the same trick wiltechs_vla used.
        #
        # The hinge is one-sided on purpose: v_right is detached, so the term
        # can only push the WRONG-instruction prediction away. Without that,
        # the cheapest way to satisfy a margin is to move the correct
        # prediction, which is the one the flow loss is trying to get right.
        cw = float(cfg.contrastive_loss_weight or 0.0)
        lang_span = spans.get("lang")
        run_hinge = cw > 0.0 and lang_span is not None and B > 1
        if run_hinge:
            k = max(1, min(int(B * float(cfg.contrastive_frac)), B))
            idx = torch.randperm(B, device=device)[:k]
            # Roll, so no sample is handed back its own index. That is not
            # enough on its own: two samples of the SAME task carry the same
            # instruction, so a rolled partner can be a CORRECT instruction,
            # and the hinge would then penalise the model for agreeing with
            # itself. Rare (~1/n_tasks, about 1.5% at batch 96 over LIBERO's
            # 40 tasks) but systematically wrong rather than noise, so repair
            # it against the instruction strings rather than the indices.
            #
            # When suite buckets are on, this is replaced outright: the
            # partner is drawn uniformly from the same suite, which excludes
            # the same instruction by construction AND makes every negative a
            # hard one. See contrastive_suite_jaccard in the config for why
            # a cross-suite negative teaches object presence, not relations.
            other = torch.roll(idx, 1)
            descs = self._resolve_descs(batch)
            if descs is not None and len(descs) == B:
                # one draw per pair, taken up front so this stays O(k) on CPU
                keep, oo, hard, dropped = self._hinge_pairs(
                    descs, idx.tolist(), torch.rand(k).tolist())
                # No fallback when keep is empty: that means the whole batch
                # carries ONE instruction, so every available partner is a
                # CORRECT instruction and the hinge would punish the model for
                # agreeing with itself. Skipping is the only sound option.
                if not keep:
                    run_hinge = False
                    self._once("contrastive_skip",
                               "[wiltechs_x] contrastive hinge SKIPPED: the "
                               "batch holds a single instruction, so it has no "
                               "valid negative. Expected only if the sampler "
                               "groups by task -- check it if this repeats.")
                else:
                    idx = torch.tensor(keep, device=device)
                    other = torch.tensor(oo, device=device)
                    self._once("contrastive",
                               f"[wiltechs_x] contrastive hinge ON (weight "
                               f"{cw}, margin {cfg.contrastive_margin}, "
                               f"{k}/{B} of the batch). First batch: "
                               f"{hard}/{k} negatives drawn same-suite, "
                               f"{dropped} dropped for having no distinct "
                               f"instruction.")
        if run_hinge:
            s, e = lang_span
            _, sub_rope, sub_pad = self._subset(cache, rope, pad_mask, idx)
            wrong_cache = [
                (torch.cat([pk[idx][:, :, :s], pk[other][:, :, s:e],
                            pk[idx][:, :, e:]], dim=2),
                 torch.cat([pv[idx][:, :, :s], pv[other][:, :, s:e],
                            pv[idx][:, :, e:]], dim=2))
                for pk, pv in raw_cache]
            d0_sub = d0[idx] if d0 is not None else None
            v_wrong, _ = self._suffix_pass(state[idx], x_t[idx], t[idx], d0_sub,
                                           wrong_cache, sub_rope, sub_pad)
            # Weighted with the SAME cells as the flow loss, not a bare mean.
            # A bare mean averages over padded steps and over the tail the
            # flow loss was told to de-emphasise, so `apart` shrinks as the
            # horizon grows even when language sensitivity is unchanged. At
            # horizon 64 with loss_exec_steps=16 that diluted it ~4x and
            # pinned `contrastive` at the margin: the language-induced
            # divergence lives in the near steps, where the "which object"
            # decision shows up, and the mean spread it over 48 steps the
            # objective barely trains plus ~27% padding.
            #
            # Same normaliser as the flow term also makes `contrastive`
            # comparable across horizons, so one margin works for all of them.
            sub_cells = cells[idx]                          # (k, H, 1)
            diff = F.mse_loss(v_wrong, v_t[idx].detach(), reduction="none")
            wsum = sub_cells.expand_as(diff).sum(dim=(1, 2)).clamp(min=1e-6)
            apart = (diff * sub_cells).sum(dim=(1, 2)) / wsum
            hinge = F.relu(float(cfg.contrastive_margin) - apart).mean()
            main = main + cw * hinge
            parts["contrastive"] = float(hinge.detach())

        # ---- gripper BCE (class-balanced) -------------------------------
        gw = float(cfg.gripper_bce_weight or 0.0)
        thr = float(cfg.gripper_threshold_norm)
        if gw > 0.0 and thr == thr:                            # NaN = uncalibrated
            g = int(cfg.gripper_action_dim)
            temp = max(float(cfg.gripper_bce_temp), 1e-3)
            a_hat = x_t - t_e * v_t                            # exact, no integration
            logit = (a_hat[..., g] - thr) / temp
            target = (actions[..., g] > thr).float()
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
            w = pos_w[None, :] * valid_t
            if cfg.gripper_class_balance:
                # Without this the term sits in the majority-class optimum:
                # the demos are ~89% "open", so "always open" already scores a
                # low BCE and transition-time agreement stays at chance.
                p = ((target * w).sum() / w.sum().clamp(min=1e-6)).clamp(1e-3, 1 - 1e-3)
                w = w * torch.where(target > 0.5, 0.5 / p, 0.5 / (1 - p))
            gl = (bce * w).sum() / w.sum().clamp(min=1e-6)
            main = main + gw * gl
            parts["gripper"] = float(gl.detach())

        # ---- discrete action head (VLM side) ----------------------------
        if self.discrete_head is not None:
            readout = prefix_out[:, spans["readout"]].float()
            logits = self.discrete_head(readout)
            tgt = self.discrete_head.tokenize(actions)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                reduction="none").view(B, H, A)
            ce = (ce * valid_t.unsqueeze(-1)).sum() / \
                 valid_t.unsqueeze(-1).expand_as(ce).sum().clamp(min=1e-6)
            main = main + float(cfg.fast_token_loss_weight) * ce
            parts["discrete"] = float(ce.detach())

        # ---- progress ----------------------------------------------------
        if self.progress_head is not None:
            tgt = self._progress_target(batch, B, device)
            if tgt is not None:
                pr = self.progress_head(self.final_norm(suffix_out[:, 0]).float())
                pl = F.mse_loss(pr, tgt)
                main = main + float(cfg.progress_loss_weight) * pl
                parts["progress"] = float(pl.detach())

        return (main, parts) if return_parts else main

    def _progress_target(self, batch, B, device):
        if "progress" in batch:
            return batch["progress"].float().view(B)
        fi, el = batch.get("frame_index"), batch.get("episode_length")
        if fi is not None and el is not None:
            return (fi.float() / (el.float() - 1).clamp(min=1)).clamp(0, 1).view(B)
        self._once("progress_missing",
                   "[wiltechs_x] progress_head ON but neither 'progress' nor "
                   "('frame_index','episode_length') is in the batch — the term "
                   "is being SKIPPED, not silently zeroed.")
        return None

    # =====================================================================
    # Sampling
    # =====================================================================
    @torch.no_grad()
    def sample_actions(self, batch: dict, full_horizon: bool = True,
                       noise: torch.Tensor | None = None) -> torch.Tensor:
        """(B, horizon, action_dim). ONE prefix pass, then N expert passes.

        full_horizon defaults True so that measurement does not move when an
        inference knob moves -- the validation metrics integrate the chunk to a
        terminal position, and truncating here would change what they mean at
        every n_action_steps setting.

        `noise` overrides the draw at x_1. The integration is deterministic
        given it, so the noise is not a perturbation -- it is the INDEX of
        which sample of p(action | obs) gets returned. Holding it fixed across
        the replans of one episode keeps the policy on one branch of a
        multimodal distribution instead of re-picking every chunk; see
        WiltechsXPolicy.select_action.
        """
        cfg = self.config
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else nullcontext())

        with ctx:
            prefix, pad_mask, segments, _ = self._build_prefix(batch)
            L_s = 1 + self.num_register_tokens + cfg.horizon
            _, cache, rope = self._run_prefix(prefix, pad_mask, segments, L_s)

            N = max(1, int(cfg.num_inference_steps))
            state = batch["observation.state"]
            shape = (B, cfg.horizon, cfg.action_dim)
            if noise is None:
                x = self.sample_noise(shape, device)
            else:
                if tuple(noise.shape) != shape:
                    raise ValueError(
                        f"noise {tuple(noise.shape)} does not match "
                        f"(B, horizon, action_dim) = {shape}")
                x = noise.to(device=device, dtype=torch.float32)
            step = 1.0 / N
            t = torch.ones(B, device=device)
            # Integrates t: 1 -> 0, matching x_t = t*noise + (1-t)*action.
            # Under "shortcut" the step size is an INPUT, so a model trained
            # with the consistency term is valid at this N; under plain "flow"
            # d is ignored and N is an ordinary Euler budget.
            d = torch.full((B,), step, device=device)
            for _ in range(N):
                v, _ = self._suffix_pass(state, x, t, d, cache, rope, pad_mask)
                x = x - step * v
                t = t - step

        return x if full_horizon else x[:, : cfg.n_action_steps]
