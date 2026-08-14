"""
WiltechsVLATransformer — Qwen3-VL-based encoder-decoder flow matching policy.

Architecture (Xiaomi-Robotics-0 / pi0-style MoT, NOT SmolVLA interleaved):

  Stage A (run ONCE per inference): VLM encoder
    Input:   [vision tokens, language tokens]
    Run:     all 36 Qwen3-VL text layers, frozen
    Capture: K, V tensors from the LAST `num_dit_layers` layers
             (these become the cross-attention memory for the DiT)

  Stage B (run num_inference_steps times during denoising): DiT decoder
    Input:   [state, register_tokens, (robot_cnn)?, (latent)?, action_tokens(t)]
    Each layer:
       1. Self-attention with full causal mask
       2. Cross-attention to ONE captured VLM KV pair (Q from DiT, K/V from cache)
       3. SwiGLU FFN
       all three sublayers modulated by adaLN-Zero from the flow-matching time t

  Properties:
    - VLM never sees action / state / robot tokens — it stays in pure VL mode,
      preserving Qwen3-VL's pretrained vision-language capabilities.
    - VLM runs once per inference (10× speedup vs interleaved at N=10 steps).
    - All 36 VLM layers are used (not truncated) — DiT only reads from the
      last N as KV memory, but earlier layers still refine those features.
    - DiT cross-attention has no RoPE on Q; the VLM K already carries
      M-RoPE rotation, which is sufficient for positional alignment.

  Mask semantics (DiT self-attention):
    Full left-to-right causal mask over [state, register, robot, latent, action_0..T-1].
    Every position can only attend to itself and earlier positions. Action
    tokens get an action_pos_emb so they can distinguish their position.

  Replaces the previous interleaved (joint attention every layer) implementation.
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
from .task_rewrites import rewrite_instruction
from ..interleaved_flow_matching.expert_layer import RMSNorm, SwiGLU
from ..transformer_flow_matching.robot_visual_encoder import RobotVisualEncoder


# ---------------------------------------------------------------------------
# Shared camera preprocessing (Qwen image_processor → pixel_values / grid_thw).
#
# Called from TWO places so the training and inference paths can never drift:
#   1. The training DataLoader workers (parallel, overlapped with GPU compute)
#      when --preprocess_in_workers is on — results land in the batch under the
#      vlm_pixels_key()/vlm_grid_key() keys.
#   2. The model's _encode_images fallback (inference, or when workers are off),
#      which preprocesses raw camera frames on the fly.
# ---------------------------------------------------------------------------
_VLM_PIX_PREFIX = "_vlmpix_"


def vlm_pixels_key(cam_key: str) -> str:
    return f"{_VLM_PIX_PREFIX}pv::{cam_key}"


def vlm_grid_key(cam_key: str) -> str:
    return f"{_VLM_PIX_PREFIX}thw::{cam_key}"


def _as_instruction_list(value, key_name: str):
    """Return `value` as a list of instruction strings, or None if absent.

    ABSENT (None / empty) returns None — a dataset with no language is a valid
    configuration and the caller falls through to the next key.

    PRESENT BUT NOT STRINGS raises. The common case is a `task` column holding
    the integer task INDEX rather than the text: default_collate turns it into a
    (B,) tensor, and every downstream truth test on it either raises deep inside
    the encoder ("Boolean value of Tensor with more than one value is
    ambiguous") or, worse, would quietly train a vision-only policy. Failing
    here names the key and shows what it actually held.

    The fix is upstream, in the train script's task-description block: make
    tasks.parquet load so `task_index` maps to text, or set
    batch["task_description"] yourself.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if all(isinstance(d, str) for d in value):
            return list(value)
    raise TypeError(
        f"batch[{key_name!r}] must be a list of instruction strings, got "
        f"{type(value).__name__}"
        + (f" of shape {tuple(value.shape)}" if hasattr(value, "shape") else "")
        + f" (first entries: {list(value[:2]) if hasattr(value, '__getitem__') else value!r}). "
        f"A (B,) tensor here means the column holds the task INDEX, not the "
        f"text — tasks.parquet did not load, so no instruction is reaching the "
        f"VLM. Fix the task-description block in the train script; do not "
        f"silence this."
    )


def preprocess_camera_to_pixels(image_processor, img: torch.Tensor, target_size: int = 0):
    """img: (B, 3, H, W) or (3, H, W) float in [0, 1]. Returns
    (pixel_values, image_grid_thw) on CPU from the Qwen image_processor.

    target_size > 0 forces a square input of that side length, which is what
    sets the vision token count (Qwen3-VL: one merged token per 32x32 input px,
    so 512 -> 16x16 tokens). Both the explicit resize AND the min/max_pixels
    bounds are applied, because the processor's smart-resize would otherwise
    clamp the upscaled image straight back down to its default budget.
    0 leaves the processor on its defaults.

    Different transformers versions accept the pixel bounds either as call
    kwargs or only as instance attributes, so try the kwargs first and fall
    back to a save/set/restore around the call. Do not trust either path to
    have worked -- read the grid_thw the caller prints at startup.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B = img.shape[0]
    img_np = (img.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_images = [Image.fromarray(img_np[i]) for i in range(B)]
    if not target_size:
        proc_out = image_processor(images=pil_images, return_tensors="pt")
        return proc_out["pixel_values"], proc_out["image_grid_thw"]

    ts = int(target_size); px = ts * ts
    pil_images = [im.resize((ts, ts), Image.BICUBIC) for im in pil_images]
    try:
        proc_out = image_processor(images=pil_images, return_tensors="pt",
                                   min_pixels=px, max_pixels=px)
    except (TypeError, ValueError):
        keys = ("min_pixels", "max_pixels")
        saved = {k: getattr(image_processor, k, None) for k in keys}
        size_saved = getattr(image_processor, "size", None)
        try:
            for k in keys:
                if saved[k] is not None:
                    setattr(image_processor, k, px)
            if isinstance(size_saved, dict):
                image_processor.size = {**size_saved,
                                        **{k: px for k in ("shortest_edge", "longest_edge")
                                           if k in size_saved}}
            proc_out = image_processor(images=pil_images, return_tensors="pt")
        finally:
            for k in keys:
                if saved[k] is not None:
                    setattr(image_processor, k, saved[k])
            if isinstance(size_saved, dict):
                image_processor.size = size_saved
    return proc_out["pixel_values"], proc_out["image_grid_thw"]


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
# RoPE helpers (used inside the VLM forward only; DiT does not use RoPE)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL interleaved-M-RoPE cos/sin (already collapsed to (B, L, head_dim))
    onto multi-head Q, K of shape (B, num_heads, L, head_dim)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _build_mrope_position_ids(
    segments: list[tuple[str, object]],
    B: int,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """(3, B, L_vlm) M-RoPE position_ids for an ordered list of segments.

    Each segment is ("text", n_tokens) or ("image", grid_thw). Mirrors HF
    Qwen3VL.get_rope_index: text tokens advance a scalar position replicated
    across the three channels; each image block gets (t, h, w) coordinates at
    the LLM-grid resolution (post spatial_merge_size) offset by the current
    position, after which the position advances by max(t, h, w).
    """
    pieces: list[torch.Tensor] = []
    cur = 0
    for kind, val in segments:
        if kind == "text":
            n = int(val)
            if n <= 0:
                continue
            p = torch.arange(cur, cur + n, device=device).unsqueeze(0).expand(3, -1)
            pieces.append(p)
            cur += n
        else:
            grid_thw = val
            t = int(grid_thw[0].item())
            h = int(grid_thw[1].item()) // spatial_merge_size
            w = int(grid_thw[2].item()) // spatial_merge_size

            pos_t = torch.arange(t, device=device).repeat_interleave(h * w) + cur
            pos_h = torch.arange(h, device=device).repeat_interleave(w).repeat(t) + cur
            pos_w = torch.arange(w, device=device).repeat(t * h) + cur
            pieces.append(torch.stack([pos_t, pos_h, pos_w], dim=0))
            cur += max(t, h, w)

    if pieces:
        full = torch.cat(pieces, dim=1)
    else:
        full = torch.zeros(3, 0, dtype=torch.long, device=device)
    return full.unsqueeze(1).expand(3, B, -1).contiguous()


# ---------------------------------------------------------------------------
# adaLN-Zero modulation (DiT-style)
# ---------------------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: (B, L, D) — shift/scale: (B, D). Broadcasts over L."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@torch.no_grad()
def cross_attention_mass(
    layer,
    x: torch.Tensor,
    t_emb: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    vlm_kv_pad_mask: Optional[torch.Tensor],
    action_start: int,
    action_len: int,
    vis_mask: Optional[torch.Tensor],
) -> dict[str, float]:
    """Fraction of a DiT layer's cross-attention that ACTION queries place on
    the vision vs language positions of the VLM KV cache.

    SDPA does not expose softmax weights, so this re-projects Q and re-runs the
    Q·K^T softmax by hand; cost is one extra layer's cross-attn projection.

    `layer` is any DiTLayer, `x` its input and `kv` the (K, V) it reads. Shared
    by both models: WiltechsVLA measures its single last layer, WiltechsMoE
    measures each expert's last layer (it has no single 'last layer', and the
    experts read different VLM depth bands, so they can differ).

    Read it against the POSITION split, not on its own: language is typically a
    small minority of L_vlm, so equal informativeness would show up as a small
    language share. Divide each share by that region's fraction of L_vlm to get
    the over-attention factor.

    Caveat: x is the layer input, pre-self-attn. The true cross-attn input is
    x + sa_gate·sa_out; the gate path is a residual refinement and x dominates,
    which is accurate enough for a diagnostic but not for anything load-bearing.
    """
    if action_len <= 0 or vis_mask is None:
        return {}
    mod = layer.adaLN_modulation(t_emb)
    chunks = mod.chunk(9, dim=-1)
    s_ca, sc_ca = chunks[3], chunks[4]
    h = _modulate(layer.ca_norm(x), s_ca, sc_ca)

    B = h.shape[0]
    H, Hk, D = layer.ca_num_heads, layer.ca_num_kv_heads, layer.ca_head_dim
    Q_full = layer.ca_q(h).view(B, -1, H, D).transpose(1, 2).float()
    Q = Q_full[:, :, action_start:action_start + action_len, :]   # action rows only

    K = kv[0].float()                                             # (B, Hk, L_vlm, D)
    if Hk != H:
        K = K.repeat_interleave(H // Hk, dim=1)

    scores = (Q @ K.transpose(-1, -2)) * (1.0 / math.sqrt(D))     # (B,H,a_len,L_vlm)
    if vlm_kv_pad_mask is not None:
        ca_mask = torch.zeros(B, 1, 1, vlm_kv_pad_mask.shape[-1],
                              device=scores.device, dtype=scores.dtype)
        ca_mask.masked_fill_((~vlm_kv_pad_mask).unsqueeze(1).unsqueeze(1), float("-inf"))
        scores = scores + ca_mask
    weights = torch.softmax(scores, dim=-1)

    stats: dict[str, float] = {}
    if bool(vis_mask.any()):
        stats["vision"] = weights[:, :, :, vis_mask].sum(dim=-1).mean().item()
    txt_mask = ~vis_mask
    if bool(txt_mask.any()):
        # "language" = every non-vision KV position: the instruction, plus the
        # ChatML prefix/markers/assistant header when the template is on.
        stats["language"] = weights[:, :, :, txt_mask].sum(dim=-1).mean().item()
    # Position split, so the caller never has to reconstruct it to normalise.
    stats["_n_vis"] = float(vis_mask.sum().item())
    stats["_n_lang"] = float(txt_mask.sum().item())
    return stats


def _merge_depths(per_depth: list[dict], vlm_layers: list[int]) -> dict:
    """Average cross-attn shares sampled at several DiT depths into one figure.

    A plain mean, which is what makes it comparable with WiltechsMoE's
    router-weighted mean over experts: that router sits near uniform
    (25.4/26.0/24.0/24.6 at ckpt 18000), so its weighted mean is a plain mean
    too. `_per_depth` keeps the breakdown, because the spread is the point —
    the MoE reads 55.8% vision at VLM layer 8 and 8.6% at layer 35, and the
    average of those two says nothing useful on its own.
    """
    valid = [d for d in per_depth if d]
    if not valid:
        return {}
    out = {k: sum(d.get(k, 0.0) for d in valid) / len(valid)
           for k in ("vision", "language")}
    out["_n_vis"] = valid[0].get("_n_vis", 0.0)
    out["_n_lang"] = valid[0].get("_n_lang", 0.0)
    out["_per_expert"] = [(d.get("vision", 0.0), d.get("language", 0.0))
                          for d in valid]
    out["_labels"] = [f"L{v}" for v in vlm_layers[:len(valid)]]
    return out


def format_xattn(s: dict) -> str:
    """Render cross_attention_mass() output as one log line.

    The raw share is not interpretable alone: language is usually a small
    minority of L_vlm (87 of 407 in a 2-camera LIBERO run at
    --vision_input_size 512), so "language=81%" has to be read against "language
    is 21% of the positions". The x= factor is share / position-share, so 1.0
    means the region is attended exactly in proportion to its size and 3.8 means
    each language token draws 3.8x its share.

    Lives here rather than in either train script so both print the identical
    format and the two models' numbers stay comparable.
    """
    n_v, n_l = s.get("_n_vis", 0.0), s.get("_n_lang", 0.0)
    total = n_v + n_l
    out = []
    for key, n in (("vision", n_v), ("language", n_l)):
        if key not in s:
            continue
        share = s[key]
        cell = f"{key}={share * 100:5.1f}%"
        if total > 0 and n > 0:
            cell += f" (x{share / (n / total):.2f})"
        out.append(cell)
    if total > 0:
        out.append(f"[{int(n_l)} lang + {int(n_v)} vis tok]")
    return "  ".join(out)


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
        sa_num_heads: int,
        sa_num_kv_heads: int,
        sa_head_dim: int,
        ca_num_heads: int,
        ca_num_kv_heads: int,
        ca_head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # Self-attention runs at the DiT width (sa_*); cross-attention bridges
        # the DiT width to the frozen VLM KV geometry (ca_*). When the DiT width
        # equals the VLM width both specs are identical (original behavior).
        self.sa_num_heads = sa_num_heads
        self.sa_num_kv_heads = sa_num_kv_heads
        self.sa_head_dim = sa_head_dim
        self.ca_num_heads = ca_num_heads
        self.ca_num_kv_heads = ca_num_kv_heads
        self.ca_head_dim = ca_head_dim

        # ── Self-attention (over DiT sequence, at the DiT width) ────────
        self.sa_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.sa_q = nn.Linear(hidden_size, sa_num_heads * sa_head_dim, bias=False)
        self.sa_k = nn.Linear(hidden_size, sa_num_kv_heads * sa_head_dim, bias=False)
        self.sa_v = nn.Linear(hidden_size, sa_num_kv_heads * sa_head_dim, bias=False)
        self.sa_o = nn.Linear(sa_num_heads * sa_head_dim, hidden_size, bias=False)
        self.sa_drop = nn.Dropout(dropout)

        # ── Cross-attention (Q from DiT, K/V from VLM KV cache) ─────────
        # Only Q has trainable projection; K, V are the cached VLM tensors.
        # ca_q projects the DiT width UP to the VLM head geometry (so the queries
        # dot-product against the cached K/V); ca_o projects back DOWN.
        self.ca_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ca_q = nn.Linear(hidden_size, ca_num_heads * ca_head_dim, bias=False)
        self.ca_o = nn.Linear(ca_num_heads * ca_head_dim, hidden_size, bias=False)
        self.ca_drop = nn.Dropout(dropout)

        # ── FFN ─────────────────────────────────────────────────────────
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn = SwiGLU(hidden_size, intermediate_size)
        self.ffn_drop = nn.Dropout(dropout)

        # ── adaLN-Zero: produces 9 modulation vectors from t_emb ────────
        # 3 sublayers × {shift, scale, gate} = 9 × hidden_size
        #
        # Zero-init the modulation linear so gates start at 0 → each block
        # acts as identity on the residual stream at init. The sublayer
        # output projections (sa_o / ca_o / ffn.down_proj) are LEFT AT
        # DEFAULT INIT. Zero-init'ing them in addition to the modulator
        # creates a dead-init deadlock: residual = x + gate · sublayer_out,
        # with gate=0 AND sublayer_out=0 the backward gradient on BOTH sides
        # is 0·(…) = 0, so neither side can ever escape — the DiT stack
        # never learns and only action_in/out + final_norm receive gradient.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
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
    ) -> torch.Tensor:
        """
        x:               (B, L_dit, H)
        t_emb:           (B, H) — per-batch time conditioning
        vlm_k, vlm_v:    (B, num_kv_heads, L_vlm, head_dim) — frozen VLM cache
        vlm_kv_pad_mask: (B, L_vlm) bool, True at valid VLM positions
        self_attn_mask:  (L_dit, L_dit) additive mask (causal)
        """
        B, L_dit, H = x.shape

        mod = self.adaLN_modulation(t_emb)
        (
            s_sa, sc_sa, g_sa,
            s_ca, sc_ca, g_ca,
            s_ff, sc_ff, g_ff,
        ) = mod.chunk(9, dim=-1)

        # ── Self-attention ────────────────────────────────────────────
        h = _modulate(self.sa_norm(x), s_sa, sc_sa)
        Q = self.sa_q(h).view(B, L_dit, self.sa_num_heads, self.sa_head_dim).transpose(1, 2)
        K = self.sa_k(h).view(B, L_dit, self.sa_num_kv_heads, self.sa_head_dim).transpose(1, 2)
        V = self.sa_v(h).view(B, L_dit, self.sa_num_kv_heads, self.sa_head_dim).transpose(1, 2)
        if self.sa_num_kv_heads != self.sa_num_heads:
            r = self.sa_num_heads // self.sa_num_kv_heads
            K = K.repeat_interleave(r, dim=1)
            V = V.repeat_interleave(r, dim=1)
        sa = F.scaled_dot_product_attention(Q, K, V, attn_mask=self_attn_mask, is_causal=False)
        sa = sa.transpose(1, 2).contiguous().view(B, L_dit, self.sa_num_heads * self.sa_head_dim)
        sa = self.sa_drop(self.sa_o(sa))
        x = x + g_sa.unsqueeze(1) * sa

        # ── Cross-attention to frozen VLM cache ──────────────────────
        h = _modulate(self.ca_norm(x), s_ca, sc_ca)
        Q = self.ca_q(h).view(B, L_dit, self.ca_num_heads, self.ca_head_dim).transpose(1, 2)
        Kv, Vv = vlm_k, vlm_v
        if self.ca_num_kv_heads != self.ca_num_heads:
            r = self.ca_num_heads // self.ca_num_kv_heads
            Kv = Kv.repeat_interleave(r, dim=1)
            Vv = Vv.repeat_interleave(r, dim=1)
        # Build cross-attn pad mask: (B, 1, 1, L_vlm)
        if vlm_kv_pad_mask is not None:
            kpad = ~vlm_kv_pad_mask                                 # True = pad
            ca_mask = torch.zeros(B, 1, 1, vlm_kv_pad_mask.shape[-1],
                                  device=x.device, dtype=Q.dtype)
            ca_mask.masked_fill_(kpad.unsqueeze(1).unsqueeze(1), float("-inf"))
        else:
            ca_mask = None
        ca = F.scaled_dot_product_attention(Q, Kv, Vv, attn_mask=ca_mask, is_causal=False)
        ca = ca.transpose(1, 2).contiguous().view(B, L_dit, self.ca_num_heads * self.ca_head_dim)
        ca = self.ca_drop(self.ca_o(ca))
        x = x + g_ca.unsqueeze(1) * ca

        # ── FFN ──────────────────────────────────────────────────────
        h = _modulate(self.ffn_norm(x), s_ff, sc_ff)
        ff = self.ffn_drop(self.ffn(h))
        x = x + g_ff.unsqueeze(1) * ff

        return x


# ---------------------------------------------------------------------------
# Latent Q-Former: learned queries distill the frozen VLM KV cache (vision +
# language) into a small set of "thought" tokens. Vision-aware, per-frame, fully
# differentiable, and computed ONCE per forward (noise-independent), so it adds
# no cost inside the N-step denoising loop. Replaces the old MLP-on-pooled-
# language latent_generator. Zero-init output gates → starts as a no-op so the
# latent tokens begin at ~0 (matching the previous safe init) and only grow if
# the action loss finds them useful.
# ---------------------------------------------------------------------------

class LatentQFormer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_queries: int,
        n_layers: int,
        ca_num_heads: int,
        ca_num_kv_heads: int,
        ca_head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ca_num_heads = ca_num_heads
        self.ca_num_kv_heads = ca_num_kv_heads
        self.ca_head_dim = ca_head_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                ca_norm=RMSNorm(dim, eps=rms_norm_eps),
                ca_q=nn.Linear(dim, ca_num_heads * ca_head_dim, bias=False),
                ca_o=nn.Linear(ca_num_heads * ca_head_dim, dim, bias=False),
                ffn_norm=RMSNorm(dim, eps=rms_norm_eps),
                ffn=SwiGLU(dim, intermediate_size),
            )) for _ in range(n_layers)
        ])
        # Per-block residual gates, small-init → gentle (non-zero) contribution
        # at start so the latents aren't inert but still don't dominate. ca_o
        # keeps its default init (no zero-init) so the gated path is non-zero.
        self.gates = nn.ParameterList([nn.Parameter(torch.full((2,), 0.1)) for _ in range(n_layers)])

    def forward(
        self,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
        vlm_kv_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """vlm_k, vlm_v: (B, num_kv_heads, L_vlm, head_dim) from one VLM layer.
        Returns latent tokens (B, num_queries, dim)."""
        B = vlm_k.shape[0]
        x = self.queries.expand(B, -1, -1).to(vlm_k.dtype)

        if vlm_kv_pad_mask is not None:
            ca_mask = torch.zeros(B, 1, 1, vlm_kv_pad_mask.shape[-1],
                                  device=x.device, dtype=x.dtype)
            ca_mask.masked_fill_((~vlm_kv_pad_mask).unsqueeze(1).unsqueeze(1), float("-inf"))
        else:
            ca_mask = None

        # GQA expand once (K/V are the same constant across blocks).
        Kv, Vv = vlm_k, vlm_v
        if self.ca_num_kv_heads != self.ca_num_heads:
            r = self.ca_num_heads // self.ca_num_kv_heads
            Kv = Kv.repeat_interleave(r, dim=1)
            Vv = Vv.repeat_interleave(r, dim=1)

        for blk, g in zip(self.layers, self.gates):
            g0, g1 = g[0].to(x.dtype), g[1].to(x.dtype)
            h = blk["ca_norm"](x)
            Q = blk["ca_q"](h).view(B, -1, self.ca_num_heads, self.ca_head_dim).transpose(1, 2)
            a = F.scaled_dot_product_attention(Q, Kv, Vv, attn_mask=ca_mask, is_causal=False)
            a = a.transpose(1, 2).contiguous().view(B, -1, self.ca_num_heads * self.ca_head_dim)
            x = x + g0 * blk["ca_o"](a)
            x = x + g1 * blk["ffn"](blk["ffn_norm"](x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class WiltechsVLATransformer(nn.Module):
    """Encoder-decoder flow matching VLA built on frozen Qwen3-VL-4B."""

    # Non-FP8 bf16 backbone: avoids the finegrained-fp8 CUDA kernel (which needs
    # the `kernels` package AND an FP8-capable GPU, sm_89+/Hopper). The VLM is
    # frozen, so this is just the bf16 view of the same weights — KV cache is
    # numerically near-identical and FP8-pretrained checkpoints load/fine-tune fine.
    VLM_MODEL_ID: str = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(self, config: WiltechsVLAConfig):
        super().__init__()
        self.config = config

        # ─────────────────────────────────────────────────────────────
        # 1. Load Qwen3-VL (frozen, ALL layers kept)
        # ─────────────────────────────────────────────────────────────
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

        # NO LAYER TRUNCATION — encoder-decoder uses all 36 layers so that the
        # last `num_dit_layers` KV caches benefit from the full upstream
        # refinement. (Truncating earlier layers would degrade the cached
        # representations the DiT cross-attends to.)
        self.num_vlm_layers = len(self.language_model.layers)

        text_cfg = self.language_model.config
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_heads = int(text_cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", self.num_heads))
        self.head_dim = int(
            getattr(text_cfg, "head_dim", None) or (self.hidden_size // self.num_heads)
        )
        self.intermediate_size = int(text_cfg.intermediate_size)
        self.rms_norm_eps = float(getattr(text_cfg, "rms_norm_eps", 1e-5))
        print(f"VLM: {self.num_vlm_layers} layers  hidden={self.hidden_size}  "
              f"heads={self.num_heads}  kv_heads={self.num_kv_heads}  "
              f"head_dim={self.head_dim}  intermediate={self.intermediate_size}")

        vis_cfg = getattr(vlm.config, "vision_config", None)
        self.spatial_merge_size = int(getattr(vis_cfg, "spatial_merge_size", 2))

        if config.d_model != self.hidden_size:
            print(f"[wiltechs_vla] forcing d_model {config.d_model} → {self.hidden_size}")
            config.d_model = self.hidden_size

        # Sanity: VLM must own its rotary_emb (used by the manual layer-by-layer
        # forward below to get M-RoPE cos/sin).
        if not hasattr(self.language_model, "rotary_emb"):
            raise RuntimeError(
                "language_model.rotary_emb not found — encoder forward expects "
                "Qwen3VLTextRotaryEmbedding to live on language_model."
            )

        # Freeze VLM
        for p in self.visual.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False
        self.visual.eval()
        self.language_model.eval()
        del vlm

        # ─────────────────────────────────────────────────────────────
        # 2. DiT (trainable) — N layers cross-attending to last N VLM KV pairs
        # ─────────────────────────────────────────────────────────────
        # `num_vlm_layers` field in config is reused as DiT depth (= number of
        # VLM KV caches the DiT cross-attends to, one per DiT layer).
        self.num_dit_layers = int(config.num_vlm_layers)
        if self.num_dit_layers > self.num_vlm_layers:
            raise ValueError(
                f"num_dit_layers ({self.num_dit_layers}) > VLM layers "
                f"({self.num_vlm_layers}); not enough KV caches to source from."
            )
        # Which VLM layers to read -- see vlm_capture_mode in the config for the
        # trade. The VLM runs all 36 either way, so any layer left out is one
        # whose KV was computed and discarded.
        capture_mode = str(getattr(config, "vlm_capture_mode", "last") or "last").lower()
        if capture_mode not in ("last", "spread"):
            raise ValueError(
                f"vlm_capture_mode must be 'last' or 'spread', got {capture_mode!r}.")
        explicit = list(getattr(config, "vlm_capture_layers", None) or [])
        if explicit:
            if len(explicit) != self.num_dit_layers:
                raise ValueError(
                    f"vlm_capture_layers has {len(explicit)} entries but "
                    f"num_dit_layers is {self.num_dit_layers}; they must match."
                )
            if max(explicit) >= self.num_vlm_layers or min(explicit) < 0:
                raise ValueError(
                    f"vlm_capture_layers {explicit} out of range for a "
                    f"{self.num_vlm_layers}-layer VLM."
                )
            self.capture_layers = sorted(explicit)
        elif self.num_dit_layers == self.num_vlm_layers:
            self.capture_layers = list(range(self.num_vlm_layers))
        elif capture_mode == "last":
            self.capture_layers = list(
                range(self.num_vlm_layers - self.num_dit_layers, self.num_vlm_layers))
        else:
            self.capture_layers = np.linspace(
                0, self.num_vlm_layers - 1, self.num_dit_layers, dtype=int
            ).tolist()
        # kv_cache stays a LIST ordered by increasing VLM layer index, so DiT
        # layer i reads capture_layers[i] and kv_cache[-1] is still the deepest
        # captured layer (what the latent Q-Former distils).
        print(f"DiT: {self.num_dit_layers} layers (mode={capture_mode}), sourcing KV "
              f"from VLM layers {self.capture_layers}")

        # ── DiT width (may be < VLM width to save params) ───────────────
        # 0 → match the VLM hidden size (original behavior). Otherwise the DiT
        # residual stream / self-attn / FFN / adaLN run at this smaller width
        # (~quadratic param savings), while cross-attention bridges back up to
        # the VLM head geometry. Must be a multiple of the VLM head_dim.
        self.dit_hidden = int(getattr(config, "dit_hidden_size", 0)) or self.hidden_size
        if self.dit_hidden % self.head_dim != 0:
            raise ValueError(
                f"dit_hidden_size ({self.dit_hidden}) must be divisible by the VLM "
                f"head_dim ({self.head_dim})."
            )
        # Cross-attn always bridges to the VLM KV geometry.
        ca_nh, ca_nkv, ca_hd = self.num_heads, self.num_kv_heads, self.head_dim
        if self.dit_hidden == self.hidden_size:
            # Unchanged default: self-attn uses the exact VLM head config and the
            # VLM FFN width, so saved checkpoints load identically.
            sa_nh, sa_nkv, sa_hd = self.num_heads, self.num_kv_heads, self.head_dim
            dit_intermediate = self.intermediate_size
        else:
            sa_hd = self.head_dim
            sa_nh = self.dit_hidden // sa_hd
            gqa_ratio = max(1, self.num_heads // max(1, self.num_kv_heads))
            sa_nkv = max(1, sa_nh // gqa_ratio)
            while sa_nh % sa_nkv != 0:
                sa_nkv -= 1
            # FFN inner width = the DiT width itself, matching WiltechsMoE.
            # Scaling the VLM's own intermediate_size by dit_hidden/hidden_size
            # (the pre-2026-08-04 formula) gives roughly 4x this, which at 36
            # layers dominates the parameter count for no measured benefit.
            dit_intermediate = self.dit_hidden
            print(f"DiT width decoupled: dit_hidden={self.dit_hidden} (VLM hidden={self.hidden_size}); "
                  f"self-attn {sa_nh}x{sa_hd} (kv {sa_nkv}), cross-attn {ca_nh}x{ca_hd} (kv {ca_nkv}), "
                  f"ffn_intermediate={dit_intermediate}")

        self.dit_layers = nn.ModuleList([
            DiTLayer(
                hidden_size=self.dit_hidden,
                sa_num_heads=sa_nh, sa_num_kv_heads=sa_nkv, sa_head_dim=sa_hd,
                ca_num_heads=ca_nh, ca_num_kv_heads=ca_nkv, ca_head_dim=ca_hd,
                intermediate_size=dit_intermediate,
                rms_norm_eps=self.rms_norm_eps,
                dropout=config.dropout,
            ) for _ in range(self.num_dit_layers)
        ])

        # ─────────────────────────────────────────────────────────────
        # 3. DiT-side embeddings: register, state, action, time MLP
        # ─────────────────────────────────────────────────────────────
        # Registers replace the old single SINK token. The sink sat at position
        # 0 as a no-op attention target; these sit AFTER the state and carry
        # actual capacity -- see num_register_tokens in the config.
        #
        # std=0.02 like every other learned token here. They are not zero-init:
        # the DiT sequence has no positional embedding outside the action slice,
        # so identical registers would be indistinguishable to self-attention
        # and could never differentiate.
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, self.dit_hidden))
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, self.dit_hidden),
            RMSNorm(self.dit_hidden, eps=self.rms_norm_eps),
        )

        self.action_in_proj = nn.Linear(config.action_dim, self.dit_hidden)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, config.horizon, self.dit_hidden))
        nn.init.normal_(self.action_pos_emb, std=0.02)

        self.final_norm = RMSNorm(self.dit_hidden, eps=self.rms_norm_eps)
        self.action_out_proj = nn.Linear(self.dit_hidden, config.action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

        # Time embedding MLP: sinusoidal → MLP → fed to every DiT layer's adaLN
        self.time_embedder = nn.Sequential(
            nn.Linear(self.dit_hidden, self.dit_hidden),
            nn.SiLU(),
            nn.Linear(self.dit_hidden, self.dit_hidden),
        )

        # ─────────────────────────────────────────────────────────────
        # 4. Robot CNN (optional parallel visual path)
        # ─────────────────────────────────────────────────────────────
        if config.use_robot_cnn:
            _pool = str(getattr(config, "robot_cnn_pool", "avg") or "avg")
            self.robot_visual_encoder = RobotVisualEncoder(
                input_size=config.robot_encoder_input_size,
                out_tokens=config.robot_encoder_tokens,
                out_dim=self.dit_hidden,
                pool=_pool,
            )
            _side = config.robot_encoder_input_size / max(config.robot_encoder_tokens, 1) ** 0.5
            print(f"[wiltechs_vla] RobotCNN: pool={_pool}, "
                  f"{config.robot_encoder_tokens} tokens/cam at "
                  f"{config.robot_encoder_input_size}px = {_side:.1f} px/token "
                  f"({'FINER' if _side < 32 else 'COARSER'} than the VLM's 32)")
        else:
            self.robot_visual_encoder = None
            print("[wiltechs_vla] use_robot_cnn=False — RobotVisualEncoder disabled")

        # ─────────────────────────────────────────────────────────────
        # 5. Latent "thought" tokens — task-conditional, zero-init output
        # ─────────────────────────────────────────────────────────────
        self.num_latent_tokens = config.num_latent_tokens
        if self.num_latent_tokens > 0:
            # Learned-query Q-Former: the latent "thought" tokens are produced by
            # cross-attending a small set of learned queries to the frozen VLM KV
            # cache (vision + language), rather than an MLP on pooled language
            # embeddings. Vision-aware, per-frame, differentiable. FFN inner dim
            # is kept at the DiT width to stay parameter-light.
            self.latent_qformer = LatentQFormer(
                dim=self.dit_hidden,
                num_queries=self.num_latent_tokens,
                n_layers=int(getattr(config, "num_latent_qformer_layers", 2)),
                ca_num_heads=self.num_heads,
                ca_num_kv_heads=self.num_kv_heads,
                ca_head_dim=self.head_dim,
                intermediate_size=self.dit_hidden,
                rms_norm_eps=self.rms_norm_eps,
            )

        # 128, not 48: the CoT rewrites in task_rewrites.py (shared with
        # WiltechsMoE) render up to ~105 tokens, and truncation is silent and
        # lands exactly on the disambiguating tail. _report_lang_budget makes
        # the headroom visible at startup.
        self._lang_max_len = int(getattr(config, "lang_max_len", 128) or 128)
        _tmpl = str(getattr(config, "instruction_template", "") or "").strip()
        if _tmpl:
            print(f"[wiltechs_vla] instruction_template ACTIVE (lang_max_len="
                  f"{self._lang_max_len}):\n  {_tmpl!r}")
        self.text_first = bool(getattr(config, "text_first", True))

        # Chat-template static token ids (lazy; only built when
        # config.use_chat_template is on) + one-shot format print.
        self._template_ids_cpu: Optional[tuple] = None
        self._template_format_printed = False
        self._lang_len_printed = False
        self._vision_grid_printed: set = set()

        # Activation checkpointing toggle for the DiT layers. The VLM runs in
        # @torch.no_grad and would not benefit from checkpointing; only the
        # trainable DiT decoder stack stores activations for backward.
        self.gradient_checkpointing = False

        # Attention-mass diagnostics (armed by the train script on the
        # gradient-analysis cadence; self-disarms after one capture so the
        # contrastive v_wrong forward doesn't overwrite the main-forward stats).
        self._capture_attention_stats = False
        self._last_attention_stats: Optional[dict] = None
        self._last_cross_attention_stats: Optional[dict] = None

    # =========================================================================
    # Keep frozen components in eval mode
    # =========================================================================
    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        self.language_model.eval()
        return self

    def gradient_checkpointing_enable(self):
        """Recompute DiT layer activations during backward instead of storing
        them. Trades extra forward compute for ~5-10× lower activation memory
        across the {self.num_dit_layers}-layer DiT stack. Frozen VLM is
        unaffected (it already runs in no_grad)."""
        self.gradient_checkpointing = True
        print(f"[wiltechs_vla] DiT gradient checkpointing ENABLED "
              f"({self.num_dit_layers} layers will be recomputed in backward)")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    # =========================================================================
    # Helpers for locating the Qwen3-VL spatial merger
    # =========================================================================
    def _find_visual_merger(self):
        """Locate the spatial-merger submodule on the vision tower.

        Qwen2/3-VL family names this differently across releases; we look at
        the most common attribute names on both `self.visual` and on
        `self.vlm_model` (sometimes vendored higher up). Returns None if no
        suitable submodule is found.
        """
        for owner in (self.visual, self.vlm_model):
            for attr in ("merger", "patch_merger", "visual_merger", "merger_module"):
                candidate = getattr(owner, attr, None)
                if candidate is not None:
                    return candidate
        return None

    # =========================================================================
    # Vision / language encoding (no gradient, frozen VLM components)
    # =========================================================================
    def cam_target_size(self, cam_key: str) -> int:
        """Square input side length for this camera's Qwen preprocessing, or 0
        for the processor default. Shared with the DataLoader-worker path so
        both produce identical grids."""
        vs = int(getattr(self.config, "vision_input_size", 0) or 0)
        if vs <= 0:
            return 0
        hires = list(getattr(self.config, "vision_hires_cameras", None) or [])
        return vs if (not hires or cam_key in hires) else 0

    def _report_vision_grid(self, cam_key: str, image_grid_thw: torch.Tensor) -> None:
        """One line per camera at startup. The pixel-bound plumbing inside
        preprocess_camera_to_pixels varies by transformers version, so this is
        the ONLY trustworthy confirmation that a resolution change took."""
        if cam_key in self._vision_grid_printed:
            return
        self._vision_grid_printed.add(cam_key)
        g = image_grid_thw[0].tolist() if image_grid_thw.dim() > 1 else image_grid_thw.tolist()
        m = self.spatial_merge_size
        gh, gw = int(g[1]) // m, int(g[2]) // m
        print(f"[wiltechs_vla] vision grid {cam_key}: patch_thw={g} -> {gh}x{gw} merged "
              f"= {gh * gw} tokens/frame "
              f"(target_size={self.cam_target_size(cam_key) or 'processor default'})")

    def _encode_images(
        self, batch: dict, B: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        device = batch["observation.state"].device
        all_vis: list[torch.Tensor] = []
        grid_thw_list: list[torch.Tensor] = []
        for cam_key in self.config.cameras_for_vision_state_concat:
            pvk, thwk = vlm_pixels_key(cam_key), vlm_grid_key(cam_key)
            with torch.no_grad():
                if pvk in batch:
                    # Fast path: the DataLoader workers already ran the (CPU)
                    # image_processor. pixel_values collate as (B, P, dim) and
                    # grid_thw as (B, 3); flatten the patches back to the
                    # (sum_patches, dim) layout the vision tower expects.
                    pv = batch[pvk]
                    image_grid_thw = batch[thwk]
                    if pv.dim() == 3:
                        pv = pv.reshape(-1, pv.shape[-1])
                    if image_grid_thw.dim() == 1:
                        image_grid_thw = image_grid_thw.unsqueeze(0)
                    pixel_values = pv.to(device=device)
                    image_grid_thw = image_grid_thw.to(device=device)
                elif cam_key in batch:
                    # Fallback (inference, or --preprocess_in_workers off): run the
                    # SAME preprocessing here on the raw camera frames.
                    imgs = batch[cam_key]
                    img = imgs[:, -1] if imgs.dim() == 5 else imgs
                    pixel_values, image_grid_thw = preprocess_camera_to_pixels(
                        self.processor.image_processor, img,
                        target_size=self.cam_target_size(cam_key),
                    )
                    pixel_values = pixel_values.to(device=device)
                    image_grid_thw = image_grid_thw.to(device=device)
                else:
                    continue
                self._report_vision_grid(cam_key, image_grid_thw)
                # Call the vision tower directly. `Qwen3VLVisionTransformer`
                # already includes the spatial merger that projects vision
                # features (vision_hidden, e.g. 1024) → text_hidden (2560),
                # AND collapses 2×2 spatial neighbours into one token.
                # `vlm_model.get_image_features` in the FP8 release returns
                # the PRE-merger vision-tower output instead (different last
                # dim, 4× more tokens), so we bypass it here.
                try:
                    vis_tokens = self.visual(
                        pixel_values, grid_thw=image_grid_thw,
                    )
                except TypeError:
                    vis_tokens = self.visual(
                        pixel_values, image_grid_thw=image_grid_thw,
                    )
                vis_tokens = getattr(vis_tokens, "last_hidden_state", vis_tokens)

                # Qwen3-VL's vision tower outputs pre-merger features in
                # vision_hidden (e.g. 1024) with `spatial_merge_size**2`× more
                # tokens than the LLM consumes. The merger then:
                #   1. group 2×2 spatial neighbours into one slot
                #   2. project (vision_hidden × 4) → text_hidden
                # In Qwen2-VL the merger fires inside `visual.__call__`; in
                # Qwen3-VL (incl. the FP8 build) it is a SEPARATE submodule
                # we must call explicitly.
                if vis_tokens.shape[-1] != self.hidden_size:
                    merger = self._find_visual_merger()
                    if merger is None:
                        raise RuntimeError(
                            f"vis_tokens hidden dim {vis_tokens.shape[-1]} != "
                            f"text hidden {self.hidden_size} and no merger / "
                            f"patch_merger submodule found on self.visual or "
                            f"self.vlm_model. Inspect the model's child modules."
                        )
                    try:
                        vis_tokens = merger(vis_tokens)
                    except TypeError:
                        # Some variants take (features, grid_thw)
                        vis_tokens = merger(vis_tokens, image_grid_thw)
                    vis_tokens = getattr(vis_tokens, "last_hidden_state", vis_tokens)

            # Qwen3-VL packs dynamic-resolution vision features as a flat
            # (sum_tokens_across_batch, text_hidden) tensor — each image's
            # tokens are concatenated along the leading dim, not a per-batch
            # axis. At fixed CANONICAL_IMAGE_SIZE every image yields the
            # same N_per_image, so a single reshape recovers the
            # (B, N, hidden) layout the rest of the pipeline expects. If a
            # future API returns (B, N, hidden) directly we keep that branch
            # untouched.
            if vis_tokens.dim() == 2:
                if vis_tokens.shape[-1] != self.hidden_size:
                    raise RuntimeError(
                        f"vis_tokens hidden dim {vis_tokens.shape[-1]} != "
                        f"text hidden {self.hidden_size} after merger. "
                        f"Merger output dim is unexpected — print "
                        f"`{type(self._find_visual_merger()).__name__}` "
                        f"to debug."
                    )
                if vis_tokens.shape[0] % B != 0:
                    raise RuntimeError(
                        f"Cannot unpack vis_tokens of shape {tuple(vis_tokens.shape)} "
                        f"into per-batch tokens: leading dim {vis_tokens.shape[0]} "
                        f"is not divisible by B={B}. Are images of mixed resolution?"
                    )
                vis_tokens = vis_tokens.reshape(B, -1, self.hidden_size)
            all_vis.append(vis_tokens)
            grid_thw_list.append(image_grid_thw[0].detach())
        if not all_vis:
            empty = torch.zeros(B, 0, self.hidden_size, device=device, dtype=torch.bfloat16)
            return empty, []
        return torch.cat(all_vis, dim=1), grid_thw_list

    def _resolve_descs(self, batch: dict, descs_override=None):
        """Single entry point for the instruction strings the model consumes.

        Pulls the task string from the batch and, when use_descriptive_objects
        is on, rewrites ambiguous object/region names via the task_rewrites
        single source of truth. Used by every language read site so the same
        phrasing reaches training, RL rollout, and eval.

        descs_override feeds the contrastive branch's permuted instructions
        through the same path. rewrite_instruction is idempotent (a rewritten
        string is not itself a key in the map), so re-resolving an already
        rewritten override is safe.
        """
        if descs_override is not None:
            descs = descs_override
        else:
            descs = _as_instruction_list(batch.get("task_description"), "task_description")
            if descs is None:
                descs = _as_instruction_list(batch.get("task"), "task")
        if descs and getattr(self.config, "use_descriptive_objects", False):
            descs = [rewrite_instruction(d) for d in descs]
        return descs

    def _format_instruction(self, descs):
        """Raw instruction -> the string actually fed to the VLM.

        Deliberately NOT folded into _resolve_descs: the contrastive branch
        compares descs for equality and builds hard negatives from them, and a
        constant template wrapper on both sides of a pair would only add
        identical text to every negative. Keeping the raw form there and
        formatting here means the template affects the VLM input and nothing else.

        instruction_template wins over chat_directive when both are set -- the
        template can place the instruction mid-prompt, which a prefix cannot.
        """
        tmpl = str(getattr(self.config, "instruction_template", "") or "").strip()
        if tmpl:
            return [tmpl.replace("{instruction}", str(d)) for d in descs]
        directive = str(getattr(self.config, "chat_directive", "") or "").strip()
        return [(f"{directive} {d}" if directive else str(d)) for d in descs]

    def _report_lang_budget(self, texts, lang_ids, lang_mask) -> None:
        """One-time print of the token budget vs. the longest instruction.

        Truncation is silent and destroys exactly the disambiguating tail of the
        CoT rewrites, so make it visible at startup instead of leaving it to be
        inferred from rollout behaviour."""
        if self._lang_len_printed:
            return
        self._lang_len_printed = True
        tok = self.processor.tokenizer
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        i = int(np.argmax(lens))
        full = tok(texts[i], add_special_tokens=False)["input_ids"]
        print(f"[wiltechs_vla] lang budget: max_len={self._lang_max_len}, longest "
              f"instruction in batch={lens[i]} tokens, "
              f"kept={int(lang_mask[i].sum().item())}")
        print(f"[wiltechs_vla]   kept: {tok.decode(lang_ids[i][lang_mask[i]])!r}")
        if lens[i] > self._lang_max_len:
            print(f"[wiltechs_vla]   *** TRUNCATED *** dropped tail: "
                  f"{tok.decode(full[self._lang_max_len:])!r}")

    def _encode_language(self, batch: dict, device: torch.device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        descs = self._resolve_descs(batch)
        if not descs or not any(descs):
            return None
        texts = self._format_instruction(descs)
        inputs = self.processor.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self._lang_max_len, add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(device)
        lang_mask = inputs["attention_mask"].bool().to(device)
        lang_tokens = self.language_model.get_input_embeddings()(input_ids)
        return lang_tokens, lang_mask

    def _get_template_ids(self, device: torch.device) -> tuple:
        """Token ids for the static ChatML pieces (cached after first call).

        Returns (prefix_ids, vision_start_id, vision_end_id, assistant_ids).
        prefix is "<|im_start|>user\\n"; the marker ids bracket each camera's
        vision block; assistant_ids is the "<|im_end|>\\n<|im_start|>assistant\\n"
        tail, which text-first has to append separately because the images —
        not the instruction — now sit last inside the user turn.
        """
        if self._template_ids_cpu is None:
            tok = self.processor.tokenizer
            prefix_ids = tok(
                "<|im_start|>user\n", add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0].long()
            asst_ids = tok(
                "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].long()
            vs = tok.convert_tokens_to_ids("<|vision_start|>")
            ve = tok.convert_tokens_to_ids("<|vision_end|>")
            if vs is None or ve is None or vs == tok.unk_token_id or ve == tok.unk_token_id:
                raise RuntimeError(
                    "use_chat_template=True but the tokenizer has no "
                    "<|vision_start|>/<|vision_end|> special tokens — check the "
                    "Qwen3-VL processor version."
                )
            self._template_ids_cpu = (
                prefix_ids,
                torch.tensor([vs], dtype=torch.long),
                torch.tensor([ve], dtype=torch.long),
                asst_ids,
            )
        return tuple(t.to(device) for t in self._template_ids_cpu)

    # =========================================================================
    # VLM encoder: run all 36 layers, cache K/V from self.capture_layers
    # =========================================================================
    @torch.no_grad()
    def _run_vlm_and_cache_kv(
        self, batch: dict, descs_override=None, vis_pack=None
    ) -> tuple:
        """
        Returns:
          kv_cache:        list of length num_dit_layers, each entry is
                           (K, V) of shape (B, num_kv_heads, L_vlm, head_dim),
                           ordered by increasing VLM layer index so DiT layer i
                           reads self.capture_layers[i]. K is post-M-RoPE.
          vlm_kv_pad_mask: (B, L_vlm) bool — True at non-padded positions.
                           Used by DiT cross-attn to ignore padded language slots.
          vis_mask:        (L_vlm,) bool — True at actual VISION token positions
                           (excludes the ChatML prefix/markers when
                           use_chat_template is on). Sample-independent.
          lang_span:       (start, end) index range of the PER-SAMPLE instruction
                           tokens inside the VLM sequence. Everything outside it
                           is identical across the batch. Under text-last this is
                           a trailing slice, which is what the legacy contrastive
                           branch shuffles; under text_first it sits mid-sequence
                           and shuffling the slice is NOT valid (see compute_loss).
          vis_pack:        the (vis_tokens, grid_thw_list) ViT output, so a second
                           call with a different instruction can skip re-encoding
                           the images.

        descs_override replaces the batch's instructions (contrastive negatives).
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        if vis_pack is not None:
            vis_tokens, grid_thw_list = vis_pack
        else:
            vis_tokens, grid_thw_list = self._encode_images(batch, B)
        vis_pack_out = (vis_tokens, grid_thw_list)
        L_vis = vis_tokens.shape[1]

        descs = self._resolve_descs(batch, descs_override)
        have_lang = bool(descs) and any(descs)
        use_template = bool(getattr(self.config, "use_chat_template", False)) and have_lang
        text_first = self.text_first and have_lang
        embed_tokens = self.language_model.get_input_embeddings()

        if text_first:
            # ---- instruction BEFORE images ------------------------------
            # Under the VLM's causal mask this makes every vision patch's K/V
            # at every layer conditioned on the instruction, so the DiT's
            # cross-attention reads a language-grounded feature map rather than
            # a language-blind one.
            m = self.spatial_merge_size
            cam_sizes = [
                int(g[0].item()) * (int(g[1].item()) // m) * (int(g[2].item()) // m)
                for g in grid_thw_list
            ]
            cam_tokens = list(vis_tokens.split(cam_sizes, dim=1)) if cam_sizes else []

            texts = self._format_instruction(descs)
            lang = self.processor.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self._lang_max_len, add_special_tokens=not use_template,
            )
            lang_ids = lang["input_ids"].to(device)
            lang_mask = lang["attention_mask"].bool().to(device)
            L_lang = lang_ids.shape[1]
            self._report_lang_budget(texts, lang_ids, lang_mask)
            lang_emb = embed_tokens(lang_ids)
            lang_emb = torch.where(
                lang_mask.unsqueeze(-1), lang_emb, torch.zeros_like(lang_emb),
            )

            parts: list = []
            segments: list[tuple[str, object]] = []
            vis_flags: list[bool] = []
            head_len = 0
            asst_ids = None
            if use_template:
                prefix_ids, vs_id, ve_id, asst_ids = self._get_template_ids(device)
                head_len = prefix_ids.shape[0]
                parts.append(embed_tokens(prefix_ids).unsqueeze(0).expand(B, -1, -1))
                segments.append(("text", head_len))
                vis_flags += [False] * head_len

            parts.append(lang_emb)
            segments.append(("text", L_lang))
            vis_flags += [False] * L_lang

            for ct, g in zip(cam_tokens, grid_thw_list):
                if use_template:
                    vs_emb = embed_tokens(vs_id).unsqueeze(0).expand(B, -1, -1)
                    ve_emb = embed_tokens(ve_id).unsqueeze(0).expand(B, -1, -1)
                    parts += [vs_emb, ct, ve_emb]
                    segments += [("text", 1), ("image", g), ("text", 1)]
                    vis_flags += [False] + [True] * ct.shape[1] + [False]
                else:
                    parts.append(ct)
                    segments.append(("image", g))
                    vis_flags += [True] * ct.shape[1]

            if use_template:
                parts.append(embed_tokens(asst_ids).unsqueeze(0).expand(B, -1, -1))
                segments.append(("text", asst_ids.shape[0]))
                vis_flags += [False] * asst_ids.shape[0]

            vlm_seq = torch.cat(parts, dim=1).to(torch.bfloat16)
            L_vlm = vlm_seq.shape[1]
            lang_span = (head_len, head_len + L_lang)
            vis_mask = torch.tensor(vis_flags, device=device, dtype=torch.bool)
            # Padded instruction positions now sit MID-sequence; they are masked
            # out as attention KEYS, so the images never read them. M-RoPE
            # positions are built from the padded length uniformly across the
            # batch, which is a constant offset for the image block — harmless
            # under RoPE.
            vlm_kv_pad_mask = torch.cat([
                torch.ones(B, head_len, device=device, dtype=torch.bool),
                lang_mask,
                torch.ones(B, L_vlm - head_len - L_lang, device=device, dtype=torch.bool),
            ], dim=1)

            if not self._template_format_printed:
                self._template_format_printed = True
                print(f"[wiltechs_vla] TEXT-FIRST layout ON "
                      f"(chat_template={use_template}) — L_vlm={L_vlm}, "
                      f"lang span={lang_span}, L_vis={L_vis}")
        elif use_template:
            # ChatML wrapping (in-distribution for the instruct-tuned VLM):
            #   <|im_start|>user\n
            #   (<|vision_start|> [cam tokens] <|vision_end|>) × num_cameras
            #   {directive }{task}<|im_end|>\n<|im_start|>assistant\n
            # Per-camera markers match Qwen's multi-image pretraining; the
            # trailing assistant header adds "answer preparation" registers
            # the DiT can cross-attend to.
            m = self.spatial_merge_size
            cam_sizes = [
                int(g[0].item()) * (int(g[1].item()) // m) * (int(g[2].item()) // m)
                for g in grid_thw_list
            ]
            cam_tokens = list(vis_tokens.split(cam_sizes, dim=1)) if cam_sizes else []

            texts = [
                t + "<|im_end|>\n<|im_start|>assistant\n"
                for t in self._format_instruction(descs)
            ]
            suf = self.processor.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self._lang_max_len + 24, add_special_tokens=False,
            )
            suffix_ids = suf["input_ids"].to(device)
            suffix_mask = suf["attention_mask"].bool().to(device)
            suffix_emb = embed_tokens(suffix_ids)
            # Zero-out padded embeddings so their K/V are uninformative.
            suffix_emb = torch.where(
                suffix_mask.unsqueeze(-1), suffix_emb, torch.zeros_like(suffix_emb),
            )

            prefix_ids, vs_id, ve_id, _ = self._get_template_ids(device)
            prefix_emb = embed_tokens(prefix_ids).unsqueeze(0).expand(B, -1, -1)
            vs_emb = embed_tokens(vs_id).unsqueeze(0).expand(B, -1, -1)
            ve_emb = embed_tokens(ve_id).unsqueeze(0).expand(B, -1, -1)

            parts = [prefix_emb]
            segments: list[tuple[str, object]] = [("text", prefix_ids.shape[0])]
            vis_flags = [False] * prefix_ids.shape[0]
            for ct, g in zip(cam_tokens, grid_thw_list):
                parts += [vs_emb, ct, ve_emb]
                segments += [("text", 1), ("image", g), ("text", 1)]
                vis_flags += [False] + [True] * ct.shape[1] + [False]
            parts.append(suffix_emb)
            segments.append(("text", suffix_ids.shape[1]))
            vis_flags += [False] * suffix_ids.shape[1]

            vlm_seq = torch.cat(parts, dim=1).to(torch.bfloat16)
            L_vlm = vlm_seq.shape[1]
            text_start = L_vlm - suffix_ids.shape[1]
            vis_mask = torch.tensor(vis_flags, device=device, dtype=torch.bool)
            vlm_kv_pad_mask = torch.cat(
                [torch.ones(B, text_start, device=device, dtype=torch.bool), suffix_mask],
                dim=1,
            )

            if not self._template_format_printed:
                self._template_format_printed = True
                decoded = self.processor.tokenizer.decode(
                    suffix_ids[0][suffix_mask[0]], skip_special_tokens=False,
                )
                print(f"[wiltechs_vla] chat template ON (text-last) — L_vlm={L_vlm} "
                      f"(prefix {prefix_ids.shape[0]} | {len(cam_tokens)} cams ×(1+vis+1) "
                      f"| suffix {suffix_ids.shape[1]})")
                print(f"[wiltechs_vla]   suffix[0]: {decoded!r}")
            lang_span = (text_start, L_vlm)
        else:
            # Legacy raw concatenation [vision | task].
            lang_result = self._encode_language(batch, device)
            if lang_result is not None:
                lang_tokens, lang_mask = lang_result
                lang_tokens = lang_tokens.to(vis_tokens.dtype)
                # Zero-out padded language embeddings so their K/V are uninformative.
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
            text_start = L_vis
            vis_mask = torch.zeros(L_vlm, device=device, dtype=torch.bool)
            vis_mask[:L_vis] = True

            segments = [("image", g) for g in grid_thw_list]
            if L_lang > 0:
                segments.append(("text", L_lang))

            # Valid-position mask: vision always valid; language follows lang_mask.
            if lang_mask is not None:
                vis_ones = torch.ones(B, L_vis, device=device, dtype=torch.bool)
                vlm_kv_pad_mask = torch.cat([vis_ones, lang_mask], dim=1)
            else:
                vlm_kv_pad_mask = torch.ones(B, L_vlm, device=device, dtype=torch.bool)
            lang_span = (text_start, L_vlm)

        # M-RoPE position_ids — image segments get (t, h, w), text monotonic
        position_ids = _build_mrope_position_ids(
            segments, B=B,
            spatial_merge_size=self.spatial_merge_size, device=device,
        )
        cos, sin = self.language_model.rotary_emb(vlm_seq, position_ids)

        # Causal mask + key-padding mask for VLM self-attention (matches the
        # mask shape Qwen3-VL was pretrained with). Shape: (B, 1, L, L).
        causal = torch.triu(
            torch.full((L_vlm, L_vlm), float("-inf"), device=device, dtype=vlm_seq.dtype),
            diagonal=1,
        )
        full_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L_vlm, L_vlm).clone()
        key_pad = ~vlm_kv_pad_mask                            # True = pad
        full_mask.masked_fill_(key_pad.unsqueeze(1).unsqueeze(1), float("-inf"))

        # Layer-by-layer forward, capturing K/V from self.capture_layers.
        # Appended in loop order, so kv_cache is sorted by VLM layer index and
        # kv_cache[-1] is the deepest captured layer.
        capture_set = set(self.capture_layers)
        hidden = vlm_seq
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.language_model.layers):
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

            # Capture K (post-RoPE) and V — these are the cross-attn memory.
            if i in capture_set:
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

        return kv_cache, vlm_kv_pad_mask, vis_mask, lang_span, vis_pack_out

    # =========================================================================
    # DiT-side helpers: robot CNN, latents, time, input assembly
    # =========================================================================
    def _compute_robot_tokens(self, batch: dict) -> Optional[torch.Tensor]:
        if self.robot_visual_encoder is None:
            return None
        toks_list = []
        # CNN-specific camera list (wrist-only specialization) when set; else
        # legacy behavior of re-encoding every VLM scene view.
        cnn_cams = getattr(self.config, "robot_cnn_cameras", None) \
            or self.config.cameras_for_vision_state_concat
        for cam_key in cnn_cams:
            if cam_key not in batch:
                continue
            img = batch[cam_key]
            if img.dim() == 5:
                img = img[:, -1]
            toks_list.append(self.robot_visual_encoder(img.float()))
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
        self,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_kv_pad_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Distill the top (most semantic) captured VLM layer's KV cache into the
        latent 'thought' tokens via the learned-query Q-Former. Noise-independent
        → computed once per forward and shared across all denoising steps."""
        if self.num_latent_tokens == 0:
            return None
        vlm_k, vlm_v = kv_cache[-1]
        return self.latent_qformer(vlm_k, vlm_v, vlm_kv_pad_mask)

    def _build_dit_input(
        self,
        batch: dict,
        noisy_actions: torch.Tensor,
        robot_tokens: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """
        Returns:
          dit_seq:          (B, L_dit, H)
          action_start_idx: int — index where action tokens begin (for readout)
        """
        B, H, _ = noisy_actions.shape
        dtype = noisy_actions.dtype

        state = batch["observation.state"].float()
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state = state.nan_to_num(0.0).clamp(-10.0, 10.0)
        state_tok = self.state_encoder(state).to(dtype)
        if state_tok.shape[1] > 1:
            state_tok = state_tok[:, -1:]

        action_emb = self.action_in_proj(noisy_actions) + self.action_pos_emb[:, :H]
        action_emb = action_emb.to(dtype)
        # Reference scale for the register-token diagnostic. Each DiT layer
        # RMSNorms per token, so a small ratio is not by itself "inert" -- a
        # ratio that never MOVES off its init is.
        self._last_action_emb_rms = float(
            action_emb.detach().float().pow(2).mean().sqrt())

        # Layout: [state(1), (robot(R'))?, (latent(K))?, register(R), action(H)]
        #
        # Ordered by role, and the order is load-bearing because self-attention
        # is causal. Everything that CARRIES observation comes first: state, then
        # the RobotCNN pixels, then the latent Q-Former's VLM summary. The
        # registers come last before the actions, so they can read all of it --
        # they are the scratchpad the stack accumulates into, and a scratchpad
        # that cannot see its inputs is just 32 more learned constants.
        #
        # The registers used to sit directly after the state, ahead of the robot
        # tokens, which left them blind to the one module that supplies fine
        # visual detail while still taking ~26-38% of the action tokens' self
        # attention. Moving them behind costs nothing: no parameter is indexed by
        # sequence position (action_pos_emb covers the action slice only), so
        # this is a pure reordering.
        parts = [state_tok]
        if robot_tokens is not None:
            parts.append(robot_tokens.to(dtype))
        if latents is not None:
            parts.append(latents.to(dtype))
        if self.register_tokens is not None:
            parts.append(self.register_tokens.expand(B, -1, -1).to(dtype))
        parts.append(action_emb)
        seq = torch.cat(parts, dim=1)

        # action starts at position = len of everything before it
        action_start_idx = seq.shape[1] - H
        return seq, action_start_idx

    def _build_dit_self_attn_mask(self, L_dit: int, device, dtype) -> torch.Tensor:
        """Full left-to-right causal mask. state / register / robot / latent all
        share the same causal regime as action tokens — they only see earlier
        positions, action tokens see everything before them including
        themselves at their position. Action_pos_emb gives each action_t a
        distinguishing position embedding."""
        return torch.triu(
            torch.full((L_dit, L_dit), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )

    # =========================================================================
    # Diagnostic: per-region attention mass from action queries
    # =========================================================================
    @torch.no_grad()
    def _compute_attention_mass(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        regions: dict[str, Optional[tuple[int, int]]],
    ) -> dict[str, float]:
        """Re-run the LAST DiT layer's self-attn Q·K^T softmax manually and
        report, for each region (state/register/robot/latent/action), the average
        attention mass that the action queries place on it.

        SDPA doesn't expose softmax weights, so this re-projects Q/K once;
        cost ≈ one extra layer's self-attn projection.
        """
        layer = self.dit_layers[-1]
        mod = layer.adaLN_modulation(t_emb)
        chunks = mod.chunk(9, dim=-1)
        s_sa, sc_sa = chunks[0], chunks[1]
        h = _modulate(layer.sa_norm(x), s_sa, sc_sa)

        B, L, _ = h.shape
        H, Hk, D = layer.sa_num_heads, layer.sa_num_kv_heads, layer.sa_head_dim
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

    @torch.no_grad()
    def _compute_cross_attention_mass(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        kv_last: tuple[torch.Tensor, torch.Tensor],
        vlm_kv_pad_mask: Optional[torch.Tensor],
        action_start: int,
        action_len: int,
        vis_mask: Optional[torch.Tensor],
    ) -> dict[str, float]:
        """Cross-attn mass at this model's last DiT layer. See
        cross_attention_mass() — WiltechsMoE calls that directly, once per
        expert, since it has no single 'last layer'."""
        return cross_attention_mass(
            self.dit_layers[-1], x, t_emb, kv_last, vlm_kv_pad_mask,
            action_start, action_len, vis_mask,
        )

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
        vis_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One DiT denoising step. Returns velocity prediction (B, H, action_dim)."""
        device = noisy_actions.device
        dtype = noisy_actions.dtype

        # Time embedding
        t_emb_raw = create_sinusoidal_pos_embedding(timesteps, self.dit_hidden).to(dtype)
        t_emb = self.time_embedder(t_emb_raw.float()).to(dtype)

        # Build sequence
        dit_seq, action_start_idx = self._build_dit_input(
            batch, noisy_actions, robot_tokens, latents,
        )
        L_dit = dit_seq.shape[1]
        causal_mask = self._build_dit_self_attn_mask(L_dit, device, dtype)

        # Region boundaries (used by attention-mass diagnostic). Layout:
        #   [state(1), (robot(R'))?, (latent(K))?, register(R), action(H)]
        # Offsets are ACCUMULATED rather than written as literals, and the tuple
        # below MUST list the blocks in the same order _build_dit_input
        # concatenates them -- get that wrong and every region is mislabelled
        # while the percentages still sum to 100, which looks entirely fine.
        reg_len = self.register_tokens.shape[1] if self.register_tokens is not None else 0
        robot_len = robot_tokens.shape[1] if robot_tokens is not None else 0
        latent_len = latents.shape[1] if latents is not None else 0
        H_horizon = noisy_actions.shape[1]
        off = 1  # state occupies position 0
        regions: dict[str, Optional[tuple[int, int]]] = {"state": (0, 1)}
        for name, n in (("robot", robot_len), ("latent", latent_len), ("register", reg_len)):
            regions[name] = (off, n) if n > 0 else None
            off += n
        assert off == action_start_idx, (
            f"region offsets ({off}) disagree with action_start_idx "
            f"({action_start_idx}) -- the loop above is out of sync with "
            f"_build_dit_input's concatenation order.")
        regions["action"] = (action_start_idx, H_horizon)

        # Run DiT layers, each cross-attending to its paired VLM cache.
        # When `gradient_checkpointing` is on we recompute the layer in
        # backward instead of storing its activations — the VLM K/V tensors
        # are already detached, so checkpointing only re-runs DiT compute.
        x = dit_seq
        use_ckpt = self.gradient_checkpointing and self.training
        capture = self._capture_attention_stats
        # Sample cross-attn mass at FOUR depths, not just the last layer.
        # WiltechsMoE's per-expert breakdown showed the shares swing enormously
        # with depth on the same model — 44% language at VLM layer 8 against 91%
        # at layer 35 — so a last-layer-only reading says nothing about whether
        # the shallow, geometric part of the stack is grounding on vision. These
        # indices are the last layer of each quarter, matching the MoE's four
        # expert-band boundaries when num_dit_layers=36, so the two models'
        # numbers line up depth for depth.
        n_l = len(self.dit_layers)
        depth_idx = sorted({max(0, (n_l * q) // 4 - 1) for q in (1, 2, 3, 4)})
        per_depth: list[dict] = []

        for i, layer in enumerate(self.dit_layers):
            # The diagnostics measure this layer's INPUT, so hold onto it and
            # run them AFTER the real forward. Ordering, not decoration:
            #
            # A no_grad capture fills the autocast weight cache with GRAD-LESS
            # bf16 casts of this layer's sa_q/sa_k/ca_q/adaLN weights. Running
            # it first meant the real forward reused those casts and the weights
            # received no gradient, so the old code called
            # torch.clear_autocast_cache() to undo it -- one line before this
            # layer's CHECKPOINTED call. That is what raised CheckpointError:
            # the forward then ran with an empty cache and recorded the weight
            # casts, while the backward recompute (autocast context long since
            # exited) recorded a different set, so the saved tensor list came
            # back shifted by one and every entry mismatched.
            #
            # Capturing after the forward fixes both at once: the real forward
            # populates the cache with grad-tracking casts, the capture merely
            # reuses them, and nothing mutates global cast state between a
            # checkpoint's forward and its recompute. Each layer owns its
            # weights, so a later layer cannot inherit an earlier capture's cast.
            x_in = x if capture else None

            vlm_k, vlm_v = kv_cache[i]
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, t_emb, vlm_k, vlm_v, vlm_kv_pad_mask, causal_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x, t_emb=t_emb,
                    vlm_k=vlm_k, vlm_v=vlm_v,
                    vlm_kv_pad_mask=vlm_kv_pad_mask,
                    self_attn_mask=causal_mask,
                )

            if capture and i in depth_idx:
                per_depth.append(cross_attention_mass(
                    layer, x_in, t_emb, kv_cache[i], vlm_kv_pad_mask,
                    action_start_idx, H_horizon, vis_mask,
                ))
            # Self-attn mass is only meaningful at the LAST layer — the "final
            # say" before the action readout.
            if capture and i == n_l - 1:
                self._last_attention_stats = self._compute_attention_mass(
                    x_in, t_emb, causal_mask, regions,
                )
                # one-shot: don't capture again if _run_dit is called twice
                # (e.g. for the contrastive-language v_wrong forward).
                self._capture_attention_stats = False
                self._last_cross_attention_stats = _merge_depths(
                    per_depth, [self.capture_layers[j] for j in depth_idx])

        action_out = self.final_norm(x[:, action_start_idx:])
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
        t = torch.rand(B, device=device)
        return t * 0.998 + 0.001

    def compute_loss(self, batch: dict) -> torch.Tensor:
        actions = batch["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0)
        B = actions.shape[0]
        device = actions.device

        # ── Encoder: run VLM once, cache KV ─────────────────────────
        kv_cache, vlm_kv_pad_mask, vis_mask, lang_span, vis_pack = \
            self._run_vlm_and_cache_kv(batch)

        # ── Vision-KV dropout (training only) ───────────────────────
        # Mask a random fraction of VISION positions in the cross-attn pad
        # mask. The VLM forward above already ran on the full sequence, so
        # this only hides vision slots from the DiT / QFormer consumers —
        # language slots are never dropped. This weakens the ~25:1
        # vision:language shortcut and is the KV-cache analog of the
        # interleaved model's vision-stream dropout (which was decisive for
        # language reliance there). Applied BEFORE latents/DiT/contrastive
        # so all consumers see one consistent mask.
        vkv_p = float(getattr(self.config, "vision_kv_dropout_prob", 0.0))
        n_vis = int(vis_mask.sum().item())
        vkv_drop = None
        if self.training and vkv_p > 0.0 and n_vis > 0:
            vis_idx = vis_mask.nonzero(as_tuple=True)[0]
            keep = torch.rand(B, n_vis, device=device) > vkv_p
            # Never let a sample lose its entire memory (all-pad rows make
            # SDPA emit NaN when no language tokens exist either).
            dead = ~keep.any(dim=1)
            if dead.any():
                keep[dead, 0] = True
            vlm_kv_pad_mask = vlm_kv_pad_mask.clone()
            vlm_kv_pad_mask[:, vis_idx] &= keep
            # Remembered so the contrastive negative gets the SAME vision KV
            # dropout — otherwise v_t and v_wrong differ because of dropout,
            # not because of the language, and the hinge measures the wrong
            # thing. (Only matters on the text_first re-run path, which builds
            # a fresh mask instead of cloning this one.)
            vkv_drop = (vis_idx, keep)

        # ── DiT-side conditioning that does NOT depend on noise ─────
        robot_tokens = self._compute_robot_tokens(batch)
        latents = self._generate_latents(kv_cache, vlm_kv_pad_mask)

        # ── Flow matching: build noisy actions, predict velocity ────
        noise = self.sample_noise(actions.shape, device)
        t = self.sample_time(B, device)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions
        u_t = noise - actions
        x_t_bf16 = x_t.to(torch.bfloat16)  # reused by the contrastive forward

        v_t = self._run_dit(
            batch, x_t_bf16, t, kv_cache, vlm_kv_pad_mask,
            robot_tokens, latents,
            vis_mask=vis_mask,
        ).float()

        # Per-position weighting (n_action_steps gets full weight; future tail
        # gets future_steps_weight; optional exponential decay).
        loss = F.mse_loss(v_t, u_t, reduction="none")
        if self.config.action_dim_weights:
            dim_w = torch.tensor(self.config.action_dim_weights, device=loss.device, dtype=loss.dtype)
            loss = loss * dim_w[None, None, :]

        H = loss.shape[1]
        # loss_exec_steps owns the loss boundary; n_action_steps is inference-only
        # (how many actions the queue pops per replan). 0 falls back to the full
        # horizon = no down-weighting. It used to fall back to n_action_steps,
        # which coupled the LOSS to an INFERENCE knob -- and since the trainer
        # pinned n_action_steps to horizon, the slice below was empty and
        # future_steps_weight did nothing. Falling back to H rather than
        # n_action_steps changes no existing checkpoint: every one this trainer
        # produced had n_action_steps == horizon, so the two are the same number.
        n_exec = int(getattr(self.config, "loss_exec_steps", 0) or 0) or H
        n_exec = max(1, min(n_exec, H))
        pos_w = torch.ones(H, device=loss.device, dtype=loss.dtype)
        pos_w[n_exec:] = self.config.future_steps_weight
        if self.config.pos_decay_lambda > 0.0:
            pos = torch.arange(H, device=loss.device, dtype=loss.dtype)
            pos_w = pos_w * torch.exp(-self.config.pos_decay_lambda * pos)
        loss = loss * pos_w[None, :, None]

        # action_is_pad / action_dim_pad masking (same as before)
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
        denom = (pos_w[None, :, None] * valid_cells).sum().clamp(min=1e-6)
        main_loss = loss.sum() / denom

        # ── Gripper auxiliary BCE ───────────────────────────────────
        # See gripper_bce_weight in the config for why this exists. The action
        # is recovered in closed form from the velocity prediction:
        #
        #   x_t = t*noise + (1-t)*a   and   u_t = noise - a
        #   =>  x_t - t*u_t = a       exactly
        #
        # so a_hat = x_t - t*v_hat needs no integration and no second forward.
        gr_w = float(getattr(self.config, "gripper_bce_weight", 0.0) or 0.0)
        gr_thr = float(getattr(self.config, "gripper_threshold_norm", float("nan")))
        gripper_v = 0.0
        gripper_open_frac = float("nan")
        if gr_w > 0.0 and gr_thr == gr_thr:   # NaN means "not calibrated"
            g = int(getattr(self.config, "gripper_action_dim", -1))
            temp = max(float(getattr(self.config, "gripper_bce_temp", 0.25)), 1e-3)
            a_hat = x_t - t_exp * v_t
            logit = (a_hat[..., g] - gr_thr) / temp
            target = (actions[..., g] > gr_thr).to(logit.dtype)
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
            # Same position weighting and padding mask as the main loss, so the
            # two terms agree about which cells are real and how much the tail
            # of the horizon counts.
            w = pos_w[None, :] * valid_t

            # Class balancing. WITHOUT this the term has a strong trivial
            # optimum and measurably sits in it: the demos are ~89% "open", so
            # "always open" already scores a low BCE, and validation showed
            # exactly that signature -- all-steps agreement climbing to 89.2%
            # (= the demo open-fraction) while agreement AT THE TRANSITIONS,
            # the only steps that decide a grasp, stayed at chance (42.9%,
            # 51.0%, 47.8% across checks). Two independently configured runs
            # converged on the same ~89%, which is what majority-class
            # prediction looks like, not what learning looks like.
            #
            # What balancing buys, stated exactly: the closing class carries
            # 10.8% of the weight and after balancing carries 50% -- 4.6x more
            # gradient on the only cells that mark a grasp. And the floor for
            # an input-independent predictor moves from H(0.892) = 0.342 to
            # ln 2 = 0.693, so twice as much of the loss is headroom that only
            # actually predicting the transitions can claim.
            #
            # Reweighting so each class carries half the total mass. Done as a per-cell
            # weight folded into `w` rather than BCEWithLogits' `pos_weight`
            # because the loss is a WEIGHTED MEAN by the same `w`: the scale
            # therefore stays interpretable (chance ~= 0.693 at any imbalance)
            # and gr_w keeps its meaning. `pos_weight` scales only the positive
            # term against an unchanged denominator, so a batch that happens to
            # be all-open would drive the whole term to zero.
            #
            # p is estimated per batch under the same weights the loss sums
            # over -- pos_decay_lambda shortens the effective horizon, so the
            # unweighted prior is the wrong one. The estimate is noisy at ~15%
            # on the minority weight, which is immaterial next to gr_w=0.05.
            #
            # p is measured whether or not it is USED, because it is what tells
            # the unbalanced floor H(p) apart from the balanced ln 2 -- an
            # ablation run with no p reported would have no baseline to read its
            # own loss against.
            p = ((target * w).sum() / w.sum().clamp(min=1e-6)).clamp(1e-3, 1 - 1e-3)
            gripper_open_frac = float(p.detach())
            if bool(getattr(self.config, "gripper_class_balance", True)):
                w = w * torch.where(target > 0.5, 0.5 / p, 0.5 / (1.0 - p))
            gripper_loss = (bce * w).sum() / w.sum().clamp(min=1e-6)
            gripper_v = float(gripper_loss.detach())
            main_loss = main_loss + gr_w * gripper_loss

        # ── Contrastive language loss ───────────────────────────────
        # Build a "wrong-language" prediction and require it to differ from the
        # right-language one by at least `contrastive_margin`.
        #
        # HOW the negative is built depends on the layout:
        #  * text-last — permute the per-sample language slice of each VLM KV
        #    pair across the batch. Cheap: no second VLM forward.
        #  * text_first — the instruction is baked into every vision K/V, so
        #    swapping only the language slice is self-inconsistent. The frozen
        #    LM is re-run with permuted instructions instead; the ViT output is
        #    reused via vis_pack, so the extra cost is the 36 LM layers under
        #    no_grad, with no activations retained.
        contrastive_w = float(getattr(self.config, "contrastive_loss_weight", 0.0))
        contrastive_v = 0.0
        lang_start, lang_end = lang_span
        L_lang_total = lang_end - lang_start
        descs = self._resolve_descs(batch)
        have_descs = descs is not None and len(descs) == B
        # The re-run path needs the actual strings; there is no valid KV slice
        # to swap without them.
        can_contrast = have_descs if self.text_first else True
        if (
            self.training and contrastive_w > 0.0
            and L_lang_total > 0 and B >= 2 and can_contrast
        ):
            # Pick the wrong-language partner for each sample. Hard negatives
            # (most-similar DIFFERENT instruction) focus the hinge on confusable
            # minimal pairs that fail at eval; random pairs (legacy) are almost
            # always grossly different and trivially satisfied.
            if (
                getattr(self.config, "contrastive_hard_negatives", False)
                and have_descs
            ):
                perm, pair_diff = _hard_negative_perm(descs, device)
            else:
                perm = torch.randperm(B, device=device)
                if (perm == torch.arange(B, device=device)).any():
                    perm = torch.roll(perm, shifts=1, dims=0)

                # Skip pairs whose language string actually matches (cross-dataset
                # collisions, e.g. "Grasp a lego block ..." appearing 4× across
                # community).
                if have_descs:
                    perm_cpu = perm.detach().cpu().tolist()
                    pair_diff = torch.tensor(
                        [descs[i] != descs[perm_cpu[i]] for i in range(B)],
                        device=device, dtype=torch.bool,
                    )
                else:
                    pair_diff = torch.ones(B, device=device, dtype=torch.bool)

            if pair_diff.any():
                if self.text_first:
                    # _hard_negative_perm is deliberately NOT a bijection, so
                    # descs_perm is a different multiset from descs and can pad
                    # to a different length than the positive forward. Nothing
                    # downstream mixes the two — shuffled_cache and
                    # shuffled_pad_mask are used together — so that is fine, but
                    # it is why the mask must come from this call rather than
                    # being derived from vlm_kv_pad_mask.
                    perm_cpu = perm.detach().cpu().tolist()
                    descs_perm = [descs[i] for i in perm_cpu]
                    shuffled_cache, shuffled_pad_mask, shuf_vis_mask, _, _ = \
                        self._run_vlm_and_cache_kv(
                            batch, descs_override=descs_perm, vis_pack=vis_pack)
                    if vkv_drop is not None:
                        # Re-apply the SAME vision dropout. The vision block is
                        # a fixed-size suffix in both forwards, so index it from
                        # this call's own vis_mask rather than reusing positions
                        # computed against a possibly different L_vlm.
                        _, v_keep = vkv_drop
                        shuf_vis_idx = shuf_vis_mask.nonzero(as_tuple=True)[0]
                        if shuf_vis_idx.numel() == v_keep.shape[1]:
                            shuffled_pad_mask = shuffled_pad_mask.clone()
                            shuffled_pad_mask[:, shuf_vis_idx] &= v_keep
                else:
                    shuffled_cache = []
                    for K, V in kv_cache:
                        K_shuf = K.clone()
                        V_shuf = V.clone()
                        K_shuf[:, :, lang_start:, :] = K[perm, :, lang_start:, :]
                        V_shuf[:, :, lang_start:, :] = V[perm, :, lang_start:, :]
                        shuffled_cache.append((K_shuf, V_shuf))
                    shuffled_pad_mask = vlm_kv_pad_mask.clone()
                    shuffled_pad_mask[:, lang_start:] = vlm_kv_pad_mask[perm][:, lang_start:]

                # Recompute the latents from the wrong-language cache so the
                # QFormer path is ALSO language-forced. Passing the correct-
                # language latents here would let the hinge be satisfied by
                # the DiT's direct cross-attention alone, leaving the latent
                # tokens free to ignore language entirely.
                latents_wrong = self._generate_latents(shuffled_cache, shuffled_pad_mask)

                v_wrong = self._run_dit(
                    batch, x_t_bf16, t,
                    shuffled_cache, shuffled_pad_mask,
                    robot_tokens, latents_wrong,
                ).float()

                diff_sq = (v_t - v_wrong).pow(2).mean(dim=[1, 2])
                margin = float(getattr(self.config, "contrastive_margin", 0.05))
                hinge = F.relu(margin - diff_sq) * pair_diff.float()
                n_valid = pair_diff.float().sum().clamp(min=1.0)
                loss_contrastive = hinge.sum() / n_valid
                contrastive_v = float(loss_contrastive.detach())
                main_loss = main_loss + contrastive_w * loss_contrastive

        # "main" is the flow-matching term ALONE -- every auxiliary weight is
        # subtracted back out. The validation loss reads this key, so leaving an
        # aux term in it would make the val/train comparison shift the moment a
        # weight changed, for reasons having nothing to do with the model.
        self._last_loss_components = {
            "main": float(main_loss.detach() - contrastive_w * contrastive_v
                          - gr_w * gripper_v),
            "contrastive": contrastive_v,
            "gripper": gripper_v,
            # Reported so the balancing is auditable from the training log: this
            # is the class prior the reweighting was derived from, and a
            # balanced term sitting at ~0.693 means the model is still
            # predicting the majority class.
            "gripper_open_frac": gripper_open_frac,
        }
        return main_loss

    def forward(self, batch: dict) -> tuple:
        if self.training:
            return self.compute_loss(batch), {}
        return self.sample_actions(batch), {}

    def flow_actions_from_noise(self, batch: dict, x_init: torch.Tensor) -> torch.Tensor:
        """Deterministic flow ODE solution from a GIVEN initial noise x_init
        (B, horizon, action_dim), returned for the FULL horizon in normalized
        action space. Differentiable through the DiT / robot CNN / QFormer (the
        frozen VLM encoder still runs under no_grad inside _run_vlm_and_cache_kv).

        Used by RL (GRPO): the policy is N(flow_actions_from_noise(s, x1), sigma^2)
        conditioned on the stored noise latent x1, so action log-probs are exact
        and importance ratios are computable. Caller controls grad/no_grad and
        autocast context. Mirrors sample_actions' integration exactly.
        """
        B = x_init.shape[0]
        device = x_init.device

        kv_cache, vlm_kv_pad_mask, *_ = self._run_vlm_and_cache_kv(batch)
        robot_tokens = self._compute_robot_tokens(batch)
        latents = self._generate_latents(kv_cache, vlm_kv_pad_mask)

        N = int(getattr(self.config, "num_inference_steps", 5))
        x_t = x_init.float()
        dt = -1.0 / N
        t = torch.ones(B, device=device, dtype=torch.float32)
        for _ in range(N):
            v_t = self._run_dit(
                batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
                robot_tokens, latents,
            ).float()
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t

    @torch.no_grad()
    def sample_actions(self, batch: dict, full_horizon: bool = False) -> torch.Tensor:
        """(B, n_action_steps, action_dim), or the full horizon if requested.

        full_horizon exists so that MEASUREMENT does not move when an
        inference knob moves. The validation metrics integrate the chunk to a
        terminal position and count gripper transitions over it; truncating to
        n_action_steps would make both quantities mean something different at
        every setting -- dropping n_action_steps from 64 to 4 would shrink the
        terminal-error denominator from 6.4s of motion to 0.4s and cut the
        transition sample from ~786 to a handful, silently ending the
        comparison with every previous run.
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )

        with autocast_ctx:
            # Encoder pass: run VLM once, get KV cache
            kv_cache, vlm_kv_pad_mask, *_ = self._run_vlm_and_cache_kv(batch)

            # DiT-side static conditioning (same across denoising steps)
            robot_tokens = self._compute_robot_tokens(batch)
            latents = self._generate_latents(kv_cache, vlm_kv_pad_mask)

            # Flow matching: N=5 inference steps (Xiaomi-Robotics-0 standard).
            # config.num_inference_steps can override.
            N = int(getattr(self.config, "num_inference_steps", 5))
            x_t = self.sample_noise(
                (B, self.config.horizon, self.config.action_dim), device=device,
            )
            dt = -1.0 / N
            t = torch.ones(B, device=device, dtype=torch.float32)

            for _ in range(N):
                v_t = self._run_dit(
                    batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
                    robot_tokens, latents,
                ).float()
                x_t = x_t + dt * v_t
                t = t + dt

        return x_t if full_horizon else x_t[:, : self.config.n_action_steps]

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
