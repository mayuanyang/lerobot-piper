# InterleavedFlowMatchingTransformer — Architecture Diagram

## Overview

SmolVLA-style **joint attention** flow matching policy for LIBERO / Piper.

Key difference from the encoder-decoder `transformer_flow_matching`:
- Encoder-decoder runs VLM to completion (frozen) and then has the action
  expert *read* its outputs via cross-attention. VLM cannot see action.
- **This model** runs VLM and a parallel trainable expert **side-by-side**:
  at every VLM layer, expert tokens (latent + action) join the VLM sequence
  in a single self-attention pass. VLM tokens still use frozen Q/K/V/FFN;
  expert tokens use a trainable copy. Joint softmax over the combined
  sequence makes VLM **activations** task-aware even though VLM **weights**
  remain frozen.

---

## Full Forward Pass

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUTS                                                                       │
│                                                                              │
│  Camera images (3 cams)   Robot state (7-DOF)   Task description (text)     │
│  (B, C, H, W) × 3         (B, 7)                list[str]                   │
└──────────────┬──────────────────┬──────────────────────┬────────────────────┘
               │                  │                      │
               ▼                  │                      ▼
   ┌───────────────────────────┐  │           ┌─────────────────────────┐
   │ VLM ENCODER [frozen, bf16]│  │           │ LANGUAGE [frozen]        │
   │                            │  │           │                          │
   │   SigLIP ViT               │  │           │   processor.tokenizer    │
   │     384×384 → ~729 patches │  │           │        ↓                 │
   │   Connector (pixel-shuffle)│  │           │   embed_tokens (frozen)  │
   │     → (B, V, 960)          │  │           │     → (B, L≤48, 960)     │
   └───────────────┬───────────┘  │           └────────────┬─────────────┘
                   │              │                        │
                   │   ┌──────────▼──────────────┐         │
                   │   │ ROBOT CNN [trainable]   │         │
                   │   │  ResNet-18 stem+L1-3    │         │
                   │   │  (per camera)           │         │
                   │   │  → (B, 16, 960) × 3     │         │
                   │   │  cat → (B, 48, 960)     │         │
                   │   └──────────┬──────────────┘         │
                   │              │                        │
                   │              ▼                        │
                   │   ┌─────────────────────┐             │
                   │   │ STATE ENCODER       │             │
                   │   │  Linear(7→960)+RMS  │             │
                   │   │  → (B, 1, 960)      │             │
                   │   └──────────┬──────────┘             │
                   │              │                        │
                   └──────────────┼────────────────────────┘
                                  │
                                  ▼
                ┌────────────────────────────────────┐
                │ VLM SIDE SEQUENCE                  │
                │ cat([vis, lang, robot, state])     │
                │ → (B, L_vlm, 960)                  │  L_vlm ≈ 600
                └────────────────┬───────────────────┘
                                 │
                                 │   ┌─── time t ──┐
                                 │   │             │
                                 ▼   ▼             ▼
                ┌────────────────────────────────────┐
                │ EXPERT SIDE SEQUENCE [trainable]   │
                │                                    │
                │  latent_embs (B, K=8, 960)         │
                │           cat                      │
                │  action_in_proj(noisy_actions)     │
                │    + action_pos_emb                │
                │    + time-conditioned MLP fusion   │
                │  → action_emb (B, H=64, 960)       │
                │                                    │
                │  cat([latent, action])             │
                │  → (B, K+H, 960)                   │  K+H = 72
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼────────────────────────────────────┐
                │ 16 × JOINT ATTENTION LAYERS                          │
                │                                                      │
                │   For layer i ∈ [0..15]:                             │
                │     vlm_seq, exp_seq = joint_attn_layer_i(           │
                │         vlm_seq,                                     │
                │         exp_seq,                                     │
                │         vlm_layer = text_model.layers[i]  (frozen)  │
                │         exp_layer = expert_layers[i]      (trainable)│
                │     )                                                │
                │                                                      │
                │   See "Joint Attention Layer" below for the inner    │
                │   workings — it's the heart of the architecture.     │
                └────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────┐
                │ READOUT                    │
                │   final_norm(exp_seq[:,K:])│  ← drop latent, keep
                │   action_out_proj(...)     │     action positions
                │   → velocity (B, H, 7)     │
                └────────────────────────────┘
```

---

## Sequence Layout (within each joint attention layer)

```
position:  0 ──────────────── L_vlm ─── L_vlm+K ───────── L_vlm+K+H = L_total
            │                  │                              │
           VLM tokens          │     Expert tokens            │
            │                  │                              │
           ┌───────────────────┐──┬────────────────────────┬──┐
           │ vision (V≈546)    │  │ latent (K=8)           │  │
           │ language (L≤48)   │  │ action (H=64)          │  │
           │ robot CNN (48)    │  │                        │  │
           │ state (1)         │  │                        │  │
           └───────────────────┘──┴────────────────────────┴──┘
            └─── frozen QKV ────┘   └─── trainable QKV ────┘
            └─── frozen FFN ────┘   └─── trainable FFN ────┘
```

Total sequence length L_total ≈ 600 + 8 + 64 = **672 tokens**.

---

## Joint Attention Layer — the core mechanism

Each layer takes **two parallel hidden streams** (vlm_seq, exp_seq) and updates
both, allowing them to influence each other through a single shared softmax.

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                       ONE JOINT ATTENTION LAYER                            ║
║                                                                            ║
║   ┌──────────────────────┐         ┌──────────────────────┐                ║
║   │   vlm_seq            │         │   exp_seq            │                ║
║   │   (B, L_vlm, 960)    │         │   (B, K+H, 960)      │                ║
║   └──────────┬───────────┘         └──────────┬───────────┘                ║
║              │                                │                            ║
║   ┌──────────▼───────────┐         ┌──────────▼───────────┐                ║
║   │ RMSNorm [frozen]     │         │ RMSNorm [trainable]  │                ║
║   │   input_layernorm    │         │   input_layernorm    │                ║
║   └──────────┬───────────┘         └──────────┬───────────┘                ║
║              │                                │                            ║
║   ┌──────────▼───────────┐         ┌──────────▼───────────┐                ║
║   │  Q,K,V = VLM proj    │         │  Q,K,V = Expert proj │                ║
║   │  [frozen Llama Q/K/V]│         │  [trainable]         │                ║
║   └──────────┬───────────┘         └──────────┬───────────┘                ║
║              │                                │                            ║
║              │       ┌─────────CAT────────────┘                            ║
║              └──────►│                                                     ║
║                      ▼                                                     ║
║   ┌─────────────────────────────────────────────────────────┐              ║
║   │  Q = cat([Q_vlm, Q_exp])      shape (B, L_total, ...)   │              ║
║   │  K = cat([K_vlm, K_exp])                                │              ║
║   │  V = cat([V_vlm, V_exp])                                │              ║
║   │                                                          │              ║
║   │  Reshape to (B, num_heads, L_total, head_dim)            │              ║
║   │  Apply RoPE on Q, K  (positions 0..L_total-1)            │              ║
║   │  GQA: repeat K, V heads if num_kv_heads < num_heads      │              ║
║   │                                                          │              ║
║   │  attn = softmax(QKᵀ / √d + mask) V    ← SINGLE softmax  │              ║
║   │                                          over the WHOLE  │              ║
║   │                                          joint sequence  │              ║
║   │  (B, num_heads, L_total, head_dim)                       │              ║
║   └─────────────────────────────────┬───────────────────────┘              ║
║                                     │                                      ║
║              ┌──────────SPLIT───────┤                                      ║
║              ▼                      ▼                                      ║
║   ┌──────────────────────┐   ┌──────────────────────┐                      ║
║   │ vlm_attn_out         │   │ exp_attn_out         │                      ║
║   │ (B, L_vlm, ...)      │   │ (B, K+H, ...)        │                      ║
║   ├──────────────────────┤   ├──────────────────────┤                      ║
║   │ o_proj [frozen]      │   │ o_proj [trainable]   │                      ║
║   │ (VLM's)              │   │ (Expert's, dropout)  │                      ║
║   └──────────┬───────────┘   └──────────┬───────────┘                      ║
║              │                          │                                  ║
║              + (residual)               + (residual)                       ║
║              │                          │                                  ║
║   ┌──────────▼───────────┐    ┌──────────▼───────────┐                     ║
║   │ RMSNorm [frozen]     │    │ RMSNorm [trainable]  │                     ║
║   │ post_attention_norm  │    │ post_attention_norm  │                     ║
║   └──────────┬───────────┘    └──────────┬───────────┘                     ║
║              │                           │                                 ║
║   ┌──────────▼───────────┐    ┌──────────▼───────────┐                     ║
║   │ SwiGLU FFN [frozen]  │    │ SwiGLU FFN [trainable│                     ║
║   │  gate/up/down_proj   │    │  gate/up/down_proj   │                     ║
║   └──────────┬───────────┘    └──────────┬───────────┘                     ║
║              │                           │                                 ║
║              + (residual)                + (residual)                      ║
║              │                           │                                 ║
║              ▼                           ▼                                 ║
║   ┌──────────────────────┐    ┌──────────────────────┐                     ║
║   │  vlm_seq (updated)   │    │  exp_seq (updated)   │                     ║
║   └──────────────────────┘    └──────────────────────┘                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Why this is "true interleaving"**:

The `softmax(QKᵀ) V` step is over the **whole concatenated sequence**. So:

| VLM query position | can attend to | VLM keys + **Expert keys** |
|---|---|---|
| Expert query position | can attend to | **VLM keys** + Expert keys |

The **bolded** parts are what make this different from encoder-decoder:

- **Expert → VLM**: trivial, this is what cross-attention already does.
- **VLM → Expert** (the bold one): this is the key. VLM tokens' outputs are
  computed as a weighted sum of values that include expert values. So a VLM
  token's *output activation* is now a function of expert state — even though
  the VLM's *weights* (Q/K/V/o_proj/FFN) never change.

This is how the action context influences VLM perception: not by changing
what the VLM knows, but by changing what the VLM **attends to** at every
layer.

---

## Attention Mask

The mask determines which Query positions can attend to which Key positions
in the joint sequence. Layout:

```
                     ┌──── VLM ────┬── Latent ──┬─────── Action ──────┐
                     │  positions  │ positions  │     positions       │
                     │ [0..L_vlm)  │[L_vlm..K_e)│   [K_e..L_total)    │
                     ├─────────────┼────────────┼─────────────────────┤
              VLM    │             │            │                     │
   QUERY  positions  │   ✓ full    │  ✓ full    │  ✓ if vlm_attends   │
         [0..L_vlm)  │             │            │    _to_expert       │
                     ├─────────────┼────────────┼─────────────────────┤
            Latent   │             │            │                     │
          positions  │   ✓ full    │  ✓ full    │     ✗ BLOCKED       │
        [L_vlm..K_e) │             │            │                     │
                     ├─────────────┼────────────┼─────────────────────┤
            Action   │             │            │                     │
          positions  │   ✓ full    │  ✓ full    │   ◢ causal only     │
       [K_e..L_total)│             │            │                     │
                     └─────────────┴────────────┴─────────────────────┘
```

Where `K_e = L_vlm + K`.

### Per-rule explanation

**1. Latent → Action: BLOCKED.** Latent tokens cannot see noisy action tokens.

Why: latents are meant to be **task-aware "thoughts"** distilled from the
scene and the language prompt — not from the noisy action sequence itself.
If a latent could read a noisy action at low noise level t (close to the
true demo action), it would short-circuit the learning: the model would
just learn "latent ≈ next action", trivially copying the leaked signal
instead of doing genuine task understanding. Blocking this preserves the
latents as task-context tokens.

**2. Action → Action: CAUSAL.** Action token at step i can only see action
tokens at steps ≤ i.

Why: standard autoregressive masking. During training each action token
sees only previous action tokens (no future-leak). During flow-matching
inference the entire chunk is denoised in parallel, but the causal mask
keeps the architecture consistent with autoregressive variants.

**3. VLM → Expert: CONFIGURABLE (default ON).**

- **ON** (`vlm_attends_to_expert=True`, default): VLM queries can attend to
  expert keys. This is what enables true SmolVLA-style interleaving — VLM
  perception becomes action-aware. Slight risk of leak through action keys
  (VLM has direct access to noisy actions), but in practice the noise
  swamps that signal at low layers; the model learns to use the *structure*
  rather than memorise the answer.
- **OFF**: behaves more like a one-way "expert reads VLM" cross-attention.
  Lose the interleaving benefit but eliminate any potential leak. Use this
  flag if you observe training loss collapsing while benchmark stagnates
  (a classical leakage signature).

**4. Everything else: full visibility.** Latents and actions can both see
all of [VLM, Latent]. This is needed because:
- Expert tokens need full task context (the entire VLM-side sequence) at
  every layer to ground their predictions.
- Latents need to see each other to specialise into distinct "thought
  channels" (otherwise they'd collapse to one token).

### How the mask is built in code

[interleaved_flow_matching_model.py:191](interleaved_flow_matching_model.py#L191):
```python
mask = torch.zeros(L_total, L_total)            # default = allow everything (0)
mask[L_vlm:L_vlm + K, a_start:] = -inf          # rule 1: latent ✗ action
if not vlm_attends_to_expert:
    mask[:L_vlm, L_vlm:] = -inf                 # rule 3 OFF: VLM ✗ expert
causal = triu(full(H, H), -inf, diagonal=1)
mask[a_start:, a_start:] = causal                # rule 2: action causal
```

---

## Component Summary

| Component | Module | ~Params | Frozen? |
|---|---|---|---|
| **VLM backbone (all 16 layers, ViT, connector)** | SmolVLM2-500M | ~500M | ✅ always |
| State encoder | Linear(7→960) + RMSNorm | ~7K | ❌ |
| RobotVisualEncoder ×3 cams | ResNet-18 stem+L1-3 + proj | ~11M | ❌ |
| Latent thought tokens | nn.Parameter(1, K, 960) | ~8K | ❌ |
| Action in proj | Linear(7→960) | ~7K | ❌ |
| Action time MLP | Linear(1920→960) + Linear(960→960) | ~2.8M | ❌ |
| Action positional emb | nn.Parameter(1, H, 960) | ~62K | ❌ |
| Final RMSNorm | scalar weight | 1K | ❌ |
| Action out proj | Linear(960→7) zero-init | ~7K | ❌ |
| **Expert attention ×16 layers** | Q,K,V,O each Linear(960→960) | ~59M | ❌ |
| **Expert FFN ×16 layers** | SwiGLU at ffn_dim≈2560 | ~118M | ❌ |
| Expert RMSNorms ×16×2 | scalar weight each | ~30K | ❌ |
| **Total trainable** | | **~190M** | |
| **Total frozen** | | **~500M** | |

---

## Comparison vs Encoder-Decoder

| Aspect | `transformer_flow_matching` | `interleaved_flow_matching` |
|---|---|---|
| VLM weights | Frozen | Frozen |
| Expert d_model | 512 | **960** (must match VLM) |
| Expert participates in VLM attention? | ❌ No | ✅ Yes (joint softmax) |
| Cross-attention | Per layer, expert → VLM_layer[i] | None — replaced by joint self-attn |
| Mask | Action causal, latent ✗ action | Same + optional VLM ✗ expert toggle |
| Number of attention computations per layer | 2 (expert SA + cross-attn) | 1 (joint SA only) |
| Trainable params | ~83M | ~190M |
| Compute per forward | Lower | Higher (longer joint seq) |
| Memory per forward | Lower | ~2-3× higher |
| Checkpoint compatibility | — | **Not loadable from encoder-decoder ckpts** |
| Best use case | Quick iteration, smaller GPU | Disambiguation-heavy tasks (task 5 ramekin) |

---

## Key Design Decisions

**Expert dim = VLM hidden (960).** Joint attention concatenates K/V across
both sides; dims must match. Could be relaxed by inserting projection layers
but at significant complexity cost.

**Latent tokens prepended to expert (not VLM) side.** This way they get the
benefit of being part of the trainable side (their effective Q/K/V come from
expert weights), but are still visible to VLM tokens via the joint softmax.

**Zero-init o_proj and mlp.down_proj.** Expert starts as identity in the
residual stream — initial loss ≈ E[||u_t||²] ≈ 2, not millions. Optional
warm-start via [transfer_pretrained_weights.py](../../transfer_pretrained_weights.py)
copies VLM weights into expert (scaled by `init_scale`).

**RMSNorm + SwiGLU + RoPE + GQA in ExpertProjections.** Mirrors SmolLM2's
Llama-style decoder layer exactly so weight transfer between VLM and expert
is shape-clean.

**Position IDs continue across the joint sequence.** Expert positions sit
*after* VLM positions in RoPE indexing. The model learns these are distinct
modalities through the attention pattern.

**Action positional embedding is a separate learned parameter** (not RoPE
alone). RoPE encodes the absolute joint-sequence position; the small
learned `action_pos_emb` lets the action segment have its own relative
position structure independent of where it sits within the joint sequence.
