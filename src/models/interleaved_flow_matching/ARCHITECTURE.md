# InterleavedFlowMatchingTransformer — Architecture Diagram

## Overview

A **joint attention** flow matching policy where a frozen VLM (SmolVLM2)
and a parallel trainable action expert share every self-attention layer.

At each of the 16 transformer layers, VLM tokens (vision + language + state)
and expert tokens (robot features + latent thoughts + noisy actions) are
concatenated into a single sequence. One joint softmax runs over the whole
thing. VLM tokens use their frozen Q/K/V/FFN weights; expert tokens use a
trainable copy.

The key property: VLM activations become **task-aware** because VLM queries
can attend to expert keys, even though VLM weights never change. The action
context influences what the VLM attends to at every layer.

**Mask semantics:**
```
                 VLM    Robot   Latent  Action
    VLM          full   ✓       ✓       ✓ (gated by vlm_attends_to_expert)
    Robot        ✓      ✓       ✓       ✗ (never see noisy actions)
    Latent       ✓      ✓       ✓       ✗ (never see noisy actions)
    Action       ✓      ✓       ✓       causal (only earlier actions)
```

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
                │ VLM SEQUENCE                       │
                │ cat([vis, lang, state])            │
                │ → (B, L_vlm, 960)                  │  L_vlm ≈ 600
                └────────────────┬───────────────────┘
                                 │
                                 │   ┌─── time t ──┐
                                 │   │             │
                                 ▼   ▼             ▼
                ┌────────────────────────────────────┐
                │ EXPERT SEQUENCE [trainable]        │
                │                                    │
                │  robot tokens: (B, 48, 960)        │
                │  latent tokens: (B, K=8, 960)      │
                │           cat                      │
                │  action_in_proj(noisy_actions)     │
                │    + action_pos_emb                │
                │    + time-conditioned MLP fusion   │
                │  → action_emb (B, H=64, 960)       │
                │                                    │
                │  cat([robot, latent, action])      │
                │  → (B, 48+K+H, 960)                │
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
                │   final_norm(exp_seq[:,-H:])│ ← drop robot+latent, keep
                │   action_out_proj(...)     │     action positions
                │   → velocity (B, H, 7)     │
                └────────────────────────────┘
```

---

## Sequence Layout (within each joint attention layer)

```
position:  0 ──────────────── L_vlm ────── L_vlm+R ────── L_vlm+R+K ───────── L_total
            │                  │             │               │
           VLM tokens          Robot         Latent          Action
            │                  │             │               │
           ┌───────────────────┐──┬──────────┬──┬────────────┬──┐
           │ vision (V≈546)    │  │ robot (48)│ │ latent (K=8)│  │
           │ language (L≤48)   │  │           │ │             │  │
           │ state (1)         │  │           │ │             │  │
           └───────────────────┘──┴──────────┴──┴────────────┴──┘
            └─── frozen QKV ────┘   └── trainable QKV ───┘
            └─── frozen FFN ────┘   └── trainable FFN ───┘
```

Total sequence length L_total ≈ 600 + 48 + 8 + 64 = **720 tokens**.

---

## Joint Attention Layer

Each layer takes **two parallel hidden streams** (vlm_seq, exp_seq) and updates
both through a single shared softmax.

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                       ONE JOINT ATTENTION LAYER                            ║
║                                                                            ║
║   ┌──────────────────────┐         ┌──────────────────────┐                ║
║   │   vlm_seq            │         │   exp_seq            │                ║
║   │   (B, L_vlm, 960)    │         │   (B, R+K+H, 960)    │                ║
║   └──────────┬───────────┘         └──────────┬───────────┘                ║
║              │                                │                            ║
║   ┌──────────▼───────────┐         ┌──────────▼───────────┐                ║
║   │ RMSNorm [frozen]     │         │ RMSNorm [trainable]  │                ║
║   │   input_layernorm    │         │   input_layernorm    │                ║
║   └──────────┬───────────┘         └──────────┬───────────┘                ║
║              │                                │                            ║
║   ┌──────────▼───────────┐         ┌──────────▼───────────┐                ║
║   │  Q,K,V = VLM proj    │         │  Q,K,V = Expert proj │                ║
║   │  [frozen]             │         │  [trainable]         │                ║
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
║   │ (B, L_vlm, ...)      │   │ (B, R+K+H, ...)      │                      ║
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

Since the `softmax(QKᵀ / √d) V` runs over the **whole concatenated sequence**:

- **Expert queries** attend to VLM keys — the expert has access to visual,
  language, and state context at every layer.
- **VLM queries** attend to expert keys — VLM tokens' output activations
  become a weighted sum that includes expert values, so even though VLM
  *weights* never change, VLM *perception* adapts to the current task
  context at every layer.

---

## Attention Mask

The mask determines which queries can attend to which keys
in the joint sequence.

```
                     ┌──── VLM ────┬── Robot ──┬── Latent ──┬─────── Action ──────┐
                     │  positions  │ positions │ positions  │     positions       │
                     │ [0..L_vlm)  │[L_vlm..R_e)│[R_e..K_e) │   [K_e..L_total)    │
                     ├─────────────┼───────────┼────────────┼─────────────────────┤
              VLM    │             │           │            │                     │
   QUERY  positions  │   ✓ full    │  ✓ full   │  ✓ full    │  ✓ if vlm_attends   │
         [0..L_vlm)  │             │           │            │    _to_expert       │
                     ├─────────────┼───────────┼────────────┼─────────────────────┤
             Robot   │             │           │            │                     │
          positions  │   ✓ full    │  ✓ full   │  ✓ full    │     ✗ BLOCKED       │
        [L_vlm..R_e) │             │           │            │                     │
                     ├─────────────┼───────────┼────────────┼─────────────────────┤
            Latent   │             │           │            │                     │
          positions  │   ✓ full    │  ✓ full   │  ✓ full    │     ✗ BLOCKED       │
        [R_e..K_e)   │             │           │            │                     │
                     ├─────────────┼───────────┼────────────┼─────────────────────┤
            Action   │             │           │            │                     │
          positions  │   ✓ full    │  ✓ full   │  ✓ full    │   ◢ causal only     │
       [K_e..L_total)│             │           │            │                     │
                     └─────────────┴───────────┴────────────┴─────────────────────┘
```

Where `R_e = L_vlm + R` and `K_e = L_vlm + R + K`.

### Rule 1: Robot/Latent → Action: BLOCKED

Robot and latent tokens cannot attend to noisy action tokens.

These tokens are **task-aware representations** distilled from the scene and
the language prompt — not from the noisy action sequence. If a robot or latent
token could read a noisy action at low noise level t (close to the true demo
action), it would short-circuit learning: the model would copy the leaked
signal instead of doing genuine task understanding.

### Rule 2: Action → Action: CAUSAL

Action token at step i can only see action tokens at steps ≤ i.

Standard autoregressive masking. During training each action token sees only
previous tokens. During flow-matching inference the whole chunk is denoised in
parallel, but the causal mask keeps the architecture consistent.

### Rule 3: VLM → Expert: CONFIGURABLE (default ON)

- **ON** (`vlm_attends_to_expert=True`): VLM queries can attend to expert keys.
  This enables VLM perception to become action-aware. There's a slight risk of
  signal leak through action keys, but in practice the noise swamps that at low
  layers — the model learns to use the *structure*, not memorize the answer.
- **OFF**: VLM attention is restricted to its own tokens only. Eliminates any
  potential leak at the cost of losing interleaving. Use this if training loss
  collapses while benchmark stagnates (a leakage signature).

### Rule 4: Everything else: FULL visibility

Expert tokens need full task context at every layer to ground predictions.
Latents need to see each other to specialize into distinct "thought channels"
(otherwise they'd collapse to one token).

### Mask construction in code

```python
mask = torch.zeros(L_total, L_total)            # default: allow everything
mask[R_start:R_start+R, A_start:] = -inf         # rule 1a: robot ✗ action
mask[R_start+R:R_start+R+K, A_start:] = -inf     # rule 1b: latent ✗ action
if not vlm_attends_to_expert:
    mask[:L_vlm, L_vlm:] = -inf                 # rule 3 OFF: VLM ✗ expert
causal = triu(full(H, H), -inf, diagonal=1)
mask[A_start:, A_start:] = causal                # rule 2: action causal
```

### Per-sample pad blocking + learnable language bias

When language tokens are present, the mask expands to `(B, L_total, L_total)`:

1. **Pad positions blocked**: `lang_mask` marks real vs pad tokens. Pad
   positions are fully blocked — no query attends to them and they produce
   no output.

2. **`lang_attn_bias`**: a per-layer learnable scalar applied via `softplus`
   (keeps it ≥ 0), added to the expert→language attention logits. Each layer
   learns its own bias because different layers have different language
   dependence: shallow layers may favor vision, mid layers may need language
   for disambiguation.

### Language adaptor and contrastive loss

Two mechanisms explicitly force the model to USE language:

1. **Language adaptor**: A zero-init trainable residual on non-pad language tokens:
   ```
   lang_final = frozen_embedding + lang_adaptor(frozen_embedding)
   ```
   Zero-init means training starts from the original frozen behavior.

2. **Contrastive auxiliary loss** (optional, `contrastive_loss_weight`):
   - Shuffle language tokens across the batch (derangement)
   - Re-run `velocity_field()` with shuffled language → `v_wrong`
   - Penalize when `v_t` and `v_wrong` are too similar:
     ```
     L_contrastive = max(0, margin - ||v_t - v_wrong||²)
     ```
   This forces the velocity prediction to CHANGE when the language changes.

---

## Component Summary

| Component | ~Params | Frozen? |
|---|---|---|
| VLM backbone (16 layers, ViT, connector) — SmolVLM2-500M | ~500M | ✅ always |
| State encoder: Linear(7→960) + RMSNorm | ~7K | ❌ |
| RobotVisualEncoder ×3 cams: ResNet-18 stem+L1-3 + projection | ~11M | ❌ |
| Latent generator MLP: Linear(960→1920) → SiLU → Linear(1920→8×960) | ~17M | ❌ |
| Action in proj: Linear(7→960) | ~7K | ❌ |
| Action time MLP: Linear(1920→960) + Linear(960→960) | ~2.8M | ❌ |
| Action positional embedding: `nn.Parameter(1, H, 960)` | ~62K | ❌ |
| Final RMSNorm | 1K | ❌ |
| Action out proj: Linear(960→7) zero-init | ~7K | ❌ |
| Expert attention ×16 layers: Q,K,V,O each Linear(960→960) | ~59M | ❌ |
| Expert FFN ×16 layers: SwiGLU at ffn_dim≈2560 | ~118M | ❌ |
| Expert RMSNorms ×16×2: scalar weight each | ~30K | ❌ |
| Language adaptor: Linear(960→960) + RMSNorm | ~1M | ❌ |
| Lang attention bias: `nn.Parameter(16,)` | 16 | ❌ |
| **Total trainable** | **~210M** | |
| **Total frozen** | **~500M** | |

---

## Key Design Decisions

**Frozen VLM + trainable expert in the same self-attention.**
The core idea: instead of running the VLM separately and then having the
expert read its outputs, both sequences go through the same softmax. VLM
weights stay frozen (Q/K/V/o_proj/FFN never change), but VLM activations
become a function of the expert tokens they attend to. This lets action
context flow into VLM perception at every layer.

**Expert dim = VLM hidden (960).**
Joint attention concatenates K/V across both sides; dimensions must match.
Could insert projection layers but at significant complexity cost.

**Latent tokens are generated dynamically from language.**
Previous design used a static `nn.Parameter(1, K, 960)` shared across all
tasks — ablation showed it was load-bearing but task-agnostic. Current
design pools language tokens via masked mean and passes through an MLP to
produce K = 8 latent tokens per sample. The output layer is zero-initialized
so training starts from "no latents" behavior.

**Language masked mean pooling.**
Pad tokens are excluded before averaging, so the latent generator sees only
the real instruction content (typically 3-10 tokens out of 48 max).

**Robot CNN tokens on the expert side.**
The expert sequence layout is `[robot, latent, action]`. The ResNet output
trains through expert Q/K/V/FFN — a fully trainable path from camera pixels
to action output.

**`use_robot_cnn` config flag for ablation** (default `True`).
When `False`, the RobotVisualEncoder isn't instantiated. Useful for measuring
whether the CNN contributes beyond what the VLM already provides.

**Zero-init action_out_proj.**
Expert starts as identity in the residual stream — initial loss ≈ E[||u_t||²]
≈ 2, not millions. Optional warm-start via `transfer_pretrained_weights.py`
copies VLM weights into expert (scaled by `init_scale`).

**RMSNorm + SwiGLU + RoPE + GQA in ExpertProjections.**
Mirrors the VLM's Llama-style decoder layer exactly, making weight transfer
between VLM and expert shape-clean.

**Position IDs continue across the joint sequence.**
Expert positions sit *after* VLM positions in RoPE indexing. The model learns
these are distinct modalities through the attention pattern.

**Action positional embedding is a separate learned parameter.**
RoPE encodes the absolute joint-sequence position; the small learnable
`action_pos_emb` lets the action segment have its own relative position
structure independent of where it sits within the joint sequence.

**Per-layer language attention bias.**
One scalar per joint layer, applied via softplus (≥ 0). Early layers may need
less language (raw visual processing), mid layers need it for disambiguation,
late layers settle in between. A single global bias would average across
these conflicting needs.

**Dual visual dropout** (`vision_dropout_prob`).
Applied independently to both VLM vision tokens and robot CNN tokens with the
same probability — prevents one stream from compensating for the other. The
model must learn to function with partial information from either source.

---

## Method Reference

### 1. `__init__(self, config)`

Loads frozen SmolVLM2 and initializes all trainable components.

| Sub-module | Details | Frozen? |
|---|---|---|
| `self.vision_model` | SigLIP ViT | ✅ |
| `self.connector` | Vision→language projection (pixel shuffle) | ✅ |
| `self.text_model` | Text transformer (first `num_vlm_layers`) | ✅ |
| `self.expert_layers` | One `ExpertProjections` per VLM layer: Q/K/V/o_proj + SwiGLU FFN + 2×RMSNorm | ❌ |
| `self.robot_visual_encoder` | RobotVisualCNN (optional, `use_robot_cnn` flag) | ❌ |
| `self.state_encoder` | Linear → RMSNorm | ❌ |
| `self.action_in_proj` | Linear(action_dim → hidden) | ❌ |
| `self.action_out_proj` | Linear(hidden → action_dim), zero-initialized | ❌ |
| `self.action_time_mlp_in/out` | Time-conditional MLP (1920→960→960) | ❌ |
| `self.action_pos_emb` | Learnable positional bias `(1, horizon, hidden)` | ❌ |
| `self.latent_generator` | Linear→SiLU→Linear, output layer zero-initialized | ❌ |
| `self.lang_adaptor` | Linear + RMSNorm, zero-initialized | ❌ |
| `self.lang_attn_bias` | Per-layer scalar, applied via softplus | ❌ |
| `self.final_norm` | RMSNorm before action readout | ❌ |

Reads VLM layer 0's `self_attn` to determine `hidden_size`, `num_heads`,
`num_kv_heads`, `head_dim`, `intermediate_size` — ensures expert projections
match exactly. If `config.d_model != hidden_size`, forcibly overrides.

### 2. `train(self, mode=True)`

Overrides `nn.Module.train()`. Forces `vision_model`, `connector`, and
`text_model` back to eval mode so frozen weights never participate in
dropout or batchnorm updates.

### 3. `_encode_images(self, batch, B) → Tensor`

Frozen visual encoding:

1. Iterates `cameras_for_vision_state_concat` for each camera key
2. Takes last frame if 5D video input
3. Normalizes to `[-1, 1]` (SigLIP convention)
4. Non-square images: pad with -1 then resize to `vision_input_size`
5. `vision_model(pixel_values)` → `last_hidden_state`
6. `connector(vis_hidden)` → `(B, num_patches, hidden)`
7. All camera tokens concatenated along dim=1

Returns `(B, V_total, hidden)` or `(B, 0, hidden)` if no cameras.

### 4. `_encode_language(self, batch, device) → (lang_tokens, lang_mask) or None`

Frozen language encoding:

1. Reads `batch["task_description"]` or `batch["task"]`
2. Tokenizes with SmolVLM2's processor (pad to max 48, truncate)
3. `text_model.get_input_embeddings()(input_ids)` → `(B, L, hidden)`
4. Returns `lang_tokens` and `lang_mask` `(B, L)` bool — the mask is critical
   because most task descriptions are 3-10 tokens padded to 48

Returns `None` if no language is present.

### 5. `_build_joint_mask(self, L_vlm, R, K, H, ...) → Tensor`

Builds the additive attention mask. Returns `(L_total, L_total)` or
`(B, 1, L_total, L_total)` when per-sample pad blocking is needed.

```python
mask = zeros(L_total, L_total)                  # default: allow everything
mask[R_start:R_start+R, A_start:] = -inf         # robot ✗ action
mask[R_start+R:R_start+R+K, A_start:] = -inf     # latent ✗ action
if not vlm_attends_to_expert:
    mask[:L_vlm, L_vlm:] = -inf                 # VLM ✗ expert
mask[A_start:, A_start:] = causal(...)           # action causal
```

Per-sample pad blocking: pad language positions are fully blocked as both
key and query. `lang_attn_bias[layer_idx]` is applied via softplus and added
to the expert→language region, only on non-pad columns.

### 6. `_joint_layer(self, vlm_seq, exp_seq, layer_idx, cos, sin, attn_mask) → (vlm_seq, exp_seq)`

One layer of joint self-attention + FFN.

**Step 1: Pre-norm** — frozen `input_layernorm` on VLM side, trainable on expert side.

**Step 2: Q/K/V projections** — frozen on VLM side, trainable on expert side.
Concat along sequence dim *before* multi-head reshape so RoPE aligns:

```python
Q = cat([Q_vlm, Q_exp], dim=1).view(B, L_total, H, D).transpose(1, 2)
```

**Step 3: RoPE** — applied to concatenated Q, K over positions 0..L_total-1.

**Step 4: GQA** — repeat K, V heads if `num_kv_heads < num_heads`.

**Step 5: Joint SDPA** — `F.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)`.
Softmax computed over the entire combined sequence.

**Step 6: Split + output projection** — split back into VLM and expert
portions, each through its own `o_proj` (frozen vs trainable+dropout). Residual.

**Step 7: FFN** — pre-norm then separate SwiGLU FFNs (frozen vs trainable+dropout). Residual.

Returns updated `(vlm_seq, exp_seq)`.

### 7. `_generate_latents(self, lang_tokens, lang_mask, B, device, dtype) → Tensor or None`

Generates task-conditional latent tokens from language:

1. If `K == 0`: returns None
2. If no language: `pooled = zeros(B, hidden)`
3. With language: **masked mean pooling** over non-pad positions
4. Pass through `self.latent_generator` (Linear→SiLU→Linear) in fp32
5. Reshape to `(B, K, hidden)`

Uses post-`lang_adaptor` representations so the generator sees the same
representation the rest of the network attends to.

### 8. `_build_expert_seq(self, noisy_actions, timesteps, robot_tokens, latents) → Tensor`

Builds the expert sequence. Layout: `[robot?, latent?, action]`.

Action embedding:
```python
action_emb = action_in_proj(noisy_actions)
action_emb = action_emb + action_pos_emb         # learned positional bias
t_emb = sinusoidal(timesteps)                    # time conditioning
fused = cat([action_emb, t_emb], dim=-1)
fused = silu(time_mlp_in(fused))
fused = time_mlp_out(fused)
action_final = fused + (1 - t) * action_emb      # residual gating
```

When t ≈ 1 (pure noise), time conditioning dominates. When t ≈ 0 (clean),
the residual passes through the original action embedding.

### 9. `_compute_robot_tokens(self, batch) → Tensor or None`

Runs RobotVisualEncoder per camera, concatenates outputs. Returns None if
`robot_visual_encoder is None`. Applies vision dropout at the same rate as
the VLM vision side.

### 10. `_build_vlm_seq(self, batch) → (vlm_seq, L_vis, lang_mask)`

Builds the VLM sequence. Layout: `[vision, language, state]`.

1. `_encode_images()` → visual tokens
2. Per-token vision dropout (each SigLIP patch independently zeroed)
3. `_encode_language()` → language tokens + lang_mask
4. Apply `lang_adaptor` only on non-pad positions:
   `lang_final = frozen_emb + lang_adaptor(frozen_emb)`
5. `state_encoder(observation.state)` → state token (NaN→0, clamp [-10, 10])

### 11. `velocity_field(self, noisy_actions, timesteps, vlm_seq, ...) → Tensor`

Runs the full joint attention stack. Given noisy action `x_t` at time `t`,
outputs `v_t = dx_t/dt` — the velocity pointing toward the clean action.

1. Generate task-conditional latents from language portion of `vlm_seq`
2. `_build_expert_seq()` → expert sequence
3. Precompute RoPE cache for L_total
4. Per-layer loop: rebuild mask (per-layer bias), call `_joint_layer()`
5. Take last H positions from expert sequence
6. `final_norm` → `action_out_proj` → `(B, H, action_dim)`

### 12. `sample_noise(self, shape, device) → Tensor`

Samples noise with optional temporal correlation (AR(1)):
```python
noise[t] = rho * noise[t-1] + sqrt(1 - rho²) * randn()
```
`rho = 0` gives independent Gaussian noise. Correlated noise produces
smoother action trajectories during inference.

### 13. `sample_time(self, B, device) → Tensor`

Samples flow matching time: `t = rand(B) * 0.998 + 0.001`. Avoids t=0, t=1
boundaries for numerical stability.

### 14. `compute_loss(self, batch) → Tensor`

**Main loss — Flow Matching MSE:**

1. `x_t = t*noise + (1-t)*actions`, target `u_t = noise - actions`
2. `v_t = velocity_field(x_t, t, ...)`
3. Weighted MSE with:
   - `action_dim_weights`: per-dimension (e.g., gripper vs end-effector)
   - Position weights: full on `n_action_steps`, reduced beyond
   - `pos_decay_lambda`: exponential decay for far-future steps
   - `action_is_pad` and `action_dim_pad`: exclude padded positions/dims

**Contrastive auxiliary loss** (optional):

1. Derange language tokens across batch
2. Filter out pairs with identical task instructions
3. Re-run `velocity_field()` with shuffled language → `v_wrong`
4. `max(0, margin - ||v_t - v_wrong||²)` — penalize language-agnostic predictions

**Final**: `loss = main_loss + contrastive_weight * contrastive_loss`

### 15. `forward(self, batch) → (loss, {}) or (actions, {})`

Dispatches to `compute_loss()` (training) or `sample_actions()` (inference).

### 16. `sample_actions(self, batch) → Tensor`

Euler integration of the flow ODE:

```python
x_t = sample_noise(shape)
for _ in range(num_inference_steps):
    v_t = velocity_field(x_t, t, ...)
    x_t = x_t + dt * v_t    # dt = -1/N, integrate backward t:1→0
    t = t + dt
return x_t[:, :n_action_steps]
```

Uses bfloat16 autocast because the frozen VLM was loaded as bf16.

### 17. `count_parameters(self) → dict`

Returns `{trainable, frozen, total}` parameter counts.

---

## Data Flow Summary

```
batch (images, language, state)
        │
        ├─→ _build_vlm_seq() ──→ vlm_seq: [vision | language | state]
        │                              │
        │              _generate_latents() ← pooled language (masked mean)
        │                              │
        ├─→ _compute_robot_tokens() ──→ robot_tokens (or None)
        │
        └─→ noisy_actions + timesteps
                │
                └─→ _build_expert_seq() ──→ exp_seq: [robot? | latent? | action]

        Joint sequence: [vlm_seq | exp_seq]
                │
                └─→ N × _joint_layer()  ← VLM Q/K/V/FFN frozen
                        │                  Expert Q/K/V/FFN trainable
                        │                  Single softmax (constrained by mask)
                        │
                        └─→ action_out_proj → velocity (train) / action (infer)
```

---

## Caveats

- VLM layer attribute paths (e.g. `layer.self_attn.q_proj`) match SmolVLM2's
  structure but may need adjustment for transformers version drift. Failures
  will be visible on first forward.
- GQA: SmolLM2 may have `num_kv_heads < num_heads`; K/V heads are replicated
  on the expert side.
- RoPE positions are assigned sequentially across [VLM, robot, latent, action].
  Action token positions sit *after* VLM — unconventional but consistent with
  treating the whole thing as one sequence.