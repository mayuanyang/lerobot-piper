# WILRO-MoE Architecture

`wiltechs_moe` with the backbone swapped from Qwen3-VL-4B to SmolVLM2-500M.
Equivalently: **wilro's encoder under wiltechs_moe's decoder.** The encoder half
is *imported* (`SmolVLMEncoderMixin`), not copied, so wilro and wilro_moe share
one implementation of image/language encoding and KV capture.

- **Encoder** = SmolVLM2-500M (SigLIP ViT + connector + 32-layer text stack),
  frozen, optionally LoRA-adapted. Runs **once per observation**. Emits three
  things: a post-RoPE K/V cache for **all 32 text layers**, the mean-pooled final
  hidden state (for the router), and spatial vision tokens.
- **Decoder** = **4 experts × 8 DiT layers**. Every expert runs on every forward
  and their outputs are combined by a softmax router. Runs **N times per
  observation** during the flow-matching loop.

The defining structural choice: **the four experts read *disjoint* bands of the
VLM's layers.** 4 × 8 = 32 = exactly SmolVLM2's text depth, so the partition is
exact with nothing left over.

> `dit_hidden_size` defaults to **960 = the VLM's hidden size**, which is what
> lets the experts reuse wilro's `DiTLayer` verbatim: self- and cross-attention
> can share one head geometry (15 heads / 5 KV heads / head_dim 64). A narrower
> expert would need the split `sa_`/`ca_` variant from `wiltechs_vla`, which is
> **not ported** — the constructor raises `NotImplementedError` rather than
> silently mis-shaping the cross-attention.

---

## Overall data flow

```mermaid
flowchart TB
    subgraph ENC["ENCODER — frozen SmolVLM2-500M, runs ONCE per observation"]
        direction TB
        IMG["images<br/>N cameras"] --> ViT["SigLIP ViT<br/>frozen + LoRA rank r"]
        ViT --> CONN["connector<br/>pixel-shuffle + MLP"]
        TASK["task text<br/>+ paraphrase aug"] --> TE["embed_tokens<br/>frozen"]
        CONN --> TXT["text_model — 32 layers<br/>frozen + optional LoRA"]
        TE --> TXT
    end

    TXT -->|"post-RoPE K/V, all 32 layers"| KV[("VLM KV cache<br/>32 x (K,V)")]
    TXT -->|"final hidden, mean-pool over valid"| SEM["vlm_semantic (960)"]
    ViT -->|"intermediate layer -3, connector-projected"| VTOK["vision tokens"]

    subgraph DEC["DECODER — 4 experts x 8 DiT layers, runs N times per observation"]
        direction TB
        SEQ["sequence:<br/>sink | state | vision | action(H)"]
        SEQ --> E0["Expert 0<br/>8 DiT layers"]
        SEQ --> E1["Expert 1<br/>8 DiT layers"]
        SEQ --> E2["Expert 2<br/>8 DiT layers"]
        SEQ --> E3["Expert 3<br/>8 DiT layers"]
    end

    VTOK --> SEQ
    XT["x_t = t*noise + (1-t)*a"] --> SEQ
    ST["observation.state"] --> SEQ

    KV -->|"layers 0-7"| E0
    KV -->|"layers 8-15"| E1
    KV -->|"layers 16-23"| E2
    KV -->|"layers 24-31"| E3

    SEM --> R{{"MoERouter"}}
    ST --> R
    XT --> R
    TEMB["t embedding"] --> R

    E0 --> MIX["weighted sum<br/>sum_e w_e * v_e"]
    E1 --> MIX
    E2 --> MIX
    E3 --> MIX
    R -->|"w (B,4) softmax"| MIX
    MIX --> V["v_t — predicted velocity<br/>(B, H, 7)"]
```

**Read the arrow counts, not the boxes:** the encoder arrow fires once, the
decoder block fires `num_inference_steps` times at eval (default 10) and once
per training step. The KV cache is what makes that cheap.

---

## Expert ↔ VLM layer bands

Bands are **disjoint and contiguous, shallow experts on shallow layers**:

| Expert | VLM text layers | reads |
|---|---|---|
| 0 | 0–7 | early / lexical |
| 1 | 8–15 | |
| 2 | 16–23 | |
| 3 | 24–31 | late / semantic |

Expert `e`'s DiT layer `i` cross-attends to `expert_kv_blocks[e][i % 8]`, i.e. a
1:1 pairing of the expert's 8 layers to its 8 captured VLM layers.

> **Why the mixin must capture ALL layers.** `expert_kv_blocks` indexes
> `kv_cache[i]` by absolute VLM layer number. The mixin appends in layer order,
> so a *partial* capture would silently renumber every band — expert 3 would
> read early layers while the config still said 24–31. This is why
> `capture = list(range(32))` rather than a subset, and why
> `vlm_capture_layers` must be divisible by `num_experts`.

> **Why 4 × 8 and not wiltechs_moe's 4 × 9.** SmolVLM2-500M has **32** text
> layers where Qwen3-VL-4B has 36. With disjoint bands, 36 does not fit; the
> constructor raises rather than overlapping them. `train_wilro_moe.py` carries
> a preflight guard for the same reason.

---

## The router

```mermaid
flowchart LR
    S["state_emb (960)"] --> C["concat (3840)"]
    P["vlm_semantic -> vlm_proj (960)"] --> C
    T["time_emb (960)"] --> C
    A["action_emb.mean over H (960)"] --> C
    C --> M["Linear 3840->960<br/>SiLU<br/>Linear 960->4"]
    M --> L["logits (B,4)"]
    L -.->|"detach, PRE-noise"| D["diagnostics:<br/>max_w, entropy"]
    L --> N["+ N(0, 0.5)<br/>TRAIN ONLY"]
    N --> SM["softmax"]
    SM --> W["w (B,4)"]
    W --> U["usage = w.mean(0)<br/>-> balance loss"]
```

Four inputs, chosen so the router sees **the whole conditioning**: the robot
state, the fused multimodal context, where in the flow it is, and what the
current noisy action looks like.

`vlm_semantic` is the mean-pooled **final hidden state**, not the KV cache's V —
hidden is always `hidden_size`, so there is no GQA head-count mismatch to
unpick. Vision and language are already fused by the text stack's causal
attention, so one pool of the last layer carries both.

| knob | default | note |
|---|---|---|
| `num_experts` | 4 | |
| `expert_num_layers` | 8 | 4 × 8 = 32 = VLM depth |
| `router_temperature` | 1.0 | |
| `router_top_k` | **0** | 0 = **dense**: all experts run, see cost note |
| `router_balance_weight` | 0.1 | CV² of usage |

### Three things about the router that are load-bearing

1. **Init is `normal_(std=0.02)`, deliberately not zeros.** Zero init makes every
   logit identical at step 0; any tiny data gradient tips one expert ahead and
   the softmax positive-feedback loop collapses to it. Observed on the sibling:
   E3 at 100% by step 200.

2. **Train-time exploration noise is fixed `N(0, 0.5)`, not scaled to the logit
   magnitude** — so it keeps feeding starved experts signal instead of washing
   out as the router grows confident. Consequence: **diagnostics must read the
   pre-noise weights.** A router with no input dependence at all still reports
   `max_w ≈ 0.39` once the noise is added, not the 0.25 that "uniform" suggests.

3. **The balance loss is applied in the *policy's* `forward`, not in
   `compute_loss`** — it reads `model._last_router_usage` *after* `compute_loss`
   returns. That is why `_run_dit` takes `record=True/False`: see Failure modes.

### Cost note: `router_top_k=0` means this is a DENSE mixture

Every expert runs on every token of every forward. **There is no compute saving
from the MoE structure** — 4 experts × 8 layers costs the same as one 32-layer
decoder. What the structure buys is *specialisation over disjoint VLM depth
bands* plus a router that can weight them per-sample. Setting `router_top_k > 0`
would make it sparse, but no run has used it.

---

## Sequence layout

```
index:   0        1 .. S      S+1 .. S+V         S+V+1 .. S+V+H
       [ sink ] [ state ] [ vision tokens ] [ noisy actions x_t ]
                                            ^ action_start_idx
```

with a **causal** mask over the whole thing. `S` is 1 unless
`use_state_history`; `V` is 0 if `vision_token_source` yields nothing;
`H = horizon`.

**Vision tokens sit BEFORE the actions because there is no Vision CA sublayer.**
Causal self-attention is the *only* path from an action query to a vision token,
so their order in the sequence is not cosmetic — put them after the actions and
the causal mask hides them completely.

> wiltechs_moe additionally places K "thought" tokens here, from a QFormer over
> the deepest VLM layer's KV. **Dropped 2026-09-05** (18.4M params + a sequence
> region) after being reported as not earning its keep there. No wilro_moe
> checkpoint existed yet, which is the only reason this is a deletion rather
> than a default-off flag.

---

## One DiT layer

wilro's `DiTLayer`, constructed with `use_vision_ca=False` — matching
wiltechs_moe. **Three sublayers, not four**, so adaLN-Zero produces 9 modulation
vectors rather than 12.

```mermaid
flowchart TB
    X["x"] --> N1["RMSNorm + adaLN shift/scale"]
    N1 --> SA["self-attention<br/>causal, over the DiT sequence"]
    SA --> G1["x gate1"] --> R1(("+"))
    X --> R1
    R1 --> N2["RMSNorm + adaLN shift/scale"]
    N2 --> CA["cross-attention<br/>Q from DiT, K/V from ONE VLM layer"]
    KVIN[/"expert_kv_cache[i mod 8]"/] --> CA
    CA --> G2["x gate2"] --> R2(("+"))
    R1 --> R2
    R2 --> N3["RMSNorm + adaLN shift/scale"]
    N3 --> FF["SwiGLU FFN<br/>960 -> 2560 -> 960"]
    FF --> G3["x gate3"] --> R3(("+"))
    R2 --> R3
    R3 --> OUT["x'"]
    TE[/"t_emb"/] -.->|"SiLU -> Linear 960 -> 9x960<br/>ZERO-INIT"| N1
    TE -.-> N2
    TE -.-> N3
```

**adaLN-Zero means every residual branch starts at exactly zero, so at init each
expert *is* the identity map.** This is why the expert-disagreement diagnostic
reads exactly `0.000` at step 0 — "not yet differentiated", which is
indistinguishable from "in agreement" by the number alone. Read it with the step
count.

---

## Per-expert vision adapters (`resnet_expert_adapter_dim`, default 0 = off)

All four experts read the **same** vision tokens. If different experts want
different things from them, they have no way to say so — the tokens are shared
and frozen relative to the expert stack. This is the fix:

```python
vis   = seq[:, lo:hi]
delta = expert_vision_gates[e] * expert_vision_adapters[e](vis)
seq_e = cat([seq[:, :lo], vis + delta, seq[:, hi:]], dim=1)
```

One `RMSNorm → Linear(960,d) → SiLU → Linear(d,960)` MLP per expert, **zero-init
output over a residual**, plus a scalar gate initialised to 0. So an adapter that
never trains is the identity map rather than noise — which matters precisely for
an expert the router has starved, since that is the one whose adapter gets no
gradient. At `d=256` this is 0.49M × 4 = **1.98M**, ~0.3% of the decoder.

---

## Parameter budget

Per DiT layer at `hidden=960`, `intermediate=2560`, 15 heads / 5 KV heads / head_dim 64:

| component | params | share |
|---|---|---|
| self-attention | 2.46M | 12.3% |
| VLM-KV cross-attention | 1.84M | 9.2% |
| SwiGLU FFN | 7.37M | 36.9% |
| **adaLN-Zero (9 vectors)** | **8.30M** | **41.6%** |
| RMSNorms | ~0.003M | 0.0% |
| **per layer** | **19.98M** | |

```
1 expert   =  8 layers          =  159.8M
4 experts                       =  639.3M
router                          =    4.61M
sink / state / action / t emb   =    1.93M
------------------------------------------
TRAINABLE (adapters off, LoRA off) ≈ 646M
```

**adaLN-Zero is the single largest block in the decoder — larger than the FFN.**
Each layer carries a `Linear(960, 9×960)` producing per-sample modulation from
`t_emb`. That is the price of making the whole stack time-conditioned.

Encoder params are frozen and do not appear above; vision/text LoRA adds ~2.4M at
rank 64 (SmolVLM2's ViT has 12 layers, so `vision_lora_num_layers > 12` clamps).

> **Optimizer memory follows from this:** Adam holds two fp32 moments per
> trainable parameter, ≈ 2 × 646M × 4 B = **5.2 GB**. This is why the resume path
> must stage the checkpoint and the optimizer state on **CPU** — loading either
> straight to the GPU transiently doubles it.

---

## Flow matching

Identical to wilro; the MoE changes only what computes `v_t`.

| | |
|---|---|
| convention | `x_t = t·noise + (1−t)·a`, target `u_t = noise − a`. **t=1 is noise, t=0 is data.** |
| t sampling | `uniform` (default) or `lognormal` (SD3 logit-normal, `mean −0.5`, `std 1.0`) |
| solver | explicit Euler, **uniform grid**, `dt = −1/N`, starting at `t=1.0`; `N = num_inference_steps` (default 10) |
| execution | `select_action` commits `n_action_steps` then re-samples — **each re-sample draws fresh noise** |

Loss reweighting carried over from wilro: `action_dim_weights`,
`future_steps_weight` past `n_action_steps`, `pos_decay_lambda`,
`gripper_phase_weight` around gripper transitions, and padding masks — all folded
into the **denominator** too, so the loss stays a weighted *mean* and the
effective LR is unchanged.

### Contrastive language loss (`contrastive_loss_weight`, default 0.1)

Permutes only the **language band** `[L_vis : L_vis+L_lang]` of the cached KV
across the batch and re-runs the DiT — no second VLM forward. The wrong-language
prediction is a **detached** negative target, and the second DiT forward runs
under `no_grad`, which avoids storing a full second backward graph (~2× memory).

---

## Failure modes already hit

Recording these because both were **silent** — the loss curve looked normal.

### 1. The router balance penalty carried no gradient at all

The contrastive negative's `_run_dit` runs *after* the real one, under
`no_grad`. Without a guard it **overwrote `_last_router_usage` with a
graph-detached tensor.** The balance penalty — which is applied later, in the
policy's `forward` — then added a *constant* to the loss, contributing exactly
zero gradient.

Measured consequence: **router collapsed to E3 = 100% by step 200**, CV² = 3.0
(the max for 4 experts), entropy 0.000, `Router - Avg Abs Grad: 0.000000`.

Fix: `_run_dit(..., record=False)` at the contrastive call site. Verified after:
`balance.requires_grad=True, grad_fn=DivBackward0`, router-head gradient from the
balance term alone `max|g| = 4.8e-01`.

### 2. `.6f` floored the gradient readout to `0.000000`

On a 646M-parameter average, `{grad:.6f}` prints `0.000000` for a perfectly
healthy gradient — which reads as *exactly* the failure above. Both trainers now
print `{grad:.3e}`.

---

## Differences from the two siblings

### vs `wiltechs_moe` (Qwen3-VL-4B)

| | wiltechs_moe | **wilro_moe** |
|---|---|---|
| backbone | Qwen3-VL-4B | **SmolVLM2-500M** |
| experts × layers | 4 × 9 = 36 | **4 × 8 = 32** (SmolVLM2 has 32 text layers) |
| `dit_hidden` | 1280 (≠ VLM hidden) | **960 (= VLM hidden)** → wilro's `DiTLayer` reused verbatim |
| thought QFormer | yes (18.4M) | **removed** |
| per-expert vision adapters | no | **yes** (opt-in) |

Nothing here imports from `wiltechs_moe`: that module pulls in
`Qwen3VLForConditionalGeneration` at import time, which `cac2de6` had to cut out
of the eval harness. `MoERouter` is reproduced with its reasoning intact.

### vs `wilro`

| | wilro | **wilro_moe** |
|---|---|---|
| decoder | one stack of `num_dit_layers` | **4 experts, disjoint VLM KV bands** |
| Vision CA sublayer | selectable (`use_vision_ca`) | **always off** — vision lives in the sequence |
| DiT layer sublayers | 3 or 4 (adaLN 9 or 12 vec) | **3 (adaLN 9 vec)** |
| latent / thought tokens | latent path exists | **`num_latent_tokens = 0`** |
| async action prefix | supported | **accepted and ignored** |
| encoder | `SmolVLMEncoderMixin` | **same mixin, shared code** |

`_run_dit` keeps wilro's exact signature so the loss and sampling code lifted
from it works unchanged; `latents`, `action_prefix` and `lang_tokens` are
accepted and ignored.

---

## Config quick reference

| field | default | |
|---|---|---|
| `num_experts` | 4 | |
| `expert_num_layers` | 8 | must satisfy `n × depth ≤ 32` |
| `vlm_capture_layers` | `[]` | empty ⇒ all 32; must be divisible by `num_experts` |
| `dit_hidden_size` | 960 | **must equal the VLM hidden** (else `NotImplementedError`) |
| `router_temperature` | 1.0 | |
| `router_top_k` | 0 | 0 = dense |
| `router_balance_weight` | 0.1 | CV² of usage; applied in the policy |
| `resnet_expert_adapter_dim` | 0 | 0 = off |
| `vision_token_source` | `"vlm"` | `"vlm"` (SigLIP layer `-3`) or `"resnet"` |
| `contrastive_loss_weight` | 0.1 | |
| `num_inference_steps` | 10 | |
| `state_dim` / `action_dim` | 7 / 7 | |

> `vision_token_source="resnet"` **severs the only gradient path to the ViT
> LoRA** (the text stack runs under `no_grad` and the KV cache is detached). The
> constructor prints a `[WARN]`; pass `--vision_lora_num_layers 0` to say so in
> the config rather than carrying adapters that never train.

Legacy 2026-06/07 field names (`robot_ca_source`, `use_robot_ca`,
`robot_encoder_*`, `robot_cnn_*`) are accepted and **mirrored** — see wilro's
[ARCHITECTURE.md § Naming](../wilro/ARCHITECTURE.md) and
`src/migrate_wilro_config.py`.
