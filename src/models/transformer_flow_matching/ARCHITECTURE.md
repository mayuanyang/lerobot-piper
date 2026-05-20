# TransformerFlowMatching — Architecture Diagram

## Overview

Flow-matching policy for the Piper 7-DOF robot arm.  
Frozen SmolVLM2-500M encodes images + language into per-layer VLM hidden states.  
A trainable action expert (16-layer TransformerDecoder) cross-attends to those states and predicts a velocity field over a horizon of actions.

---

## Full Forward Pass

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  INPUTS                                                                         │
│                                                                                 │
│  Camera images (3 cams)   Robot state (7-DOF)   Task description (text)        │
│  (B, C, H, W) × 3        (B, 7)                 list[str]                      │
└──────────────┬──────────────────────┬───────────────────┬────────────────────┘
               │                      │                   │
               ▼                      │                   ▼
┌──────────────────────────────────┐  │  ┌────────────────────────────────────┐
│  VLM ENCODER  [frozen, bfloat16] │  │  │  LANGUAGE ENCODER  [frozen]        │
│                                  │  │  │                                    │
│  ┌───────────────────────────┐   │  │  │  processor.tokenizer               │
│  │  SigLIP ViT               │   │  │  │      ↓                             │
│  │  384×384 → 27×27 patches  │   │  │  │  embed_tokens (frozen)             │
│  │  patch_size=14, dim=768   │   │  │  │      ↓                             │
│  │  → (B, 729, 768)          │   │  │  │  lang_tokens (B, L≤48, 960)        │
│  └───────────┬───────────────┘   │  │  └──────────────────┬─────────────────┘
│              ↓                   │  │                     │
│  ┌───────────────────────────┐   │  │                     │
│  │  Connector                │   │  │                     │
│  │  pixel-shuffle + MLP      │   │  │                     │
│  │  (B, 729, 768)            │   │  │                     │
│  │    → (B, V_tok, 960)      │   │  │                     │
│  └───────────┬───────────────┘   │  │                     │
└──────────────┼───────────────────┘  │                     │
               │                      │                     │
               └──────────────────────▼─────────────────────▘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  prefix_embs         │
                           │  cat([vis, lang])    │
                           │  (B, N, 960)         │
                           └──────────┬───────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │  SmolLM2 Text Transformer  [frozen]   │
                  │                                       │
                  │  First 16 of 32 layers                │
                  │  (B, N, 960) → 16 × (B, N, 960)      │
                  │                                       │
                  │  hidden_states[0] = embeddings        │
                  │  hidden_states[1..16] = layer outputs │
                  └───────────────────────────────────────┘
                  ↓ vlm_layers: list of 16 × (B, N, 960)

═══════════════════════════════════════════════════════════════════

  PARALLEL TRAINABLE BRANCH: Robot-Specific Visual Encoder

  Camera images (3 cams)
  (B, C, H, W) × 3
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │  RobotVisualEncoder  [trainable, ResNet-18 based]    │
  │  (per camera)                                        │
  │                                                      │
  │  Resize → 224×224                                    │
  │  ImageNet normalize                                  │
  │                                                      │
  │  stem: conv1+bn+relu+maxpool  → (B, 64,  56, 56)    │
  │  layer1                       → (B, 64,  56, 56)    │
  │  layer2                       → (B, 128, 28, 28)    │
  │  layer3  [layer4 excluded]    → (B, 256, 14, 14)    │
  │  AdaptiveAvgPool → (4×4)      → (B, 256,  4,  4)    │
  │  flatten+transpose            → (B, 16, 256)         │
  │  Linear(256 → 512) + LayerNorm → (B, 16, 512)       │
  └──────────────────────────────────────────────────────┘
  robot_tokens per camera: (B, 16, 512)
  concat 3 cams → robot_tokens: (B, 48, 512)

═══════════════════════════════════════════════════════════════════

  CONTEXT ASSEMBLY  [trainable projections]

  For each decoder layer i ∈ [0..15]:

    vlm_i = vlm_layers[i]                    # 1:1 mapping (n_dec == n_vlm == 16)
    ctx   = LayerNorm(MLP(vlm_i))            # (B, N, 512)   [MLP shared across layers]
           # context_proj = Linear(960→512) → GELU → Linear(512→512)
           # context_norm = LayerNorm(512)

    robot_ctx_i = robot_layer_projs[i](robot_tokens)  # (B, 48, 512)
                                                       # per-layer Linear, no bias

    state = Linear(7→512) + LayerNorm        # state_encoder: (B, 1, 512)

    per_layer_contexts[i] = cat([ctx, robot_ctx_i, state])
                          # (B, N+48+1, 512)

═══════════════════════════════════════════════════════════════════

  FLOW MATCHING — TRAINING

  actions (B, H=4, 7)   noise ~ AR(1) or N(0,I)
          │                       │
          └──────────┬────────────┘
                     │   t ~ Uniform(0.001, 0.999)
                     ▼
            x_t = t·noise + (1-t)·actions     (noisy sample)
            u_t = noise - actions              (target velocity)
                     │
                     ▼
             velocity_field(x_t, t, contexts) → v_t
                     │
                     ▼
            loss = MSE(v_t, u_t)
                   × dim_weights  (per action dim)
                   × pos_weights  (exp decay + future_steps_weight=0.3)
                   masked by action_is_pad

═══════════════════════════════════════════════════════════════════

  ACTION EXPERT (velocity_field)

  noisy_actions (B, H, 7)
        │
        ▼
  ┌──────────────────────────────────────────────┐
  │  action_in_proj  Linear(7 → 512)             │
  │  PositionalEncoding (sinusoidal, H steps)    │
  │  → action_emb  (B, H, 512)                   │
  └───────────────────┬──────────────────────────┘
                      │
  timestep t (B,)     │
        │             │
        ▼             │
  sinusoidal_emb      │
  (B, 512) expand     │
  → time_emb (B,H,512)│
                      │
        ┌─────────────┘
        │   action_emb
        │
        ▼
  cat([action_emb, time_emb]) → (B, H, 1024)
        │
        ▼
  SiLU(Linear(1024 → 512)) → Linear(512 → 512)
        │   action_time_mlp_in/out
        │
        ▼
  fused + (1-t) × action_emb    # time-scaled residual blend
  → tgt  (B, H, 512)
        │
        ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  16 × TransformerDecoderLayer (Pre-LN, d=512, nhead=8, ff=2048)│
  │                                                                │
  │  Layer i:                                                      │
  │    ┌─────────────────────────────────────────────────────┐     │
  │    │  Self-Attention (causal mask)                       │     │
  │    │    tgt (B, H, 512) attends to itself                │     │
  │    ├─────────────────────────────────────────────────────┤     │
  │    │  Cross-Attention                                    │     │
  │    │    Q = tgt                                          │     │
  │    │    K,V = per_layer_contexts[i]  (B, N+49, 512)     │     │
  │    │          = VLM layer i + robot_layer_projs[i] + state│    │
  │    ├─────────────────────────────────────────────────────┤     │
  │    │  FFN: Linear(512→2048) → GELU → Linear(2048→512)   │     │
  │    └─────────────────────────────────────────────────────┘     │
  └────────────────────────────────────────────────────────────────┘
        │
        ▼
  LayerNorm (action_expert_norm)
        │
        ▼
  Linear(512 → 7)  [action_out_proj, zero-init]
        │
        ▼
  velocity  (B, H, 7)

═══════════════════════════════════════════════════════════════════

  INFERENCE — Euler ODE Integration

  x_1 ~ sample_noise(B, H=4, 7)      # t=1: pure noise
  t   = 1.0

  for step in range(10):              # num_inference_steps=10
      v_t = velocity_field(x_t, t, contexts)
      x_t = x_t + (-1/10) × v_t      # Euler step toward t=0
      t   = t - 1/10

  return x_0[:, :n_action_steps]     # (B, 4, 7) — first 4 of horizon
```

---

## Component Summary Table

| Component | Module | Params | Frozen? |
|---|---|---|---|
| SigLIP ViT | `vision_model` (SmolVLM2) | ~300M | ✅ (LoRA opt-in: last 8 layers) |
| Connector (pixel-shuffle MLP) | `connector` | ~5M | ✅ (opt-in unfreeze) |
| SmolLM2 text transformer | `text_model` (16 of 32 layers) | ~240M | ✅ always |
| context_proj + context_norm | `Linear(960→512) → GELU → Linear(512→512) + LN` | 0.76M | ❌ |
| state_encoder | `Linear(7→512) + LN` | 0.3M | ❌ |
| RobotVisualEncoder (×3 cams) | ResNet-18 stem+L1-3 + proj | ~11M | ❌ |
| robot_layer_projs (16 layers) | `Linear(512→512)` × 16 (no bias) | 4.2M | ❌ |
| action_in_proj | `Linear(7→512)` | 4K | ❌ |
| action_time_mlp | `Linear(1024→512) + Linear(512→512)` | 0.8M | ❌ |
| PositionalEncoding | sinusoidal buffer | 0 | — |
| ActionExpert (16 layers) | `TransformerDecoderLayer` × 16 | ~67M | ❌ |
| action_expert_norm | `LayerNorm(512)` | 1K | ❌ |
| action_out_proj | `Linear(512→7)` (zero-init) | 4K | ❌ |
| **Total trainable** | | **~83M** | |
| **Total frozen** | | **~545M** | |

---

## Data Flow Dimensions (default config)

```
vision_input_size = 384     → ViT patches: 27×27 = 729 per cam
connector output            → V_tok ≈ 182 tokens per cam (pixel-shuffle 4:1)
num_cameras = 3             → V_tok_total ≈ 546 vision tokens
language tokens (L ≤ 48)   → N = V_tok_total + L ≈ 594 context tokens
VLM text layers used        = 16 of 32

state_dim        = 7        → 1 state token per obs step (T_obs=1)
action_dim       = 7
horizon H        = 4        → action sequence length
n_action_steps   = 4        → steps executed per inference call
d_model          = 512
nhead            = 8        → 64-dim per head
num_decoder_layers = 16     → 1:1 with num_vlm_layers
dim_feedforward  = 2048

robot_encoder_tokens = 16   → 4×4 spatial grid per camera
                               × 3 cams = 48 robot tokens per layer

per_layer_context size = 594 (VLM) + 48 (robot) + 1 (state) = 643 tokens
```

---

## Key Design Decisions

**Interleaved VLM-depth cross-attention** — each of the 16 decoder layers
cross-attends to its corresponding VLM text layer output (1:1 mapping now that
`num_decoder_layers == num_vlm_layers == 16`). Early decoder layers see
low-level visual features; later ones see high-level semantics. Mirrors the
SmolVLA architecture.

**Per-layer robot token projections** (`robot_layer_projs`) — the same
ResNet base features are projected by a dedicated `Linear` for each decoder
layer. This eliminates conflicting gradients that would arise from sharing
identical robot tokens across all 16 cross-attention operations.

**Shared VLM context projection** (`context_proj`) — by contrast, the VLM
adapter is a *single* MLP used by all 16 layers. The input differs each
iteration (different VLM depth), so the shared MLP still produces different
outputs per layer, while saving ~11M params versus a per-layer variant.

**Zero-init action_out_proj** — initial velocity predictions are ~0, so the
initial loss ≈ E[||u_t||²] ≈ 2 (not millions), preventing early training
instability.

**Two-tier loss weighting** — executed steps (0..3) weight 1.0; future steps
(4..H-1) weight 0.3 (`future_steps_weight`). An exponential decay
`exp(-0.1 × position)` further concentrates gradient on earlier steps within
each tier.

**AR(1) noise** (`noise_temporal_correlation`) — optional temporally
correlated source noise matches robot trajectory smoothness, creating straighter
flow paths from noise to action. Default 0.0 (white Gaussian).
