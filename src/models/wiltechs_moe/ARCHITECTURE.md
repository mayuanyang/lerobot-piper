# WiltechsMoE Architecture (num_experts=4, expert_num_layers=9, thought_tokens=8)

> **Layout change:** the instruction now goes **before** the images in the VLM
> sequence (`text_first=True`) and the language budget is **128 tokens**
> (was 48). See [VLM Input Layout](#vlm-input-layout-text_first-new---default-true).
> This changes what the captured KV contains, so an existing checkpoint will
> see a distribution shift on resume.

## Overview

```
                    ┌──────────────────────────────────┐
                    │     Raw Camera Images            │
                    │  (from dataset, 224x224 each)    │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────┴─────────────────────┐
                    │                                    │
                    ▼                                    ▼
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│  FROZEN: Qwen Image Processor       │    │  TRAINABLE: Robot CNN       │
│  (pixel_values, grid_thw)           │    │  (per-camera CNN +          │
│         │                           │    │   Spatial Softmax)          │
│         ▼                           │    │         │                   │
│  ┌───────────┐                      │    │  ┌──────────────────────┐   │
│  │  Visual   │                      │    │  │ Camera 0 -> CNN      │   │
│  │  Encoder  │                      │    │  │  -> SpatialSoftmax   │   │
│  │  (frozen) │                      │    │  │  -> 16 tokens (1280d)│   │
│  └─────┬─────┘                      │    │  ├──────────────────────┤   │
│        │ vision_tokens               │    │  │ Camera 1 -> CNN      │   │
│        ▼                            │    │  │  -> SpatialSoftmax   │   │
│  ┌───────────────────────────────┐  │    │  │  -> 16 tokens (1280d)│   │
│  │  Language Model (36 layers)   │  │    │  ├──────────────────────┤   │
│  │  (frozen)                     │  │    │  │ Camera 2 -> CNN      │   │
│  │                               │  │    │  │  -> SpatialSoftmax   │   │
│  │  Input: [INSTRUCTION | vision]│  │    │  │  -> 16 tokens (1280d)│   │
│  │  L0 -> L1 -> ... -> L35       │  │    │  └──────────┬───────────┘   │
│  │                               │  │    │             │               │
│  │  Captures KV at 36 layers:    │  │    │  Training-time dropout     │
│  │  E0: L0-8   E1: L9-17        │  │    │  (prob=0.3, forces VLM     │
│  │  E2: L18-26 E3: L27-35       │  │    │   vision grounding)        │
│  └──────────────┬────────────────┘  │    │             │               │
└─────────────────┼───────────────────┘    └─────────────┼───────────────┘
                  │                                      │
                  │ KV cache (36 layers)                 │ robot_tokens
                  │ (B, seq, 2560)                       │ (B, 16-48, 1280)
                  │                                      │
         ┌────────┴─────────┐                           │
         │                  │                           │
         ▼                  ▼                           │
┌──────────────────┐  ┌──────────────────┐              │
│  Thought QFormer │  │  Expert KV Cache │              │
│  (TRAINABLE)     │  │  (per expert)    │              │
│                  │  │                  │              │
│  Reads KV from   │  │  E0: L0-8       │              │
│  deepest VLM     │  │  E1: L9-17      │              │
│  layer           │  │  E2: L18-26     │              │
│                  │  │  E3: L27-35     │              │
│  8 learnable     │  │                  │              │
│  query tokens    │  └────────┬─────────┘              │
│  cross-attend    │           │                        │
│  to VLM KV       │           │                        │
│                  │           │                        │
│  dim=1280        │           │                        │
│  heads=8         │           │                        │
│  kv_heads=4      │           │                        │
│  2 layers        │           │                        │
│  ~12.3M params   │           │                        │
└────────┬─────────┘           │                        │
         │                     │                        │
         │ thought_tokens      │                        │
         │ (B, 8, 1280)        │                        │
         │                     │                        │
         ▼                     ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINABLE MoE Decoder (~1.53B params)                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Expert Input Sequence                            │  │
│  │                                                                      │  │
│  │  [sink] [state] [robot_cnn_toks] [thought_toks] [noisy_action_1...64]│  │
│  │    1      1       16-48           8             64                    │  │
│  │                                                                      │  │
│  │  + timestep embedding (adaLN modulation)                             │  │
│  └──────────────────────────┬───────────────────────────────────────────┘  │
│                             │                                              │
│              ┌──────────────┼──────────────┐                              │
│              │              │              │                              │
│              ▼              ▼              ▼                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ │
│  │   Expert 0    │ │   Expert 1    │ │   Expert 2    │ │   Expert 3    │ │
│  │  (9 layers)   │ │  (9 layers)   │ │  (9 layers)   │ │  (9 layers)   │ │
│  │               │ │               │ │               │ │               │ │
│  │ DiT Layer 0   │ │ DiT Layer 0   │ │ DiT Layer 0   │ │ DiT Layer 0   │ │
│  │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │
│  │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │
│  │  K,V<-L0-8   │ │  K,V<-L9-17  │ │  K,V<-L18-26 │ │  K,V<-L27-35 │ │
│  │  MLP          │ │  MLP          │ │  MLP          │ │  MLP          │ │
│  │     ...       │ │     ...       │ │     ...       │ │     ...       │ │
│  │ DiT Layer 8   │ │ DiT Layer 8   │ │ DiT Layer 8   │ │ DiT Layer 8   │ │
│  │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │
│  │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │
│  │  K,V<-L0-8   │ │  K,V<-L9-17  │ │  K,V<-L18-26 │ │  K,V<-L27-35 │ │
│  │  MLP          │ │  MLP          │ │  MLP          │ │  MLP          │ │
│  │               │ │               │ │               │ │               │ │
│  │ action_out    │ │ action_out    │ │ action_out    │ │ action_out    │ │
│  │ v0 (64x7)     │ │ v1 (64x7)     │ │ v2 (64x7)     │ │ v3 (64x7)     │ │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ │
│          │    w0           │    w1           │    w2           │    w3   │
│          └─────────────────┴─────────────────┴─────────────────┘          │
│                                    │                                      │
│                           ┌────────▼────────┐                             │
│                           │  Weighted Sum   │                             │
│                           │  v_t = Sum(wi*vi)│                            │
│                           └────────┬────────┘                             │
│                                    │                                      │
│                                    ▼                                      │
│                              v_t (64x7)                                    │
│                        (velocity field)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## VLM Input Layout: text_first (NEW - default True)

The VLM is **causal**. Token order therefore decides what the captured KV
actually contains, and it is the single biggest lever on referring-expression
grounding.

```
LEGACY (text_first=False)                  CURRENT (text_first=True, default)
─────────────────────────                  ──────────────────────────────────
[<|im_start|>user\n]                       [<|im_start|>user\n]
[<vis> cam0 </vis>]                        [ INSTRUCTION  (<=128 tokens) ]
[<vis> cam1 </vis>]                        [<vis> cam0 </vis>]
[<vis> cam2 </vis>]                        [<vis> cam1 </vis>]
[ INSTRUCTION ]                            [<vis> cam2 </vis>]
[<|im_end|> assistant]                     [<|im_end|> assistant]

   Under the causal mask:                     Under the causal mask:
   vision KV attends to                       vision KV attends to
   -> images only                             -> images AND the instruction
   = LANGUAGE-BLIND                           = LANGUAGE-GROUNDED

   ~590 vision positions carry                every one of the ~590 vision
   no language; the referring                 positions is conditioned on
   expression survives only in                "the black bowl BETWEEN the
   the ~50 trailing text KV,                  plate and the ramekin", so the
   which the experts' cross-attn              experts cross-attend to a
   softmax dilutes to ~8%.                    language-grounded feature map.
```

**Failure mode this fixes.** With the legacy layout the cheapest thing the
DiT can learn is to treat the language embedding as a coarse *location prior*
("somewhere near the plate and the ramekin") rather than as an *object
selector*. Observed behaviour on
`pick up the black bowl between the plate and the ramekin ...`: the gripper
descends on the geometric midpoint between the two anchors, where no object
exists.

**Token budget.** `_lang_max_len = 128` (was 48). The CoT rewrites emitted by
`task_rewrites.cot()` under `--use_descriptive_objects` run ~110 tokens; at 48
they were truncated mid-Location, silently dropping both the anti-grounding
cue (`not the midpoint between the plate and the ramekin`) and the entire
`Action:` clause — the model was trained on a prompt that literally ended with
"in the gap between the plate and the ramekin". The model prints the longest
instruction, the kept text, and any dropped tail once at startup:

```
[wiltechs_moe] lang budget: max_len=128, longest instruction in batch=108 tokens, kept=108
[wiltechs_moe]   kept: 'Target: the black bowl — a round dark bowl. Location: ... Action: ...'
```

**Cost.** Padded instruction positions sit mid-sequence; they are masked out as
attention *keys*, so the images never read them. M-RoPE positions are built
from the padded length uniformly across the batch — a constant offset on the
image block, which RoPE is invariant to. Forward cost is unchanged except for
the contrastive branch (see Training Losses).

Set `--text_last` to restore the legacy layout.

## Thought QFormer Detail (NEW - Trainable, ~12.3M params)

```
   VLM KV Cache (deepest captured layer)
   (B, num_kv_heads=4, L_vlm, head_dim=128)
            │
            ▼
   ┌────────────────────────┐
   │  8 Learnable Query     │    nn.Parameter(1, 8, 1280)
   │  Tokens (1280d)        │    Random init (std=0.02)
   └────────┬───────────────┘
            │
            ▼
   ┌────────────────────────┐
   │  Cross-Attention Layer │    Q: Linear(1280 -> 8*128=1024)
   │  (Layer 0)             │    K,V: from VLM (4 kv_heads, GQA expand)
   │                        │    O: Linear(1024 -> 1280)
   │  + Residual gate (0.1) │    Gate init: 0.1 (gentle start)
   │  + SwiGLU FFN          │    intermediate_size = dit_intermediate
   │  + Residual gate (0.1) │
   └────────┬───────────────┘
            │
            ▼
   ┌────────────────────────┐
   │  Cross-Attention Layer │    Same structure as Layer 0
   │  (Layer 1)             │
   │                        │
   │  + Residual gate (0.1) │
   │  + SwiGLU FFN          │
   │  + Residual gate (0.1) │
   └────────┬───────────────┘
            │
            ▼
      8 thought tokens (1280d)

   Purpose:
     - Compresses VLM visual+language understanding into
       compact spatial reasoning tokens
     - QFormer cross-attends to VLM KV, extracting:
       * Object positions (where to reach)
       * Spatial relationships (between plate and ramekin)
       * Task-relevant visual features
     - These tokens are injected into expert input sequence,
       giving all experts access to VLM's reasoning
     - Contrastive loss ensures thoughts are language-grounded
       (shuffled language -> different thoughts -> different actions)

   Head config:
     - ca_num_heads = 8 (divisible by VLM num_kv_heads=4)
     - ca_num_kv_heads = 4 (matches VLM, K/V come from VLM)
     - ca_head_dim = 128
     - GQA ratio = 8/4 = 2 (repeat_interleave K/V by 2)
```

## Robot CNN Detail (Trainable, parallel to VLM)

```
   Raw Camera Image (224x224)
            │
            ▼
   ┌────────────────────┐
   │  Conv layers       │    4x Conv2d + ReLU + BatchNorm
   │  (feature extract) │    224 -> 112 -> 56 -> 28 -> 14
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────┐
   │  Spatial Softmax   │    14x14 feature map -> 16 tokens
   │  (4x4 grid)        │    Each token = softmax over spatial dims
   └────────┬───────────┘    -> select max-activation positions
            │
            ▼
   ┌────────────────────┐
   │  Linear projection │    -> 1280 dim (match dit_hidden_size)
   └────────┬───────────┘
            │
            ▼
      16 tokens (1280d)     per camera

   If 3 cameras: concat -> 48 tokens total
   If 1 camera (wrist only): 16 tokens

   Training: 30% of tokens randomly dropped to zero
   (forces model to also use VLM vision, not just CNN)
```

## Router Detail

```
                    ┌─────────────────────────────────────┐
                    │          MoE Router                 │
                    │                                     │
                    │  Input:                             │
                    │    state_emb      (B, 1, 1280)     │
                    │    vlm_semantic   (B, 2560)        │
                    │    time_emb       (B, 1280)        │
                    │    action_emb     (B, 64, 1280)    │
                    │                                     │
                    │  vlm_semantic = mean_pool(          │
                    │    VLM final hidden state)          │
                    │  Contains BOTH vision + language    │
                    │  (VLM causal attention mixes them)  │
                    │                                     │
                    │  vlm_proj: Linear(2560->1280)       │
                    │  Concat [state|vlm_proj|time|action]│
                    │    -> Linear(5120->1280) -> SiLU    │
                    │    -> Linear(1280->4)               │
                    │    -> /temperature                  │
                    │    -> softmax                       │
                    │                                     │
                    │  Output: weights (B, 4)             │
                    │    w0, w1, w2, w3                   │
                    └─────────────────────────────────────┘

  VLM semantic input:
    - Router sees what the VLM understands about the
      scene and instruction -> can route to deeper experts
      for complex reasoning, shallower for precise control
    - Training noise (0.5 std) prevents router collapse
```

## VLM KV Cache Flow (4 experts x 9 layers = 36 layers captured)

```
Qwen3-VL-4B (36 layers)
  Layer 0  ----------------> Expert 0 -+
  Layer 1  ----------------> Expert 0 -+
  Layer 2  ----------------> Expert 0 -+
  Layer 3  ----------------> Expert 0 -+
  Layer 4  ----------------> Expert 0 -+
  Layer 5  ----------------> Expert 0 -+
  Layer 6  ----------------> Expert 0 -+
  Layer 7  ----------------> Expert 0 -+
  Layer 8  ----------------> Expert 0 -+
  --------------------------------------
  Layer 9  ----------------> Expert 1 -+
  Layer 10 ----------------> Expert 1 -+
  Layer 11 ----------------> Expert 1 -+
  Layer 12 ----------------> Expert 1 -+
  Layer 13 ----------------> Expert 1 -+
  Layer 14 ----------------> Expert 1 -+
  Layer 15 ----------------> Expert 1 -+
  Layer 16 ----------------> Expert 1 -+
  Layer 17 ----------------> Expert 1 -+
  --------------------------------------
  Layer 18 ----------------> Expert 2 -+
  Layer 19 ----------------> Expert 2 -+
  Layer 20 ----------------> Expert 2 -+
  Layer 21 ----------------> Expert 2 -+
  Layer 22 ----------------> Expert 2 -+
  Layer 23 ----------------> Expert 2 -+
  Layer 24 ----------------> Expert 2 -+
  Layer 25 ----------------> Expert 2 -+
  Layer 26 ----------------> Expert 2 -+
  --------------------------------------
  Layer 27 ----------------> Expert 3 -+
  Layer 28 ----------------> Expert 3 -+
  Layer 29 ----------------> Expert 3 -+
  Layer 30 ----------------> Expert 3 -+
  Layer 31 ----------------> Expert 3 -+
  Layer 32 ----------------> Expert 3 -+
  Layer 33 ----------------> Expert 3 -+
  Layer 34 ----------------> Expert 3 -+
  Layer 35 ----------------> Expert 3 -+

  Thought QFormer reads from:
    Layer 35 (deepest, -1 index)
    (configurable via thought_vlm_layer_idx)
```

## DiT Layer Detail (inside each expert)

```
┌─────────────────────────────────────────────────────────┐
│                    DiT Layer                             │
│                                                          │
│  x --+--> RMSNorm --> Self-Attention --> +              │
│      |         ^                          |              │
│      |         | adaLN modulation         |              │
│      |         | (from timestep emb)      |              │
│      |         v                          v              │
│      |      [shift/scale/gate]    +-------+-------+      │
│      |                           |     + Add     |<- x   │
│      |                           +-------+-------+      │
│      |                                   |               │
│      +<----------------------------------+               │
│      |                                                   │
│      +--> RMSNorm --> Cross-Attention --> +             │
│      |         ^             Q = x          |            │
│      |         | adaLN       K,V = VLM KV   |            │
│      |         | modulation  (expert's      |            │
│      |         |             layers)        |            │
│      |         v                          v             │
│      |      [shift/scale/gate]    +-------+-------+     │
│      |                           |     + Add     |<-res │
│      |                           +-------+-------+     │
│      |                                   |              │
│      +<----------------------------------+              │
│      |                                                   │
│      +--> RMSNorm --> SwiGLU MLP --> +                  │
│      |         ^       gate_proj      |                 │
│      |         |       up_proj        |                 │
│      |         | adaLN  down_proj     |                 │
│      |         v                     v                  │
│      |      [shift/scale/gate]    +-------+-------+     │
│      |                           |     + Add     |<-res │
│      |                           +-------+-------+     │
│      |                                   |              │
│      +<----------------------------------+              │
│      |                                                   │
│      v                                                   │
│      x' (output to next layer)                          │
└─────────────────────────────────────────────────────────┘
```

## Parameter Summary (num_experts=4, expert_num_layers=9, dit_hidden=1280)

```
┌────────────────────────────────────────────────────────────┐
│ Component              │   Params   │  Trainable?         │
├────────────────────────────────────────────────────────────┤
│ Qwen3-VL-4B (frozen)   │  ~4.0B     │  Frozen             │
│ Expert 0 (9 layers)    │  ~313M     │  Trainable          │
│ Expert 1 (9 layers)    │  ~313M     │  Trainable          │
│ Expert 2 (9 layers)    │  ~313M     │  Trainable          │
│ Expert 3 (9 layers)    │  ~313M     │  Trainable          │
│ Router                 │  ~9.8M     │  Trainable          │
│ Robot CNN encoder      │  ~3.1M     │  Trainable          │
│ Thought QFormer        │  ~12.3M    │  Trainable  (NEW)   │
│ Sink token             │  ~1.3K     │  Trainable          │
│ State encoder          │  ~0.9M     │  Trainable          │
│ Action in/out proj     │  ~0.1M     │  Trainable          │
│ Action pos emb         │  ~0.08M    │  Trainable          │
│ Time embedder          │  ~3.3M     │  Trainable          │
│ Final norm             │  ~2.6K     │  Trainable          │
├────────────────────────────────────────────────────────────┤
│ Total trainable        │  ~1.28B    │                     │
│ Total frozen           │  ~4.0B     │                     │
│ Grand total            │  ~5.3B     │                     │
└────────────────────────────────────────────────────────────┘
```

## Expert Input Sequence (NEW: includes thought tokens)

```
Position:  0      1      2..49       50..57      58..121
         [sink] [state] [robot_cnn] [thoughts] [actions]
           1      1      16-48         8          64

  sink     - learnable global context token
  state    - current robot joint angles (7d -> 1280d)
  robot    - CNN spatial tokens (16 per camera)
  thoughts - QFormer output (8 tokens, VLM spatial reasoning)
  actions  - noisy action trajectory (64 steps x 7d -> 1280d)

  Thought tokens give experts access to:
    - Object spatial positions from VLM
    - Language-grounded task understanding
    - Visual reasoning compressed into compact tokens
```

## Flow Matching Inference

```
t=1.0 (noise) --> [VLM KV cache + thoughts computed once]
                   |
                   Expert 0 --+
                   Expert 1 --+--> v_t --> x_{t-dt}
                   Expert 2 --+               |
                   Expert 3 --+               |
                                             |
t=0.8           --> ... ------------------> |
t=0.6           --> ... ------------------> |
t=0.4           --> ... ------------------> |
t=0.2           --> ... ------------------> |
t=0.0 (action)  <---------------------------+
                   (5 inference steps)

  Note: VLM forward pass runs ONCE at t=1.0
        KV cache + thought tokens reused for all 5 steps
```

## Training Losses

```
  Total Loss = main_loss + w_contrastive * contrastive_loss + w_balance * balance_loss

  main_loss:       Flow matching MSE (weighted by position decay)
  contrastive:     Hinge loss on action difference when language is shuffled
                   (ensures thought tokens are language-grounded)
  balance:         CV² of router usage (prevents expert collapse)

  How the "wrong language" negative is built depends on the layout:

    text_first=False   swap the KV slice [lang_start:] between batch members.
    (legacy)           Valid because vision KV is language-blind, so only the
                       trailing text positions carry the instruction.
                       Cost: 0 extra VLM forwards.

    text_first=True    the instruction is baked into EVERY vision KV, so there
    (default)          is no slice to swap -- swapping only the text span would
                       leave the vision KV still carrying the correct language
                       and the negative would be a lie. Instead the frozen LM
                       is re-run with permuted instructions.
                       Cost: +1 LM forward (36 layers, no_grad). The ViT output
                       is reused via vis_pack, so the image encoder does NOT
                       re-run. Expect roughly +25-40% step time while
                       contrastive_loss_weight > 0.
                       The same vision-KV-dropout mask is re-applied to the
                       negative, so the hinge measures language, not dropout.

  Defaults:
    contrastive_loss_weight = 0.1
    router_balance_weight   = 0.1
    contrastive_margin      = 0.05