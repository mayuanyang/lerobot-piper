# WiltechsMoE Architecture (num_experts=4, expert_num_layers=8)

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
│  │  Input: [vision | language]   │  │    │  │  -> 16 tokens (1280d)│   │
│  │  L0 -> L1 -> ... -> L35       │  │    │  └──────────┬───────────┘   │
│  │                               │  │    │             │               │
│  │  Captures KV at 32 layers:    │  │    │  Training-time dropout     │
│  │  E0: L0-7  E1: L8-15         │  │    │  (prob=0.3, forces VLM     │
│  │  E2: L16-23 E3: L24-31       │  │    │   vision grounding)        │
│  └──────────────┬────────────────┘  │    │             │               │
└─────────────────┼───────────────────┘    └─────────────┼───────────────┘
                  │                                      │
                  │ KV cache (32 layers)                 │ robot_tokens
                  │ (B, seq, 2560)                       │ (B, 16-48, 1280)
                  ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINABLE MoE Decoder (~1.1B params)                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Expert Input Sequence                            │  │
│  │                                                                      │  │
│  │  [sink] [state] [robot_cnn_toks] [noisy_action_1...action_64]       │  │
│  │    1      1       16-48           64                                 │  │
│  │                                                                      │  │
│  │  + timestep embedding (adaLN modulation)                             │  │
│  └──────────────────────────┬───────────────────────────────────────────┘  │
│                             │                                              │
│              ┌──────────────┼──────────────┐                              │
│              │              │              │                              │
│              ▼              ▼              ▼                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ │
│  │   Expert 0    │ │   Expert 1    │ │   Expert 2    │ │   Expert 3    │ │
│  │  (8 layers)   │ │  (8 layers)   │ │  (8 layers)   │ │  (8 layers)   │ │
│  │               │ │               │ │               │ │               │ │
│  │ DiT Layer 0   │ │ DiT Layer 0   │ │ DiT Layer 0   │ │ DiT Layer 0   │ │
│  │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │
│  │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │
│  │  K,V<-L0-7   │ │  K,V<-L8-15  │ │  K,V<-L16-23 │ │  K,V<-L24-31 │ │
│  │  MLP          │ │  MLP          │ │  MLP          │ │  MLP          │ │
│  │     ...       │ │     ...       │ │     ...       │ │     ...       │ │
│  │ DiT Layer 7   │ │ DiT Layer 7   │ │ DiT Layer 7   │ │ DiT Layer 7   │ │
│  │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │  Self-Attn    │ │
│  │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │  Cross-Attn   │ │
│  │  K,V<-L0-7   │ │  K,V<-L8-15  │ │  K,V<-L16-23 │ │  K,V<-L24-31 │ │
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
                    │    time_emb       (B, 1280)        │
                    │    action_emb     (B, 64, 1280)    │
                    │    [latent=None -> zeros]           │
                    │                                     │
                    │  Concat -> Linear(5120->1280) ->SiLU│
                    │         -> Linear(1280->4)          │
                    │         -> /temperature             │
                    │         -> softmax                  │
                    │                                     │
                    │  Output: weights (B, 4)             │
                    │    w0, w1, w2, w3                   │
                    └─────────────────────────────────────┘
```

## VLM KV Cache Flow (4 experts x 8 layers = 32 layers captured)

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
  --------------------------------------
  Layer 8  ----------------> Expert 1 -+
  Layer 9  ----------------> Expert 1 -+
  Layer 10 ----------------> Expert 1 -+
  Layer 11 ----------------> Expert 1 -+
  Layer 12 ----------------> Expert 1 -+
  Layer 13 ----------------> Expert 1 -+
  Layer 14 ----------------> Expert 1 -+
  Layer 15 ----------------> Expert 1 -+
  --------------------------------------
  Layer 16 ----------------> Expert 2 -+
  Layer 17 ----------------> Expert 2 -+
  Layer 18 ----------------> Expert 2 -+
  Layer 19 ----------------> Expert 2 -+
  Layer 20 ----------------> Expert 2 -+
  Layer 21 ----------------> Expert 2 -+
  Layer 22 ----------------> Expert 2 -+
  Layer 23 ----------------> Expert 2 -+
  --------------------------------------
  Layer 24 ----------------> Expert 3 -+
  Layer 25 ----------------> Expert 3 -+
  Layer 26 ----------------> Expert 3 -+
  Layer 27 ----------------> Expert 3 -+
  Layer 28 ----------------> Expert 3 -+
  Layer 29 ----------------> Expert 3 -+
  Layer 30 ----------------> Expert 3 -+
  Layer 31 ----------------> Expert 3 -+
  --------------------------------------
  Layer 32  (not captured)
  Layer 33  (not captured)
  Layer 34  (not captured)
  Layer 35  (not captured)
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

## Parameter Summary (num_experts=4, expert_num_layers=8, dit_hidden=1280)

```
┌────────────────────────────────────────────────────────────┐
│ Component              │   Params   │  Trainable?         │
├────────────────────────────────────────────────────────────┤
│ Qwen3-VL-4B (frozen)   │  ~4.0B     │  Frozen             │
│ Expert 0 (8 layers)    │  ~279M     │  Trainable          │
│ Expert 1 (8 layers)    │  ~279M     │  Trainable          │
│ Expert 2 (8 layers)    │  ~279M     │  Trainable          │
│ Expert 3 (8 layers)    │  ~279M     │  Trainable          │
│ Router                 │  ~6.6M     │  Trainable          │
│ Robot CNN encoder      │  ~3.1M     │  Trainable          │
│ Sink token             │  ~1.3K     │  Trainable          │
│ State encoder          │  ~0.9M     │  Trainable          │
│ Action in/out proj     │  ~0.1M     │  Trainable          │
│ Action pos emb         │  ~0.08M    │  Trainable          │
│ Time embedder          │  ~3.3M     │  Trainable          │
│ Final norm             │  ~2.6K     │  Trainable          │
├────────────────────────────────────────────────────────────┤
│ Total trainable        │  ~1.13B    │                     │
│ Total frozen           │  ~4.0B     │                     │
│ Grand total            │  ~5.1B     │                     │
└────────────────────────────────────────────────────────────┘
```

## Flow Matching Inference

```
t=1.0 (noise) --> Expert 0 --+
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