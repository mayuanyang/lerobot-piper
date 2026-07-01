# WILRO Architecture

Encoder-decoder flow matching VLA built on a frozen SmolVLM2-500M backbone.
Mixture-of-Transformers (MoT) layout — the VLM never sees state/action tokens.

- **Encoder** = the SmolVLM2 text stack. Runs **once per observation**, captures
  post-RoPE K/V from its trailing `num_dit_layers` layers as cross-attention
  memory for the DiT.
- **Decoder** = a `num_dit_layers`-deep DiT. Runs **N times per observation**
  during the flow-matching denoising loop. Each DiT layer cross-attends to one
  matched VLM KV pair **and** to Robot CNN's high-resolution feature map.

```
╔════════════════════════════════════════════════════════════════════════════╗
║                    Top-level forward pass (inference)                      ║
╚════════════════════════════════════════════════════════════════════════════╝

   cameras     task string    state(ℝ⁷)    horizon×ℝ⁷ noise x_T
      │            │             │              │
      │            │             │              │
      ▼            ▼             │              │
  ┌──────────────────────┐       │              │
  │  STAGE A: Encoder    │       │              │
  │  (frozen SmolVLM2)   │       │              │
  │  runs ONCE           │       │              │
  │                      │       │              │
  │  output: per-layer   │       │              │
  │  K/V cache (last N   │       │              │
  │  text layers)        │       │              │
  └──────────┬───────────┘       │              │
             │ kv_cache          │              │
             │ [(K₀,V₀)..(K_{N-1},V_{N-1})]     │
             │                   │              │
             ▼                   ▼              ▼
         ┌──────────────────────────────────────────┐
         │  STAGE B: Decoder (trainable DiT)        │
         │                                          │
         │  for step in range(num_inference_steps): │
         │     v_t = DiT(x_t, t, kv_cache, state,   │
         │              robot, robot_k, robot_v)    │
         │     x_t = x_t + dt · v_t                 │
         │     t  += dt                             │
         └──────────────────┬───────────────────────┘
                            ▼
                   actions[:, :n_action_steps]
```


## Stage A — VLM encoder (frozen, one-shot)

```
   3 cameras                       task string                        
   (B,3,H,W)                       (B,) of str                        
       │                                │                             
       ▼                                ▼                             
┌──────────────────┐           ┌─────────────────────┐                
│ vision_model     │           │ tokenizer +         │                
│ (SigLIP ViT,     │           │ text_model          │                
│  frozen)         │           │ .embed_tokens       │                
│                  │           │ (frozen)            │                
│  └ connector ──► │           └──────────┬──────────┘                
│  (frozen MLP +   │                      │                           
│   pixel-shuffle) │                      ▼                           
│                  │           ┌─────────────────────┐                
│  per-camera      │           │ lang_adaptor        │                
│  patch tokens    │           │ (zero-init residual,│                
│                  │           │  trainable)         │                
└────────┬─────────┘           │  pad slots → zero   │                
         │                     └──────────┬──────────┘                
   V_tok │                          L_tok │                           
         │  (B, L_vis, h)                 │  (B, L_lang, h)           
         └────────────┬───────────────────┘                           
                      ▼                                               
              concat → vlm_seq (B, L_vlm, h)                          
                      │                                               
                      ▼                                               
    ┌────────────────────────────────────────────┐                    
    │  build RoPE cos/sin for positions 0..L_vlm │                    
    │  build causal mask + KV-pad mask           │                    
    └────────────────────────────────────────────┘                    
                      │                                               
                      ▼                                               
    ┌───────────────────────────────────────────────────────┐         
    │ SmolVLM2 text_model — manual layer-by-layer forward   │         
    │                                                       │         
    │   layer 0 ─ input_layernorm                           │         
    │     ↓     q/k/v projections                           │         
    │     ↓     apply RoPE to Q,K                           │         
    │     ↓     causal SDPA  +  o_proj  +  residual         │         
    │     ↓     post_attention_layernorm                    │         
    │     ↓     SwiGLU MLP  +  residual                     │         
    │   layer 1 …                                           │         
    │      …                                                │         
    │   layer (M-N) ──► capture (K₀,V₀)  ──┐                │         
    │   layer (M-N+1) ─► capture (K₁,V₁)  ─┤                │         
    │      …                                ├──► kv_cache   │         
    │   layer (M-1) ────► capture (K_{N-1})─┘  list of N    │         
    │                                          (K,V) pairs  │         
    └───────────────────────────────────────────────────────┘         

  M = total VLM text layers (all kept, NOT truncated)                 
  N = config.num_vlm_layers (= DiT depth = #trailing KV pairs)        
  K, V shape: (B, num_kv_heads, L_vlm, head_dim)   in bfloat16        
  K is POST-RoPE — positional rotation already applied.               
```

Also emitted by Stage A:

- `vlm_kv_pad_mask`: `(B, L_vlm)` bool — True at vision positions and at real
  language tokens; False at padded language slots. Used by the DiT's
  cross-attention to mask padded keys.


## Stage B — DiT decoder (trainable, runs N times)

### Building the DiT input sequence

```
  state(B,7) ──► state_encoder ──► state_tok (B,1,h)
  cameras   ──► RobotVisualEncoder x3 ─► robot_tok (B, 3·R, h)
  x_t (B,H,7) ─► action_in_proj + action_pos_emb ─► action_emb (B,H,h)
  prefix?  ───► action_in_proj.detach()  ─────────► prefix_emb (B,P,h)
  sink_token ─► learned 1-token parameter (B,1,h)

  DiT sequence (concatenated):

  ┌──────┬───────┬────────────┬─────────┬─────────────────┐
  │ SINK │ state │ prefix(P)? │ robot   │  action(H)      │
  │  1   │   1   │     P      │  3·R    │       H         │
  └──────┴───────┴────────────┴─────────┴─────────────────┘
                                         ▲
                                         │
                             action_start_idx = 1 + 1 + P + 3·R
                                         │
                             readout slice for v_t

  Note: latent tokens are DISABLED (num_latent_tokens=0). Robot CNN
  spatial grounding replaces the role of latent thought tokens — action
  queries directly attend to high-res Robot CNN features via cross-attn.
```

Self-attention mask: full lower-triangular causal. When `action_prefix` is
present, the **Λ-shape** modification additionally blocks noisy actions beyond
the first `lambda_mask_window` from attending to the clean prefix slots — this
forces later actions to rely on vision/language via cross-attention rather than
copying nearby clean steps.

### A single DiT layer

```
            t  ─► sinusoidal ─► time_embedder ─► t_emb (B,h)
                                       │
                                       ▼
                       ┌─────────────────────────────────┐
                       │  adaLN_modulation(t_emb)        │
                       │  → 12 vectors, chunked into:    │
                       │   (s_sa, sc_sa, g_sa,           │
                       │    s_ca, sc_ca, g_ca,           │
                       │    s_rca, sc_rca, g_rca,        │  ← NEW
                       │    s_ff, sc_ff, g_ff)           │
                       └─────────────────────────────────┘
                                       │
                                       ▼ (modulates each sublayer)
   x ─┬─► RMSNorm ─► shift/scale ─► self-attn (causal/Λ) ─┐
      │                                                   │
      │           ◄────── gate · ───────────────────────── ◄
      ├───────────────────────────────►(+)
      │
      ▼
   x ─┬─► RMSNorm ─► shift/scale ─► cross-attn(Q = x,    ─┐
      │                              K,V = kv_cache[i],   │
      │                              mask = pad_mask)     │
      │           ◄────── gate · ───────────────────────── ◄
      ├───────────────────────────────►(+)
      │
      ▼
   x ─┬─► RMSNorm ─► shift/scale ─► Robot cross-attn     ─┐  ← NEW
      │                              (Q = x,               │
      │                               K,V = robot_k/v,     │
      │                               no mask)             │
      │           ◄────── gate · ───────────────────────── ◄
      ├───────────────────────────────►(+)
      │
      ▼
   x ─┬─► RMSNorm ─► shift/scale ─► SwiGLU FFN          ─┐
      │                                                   │
      │           ◄────── gate · ───────────────────────── ◄
      └───────────────────────────────►(+)
                                       │
                                       ▼  next DiT layer
```

**NEW: Robot CNN cross-attention** — action queries directly attend to
Robot CNN's high-resolution feature map (14×14 grid @ 224×224) instead
of only VLM's coarse SigLIP patches (~729 patches @ 384×384). This
provides fine-grained spatial grounding for precise object localization
in spatial reasoning tasks (e.g., "pick up the bowl closer to the plate").

All four output projections (`sa_o`, `ca_o`, `robot_ca_o`, `ffn.down_proj`)
are **zero-init** so each layer starts as the identity transform on the
residual stream. The `adaLN_modulation` last-linear is also zero-init, so
at step 0 the model behaves exactly like a stack of residual no-ops on
top of the input embedding.

### Robot CNN K/V projections

```
  robot_tokens (B, 3·R, h)
       │
       ▼
  robot_ca_norm (RMSNorm)
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
  robot_ca_k_proj                        robot_ca_v_proj
  Linear(h → kv_heads·head_dim)          Linear(h → kv_heads·head_dim)
       │                                     │
       ▼                                     ▼
  reshape → (B, kv_heads, 3·R, head_dim)    same
       │                                     │
       └──────────► robot_k, robot_v ────────┘
                    passed to every DiT layer's Robot cross-attn
```

### DiT stack and readout

```
  dit_seq (B, L_dit, h)
       │
       ▼
  ┌──────────────────────────────────────────────┐
  │ DiTLayer 0  ── cross-attn → kv_cache[0]      │
  │            ── robot-ca  → robot_k, robot_v   │  ← NEW
  │ DiTLayer 1  ── cross-attn → kv_cache[1]      │
  │            ── robot-ca  → robot_k, robot_v   │  ← NEW
  │ DiTLayer 2  ── cross-attn → kv_cache[2]      │
  │            ── robot-ca  → robot_k, robot_v   │  ← NEW
  │     …                                        │
  │ DiTLayer N-1 ── cross-attn → kv_cache[N-1]   │
  │             ── robot-ca → robot_k, robot_v   │  ← NEW
  └──────────────────────┬───────────────────────┘
                         ▼
              slice rows [action_start : action_start + H]
                         ▼
                   final_norm (RMSNorm)
                         ▼
                action_out_proj  (zero-init Linear)
                         ▼
                v_t  ∈  ℝ^(B, H, 7)        ← velocity prediction
```


## Flow-matching denoising loop (inference)

```
  x_T ~ N(0, I)        # (B, H, 7) initial noise
  t   = 1.0
  dt  = -1 / N

  for step in range(N):                       # N = num_inference_steps
      v_t = DiT(x_t, t, kv_cache, ...)        # Stage B only
      x_t = x_t + dt · v_t
      t   = t  + dt

  return x_t[:, :n_action_steps]              # first n executed on robot
```

The VLM KV cache is computed ONCE before the loop; only the lightweight DiT
runs each step. With N=10 and 16 DiT layers this is ~10× cheaper than running
SmolVLM2 every denoising step (the interleaved variant's failure mode).


## Training loss

```
  target velocity:   u_t = noise − action
  predicted:         v_t = DiT(t · noise + (1−t) · action, t, kv_cache, …)

  main_loss = mean( pos_w · dim_w · (v_t − u_t)² )      over valid cells

  contrastive_loss (optional, training only):
      permute language slice of kv_cache across batch
      v_wrong = DiT(x_t, t, shuffled_cache, …)
      hinge   = max(0, margin − mean‖v_t − v_wrong‖²)

  total = main_loss + contrastive_weight · contrastive_loss
```

The contrastive loss perturbs only the language slot of the cached K/V — no
second VLM forward needed. This pushes the model to produce different
velocities for different task instructions ("language forcing").


## Component summary

| Component                | Trainable | Notes                                            |
|--------------------------|-----------|--------------------------------------------------|
| `vision_model`           | ❌ frozen | SmolVLM2 SigLIP ViT                              |
| `connector`              | ❌ frozen | SmolVLM2 pixel-shuffle resampler                 |
| `text_model` (all layers)| ❌ frozen | Llama-style causal LM, RoPE                      |
| `lang_adaptor`           | ✅        | Zero-init residual on language embeddings        |
| `state_encoder`          | ✅        | Linear + RMSNorm                                 |
| `robot_visual_encoder`   | ✅        | Parallel ResNet-18 per camera                    |
| `robot_ca_k_proj`        | ✅        | **NEW** Robot CNN → K projection for cross-attn  |
| `robot_ca_v_proj`        | ✅        | **NEW** Robot CNN → V projection for cross-attn  |
| `robot_ca_norm`          | ✅        | **NEW** RMSNorm before Robot K/V projection      |
| `sink_token`             | ✅        | Single learnable token, attention anchor         |
| `action_in_proj`         | ✅        | Linear: action_dim → h                           |
| `action_pos_emb`         | ✅        | Learned position embedding for action positions  |
| `time_embedder`          | ✅        | Sinusoidal → MLP → t_emb for adaLN               |
| `latent_generator`       | ✅        | **DISABLED** (num_latent_tokens=0)               |
| `dit_layers` × N         | ✅        | Self-attn + VLM cross-attn + **Robot cross-attn** + FFN + adaLN-Zero |
| `final_norm`             | ✅        | RMSNorm before readout                           |
| `action_out_proj`        | ✅        | Linear: h → action_dim (zero-init)               |


## Shape reference

| Symbol  | Meaning                                       | Default |
|---------|-----------------------------------------------|---------|
| `B`     | batch size                                    | —       |
| `h`     | hidden size (VLM text hidden_size)            | 960     |
| `H`     | action horizon (`config.horizon`)             | 64      |
| `L_vis` | total vision tokens (sum across cameras)      | —       |
| `L_lang`| language tokens after tokenization (padded)   | ≤48     |
| `L_vlm` | `L_vis + L_lang`                              | —       |
| `M`     | total SmolVLM2 text layers                    | depends |
| `N`     | DiT depth = `config.num_vlm_layers`           | 16      |
| `R`     | tokens per robot CNN view                     | 100     |
| `K`     | latent thought tokens                         | **0**   |
| `P`     | action prefix length (0 in synchronous mode)  | 0       |


## Quick comparison vs siblings

| Property                       | Interleaved      | WiltechsVLA      | **WILRO**         |
|--------------------------------|------------------|------------------|-------------------|
| VLM backbone                   | SmolVLM2-500M    | Qwen3-VL-4B      | SmolVLM2-500M     |
| VLM runs per inference         | N (≈10)          | 1                | **1**             |
| VLM sees action/state          | yes (joint attn) | no               | **no**            |
| VLM layer truncation           | yes              | no               | **no**            |
| Time conditioning              | fused into emb   | adaLN-Zero       | **adaLN-Zero**    |
| Action position in DiT seq     | n/a              | last             | **last**          |
| Contrastive loss path          | full re-forward  | KV permute       | **KV permute**    |
| Robot CNN cross-attn           | n/a (joint)      | no               | **yes (NEW)**     |
| Latent tokens                  | yes (dynamic)    | yes              | **no (disabled)** |
| GPU memory (relative)          | high             | very high        | low               |


## Robot CNN Cross-Attention Detail

### Why Robot CNN cross-attention?

The VLM's SigLIP ViT produces ~729 patches at 384×384 resolution — each patch
covers ~16×16 pixels. For spatial reasoning tasks like "pick up the bowl closer
to the plate", this coarse resolution is insufficient: the model learns to go
to the geometric midpoint between landmarks rather than the actual object position.

Robot CNN (ResNet-18) produces a 14×14 feature map at 224×224 — each token
covers ~16×16 pixels but with **much finer spatial structure** because:
1. ResNet preserves spatial locality through convolutions (vs ViT's global attention)
2. The feature map is directly projected to K/V without going through VLM's connector
3. Action queries can attend to specific spatial locations without VLM's abstraction

### Architecture change

**Before** (3 sublayers per DiT layer):
```
self-attn → VLM cross-attn → FFN
adaLN: 9 vectors (3 × 3)
```

**After** (4 sublayers per DiT layer):
```
self-attn → VLM cross-attn → Robot cross-attn → FFN
adaLN: 12 vectors (4 × 3)
```

### Parameter cost

Per DiT layer:
- `robot_ca_q`: Linear(960 → 960) = 921K params
- `robot_ca_o`: Linear(960 → 960) = 921K params
- `robot_ca_norm`: RMSNorm(960) = 960 params
- `robot_ca_k_proj` (shared): Linear(960 → 960) = 921K params
- `robot_ca_v_proj` (shared): Linear(960 → 960) = 921K params

Total: ~59M additional params across 16 DiT layers + shared projections.

### Compatibility

**NOT compatible with old checkpoints** — the adaLN modulation dimension changes
from 9×960=8640 to 12×960=11520, and new projection layers are added. Must
train from scratch or fine-tune with shape-matched weight loading.