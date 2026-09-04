# WILRO Architecture

Encoder-decoder flow matching VLA built on a frozen SmolVLM2-500M backbone with
**LoRA-adapted SigLIP ViT + text_model** for trainable vision and language.
Mixture-of-Transformers (MoT) layout — the VLM never sees state/action tokens.

- **Encoder** = the SmolVLM2 text stack (frozen base + LoRA adapters on last 8
  layers) + SigLIP ViT (frozen base + LoRA adapters on last 8 layers). Runs
  **once per observation**, captures post-RoPE K/V from trailing `num_dit_layers`
  text layers as cross-attention memory for the DiT. Also extracts intermediate
  SigLIP features for Robot CA, unless the ResNet source is selected.
- **Decoder** = a `num_dit_layers`-deep DiT. Runs **N times per observation**
  during the flow-matching denoising loop. Each DiT layer cross-attends to one
  matched VLM KV pair **and** to robot visual tokens (Robot CA).

**Robot CA has two selectable sources** (`config.robot_ca_source`), and the
choice is the single largest open question about this model — see
[Robot Cross-Attention Detail](#robot-cross-attention-detail):

| value | tokens from | trainable in that path |
|---|---|---|
| `vlm_intermediate` *(default)* | SigLIP ViT layer `robot_vlm_layer_offset`, connector-projected | 0.39M (LoRA only; base frozen) |
| `resnet` | a separate ResNet-18 truncated after layer3 | 3.03M (fully trainable) |

Two further pathways are **off by default** and exist to give the model temporal
input, which it otherwise has none of (image and state are both single-frame):

| flag | effect |
|---|---|
| `use_state_history` | stop slicing `state_tok[:, -1:]`, so all `n_obs_steps` frames enter the DiT |
| `robot_cnn_motion_tokens` | extra tokens from differencing the ResNet feature maps of two camera frames (needs `robot_ca_source="resnet"`) |

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
  │  (SmolVLM2)          │       │              │
  │  runs ONCE           │       │              │
  │                      │       │              │
  │  SigLIP ViT:         │       │              │
  │    base = frozen     │       │              │
  │    LoRA = trainable  │       │              │
  │                      │       │              │
  │  output: per-layer   │       │              │
  │  K/V cache (last N   │       │              │
  │  text layers) +      │       │              │
  │  intermediate SigLIP │       │              │
  │  features (Robot CA, │       │              │
  │  vlm_intermediate    │       │              │
  │  source only)        │       │              │
  └──────────┬───────────┘       │              │
             │ kv_cache          │              │
             │ [(K₀,V₀)..(K_{N-1},V_{N-1})]     │
             │ + robot_features  │              │
             │                   │              │
  ┌──────────┴───────────┐       │              │
  │  ResNet-18 → layer3  │       │              │  robot_ca_source
  │  (trainable, 3.03M)  │       │              │  == "resnet":
  │  OPTIONAL — replaces │       │              │  robot tokens come
  │  the SigLIP source,  │       │              │  from here instead,
  │  never runs beside   │       │              │  and the VLM's
  │  it                  │       │              │  intermediate is
  └──────────┬───────────┘       │              │  never computed
             │                   │              │
             ▼                   ▼              ▼
         ┌──────────────────────────────────────────┐
         │  STAGE B: Decoder (trainable DiT)        │
         │                                          │
         │  for step in range(num_inference_steps): │
         │     v_t = DiT(x_t, t, kv_cache, state,   │
         │              robot_k, robot_v)           │
         │     x_t = x_t + dt · v_t                 │
         │     t  += dt                             │
         └──────────────────┬───────────────────────┘
                            ▼
                   actions[:, :n_action_steps]
```


## Stage A — VLM encoder (frozen base + LoRA vision, one-shot)

```
   3 cameras                       task string                        
   (B,3,H,W)                       (B,) of str                        
       │                                │                             
       ▼                                ▼                             
┌──────────────────────────────────────────────────┐  ┌─────────────────────┐
│ vision_model (SigLIP ViT)                        │  │ tokenizer +         │
│                                                  │  │ text_model          │
│  ┌──────────────────────────────────────────┐    │  │ .embed_tokens       │
│  │ Layers 0..(M_v-k-1): frozen base only   │    │  │ (frozen)            │
│  │ Layers (M_v-k)..M_v: frozen base + LoRA │    │  └──────────┬──────────┘
│  │   q_proj = LoRALinear(frozen_W + B·A)   │    │             │           
│  │   v_proj = LoRALinear(frozen_W + B·A)   │    │             ▼           
│  └──────────────────────────────────────────┘    │  ┌─────────────────────┐
│                                                  │  │ lang tokens         │
│  └ connector ──► pixel-shuffle MLP (frozen)      │  │ (B, L_lang, h)      │
│                                                  │  └──────────┬──────────┘
│  per-camera patch tokens (B, L_cam, h)           │             │           
│  + intermediate features from layer_offset       │             ▼           
│    (for Robot CA, LoRA-adapted)                  │  zero pad slots → 0    
└────────┬─────────────────────────────────────────┘             │           
         │                                                       │           
   V_tok │ (B, L_vis, h)                                         │           
         └────────────┬──────────────────────────────────────────┘           
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
    ┌──────────────────────────────────────────────────────────┐            
    │ SmolVLM2 text_model — manual layer-by-layer forward      │            
    │ (runs under torch.no_grad() — KV caches are detached)    │            
    │                                                          │            
    │   layer 0 ─ input_layernorm                              │            
    │     ↓     q/k/v projections                              │            
    │     ↓     apply RoPE to Q,K                              │            
    │     ↓     causal SDPA  +  o_proj  +  residual            │            
    │     ↓     post_attention_layernorm                       │            
    │     ↓     SwiGLU MLP  +  residual                        │            
    │   layer 1 …                                              │            
    │      …                                                   │            
    │   layer (M-N) ──► capture (K₀,V₀)  ──┐                   │            
    │   layer (M-N+1) ─► capture (K₁,V₁)  ─┤                   │            
    │      …                                ├──► kv_cache      │            
    │   layer (M-1) ────► capture (K_{N-1})─┘  list of N       │            
    │                                          (K,V) pairs     │            
    └──────────────────────────────────────────────────────────┘            
    ┌──────────────────────────────────────────────────────────┐            
    │ lang_embeddings: final hidden state[:, L_vis:L_vis+L_lang]│           
    │ (detached, injected into DiT sequence as tokens)          │           
    └──────────────────────────────────────────────────────────┘            

  M = total VLM text layers (all kept, NOT truncated)                      
  M_v = total SigLIP ViT layers (~27 in SmolVLM2-500M)                     
  k = vision_lora_num_layers (default 5, last k layers get LoRA)           
  N = config.num_vlm_layers (= DiT depth = #trailing KV pairs)             
  K, V shape: (B, num_kv_heads, L_vlm, head_dim)   in bfloat16             
  K is POST-RoPE — positional rotation already applied.                    
```

Also emitted by Stage A:

- `vlm_kv_pad_mask`: `(B, L_vlm)` bool — True at vision positions and at real
  language tokens; False at padded language slots. Used by the DiT's
  cross-attention to mask padded keys.

- `lang_embeddings`: `(B, L_lang, h)` — VLM-processed language embeddings
  extracted from the **final hidden state** (after all VLM layers). These are
  injected into the DiT sequence as tokens so that robot and action tokens can
  **self-attend** to language directly, providing language grounding for Robot
  CA features. Detached from the VLM graph (no gradient flows back to VLM text).

- `intermediate_features`: `(B, L_vis, h)` — SigLIP ViT intermediate layer
  features (from `robot_vlm_layer_offset`, default -3 = third-to-last layer),
  projected through the connector. These are **LoRA-adapted** and naturally
  language-vision aligned through SigLIP's contrastive pretraining. Used as
  the source for Robot cross-attention K/V.
  **`None` under `robot_ca_source="resnet"`** — `output_hidden_states` and the
  second connector pass are both skipped, since nothing would read the result.

### Gradient flow in Stage A

```
  Vision LoRA gradient path (robot_ca_source = "vlm_intermediate"):
    loss → DiT → robot_tokens → intermediate_features → connector → vision_model
                                                                  │
                                                                  └── LoRA adapters
                                                                      receive gradient
                                                                      (lora_A, lora_B)

  Under robot_ca_source = "resnet" that arm is GONE — robot_tokens no longer
  touch the ViT. The vision LoRA then receives gradient only through the main
  vision tokens in the VLM KV cache, and the Robot CA arm instead trains the
  ResNet end to end:
    loss → DiT → robot_tokens → RobotVisualEncoder (stem/layer1-3/proj, 3.03M)

  Text LoRA gradient path (when enabled):
    loss → DiT cross-attn → KV cache (not detached) → text_model LoRA adapters
                                                           │
                                                           └── lora_A, lora_B
                                                               receive gradient

  connector:  frozen weights but gradient flows THROUGH to vision_model LoRA
  text_model: when text LoRA enabled → runs WITHOUT no_grad, KV not detached
              when text LoRA disabled → runs under no_grad (saves memory)
  vision_model: base weights frozen (requires_grad=False), but LoRA adapters
                are trainable Parameters that receive gradient
```


## LoRA on SigLIP ViT + text_model

### Architecture

LoRA adapters are applied to `q_proj` and `v_proj` of:
- **SigLIP ViT**: last k layers (default k=5) for trainable vision
- **text_model**: last m layers (default m=8) for trainable language

Each adapter wraps a frozen `nn.Linear`:

```
  LoRALinear:
    base: nn.Linear (frozen, requires_grad=False)
    lora_A: Parameter (rank, in_features)   — normal init σ=0.02
    lora_B: Parameter (out_features, rank)  — zero init

  forward(x):
    base_out = base(x)                                    # frozen path
    lora_out = (x @ lora_A.T) @ lora_B.T * (alpha/rank)   # adapter path
    return base_out + lora_out
```

At initialization, `lora_B = 0` so `lora_out = 0` — the adapter starts as
identity and the model behavior is unchanged. Gradients wake up `lora_B` during
the first few training steps.

### Parameter count

For rank=16, alpha=32, 5 layers, 2 projections (q_proj, v_proj) per layer:

| Layer | lora_A | lora_B | Total |
|-------|--------|--------|-------|
| Per projection | 16 × 960 = 15,360 | 960 × 16 = 15,360 | 30,720 |
| Per layer (q+v) | 30,720 | 30,720 | 61,440 |
| 5 layers | 153,600 | 153,600 | **307,200** |

~307K trainable LoRA parameters (negligible vs ~393M DiT params).

### Parameter count (vision LoRA)

For rank=16, alpha=32, 8 layers, 2 projections (q_proj, v_proj) per layer:

| Layer | lora_A | lora_B | Total |
|-------|--------|--------|-------|
| Per projection | 16 × 960 = 15,360 | 960 × 16 = 15,360 | 30,720 |
| Per layer (q+v) | 30,720 | 30,720 | 61,440 |
| 8 layers | 245,760 | 245,760 | **491,520** |

### Parameter count (text LoRA)

For rank=16, alpha=32, 8 layers, 2 projections (q_proj, v_proj) per layer:

| Layer | lora_A | lora_B | Total |
|-------|--------|--------|-------|
| Per projection | 16 × 960 = 15,360 | 960 × 16 = 15,360 | 30,720 |
| Per layer (q+v) | 30,720 | 30,720 | 61,440 |
| 8 layers | 245,760 | 245,760 | **491,520** |

**Total LoRA parameters**: ~983K (492K vision + 492K text), negligible vs ~393M DiT params.

### Why LoRA on vision?

SigLIP's contrastive pretraining provides strong language-vision alignment.
Freezing the base preserves this alignment; LoRA adapters allow the vision
encoder to adapt to **robot-domain features**:

- Gripper aperture and contact state
- Object distance and spatial relationships
- Robot-specific visual cues (wrist camera perspective)

This is more parameter-efficient than training a separate ResNet and leverages
SigLIP's pretrained alignment directly.

### Why LoRA on text?

The text model's pretrained language understanding is preserved by freezing base
weights. LoRA adapters on the last 8 layers enable adaptation to:

- **Robot-specific instructions** (LIBERO's templated task descriptions)
- **Spatial grounding** (rewritten instructions with visual descriptions)
- **Better-conditioned KV caches** for the DiT's cross-attention

When text LoRA is enabled, the text_model forward runs **without** `torch.no_grad()`
and KV caches are **not detached**, allowing gradients to flow:
`loss → DiT cross-attn → KV cache → text_model LoRA adapters`.


## Stage B — DiT decoder (trainable, runs N times)

### Building the DiT input sequence

```
  state(B,T_s,7) ► state_encoder ──► state_tok (B,S,h)
                   S = T_s if use_state_history else 1 (sliced to the last frame)
  cameras   ──► Robot CA source ──► robot_tok (B, R, h)
                vlm_intermediate: SigLIP ViT layer_offset, connector-projected
                                  → R = L_vis
                resnet:           ResNet-18→layer3, pooled per camera
                                  → R = n_cam·(robot_encoder_tokens
                                               + robot_cnn_motion_tokens)
  x_t (B,H,7) ─► action_in_proj + action_pos_emb ─► action_emb (B,H,h)
  prefix?  ───► action_in_proj.detach()  ─────────► prefix_emb (B,P,h)
  sink_token ─► learned 1-token parameter (B,1,h)
  lang_emb ───► from VLM final hidden state ──────► lang_tok (B, L_lang, h)

  DiT sequence (concatenated):

  ┌──────┬────────┬───────┬──────────────┬────────────┬───────┬────────────┐
  │ SINK │ latent │ state │ language(L)? │ prefix(P)? │ robot │ action(H)  │
  │  1   │   K    │   S   │      L       │     P      │   R   │     H      │
  └──────┴────────┴───────┴──────────────┴────────────┴───────┴────────────┘
                                                                ▲
                                                                │
                            action_start_idx = 1 + K + S + L + P + R
                                                                  │
                            readout slice for v_t

  Language tokens (from VLM's final hidden state, detached) are inserted
  AFTER state so that robot and action tokens can self-attend to language
  directly. This provides language grounding for Robot CA features —
  robot tokens learn to condition on the task instruction through
  self-attention, complementing the VLM cross-attention path.

  Robot tokens come from whichever source config.robot_ca_source names. The
  layout is identical either way — only R changes — so the two are a clean
  A/B, and switching is NOT resume-compatible (robot_ca_k/v_proj are trained
  against one source's statistics).

  Note: latent tokens are DISABLED by default (num_latent_tokens=0).
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
                       │    s_rca, sc_rca, g_rca,        │
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
   x ─┬─► RMSNorm ─► shift/scale ─► Robot cross-attn     ─┐
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

All four output projections (`sa_o`, `ca_o`, `robot_ca_o`, `ffn.down_proj`)
are **zero-init** so each layer starts as the identity transform on the
residual stream. The `adaLN_modulation` last-linear is also zero-init, so
at step 0 the model behaves exactly like a stack of residual no-ops on
top of the input embedding.

### Robot CA K/V projections (from either source)

```
  robot_tokens (B, R, h)   ── ONE of:
    vlm_intermediate: SigLIP ViT layer_offset, LoRA-adapted, connector-projected
    resnet:           ResNet-18→layer3 per camera, pooled to a token grid,
                      optionally concatenated with gate·pool(f_t − f_{t−k})
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
  reshape → (B, kv_heads, R, head_dim)      same
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
  │            ── robot-ca  → robot_k, robot_v   │
  │ DiTLayer 1  ── cross-attn → kv_cache[1]      │
  │            ── robot-ca  → robot_k, robot_v   │
  │ DiTLayer 2  ── cross-attn → kv_cache[2]      │
  │            ── robot-ca  → robot_k, robot_v   │
  │     …                                        │
  │ DiTLayer N-1 ── cross-attn → kv_cache[N-1]   │
  │             ── robot-ca → robot_k, robot_v   │
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
SmolVLM2 every denoising step.


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
| `vision_model` base      | ❌ frozen | SmolVLM2 SigLIP ViT base weights                 |
| `vision_model` LoRA      | ✅        | LoRA adapters on last k layers (q_proj, v_proj)  |
| `connector`              | ❌ frozen | SmolVLM2 pixel-shuffle resampler (gradient flows through) |
| `text_model` base        | ❌ frozen | Llama-style causal LM, RoPE                      |
| `text_model` LoRA        | ✅        | LoRA adapters on last m layers (q_proj, v_proj)  |
| `state_encoder`          | ✅        | Linear + RMSNorm                                 |
| `robot_ca_k_proj`        | ✅        | SigLIP intermediate → K projection for Robot CA  |
| `robot_ca_v_proj`        | ✅        | SigLIP intermediate → V projection for Robot CA  |
| `robot_ca_norm`          | ✅        | RMSNorm before Robot K/V projection              |
| `sink_token`             | ✅        | Single learnable token, attention anchor         |
| `action_in_proj`         | ✅        | Linear: action_dim → h                           |
| `action_pos_emb`         | ✅        | Learned position embedding for action positions  |
| `time_embedder`          | ✅        | Sinusoidal → MLP → t_emb for adaLN               |
| `latent_generator`       | ✅        | DISABLED by default (num_latent_tokens=0)        |
| `dit_layers` × N         | ✅        | Self-attn + VLM cross-attn + Robot cross-attn + FFN + adaLN-Zero |
| `final_norm`             | ✅        | RMSNorm before readout                           |
| `action_out_proj`        | ✅        | Linear: h → action_dim (zero-init)               |


## Shape reference

| Symbol  | Meaning                                       | Default |
|---------|-----------------------------------------------|---------|
| `B`     | batch size                                    | —       |
| `h`     | hidden size (VLM text hidden_size)            | 960     |
| `H`     | action horizon (`config.horizon`)             | 64      |
| `L_vis` | total vision tokens (sum across cameras)      | ~729/cam|
| `R`     | robot tokens reaching Robot CA                | `L_vis`, or `n_cam·(64+M)` under the ResNet source |
| `S`     | state tokens in the DiT sequence              | **1** (`n_obs_steps` if `use_state_history`) |
| `M`ᵣ    | ResNet motion tokens per camera               | **0** |
| `L_lang`| language tokens after tokenization (padded)   | ≤48     |
| `L_vlm` | `L_vis + L_lang`                              | —       |
| `M`     | total SmolVLM2 text layers                    | depends |
| `M_v`   | total SigLIP ViT layers                       | ~27     |
| `N`     | DiT depth = `config.num_vlm_layers`           | 16      |
| `k`     | LoRA layers (last k of SigLIP ViT)            | 8       |
| `r`     | LoRA rank                                     | 16      |
| `m`     | Text LoRA layers (last m of text_model)       | 8       |
| `K`     | latent thought tokens                         | **0**   |
| `L`     | language tokens in DiT sequence (from VLM)    | ≤48     |
| `P`     | action prefix length (0 in synchronous mode)  | 0       |


## Quick comparison vs siblings

| Property                       | Interleaved      | WiltechsVLA      | **WILRO**              |
|--------------------------------|------------------|------------------|------------------------|
| VLM backbone                   | SmolVLM2-500M    | Qwen3-VL-4B      | SmolVLM2-500M          |
| VLM runs per inference         | N (≈10)          | 1                | **1**                  |
| VLM sees action/state          | yes (joint attn) | no               | **no**                 |
| VLM layer truncation           | yes              | no               | **no**                 |
| Vision + Language adaptation   | none             | none             | **LoRA on ViT + text** |
| Time conditioning              | fused into emb   | adaLN-Zero       | **adaLN-Zero**         |
| Action position in DiT seq     | n/a              | last             | **last**               |
| Contrastive loss path          | full re-forward  | KV permute       | **KV permute**         |
| Robot CNN cross-attn           | n/a (joint)      | no               | **yes (source selectable)** |
| Image frames the VLM sees      | 1                | 1                | **1** (2 to the ResNet only, if motion is on) |
| State frames reaching the model| all T            | 1                | **1** (all T if `use_state_history`) |
| Language in DiT sequence       | yes (joint)      | no               | **yes**                |
| Latent tokens                  | yes (dynamic)    | yes              | **no (disabled)**      |
| GPU memory (relative)          | high             | very high        | low                    |


## Robot Cross-Attention Detail

### The two sources, and why the choice is still open

**`vlm_intermediate` (default).** The VLM's SigLIP ViT produces ~729 patches at
384×384. The final layer is highly semantic but loses fine spatial detail
through global self-attention; an intermediate layer (default -3) retains more
spatial structure while staying semantically rich. Projecting it through the
connector gives:

1. **LoRA-adapted features** — the trailing ViT layers carry adapters, so the
   features are robot-domain adapted
2. **Language-vision alignment** — free from SigLIP's contrastive pretraining
3. **Connector-projected** — the same pixel-shuffle resampler as the main
   vision tokens, so the representation is consistent
4. **No second encoder to train** — 0.39M trainable in this path

**`resnet`.** A separate ResNet-18 truncated after layer3 (3.03M, fully
trainable, ImageNet init), pooled to a token grid per camera. Spatial rather
than semantic: at `robot_encoder_input_size=256` the feature map is 16×16, and
the default 64 tokens give 32 native px each — parity with the VLM's merged
patches, which is the point. It is what this model ran until 2026-07-06 and
what `wiltechs_moe` still runs.

**Neither list settles it, and the swap between them was never A/B'd.** The
record, kept because each line cost an experiment:

* wilro's best spatial, 82.5 (2026-06-21, ckpt 144k), predates the swap. That
  checkpoint no longer exists, so 82.5 cannot be re-measured on the current
  harness. Every 2026-08/09 number (66–69) is post-swap.
* Between those two numbers sit a dataset change, a VLABench pretrain and an
  unrecorded inference cadence, so the gap is not cleanly attributable either.
* The sibling's ablation is the only controlled measurement of the pathway:
  `wiltechs_moe` lost **34 points** of spatial success removing its RobotCNN
  (92 → 58), same checkpoint lineage, same cadence.
* Four capacity-flavoured interventions on the current architecture (training
  budget 30k→40k, vision LoRA 8→16 layers, RFT data, +8000 steps) all landed
  within ±2 points.

**Trainable count is probably not the mechanism.** A rank-64 vision LoRA over
27 ViT layers is 5.31M — already larger than the ResNet's 3.03M — and that run
is not ahead. What differs is the *kind* of pathway: full-resolution pixels, a
16×16 spatial map, ImageNet init, and no frozen semantic tower underneath.

**Do not run both.** `resnet` replaces the source rather than adding a parallel
encoder, deliberately: a second visual pathway alongside a trained one was
measured getting gated off on the sibling (wiltechs_x wrist encoder, 1e-3 →
6.2e-4, confirmed twice). "Add it and let the model choose" reliably chooses
the incumbent.

### Temporal pathways (both off by default)

wilro has **no temporal input at all** in its default configuration — not a
short history, one frame of everything. `_encode_images` takes `imgs[:, -1]`
and `_suffix_pass` slices `state_tok[:, -1:]`, so `--n_obs_steps` has never
changed what the model sees: the extra state frames are encoded and discarded.
There is no velocity, no acceleration, and no way to tell a half-open drawer
from a closing one.

| flag | what it adds | cost |
|---|---|---|
| `use_state_history` | drops the state slice; all `n_obs_steps` frames enter the DiT | +`n_obs_steps−1` sequence positions |
| `robot_cnn_motion_tokens` | `gate · pool(f_t − f_{t−k})` from the same ResNet backbone, concatenated after the current-frame grid | +1 ResNet pass and +1 video decode per camera |

Three notes that are not obvious from the code:

* **The VLM still sees one frame under the motion path, by design.** It is
  40.8% of step time, and over 100 ms the semantics do not change — only the
  motion, which is what the ResNet extracts. Doubling the VLM pass buys the
  wrong thing at the highest price.
* **The difference is taken on feature MAPS, not on pooled tokens.** Pooling is
  followed by LayerNorm, so differencing after it is a different quantity. The
  diff gets its own, smaller grid from the same backbone at no parameter cost
  (`proj`/`norm` are per-token). The two frames are NOT stacked into a
  6-channel `conv1`; that would destroy the ImageNet stem.
* **The motion gate is zero-init**, so Stage C starts as an exact no-op over
  Stage A, and the gate's magnitude is the instrument that detects suppression.

Counter-evidence to hold onto for `use_state_history`: the leak control is
already run and clears it (the momentum shortcut sits 33× above the model's own
residual; frozen < noise < shuffled < real is a dose-response at z=4.77). But
on the sibling's task 5 three independent corruptions of the window each cut
time-to-success 195 → ~110 steps — the window can sustain a dithering loop —
and wilro pins its step cap on exactly the five tasks with that signature.
Read the result per task, not in aggregate.

### Architecture

**Per DiT layer** (4 sublayers):
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