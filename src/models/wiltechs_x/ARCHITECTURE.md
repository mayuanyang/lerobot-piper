# WiltechsX Architecture

> **Status: runs against a real Qwen3-VL-4B on a 22 GiB card; not yet
> evaluated.** The memory budget fits (13.5 GiB fixed + ~5–8 GiB activations at
> `batch_size=8–12`, `grad_accum=8`, gradient checkpointing on) and every loss
> term now starts on its chance baseline (§5). No success rate has been measured
> yet; `src/eval_wiltechs_x.py` is the entry point, and it reports the per-task
> **min** and the §6 stage-A gate rather than the average. Runs started before
> 2026-08-15 carry three fixed defects and should be discarded, not evaluated
> (§8.0).
>
> ```
> python src/models/wiltechs_x/test_components.py        # pure torch, seconds
> python src/train_wiltechs_x.py --dataset_ids ... --training_steps 30 \
>        --profile_steps 20                              # where the time goes
> python src/eval_wiltechs_x.py --checkpoint outputs/wx_a/checkpoint-5000
> ```
>
> Before training:
> ```
> python src/models/wiltechs_x/test_components.py   # pure torch, seconds
> python src/models/wiltechs_x/smoke_test.py --tiny # plumbing, no 4B download
> python src/models/wiltechs_x/smoke_test.py        # real backbone
> python src/train_wiltechs_x.py --dataset_ids ... --training_steps 200
> ```
>
> Known deviations from this document, all in `wiltechs_x_model.py`'s docstring:
> the VLM-side discrete head uses uniform binning rather than FAST (knowledge
> insulation needs *a* token objective, not specifically FAST's), and
> `flow_objective="meanflow"` raises rather than ship an unverified identity.

---

## 0. Why this exists, and what it is NOT trying to beat

Surveyed 2026-08-15. Standard LIBERO (4 suites × 10 tasks, 500 trials/suite):

| Method | Avg | Long | Note |
|---|---|---|---|
| OpenVLA | 76.5 | 53.7 | autoregressive discrete |
| OpenVLA-OFT | 97.1 | — | parallel decode + chunk + continuous |
| X-VLA-0.9B | 98.1 | — | soft-prompt, 9M trainable |
| HiF-VLA | 98.0 | 96.4 | motion-vector history |
| VLA-GSE | 98.4 | 96.8 | spectral expert split, 2.51% params |
| ElasticFlow | 98.5 | — | 1-NFE mean-flow |
| **RIPT-VLA** (RL) | **97.5** | — | OpenVLA-OFT + interactive post-training |
| **SimpleVLA-RL** (RL) | **~99** | **98.5** | OpenVLA-OFT 91→99, Long 86.5→98.5 |

Two things follow, and they set every decision below.

**(1) The SFT architecture race is decided by <1.5 points; RL post-training is
worth 8–12.** SimpleVLA-RL takes a 2025 architecture past every 2026 one. The
top-5 SFT spread is within seed noise. So WiltechsX is **not** designed to be the
best SFT policy — it is designed to be *the best substrate for RL post-training*,
which is a different objective (see §1).

**(2) LIBERO's standard setting is a memorization benchmark.** LIBERO-PRO
perturbs objects, initial states, instruction phrasing and environment; models
scoring >90% standard drop to **0.0%**. Anything above 0 on PRO is a more
interesting result than tying at 98.5 standard.

Sources are listed in §9.

---

## 1. Design objective: an RL substrate, not an SFT champion

Optimizing for "highest SFT score" and "highest post-RL score" pull in different
directions. WiltechsX optimizes for the second, which means three hard
constraints that a pure-SFT design would not accept:

| Constraint | Why | Consequence |
|---|---|---|
| **Rollout throughput is the RL budget** | RL wall-clock is dominated by env rollouts, not gradient steps | Small backbone; 1–4 NFE decoding; chunk executed in full |
| **Every task must start non-zero** | Binary-reward RL has no gradient where all K rollouts fail. RIPT-VLA lifts 4%→97%, SimpleVLA-RL 17%→91% — the bar is low, but it is not zero | Gate stage A on **per-task min**, never on the average |
| **The policy must not pre-collapse** | A flow policy annealed to near-determinism cannot explore, and RL dies silently | Keep noise/temperature knobs; do not over-train stage A |

**The average success rate is the wrong stage-A metric.** 95% average with two
tasks at 0 is strictly worse for this pipeline than 93% average with a floor of
15%: RL recovers the second, never the first.

---

## 2. What is kept from the existing repo, and what is deleted

**Kept** — these are the scarce assets, and they are why this is worth building
here rather than forking OpenVLA-OFT:

- `wilro`'s GRPO infrastructure (stage B runs on it)
- 10 Hz eval harness — LIBERO's dataset is 10 Hz and stock lerobot env is 20; RL
  done at 20 Hz does not transfer
- The two-layout-distribution finding — LIBERO's canonical 50 initial states vs
  the ~10× wider sampler distribution. This is directly LIBERO-PRO's
  initial-state axis, discovered independently here
- `task_rewrites.py` and the descriptive-object rewrites — directly PRO's
  instruction axis
- WiltechsMoE's per-token expert form (92% on the "between" task is the evidence)
- The executed-prefix / per-task validation metrics

**Deleted**, with the measurement that killed each:

| Removed | Why |
|---|---|
| Frozen VLM | Nothing in the top-10 freezes its backbone. Freezing produced the vision collapse this repo has been fighting for months |
| KV-cache capture + `capture_layers` + `vlm_capture_mode` | "Which layers should the decoder read" has no good answer — `spread` scored 39.8→13.3 and reliance migrated to the trainable CNN. Joint attention deletes the question |
| External cross-attention (`ca_q`/`ca_o`) | ~31% of decoder params spent bridging to the frozen VLM's 32×128 head geometry, plus the rotated-K/unrotated-Q asymmetry |
| Contrastive language hinge | Knowledge insulation (§3.3) is the principled version of the same goal |
| RobotCNN as a privileged side channel | Worth 34 points, which is the problem: a trainable side channel is the path of least gradient resistance. Same information, put in the shared sequence (§3.4) |

---

## 3. Architecture

### 3.0 Data flow

Token counts below are the **observed 2-camera LIBERO run** (`--expert_num_layers
18 --wrist_tokens 64`), so the arithmetic can be checked against the startup
banner. §4 gives the same layout with the config defaults.

```
   task string        base cam         wrist cam                  state history
  "pick up the…"     (B,3,H,W)      ┌──(B,3,H,W)──┐               (B, T=8, 8)
        │                 │         │             │                     │
        ▼                 ▼         ▼             ▼                     ▼
 ┌────────────┐    ┌────────────┐ ┌────────────┐ ┌───────────┐  ┌──────────────┐
 │embed_tokens│    │ Qwen3-VL   │ │ Qwen3-VL   │ │ DINOv2-S  │  │ MotionVector │
 │  FROZEN    │    │ ViT + 2×2  │ │ ViT + 2×2  │ │ + avgpool │  │   encoder    │
 │            │    │  merger    │ │  merger    │ │ TRAINABLE │  │  TRAINABLE   │
 │  48 tok    │    │  64 tok    │ │  64 tok    │ │  64 tok   │  │    8 tok     │
 └─────┬──────┘    └─────┬──────┘ └─────┬──────┘ └─────┬─────┘  └──────┬───────┘
       └─────────────────┴──────────────┴──────────────┴───────────────┘
                                        │   all projected to d = 2560 (VLM width)
                                        ▼
 PREFIX (B, 260, 2560) — bidirectional, M-RoPE, lang block key-padded
 ┌──────┬─────────┬──────────────────┬──────────────────┬────────┬────────┬──────┐
 │ user │ lang 48 │ <v> cam0 64 </v> │ <v> cam1 64 </v> │wrist 64│motion 8│ asst │
 │  3   │         │        66        │        66        │        │        │  5   │
 └──────┴─────────┴──────────────────┴──────────────────┴────────┴────────┴───┬──┘
                                                        discrete-head readout ─┘

 ┌──────────────────────────────── THE STACK ─────────────────────────────────┐
 │  VLM layer  0 … 17   prefix ONLY — plain Qwen layer, suffix does not exist  │
 │  VLM layer 18 … 35   ONE SDPA over [prefix | suffix], per-segment weights:  │
 │                        prefix half → Qwen3-VL weights + LoRA (r=32)         │
 │                        suffix half → expert weights, adaLN-Zero on t        │
 └────────────────────────────────────────────────────────────────────────────┘
                                        ▲
 SUFFIX (B, 25, 1024) — causal ─────────┘        t ──► time embedder ──► adaLN
 ┌─────────┬──────────────┬───────────────────┐        (modulation only —
 │ state 1 │ register × 8 │ x_t × 16 (horizon)│         t is NOT a token)
 └─────────┴──────────────┴───────────────────┘

        prefix_out[readout]                          suffix_out
                │                                        │
   ┌────────────▼────────────┐           ┌───────────────▼───────────────┐
   │ discrete action head    │           │ action_out_proj → v_t (B,16,7)│
   │ 16×7×256 bins, CE       │           │ progress head   → scalar      │
   │ (VLM side, insulation)  │           │ (expert side, continuous)     │
   └────────────┬────────────┘           └───────────────┬───────────────┘
                │                                        │
                └───────────── stop-grad ────────────────┘
                        (the K/V cache is detached)
```

### 3.0.1 Input composition

Every input is a token in **one shared sequence**. There is no side channel and
no cross-attention module — that is the whole architectural claim (§2, §3.4).

| Segment | Source key | Encoder | Tokens | Grad path |
|---|---|---|---|---|
| header | `<\|im_start\|>user\n` | `embed_tokens` | 3 | frozen |
| **lang** | `task` / `task_description` | `embed_tokens`, `padding="max_length"` | `lang_max_len` = **48** | frozen embed; LoRA downstream |
| **vision** | every key in `cameras_for_vlm` | Qwen3-VL ViT → 2×2 spatial merge | `(grid_h/2)·(grid_w/2)` per camera = **64** at a 16×16 patch grid, **+2** for the `<\|vision_start\|>`/`<\|vision_end\|>` brackets | frozen unless `lora_on_vision_tower` |
| **wrist** | keys matching `WRIST_HINTS` (`image2`, `wrist`, `gripper`, `eye_in_hand`, `hand`) | DINOv2 → drop CLS → adaptive-avg-pool to a √N×√N grid → Linear | `wrist_tokens` = **256** default | trainable (backbone + proj) |
| **motion** | `observation.state` history, `T = motion_history_len` | `[h, Δh]` → 256-d MLP → mix → Linear | `motion_vector_tokens` = **8** | trainable |
| tail | `<\|im_end\|>\n<\|im_start\|>assistant\n` | `embed_tokens` | 5 | frozen; **last position is the discrete-head readout** |
| **state** | `observation.state` (last step) | Linear + RMSNorm | 1 | trainable (expert) |
| **register** | learned parameter | — | `num_register_tokens` = **8**, std=0.02 init | trainable (expert) |
| **x_t** | noisy action chunk | `action_in_proj` + `action_pos_emb` | `horizon` = **16** | trainable (expert) |
| *t (and step size d)* | flow time | sinusoid → MLP → per-layer adaLN | **not a token** | trainable (expert) |

Three things this table is meant to make un-missable:

- **The wrist camera is encoded twice.** `train_wiltechs_x.py` passes *all*
  cameras to `cameras_for_vlm`, and then selects the wrist-like ones again for
  the DINO path. That is deliberate — the two towers carry different features —
  but it means the wrist frame costs `64 + wrist_tokens` prefix positions, and
  the DINO path only earns them if `wrist_tokens > 64` (§3.4).
- **`motion` is the only path that sees more than one timestep.** There is no
  frame stacking anywhere; history enters as 8 low-dimensional tokens.
- **The prefix is a function of the observation ONLY.** No suffix token, no
  noise level, and no flow time appears in it. §4 explains why that is
  load-bearing rather than incidental.

### 3.1 Backbone: Qwen3-VL, LoRA, unfrozen

Same backbone family as `wiltechs_vla` so the ViT/processor/M-RoPE plumbing
carries over. Two changes:

- **LoRA instead of frozen.** X-VLA is SOTA-level at 0.9B total with 9M
  trainable; VLA-GSE reaches 98.4 updating 2.51% of parameters. Capacity is not
  the LIBERO bottleneck — a *trainable* backbone is. LoRA also caps how far the
  representation can drift, which is half of what knowledge insulation buys.
- **Prefer the smallest Qwen3-VL that clears stage A.** Backbone size sets
  rollout throughput, which sets the stage-B budget. Verify which sizes exist
  before pinning `vlm_model_id`.

### 3.2 Bidirectional prefix — trained that way

The prefix (vision + language + motion) attends bidirectionally; only the action
suffix is causal. This is what π0 / PaliGemma / X-VLA do.

This is the same mask change added to `wiltechs_vla` as
`--bidirectional_prompt`, but the two are not the same experiment. There it is
applied zero-shot to frozen weights, which is out of distribution for a causally
pretrained model with nothing able to adapt. Here LoRA trains under the mask, so
the adaptation exists. **If bidirectional attention is worth anything, it is
worth it here and not there.**

### 3.3 Knowledge insulation (replaces the contrastive hinge)

Two couplings, deliberately asymmetric:

- **stop-grad on the action expert → VLM path.** Flow-matching gradients
  flowing into the VLM degrade language grounding. The expert reads the VLM;
  it does not rewrite it.
- **A discrete FAST action-token head on the VLM side**, trained with
  cross-entropy. The VLM still learns the task — through a token objective it
  was pretrained for, not through a regression objective it was not.

This is the principled form of what `contrastive_loss_weight` /
`contrastive_hard_negatives` were approximating: *keep the language pathway
alive under action supervision*. The hinge attacked the symptom (the model
ignores language) with a hand-built negative; KI removes the cause.

> **First ablation to run in stage A.** The KI result comes from large
> cross-embodiment corpora. On LIBERO's 50 demos/task, LoRA's own rank
> constraint may already supply the insulation, making the FAST head dead
> weight. Measure it; do not assume it transfers.

### 3.4 Precision path: DINO features in the shared sequence

The 34-point RobotCNN result says the missing ingredient is high-frequency
detail near the gripper. Two changes to how it is supplied:

- **A self-supervised ViT (DINOv2/v3) instead of a from-scratch CNN.**
  Self-supervised features carry far better dense spatial correspondence than
  SigLIP-style contrastive ones; OpenVLA fuses SigLIP+DINOv2 for exactly this.
- **The budget test is TOKENS PER CAMERA, not pixels per token.** `wrist_tokens`
  must exceed the VLM's own per-camera count — `(grid_h/2)·(grid_w/2)`, 64 at a
  16×16 patch grid — or this path resolves nothing the prefix does not already
  have and is pure parameter cost. `wrist_input_size` does **not** help: it
  upsamples before an adaptive pool that discards the extra patches. The startup
  banner prints the verdict (`FINER` / `IDENTICAL` / `COARSER`) once the
  processor has revealed the real grid.
- **Its tokens go in the shared prefix, not into a side channel.** The observed
  "reliance migrated to the RobotCNN" is a *consequence* of privileged-side-channel
  placement — gradient descent takes the trainable shortcut. In the shared
  sequence those tokens compete in the same attention as everything else and
  are subject to the same language conditioning.

### 3.5 Long horizon: motion vectors + progress

The only suite with headroom. Two cheap additions:

- **Motion vectors as hindsight.** HiF-VLA encodes history as low-dimensional
  motion vectors rather than stacked frames — currently the Long SOTA mechanism
  (96.4) at 58% lower latency than frame stacking. Frame stacking also invites
  causal confusion, which is precisely the failure LIBERO-Long punishes.
- **A predicted progress scalar** (normalized time-to-completion), auxiliary
  regression. Cheap, gives the policy an explicit "which stage am I in", and it
  makes stage-B credit assignment tractable — a binary terminal reward on a
  10-stage task is the hardest credit-assignment problem in the pipeline.

### 3.6 Decoding: few-step flow, chunk executed in full

- Chunk `horizon=16`, execute `n_action_steps=8` — the OFT setting. Not 32.
- Train toward few-step inference (mean-flow / shortcut objective) so inference
  runs at 1–4 NFE. **This is for the RL rollout budget, not for a headline Hz
  number.** At 5 Euler steps, stage B costs 5× per env step.

### 3.7 Component internals

#### One `JointExpertLayer` — the whole architecture in one box

Head geometry is **shared** with the VLM (q 32×128, k/v 8×128, GQA 4:1) because
Q and K meet in a single dot product. The expert's *width* is free: it projects
from `d_exp` into that head space and back. That is what removes the
cross-attention bridge `wiltechs_vla` spent 31% of its decoder on.

```
 ══ PREFIX half — VLM weights + LoRA ══   ══ SUFFIX half — expert weights ══
 h_p                                      h_s              t_emb
  │                                        │                 │
  ▼                                        ▼                 ▼
 vlm.input_layernorm                     attn_norm          ada  (SiLU→r=64→6d)
  │                                        │                 │
  │                                        ▼        shift_a scale_a gate_a
  │                                   modulate(shift_a, scale_a)  shift_f
  ▼                                        │                      scale_f gate_f
 vlm.self_attn.q/k/v_proj                q/k/v_proj (expert, no bias)
 (LoRALinear r=32 a=64)                    │
  │  pq (32h) pk,pv (8h)                   │  sq (32h) sk,sv (8h)
  └──────────────────┬─────────────────────┘
                     ▼
        q = cat[pq,sq]   k = cat[pk,sk]   v = cat[pv,sv]        ← ONE sequence
                     │
              M-RoPE, then ONE scaled_dot_product_attention
              mask: prefix↔prefix full · suffix→prefix full
                    suffix→suffix causal · prefix→suffix BLOCKED
                     │
          ┌──────────┴──────────┐
     p_attn[:, :L_p]       s_attn[:, L_p:]
          │                     │
     ┌────▼─────┐               │
     │ .detach()│ ← knowledge insulation: the expert READS the VLM,
     └────┬─────┘   it never rewrites it (§3.3)
          ▼                     ▼
 h_p += vlm.o_proj(p_attn)   h_s += gate_a · o_proj(s_attn)
 h_p += vlm.mlp(post_ln)     h_s += gate_f · SwiGLU(modulate(shift_f, scale_f))
```

Per layer at `d_exp=1024`, `ada_rank=64`, `ffn=d_exp` — this reconciles with the
`expert 253.8M params` line the trainer prints (18 × 14.1M):

| Block | Shapes | Params |
|---|---|---|
| `q_proj` | 1024 → 32·128 | 4.19M |
| `k_proj`, `v_proj` | 1024 → 8·128, ×2 | 2.10M |
| `o_proj` | 4096 → 1024 | 4.19M |
| `ffn` SwiGLU | gate/up/down, 3 × 1024² | 3.15M |
| `ada` low rank | 1024→64→6·1024 | 0.47M |
| `attn_norm`, `ffn_norm` | RMSNorm ×2 | 0.002M |
| **per layer** | | **14.10M** |

`ada` is factorised for a measured reason: a plain `Linear(d, 6d)` is 6.29M per
layer, 226M over 36 — 32% of the expert, spent on six vectors per layer. It was
the difference between OOM and fitting on a 22 GiB card.

**Layers 0…`first_joint_layer`-1** have no expert and run `_prefix_only_layer`:
a plain Qwen block (LoRA still active on q/k/v/o) over the prefix alone, with
`bidirectional_prefix` deciding whether the causal triangle is applied.

#### Leaf modules

| Module | Stack | Out | Params |
|---|---|---|---|
| `LoRALinear` | frozen `base(x)` + `B(A(dropout(x))) · α/r`, **B zero-init** so it is an exact identity at step 0 | — | 23.6M total (4 projections × 36 layers) |
| `WristTokenizer` | DINOv2 → drop CLS → `(B,D,s,s)` → `adaptive_avg_pool2d(√N)` → `Linear(384→2560)` → RMSNorm | (B, `wrist_tokens`, 2560) | 23.0M |
| `MotionVectorEncoder` | `cat[h, Δh]` → `Linear(2D→256)` → SiLU → `Linear(256→256)` → flatten → `Linear(256·T → n_tok·256)` → `Linear(256→2560)` → RMSNorm | (B, 8, 2560) | 4.9M |
| `DiscreteActionHead` | **RMSNorm first** → `Linear(2560→1280)` → SiLU → `Linear(1280→H·A·256)`, last weight init `std=1e-3` | (B, 16, 7, 256) | 40.0M |
| `ProgressHead` | RMSNorm → `Linear(1024→256)` → SiLU → `Linear(256→1)` → sigmoid, reads `suffix_out[:, 0]` (the state token) | (B,) | 0.26M |
| suffix embeds | `state_encoder` = Linear+RMSNorm · `action_in_proj` = Linear · `action_pos_emb` (H, d) std 0.02 · `register_tokens` (R, d) std 0.02 | — | small |
| `time_embedder` | sinusoid(t) [‖ sinusoid(d) under shortcut] → `Linear→SiLU→Linear` | (B, d) | small |
| velocity head | `final_norm` → `action_out_proj`, **weight and bias zero-init** | (B, 16, 7) | small |

Three of those initialisations are load-bearing, not style:

- **`action_out_proj` zero** makes `v(0) = 0` exactly, which is why the `flow`
  loss has a closed-form value at step 0 and why §5's init column can be checked
  against a log.
- **`DiscreteActionHead`'s leading RMSNorm** exists because without it the head
  reads a 36-layer Qwen hidden state whose norm is large, and a default-init
  `Linear` turned that into **CE 25.5 at init against ln(256) = 5.545** — 20 nats
  of pure initialisation error on the only gradient path into the VLM. The final
  weight is `std=1e-3` rather than zero for the same reason in reverse: exact
  zero gives `dL/d(input) = 0`, so nothing would reach the VLM on step 1.
- **`register_tokens` std 0.02, not zero.** Outside the action slice the suffix
  has no positional embedding, so identical registers would be
  indistinguishable to self-attention and could never differentiate.

#### Two execution paths, and why they are the same computation

```
 forward()                       _run_prefix()  →  forward_cached() × N
 ─────────                       ────────────────────────────────────
 joint attention over            phase 1: prefix alone, per-layer K/V cached
 [prefix | suffix] in one go     phase 2: suffix vs that cache, once per
 REFERENCE ONLY                           denoising step
```

This is not an inference-time approximation. `prefix→suffix` is masked, so the
prefix's per-layer K/V are *the same tensors* the joint forward produces —
running it once and reusing it is exact. `test_components.py` asserts
`forward_cached == forward`; if that ever fails, the fast path is wrong.

Both training and sampling take the cached path, deliberately: a separate
training path is where train/inference skew comes from. It is also what makes
the shortcut consistency term affordable — three extra *suffix* passes on a
fraction of the batch, with the VLM not re-run.

Gradient checkpointing wraps **both** halves (36 prefix layers + `n_exp` expert
layers). It used to wrap only the expert, which is the cheap half: at B=8 the
prefix stores roughly 9 GiB of activations against 8 GiB of bf16 weights.

---

## 4. Sequence layout and masks

```
   ┌──────────────────────── PREFIX (bidirectional) ───────────────────────┐┌───── SUFFIX (causal) ─────┐
 seg   hdr  lang₀..₄₇  <v>vis₀..₆₃</v> ×cams  wrist₀..W  mv₀..₇  tail │ state  reg₀..₇  x_t⁰..x_t^{H-1}
 wts   ├──────────── Qwen3-VL layer i + LoRA (r=32) ──────────────────┤ ├──── expert layer i, adaLN(t) ────┤
 grad  └──── LoRA + wrist + motion, via the discrete CE head ─────────┘ └── full; stop-grad into the prefix ┘
```

**Length arithmetic** — `L_prefix = 3 + lang + Σ_cams(vis_c + 2) + wrist + motion + 5`:

| Config | lang | vision | wrist | motion | `L_prefix` | `L_suffix` | total |
|---|---|---|---|---|---|---|---|
| defaults, 2 cams, `wrist_tokens=256` | 48 | 2×66 | 256 | 8 | **452** | 25 | 477 |
| the run in the banner (`wrist_tokens=64`) | 48 | 2×66 | 64 | 8 | **260** | 25 | 285 |
| `--lang_max_len 24`, 1 wrist cam @ 256 | 24 | 2×66 | 256 | 8 | 428 | 25 | 453 |

`L_suffix = 1 + num_register_tokens + horizon`. It is paid **in every joint
layer**, which is why `horizon` is 16 and not WiltechsVLA's 64.

- Prefix ↔ prefix: **full** (both directions, key padding enforced)
- Suffix → prefix: **full**
- Suffix → suffix: **causal**
- Prefix → suffix: **masked out**. The prefix must not depend on the noise
  level, or it would have to be recomputed at every denoising step — this is
  what keeps the VLM a *once-per-chunk* cost during rollouts.

One SDPA per layer over the concatenated sequence, with per-segment weights.
That last property is what makes this a Mixture-of-Transformers rather than an
encoder-decoder: there is no separate cross-attention module, and no question of
which layer's KV the decoder reads.

---

## 5. Losses

`total` printed by the trainer is the weighted sum of exactly these terms, so it
reconciles arithmetically — useful for confirming a term is actually on.

| Term (log key) | Side | Weight | Value at init | Purpose |
|---|---|---|---|---|
| `flow` — velocity MSE | expert | `action_loss_weight` 1.0 | **2.0** on normalized data (`action_out_proj` is zero-init, so `v=0` and the loss is `E‖noise−a‖²`) | main objective |
| `shortcut` — self-consistency | expert | 1.0, on `shortcut_consistency_frac` = 0.25 of the batch | **≈0** — an untrained near-constant field satisfies the identity trivially | makes 1–4 NFE inference valid rather than an under-integrated Euler solve |
| `gripper` — BCE, class-balanced | expert | 0.05 | ln2 = **0.693** | the gripper dim sits in the majority-class optimum otherwise |
| `discrete` — binned action CE | VLM | `fast_token_loss_weight` 0.5 | ln(256) = **5.545** | knowledge insulation |
| `progress` — regression | expert | 0.1 | 1/12 = **0.083** (variance of uniform progress) | long-horizon phase signal |

The init column is the whole point of the table: at step 20 every term should be
sitting on its chance baseline, and any term that is *not* is miswired. Two bugs
were found by exactly this check and are fixed as of 2026-08-15 — if you are
reading a log from before it, both are visible in the first line:

- **`flow` started at 8.5, not 2.0.** Two independent causes multiplying:
  `cells.sum()` was taken on a `(B, H, 1)` tensor while the numerator summed
  `(B, H, A)`, so the mean was short by a factor of `action_dim` (7); and the
  trainer never applied the preprocessor, so `E[a²]` was the raw 0.21 rather
  than 1.0. `7 × (1 + 0.21) = 8.48`, against 8.42 and 8.59 measured on two runs.
- The A-factor was **not** cosmetic. `shortcut`, `gripper` and `progress` are
  added to the same `main` and share the expert, so every one of them ran at
  1/7 of its stated weight. A `shortcut` term pinned near 0.003 while `flow`
  is at 1.5 is the signature.

`shortcut ≈ 0` is ambiguous, not good: a collapsed constant field satisfies the
identity as well as a correct one does. It should **rise** as `flow` falls, then
settle. Flat-zero alongside a flat `flow` is a collapse, not convergence.

No contrastive term. If language-following regresses, the diagnosis goes to the
discrete head's weight, not to a new hinge.

---

## 6. Training stages

| Stage | What | Gate to advance |
|---|---|---|
| **A. SFT** | 4 suites co-trained + instruction paraphrase augmentation | avg ≥93% **and per-task min >5%** |
| **B. RL** | SimpleVLA-RL recipe on `wilro`'s GRPO | Long ≥97%, avg ≥98.5 |
| **C. PRO** | layout randomization + paraphrase augmentation; evaluate on LIBERO-PRO | any score >0 |

**Stage B specifics** (the part that actually produces the number):

- Binary terminal success reward, K rollouts per initial state
- Leave-one-out / group-relative advantage — no learned critic
- **Dynamic sampling**: drop initial states where all K rollouts agree. Their
  advantage is identically zero, so they contribute nothing but rollout cost.
  This is the single highest-leverage efficiency item; check whether `wilro`
  already has it before anything else
- Run at `control_freq=10`
- Do not judge per-task success at 3–5 iterations — the T8 U-shape (50→22→44)
  in `wilro` was transient negative transfer, not reward hacking

---

## 7. What this is expected to achieve, honestly

**Standard LIBERO 98.5–99 is a tie, not a win**, and ~90% of it comes from stage
B rather than from anything in §3. If the only goal is that number, the shortest
path is not this module — it is stage B applied to existing OpenVLA-OFT weights,
which needs no new architecture at all. That comparison should be run before
committing to a full stage A here.

**LIBERO-PRO is where the design could earn its keep, and there is no
confidence interval to offer** — everyone is at 0.0%. The components with a
plausible claim on it are §3.3 (real language pathway), §3.4 (spatial features
that transfer across layouts), and stage C's augmentation. All three are
hypotheses.

---

## 8. Known risks

0. **Checkpoints from before 2026-08-15 are not comparable.** Three defects,
   all fixed, all of which changed what was optimized rather than only what was
   printed:
   - the trainer never applied the preprocessor, so state and action reached
     the loss raw. On this dataset's stats the three rotation dims carry
     std 0.04–0.08 against the gripper's 1.0, so rotation took **0.79%** of the
     flow loss where MEAN_STD gives it 42.9% — grasp pose was barely trained.
     Fixed in `train_wiltechs_x.py` (`prepare()`), which also has to restore
     `progress`: LeRobot's `transition_to_batch` keeps only
     `observation.*` / `action` / `*_is_pad` / `task` / `index` / `task_index`.
   - the flow loss divided by `cells.sum()` on a `(B, H, 1)` tensor, making it
     `action_dim`× too large and every term sharing the expert
     correspondingly under-weighted (§5).
   - the LR schedule stepped per micro-batch, stretching warmup by
     `grad_accum` (fixed earlier, in `51c616e`).
1. **KI may be unnecessary at this data scale** (§3.3). First ablation.
2. **Motion vectors may leak the demonstrator's action**, reintroducing causal
   confusion through the back door — the exact failure frame stacking has. Check
   that a motion-vector-only model does *not* score above chance.
3. **Few-step flow trades sample quality for speed**, and the trade is only
   worth it if stage B actually runs. If stage B slips, this is a pure loss.
4. **The prefix must stay noise-independent** (§4). It is easy to break this
   accidentally when adding a feature, and the symptom is a silent 5× slowdown
   in rollouts, not an error.
5. **`dinov3` model ids on HF were not verified** when this was written. The
   config defaults to a DINOv2 id; confirm before switching.

---

## 9. Sources

- LIBERO-PRO — arXiv:2510.03827
- SimpleVLA-RL — arXiv:2509.09674
- RIPT-VLA — ariostgx.github.io/ript_vla
- VLA-GSE — arXiv:2605.06175
- X-VLA — arXiv:2510.10274
- ElasticFlow — arXiv:2605.08799
- HiF-VLA — arXiv:2512.09928
- OpenVLA-OFT — arXiv:2502.19645
