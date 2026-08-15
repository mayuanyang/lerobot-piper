# WiltechsX Architecture

> **Status: implemented, never run against a real backbone.** Every method is
> written; 27 component tests pass on pure torch. Nothing has touched a real
> Qwen3-VL, a real dataset, or a GPU, so treat the first Colab run as debugging,
> not as an experiment.
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

```
                     ┌──────────── ONE joint attention per layer ────────────┐
 images ──► Qwen3-VL ViT ──┐
 wrist  ──► DINOv2/v3 ─────┤
 language ► embed_tokens ──┤  PREFIX (bidirectional, VLM weights + LoRA)
 motion  ► MV encoder ─────┘
                            ├─────────────────────────────────────────────────┤
 state   ► state proj ─────┐
 x_t     ► action proj ────┤  SUFFIX (causal, EXPERT weights, full gradient)
 t       ► time embed  ────┘
                            └─────────────────────────────────────────────────┘
                                        │              │
                       FAST token head ─┘              └─ velocity head + progress
                       (VLM side, discrete)               (expert side, continuous)
                                ▲                                │
                                └──── stop-grad ─────────────────┘
```

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

---

## 4. Sequence layout and masks

```
              ┌──────────────────── PREFIX (bidirectional) ─────────────────────┐┌── SUFFIX (causal) ──┐
 position:    lang₀..lang_L  vis₀..vis_N  wrist₀..wrist_W  mv₀..mv_M  │ state  x_t⁰ .. x_t^{H-1}
 weights:     ├──────────── Qwen3-VL layer i + LoRA ──────────────────┤ ├── expert layer i ────┤
 gradient:    └──── LoRA only, + FAST CE head ──────────────────────┘ └── full, stop-grad in ─┘
```

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

| Term | Side | Default weight | Purpose |
|---|---|---|---|
| Flow / mean-flow velocity | expert | 1.0 | main objective |
| Gripper BCE (class-balanced) | expert | 0.05 | ported from `wiltechs_vla`; the gripper dim sits in the majority-class optimum otherwise |
| FAST token CE | VLM | 0.5 | knowledge insulation |
| Progress regression | expert | 0.1 | long-horizon phase signal |

No contrastive term. If language-following regresses, the diagnosis goes to the
FAST head weight, not to a new hinge.

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
