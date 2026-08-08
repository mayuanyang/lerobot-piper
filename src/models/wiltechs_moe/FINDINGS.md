# WiltechsMoE — empirical notes

Companion to `ARCHITECTURE.md`, which describes the structure. This file records
what has actually been **measured**, what it means, and what is still open.
Every claim is tagged:

- **[measured]** — a number produced by a script in this repo
- **[inferred]** — a mechanism argument, not verified by a controlled test
- **[wrong]** — believed at some point, disproved; kept so it is not redone

---

## 1. Status

**libero_spatial task 0** — "pick up the black bowl between the plate and the
ramekin and place it on the plate" — was the driving failure. It now scores
**92% (46/50)** at checkpoint 18000. **[measured]**

The four failures are init states 13, 14, 26, 27. Videos show grasp-precision
failures, not wrong-bowl selection. Geometry agrees: those are the layouts where
the target sits nearest the ramekin end of the segment, so clearance is tightest.
**[measured]**

| metric | FAILED (n=4) | succeeded (n=46) | Welch t |
|---|---|---|---|
| position along segment `t` | 0.460 ± 0.039 | 0.525 ± 0.041 | −3.2 |
| plate-selector margin | 1.90 ± 0.12 | 2.19 ± 0.17 | −4.5 |
| perpendicular offset | 0.0127 ± 0.0084 | 0.0095 ± 0.0064 | 0.7 |

Distance **off** the line does not separate the failures; position **along** it
does. Reproduce with `src/libero_layout_geometry.py`.

Config that produced it:

```
--use_descriptive_objects
--vision_input_size 512 --vision_hires_cameras observation.images.image
--robot_cnn_wrist_only
--contrastive_loss_weight 0.1 --contrastive_hard_negatives
--num_experts 4 --expert_num_layers 9 --dit_hidden_size 1280
--batch_size 60 --dataset_id lerobot/libero
```

Four changes landed together, so the 92% is **not attributed to any one of
them**. The pre-change baseline on a corrected eval harness was never run.

---

## 2. The two distributions — read this before comparing any eval number

lerobot's `LiberoEnv.reset()` calls `set_init_state()` **before** `_env.reset()`.
robosuite's reset re-samples the BDDL placement initializer and discards the
state. LIBERO's own eval loop is the other order. **[measured]**, per-object xyz
spread over 12 layouts, metres:

| object | canonical states | stock lerobot | fixed order | fixed + same id |
|---|---|---|---|---|
| akita_black_bowl_1 | 0.0089 | **0.0876** | 0.0143 | 0.0000 |
| plate_1 | 0.0080 | **0.0503** | 0.0076 | 0.0000 |
| ramekin_1 | 0.0094 | **0.0284** | 0.0092 | 0.0000 |

So there are two distributions and they are not interchangeable:

- **canonical 50** — what the demos were recorded on, what LIBERO's official
  harness and the OpenVLA-derived scripts most papers use evaluate on, ±1.5 cm
  of jitter. Numbers here are comparable to published results.
- **placement sampler** — what stock lerobot serves, ~10× wider. Nothing in the
  literature reports on it, and training has never seen it.

Consequences:

- Every eval run before `src/libero_env_fixed.py` measured the sampler
  distribution. Those numbers cannot be compared with the ones after.
- A layout the eyes remember as "the bowl is way off the line" is a sampler
  layout; it is not among the canonical 50.
- GRPO groups were broken: `TaskEnvGroup.reset_group()` sets one
  `_init_state_id` for all G members but a different seed each, on the premise
  that the group shares an initial state. Under the stock ordering the seed
  drives the sampler, so members got different layouts and the group-mean
  baseline absorbed layout difficulty instead of action quality. **[inferred]**
  from the ordering; the spread above is the measured part.

Use `patch_lerobot_libero()` for every eval and rollout from now on. A/B with
`LIBERO_FIXED_INIT=0/1`; `LIBERO_INIT_FROM_SEED=1` for eval only (it would
re-break GRPO groups).

---

## 3. Chain of causes on the "between" task

In the order they were found and fixed. Each was real; none alone was enough.

1. **Vision KV was language-blind.** Images sat before the instruction, so under
   the VLM's causal mask no vision position could attend to the text. Fixed by
   `text_first=True`. **[inferred]** mechanism, **[measured]** cost +10.6% step
   time (predicted +25–40%, which was wrong).
2. **The CoT rewrite was truncated.** `_lang_max_len` was 48; the rewrites run to
   ~105 tokens, so the selector and the whole `Action:` clause never reached the
   model. Raised to 128 with a startup report. **[measured]**
3. **The relation was used as a coordinate.** After text-first the arm stopped
   aiming at the midpoint and started landing on bare table *on* the
   ramekin→plate line. "Between" pins a line; the target's position along it
   changes per episode. Rewritten as a selector over two candidates.
4. **8×8 vision grid.** One merged token covers 32×32 input px. At 64 tokens the
   Qwen probe failed its own consistency control, calling one bowl both nearest
   to and farthest from the plate. `--vision_input_size 512` → 256 tok/frame for
   the third-person camera only. **[measured]**
5. **Vocabulary was inconsistent, not merely wrong.** The ramekin was "a small
   shallow empty cup" in one task, "a small round silver container" in two more,
   and bare "ramekin" in six. All ten libero_spatial tasks now build from one set
   of module constants; relations stay per-task.
6. **`contrastive_loss_weight` was 0.** The one term that forces the output to
   depend on the instruction was off for the whole 25k→35k run, which is why
   raising the resolution in that run showed no effect — the two changes were
   confounded. Restored to 0.1.
7. **The contrastive branch fed the router the correct language.**
   `_run_moe_dit` got the shuffled KV cache but the original `vlm_hidden`, so
   `v_t` and `v_wrong` were mixed by identical expert weights and the gradient
   never reached the router. It also crashed outright under
   `--contrastive_hard_negatives`, whose permutation is not a bijection and can
   change the padded language length (406 vs 416). **[measured]** — it crashed.

Anchor strength per task, measured on init state 0, distance ratio between the
two bowls under each task's own named anchor. Only "between" is weak, which is
why only it failed while the suite sat around 87%:

```
t0 between the plate and the ramekin  ramekin  1.20x   <- outlier
t1 next to the ramekin                ramekin  3.61x
t2 from table center                  centre   4.02x
t3 on the cookie box                  cookies    inf   (stacked)
t4 in the top drawer                  cabinet 13.28x
t5 on the ramekin                     ramekin 20.82x   (stacked)
t6 next to the cookie box             cookies  3.33x
t7 on the stove                       stove    2.91x
t8 next to the plate                  plate    2.19x
t9 on the wooden cabinet              cabinet 17.83x   (stacked)
```

---

## 4. Reading the training diagnostics

`_log_gradient_analysis` prints every `progress_update_freq` steps.

**`Thought gates`** — the model's own "how much do I use the thought tokens"
knob, four scalars initialised to +0.100. Far lower variance than the gradient
RMS above it, which swings 2–3× between adjacent readings. Observed trajectory:

| step | L0 ca / ffn | L1 ca / ffn | thought/action RMS |
|---|---|---|---|
| 400 | 0.099 / 0.101 | 0.101 / 0.101 | 0.98 |
| 3000 | 0.125 / 0.092 | 0.119 / 0.088 | 2.29 |
| 15600 | 0.168 / 0.096 | 0.114 / 0.084 | 3.89 |
| 18000 | 0.171 / 0.093 | 0.113 / 0.066 | 3.86 |

L0 climbs and holds; L1 is being switched off. **The second Q-Former layer is
not earning its keep** — try `--thought_qformer_layers 1` next run. **[measured]**

A low `Thought QFormer` gradient does *not* mean the module is inert: the
expert's first RMSNorm renormalises per token, so magnitude is rescaled away.
The gates are the honest signal. Pinned at 0.100 after thousands of steps is the
failure state, and that is what the pre-text-first runs showed.

**The x-attn null is NOT x1.0 — always compare against step 0.** At random
init the shares should be proportional to the position counts, i.e. `x1.00`
each. Measured on an untrained WiltechsVLA (step 0, `[96 lang + 320 vis]`):

```
vision= 89.5% (x1.16)   language= 10.5% (x0.45)
L8= 6.3%  L17= 6.3%  L26= 7.0%  L35=22.3%      (language %, shallow→deep)
```

The frozen VLM's text-position K vectors have smaller norm than its vision-token
K, so random queries score them lower — and the gap widens with depth. The null
is a property of the backbone, not of the policy. **[measured]**

Read every x-attn number against it, not against x1.0:

| | language | vs positions | **vs null** |
|---|----------|--------------|---------|
| step 0 (null) | 10.5% | x0.45 | — |
| MoE @18k (92%) | 76.7% | x3.32 | **7.4×** |
| VLA @8.6k, `dit_hidden=640` | 95.6% | x4.14 | **9.2×** |

The same applies per depth: the MoE's `L8=44.2%` is a **7×** move off a 6.3%
null, while its `L35=91.4%` is only **4×** off a 22.3% null. So "E0 is the
vision expert" is mostly the null showing through — in *relative* terms E0
shifts toward language harder than E3 does, not less.

The self-attn regions have a null too: `robot=30.7% latent=16.0% action=49.3%`
at step 0. A later `robot=36.4%` is therefore a small move, not the RobotCNN
taking over; what actually moves is `latent`, which halves.

**Trajectory** (VLA at `dit_hidden=1280`, matched config): language attention is
learned **deep-first and propagates shallow-ward** — L35 reaches 69.5% by step
400, L26 by 800, L17 by 1000, L8 last. By step 1200 the whole profile sits on
top of the MoE's 18k profile (mean 71.1% vs 76.7%, every depth slightly *less*
language-dominated). A single deep-layer reading taken early therefore looks
alarming for reasons that resolve on their own. **[measured]**

**`Router usage` / `Router per-samp`** — batch-mean usage CV² near zero is
ambiguous: healthy per-sample specialisation averages out the same way an
input-independent uniform router does, and the latter zeroes the CV² balance
penalty for free. The per-sample line separates them, because CV² is taken over
the batch mean while `max_w` / `entropy` are taken per sample and then averaged.

`max_w` is the weight of the winning expert; `entropy` is how spread the mix is.
With `num_experts=4`, an input-independent router gives `max_w=0.250`,
`entropy=ln 4=1.386`; a hard one-expert-per-sample router gives `1.000` / `0.000`.
Both statistics track the same underlying quantity — the spread of the router's
logits — so read them together as one number, not two.

**The step-18000 reading `max_w=0.440 entropy=1.237` overstated this.** Until
2026-08-04 the statistics were computed from the *noisy* logits: training adds a
fixed `N(0, 0.5)` for exploration, and at these logit scales that alone drives a
router with **zero** input dependence to `max_w=0.388 / entropy=1.301`. So the
noise floor is 0.388, not 0.250, and 0.440 sits just above it — consistent with a
learned logit spread of only ≈0.5, i.e. an eval-time mix near 0.39/0.25/0.20/0.16.
That is real differentiation but far weaker than the reading suggested, and the
old note claiming the training-mode value was a *lower* bound on eval peakedness
had the sign backwards: noise only ever adds logit variance, so it was an upper
bound. **[measured]** Fixed by reading the pre-noise weights, so the 0.250/1.386
reference is now the correct comparison. Rough decode of the corrected line:
`≤0.30` collapsed, `0.35–0.55` differentiating, `>0.8` near-hard routing.

The same reading was also taken from the wrong forward pass: with contrastive on,
the negative branch ran second and overwrote every router statistic, so the log
described routing under *permuted* instructions. Also fixed. Neither bug touched
training — both are diagnostic-only, and the balance loss always read the correct
branch — but no router number recorded before 2026-08-04 is comparable with one
taken after.

**`Action→ x-attn`** — added 2026-08-05, so **no run that produced a published
number here has it**, including the 92% checkpoint. To get a reference, load
ckpt 18000 and take one training step; the line prints on the first
gradient-analysis tick.

It reports the share of the action queries' cross-attention landing on vision
vs language positions of the VLM KV, router-weighted across experts, with the
`(xN)` factor normalising by each region's share of `L_vlm` — the share alone is
uninterpretable because language is a small minority of positions. `x1.0` is
proportional attention; the WiltechsVLA run at step 1200 read
`language=81.1% (x3.79)` against `[87 lang + 320 vis tok]`.

`x-attn lang/exp` breaks it out per expert, shallow band → deep. Measured on the
92% checkpoint (resumed for 200 steps), and the spread is the whole story:

| expert | VLM layers | language | **vision** |
|--------|-----------|----------|--------|
| E0 | 0–8 | 44.2% | **55.8%** |
| E1 | 9–17 | 90.6% | 9.4% |
| E2 | 18–26 | 81.1% | 18.9% |
| E3 | 27–35 | 91.4% | 8.6% |
| router-weighted | | 76.7% (x3.53) | 23.3% (x0.30) |

**Visual grounding lives almost entirely in the SHALLOW band.** E0 is the only
expert reading vision more than language; by layer 35 vision is down to 8.6%.
That fits what the layers hold — geometry early, task semantics late — and it
means a deep-layer-only reading says nothing about whether the model is
grounding. **[measured]**

Consequence for any single-stack model (WiltechsVLA): measuring the last layer
alone is misleading. 91% language at the deepest layer is *normal*, not a
failure. The VLA now samples four depths matching these band boundaries
(DiT/VLM layers 8, 17, 26, 35 at `num_dit_layers=36`) so the two line up depth
for depth; its mean over those four is what compares with 76.7% here.

**`Loss components`** — `contrastive` is a hinge, `relu(margin − diff_sq)`. It
falls to 0 as the constraint is satisfied; that is the design, not a fault.
Observed: 0.0204 (s400) → 0.0058 (s1800) → 0.0032 (s3000) → 0.0000 (s15400). The
bar is low — `margin=0.05` against a main loss around 0.4 — so from ~15k there is
no explicit language-dependence pressure left. **Raising `--contrastive_margin`
is the lever for the next run**, not mid-run, which would break comparability.

What the hinge does *not* measure: it requires `v_wrong ≠ v_t`, never that `v_t`
is right. Saturation means language is not being ignored; it says nothing about
whether it is being used correctly.

**Startup lines that must be checked** — each corresponds to a flag that fails
silently if it does not take:

```
RobotCNN cameras (wrist-specialized): ['observation.images.image2']
Image aug: colour/blur only (no geometric)
Vision input size 512px -> 256 tokens/frame for ['observation.images.image']
[task_rewrites] OK — all 17 rephrasing keys match real LIBERO task strings.
[wiltechs_moe] lang budget: max_len=128, longest ... kept=N / 128
```

---

## 5. Why each non-default flag

| flag | reason |
|---|---|
| `--use_descriptive_objects` | `store_true`, default off. Without it `task_rewrites.py` does nothing at all and every prompt change is a no-op. |
| `--vision_input_size 512` + hires on the third-person camera only | The between rewrite requires it; at 8×8 the probe fails its consistency control. Wrist camera cannot resolve those relations at any scale, so restricting roughly halves the added cost. `L_vlm` is also every expert layer's cross-attn K/V length, so cost multiplies by `num_experts × expert_num_layers`. |
| `--robot_cnn_wrist_only` | The trainable ResNet on the third-person view is a language-blind shortcut to the same object positions the VLM KV carries, and it converges faster than cross-attention to a frozen representation. Wrist-only splits the labour: selection via language, last-cm servoing via CNN. **[inferred]** — the ablation that would confirm it was not completed. |
| `--contrastive_loss_weight 0.1` | 0 reproduced language-ignoring in WiltechsVLA; 0.1 fixed it. |
| `--contrastive_hard_negatives` | Random in-batch pairing almost never lands a same-template pair, so the hinge is satisfied by trivially different instructions. |
| geometric image aug off (now default) | `RandomAffine` moved objects in the frame while the action label stayed put — training position-invariance in a spatial-referring task. On a 256px frame `translate=0.03` is ±7.7px against the separations being resolved. The LIBERO camera is fixed, so no viewpoint robustness is bought in exchange. **[inferred]** |

---

## 5b. The 92% run's actual size — read this before any comparison

`dit_hidden=1280`, giving **1,227,386,880** expert params (4 × 9 layers at
34.1M). Confirmed three independent ways from its training log: the expert
total, `Sink Token` = 1280 params, `Time Embedder` = 3,279,360 = 2·(1280²+1280).
**[measured]**

The config file's *default* is 640. Reading the default instead of the log is
how a WiltechsVLA run got built at 1/3 the parameters and scored 25% against
this 92% — with width, gradient-update count (10k at batch 65 vs 18k at batch
40) and topology all differing at once, so it attributed nothing. Take sizes
from the log, not the defaults.

---

## 6. Open

- **Is the router doing anything?** The corrected per-sample statistic (§4) puts
  the learned routing barely above the collapse floor, so this is now the
  cheapest high-value question: no retraining, existing checkpoint, one env var.

  ```bash
  WILTECHS_MOE_ROUTER=uniform python <eval>   # router removed: fixed 1/E average
  WILTECHS_MOE_ROUTER=0       python <eval>   # expert 0 alone (VLM layers 0-8)
  ```

  `uniform` vs learned isolates the **router** — same parameters, same compute,
  only the mixing weights change. This is now the most informative experiment
  available, because §8 showed the parallel topology is what wins and the router
  is the only part of it not yet tested: if `uniform` also scores ~92%, the win
  is the 4-way *average*, not the routing, and the router plus its balance loss
  can go while the topology stays.

  (An earlier version of this note said a matching `uniform` would justify
  replacing the experts with one deeper 36-layer decoder, "strictly more
  expressive, a fixed average being a special case of it". That is wrong on both
  counts — see §8.)

  Single-expert vs uniform isolates the **ensemble**, and is confounded — one
  expert is 1/E of the parameters *and* sees only its own 9-layer block, so a
  drop there is not evidence that routing matters. Do not read it as one.

  Power: at n=50 and ~92%, the 2σ band on an unpaired rate *difference* is about
  11 points, so only a collapse is visible that way. The runs share initial
  states, so compare the per-episode success **sets** — the discordant episodes
  are the signal.
- **Suite-level number.** 92% is one task. Published LIBERO figures are 10 tasks
  × 50 episodes. The vocabulary unification touched nine previously-working
  tasks, and that is this round's biggest risk — it has not been measured.
- **Attribution.** Four changes shipped together; no baseline on the corrected
  harness. If it matters, the old 25k checkpoint under `patch_lerobot_libero()`
  gives it for free (no training).
- **Does the model use language, or just avoid contradicting it?** The ablation
  — swap in another task's instruction and re-eval — was written but not run.
  Zeroing the RobotCNN tokens instead is **not** a valid substitute: per-token
  dropout at p=0.3 never produces an all-zero input, so all-zero is out of
  distribution and the 0% it produced is uninformative. Shuffle the tokens
  across the batch instead.
- **The sampler distribution.** Whether the rewrite's plate-anchored selector
  even stays *correct* out there is unmeasured; run
  `libero_layout_geometry.py --source sampler`. If `ratio < 1` occurs, the
  instruction names the distractor in those layouts and no training fixes it.
- **`control_freq=10` and `max_episode_steps=200`** both deviate from the stock
  harness. The first is justified (the dataset is 10 Hz); both affect
  comparability with published numbers and should be stated when reporting.

---

## 7. Disproved — do not redo

- **"The three objects are not collinear."** From eyeballing a screenshot. Sim
  ground truth: the target is 4 mm off the ramekin→plate line at t=0.58.
- **"bowl_2 is the target of the between task."** The BDDL `goal_state` is
  authoritative and says `['on', 'akita_black_bowl_1', 'plate_1']`. Every
  libero_spatial task is its own scene and names `akita_black_bowl_1` as its
  target, so the numbering carries no meaning across tasks.
- **"Unify all tasks onto the plate anchor."** It is right for t0 (2.30× vs the
  ramekin's 1.20×) and t8, and useless elsewhere — for t1 the plate separates
  the bowls 1.01×, a coin flip.
- **"The 50 canonical layouts vary a lot, so init states explain the variation."**
  They span ±1.5 cm and objects never swap. The variation seen in early probes
  was the placement sampler, because `_init_state_id` was not selecting anything.
- **"~17 / ~19 native px between the ramekin and the distractor bowl."** Both
  eyeballed, never measured, and mutually inconsistent. The measured figure is
  0.127 m; convert it yourself from `--annotate`'s row/col output if a pixel
  number is needed.
- **"Increase `num_thought_tokens`, the gradient is low."** The gradient is low
  because RMSNorm rescales magnitude away. The gates moved from 0.100 to 0.171,
  so the tokens are used and capacity is not the constraint.

---

## 8. The parallel topology is doing the work — WiltechsVLA head-to-head

A single 36-layer sequential DiT was built to test whether the MoE's four
parallel 9-layer experts are needed. Everything else was matched: same frozen
backbone, `dit_hidden=1280`, `batch=40`, same `--dataset_id`, same 50k cosine
schedule (both read `lr=7.4e-05` at step 18k), same 36 VLM layers read 1:1, same
CoT rewrites, `--vision_input_size 512` on the third-person camera only,
`--robot_cnn_wrist_only`, `--contrastive_loss_weight 0.1`.

| run | libero_spatial t0 | 95% CI |
|-----|-------------------|--------|
| VLA 10k, `dit_hidden=640`, batch 65 | 3/12 = 25.0% | [9%, 53%] |
| VLA 12k, matched config | 4/12 = 33.3% | [14%, 61%] |
| **MoE 18k** | **46/50 = 92.0%** | [81%, 97%] |

**[measured]** `P(≤4 | p=0.92, n=12) = 6.2e-07`, so the gap is decisive even at
n=12. The width fix bought nothing measurable: 3/12 vs 4/12 is Fisher p=1.00, so
the 3× parameter deficit that looked like the obvious cause was not it.

The VLA run was stopped at 12k rather than 18k — 10k→12k showed no movement
toward 92%, so the remaining 6k had low expected information. **The comparison
is therefore 12k vs 18k, not step-matched**; state that whenever the number is
quoted.

**The claim that motivated the experiment was wrong.** A 36-layer sequential
stack was said to be "strictly more expressive" than four parallel 9-layer
experts because "a fixed average is a special case of it". It is not:
`f4∘f3∘f2∘f1 ≠ (f1+f2+f3+f4)/4`. Sequential composition and parallel averaging
are different function classes at equal width, neither containing the other;
embedding four independent branches in one residual stream needs roughly 4× the
width or they interfere. The claim was asserted from intuition and never
checked, and it was the entire justification for "delete the router, use one
deep decoder".

Unverified hypotheses for *why* parallel wins, in the order worth testing:
1. **Ensemble variance reduction** — flow matching is regression; averaging four
   predictors helps for reasons that have nothing to do with routing. The
   `WILTECHS_MOE_ROUTER=uniform` ablation (§6) tests exactly this.
2. **Gradient path length** — 9 layers to the loss instead of 36.
3. **Depth-band commitment** — each expert owns one band of VLM layers; the
   sequential stack must push all 36 bands through one residual stream.

Two diagnostics diverged before the eval did, and now correlate with it. Neither
is established as causal, but they are the leading candidates:

| | VLA @12k | MoE @18k |
|---|---|---|
| `L8` language (null 6.3%) | 69.6% = **11.0×** | 44.2% = 7.0× |
| Q-Former gate mean (init 0.100) | 0.087, **falling** | 0.110, rising |

The Q-Former collapse is not weight decay — at `wd=1e-6` the cumulative shrink
over 12k steps is 0.9999988, while the observed weight norm went 106.15 → 43.57
(0.41×). The task loss is switching the module off. **[measured]**

## 9. The grounding probe — and three artifacts that each reversed its sign

`kv_grounding_probe.py` fits a ridge from the frozen VLM's vision-position
hidden states to a bowl's xyz across LIBERO's 50 canonical layouts, and fits the
same probe against the distractor bowl as a control. It answers the question a
10-hour run cannot: is the target's position *in* the representation the DiT
reads, or is the policy being asked to infer something that is not there?

**[measured, task 0, layer 8, after all three fixes below]**

| probe | CV R² | R² x | R² y | err x | err y |
|---|---|---|---|---|---|
| TARGET bowl | 0.366 | 0.52 | 0.22 | 8.4 mm | 8.4 mm |
| DISTRACTOR bowl | 0.419 | 0.35 | 0.49 | 6.4 mm | 6.5 mm |
| shuffled labels | −0.051 | −0.04 | −0.06 | 12.4 mm | 9.8 mm |

Normalised by each object's own travel (target moves 12.1/9.5 mm, distractor
8.0/9.0 mm), the errors are **0.79 vs 0.76** — the same to 0.026, where 1.00 is
the predict-the-mean baseline.

**Conclusion: the representation encodes where the bowls ARE, to ~8 mm against a
~150 mm bowl separation, but it does NOT encode which one the instruction
selects.** Three runs at layer 8 put the target at 0.434 / 0.463 / 0.366 and the
distractor at 0.379 / 0.366 / 0.419 — the *ordering flips between runs*, which
is what no real effect looks like.

Two consequences follow. Under `text_first=True` the language positions precede
the images under the causal mask, so they carry zero scene content; if the
vision positions also do not privilege the target, then nothing in the VLM
output identifies it and the DiT must derive the relation itself from the
encoded positions of all objects. And the run-to-run spread (0.097 on the
aggregate) is 2–3× the ± the script prints, because that ± covers only fold
reassignment, not the bf16 forward. **Do not read any R² gap below ~0.1.**

### The artifacts

Each was found only because a result looked wrong, and each on its own was
enough to reverse the conclusion. All three are now guarded in the script.

**1. A zero-variance label dimension scored +1.0.** `R² = 1 − ss_res/max(ss_tot,
1e-12)`. Both bowls rest on the table, so z barely varies — but *unequally*:
the distractor's z std is exactly 0.000 and the target's is 1e-4. For the
distractor the ridge predicts the constant exactly, giving `1 − 0/1e-12 = +1.0`
folded into the 3-dim mean; for the target z scored −0.37. On synthetic data
with **identical** x,y signal (0.257 each) that alone produced 0.505 vs 0.149 —
against the 0.590 vs 0.166 then being measured. Every target-vs-distractor
number the script printed before the dim mask was reading this.

**2. The layouts were resampled on every run.** The probe built the stock
`lerobot.envs.libero.LiberoEnv`, whose `reset()` writes the init state and then
calls the underlying `reset()`, which re-runs the placement initialiser and
discards it — and the probe passes no seed. Two identical invocations reported
0.164 and 0.392 because they fitted **different datasets**. The script now
applies `libero_env_fixed.patch_lerobot_libero` and prints a fingerprint of the
ground-truth positions; two runs of one command must print the same value.

Before finding this I had diagnosed the swing as estimator noise at d/n = 256:1
and proposed fold-internal PCA as the fix. Synthetic data at the same d, n and
signal strength said otherwise: PCA moved R² 0.246 → 0.243 and left the spread
unchanged under both feature perturbation (0.000 either way) and fold
reassignment (±0.017 vs ±0.016). **The proposed fix addressed nothing.** It is
kept as `--pca_dim`, defaulting to off.

**3. R² rewards the object that moves further.** `R² = 1 − MSE/Var`, so against
a *fixed* feature precision — an 8×8 grid quantises position to one token per
32×32 source px regardless of travel — the object with more travel scores higher
for the same absolute error. The target's x std is 1.52× the distractor's and
its x R² was 0.56 vs 0.29; on y, where the two move within 5% of each other, it
was 0.37 vs 0.44 and the *distractor* won. Modelling a fixed 4 mm precision
reproduces this axis by axis: 0.84 vs 0.66 for objects travelling 12.1 vs 8.0
mm, at RMSE 4.47 vs 4.27 mm.

A first synthetic check missed it by scaling the signal proportionally, which
makes R² exactly scale-invariant (0.579 both) — the confound needs an
*absolute* error floor to appear. The script now reports per-dimension RMSE and
normalises each object's error by its own travel.

### Method note

Three independent artifacts, each individually sign-reversing, in one ~50-line
diagnostic. Two were caught by a result that did not match any anticipated case
and one by a synthetic check of a proposed fix. **Before a probe result changes
a training decision, build the null on synthetic data with the same n, d and
effect size** — that is what separated the real confound from the invented one
here, and it costs minutes.

### The VLM never resolves the referring expression — anywhere

Four configurations at layer 8, reported as error / the object's own travel,
where 1.00 is predict-the-mean. **[measured]**

| | read | target | distractor |
|---|---|---|---|
| A text-first | vision | 0.65 / 0.78 | 0.81 / 0.73 |
| B **text-last** | **language** | **0.92 / 1.02** | **0.96 / 0.90** |
| C text-last, WRONG instruction | language | 0.99 / 0.96 | 0.90 / 0.83 |
| D text-last | vision | 0.70 / 0.78 | 0.82 / 0.71 |

Plus the no-language control: `--instruction ""` on A's configuration gives
target 0.437 against 0.434 / 0.463 / 0.366 / 0.429 with the real instruction.

**B closes the `--text_last` branch.** Text-last is the only ordering in which
the language positions can attend to the image, so it was the one route by which
the VLM's own 36 layers could resolve "the bowl between the plate and the
ramekin" before the DiT sees anything. They carry essentially nothing: both
bowls sit at 0.92–1.02, i.e. no better than guessing the average layout.

**D says the ordering changes nothing at all.** A and D agree to within the
run-to-run spread on both objects, so text-first and text-last are equivalent
for every readout tested. The `text_first` flag is neither delivering its
premise nor costing anything measurable.

Putting it together: the bowls' positions are in the vision tokens to ~8 mm
against a ~150 mm separation, and **that is the only thing the VLM output
contains.** The instruction is not fused into the vision positions, the
selection is not written into the language positions, and neither ordering
changes either fact. Nothing in the frozen backbone's output identifies which
bowl is meant — the DiT would have to compute the relation itself from
language × coordinates, across cross-attention, with no supervision that
rewards doing so.

That is consistent with every training measurement: 77/23 language-vs-vision
cross-attention mass, `--no_robot_cnn` not moving the vision share over ~10k
steps, 40 tasks of scene diversity not moving it, and 92% on the task being
reachable without fine grounding — a prior over where "between" lands is enough.

### Open

- **`box_encoder.py` is the remaining path.** Resolve the relation offline and
  hand the DiT coordinates, rather than asking it to infer a selection the
  backbone never encodes. `--annotate_detect` already shows whether Qwen can
  produce the boxes that readout would consume.
- Not yet tested: whether a *non-relational* instruction changes B. Every
  instruction tried names the target by a spatial relation. An instruction that
  names it by an intrinsic visual property is the one case where the VLM would
  not need to compute anything — but two identical black bowls may not admit
  one, which is itself the reason this task is hard.
- `--instruction ""` was inert until 2026-08-07: the argument was read with
  `or`, so the empty string fell back to the task's own language and the
  ablation would have silently re-run the baseline.
