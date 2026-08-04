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

## 6. Open

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
