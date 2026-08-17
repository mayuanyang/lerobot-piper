"""Does the WiltechsX policy actually USE the instruction?

    python src/probe_language_wiltechs_x.py \
        --checkpoint outputs/wiltechs_x/checkpoint-5000 \
        --dataset_ids lerobot/libero

No environment, no rollout: this is a forward pass over training batches, so it
costs minutes rather than hours and it measures the thing directly instead of
inferring it through the control stack.

WHY THIS EXISTS. Knowledge insulation (ARCHITECTURE.md 3.3) replaced the
contrastive hinge on the premise that a discrete CROSS-ENTROPY head keeps the
VLM's language pathway alive under action supervision. But that head predicts
ACTIONS, and on LIBERO the scene alone very nearly determines the action -- so
its CE can fall from ln(256)=5.545 to ~3.2 without the model ever reading the
instruction. `contrastive_loss_weight=0` reproduced exactly that failure in this
repo's earlier lineage. libero_spatial is where it bites hardest: ten tasks
share one tabletop and differ ONLY by instruction.

WHAT IT DOES. For each batch, recompute the loss with one input corrupted:

    lang    every sample gets a DIFFERENT sample's instruction
    vision  every camera rolled by one permutation (scene from another sample)
    state   observation.state rolled (proprioception from another sample)

`discrete` is the number to read -- it is noise-free (it depends on the prefix
readout and the binned action, not on the flow time or the sampled noise), so
the comparison is exact rather than statistical. The vision and state rows are
POSITIVE CONTROLS: without them a flat language row cannot be told apart from a
broken probe.

HOW TO READ IT. If d(lang) is ~0 while d(vision) and d(state) are large, the
policy is not conditioned on language and no amount of further training fixes
it -- the fix is architectural (restore a contrastive term, or
--no_knowledge_insulation so the flow loss trains the prefix too, or a much
larger --fast_token_loss_weight). If all three are ~0 the probe is broken; fix
that first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_wiltechs_x import load_policy, load_processors, pick_device
from train_wiltechs_x import ProgressDataset, build_datasets


def shuffle_language(tasks, rng):
    """Give every sample a DIFFERENT task's instruction.

    Not a roll: a batch drawn from one suite has many neighbours sharing a
    task string, and those samples would silently stay intact and dilute the
    effect toward zero -- which is the result we are trying to rule out.
    """
    n = len(tasks)
    out, changed = list(tasks), 0
    for i in range(n):
        cands = [j for j in range(n) if tasks[j] != tasks[i]]
        if cands:
            out[i] = tasks[cands[int(rng.integers(len(cands)))]]
            changed += 1
    return out, changed


def corrupt(batch, mode, cameras, rng):
    """Return a shallow copy of `batch` with exactly one input group replaced."""
    out = dict(batch)
    if mode == "intact":
        return out, 1.0
    b = batch["observation.state"].shape[0]
    if mode == "lang":
        tasks = list(batch["task"])
        out["task"], changed = shuffle_language(tasks, rng)
        return out, changed / max(b, 1)
    # One permutation shared across every corrupted key, so the sample gets a
    # coherent other-observation rather than a chimera of several.
    perm = torch.from_numpy(rng.permutation(b)).to(batch["observation.state"].device)
    same = float((perm == torch.arange(b, device=perm.device)).float().mean())
    keys = cameras if mode == "vision" else ["observation.state"]
    for k in keys:
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k][perm]
    return out, 1.0 - same


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_ids", nargs="+", required=True)
    p.add_argument("--batches", type=int, default=8,
                   help="Paired comparison, so the batch-to-batch variance "
                        "cancels; 8 is usually plenty for a verdict.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    device = a.device or pick_device()
    ckpt = Path(a.checkpoint)
    policy = load_policy(ckpt, device, None)
    pre, _ = load_processors(ckpt, device, None)
    cfg = policy.config
    policy.eval()                      # also puts the prefix under no_grad

    D = build_datasets(a.dataset_ids, cfg.n_obs_steps, cfg.horizon, None)
    ds = (ProgressDataset(D["dataset"], D["ep_from"], D["ep_to"])
          if cfg.progress_head else D["dataset"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
        drop_last=True, generator=torch.Generator().manual_seed(a.seed))
    cameras = list(cfg.cameras_for_vlm)
    print(f"\n[probe] {ckpt}  batches={a.batches} x {a.batch_size}  "
          f"cameras={cameras}")

    rng = np.random.default_rng(a.seed)
    MODES = ["intact", "lang", "vision", "state"]
    rows: dict[str, list[dict]] = {m: [] for m in MODES}
    frac: dict[str, list[float]] = {m: [] for m in MODES}

    it = iter(loader)
    for bi in range(a.batches):
        raw = next(it)
        raw = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
               for k, v in raw.items()}
        out = pre(raw)
        for k, v in raw.items():
            out.setdefault(k, v)                       # see train_wiltechs_x.prepare
        for m in MODES:
            batch, f = corrupt(out, m, cameras, rng)
            # `discrete` is noise-free, but `flow` is not: seed identically so
            # the two conditions see the same flow time and the same noise.
            torch.manual_seed(a.seed * 1000 + bi)
            _, parts = policy.model.compute_loss(batch, return_parts=True)
            rows[m].append(parts)
            frac[m].append(f)
        print(f"  batch {bi + 1}/{a.batches}  "
              + "  ".join(f"{m}:{rows[m][-1]['discrete']:.4f}" for m in MODES),
              flush=True)

    def col(m, key):
        return np.array([r[key] for r in rows[m]])

    print(f"\n{'=' * 72}\ndiscrete CE (the readout's dependence on each input)")
    print(f"  uniform baseline ln(256) = {np.log(256):.4f}")
    base = col("intact", "discrete")
    print(f"  {'intact':<8} {base.mean():.4f} +/- {base.std():.4f}")
    verdict = {}
    for m in MODES[1:]:
        d = col(m, "discrete") - base                  # PAIRED: same batch
        # paired std over batches, so batch difficulty cancels
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else float("inf")
        verdict[m] = float(d.mean())
        print(f"  {m:<8} {col(m, 'discrete').mean():.4f}   "
              f"delta {d.mean():+.4f} +/- {d.std(ddof=1):.4f}   "
              f"t={t:+.1f}   ({100 * np.mean(frac[m]):.0f}% of samples corrupted)")

    fl = col("intact", "flow")
    print("\nflow (same noise per pair; the expert's dependence)")
    print(f"  {'intact':<8} {fl.mean():.4f}")
    for m in MODES[1:]:
        d = col(m, "flow") - fl
        print(f"  {m:<8} {col(m, 'flow').mean():.4f}   delta {d.mean():+.4f}")

    print(f"\n{'=' * 72}")
    dl, dv, dst = verdict["lang"], verdict["vision"], verdict["state"]
    ctrl = max(dv, dst)
    if ctrl <= 0.02:
        print("PROBE INCONCLUSIVE: corrupting vision AND state barely moved the\n"
              "CE either, so this is not measuring input dependence at all.\n"
              "Fix the probe before reading the language row.")
    elif dl <= 0.02:
        print(f"LANGUAGE IS NOT USED. Corrupting the instruction costs "
              f"{dl:+.4f} nats\nagainst {ctrl:+.4f} for the best control "
              f"({dl / ctrl:.1%} of it). The discrete head predicts\nactions "
              f"from the scene and ignores the instruction, so knowledge\n"
              f"insulation is not doing what it replaced the contrastive hinge "
              f"to do.\nMore training will not fix this; see the module "
              f"docstring for the levers.")
    else:
        print(f"Language IS used: {dl:+.4f} nats, {dl / ctrl:.1%} of the best "
              f"control ({ctrl:+.4f}).\nThe grounding failures are not a "
              f"language-conditioning problem -- look at\ncontrol precision, "
              f"the 150-step eval cap, and NFE 4 vs 16 instead.")

    payload = {"checkpoint": str(ckpt), "batches": a.batches,
               "batch_size": a.batch_size,
               "discrete": {m: col(m, "discrete").tolist() for m in MODES},
               "flow": {m: col(m, "flow").tolist() for m in MODES},
               "delta_discrete": verdict}
    out_p = Path(a.out) if a.out else ckpt / "probe_language.json"
    out_p.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
