"""Where does the action expert actually LOOK?

    python src/attention_mass_wiltechs_x.py \
        --checkpoint outputs/wiltechs_x/checkpoint-14000 \
        --dataset_ids lerobot/libero

Forward-only, minutes not hours. The expert's suffix tokens attend over
[lang | vision | wrist | motion | readout | suffix] in one SDPA per joint
layer (ARCHITECTURE.md 3.1), so the share of attention mass landing on each
segment is a direct read of what the policy conditions on -- more direct than
the probe (which infers it from a loss delta) and far cheaper than the rollout
ablation (which infers it from behaviour).

WHY THE NORMALISED COLUMN IS THE ONE TO READ. The segments have wildly
different lengths -- 48 language tokens against 256 wrist and 128 vision -- so
raw mass says almost nothing: attending UNIFORMLY over the sequence already
puts ~10% on language. Every table below therefore reports

    x uniform  =  (segment's share of attention) / (segment's share of tokens)

1.0 means "indistinguishable from attending everywhere equally", which for a
policy that is supposed to be reading an instruction is the same as ignoring
it. Below 1.0 means the segment is actively suppressed.

WHY IT SWEEPS t. x_t = t*noise + (1-t)*action, so at t->0 the action is nearly
handed to the model in its own input and no conditioning is needed; at t->1 the
input is pure noise and EVERYTHING must come from the prefix. Language
dependence that exists at all should be strongest at high t. A flat profile
across t is itself a finding: it means the segment is being read (or ignored)
regardless of how much information the model actually needs.

WHAT IT CANNOT TELL YOU. Attention mass is not causation -- a head can attend
to a token and route none of its value into the output. Treat a high language
share as necessary-but-not-sufficient, and confirm with the rollout ablation.
A LOW share is the stronger signal: a segment that is not attended to cannot
be driving anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_wiltechs_x import load_policy, load_processors, pick_device
from train_wiltechs_x import ProgressDataset, build_datasets, resolve_checkpoint


@contextmanager
def record_attention(store: list, query_len: int):
    """Capture attention probabilities for calls whose query length matches.

    Patching F.scaled_dot_product_attention rather than adding a hook inside
    JointExpertLayer keeps the hot training path untouched, and guarantees the
    recorded probabilities come from the SAME q/k/mask the model used -- a
    reimplementation could drift from the real path, which is exactly the
    failure mode a diagnostic must not have.

    Only calls with `query_len` queries are recorded. In one forward that
    selects the joint layers' suffix passes and skips the prefix-only VLM
    layers, whose queries number L_prefix.
    """
    real = F.scaled_dot_product_attention

    def patched(q, k, v, attn_mask=None, is_causal=False, **kw):
        if q.shape[-2] == query_len:
            logits = (q.float() @ k.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            if attn_mask is not None:
                logits = logits + attn_mask.float()
            store.append(logits.softmax(-1).mean(dim=1).detach().cpu())  # mean over heads
        return real(q, k, v, attn_mask=attn_mask, is_causal=is_causal, **kw)

    F.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = real


def segment_ranges(spans, L_prefix, L_suffix, horizon):
    """Named (start, end) key ranges over the concatenated [prefix | suffix].

    Built from the spans _build_prefix already tracks, so this cannot drift
    from the real layout the way a hand-computed offset table would.
    """
    segs = {}
    if spans.get("lang"):
        segs["lang"] = [tuple(spans["lang"])]
    if spans.get("vision"):
        segs["vision"] = [tuple(s) for s in spans["vision"]]
    if spans.get("wrist"):
        segs["wrist"] = [tuple(s) for s in spans["wrist"]]
    if spans.get("motion"):
        segs["motion"] = [tuple(spans["motion"])]
    if spans.get("readout") is not None:
        r = int(spans["readout"])
        segs["readout"] = [(r, r + 1)]
    # Everything in the prefix that is not one of the above: the chat-template
    # header/tail and the <|vision_start|>/<|vision_end|> brackets. Reported so
    # the shares sum to 1 and a missing segment cannot hide in the remainder.
    claimed = torch.zeros(L_prefix, dtype=torch.bool)
    for rs in segs.values():
        for s, e in rs:
            claimed[s:e] = True
    other = []
    i = 0
    while i < L_prefix:
        if not claimed[i]:
            j = i
            while j < L_prefix and not claimed[j]:
                j += 1
            other.append((i, j))
            i = j
        else:
            i += 1
    if other:
        segs["template"] = other
    segs["self(suffix)"] = [(L_prefix, L_prefix + L_suffix)]
    return segs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Local checkpoint directory OR a Hugging Face repo id.")
    p.add_argument("--dataset_ids", nargs="+", required=True)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--times", nargs="+", type=float, default=[0.9, 0.5, 0.1],
                   help="Flow times to evaluate. t->1 is pure noise, where the "
                        "prefix is the ONLY source of information; t->0 hands "
                        "the action to the model in its own input.")
    p.add_argument("--per_layer", action="store_true",
                   help="Also print the per-layer breakdown. The mean can hide "
                        "language being read in a few layers and nowhere else.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    device = a.device or pick_device()
    ckpt = resolve_checkpoint(a.checkpoint, for_resume=False)
    policy = load_policy(ckpt, device, None)
    pre, _ = load_processors(ckpt, device, None)
    cfg = policy.config
    model = policy.model
    model.eval()

    D = build_datasets(a.dataset_ids, cfg.n_obs_steps, cfg.horizon, None)
    ds = (ProgressDataset(D["dataset"], D["ep_from"], D["ep_to"])
          if cfg.progress_head else D["dataset"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
        drop_last=True, generator=torch.Generator().manual_seed(a.seed))

    print(f"\n[attn] {ckpt}  {a.batches} x {a.batch_size}  t={a.times}")

    H = int(cfg.horizon)
    L_s = 1 + int(model.num_register_tokens) + H
    # totals[t][segment] -> [summed mass, n_layers*n_batches]; per_layer[t][seg]
    totals: dict = {t: defaultdict(float) for t in a.times}
    per_layer: dict = {t: defaultdict(lambda: defaultdict(float)) for t in a.times}
    counts: dict = {t: 0 for t in a.times}
    lengths: dict = {}

    it = iter(loader)
    with torch.no_grad():
        for bi in range(a.batches):
            raw = next(it)
            raw = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                   for k, v in raw.items()}
            batch = pre(raw)
            for k, v in raw.items():
                batch.setdefault(k, v)

            actions = batch["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0)
            actions = actions[:, :H]
            B = actions.shape[0]
            prefix, pad_mask, segments, spans = model._build_prefix(batch)
            _, cache, rope = model._run_prefix(prefix, pad_mask, segments, L_s)
            L_p = prefix.shape[1]
            segs = segment_ranges(spans, L_p, L_s, H)
            lengths = {n: sum(e - s for s, e in rs) for n, rs in segs.items()}

            noise = model.sample_noise(actions.shape, device)
            state = batch["observation.state"]
            for tv in a.times:
                t = torch.full((B,), float(tv), device=device)
                x_t = t[:, None, None] * noise + (1.0 - t[:, None, None]) * actions
                d0 = (torch.zeros(B, device=device)
                      if cfg.flow_objective == "shortcut" else None)
                store: list = []
                with record_attention(store, L_s):
                    model._suffix_pass(state, x_t, t, d0, cache, rope, pad_mask)
                if not store:
                    raise SystemExit(
                        f"recorded no attention: expected queries of length "
                        f"{L_s}. Did the suffix layout change?")
                for li, probs in enumerate(store):
                    # Queries: the ACTION tokens only. The state token and the
                    # registers are the expert's own scratch space; including
                    # them would average the thing we are measuring with
                    # something that has no reason to read the instruction.
                    act = probs[:, -H:, :]                      # (B, H, L_p+L_s)
                    for name, rs in segs.items():
                        m = sum(float(act[:, :, s:e].sum()) for s, e in rs)
                        m /= act.shape[0] * act.shape[1]         # per query
                        totals[tv][name] += m
                        per_layer[tv][li][name] += m
                counts[tv] += len(store)
            print(f"  batch {bi + 1}/{a.batches}", flush=True)

    n_layers = counts[a.times[0]] // a.batches
    L_total = L_p + L_s
    print(f"\nL_prefix={L_p}  L_suffix={L_s}  joint layers={n_layers}")
    print("segment lengths: "
          + "  ".join(f"{n}={l}" for n, l in lengths.items()))

    report = {"checkpoint": str(ckpt), "lengths": lengths,
              "L_prefix": L_p, "L_suffix": L_s, "layers": n_layers, "times": {}}

    names = list(lengths)
    for tv in a.times:
        norm = counts[tv]
        print(f"\n--- t = {tv}  ({'pure noise, prefix is everything' if tv >= 0.8 else 'action nearly given' if tv <= 0.2 else 'midpoint'}) ---")
        print(f"{'segment':>14} {'tokens':>7} {'share':>8} {'x uniform':>10}")
        row = {}
        for n in names:
            share = totals[tv][n] / norm
            uni = lengths[n] / L_total
            print(f"{n:>14} {lengths[n]:>7} {share:>7.1%} {share / uni:>9.2f}")
            row[n] = {"share": share, "x_uniform": share / uni,
                      "tokens": lengths[n]}
        report["times"][str(tv)] = row

    if a.per_layer:
        print("\nper-layer 'x uniform' (joint layer 0 = first_joint_layer)")
        for tv in a.times:
            print(f"\n  t={tv}")
            print("   layer " + "".join(f"{n:>13}" for n in names))
            for li in sorted(per_layer[tv]):
                cells = []
                for n in names:
                    share = per_layer[tv][li][n] / a.batches
                    cells.append(f"{share / (lengths[n] / L_total):>13.2f}")
                print(f"   {li:>5} " + "".join(cells))
        report["per_layer"] = {
            str(tv): {str(li): {n: per_layer[tv][li][n] / a.batches for n in names}
                      for li in per_layer[tv]} for tv in a.times}

    print("""
HOW TO READ
  x uniform ~ 1.0   the segment is attended no more than anything else, which
                    for language means the instruction is not being singled out
  x uniform  < 1.0  actively suppressed
  language rising as t -> 1   conditioning is real: the model leans on the
                    instruction exactly when it has nothing else to go on
  language flat across t      it is not being used as information
  Attention is necessary, not sufficient -- confirm a HIGH share with the
  rollout ablation. A LOW share needs no confirmation.""")

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
