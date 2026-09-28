"""One file of golden tickets, keyed by (suite, task_id), living next to a checkpoint.

A ticket is bound to the policy that was searched with it -- same weights,
same horizon, same control frequency -- so the natural place to keep it is the
checkpoint repo itself:

    huggingface-cli upload <repo> ./tickets/golden_tickets.safetensors
    huggingface-cli upload <repo> ./tickets/golden_tickets.json

Eval then takes `--noise_tickets auto` and looks each task up by id.

PARTIAL BUNDLES ARE NORMAL AND MUST BE VISIBLE. Search is hours per task, so a
bundle will usually cover some tasks and not others. A suite average that
mixes ticketed and Gaussian tasks is not comparable to anything, so the loader
reports coverage and the eval writes it into the result JSON rather than
letting it pass silently.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

BUNDLE = "golden_tickets.safetensors"
META = "golden_tickets.json"


def key(suite: str, task_id: int) -> str:
    return f"{suite}.{int(task_id)}"


def save_ticket(out_dir, suite: str, task_id: int, ticket: np.ndarray, meta: dict):
    """Read-modify-write. Bundles are a few hundred KB; rewriting is free and
    it means a run killed mid-search still leaves every finished task on disk.

    NOT SAFE FOR CONCURRENT SEARCHES ON ONE DIRECTORY, and there is no lock.
    Two processes that both read {goal.0}, then write {goal.0, goal.1} and
    {goal.0, object.0}, leave whichever finished first erased. Give each
    concurrent search its own --out and `merge()` them at the end.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f, m = out / BUNDLE, out / META
    tensors = load_file(str(f)) if f.exists() else {}
    info = json.loads(m.read_text()) if m.exists() else {}
    k = key(suite, task_id)
    tensors[k] = np.asarray(ticket, dtype=np.float32)
    info[k] = meta
    save_file(tensors, str(f))
    m.write_text(json.dumps(info, indent=1, sort_keys=True))
    return f


def load_bundle(path):
    """-> (dict[key] -> (H, D) array, dict[key] -> meta). `path` may be the
    bundle file or a directory containing it (a checkpoint, for instance)."""
    p = Path(path)
    f = p if p.is_file() else p / BUNDLE
    if not f.exists():
        raise FileNotFoundError(
            f"no {BUNDLE} at {p}. Search produces it; if the tickets live on "
            f"the hub, they must be downloaded with the checkpoint.")
    m = f.with_name(META)
    return load_file(str(f)), (json.loads(m.read_text()) if m.exists() else {})


def coverage(tensors, suite: str, task_ids) -> dict:
    """Which of the tasks about to be evaluated actually have a ticket."""
    have = [t for t in task_ids if key(suite, t) in tensors]
    miss = [t for t in task_ids if key(suite, t) not in tensors]
    return {"with_ticket": have, "gaussian": miss,
            "n_with_ticket": len(have), "n_tasks": len(list(task_ids))}


def top_k(done_npz, k: int = 8, min_rate: float = 0.0):
    """-> (k, H, D) of the best-scoring candidates from a finished search.

    The bundle keeps one ticket per task; this recovers the rest from the
    _done_*.npz the search now leaves behind. The paper (D.6.2) finds that
    drawing uniformly from the top-k performs as well as the single best while
    restoring stochasticity -- which matters here, because one fixed ticket
    makes the policy deterministic and removes the per-chunk re-draw this
    benchmark measured at 25 points.

    Ranked on CUMULATIVE wins/runs, so candidates eliminated early (5 episodes)
    are compared against survivors (15). That favours survivors, which is the
    intent: an early exit means the evidence stopped at "not promising".
    """
    z = np.load(done_npz, allow_pickle=True)
    w, r, cands = z["wins"], z["runs"], z["cands"]
    rate = np.where(r > 0, w / np.maximum(r, 1), -1.0)
    order = sorted(range(len(rate)), key=lambda i: (-rate[i], -r[i]))
    keep = [i for i in order if rate[i] >= min_rate][:k]
    return cands[keep], [(int(i), int(w[i]), int(r[i])) for i in keep]


def report(path) -> int:
    """Print what a bundle actually contains. -> number of weak entries.

    Search banks a ticket for every task it finishes, including the ones it
    could not beat Gaussian on -- deliberately, so the candidates and the
    metadata survive -- and the search log then reports every finished task
    the same way. object T5 went in at 53% against an 85% baseline and reads
    as "already in the bundle" next to T9's 15/15.
    """
    tensors, info = load_bundle(path)
    rows, weak = [], 0
    for k in sorted(tensors):
        m = info.get(k, {})
        b = m.get("beats_baseline")
        verdict = {True: "BEATS", False: "weak -- eval falls back to Gaussian",
                   None: "banked, unresolved"}[b if b in (True, False) else None]
        if b is False:
            weak += 1
        c = m.get("certification")
        cert = (f"{c['ticket']} vs {c['gaussian']}" if c else "not certified")
        # PROVENANCE, because the two kinds of ticket produce numbers that look
        # identical. One was searched on layouts the eval never touches and its
        # suite average is a held-out result; the other was picked by running
        # candidates on the reported layouts, so its average is exact but
        # in-sample. A bundle holding both is fine -- the workflow wants one
        # bundle -- as long as the report says which is which.
        if m.get("selected_on_reported_layouts"):
            verdict = ("SWEPT the reported layouts (in-sample)"
                       if m.get("swept_reported_layouts")
                       else "better on the reported layouts (in-sample)")
        rows.append((k, m.get("search_success", "?"),
                     m.get("baseline_search", "?"), cert, verdict))
    w0 = max([len(r[0]) for r in rows] + [4])
    w3 = max([len(r[3]) for r in rows] + [13])
    print(f"{'task':<{w0}}  {'search':>8}  {'gaussian':>9}  "
          f"{'certified':<{w3}}  verdict")
    for k, sc, bl, ct, v in rows:
        print(f"{k:<{w0}}  {sc:>8}  {bl:>9}  {ct:<{w3}}  {v}")
    print(f"\n{len(rows)} tickets, {weak} weak.")
    if weak:
        print("The weak ones are inert at eval time -- it checks "
              "beats_baseline and uses Gaussian -- so they cost nothing to "
              "leave in place. Re-search them with more --tickets; the search "
              "skips them by default, --retry_failed redoes them.")
    print("SEARCH RATE IS NOT THE RESULT. It is the number the ticket was "
          "selected on. Report with eval_wiltechs_x.py --init_state_offset 0.")
    return weak


def merge(out_dir, *in_dirs):
    """Combine bundles from concurrent searches into one.

    Refuses to silently drop a ticket: a key present in two inputs is an
    error, because the two were searched separately and picking one by
    directory order would make the result depend on argument order.
    """
    tensors, info, src, missing = {}, {}, {}, []
    for d in in_dirs:
        try:
            t, m = load_bundle(d)
        except FileNotFoundError:
            # Searches finish at different times and merging is how a partial
            # set gets evaluated, so a directory with nothing in it yet is an
            # ordinary state, not an error.
            missing.append(d)
            continue
        dup = [k for k in t if k in tensors]
        if dup:
            def _v(meta, k):
                b = meta.get(k, {}).get("beats_baseline")
                return (f"{meta.get(k, {}).get('search_success', '?')} "
                        + {True: "BEATS", False: "weak",
                           None: "unresolved"}[b if b in (True, False) else None])
            raise ValueError(
                f"{d} and {src[dup[0]]} both define {dup}; merging would pick "
                f"one by argument order.\n"
                + "\n".join(f"  {k}: {src[k]} -> {_v(info, k)};  "
                            f"{d} -> {_v(m, k)}" for k in dup)
                + "\nDelete the one you do not want, then merge again.")
        for k in t:
            src[k] = d
        tensors.update(t)
        info.update(m)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / BUNDLE))
    (out / META).write_text(json.dumps(info, indent=1, sort_keys=True))
    print(f"{len(tensors)} tickets -> {out / BUNDLE}")
    for k in sorted(tensors):
        b = info.get(k, {}).get("beats_baseline")
        tag = {True: "", False: "   (weak -- eval uses Gaussian)",
               None: "   (unresolved)"}[b if b in (True, False) else None]
        print(f"  {k:<24} from {src[k]}{tag}")
    for d in missing:
        print(f"  (no bundle yet in {d})")
    return out / BUNDLE


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "runners":
        # THE BUNDLE KEEPS ONE TICKET; THE SEARCH KEPT ALL OF THEM. Sequential
        # halving usually leaves several candidates tied at the top tier's
        # score, and only the first was banked. When the banked one turns out
        # not to solve every canonical layout, a tied runner-up may -- and
        # testing one is 35 episodes (the layouts it has never run) against
        # hours for a fresh search.
        import numpy as _np
        z = _np.load(sys.argv[2], allow_pickle=True)
        w, r, cands = z["wins"], z["runs"], z["cands"]
        rate = _np.where(r > 0, w / _np.maximum(r, 1), -1.0)
        # DEPTH FIRST, then rate. Ranking by rate alone put candidates that
        # were eliminated in tier 1 above the deep survivors: object T5 listed
        # six tickets at 2/3 -- three layouts each, barely tested -- above the
        # 8/15 that the search actually banked, and object T3 listed a 7/9
        # above its 9/15. Runs differ BECAUSE the search stopped the weak ones
        # early, so a high rate over few layouts is the absence of evidence.
        order = sorted(range(len(rate)), key=lambda i: (-r[i], -rate[i]))
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        top = [i for i in order if r[i] > 0][:k]
        deep = max(r)
        # The banked one is named in the bundle metadata, not inferred from the
        # ranking -- which is what produced the wrong "<- banked" marks.
        banked = None
        d = Path(sys.argv[2])
        m = d.parent / META
        if m.exists():
            stem = d.stem.replace("_done_", "")
            suite_, tid_ = stem.rsplit("_t", 1)
            info = json.loads(m.read_text()).get(key(suite_, int(tid_)), {})
            banked = info.get("ticket_index")
        g = json.loads(str(z["geom"])) if "geom" in z.files else None
        print(f"{str(z['desc'])}")
        if g:
            print(f"  searched with {g.get('tickets')} tickets, "
                  f"{g.get('envs_per_tier')} layouts per tier, from id "
                  f"{g.get('init_state_offset')}")
        print(f"  deepest candidates ran {int(deep)} layouts\n")
        print(f"{'rank':>4}  {'ticket':>6}  {'score':>7}  rate")
        for n, i in enumerate(top):
            mark = "   <- banked" if banked is not None and i == banked else ""
            if r[i] < deep:
                mark += f"   (only {int(r[i])} layouts -- eliminated early)"
            print(f"{n:>4}  {i:>6}  {int(w[i]):>3}/{int(r[i]):<3}  "
                  f"{rate[i]:.0%}{mark}")
        at_depth = [i for i in range(len(r)) if r[i] == deep]
        best_at_depth = max(rate[i] for i in at_depth)
        tied = [i for i in at_depth if rate[i] >= best_at_depth - 1e-9]
        print(f"\n{len(at_depth)} candidate(s) reached {int(deep)} layouts; "
              f"{len(tied)} of them tied at the top score "
              f"{int(w[tied[0]])}/{int(deep)}.")
        if best_at_depth < 1.0:
            print("  The best full-depth candidate is not perfect even on the "
                  "layouts it was searched on, so no runner-up here is a "
                  "likely 20/20. Re-searching this task is the honest option.")
        if len(sys.argv) > 4 and sys.argv[4] == "--export":
            out = Path(sys.argv[2]).parent
            for n, i in enumerate(tied):
                f = out / f"{Path(sys.argv[2]).stem.replace('_done_', 'alt_')}_c{i}.npy"
                _np.save(f, cands[i])
                print(f"  wrote {f}")
            print("\nTest one with eval_wiltechs_x.py --noise_ticket <file> "
                  "--task_ids <id> --episodes 20 --init_state_offset 0,\n"
                  "then again at --init_state_offset 35 --episodes 15. A "
                  "candidate that takes both is 50/50.")
        raise SystemExit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        raise SystemExit(0 if report(sys.argv[2]) == 0 else 0)
    if len(sys.argv) < 4 or sys.argv[1] != "merge":
        print("usage: python ticket_bundle.py report <dir>\n"
              "       python ticket_bundle.py runners <_done_*.npz> [k] [--export]\n"
              "       python ticket_bundle.py merge <out_dir> <in_dir> [<in_dir> ...]",
              file=sys.stderr)
        raise SystemExit(2)
    merge(sys.argv[2], *sys.argv[3:])
