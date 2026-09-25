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


def merge(out_dir, *in_dirs):
    """Combine bundles from concurrent searches into one.

    Refuses to silently drop a ticket: a key present in two inputs is an
    error, because the two were searched separately and picking one by
    directory order would make the result depend on argument order.
    """
    tensors, info, src = {}, {}, {}
    for d in in_dirs:
        t, m = load_bundle(d)
        dup = [k for k in t if k in tensors]
        if dup:
            raise ValueError(
                f"{d} and {src[dup[0]]} both define {dup}; merging would pick "
                f"one by argument order. Delete the one you do not want.")
        for k in t:
            src[k] = d
        tensors.update(t)
        info.update(m)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / BUNDLE))
    (out / META).write_text(json.dumps(info, indent=1, sort_keys=True))
    print(f"{len(tensors)} tickets -> {out / BUNDLE}")
    for k in sorted(tensors):
        print(f"  {k:<24} from {src[k]}")
    return out / BUNDLE


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4 or sys.argv[1] != "merge":
        print("usage: python ticket_bundle.py merge <out_dir> <in_dir> [<in_dir> ...]",
              file=sys.stderr)
        raise SystemExit(2)
    merge(sys.argv[2], *sys.argv[3:])
