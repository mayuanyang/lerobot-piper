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
    it means a run killed mid-search still leaves every finished task on disk."""
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
