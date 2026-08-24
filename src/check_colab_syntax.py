#!/usr/bin/env python
"""Parse every source file with the OLDEST Python it has to run on.

This repo trains on Colab against a pinned Python 3.10 env, while development
happens on 3.12. The two disagree about f-strings: PEP 701 relaxed the grammar
in 3.12 so that a newline inside `{...}`, or a same-type quote nested in one,
is legal. On 3.10 both are `SyntaxError: unterminated string literal` -- raised
at IMPORT time, so it does not surface until the training command is already
running on the remote box.

That has now cost two round trips (see cf343af, and the VAL verdict line), and
`python -m py_compile` on the dev machine cannot catch it: 3.12 is exactly the
version that accepts it.

    python check_colab_syntax.py [--target 3.10] [paths...]

Finds a real interpreter of the target version if one is installed, because
nothing else reproduces the tokenizer. `ast.parse(feature_version=...)` does
NOT: feature_version gates AST-level features, and this difference lives in
the tokenizer. Verified -- it accepts the broken form.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CANDIDATES = (
    "python{v}",
    "/opt/homebrew/bin/python{v}",
    "/usr/local/bin/python{v}",
    "/usr/bin/python{v}",
    "/opt/anaconda3/envs/for_lerobot/bin/python",
)


def find_interpreter(version: str) -> str | None:
    for pat in CANDIDATES:
        cand = pat.format(v=version)
        exe = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if not exe:
            continue
        try:
            out = subprocess.run([exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                                 capture_output=True, text=True, timeout=10)
        except Exception:
            continue
        if out.stdout.strip() == version:
            return exe
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--target", default="3.10")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent
    files = sorted(Path(p) for p in a.paths) if a.paths else sorted(root.rglob("*.py"))
    files = [f for f in files if "__pycache__" not in f.parts]

    exe = find_interpreter(a.target)
    if exe is None:
        print(f"[check] no Python {a.target} found -- SKIPPED. Nothing else "
              f"reproduces its tokenizer, so this check is all-or-nothing.\n"
              f"        Tried: {', '.join(c.format(v=a.target) for c in CANDIDATES)}")
        return 0

    print(f"[check] {exe} ({a.target}) over {len(files)} file(s)")
    bad = []
    for f in files:
        r = subprocess.run([exe, "-c", "import sys;compile(open(sys.argv[1]).read(),sys.argv[1],'exec')",
                            str(f)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append((f, r.stderr.strip().splitlines()[-1] if r.stderr else "?"))

    for f, msg in bad:
        print(f"  FAIL {f.relative_to(root)}: {msg}")
    print(f"[check] {len(files) - len(bad)}/{len(files)} parse under {a.target}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
