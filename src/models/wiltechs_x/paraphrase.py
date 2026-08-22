"""Surface-form augmentation for instructions.

WHY THIS EXISTS. Measured on wiltechs-x-114k (libero_spatial task 7, 20
episodes, identical harness):

    told its own instruction
      "pick up the black bowl on the stove and place it on the plate"        60%
    told ANOTHER TASK's instruction, verbatim from the suite
      "pick up the black bowl on the wooden cabinet and place it on the plate" 0%
    told a PARAPHRASE of its own instruction
      "pick up the black bowl that is on the stove and put it onto the plate"  0%

The middle row looks like obedience -- the video shows the arm going cleanly to
the cabinet. The third row is what it actually means. A model that understood
the sentence would be untouched by `that is` and `put it onto`; this one
collapses. Both rows are explained by the same thing: it has memorised the ~40
instruction strings in the dataset and maps each to a behaviour. Give it a
string from the table and it retrieves; give it anything else and it retrieves
nothing.

The language probe agrees once you read it correctly. It swaps in OTHER TASKS'
instructions, all of which are table entries, so its d(lang) measures "can the
model tell the 40 known strings apart" -- exactly the lookup, and blind to
whether any of it is understanding.

WHAT THIS DOES. Trains on several phrasings per task, sampled per sample per
step, so surface form stops being a usable key. The relation phrase ("on the
stove") is the discriminative content and is varied too, or the model just
memorises relation strings instead of whole ones.

The original string is always in the set: evaluation uses it, and a number
taken on a phrasing the model never saw is not comparable to the ones already
recorded.

Deterministic templates rather than an LLM: the augmentation set has to be
reproducible across runs and machines, or two runs differ by an input
distribution nobody wrote down.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Locative relations may take "that is"/"which is"; source/path ones may not
# ("the bowl that is from table center" is not English). Ordered longest-first
# so "next to the" cannot be matched by a shorter prefix.
_LOCATIVE = ("between the", "next to the", "in the top drawer of the",
             "on top of the", "on the", "in the", "inside the", "near the")
_NON_LOCATIVE = ("from the", "from")

_GRASP = ("pick up", "grasp", "take", "lift")
_PLACE = (("place", "on"), ("put", "on"), ("place", "onto"), ("put", "onto"),
          ("set", "on"))
_QUALIFIER = ("", "that is ", "which is ")

_PATTERN = re.compile(
    r"^(?P<verb>pick up|grasp|take|lift)\s+(?P<obj>.+?)\s+"
    r"(?P<rel>(?:%s)\s+.+?|(?:%s)\s+.+?)\s+"
    r"and\s+(?P<pverb>place|put|set)\s+it\s+(?P<prep>onto|on|in|into)\s+(?P<dest>.+)$"
    % ("|".join(re.escape(r) for r in _LOCATIVE),
       "|".join(re.escape(r) for r in _NON_LOCATIVE)),
    re.IGNORECASE)


def paraphrases(instruction: str, limit: int = 8) -> list[str]:
    """Surface variants of one instruction, original first.

    Returns [instruction] alone when the sentence does not match the
    pick-and-place pattern. That is deliberate: a wrong guess at the structure
    would hand the policy a sentence meaning something else, and silently
    train it on the wrong task.
    """
    text = " ".join(str(instruction).split())
    m = _PATTERN.match(text)
    if not m:
        return [text]

    obj, rel, dest = m["obj"], m["rel"], m["dest"]
    locative = rel.lower().startswith(tuple(r.lower() for r in _LOCATIVE))

    out, seen = [text], {text.lower()}
    for grasp in _GRASP:
        for qual in (_QUALIFIER if locative else ("",)):
            for pverb, prep in _PLACE:
                cand = f"{grasp} {obj} {qual}{rel} and {pverb} it {prep} {dest}"
                key = cand.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(cand)
    # Deterministic order, and a stable subset when limit < len(out): the
    # nested loops already vary the grasp verb slowest, so a prefix would be
    # all "pick up". Stride instead, keeping the original at index 0.
    if limit and len(out) > limit:
        rest = out[1:]
        step = len(rest) / (limit - 1)
        out = [out[0]] + [rest[int(i * step)] for i in range(limit - 1)]
    return out


def build_table(instructions, limit: int = 8) -> dict[str, list[str]]:
    """instruction -> its variants, for every unique instruction given."""
    table: dict[str, list[str]] = {}
    for ins in instructions:
        key = " ".join(str(ins).split())
        if key not in table:
            table[key] = paraphrases(key, limit)
    return table


def load_table(path: str | Path) -> dict[str, list[str]]:
    """Read a paraphrase table, normalising keys the way lookups will.

    Raises on a variant list that omits its own key: evaluation is run on the
    original string, and a table that dropped it would train a model on
    phrasings it is never scored with.
    """
    data = json.loads(Path(path).read_text())
    table = {}
    for k, v in data.items():
        key = " ".join(str(k).split())
        variants = [" ".join(str(x).split()) for x in v]
        if key not in variants:
            raise ValueError(
                f"paraphrase table entry {key!r} does not contain the original "
                f"string among its variants; eval uses the original, so "
                f"training must see it too")
        table[key] = variants
    return table


if __name__ == "__main__":  # quick look at what a suite would get
    import sys
    samples = sys.argv[1:] or [
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "open the top drawer of the cabinet",
    ]
    for s in samples:
        v = paraphrases(s)
        print(f"\n{s}\n  -> {len(v)} variants")
        for x in v:
            print(f"     {x}")
