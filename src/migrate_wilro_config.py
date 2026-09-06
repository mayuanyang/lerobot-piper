#!/usr/bin/env python
"""Repair a wilro / wilro_moe config.json that cannot be loaded.

Between 2026-09-05 and this fix, WilroConfig.__post_init__ set its legacy
aliases to None AFTER translating them, and draccus serialises that as JSON
null. On draccus older than 0.10 an `Optional[bool]` field resolves to
decode_bool rather than the union decoder, so loading raises

    DecodingError: `use_robot_ca`: Couldn't parse 'None' into a bool

and the checkpoint is unopenable by the code that wrote it. The class no longer
emits null (it mirrors each alias onto the field it aliases), but files already
written need this.

    python migrate_wilro_config.py <ckpt-dir-or-config.json> [more...]
    python migrate_wilro_config.py --dry-run outputs/run/checkpoint-*/

Only null-valued keys are dropped, so a missing key falls back to the dataclass
default -- which is exactly what "absent" means to the translation. Nothing
else in the file is touched, and a backup is written next to it.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def repair(path: Path, dry_run: bool) -> bool:
    raw = json.loads(path.read_text())
    nulls = sorted(k for k, v in raw.items() if v is None)
    # Only these are safe to drop: they are the compatibility aliases, and
    # "absent" is a meaning the translation already handles. A null anywhere
    # else is left alone and reported, because dropping it would silently
    # substitute a default for something the run actually chose.
    ALIASES = {
        "robot_ca_source", "use_robot_ca", "robot_vlm_layer_offset",
        "robot_encoder_tokens", "robot_encoder_input_size", "robot_encoder_pool",
        "robot_cnn_cameras", "robot_cnn_fine_cameras", "robot_cnn_fine_tokens",
        "robot_cnn_motion_tokens", "robot_cnn_motion_stride",
        "use_robot_cnn", "gripper_camera", "gripper_encoder_tokens",
    }
    drop = [k for k in nulls if k in ALIASES]
    other = [k for k in nulls if k not in ALIASES]
    if other:
        print(f"  [warn] {path}: null in non-alias field(s) {other}; left as-is")
    if not drop:
        print(f"  ok    {path}: nothing to repair")
        return False
    print(f"  FIX   {path}: dropping {len(drop)} null alias(es) {drop}")
    if dry_run:
        return True
    shutil.copy2(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps({k: v for k, v in raw.items() if k not in drop},
                               indent=4) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="checkpoint dirs or config.json files")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = 0
    for spec in a.paths:
        p = Path(spec)
        files = [p] if p.suffix == ".json" else sorted(p.glob("**/config.json"))
        if not files:
            print(f"  [warn] {p}: no config.json found")
        for f in files:
            n += bool(repair(f, a.dry_run))
    print(f"\n{n} file(s) {'would be' if a.dry_run else ''} repaired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
