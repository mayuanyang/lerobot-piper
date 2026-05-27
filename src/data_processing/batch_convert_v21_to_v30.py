#!/usr/bin/env python
"""
Batch convert all sub-datasets in community_dataset_v3 from LeRobot v2.1 to v3.0 format.

This script iterates over every sub-dataset (contributor/dataset_name) inside the
community_dataset_v3 directory and runs the official lerobot v2.1→v3.0 conversion
on each one locally (without pushing to HuggingFace).

Usage:
    # First, download the dataset (758 GB total; consider filtering if needed):
    huggingface-cli download HuggingFaceVLA/community_dataset_v3 \
        --repo-type=dataset \
        --local-dir /path/to/community_dataset_v3

    # Then run the batch conversion:
    python src/data_processing/batch_convert_v21_to_v30.py \
        --input-dir /path/to/community_dataset_v3

    # Dry-run to see what would be converted:
    python src/data_processing/batch_convert_v21_to_v30.py \
        --input-dir /path/to/community_dataset_v3 \
        --dry-run

    # Convert only a specific contributor:
    python src/data_processing/batch_convert_v21_to_v30.py \
        --input-dir /path/to/community_dataset_v3 \
        --filter-contributor shuohsuan

    # Convert only specific datasets (comma-separated, supports glob patterns):
    python src/data_processing/batch_convert_v21_to_v30.py \
        --input-dir /path/to/community_dataset_v3 \
        --filter-dataset "pick_and_place,sorting_*"

    # Resume from a specific sub-dataset (skip already converted):
    python src/data_processing/batch_convert_v21_to_v30.py \
        --input-dir /path/to/community_dataset_v3 \
        --resume
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import lerobot

# Dynamically resolve the lerobot conversion script path (works on any machine)
_CONVERT_SCRIPT_PATH = Path(lerobot.__file__).parent / "datasets" / "v30" / "convert_dataset_v21_to_v30.py"
CONVERT_SCRIPT = _CONVERT_SCRIPT_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_sub_datasets(input_dir: Path) -> list[Path]:
    """
    Find all sub-dataset directories under input_dir.

    A sub-dataset is identified by the presence of meta/info.json.
    Expected structure: input_dir/contributor_name/dataset_name/meta/info.json
    """
    sub_datasets = []
    for info_path in sorted(input_dir.glob("*/*/meta/info.json")):
        dataset_dir = info_path.parent.parent  # contributor/dataset_name
        sub_datasets.append(dataset_dir)
    return sub_datasets


def get_codebase_version(dataset_dir: Path) -> Optional[str]:
    """Read the codebase_version from meta/info.json."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            info = json.load(f)
        return info.get("codebase_version")
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read {info_path}: {e}")
        return None


def convert_sub_dataset(dataset_dir: Path, dry_run: bool = False) -> bool:
    """
    Convert a single sub-dataset from v2.1 to v3.0 using the lerobot conversion script.

    Uses the trick: --repo-id . --root /path/to/dataset_dir
    so that Path(root) / repo_id == dataset_dir itself.
    """
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--repo-id", ".",
        "--root", str(dataset_dir),
        "--push-to-hub", "false",
        "--force-conversion",
    ]

    rel_path = str(dataset_dir)

    if dry_run:
        logger.info(f"[DRY-RUN] Would convert: {rel_path}")
        return True

    logger.info(f"Converting: {rel_path}")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per sub-dataset
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✓ Converted {rel_path} in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"✗ Failed to convert {rel_path} (exit code {result.returncode})")
            if result.stderr:
                # Print last 10 lines of stderr for debugging
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-10:]:
                    logger.error(f"  stderr: {line}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout converting {rel_path} (>1 hour)")
        return False
    except Exception as e:
        logger.error(f"✗ Exception converting {rel_path}: {e}")
        return False


def should_process(dataset_dir: Path, contributor_filter: str, dataset_filter: str) -> bool:
    """Check if a sub-dataset should be processed based on filters."""
    rel = dataset_dir.relative_to(dataset_dir.parent.parent)
    parts = rel.parts  # (contributor, dataset_name)

    if contributor_filter and parts[0] != contributor_filter:
        return False

    if dataset_filter:
        import fnmatch
        patterns = [p.strip() for p in dataset_filter.split(",")]
        if not any(fnmatch.fnmatch(parts[1], p) for p in patterns):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert community_dataset_v3 sub-datasets from v2.1 to v3.0"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the root of community_dataset_v3 (containing contributor directories).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be converted without actually converting.",
    )
    parser.add_argument(
        "--filter-contributor",
        type=str,
        default=None,
        help="Only process sub-datasets from this contributor.",
    )
    parser.add_argument(
        "--filter-dataset",
        type=str,
        default=None,
        help="Only process datasets matching these comma-separated patterns (supports * glob).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sub-datasets that already have codebase_version v3.0.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=True,
        help="Continue processing remaining sub-datasets even if some fail. (default: True)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on first conversion error.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if not CONVERT_SCRIPT.exists():
        logger.error(f"Conversion script not found: {CONVERT_SCRIPT}")
        logger.error("Make sure lerobot is installed: pip install lerobot")
        sys.exit(1)

    # Find all sub-datasets
    all_sub_datasets = find_sub_datasets(input_dir)
    logger.info(f"Found {len(all_sub_datasets)} sub-datasets in {input_dir}")

    if not all_sub_datasets:
        logger.warning("No sub-datasets found! Check the directory structure.")
        logger.warning("Expected: input_dir/contributor_name/dataset_name/meta/info.json")
        sys.exit(1)

    # Filter and classify
    v21_datasets = []
    v30_datasets = []
    unknown_datasets = []
    skipped_by_filter = []

    for ds_dir in all_sub_datasets:
        if not should_process(ds_dir, args.filter_contributor, args.filter_dataset):
            skipped_by_filter.append(ds_dir)
            continue

        version = get_codebase_version(ds_dir)
        if version == "v2.1":
            v21_datasets.append(ds_dir)
        elif version == "v3.0":
            v30_datasets.append(ds_dir)
        else:
            unknown_datasets.append((ds_dir, version))

    # Print summary
    logger.info("=" * 60)
    logger.info(f"Total sub-datasets found:     {len(all_sub_datasets)}")
    logger.info(f"  v2.1 (will convert):        {len(v21_datasets)}")
    logger.info(f"  v3.0 (already converted):   {len(v30_datasets)}")
    logger.info(f"  Unknown version:            {len(unknown_datasets)}")
    logger.info(f"  Skipped by filter:          {len(skipped_by_filter)}")
    logger.info("=" * 60)

    if unknown_datasets:
        logger.warning("Datasets with unknown codebase_version:")
        for ds_dir, version in unknown_datasets[:10]:
            logger.warning(f"  {ds_dir.relative_to(input_dir)} -> {version}")
        if len(unknown_datasets) > 10:
            logger.warning(f"  ... and {len(unknown_datasets) - 10} more")

    if args.resume and v30_datasets:
        logger.info(f"Skipping {len(v30_datasets)} already-converted v3.0 datasets (--resume)")

    if not v21_datasets:
        logger.info("No v2.1 datasets to convert. Done!")
        return

    logger.info(f"\nStarting conversion of {len(v21_datasets)} datasets...")
    logger.info("-" * 60)

    success_count = 0
    fail_count = 0
    failed_datasets = []

    for i, ds_dir in enumerate(v21_datasets, 1):
        rel_path = ds_dir.relative_to(input_dir)
        logger.info(f"\n[{i}/{len(v21_datasets)}] {rel_path}")

        ok = convert_sub_dataset(ds_dir, dry_run=args.dry_run)

        if ok:
            success_count += 1
        else:
            fail_count += 1
            failed_datasets.append(rel_path)
            if args.stop_on_error:
                logger.error("Stopping due to --stop-on-error")
                break

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed:     {fail_count}")
    if v30_datasets:
        logger.info(f"  Skipped (already v3.0): {len(v30_datasets)}")
    logger.info("=" * 60)

    if failed_datasets:
        logger.info("\nFailed datasets:")
        for ds in failed_datasets:
            logger.info(f"  - {ds}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()