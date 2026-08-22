#!/usr/bin/env python3
"""Process raw experiment datasets into training-ready files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


XLAM_INPUT = Path("dataset/xlam-function-calling-60k/xlam_function_calling_60k.json")
XLAM_OUTPUT = Path("processed/xlam_tool_count_trigger_1to8.jsonl")
NEMOTRON_STATS = Path("processed/nemotron_uuid_same_tool_success_stats.csv")
NEMOTRON_SPLITS = Path("processed/nemotron_splits")
NEMOTRON_SFT = Path("processed/nemotron_sft")


def run_command(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def process_xlam() -> None:
    print("===== [xlam] generate tool-count-trigger dataset =====", flush=True)
    if not XLAM_INPUT.is_file():
        raise SystemExit(
            f"ERROR: {XLAM_INPUT} not found. See README.md for dataset download commands."
        )
    run_command(
        [
            sys.executable,
            "scripts/generate_tool_count_trigger_dataset.py",
            "--input",
            str(XLAM_INPUT),
            "--output",
            str(XLAM_OUTPUT),
            "--tool-counts",
            "1,2,3,4,5,6,7,8",
            "--threshold",
            "3",
            "--variants-per-count",
            "1",
            "--seed",
            "42",
        ]
    )
    print(f"[ok] -> {XLAM_OUTPUT}", flush=True)


def process_nemotron(parquet: Path | None, stats_csv: Path) -> None:
    print("===== [nemotron] 1/2 split UUIDs =====", flush=True)
    if not stats_csv.is_file():
        raise SystemExit(
            f"ERROR: {stats_csv} not found (UUID-level stats CSV).\n"
            "  Generate it from the dataset parquet first, or pass --stats-csv."
        )
    run_command(
        [
            sys.executable,
            "scripts/split_nemotron_uuids.py",
            "--input",
            str(stats_csv),
            "--output-dir",
            str(NEMOTRON_SPLITS),
            "--train-ratio",
            "0.8",
            "--validation-ratio",
            "0.1",
            "--test-ratio",
            "0.1",
            "--seed",
            "42",
        ]
    )

    print("===== [nemotron] 2/2 build SFT samples =====", flush=True)
    if parquet is None:
        raise SystemExit(
            "ERROR: need --parquet PATH. See README.md for Nemotron download commands."
        )
    if not parquet.is_file():
        raise SystemExit(f"ERROR: parquet not found: {parquet}")
    run_command(
        [
            sys.executable,
            "scripts/build_nemotron_sft.py",
            "--parquet",
            str(parquet),
            "--splits",
            str(NEMOTRON_SPLITS / "all_uuid_splits.csv"),
            "--output-dir",
            str(NEMOTRON_SFT),
            "--threshold",
            "3",
        ]
    )
    print(f"[ok] -> {NEMOTRON_SFT}/", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process raw xlam or nemotron data into training-ready files."
    )
    subparsers = parser.add_subparsers(dest="family", required=True)
    subparsers.add_parser("xlam")

    nemotron = subparsers.add_parser("nemotron")
    nemotron.add_argument("--parquet", type=Path)
    nemotron.add_argument("--stats-csv", type=Path, default=NEMOTRON_STATS)

    args = parser.parse_args()
    if args.family == "xlam":
        process_xlam()
    else:
        process_nemotron(args.parquet, args.stats_csv)
    print("===== done =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
