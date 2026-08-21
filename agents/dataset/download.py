#!/usr/bin/env python3
"""Download experiment datasets from HuggingFace.

Requires: pip install huggingface_hub

Usage:
    python -m agents.dataset.download xlam
    python -m agents.dataset.download nemotron [--target DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


XLAM_REPO = "Salesforce/xlam-function-calling-60k"
NEMOTRON_REPO = "nvidia/Nemotron-Agentic-v1"

DEFAULT_XLAM_DIR = "raw/xlam-function-calling-60k"
DEFAULT_NEMOTRON_DIR = "raw/nemotron_agentic_v1"


def _snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub 未安装，请先执行: pip install huggingface_hub"
        ) from exc
    return snapshot_download


def download_dataset(
    repo_id: str,
    local_dir: Path,
    allow_patterns: Optional[List[str]] = None,
) -> Path:
    """Download a HF dataset repo into local_dir; returns local_dir."""
    snapshot = _snapshot_download()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    return local_dir


def find_parquet(local_dir: Path) -> List[Path]:
    return sorted(local_dir.rglob("*.parquet"))


def main() -> int:
    parser = argparse.ArgumentParser(description="下载实验数据集（HuggingFace）")
    parser.add_argument(
        "name",
        choices=["xlam", "nemotron"],
        help="数据集名称：xlam（Salesforce/xlam-function-calling-60k）或 nemotron（nvidia/Nemotron-Agentic-v1）",
    )
    parser.add_argument("--target", type=Path, default=None, help="下载目标目录")
    args = parser.parse_args()

    if args.name == "xlam":
        target = args.target or Path(DEFAULT_XLAM_DIR)
        download_dataset(XLAM_REPO, target)
        print(f"[ok] xlam 数据集已下载到 {target}")
        print("提示：count_xlam_tools.py / generate_tool_count_trigger_dataset.py")
        print(f"  期望的输入文件为 {target / 'xlam_function_calling_60k.json'}")
    else:
        target = args.target or Path(DEFAULT_NEMOTRON_DIR)
        download_dataset(NEMOTRON_REPO, target)
        print(f"[ok] Nemotron 数据集已下载到 {target}")
        parquet_files = find_parquet(target)
        if parquet_files:
            print(f"找到 {len(parquet_files)} 个 parquet 文件，例如：")
            for path in parquet_files[:5]:
                print(f"  {path}")
            print("build_nemotron_sft.py 用法示例：")
            print(
                f"  python scripts/build_nemotron_sft.py "
                f"--parquet {parquet_files[0]} "
                f"--splits processed/nemotron_splits.csv "
                f"--output-dir processed/nemotron_sft"
            )
        else:
            print("未找到 .parquet 文件，请用 --target 指定实际下载目录后检查结构。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
