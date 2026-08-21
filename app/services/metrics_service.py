"""Read run metrics from artifact directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings


METRICS_FILENAMES = ("metrics.json", "eval_results.json", "train_results.json")


def _find_metrics_files(run_dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for filename in METRICS_FILENAMES:
        for path in run_dir_path.rglob(filename):
            found.append(path)
    return sorted(found)


def collect_metrics(run_id: int) -> Dict[str, Any]:
    """Collect all metrics files under the run directory as a nested dict."""
    root = settings.runs_dir / f"run-{run_id}"
    result: Dict[str, Any] = {"files": []}
    if not root.exists():
        return result
    for path in _find_metrics_files(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rel = str(path.relative_to(root))
        result["files"].append({"path": rel, "metrics": data})
    return result
