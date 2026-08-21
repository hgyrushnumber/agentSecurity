"""Run lifecycle: creation, config hashing, status aggregation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from app.config import settings
from app.models import Job, Run


def canonical_config_json(config: Dict[str, Any]) -> str:
    """Deterministic JSON serialization of a frozen config (sorted keys)."""
    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_config_json(config).encode("utf-8")
    ).hexdigest()[:16]


def current_git_commit() -> Optional[str]:
    """Best-effort HEAD commit of the repository containing this package."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def run_dir(run_id: int) -> str:
    path = settings.runs_dir / f"run-{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def write_run_config(run: Run) -> None:
    """Persist the frozen config next to the run artifacts."""
    path = settings.runs_dir / f"run-{run.id}" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.config_json, encoding="utf-8")


def aggregate_run_status(jobs: List[Job]) -> str:
    """Derive run status from its jobs (running > queued > failed/cancelled > succeeded)."""
    if not jobs:
        return "created"
    statuses = {job.status for job in jobs}
    if "running" in statuses:
        return "running"
    if "queued" in statuses:
        return "queued"
    if "failed" in statuses:
        return "failed"
    if "cancelled" in statuses:
        return "cancelled"
    if statuses == {"succeeded"}:
        return "succeeded"
    return "created"


def refresh_run_status(session: Session, run: Run) -> None:
    """Recompute run.status from its jobs and touch start/finish timestamps."""
    jobs = list(session.exec(select(Job).where(Job.run_id == run.id)).all())
    run.status = aggregate_run_status(jobs)
    now = datetime.utcnow()
    if run.status in {"running", "queued"} and run.started_at is None:
        run.started_at = now
    if run.status in {"succeeded", "failed", "cancelled"}:
        run.finished_at = now
    session.add(run)
