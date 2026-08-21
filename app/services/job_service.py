"""Job management: creation, cancellation, log paths."""

from __future__ import annotations

import os
from typing import List, Optional

from sqlmodel import Session

from app.config import settings
from app.models import Job, Run
from app.services.run_service import refresh_run_status, run_dir


def job_log_path(job_id: int) -> str:
    path = settings.logs_dir / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return str(path / f"job-{job_id}.log")


def create_jobs(
    session: Session,
    run: Run,
    specs: List[dict],
) -> List[Job]:
    """Create queued jobs for a run (specs are validated dicts)."""
    jobs: List[Job] = []
    workdir_default = run_dir(run.id) if run.id else None
    for spec in specs:
        job = Job(
            run_id=run.id,
            stage=spec.get("stage", "task"),
            command=spec["command"],
            workdir=spec.get("workdir") or workdir_default,
            node_id=spec.get("node_id") or run.node_id,
            status="queued",
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job.log_path = job_log_path(job.id)
        session.add(job)
        jobs.append(job)
    session.commit()
    refresh_run_status(session, run)
    session.commit()
    return jobs


def cancel_job(session: Session, job: Job) -> Job:
    """Request cancellation; the worker kills the process on next poll."""
    if job.status in {"running", "queued"}:
        job.status = "cancelled"
        if job.pid is not None:
            _kill_pid(job.pid)
        session.add(job)
        session.commit()
        run = session.get(Run, job.run_id)
        if run is not None:
            refresh_run_status(session, run)
            session.commit()
    return job


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, 15)  # SIGTERM
    except (OSError, ProcessLookupError):
        pass


def read_job_log(job: Job, offset: int = 0) -> str:
    """Read log content starting at byte offset (None-safe)."""
    if not job.log_path or not os.path.exists(job.log_path):
        return ""
    try:
        with open(job.log_path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            return handle.read()
    except OSError:
        return ""
