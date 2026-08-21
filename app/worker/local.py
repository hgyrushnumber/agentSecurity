"""Local subprocess worker.

Polls the job table for queued jobs (node_id is NULL or matches LOCAL_NODE),
runs them as subprocesses, streams output into logs/jobs/job-{id}.log and
updates status. Run with:  python -m app.worker.local
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import or_
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import Job, Run
from app.services.run_service import refresh_run_status

LOCAL_NODE_NAME = os.environ.get("AGENTSEC_LOCAL_NODE", "local")
LOCAL_NODE_ID = os.environ.get("AGENTSEC_LOCAL_NODE_ID")


def _log(job: Job, message: str) -> None:
    """Append a worker-side line to the job log file."""
    if not job.log_path:
        return
    try:
        Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(job.log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def _claim_job(session: Session, running_count: int) -> Optional[Job]:
    """Find the next queued job this worker may run (respecting parallel cap).

    With AGENTSEC_LOCAL_NODE_ID set, claims jobs targeting that node or with no
    node. Without it, only claims jobs with no node assignment.
    """
    if running_count >= settings.worker_max_parallel:
        return None
    statement = select(Job).where(Job.status == "queued")
    if LOCAL_NODE_ID is not None:
        statement = statement.where(
            or_(Job.node_id.is_(None), Job.node_id == int(LOCAL_NODE_ID))
        )
    else:
        statement = statement.where(Job.node_id.is_(None))
    statement = statement.order_by(Job.id)
    jobs = list(session.exec(statement).all())
    return jobs[0] if jobs else None


def _execute(session: Session, job: Job) -> None:
    """Run one job to completion and persist the outcome."""
    job.status = "running"
    job.started_at = datetime.utcnow()
    session.add(job)
    session.commit()
    session.refresh(job)

    _log(job, f"=== job {job.id} ({job.stage}) started at "
             f"{job.started_at.isoformat()} ===")
    _log(job, f"$ {job.command}")

    log_handle = open(job.log_path, "a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            job.command,
            shell=True,
            cwd=job.workdir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        job.pid = process.pid
        session.add(job)
        session.commit()

        # Wait, polling for cancellation requests.
        while process.poll() is None:
            session.refresh(job)
            if job.status == "cancelled":
                _terminate_process(process)
                break
            time.sleep(1.0)

        exit_code = process.wait()
        session.refresh(job)
        if job.status == "cancelled":
            job.exit_code = None
        else:
            job.exit_code = exit_code
            job.status = "succeeded" if exit_code == 0 else "failed"
        job.finished_at = datetime.utcnow()
        session.add(job)
        session.commit()
        _log(
            job,
            f"=== job {job.id} finished: status={job.status} "
            f"exit_code={job.exit_code} ===",
        )
    finally:
        log_handle.close()
        session.refresh(job)
        run = session.get(Run, job.run_id)
        if run is not None:
            refresh_run_status(session, run)
            session.commit()


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass


def _reap_cancelled(session: Session) -> None:
    """Kill running jobs whose cancel was requested through the API."""
    running = list(session.exec(select(Job).where(Job.status == "running")).all())
    for job in running:
        if job.pid is None:
            continue
        try:
            os.kill(job.pid, 0)  # still alive?
        except OSError:
            continue
        # alive but worker-side loop already handles cancellation; nothing to do here
    queued = list(session.exec(select(Job).where(Job.status == "cancelled", Job.pid.isnot(None))).all())
    for job in queued:
        if job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
            except OSError:
                pass


def run_once(session: Session) -> None:
    """One poll iteration; returns after claiming at most one job."""
    running = list(session.exec(select(Job).where(Job.status == "running")).all())
    job = _claim_job(session, len(running))
    if job is not None:
        _execute(session, job)


def main_loop() -> None:
    print(f"[worker:{LOCAL_NODE_NAME}] poll interval {settings.worker_poll_interval}s", flush=True)
    while True:
        try:
            with Session(engine) as session:
                run_once(session)
        except KeyboardInterrupt:
            print("[worker] interrupted", flush=True)
            break
        except Exception as exc:  # keep the loop alive
            print(f"[worker] error: {exc!r}", flush=True)
        time.sleep(settings.worker_poll_interval)


if __name__ == "__main__":
    main_loop()
