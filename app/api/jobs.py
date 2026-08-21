from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.db import get_session
from app.models import Job
from app.schemas.job import JobRead, LogSlice
from app.services import job_service

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=List[JobRead])
def list_jobs(
    run_id: Optional[int] = None,
    status: Optional[str] = None,
    session: Session = Depends(get_session),
) -> List[Job]:
    statement = select(Job)
    if run_id is not None:
        statement = statement.where(Job.run_id == run_id)
    if status is not None:
        statement = statement.where(Job.status == status)
    statement = statement.order_by(Job.id.desc())
    return list(session.exec(statement).all())


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: int, session: Session = Depends(get_session)) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/{job_id}/logs", response_model=LogSlice)
def job_logs(
    job_id: int,
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> LogSlice:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    text = job_service.read_job_log(job, offset)
    return LogSlice(
        job_id=job_id,
        offset=offset + len(text.encode("utf-8")),
        text=text,
        finished=job.status in {"succeeded", "failed", "cancelled"},
    )


@router.post("/{job_id}/cancel", response_model=JobRead)
def cancel_job(job_id: int, session: Session = Depends(get_session)) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job_service.cancel_job(session, job)
