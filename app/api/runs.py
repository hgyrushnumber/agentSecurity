from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import Experiment, Job, Run
from app.schemas.job import JobRead
from app.schemas.run import RunCreate, RunDetail, RunRead
from app.services import job_service, metrics_service, run_service

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", response_model=RunDetail, status_code=201)
def create_run(
    payload: RunCreate,
    session: Session = Depends(get_session),
) -> RunDetail:
    if session.get(Experiment, payload.experiment_id) is None:
        raise HTTPException(status_code=404, detail="experiment not found")

    frozen_config = run_service.canonical_config_json(payload.config)
    run = Run(
        experiment_id=payload.experiment_id,
        name=payload.name,
        config_json=frozen_config,
        config_hash=run_service.config_hash(payload.config),
        dataset_id=payload.dataset_id,
        node_id=payload.node_id,
        status="created",
        git_commit=run_service.current_git_commit(),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    run_service.write_run_config(run)
    run_service.run_dir(run.id)

    specs = [job.model_dump() for job in payload.jobs]
    jobs = job_service.create_jobs(session, run, specs) if specs else []

    run_detail = RunDetail.model_validate(run, from_attributes=True)
    run_detail.jobs = [JobRead.model_validate(job, from_attributes=True) for job in jobs]
    return run_detail


@router.get("", response_model=List[RunRead])
def list_runs(
    experiment_id: Optional[int] = None,
    session: Session = Depends(get_session),
) -> List[Run]:
    statement = select(Run)
    if experiment_id is not None:
        statement = statement.where(Run.experiment_id == experiment_id)
    statement = statement.order_by(Run.id.desc())
    return list(session.exec(statement).all())


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: int, session: Session = Depends(get_session)) -> RunDetail:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    run_detail = RunDetail.model_validate(run, from_attributes=True)
    run_detail.jobs = [
        JobRead.model_validate(job, from_attributes=True)
        for job in session.exec(select(Job).where(Job.run_id == run_id).order_by(Job.id)).all()
    ]
    return run_detail


@router.get("/{run_id}/jobs", response_model=List[Job])
def list_run_jobs(run_id: int, session: Session = Depends(get_session)) -> List[Job]:
    if session.get(Run, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return list(session.exec(select(Job).where(Job.run_id == run_id).order_by(Job.id)).all())


@router.get("/{run_id}/metrics")
def run_metrics(run_id: int, session: Session = Depends(get_session)):
    if session.get(Run, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return metrics_service.collect_metrics(run_id)
