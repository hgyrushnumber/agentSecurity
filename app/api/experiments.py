from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import Experiment, Run
from app.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentUpdate
from app.schemas.run import RunRead

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("", response_model=List[ExperimentRead])
def list_experiments(session: Session = Depends(get_session)) -> List[Experiment]:
    return list(session.exec(select(Experiment).order_by(Experiment.id.desc())).all())


@router.post("", response_model=ExperimentRead, status_code=201)
def create_experiment(
    payload: ExperimentCreate,
    session: Session = Depends(get_session),
) -> Experiment:
    experiment = Experiment(name=payload.name, description=payload.description)
    session.add(experiment)
    session.commit()
    session.refresh(experiment)
    return experiment


@router.get("/{experiment_id}", response_model=ExperimentRead)
def get_experiment(
    experiment_id: int,
    session: Session = Depends(get_session),
) -> Experiment:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return experiment


@router.patch("/{experiment_id}", response_model=ExperimentRead)
def update_experiment(
    experiment_id: int,
    payload: ExperimentUpdate,
    session: Session = Depends(get_session),
) -> Experiment:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    if payload.name is not None:
        experiment.name = payload.name
    if payload.description is not None:
        experiment.description = payload.description
    session.add(experiment)
    session.commit()
    session.refresh(experiment)
    return experiment


@router.delete("/{experiment_id}", status_code=204)
def delete_experiment(
    experiment_id: int,
    session: Session = Depends(get_session),
) -> None:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    session.delete(experiment)
    session.commit()


@router.get("/{experiment_id}/runs", response_model=List[RunRead])
def list_experiment_runs(
    experiment_id: int,
    session: Session = Depends(get_session),
) -> List[Run]:
    if session.get(Experiment, experiment_id) is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return list(
        session.exec(
            select(Run).where(Run.experiment_id == experiment_id).order_by(Run.id.desc())
        ).all()
    )
