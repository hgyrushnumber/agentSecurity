from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import Dataset
from app.schemas.dataset import DatasetCreate, DatasetRead

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=List[DatasetRead])
def list_datasets(session: Session = Depends(get_session)) -> List[Dataset]:
    return list(session.exec(select(Dataset).order_by(Dataset.id.desc())).all())


@router.post("", response_model=DatasetRead, status_code=201)
def create_dataset(
    payload: DatasetCreate,
    session: Session = Depends(get_session),
) -> Dataset:
    dataset = Dataset(**payload.model_dump())
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


@router.get("/{dataset_id}", response_model=DatasetRead)
def get_dataset(
    dataset_id: int,
    session: Session = Depends(get_session),
) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return dataset
