from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.job import JobRead, JobSpec


class RunCreate(BaseModel):
    experiment_id: int
    name: str = Field(min_length=1)
    config: Dict[str, Any] = Field(default_factory=dict)
    dataset_id: Optional[int] = None
    node_id: Optional[int] = None
    jobs: List[JobSpec] = Field(default_factory=list)


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    name: str
    config_json: str
    config_hash: str
    dataset_id: Optional[int] = None
    node_id: Optional[int] = None
    status: str
    git_commit: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class RunDetail(RunRead):
    jobs: List[JobRead] = Field(default_factory=list)
