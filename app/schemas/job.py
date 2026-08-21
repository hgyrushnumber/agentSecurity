from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobSpec(BaseModel):
    """A job to attach to a run: command plus optional metadata."""

    stage: str = "task"
    command: str = Field(min_length=1)
    workdir: Optional[str] = None
    node_id: Optional[int] = None


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    stage: str
    command: str
    workdir: Optional[str] = None
    status: str
    node_id: Optional[int] = None
    pid: Optional[int] = None
    log_path: Optional[str] = None
    exit_code: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class LogSlice(BaseModel):
    job_id: int
    offset: int
    text: str
    finished: bool
