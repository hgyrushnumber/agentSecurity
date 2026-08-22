from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Job(SQLModel, table=True):
    """An executable unit of a run (e.g. train / eval / data stage)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    stage: str = "task"  # e.g. train | eval | data
    command: str = Field(index=True)
    workdir: Optional[str] = None
    status: str = "queued"  # queued|running|succeeded|failed|cancelled
    node_id: Optional[int] = Field(default=None, foreign_key="node.id")
    pid: Optional[int] = None
    log_path: Optional[str] = None
    exit_code: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
