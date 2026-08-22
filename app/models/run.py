from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Run(SQLModel, table=True):
    """One experiment run: frozen config snapshot + provenance."""

    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id", index=True)
    name: str = Field(index=True)
    config_json: str = Field(default="{}")  # frozen config snapshot
    config_hash: str = Field(index=True)
    dataset_id: Optional[int] = Field(default=None, foreign_key="dataset.id")
    node_id: Optional[int] = Field(default=None, foreign_key="node.id")
    status: str = "created"  # created|queued|running|succeeded|failed|cancelled
    git_commit: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
