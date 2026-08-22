from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Dataset(SQLModel, table=True):
    """A registered dataset (path + provenance)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    path: str = Field(index=True)
    format: str = "jsonl"
    rows: Optional[int] = None
    sha256: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
