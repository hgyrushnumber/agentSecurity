from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    format: str = "jsonl"
    rows: Optional[int] = None
    sha256: Optional[str] = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    path: str
    format: str
    rows: Optional[int] = None
    sha256: Optional[str] = None
    created_at: datetime
