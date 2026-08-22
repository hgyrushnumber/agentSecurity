from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Node(SQLModel, table=True):
    """A GPU server that can execute jobs."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    hostname: str = ""
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    gpu_info: Optional[str] = None
    status: str = "unknown"  # unknown | alive | offline
    last_heartbeat_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
