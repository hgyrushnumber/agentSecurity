from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NodeCreate(BaseModel):
    name: str = Field(min_length=1)
    hostname: str = ""
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    gpu_info: Optional[str] = None


class NodeUpdate(BaseModel):
    hostname: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: Optional[int] = None
    gpu_info: Optional[str] = None


class NodeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    hostname: str
    ssh_user: Optional[str] = None
    ssh_port: int
    gpu_info: Optional[str] = None
    status: str
    last_heartbeat_at: Optional[datetime] = None
    created_at: datetime
