"""Node registry helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Session, select

from app.models import Node


def register_or_update(
    session: Session,
    name: str,
    hostname: str = "",
    ssh_user: Optional[str] = None,
    ssh_port: int = 22,
    gpu_info: Optional[str] = None,
) -> Node:
    node = session.exec(select(Node).where(Node.name == name)).first()
    now = datetime.utcnow()
    if node is None:
        node = Node(
            name=name,
            hostname=hostname,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            gpu_info=gpu_info,
            status="alive",
            last_heartbeat_at=now,
        )
        session.add(node)
    else:
        node.hostname = hostname or node.hostname
        node.ssh_user = ssh_user if ssh_user is not None else node.ssh_user
        node.ssh_port = ssh_port
        node.gpu_info = gpu_info or node.gpu_info
        node.status = "alive"
        node.last_heartbeat_at = now
        session.add(node)
    session.commit()
    session.refresh(node)
    return node


def heartbeat(session: Session, node_id: int) -> Optional[Node]:
    node = session.get(Node, node_id)
    if node is None:
        return None
    node.last_heartbeat_at = datetime.utcnow()
    node.status = "alive"
    session.add(node)
    session.commit()
    session.refresh(node)
    return node


def mark_offline(session: Session, node_id: int) -> None:
    node = session.get(Node, node_id)
    if node is not None and node.status != "offline":
        node.status = "offline"
        session.add(node)
        session.commit()
