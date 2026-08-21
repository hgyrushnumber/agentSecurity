from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import Node
from app.schemas.node import NodeCreate, NodeRead, NodeUpdate
from app.services import node_service

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


@router.get("", response_model=List[NodeRead])
def list_nodes(session: Session = Depends(get_session)) -> List[Node]:
    return list(session.exec(select(Node).order_by(Node.id.desc())).all())


@router.post("", response_model=NodeRead, status_code=201)
def register_node(
    payload: NodeCreate,
    session: Session = Depends(get_session),
) -> Node:
    return node_service.register_or_update(
        session,
        name=payload.name,
        hostname=payload.hostname,
        ssh_user=payload.ssh_user,
        ssh_port=payload.ssh_port,
        gpu_info=payload.gpu_info,
    )


@router.get("/{node_id}", response_model=NodeRead)
def get_node(node_id: int, session: Session = Depends(get_session)) -> Node:
    node = session.get(Node, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="node not found")
    return node


@router.patch("/{node_id}", response_model=NodeRead)
def update_node(
    node_id: int,
    payload: NodeUpdate,
    session: Session = Depends(get_session),
) -> Node:
    node = session.get(Node, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="node not found")
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(node, key, value)
    session.add(node)
    session.commit()
    session.refresh(node)
    return node


@router.post("/{node_id}/heartbeat", response_model=NodeRead)
def heartbeat(node_id: int, session: Session = Depends(get_session)) -> Node:
    node = node_service.heartbeat(session, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="node not found")
    return node
