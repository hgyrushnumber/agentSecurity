"""FastAPI entrypoint for the agentSecurity control plane."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, experiments, jobs, nodes, runs
from app.config import settings
from app.db import init_db


def create_app() -> FastAPI:
    app = FastAPI(
        title="agentSecurity Control Plane",
        version="0.1.0",
        description="Experiment / run / job management for agent SFT experiments.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        init_db()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def root() -> dict:
        return {
            "service": "agentSecurity control plane",
            "docs": "/docs",
            "health": "/health",
        }

    app.include_router(experiments.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
    app.include_router(nodes.router)
    app.include_router(datasets.router)
    return app


app = create_app()
