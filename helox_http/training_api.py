"""
HTTP training trigger for deepiri-jobs helox.train -> POST /training/runs.

This is currently a contract-validation scaffold: it proves the HTTP surface,
run registry, and Synapse training event flow before the endpoint is wired to
Helox's production UnifiedTrainingOrchestrator.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from deepiri_training_orchestrator.orchestrator import TrainingOrchestrator
from deepiri_training_orchestrator.reproducibility import ReproducibilityController
from integrations.synapse_event_publisher import SynapseEventPublisher

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])

_publisher = SynapseEventPublisher()
_runs: Dict[str, Dict[str, Any]] = {}
_CONTRACT_VALIDATION_MODE = "contract_validation"


class TrainingRunRequest(BaseModel):
    jobId: Optional[str] = None
    model_name: str = "helox-default"
    max_steps: int = Field(default=10, ge=1, le=10_000)
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainingRunResponse(BaseModel):
    run_id: str
    job_id: Optional[str] = None
    status: str


def _noop_batches(max_steps: int):
    """Yield deterministic placeholder batches for HTTP/event contract validation."""
    for i in range(max_steps):
        yield {"step": i, "loss": 1.0 / (i + 1)}


async def _execute_run(run_id: str, req: TrainingRunRequest) -> None:
    run = _runs[run_id]
    run["status"] = "running"
    run["mode"] = _CONTRACT_VALIDATION_MODE
    model = req.model_name
    try:
        await _publisher.publish_training_event(
            "training.started",
            model,
            0,
            metrics={"run_id": run_id, "mode": _CONTRACT_VALIDATION_MODE},
            jobId=req.jobId,
        )
        repro = ReproducibilityController(seed=int(req.config.get("seed", 42)))
        repro.set_seeds()

        # TODO: replace this contract-validation shim with UnifiedTrainingOrchestrator
        # once deepiri-jobs passes the approved dataset/model/config contract.
        orch = TrainingOrchestrator(
            config=req.config,
            reproducibility=repro,
            max_steps=req.max_steps,
            log_every=max(1, req.max_steps // 10),
            run_name=run_id,
        )

        def train_step(step: int, batch: Dict[str, Any]) -> Dict[str, float]:
            return {"loss": float(batch.get("loss", 0.0))}

        orch.fit(_noop_batches(req.max_steps), train_step=train_step)
        run["status"] = "completed"
        await _publisher.publish_training_event(
            "training.completed",
            model,
            req.max_steps,
            metrics={"run_id": run_id, "mode": _CONTRACT_VALIDATION_MODE},
            jobId=req.jobId,
        )
    except Exception as exc:
        logger.exception("training run failed run_id=%s", run_id)
        run["status"] = "failed"
        run["error"] = str(exc)
        await _publisher.publish_training_event(
            "training.failed",
            model,
            0,
            metrics={"run_id": run_id, "mode": _CONTRACT_VALIDATION_MODE},
            error=str(exc),
            jobId=req.jobId,
        )


@router.post("/runs", response_model=TrainingRunResponse)
async def create_training_run(
    body: TrainingRunRequest,
    background_tasks: BackgroundTasks,
) -> TrainingRunResponse:
    run_id = body.jobId or f"run_{uuid.uuid4().hex[:12]}"
    _runs[run_id] = {
        "run_id": run_id,
        "job_id": body.jobId,
        "status": "queued",
        "mode": _CONTRACT_VALIDATION_MODE,
        "request": body.model_dump(),
    }
    background_tasks.add_task(_execute_run, run_id, body)
    return TrainingRunResponse(run_id=run_id, job_id=body.jobId, status="queued")


@router.get("/runs/{run_id}")
async def get_training_run(run_id: str) -> Dict[str, Any]:
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
