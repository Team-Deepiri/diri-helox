"""
Central bridge between Helox training code and shared libraries.

Consolidates imports from deepiri-training-orchestrator and deepiri-modelkit
so pipelines use one integration surface.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from deepiri_modelkit import (
    TrainingRunContext,
    create_run_context,
    emit_training_lifecycle_event,
    pipeline_metadata,
    register_model_ready,
    validate_manifest_against_path,
    write_manifest,
    get_logger,
)
from deepiri_training_orchestrator import (
    DistributedConfig,
    ExperimentTracker,
    HFTrainingAdapter,
    LoggingCallback,
    ReproducibilityController,
    TorchCheckpointCallback,
    TrainingOrchestrator,
    TrainingRunConfig,
    build_dataset_manifest,
    init_distributed,
    prepare_training_run,
    provenance_from_manifest,
)
from deepiri_modelkit.training.job_queue import TrainingJobQueue
from deepiri_modelkit.contracts.training import TrainingPriority, TrainingRunRequest

logger = get_logger("helox.training_bridge")


def make_run_context(
    experiment_id: str,
    model_name: str,
    *,
    fingerprint: Optional[str] = None,
    correlation_id: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> TrainingRunContext:
    """Create a modelkit TrainingRunContext for Helox pipelines."""
    return create_run_context(
        experiment_id,
        model_name,
        source="helox",
        fingerprint=fingerprint,
        correlation_id=correlation_id,
        manifest_path=manifest_path,
    )


def prepare_training_dataset(
    dataset_path: str | Path,
    *,
    preset: str = "training",
    eval_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Prepare a dataset via orchestrator prepare_training_run preset."""
    prepared = prepare_training_run(
        dataset_path,
        preset=preset,  # type: ignore[arg-type]
        eval_path=eval_path,
    )
    validation = validate_manifest_against_path(prepared.manifest, str(prepared.path))
    if not validation.get("valid"):
        logger.warning("dataset_manifest_validation_failed", report=validation)
    return {
        "path": prepared.path,
        "manifest": prepared.manifest,
        "provenance": prepared.provenance,
        "validation_report": validation,
        "quality_report": prepared.quality_report,
    }


def persist_manifest(manifest: Any, output_dir: str | Path) -> Path:
    """Write manifest JSON via modelkit."""
    return write_manifest(manifest, Path(output_dir) / f"{manifest.id}.manifest.json")


def build_run_config(
    *,
    max_steps: int = 1000,
    seed: int = 1337,
    hyperparameters: Optional[Dict[str, Any]] = None,
    tracking_uri: Optional[str] = None,
    use_wandb: bool = False,
    correlation_id: Optional[str] = None,
    dataset_provenance: Optional[Any] = None,
) -> TrainingRunConfig:
    """Build orchestrator TrainingRunConfig from Helox-style kwargs."""
    from deepiri_training_orchestrator.config import TrackingConfig

    return TrainingRunConfig(
        seed=seed,
        max_steps=max_steps,
        correlation_id=correlation_id,
        hyperparameters=hyperparameters or {},
        dataset=dataset_provenance,
        tracking=TrackingConfig(
            mlflow_uri=tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
            use_wandb=use_wandb,
        ),
    )


def create_experiment_tracker(
    experiment_name: str,
    *,
    tracking_uri: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "deepiri",
) -> ExperimentTracker:
    """Create orchestrator ExperimentTracker with Helox defaults."""
    return ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        use_wandb=use_wandb,
        wandb_project=wandb_project,
    )


def create_orchestrator(
    config: Dict[str, Any],
    *,
    reproducibility: Optional[ReproducibilityController] = None,
    experiment_tracker: Optional[ExperimentTracker] = None,
    max_steps: Optional[int] = None,
    log_every: int = 50,
    eval_every: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    callbacks: Optional[List[Any]] = None,
    run_name: Optional[str] = None,
    run_config: Optional[TrainingRunConfig] = None,
    dataset_provenance: Optional[Any] = None,
    correlation_id: Optional[str] = None,
) -> TrainingOrchestrator:
    """Create TrainingOrchestrator with standard Helox callbacks."""
    repro = reproducibility or ReproducibilityController()
    cb = callbacks or []
    if checkpoint_dir:
        cb = list(cb) + [
            TorchCheckpointCallback(Path(checkpoint_dir)),
            LoggingCallback(),
        ]
    return TrainingOrchestrator(
        config=config,
        reproducibility=repro,
        max_steps=max_steps or (run_config.max_steps if run_config else 1000),
        log_every=log_every,
        eval_every=eval_every,
        experiment_tracker=experiment_tracker,
        callbacks=cb or None,
        run_name=run_name,
        run_config=run_config,
        dataset_provenance=dataset_provenance,
        correlation_id=correlation_id,
    )


def setup_distributed(
    *,
    mixed_precision: Optional[str] = None,
    local_rank: Optional[int] = None,
) -> Any:
    """Initialize distributed training via orchestrator (falls back gracefully)."""
    try:
        cfg = DistributedConfig(
            local_rank=local_rank if local_rank is not None else int(os.getenv("LOCAL_RANK", "0")),
            world_size=int(os.getenv("WORLD_SIZE", "1")),
        )
        return init_distributed(cfg, mixed_precision=mixed_precision)
    except Exception as exc:
        logger.warning("distributed_init_failed", error=str(exc))
        return None


def publish_lifecycle(
    ctx: TrainingRunContext,
    event_type: str,
    *,
    status: str = "running",
    progress: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Optional[str]:
    """Publish training lifecycle event via modelkit."""
    return emit_training_lifecycle_event(
        event_type,
        ctx,
        status=status,
        progress=progress,
        metrics=metrics,
        error=error,
        redis_url=os.getenv("REDIS_URL"),
    )


def register_trained_model(
    registry_client: Any,
    model_name: str,
    version: str,
    model_path: str,
    ctx: TrainingRunContext,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Register model and publish model-ready via modelkit."""
    metadata = pipeline_metadata(ctx, extra_metadata)
    return register_model_ready(
        registry_client,
        model_name,
        version,
        model_path,
        metadata,
        source="helox",
        correlation_id=ctx.correlation_id,
        model_type=extra_metadata.get("model_type") if extra_metadata else None,
        accuracy=extra_metadata.get("accuracy") if extra_metadata else None,
        redis_url=os.getenv("REDIS_URL"),
    )


def create_hf_orchestrator(
    trainer: Any,
    *,
    experiment_tracker: Optional[ExperimentTracker] = None,
    max_steps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    run_config: Optional[TrainingRunConfig] = None,
) -> tuple[TrainingOrchestrator, HFTrainingAdapter]:
    """Build TrainingOrchestrator + HFTrainingAdapter for a HF Trainer instance."""
    adapter = HFTrainingAdapter(trainer)
    repro = ReproducibilityController()
    repro.set_seeds()
    max_s = max_steps or getattr(trainer.args, "max_steps", 1000)
    orch = create_orchestrator(
        config={"trainer": "huggingface"},
        reproducibility=repro,
        experiment_tracker=experiment_tracker,
        max_steps=max_s,
        checkpoint_dir=checkpoint_dir,
        run_config=run_config,
    )
    return orch, adapter


def consume_training_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a training job payload by priority.

    Accepts TrainingRunRequest JSON or AgentTrainingJob-shaped dict.
    """
    request = TrainingRunRequest.model_validate(payload.get("training_run_request", payload))
    preset = "feedback" if request.priority == TrainingPriority.LIVE else "training"
    manifest = request.dataset_manifest
    prepared = prepare_training_run(
        manifest.path,
        preset=preset,
        dataset_id=manifest.id,
    )
    ctx = make_run_context(
        request.experiment_id,
        request.model_name,
        fingerprint=request.fingerprint,
        correlation_id=request.fingerprint,
    )
    publish_lifecycle(ctx, "started", status="running")
    return {
        "experiment_id": request.experiment_id,
        "model_name": request.model_name,
        "priority": request.priority.value,
        "prepared_path": str(prepared.path),
        "provenance": (
            prepared.provenance.model_dump()
            if hasattr(prepared.provenance, "model_dump")
            else prepared.provenance
        ),
        "hyperparameters": request.hyperparameters or {},
    }

