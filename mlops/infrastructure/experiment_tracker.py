"""
Advanced Experiment Tracking with MLflow and W&B Integration
Reproducibility, dataset versioning, model registry
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import json
import yaml
import hashlib
from datetime import datetime
from typing import Dict, Optional, List
import logging

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger



logger = get_logger("experiment.tracker")

# Optional wandb import - handle version conflicts gracefully
try:
    import wandb
    HAS_WANDB = True
except (ImportError, Exception) as e:
    HAS_WANDB = False
    wandb = None
    logger.warning(f"wandb not available: {e}. W&B features will be disabled.")


class ExperimentTracker:
    """Unified experiment tracking with MLflow and optional W&B."""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://localhost:5000",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.use_wandb = use_wandb
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        if use_wandb and wandb_project:
            if not HAS_WANDB:
                logger.warning("wandb requested but not available. Continuing without W&B.")
            else:
                wandb.init(project=wandb_project, name=experiment_name)
        
        self.client = MlflowClient()
        self.current_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Start new experiment run."""
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = mlflow.start_run(run_name=run_name, tags=tags or {})
        
        if self.use_wandb and HAS_WANDB:
            wandb.run.name = run_name
        
        logger.info(f"Experiment run started: run_name={run_name}")
        return self.current_run
    
    def log_params(self, params: Dict):
        """Log hyperparameters."""
        mlflow.log_params(params)
        if self.use_wandb and HAS_WANDB:
            wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
        if self.use_wandb and HAS_WANDB:
            wandb.log(metrics, step=step)
    
    def log_dataset(self, dataset_path: str, dataset_hash: Optional[str] = None):
        """Log dataset with versioning."""
        if dataset_hash is None:
            dataset_hash = self._compute_dataset_hash(dataset_path)
        
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_hash", dataset_hash)
        
        if self.use_wandb and HAS_WANDB:
            wandb.config.update({"dataset_path": dataset_path, "dataset_hash": dataset_hash})
    
    def log_model(self, model_or_dir, artifact_path: str = "model"):
        """
        For LoRA/QLoRA we should not pickle the in-memory model (can fail with Accelerate/AMP).
        Instead, log the saved adapter/tokenizer directory as MLflow artifacts.
        """
        import os

        if isinstance(model_or_dir, (str, os.PathLike)) and os.path.isdir(model_or_dir):
            mlflow.log_artifacts(str(model_or_dir), artifact_path=artifact_path)
            return

        raise ValueError(
            "ExperimentTracker.log_model expected a directory path (e.g., output_dir). "
            "Pass the folder where save_pretrained() wrote adapter/tokenizer files."
        )


    
    def log_code(self, code_path: str = "."):
        """Log code snapshot."""
        try:
            mlflow.log_artifacts(code_path, "code")
        except Exception as e:
            logger.warning("Code logging failed", error=str(e))
    
    def end_run(self, status: str = "FINISHED"):
        """End current run."""
        if self.current_run:
            mlflow.end_run(status=status)
            if self.use_wandb and HAS_WANDB:
                wandb.finish()
            logger.info(f"Experiment run ended: status={status}")

    
    def _compute_dataset_hash(self, dataset_path: str) -> str:
        """Compute SHA256 hash of dataset."""
        sha256 = hashlib.sha256()
        with open(dataset_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def register_model(self, run_id: str, model_name: str, stage: str = "Staging"):
        """Register model in MLflow registry."""
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
        
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage=stage
        )
        
        logger.info("Model registered", model_name=model_name, stage=stage)


class DatasetVersioning:
    """Dataset versioning with DVC integration."""
    
    def __init__(self, dvc_repo_path: str = "."):
        self.dvc_repo_path = Path(dvc_repo_path)
    
    def version_dataset(self, dataset_path: str, description: str = ""):
        """Version dataset using DVC."""
        import subprocess
        
        dvc_path = self.dvc_repo_path / "data" / Path(dataset_path).name
        
        subprocess.run([
            "dvc", "add", str(dvc_path),
            "-f", str(dvc_path.with_suffix('.dvc'))
        ], check=True)
        
        dvc_file = str(dvc_path.with_suffix('.dvc'))
        subprocess.run(["git", "add", dvc_file, dvc_file + ".gitignore"], check=True)
        
        logger.info("Dataset versioned", path=dataset_path)


class ModelRegistry:
    """Model registry with versioning and staging."""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.client = MlflowClient(tracking_uri)
    
    def list_models(self, filter_string: Optional[str] = None) -> List[Dict]:
        """List registered models."""
        models = self.client.search_registered_models(filter_string=filter_string)
        return [{"name": m.name, "versions": len(m.latest_versions)} for m in models]
    
    def get_latest_model(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """Get latest model version URI."""
        try:
            model = self.client.get_latest_versions(model_name, stages=[stage])
            if model:
                return model[0].source
        except Exception as e:
            logger.error("Model retrieval failed", error=str(e))
        return None
    
    def promote_model(self, model_name: str, version: int, stage: str):
        """Promote model to stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info("Model promoted", model_name=model_name, version=version, stage=stage)


