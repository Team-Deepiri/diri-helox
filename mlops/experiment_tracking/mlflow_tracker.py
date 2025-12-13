"""
MLflow experiment tracker for Helox
Wraps MLflow for experiment tracking
"""
import os
import mlflow
from typing import Dict, Any, Optional


class MLflowTracker:
    """MLflow experiment tracker"""
    
    def __init__(self):
        """Initialize MLflow tracker"""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None
    ):
        """Start MLflow run"""
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path: str):
        """Log model"""
        mlflow.log_model(model, artifact_path)
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()

