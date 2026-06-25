"""
Model registrar for Helox
Registers trained models to registry and publishes events
"""
import os
from typing import Dict, Any
from deepiri_modelkit import ModelRegistryClient, register_model_ready
from deepiri_modelkit.training.pipeline_factory import create_run_context


class ModelRegistrar:
    """Registers models to registry and publishes events"""

    def __init__(self):
        """Initialize model registrar"""
        self.registry = ModelRegistryClient(
            registry_type=os.getenv("MODEL_REGISTRY_TYPE", "mlflow"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            s3_endpoint=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            s3_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            s3_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            s3_bucket=os.getenv("S3_BUCKET", "mlflow-artifacts"),
        )

    async def register_and_publish(
        self, model_name: str, version: str, model_path: str, metadata: Dict[str, Any]
    ) -> bool:
        """
        Register model to registry and publish model-ready event

        Args:
            model_name: Model name
            version: Model version
            model_path: Path to model file/directory
            metadata: Model metadata

        Returns:
            True if successful
        """
        # Register to registry and publish via modelkit helper
        ctx = create_run_context(
            experiment_id=metadata.get("experiment_id", model_name),
            model_name=model_name,
            source="helox",
            correlation_id=metadata.get("correlation_id"),
        )
        metadata = dict(metadata)
        metadata.setdefault(
            "registry_path",
            f"s3://{os.getenv('S3_BUCKET', 'mlflow-artifacts')}/models/{model_name}/{version}",
        )
        return register_model_ready(
            self.registry,
            model_name=model_name,
            version=version,
            model_path=model_path,
            metadata=metadata,
            source="helox",
            correlation_id=ctx.correlation_id,
            model_type=metadata.get("model_type"),
            accuracy=metadata.get("accuracy"),
            size_mb=metadata.get("size_mb"),
            redis_url=os.getenv("REDIS_URL"),
        )
