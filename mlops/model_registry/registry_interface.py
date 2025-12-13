"""
Model registry interface for Helox
Wraps deepiri-modelkit registry client
"""
import os
from deepiri_modelkit import ModelRegistryClient


class HeloxModelRegistry:
    """Model registry interface for Helox training"""
    
    def __init__(self):
        """Initialize registry interface"""
        self.client = ModelRegistryClient(
            registry_type=os.getenv("MODEL_REGISTRY_TYPE", "mlflow"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            s3_endpoint=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            s3_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            s3_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            s3_bucket=os.getenv("S3_BUCKET", "mlflow-artifacts")
        )
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: dict
    ) -> bool:
        """Register model to registry"""
        return self.client.register_model(model_name, version, model_path, metadata)
    
    def list_models(self, model_name: str = None) -> list:
        """List available models"""
        return self.client.list_models(model_name)

