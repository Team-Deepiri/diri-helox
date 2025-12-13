"""
Model registrar for Helox
Registers trained models to registry and publishes events
"""
import os
from typing import Dict, Any
from deepiri_modelkit import ModelRegistryClient
from deepiri_modelkit import StreamingClient
from deepiri_modelkit.streaming.topics import StreamTopics
from deepiri_modelkit.contracts.events import ModelReadyEvent


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
            s3_bucket=os.getenv("S3_BUCKET", "mlflow-artifacts")
        )
        
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", "redispassword")
        
        self.streaming = StreamingClient(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password
        )
    
    async def register_and_publish(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Dict[str, Any]
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
        # Register to registry
        success = self.registry.register_model(
            model_name=model_name,
            version=version,
            model_path=model_path,
            metadata=metadata
        )
        
        if not success:
            return False
        
        # Get registry path
        registry_path = f"s3://{os.getenv('S3_BUCKET', 'mlflow-artifacts')}/models/{model_name}/{version}"
        
        # Publish model-ready event
        await self.streaming.connect()
        
        event = ModelReadyEvent(
            event="model-ready",
            source="helox",
            model_name=model_name,
            version=version,
            registry_path=registry_path,
            metadata=metadata,
            model_type=metadata.get("model_type"),
            accuracy=metadata.get("accuracy"),
            size_mb=metadata.get("size_mb")
        )
        
        await self.streaming.publish(
            StreamTopics.MODEL_EVENTS,
            event.dict()
        )
        
        await self.streaming.disconnect()
        
        return True

