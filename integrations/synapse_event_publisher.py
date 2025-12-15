"""
Synapse event publisher integration.

Publishes training events to Synapse (Redis Streams) for integration
with Cyrex and other platform services.
"""

import logging
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class SynapseEventPublisher:
    """
    Publishes training events to Synapse.
    
    Integrates with platform-services/shared/deepiri-synapse
    for event-driven architecture.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize Synapse event publisher.
        
        Args:
            redis_url: Redis URL (defaults to env var)
        """
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://redis:6379",
        )
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
            )
            await self.redis_client.ping()
            self._connected = True
            logger.info("Connected to Synapse (Redis)")
        except Exception as e:
            logger.error(f"Failed to connect to Synapse: {e}")
            self._connected = False
    
    async def publish_training_event(
        self,
        event_type: str,
        model_name: str,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Publish training event.
        
        Args:
            event_type: Event type (started, progress, completed, checkpoint)
            model_name: Model name
            step: Training step
            metrics: Optional metrics dictionary
            **kwargs: Additional event data
        """
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            logger.warning("Not connected to Synapse, skipping event")
            return
        
        event = {
            "event": event_type,
            "model_name": model_name,
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": json.dumps(metrics) if metrics else None,
            **kwargs,
        }
        
        try:
            await self.redis_client.xadd(
                "training-events",
                event,
                maxlen=10000,
                approximate=True,
            )
            logger.debug(f"Published training event: {event_type} for {model_name}")
        except Exception as e:
            logger.error(f"Failed to publish training event: {e}")
    
    async def publish_model_ready_event(
        self,
        model_name: str,
        version: str,
        checkpoint_path: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Publish model-ready event (for Cyrex to auto-load).
        
        Args:
            model_name: Model name
            version: Model version
            checkpoint_path: Path to checkpoint
            metrics: Model metrics
        """
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return
        
        event = {
            "event": "model-ready",
            "model_name": model_name,
            "version": version,
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": json.dumps(metrics) if metrics else None,
        }
        
        try:
            await self.redis_client.xadd(
                "model-events",
                event,
                maxlen=10000,
                approximate=True,
            )
            logger.info(f"Published model-ready event: {model_name} v{version}")
        except Exception as e:
            logger.error(f"Failed to publish model-ready event: {e}")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False

