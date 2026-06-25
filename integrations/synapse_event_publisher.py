"""
Synapse event publisher integration.

Publishes training events to Synapse (Redis Streams) for integration
with Cyrex and other platform services.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis
from deepiri_modelkit import TrainingRunContext, build_training_event
from deepiri_modelkit.training.integration import publish_training_event

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
        *,
        experiment_id: str = "helox",
        correlation_id: Optional[str] = None,
    ):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://redis:6379",
        )
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
        self._ctx = TrainingRunContext(
            experiment_id=experiment_id,
            model_name="llm-training",
            source="helox",
            correlation_id=correlation_id,
        )

    def set_context(
        self,
        *,
        experiment_id: Optional[str] = None,
        model_name: Optional[str] = None,
        fingerprint: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        if experiment_id:
            self._ctx.experiment_id = experiment_id
        if model_name:
            self._ctx.model_name = model_name
        if fingerprint:
            self._ctx.fingerprint = fingerprint
        if correlation_id:
            self._ctx.correlation_id = correlation_id

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
        Publish training event using modelkit TrainingEvent schema.
        """
        self._ctx.model_name = model_name
        progress = None
        if kwargs.get("total_steps"):
            progress = min(1.0, step / max(kwargs["total_steps"], 1))

        status = "running"
        if event_type in ("completed", "complete", "done"):
            status = "completed"
        elif event_type in ("failed", "error"):
            status = "failed"

        event_metrics = dict(metrics or {})
        event_metrics["step"] = step
        event_metrics.update({k: v for k, v in kwargs.items() if k != "total_steps"})

        typed_event = build_training_event(
            event_type,
            self._ctx,
            status=status,
            progress=progress,
            metrics=event_metrics,
        )

        try:
            publish_training_event(typed_event, redis_url=self.redis_url)
            logger.debug(f"Published training event: {event_type} for {model_name}")
            return
        except Exception as sync_err:
            logger.debug(f"Sync modelkit publish failed, trying async: {sync_err}")

        if not self._connected:
            await self.connect()

        if not self._connected:
            logger.warning("Not connected to Synapse, skipping event")
            return

        payload = typed_event.model_dump(mode="json")
        flat = {
            key: json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            for key, value in payload.items()
            if value is not None
        }

        try:
            await self.redis_client.xadd(
                "training-events",
                flat,
                maxlen=10000,
                approximate=True,
            )
            logger.debug(f"Published training event (async): {event_type} for {model_name}")
        except Exception as e:
            logger.error(f"Failed to publish training event: {e}")

    async def publish_model_ready_event(
        self,
        model_name: str,
        version: str,
        checkpoint_path: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Publish model-ready event (for Cyrex to auto-load)."""
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
