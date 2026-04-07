"""
Synapse event publisher integration.

Publishes training events to Synapse (Redis Streams) for integration
with Cyrex and other platform services.
"""

import logging
import os
import json
import asyncio
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import grpc
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Generated stubs import `proto.synapse.v1...`, so add the gen root to sys.path.
_GEN_ROOT = Path(__file__).resolve().parent / "streaming" / "gen"
if str(_GEN_ROOT) not in sys.path:
    sys.path.append(str(_GEN_ROOT))

from proto.synapse.v1 import sidecar_pb2, sidecar_pb2_grpc  # type: ignore


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid float for %s=%r; using default=%s",
            name,
            raw,
            default,
        )
        return default


def _derive_grpc_addr(sidecar_url: str) -> str:
    explicit = os.getenv("SYNAPSE_GRPC_ADDR")
    if explicit:
        return explicit

    try:
        from urllib.parse import urlparse

        parsed = urlparse(sidecar_url)
        if parsed.scheme in {"http", "https"}:
            host = parsed.hostname or "localhost"
            port = parsed.port
            if port is None:
                port = 443 if parsed.scheme == "https" else 80
            if port == 8081:
                port = 50051
            return f"{host}:{port}"
    except Exception:
        pass

    return sidecar_url or "localhost:50051"


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
        self.transport = (os.getenv("SYNAPSE_TRANSPORT", "redis") or "redis").strip().lower()
        self.use_sidecar = self.transport == "sidecar"
        self.sidecar_url = (
            os.getenv("SYNAPSE_SIDECAR_URL", "http://synapse-sidecar:8081").rstrip("/")
        )
        self.sidecar_grpc_addr = _derive_grpc_addr(self.sidecar_url)
        self.sidecar_timeout_sec = _env_float("SYNAPSE_SIDECAR_TIMEOUT_SEC", 5.0)
        self.sidecar_sender = os.getenv("SYNAPSE_SIDECAR_SENDER", "helox")
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self):
        """Connect to Redis."""
        if self.use_sidecar:
            try:
                ready = await asyncio.to_thread(self._sidecar_ready_sync)
                self._connected = ready
                if ready:
                    logger.info("Connected to Synapse sidecar at %s", self.sidecar_url)
                else:
                    logger.error("Synapse sidecar not ready at %s", self.sidecar_url)
            except Exception as e:
                logger.error(f"Failed to connect to Synapse sidecar: {e}")
                self._connected = False
            return

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
            "metrics": metrics if metrics else None,
            **kwargs,
        }

        await self._publish(
            stream="training-events",
            event_type=event_type,
            payload=event,
        )
        logger.debug(f"Published training event: {event_type} for {model_name}")

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
            "metrics": metrics if metrics else None,
        }

        await self._publish(
            stream="model-events",
            event_type="model-ready",
            payload=event,
        )
        logger.info(f"Published model-ready event: {model_name} v{version}")

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        self._connected = False

    async def _publish(self, stream: str, event_type: str, payload: Dict[str, Any]):
        if self.use_sidecar:
            await asyncio.to_thread(
                self._sidecar_publish_sync,
                stream,
                event_type,
                payload,
            )
            return

        if not self.redis_client:
            raise RuntimeError("Redis client is not connected")

        payload_for_redis: Dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                payload_for_redis[key] = ""
            elif isinstance(value, (dict, list)):
                payload_for_redis[key] = json.dumps(value)
            else:
                payload_for_redis[key] = str(value)

        try:
            await self.redis_client.xadd(
                stream,
                payload_for_redis,
                maxlen=10000,
                approximate=True,
            )
        except Exception as e:
            logger.error(f"Failed to publish event via Redis stream={stream}: {e}")
            raise

    def _sidecar_ready_sync(self) -> bool:
        try:
            with grpc.insecure_channel(self.sidecar_grpc_addr) as channel:
                stub = sidecar_pb2_grpc.SynapseSidecarStub(channel)
                response = stub.Health(
                    sidecar_pb2.HealthRequest(),
                    timeout=self.sidecar_timeout_sec,
                )
            return bool(response.healthy)
        except Exception:
            return False

    def _sidecar_publish_sync(self, stream: str, event_type: str, payload: Dict[str, Any]):
        try:
            with grpc.insecure_channel(self.sidecar_grpc_addr) as channel:
                stub = sidecar_pb2_grpc.SynapseSidecarStub(channel)
                response = stub.Publish(
                    sidecar_pb2.PublishRequest(
                        stream=stream,
                        event_type=event_type,
                        sender=self.sidecar_sender,
                        priority="normal",
                        payload=json.dumps(payload).encode("utf-8"),
                    ),
                    timeout=self.sidecar_timeout_sec,
                )
            if not (response.entry_id or "").strip():
                logger.warning(
                    "Synapse sidecar queued event in WAL stream=%s event_type=%s",
                    stream,
                    event_type,
                )
        except grpc.RpcError as err:
            raise RuntimeError(f"sidecar publish failed ({err.code().name}): {err.details()}") from err
