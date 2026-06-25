"""
Training job worker — consumes Cyrex training-jobs Redis stream.

LIVE priority jobs are processed before BATCH jobs.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from deepiri_modelkit.contracts.training import TrainingPriority, TrainingRunRequest
from deepiri_modelkit.training.job_queue import TrainingJobQueue

from .training_bridge import consume_training_job, get_logger, publish_lifecycle, make_run_context

logger = get_logger("helox.training_job_worker")


class TrainingJobWorker:
    """Polls training-jobs stream and dispatches to training bridge."""

    def __init__(
        self,
        *,
        redis_url: Optional[str] = None,
        consumer_group: str = "helox-workers",
        consumer_name: str = "helox-worker-1",
    ) -> None:
        self.queue = TrainingJobQueue(redis_url=redis_url)
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self._pending_live: List[Tuple[str, TrainingRunRequest]] = []
        self._pending_batch: List[Tuple[str, TrainingRunRequest]] = []

    def _classify(self, request: TrainingRunRequest) -> None:
        """Buffer jobs by priority (LIVE first)."""
        entry = ("", request)
        if request.priority == TrainingPriority.LIVE:
            self._pending_live.append(entry)
        else:
            self._pending_batch.append(entry)

    def _next_job(self) -> Optional[TrainingRunRequest]:
        if self._pending_live:
            _, req = self._pending_live.pop(0)
            return req
        if self._pending_batch:
            _, req = self._pending_batch.pop(0)
            return req
        return None

    def process_request(self, request: TrainingRunRequest) -> Dict[str, Any]:
        """Run a single training job through the bridge."""
        payload = request.model_dump(mode="json")
        result = consume_training_job({"training_run_request": payload})
        ctx = make_run_context(
            request.experiment_id,
            request.model_name,
            fingerprint=request.fingerprint,
        )
        publish_lifecycle(ctx, "completed", status="completed", metrics=result)
        logger.info("training_job_completed", experiment_id=request.experiment_id)
        return result

    def run_once(self, count: int = 10) -> int:
        """Poll stream once and process up to count jobs."""
        messages = self.queue.read_messages(count=count, last_id=">", block_ms=1000)
        for _msg_id, request in messages:
            self._classify(request)

        processed = 0
        while processed < count:
            job = self._next_job()
            if job is None:
                break
            try:
                self.process_request(job)
                processed += 1
            except Exception as exc:
                logger.error("training_job_failed", error=str(exc))
                ctx = make_run_context(job.experiment_id, job.model_name)
                publish_lifecycle(ctx, "failed", status="failed", error=str(exc))
        return processed

    def run_forever(self, poll_interval: float = 1.0) -> None:
        """Blocking worker loop."""
        logger.info("training_job_worker_started", group=self.consumer_group)
        while True:
            self.run_once()
            time.sleep(poll_interval)


def main() -> None:
    worker = TrainingJobWorker(
        redis_url=os.getenv("REDIS_URL"),
        consumer_name=os.getenv("HELOX_WORKER_NAME", "helox-worker-1"),
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
