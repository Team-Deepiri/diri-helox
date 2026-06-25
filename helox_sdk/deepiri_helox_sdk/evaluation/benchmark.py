"""Inference latency and throughput benchmarking."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List


class InferenceBenchmark:
    """Measure latency and throughput for a prediction callable."""

    def __init__(self, warmup_runs: int = 3) -> None:
        self.warmup_runs = warmup_runs

    def run(
        self,
        predict_fn: Callable[[], None],
        sample_count: int = 50,
    ) -> Dict[str, Any]:
        """Benchmark a zero-arg callable that performs one inference."""
        for _ in range(self.warmup_runs):
            predict_fn()

        latencies_ms: List[float] = []
        for _ in range(sample_count):
            start = time.perf_counter()
            predict_fn()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

        if not latencies_ms:
            return {
                "sample_count": 0,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "throughput_per_sec": None,
            }

        sorted_latencies = sorted(latencies_ms)
        p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
        avg_latency = statistics.mean(sorted_latencies)
        throughput = 1000.0 / avg_latency if avg_latency > 0 else None

        return {
            "sample_count": sample_count,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": sorted_latencies[p95_index],
            "throughput_per_sec": throughput,
            "latencies_ms": latencies_ms,
        }

    def benchmark_texts(
        self,
        predict_batch_fn: Callable[[List[str]], None],
        texts: List[str],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Benchmark batched text inference."""
        if not texts:
            return self.run(lambda: None, sample_count=0)

        index = 0

        def _step() -> None:
            nonlocal index
            batch = texts[index : index + batch_size]
            if not batch:
                index = 0
                batch = texts[:batch_size]
            predict_batch_fn(batch)
            index = (index + batch_size) % len(texts)

        return self.run(_step, sample_count=min(len(texts), 50))

    @staticmethod
    def merge_into_report(report: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Attach benchmark metrics to an evaluation report dict."""
        merged = dict(report)
        merged["benchmark"] = {
            "avg_latency_ms": benchmark.get("avg_latency_ms"),
            "p95_latency_ms": benchmark.get("p95_latency_ms"),
            "throughput_per_sec": benchmark.get("throughput_per_sec"),
            "sample_count": benchmark.get("sample_count"),
        }
        return merged
