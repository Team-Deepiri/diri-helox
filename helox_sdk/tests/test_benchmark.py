from __future__ import annotations

from deepiri_helox_sdk.evaluation.benchmark import InferenceBenchmark


def test_inference_benchmark_runs():
    counter = {"n": 0}

    def predict() -> None:
        counter["n"] += 1

    bench = InferenceBenchmark(warmup_runs=1)
    result = bench.run(predict, sample_count=5)
    assert result["sample_count"] == 5
    assert result["avg_latency_ms"] is not None
    assert result["throughput_per_sec"] is not None
    assert counter["n"] == 6  # 1 warmup + 5 measured


def test_benchmark_texts():
    seen: list[list[str]] = []

    def predict_batch(batch: list[str]) -> None:
        seen.append(batch)

    texts = ["a", "b", "c"]
    result = InferenceBenchmark(warmup_runs=0).benchmark_texts(predict_batch, texts)
    assert result["sample_count"] == 3
    assert len(seen) >= 3
