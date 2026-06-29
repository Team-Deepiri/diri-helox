from __future__ import annotations

from deepiri_helox_sdk.evaluation.generation import GenerationEvaluator
from deepiri_helox_sdk.evaluation.samples import EvaluationSample


def test_generation_evaluator_callable():
    evaluator = GenerationEvaluator(max_new_tokens=16)
    samples = [
        EvaluationSample(
            prompt="Return status:",
            expected="ok",
            test_type="exact_match",
            threshold=1.0,
            test_id="1",
        ),
        EvaluationSample(
            prompt="Summarize:",
            expected="deployment complete",
            test_type="similarity",
            threshold=0.4,
            test_id="2",
        ),
    ]

    def fake_generate(prompt: str, max_new_tokens: int) -> str:
        if "status" in prompt:
            return "ok"
        return "deployment complete with notes"

    report = evaluator.evaluate_callable(fake_generate, samples)
    assert report["total_tests"] == 2
    assert report["passed_tests"] == 2
    assert report["pass_rate"] == 1.0
    assert report["results"][0]["passed"] is True
