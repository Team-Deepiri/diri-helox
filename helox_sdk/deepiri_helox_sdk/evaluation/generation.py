"""Text generation evaluation for post-training checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol

from .metrics import aggregate_generation_scores, score_text_response
from .samples import EvaluationSample


class TextGenerator(Protocol):
    """Minimal protocol for generation backends."""

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


class GenerationEvaluator:
    """Evaluate generative models against fixed prompt suites."""

    def __init__(self, max_new_tokens: int = 100) -> None:
        self.max_new_tokens = max_new_tokens

    def evaluate_samples(
        self,
        generator: TextGenerator,
        samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """Score each sample and return aggregate + per-item results."""
        results: List[Dict[str, Any]] = []
        scores: List[float] = []
        passed_flags: List[bool] = []

        for sample in samples:
            prompt = sample.prompt or sample.text
            expected = sample.expected or ""
            generated = generator.generate(prompt, self.max_new_tokens)
            score = score_text_response(generated, expected, sample.test_type)
            passed = score >= sample.threshold
            scores.append(score)
            passed_flags.append(passed)
            results.append(
                {
                    "test_id": sample.test_id,
                    "prompt": prompt,
                    "expected": expected,
                    "generated": generated,
                    "score": score,
                    "passed": passed,
                    "test_type": sample.test_type,
                }
            )

        aggregate = aggregate_generation_scores(scores, passed_flags)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **aggregate,
            "results": results,
        }

    def evaluate_callable(
        self,
        generate_fn: Callable[[str, int], str],
        samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """Evaluate using a plain callable instead of a protocol object."""

        class _FnGenerator:
            def generate(self, prompt: str, max_new_tokens: int) -> str:
                return generate_fn(prompt, max_new_tokens)

        return self.evaluate_samples(_FnGenerator(), samples)

    def evaluate_hf_causal_lm(
        self,
        model,
        tokenizer,
        samples: List[EvaluationSample],
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a HuggingFace causal LM with a standard generate loop."""
        import torch

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(resolved_device)
        model = model.to(torch_device)
        model.eval()

        def _generate(prompt: str, max_new_tokens: int) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

        return self.evaluate_callable(_generate, samples)
