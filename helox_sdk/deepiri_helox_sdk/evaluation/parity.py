"""Train/inference and quantization parity checks."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InferenceParityTester:
    """Detect train/inference, quantization, and batch-size parity regressions."""

    def __init__(self, tolerance: float = 1e-5, rtol: float = 1e-4) -> None:
        self.tolerance = tolerance
        self.rtol = rtol

    def test_train_inference_parity(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        model.train()
        train_logits = model(input_ids=input_ids)["logits"]

        model.eval()
        with torch.no_grad():
            eval_logits = model(input_ids=input_ids)["logits"]

        max_diff = (train_logits - eval_logits).abs().max().item()
        mean_diff = (train_logits - eval_logits).abs().mean().item()
        is_close = torch.allclose(
            train_logits,
            eval_logits,
            atol=self.tolerance,
            rtol=self.rtol,
        )
        passed = bool(is_close.item() if torch.is_tensor(is_close) else is_close)
        if not passed:
            logger.warning(
                "Train/inference parity failed: max_diff=%.2e mean_diff=%.2e",
                max_diff,
                mean_diff,
            )
        return {
            "parity_passed": passed,
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "tolerance": self.tolerance,
        }

    def test_quantization_parity(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        quantization_bits: int = 8,
    ) -> Dict[str, Any]:
        model.eval()
        with torch.no_grad():
            full_logits = model(input_ids=input_ids)["logits"]

        if quantization_bits != 8:
            return {
                "parity_passed": False,
                "error": f"Unsupported quantization bits: {quantization_bits}",
            }

        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            with torch.no_grad():
                quant_logits = quantized_model(input_ids=input_ids)["logits"]

            max_diff = (full_logits - quant_logits).abs().max().item()
            mean_diff = (full_logits - quant_logits).abs().mean().item()
            relaxed = self.tolerance * 10
            is_close = torch.allclose(
                full_logits,
                quant_logits,
                atol=relaxed,
                rtol=self.rtol * 10,
            )
            passed = bool(is_close.item() if torch.is_tensor(is_close) else is_close)
            return {
                "parity_passed": passed,
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "quantization_bits": quantization_bits,
            }
        except Exception as exc:  # noqa: BLE001 — surface parity failures to caller
            logger.error("Quantization parity test failed: %s", exc)
            return {"parity_passed": False, "error": str(exc)}

    def test_batch_size_parity(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        model.eval()
        with torch.no_grad():
            single_logits = model(input_ids=input_ids)["logits"]

        batched_input = torch.cat([input_ids, input_ids], dim=0)
        with torch.no_grad():
            batched_logits = model(input_ids=batched_input)["logits"]

        first_item = batched_logits[0:1]
        max_diff = (single_logits - first_item).abs().max().item()
        is_close = torch.allclose(
            single_logits,
            first_item,
            atol=self.tolerance,
            rtol=self.rtol,
        )
        passed = bool(is_close.item() if torch.is_tensor(is_close) else is_close)
        return {"parity_passed": passed, "max_difference": max_diff}

    def run_full_parity_suite(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        results = {
            "train_inference": self.test_train_inference_parity(model, input_ids),
            "quantization": self.test_quantization_parity(model, input_ids),
            "batch_size": self.test_batch_size_parity(model, input_ids),
        }
        results["all_tests_passed"] = all(
            section.get("parity_passed", False) for section in results.values()
        )
        return results
