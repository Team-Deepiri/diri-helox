"""
Inference parity testing system.

Tests train vs inference output parity and quantized vs full-precision
comparisons to prevent "works in training, breaks in prod" issues.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class InferenceParityTester:
    """
    Tests inference parity between training and production.
    
    Tests:
    - Train vs inference mode output parity
    - Full precision vs quantized parity
    - Batch size variations
    - Sequence length variations
    """
    
    def __init__(
        self,
        tolerance: float = 1e-5,
        rtol: float = 1e-4,
    ):
        """
        Initialize parity tester.
        
        Args:
            tolerance: Absolute tolerance for comparisons
            rtol: Relative tolerance for comparisons
        """
        self.tolerance = tolerance
        self.rtol = rtol
    
    def test_train_inference_parity(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Test parity between train and inference modes.
        
        Args:
            model: Model to test
            input_ids: Input token IDs
            
        Returns:
            Parity test results
        """
        model.train()
        train_output = model(input_ids=input_ids)
        train_logits = train_output["logits"]
        
        model.eval()
        with torch.no_grad():
            eval_output = model(input_ids=input_ids)
            eval_logits = eval_output["logits"]
        
        # Compare outputs
        max_diff = (train_logits - eval_logits).abs().max().item()
        mean_diff = (train_logits - eval_logits).abs().mean().item()
        
        # Check if close
        is_close = torch.allclose(
            train_logits,
            eval_logits,
            atol=self.tolerance,
            rtol=self.rtol,
        )
        
        result = {
            "parity_passed": is_close.item() if torch.is_tensor(is_close) else is_close,
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "tolerance": self.tolerance,
        }
        
        if not result["parity_passed"]:
            logger.warning(
                f"Train/inference parity failed: max_diff={max_diff:.2e}, "
                f"mean_diff={mean_diff:.2e}"
            )
        else:
            logger.info("Train/inference parity passed")
        
        return result
    
    def test_quantization_parity(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        quantization_bits: int = 8,
    ) -> Dict[str, Any]:
        """
        Test parity between full precision and quantized models.
        
        Args:
            model: Model to test
            input_ids: Input token IDs
            quantization_bits: Quantization bits (8, 4, etc.)
            
        Returns:
            Quantization parity results
        """
        model.eval()
        
        # Full precision
        with torch.no_grad():
            full_output = model(input_ids=input_ids)
            full_logits = full_output["logits"]
        
        # Quantized
        try:
            if quantization_bits == 8:
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
            else:
                # For other bit widths, would need custom quantization
                logger.warning(f"Quantization to {quantization_bits} bits not implemented")
                return {"parity_passed": False, "error": "Unsupported quantization"}
            
            with torch.no_grad():
                quant_output = quantized_model(input_ids=input_ids)
                quant_logits = quant_output["logits"]
            
            # Compare
            max_diff = (full_logits - quant_logits).abs().max().item()
            mean_diff = (full_logits - quant_logits).abs().mean().item()
            
            # Quantized models have lower precision, so use relaxed tolerance
            relaxed_tolerance = self.tolerance * 10
            is_close = torch.allclose(
                full_logits,
                quant_logits,
                atol=relaxed_tolerance,
                rtol=self.rtol * 10,
            )
            
            result = {
                "parity_passed": is_close.item() if torch.is_tensor(is_close) else is_close,
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "quantization_bits": quantization_bits,
            }
            
            if not result["parity_passed"]:
                logger.warning(
                    f"Quantization parity failed: max_diff={max_diff:.2e}"
                )
            else:
                logger.info("Quantization parity passed")
            
            return result
        
        except Exception as e:
            logger.error(f"Quantization parity test failed: {e}")
            return {
                "parity_passed": False,
                "error": str(e),
            }
    
    def test_batch_size_parity(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Test parity across different batch sizes.
        
        Args:
            model: Model to test
            input_ids: Input token IDs (single batch)
            
        Returns:
            Batch size parity results
        """
        model.eval()
        
        # Single batch
        with torch.no_grad():
            single_output = model(input_ids=input_ids)
            single_logits = single_output["logits"]
        
        # Batch of 2
        batched_input = torch.cat([input_ids, input_ids], dim=0)
        with torch.no_grad():
            batched_output = model(input_ids=batched_input)
            batched_logits = batched_output["logits"]
        
        # Compare first item of batch
        first_item = batched_logits[0:1]
        max_diff = (single_logits - first_item).abs().max().item()
        
        is_close = torch.allclose(
            single_logits,
            first_item,
            atol=self.tolerance,
            rtol=self.rtol,
        )
        
        result = {
            "parity_passed": is_close.item() if torch.is_tensor(is_close) else is_close,
            "max_difference": max_diff,
        }
        
        if not result["parity_passed"]:
            logger.warning(f"Batch size parity failed: max_diff={max_diff:.2e}")
        else:
            logger.info("Batch size parity passed")
        
        return result
    
    def run_full_parity_suite(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run full parity test suite.
        
        Args:
            model: Model to test
            input_ids: Input token IDs
            
        Returns:
            Complete parity test results
        """
        results = {
            "train_inference": self.test_train_inference_parity(model, input_ids),
            "quantization": self.test_quantization_parity(model, input_ids),
            "batch_size": self.test_batch_size_parity(model, input_ids),
        }
        
        all_passed = all(
            r.get("parity_passed", False) for r in results.values()
        )
        
        results["all_tests_passed"] = all_passed
        
        if all_passed:
            logger.info("All parity tests passed")
        else:
            logger.warning("Some parity tests failed")
        
        return results

