"""
Automatic evaluation harness.

Provides fixed eval prompts, domain-specific test sets, and
regression tracking for comprehensive model evaluation.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class AutomaticEvaluationHarness:
    """
    Automatic evaluation system for LLMs.
    
    Features:
    - Fixed evaluation prompts
    - Domain-specific test sets
    - Regression tracking
    - Pass/fail thresholds
    """
    
    def __init__(
        self,
        eval_dir: Path = Path("evaluation"),
        regression_threshold: float = 0.05,
    ):
        """
        Initialize evaluation harness.
        
        Args:
            eval_dir: Directory for evaluation data
            regression_threshold: Threshold for regression detection
        """
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.regression_threshold = regression_threshold
        
        self.test_suites: Dict[str, List[Dict[str, Any]]] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def load_test_suite(
        self,
        suite_name: str,
        test_file: Path,
    ):
        """
        Load test suite from file.
        
        Args:
            suite_name: Name of test suite
            test_file: Path to test file (JSONL)
        """
        test_file = Path(test_file)
        if not test_file.exists():
            raise ValueError(f"Test file not found: {test_file}")
        
        tests = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    test = json.loads(line)
                    tests.append(test)
                except json.JSONDecodeError:
                    continue
        
        self.test_suites[suite_name] = tests
        logger.info(f"Loaded {len(tests)} tests for suite: {suite_name}")
    
    def evaluate_model(
        self,
        model,
        tokenizer_manager,
        suite_name: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test suite.
        
        Args:
            model: Model to evaluate
            tokenizer_manager: Tokenizer manager
            suite_name: Test suite name
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Evaluation results
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite not found: {suite_name}")
        
        tests = self.test_suites[suite_name]
        results = []
        
        model.eval()
        with torch.no_grad():
            for test in tests:
                prompt = test.get("prompt", "")
                expected = test.get("expected", "")
                test_type = test.get("type", "generation")
                
                # Tokenize prompt
                input_ids = tokenizer_manager.encode(prompt, add_bos=True, add_eos=False)
                input_tensor = torch.tensor([input_ids], dtype=torch.long)
                
                # Generate
                generated = model.generate(
                    input_tensor,
                    max_length=len(input_ids) + max_new_tokens,
                )
                
                # Decode
                generated_ids = generated[0].tolist()
                generated_text = tokenizer_manager.decode(generated_ids)
                
                # Score (simple for now, can be extended)
                score = self._score_response(generated_text, expected, test_type)
                
                results.append({
                    "test_id": test.get("id", ""),
                    "prompt": prompt,
                    "expected": expected,
                    "generated": generated_text,
                    "score": score,
                    "passed": score >= test.get("threshold", 0.5),
                })
        
        # Compute aggregate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passed"])
        avg_score = sum(r["score"] for r in results) / total_tests if total_tests > 0 else 0.0
        
        evaluation_result = {
            "suite_name": suite_name,
            "timestamp": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "avg_score": avg_score,
            "results": results,
        }
        
        # Save evaluation
        self._save_evaluation(evaluation_result)
        
        # Check for regression
        regression = self._check_regression(suite_name, avg_score)
        if regression:
            logger.warning(f"Regression detected in {suite_name}: {regression}")
            evaluation_result["regression"] = regression
        
        logger.info(
            f"Evaluation complete: {suite_name} - "
            f"{passed_tests}/{total_tests} passed ({avg_score:.2%})"
        )
        
        return evaluation_result
    
    def _score_response(
        self,
        generated: str,
        expected: str,
        test_type: str,
    ) -> float:
        """
        Score generated response against expected.
        
        Args:
            generated: Generated text
            expected: Expected text
            test_type: Type of test
            
        Returns:
            Score between 0 and 1
        """
        if test_type == "exact_match":
            return 1.0 if generated.strip() == expected.strip() else 0.0
        elif test_type == "contains":
            return 1.0 if expected.lower() in generated.lower() else 0.0
        elif test_type == "similarity":
            # Simple word overlap
            gen_words = set(generated.lower().split())
            exp_words = set(expected.lower().split())
            if not exp_words:
                return 0.0
            overlap = len(gen_words & exp_words) / len(exp_words)
            return overlap
        else:
            # Default: similarity
            return self._score_response(generated, expected, "similarity")
    
    def _save_evaluation(self, result: Dict[str, Any]):
        """Save evaluation result."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        result_file = self.eval_dir / f"eval_{result['suite_name']}_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        self.evaluation_history.append(result)
    
    def _check_regression(
        self,
        suite_name: str,
        current_score: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Check for regression compared to previous evaluations.
        
        Args:
            suite_name: Test suite name
            current_score: Current evaluation score
            
        Returns:
            Regression info if detected, None otherwise
        """
        # Find previous evaluations for this suite
        previous_evals = [
            e for e in self.evaluation_history
            if e["suite_name"] == suite_name
        ]
        
        if not previous_evals:
            return None
        
        # Get best previous score
        best_previous = max(e["avg_score"] for e in previous_evals)
        
        score_drop = best_previous - current_score
        
        if score_drop > self.regression_threshold:
            return {
                "detected": True,
                "current_score": current_score,
                "previous_best": best_previous,
                "score_drop": score_drop,
            }
        
        return None
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        suite_summaries = {}
        for eval_result in self.evaluation_history:
            suite_name = eval_result["suite_name"]
            if suite_name not in suite_summaries:
                suite_summaries[suite_name] = {
                    "count": 0,
                    "scores": [],
                }
            
            suite_summaries[suite_name]["count"] += 1
            suite_summaries[suite_name]["scores"].append(eval_result["avg_score"])
        
        # Compute statistics
        for suite_name, summary in suite_summaries.items():
            scores = summary["scores"]
            summary["mean_score"] = sum(scores) / len(scores)
            summary["max_score"] = max(scores)
            summary["min_score"] = min(scores)
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "suites": suite_summaries,
        }

