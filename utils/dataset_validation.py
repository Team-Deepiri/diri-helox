"""
Dataset Validation Utilities
Provides validation and quality checks for language intelligence datasets
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from collections import Counter

from deepiri_modelkit.logging import get_logger

logger = get_logger("helox.dataset_validation")


class DatasetValidator:
    """
    Validates dataset quality and integrity for language intelligence tasks.

    Supports validation for:
    - Lease abstraction datasets
    - Contract intelligence datasets
    - General text quality checks
    """

    def __init__(self, dataset_type: str = "general"):
        self.dataset_type = dataset_type
        self.validation_rules = self._get_validation_rules()

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules based on dataset type."""
        base_rules = {
            "min_samples": 10,
            "max_samples": 100000,
            "min_text_length": 10,
            "max_text_length": 10000,
            "required_fields": ["text"],
            "text_quality_checks": True
        }

        type_specific_rules = {
            "lease_abstraction": {
                "min_samples": 50,
                "lease_keywords": [
                    "lease", "agreement", "landlord", "tenant", "rent",
                    "premises", "term", "commencement", "expiration"
                ],
                "min_keyword_matches": 2,
                "check_address_patterns": True,
                "check_rent_patterns": True
            },
            "contract_intelligence": {
                "min_samples": 50,
                "contract_keywords": [
                    "contract", "agreement", "party", "obligation",
                    "clause", "provision", "section", "article"
                ],
                "min_keyword_matches": 2,
                "check_legal_patterns": True
            }
        }

        if self.dataset_type in type_specific_rules:
            base_rules.update(type_specific_rules[self.dataset_type])

        return base_rules

    def validate_dataset(self, data_path: Path) -> Dict[str, Any]:
        """
        Comprehensive dataset validation.

        Args:
            data_path: Path to dataset files

        Returns:
            Validation results dictionary
        """
        logger.info("Starting dataset validation", path=str(data_path), type=self.dataset_type)

        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
            "quality_score": 0.0
        }

        try:
            # Load and parse data
            samples = self._load_samples(data_path)
            results["statistics"]["total_samples"] = len(samples)

            if not samples:
                results["is_valid"] = False
                results["errors"].append("No samples found in dataset")
                return results

            # Basic structure validation
            self._validate_structure(samples, results)

            # Content quality validation
            if results["is_valid"]:
                self._validate_content_quality(samples, results)

            # Type-specific validation
            if self.dataset_type != "general":
                self._validate_type_specific(samples, results)

            # Calculate overall quality score
            results["quality_score"] = self._calculate_quality_score(results)

            # Determine final validity
            results["is_valid"] = len(results["errors"]) == 0

        except Exception as e:
            results["is_valid"] = False
            results["errors"].append(f"Validation failed with error: {str(e)}")
            logger.error("Dataset validation failed", error=str(e))

        logger.info("Dataset validation complete",
                   valid=results["is_valid"],
                   quality_score=results["quality_score"],
                   errors=len(results["errors"]),
                   warnings=len(results["warnings"]))

        return results

    def _load_samples(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load samples from dataset files."""
        samples = []

        if data_path.is_file() and data_path.suffix == ".jsonl":
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at line {line_num}: {e}")

        elif data_path.is_dir():
            for file_path in data_path.glob("*.jsonl"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                samples.append(sample)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON in {file_path} at line {line_num}: {e}")

        return samples

    def _validate_structure(self, samples: List[Dict], results: Dict):
        """Validate basic dataset structure."""
        if len(samples) < self.validation_rules["min_samples"]:
            results["errors"].append(
                f"Insufficient samples: {len(samples)} < {self.validation_rules['min_samples']}"
            )

        if len(samples) > self.validation_rules["max_samples"]:
            results["warnings"].append(
                f"Large dataset: {len(samples)} > {self.validation_rules['max_samples']}"
            )

        # Check required fields
        required_fields = self.validation_rules["required_fields"]
        for i, sample in enumerate(samples[:100]):  # Check first 100 samples
            for field in required_fields:
                if field not in sample:
                    results["errors"].append(f"Missing required field '{field}' in sample {i}")

    def _validate_content_quality(self, samples: List[Dict], results: Dict):
        """Validate content quality."""
        text_lengths = []
        empty_texts = 0
        duplicate_texts = set()
        seen_texts = set()

        for sample in samples:
            text = sample.get("text", "").strip()

            # Check text length
            text_len = len(text)
            text_lengths.append(text_len)

            if text_len < self.validation_rules["min_text_length"]:
                results["errors"].append(f"Text too short: {text_len} chars")
            elif text_len > self.validation_rules["max_text_length"]:
                results["warnings"].append(f"Text too long: {text_len} chars")

            if not text:
                empty_texts += 1

            # Check for duplicates
            if text in seen_texts:
                duplicate_texts.add(text)
            else:
                seen_texts.add(text)

        # Statistics
        results["statistics"].update({
            "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "empty_texts": empty_texts,
            "duplicate_texts": len(duplicate_texts)
        })

        if empty_texts > 0:
            results["errors"].append(f"Found {empty_texts} empty text samples")

        if len(duplicate_texts) > len(samples) * 0.01:  # >1% duplicates
            results["warnings"].append(f"High duplicate rate: {len(duplicate_texts)} duplicates")

    def _validate_type_specific(self, samples: List[Dict], results: Dict):
        """Type-specific validation."""
        if self.dataset_type == "lease_abstraction":
            self._validate_lease_abstraction(samples, results)
        elif self.dataset_type == "contract_intelligence":
            self._validate_contract_intelligence(samples, results)

    def _validate_lease_abstraction(self, samples: List[Dict], results: Dict):
        """Validate lease abstraction dataset."""
        keywords = self.validation_rules["lease_keywords"]
        min_matches = self.validation_rules["min_keyword_matches"]

        low_keyword_samples = 0
        address_pattern_matches = 0
        rent_pattern_matches = 0

        # Address patterns (street numbers, street names, cities)
        address_pattern = r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl|Court|Ct)\s*,?\s*[A-Za-z\s]+,?\s*\d{5}'

        # Rent patterns (dollar amounts)
        rent_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'

        for sample in samples[:500]:  # Check first 500 samples for performance
            text = sample.get("text", "").lower()

            # Keyword matching
            keyword_matches = sum(1 for keyword in keywords if keyword in text)
            if keyword_matches < min_matches:
                low_keyword_samples += 1

            # Pattern matching
            if re.search(address_pattern, sample.get("text", "")):
                address_pattern_matches += 1

            if re.search(rent_pattern, sample.get("text", "")):
                rent_pattern_matches += 1

        total_checked = min(500, len(samples))
        keyword_failure_rate = low_keyword_samples / total_checked

        if keyword_failure_rate > 0.3:  # >30% samples lack keywords
            results["warnings"].append(
                f"Low keyword relevance: {keyword_failure_rate:.1%} samples lack lease keywords"
            )

        results["statistics"].update({
            "address_pattern_matches": address_pattern_matches,
            "rent_pattern_matches": rent_pattern_matches,
            "keyword_relevance_score": 1.0 - keyword_failure_rate
        })

    def _validate_contract_intelligence(self, samples: List[Dict], results: Dict):
        """Validate contract intelligence dataset."""
        keywords = self.validation_rules["contract_keywords"]
        min_matches = self.validation_rules["min_keyword_matches"]

        low_keyword_samples = 0
        legal_pattern_matches = 0

        # Legal clause patterns
        legal_patterns = [
            r'\bsection\s+\d+',
            r'\barticle\s+\d+',
            r'\bclause\s+\d+',
            r'\bparagraph\s+\d+',
            r'\bsubsection\s+\d+'
        ]

        for sample in samples[:500]:  # Check first 500 samples
            text = sample.get("text", "").lower()

            # Keyword matching
            keyword_matches = sum(1 for keyword in keywords if keyword in text)
            if keyword_matches < min_matches:
                low_keyword_samples += 1

            # Legal pattern matching
            if any(re.search(pattern, sample.get("text", "")) for pattern in legal_patterns):
                legal_pattern_matches += 1

        total_checked = min(500, len(samples))
        keyword_failure_rate = low_keyword_samples / total_checked

        if keyword_failure_rate > 0.3:
            results["warnings"].append(
                f"Low keyword relevance: {keyword_failure_rate:.1%} samples lack contract keywords"
            )

        results["statistics"].update({
            "legal_pattern_matches": legal_pattern_matches,
            "keyword_relevance_score": 1.0 - keyword_failure_rate
        })

    def _calculate_quality_score(self, results: Dict) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        score = 1.0

        # Penalize errors heavily
        error_penalty = len(results["errors"]) * 0.2
        score -= min(error_penalty, 0.8)

        # Penalize warnings moderately
        warning_penalty = len(results["warnings"]) * 0.05
        score -= min(warning_penalty, 0.2)

        # Bonus for good statistics
        stats = results["statistics"]

        if stats.get("avg_text_length", 0) > 100:
            score += 0.05  # Good average text length

        if stats.get("duplicate_texts", 0) == 0:
            score += 0.1  # No duplicates

        if stats.get("empty_texts", 0) == 0:
            score += 0.1  # No empty texts

        # Type-specific bonuses
        if self.dataset_type == "lease_abstraction":
            if stats.get("keyword_relevance_score", 0) > 0.7:
                score += 0.1
            if stats.get("address_pattern_matches", 0) > 0:
                score += 0.05

        elif self.dataset_type == "contract_intelligence":
            if stats.get("keyword_relevance_score", 0) > 0.7:
                score += 0.1
            if stats.get("legal_pattern_matches", 0) > 0:
                score += 0.05

        return max(0.0, min(1.0, score))


def validate_dataset_quality(data_path: Path, dataset_type: str = "general") -> Dict[str, Any]:
    """
    Convenience function to validate dataset quality.

    Args:
        data_path: Path to dataset
        dataset_type: Type of dataset for specialized validation

    Returns:
        Validation results
    """
    validator = DatasetValidator(dataset_type)
    return validator.validate_dataset(data_path)


# Example usage
if __name__ == "__main__":
    # Validate lease abstraction dataset
    results = validate_dataset_quality(
        Path("./data/samples/lease_abstraction_v1"),
        "lease_abstraction"
    )

    print(f"Validation Results:")
    print(f"Valid: {results['is_valid']}")
    print(f"Quality Score: {results['quality_score']:.2f}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Statistics: {results['statistics']}")
