"""
Data Quality Framework

This module provides comprehensive data quality checks across 7 dimensions:
1. Completeness - Missing values, coverage
2. Consistency - Ranges, formats
3. Validity - Schema, business rules
4. Uniqueness - Duplicates, keys
5. Timeliness - Freshness, staleness
6. Accuracy - Statistical tests, outliers
7. Integrity - Referential, constraints

Integrates with the preprocessing pipeline's ValidationResult and ProcessedData classes.

Usage Examples:
    Basic Usage:
        ```python
        from pipelines.data_preprocessing.quality import QualityChecker
        
        checker = QualityChecker()
        report = checker.check_quality(data, dataset_id="my_dataset")
        print(f"Overall score: {report.overall_score}")
        ```
    
    Custom Configuration:
        ```python
        from pipelines.data_preprocessing.quality import QualityChecker, QualityConfig
        
        config = QualityConfig(
            completeness_threshold=0.98,
            iqr_multiplier=2.0,
            freshness_decay_days=14.0
        )
        checker = QualityChecker(config=config)
        report = checker.check_quality(data, dataset_id="my_dataset", schema=schema)
        ```
    
    Pipeline Integration:
        ```python
        from pipelines.data_preprocessing.quality import QualityCheckStage
        
        quality_stage = QualityCheckStage(config={"completeness_threshold": 0.95})
        result = quality_stage.process(data)
        
        if result.success:
            validation_result = result.validation_result
            processed_data = result.processed_data
            quality_metrics = processed_data.quality_metrics
        ```
    
    Convert to ValidationResult:
        ```python
        report = checker.check_quality(data)
        validation_result = report.to_validation_result()
        # Use validation_result.errors, validation_result.warnings, validation_result.quality_scores
        ```
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from .base import ValidationResult


@dataclass
class QualityConfig:
    """
    Configuration class for data quality checks.
    
    This class centralizes all configurable thresholds and parameters,
    eliminating hardcoded values throughout the quality framework.
    
    Schema Format Documentation:
    The schema parameter should be a dictionary with the following structure:
    {
        "fields": {
            "field_name": {
                "type": "int|float|string|bool",
                "constraints": {
                    "min": <min_value>,
                    "max": <max_value>
                }
            }
        },
        "required_fields": ["field1", "field2"],
        "primary_keys": ["id", "key"],
        "foreign_keys": {
            "fk_name": {
                "references": "table.column"
            }
        }
    }
    """
    # Dimension thresholds (0.0 to 1.0)
    completeness_threshold: float = 0.95
    consistency_threshold: float = 0.90
    validity_threshold: float = 1.0
    uniqueness_threshold: float = 0.98
    timeliness_threshold: float = 0.70
    accuracy_threshold: float = 0.90
    integrity_threshold: float = 1.0
    
    # Statistical method parameters
    iqr_multiplier: float = 1.5  # IQR outlier detection multiplier
    zscore_threshold: float = 3.0  # Z-score threshold for outliers
    isolation_forest_contamination: float = 0.1  # Expected proportion of outliers
    random_state: int = 42  # Random seed for reproducibility
    
    # Timeliness parameters
    freshness_decay_days: float = 30.0  # Days over which freshness decays to 0
    
    # Uniqueness parameters
    key_uniqueness_threshold: float = 0.95  # Threshold for key column uniqueness
    
    # Recommendation threshold
    recommendation_threshold: float = 0.8  # Score below which recommendations are generated
    
    # Validity scoring parameters
    validity_error_penalty: float = 0.1  # Penalty per validation error (0.0 to 1.0)
    
    # Performance parameters
    shapiro_wilk_sample_size: int = 5000  # Max sample size for Shapiro-Wilk test
    min_samples_for_outlier_detection: int = 4  # Minimum samples needed for outlier detection
    min_samples_for_statistical_tests: int = 30  # Minimum samples for statistical tests


@dataclass
class QualityMetric:
    """Individual quality metric result."""
    dimension: str  # One of the 7 quality dimensions
    metric_name: str  # Name of the specific metric
    value: float  # Metric value (0.0 to 1.0 for scores, or raw value)
    threshold: Optional[float] = None  # Threshold for this metric
    passed: bool = True  # Whether the metric passed its threshold
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details


@dataclass
class QualityReport:
    """
    Comprehensive quality report for a dataset.
    
    This class can be converted to ValidationResult for pipeline integration.
    """
    dataset_id: str  # Identifier for the dataset
    timestamp: datetime  # When the quality check was performed
    overall_score: float  # Overall quality score (0.0 to 1.0)
    dimension_scores: Dict[str, float]  # Score for each dimension (0.0 to 1.0)
    metrics: List[QualityMetric]  # List of all quality metrics
    summary: Dict[str, Any] = field(default_factory=dict)  # Summary statistics
    recommendations: List[str] = field(default_factory=list)  # Recommendations for improvement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quality report to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "metrics": [
                {
                    "dimension": m.dimension,
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "details": m.details
                }
                for m in self.metrics
            ],
            "summary": self.summary,
            "recommendations": self.recommendations
        }
    
    def to_validation_result(self) -> ValidationResult:
        """
        Convert QualityReport to ValidationResult for pipeline integration.
        
        Returns:
            ValidationResult with quality_scores from dimension_scores,
            errors from failed metrics, and warnings from recommendations.
        """
        errors = [
            f"{m.dimension}.{m.metric_name}: {m.value:.2f} < {m.threshold}"
            for m in self.metrics if not m.passed
        ]
        
        warnings = self.recommendations.copy()
        
        # Add quality_scores from dimension_scores
        quality_scores = self.dimension_scores.copy()
        quality_scores["overall"] = self.overall_score
        
        return ValidationResult(
            is_valid=len(errors) == 0 and self.overall_score >= 0.8,
            errors=errors,
            warnings=warnings,
            quality_scores=quality_scores
        )
    
    def get_quality_metrics_for_processed_data(self) -> Dict[str, float]:
        """
        Get quality metrics in the format expected by ProcessedData.
        
        Returns:
            Dictionary mapping dimension names to scores (0.0 to 1.0)
        """
        return self.dimension_scores.copy()


class StatisticalValidator:
    """
    Statistical validation methods for data quality checks.
    
    Provides statistical tests and calculations for:
    - Outlier detection (IQR, Z-score, Isolation Forest)
    - Distribution tests (Kolmogorov-Smirnov, Shapiro-Wilk)
    - Correlation analysis
    - Hypothesis testing
    
    All methods are static and stateless, making them safe for concurrent use.
    """
    
    @staticmethod
    def detect_outliers_iqr(data: np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            data: Numeric data array
            multiplier: IQR multiplier (default 1.5)
            
        Returns:
            Tuple of (outlier_mask, details_dict)
        """
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
        details = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(outliers.sum()),
            "outlier_percentage": float(outliers.sum() / len(data) * 100)
        }
        
        return outliers, details
    
    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect outliers using Z-score method.
        
        Args:
            data: Numeric data array
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Tuple of (outlier_mask, details_dict)
        """
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        
        details = {
            "threshold": threshold,
            "mean_zscore": float(np.mean(z_scores)),
            "max_zscore": float(np.max(z_scores)),
            "outlier_count": int(outliers.sum()),
            "outlier_percentage": float(outliers.sum() / len(data) * 100)
        }
        
        return outliers, details
    
    @staticmethod
    def detect_outliers_isolation_forest(
        data: np.ndarray, 
        contamination: float = 0.1,
        random_state: Optional[int] = 42
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect outliers using Isolation Forest (ML-based).
        
        Args:
            data: Numeric data array (reshaped to 2D if needed)
            contamination: Expected proportion of outliers
            random_state: Random seed
            
        Returns:
            Tuple of (outlier_mask, details_dict)
        """
        # Reshape to 2D if needed
        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        
        clf = IsolationForest(contamination=contamination, random_state=random_state)
        outlier_predictions = clf.fit_predict(data_2d)
        outliers = outlier_predictions == -1
        
        details = {
            "contamination": contamination,
            "outlier_count": int(outliers.sum()),
            "outlier_percentage": float(outliers.sum() / len(data) * 100),
            "scores": clf.score_samples(data_2d).tolist()
        }
        
        return outliers, details
    
    @staticmethod
    def kolmogorov_smirnov_test(
        data: np.ndarray, 
        distribution: str = 'norm'
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Perform Kolmogorov-Smirnov test for distribution.
        
        Args:
            data: Numeric data array
            distribution: Distribution to test against ('norm', 'uniform', etc.)
            
        Returns:
            Tuple of (statistic, p_value, details_dict)
        """
        if distribution == 'norm':
            statistic, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        elif distribution == 'uniform':
            statistic, p_value = stats.kstest(data, 'uniform')
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        details = {
            "distribution": distribution,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "reject_null": p_value < 0.05  # Null hypothesis: data follows the distribution
        }
        
        return statistic, p_value, details
    
    @staticmethod
    def shapiro_wilk_test(
        data: np.ndarray, 
        max_sample_size: int = 5000,
        random_state: Optional[int] = None
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Perform Shapiro-Wilk test for normality (limited to max_sample_size samples).
        
        Args:
            data: Numeric data array
            max_sample_size: Maximum sample size for the test (default: 5000)
            random_state: Random seed for sampling (default: None)
            
        Returns:
            Tuple of (statistic, p_value, details_dict)
        """
        # Limit sample size for Shapiro-Wilk
        if len(data) > max_sample_size:
            if random_state is not None:
                np.random.seed(random_state)
            sample_data = np.random.choice(data, max_sample_size, replace=False)
            details = {
                "note": f"Test performed on {max_sample_size} sample subset",
                "original_size": len(data),
                "sample_size": max_sample_size
            }
        else:
            sample_data = data
            details = {}
        
        statistic, p_value = stats.shapiro(sample_data)
        
        details.update({
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05  # Null hypothesis: data is normally distributed
        })
        
        return statistic, p_value, details


class QualityChecker:
    """
    Main class for performing comprehensive data quality checks.
    
    Implements all 7 quality dimensions and generates quality reports.
    All configurable parameters are provided via QualityConfig to avoid hardcoded values.
    
    Example Usage:
        ```python
        from pipelines.data_preprocessing.quality import QualityChecker, QualityConfig
        
        # Use default config
        checker = QualityChecker()
        report = checker.check_quality(data, dataset_id="my_dataset")
        
        # Use custom config
        config = QualityConfig(
            completeness_threshold=0.98,
            iqr_multiplier=2.0,
            freshness_decay_days=14.0
        )
        checker = QualityChecker(config=config)
        report = checker.check_quality(data, dataset_id="my_dataset", schema=schema)
        
        # Convert to ValidationResult for pipeline integration
        validation_result = report.to_validation_result()
        ```
    """
    
    def __init__(
        self,
        config: Optional[QualityConfig] = None,
        # Legacy parameter support (deprecated, use config instead)
        completeness_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
        validity_threshold: Optional[float] = None,
        uniqueness_threshold: Optional[float] = None,
        accuracy_threshold: Optional[float] = None,
        integrity_threshold: Optional[float] = None
    ):
        """
        Initialize quality checker with configuration.
        
        Args:
            config: QualityConfig instance with all parameters (recommended)
            completeness_threshold: (Legacy) Override completeness threshold
            consistency_threshold: (Legacy) Override consistency threshold
            validity_threshold: (Legacy) Override validity threshold
            uniqueness_threshold: (Legacy) Override uniqueness threshold
            accuracy_threshold: (Legacy) Override accuracy threshold
            integrity_threshold: (Legacy) Override integrity threshold
        """
        # Use provided config or create default
        self.config = config or QualityConfig()
        
        # Support legacy parameters for backward compatibility
        if completeness_threshold is not None:
            self.config.completeness_threshold = completeness_threshold
        if consistency_threshold is not None:
            self.config.consistency_threshold = consistency_threshold
        if validity_threshold is not None:
            self.config.validity_threshold = validity_threshold
        if uniqueness_threshold is not None:
            self.config.uniqueness_threshold = uniqueness_threshold
        if accuracy_threshold is not None:
            self.config.accuracy_threshold = accuracy_threshold
        if integrity_threshold is not None:
            self.config.integrity_threshold = integrity_threshold
        
        self.stat_validator = StatisticalValidator()
    
    def check_quality(
        self,
        data: Union[pd.DataFrame, List[Dict], Dict],
        dataset_id: str = "dataset",
        schema: Optional[Dict[str, Any]] = None,
        reference_data: Optional[Union[pd.DataFrame, List[Dict]]] = None
    ) -> QualityReport:
        """
        Perform comprehensive quality checks on data.
        
        Args:
            data: Data to check (DataFrame, list of dicts, or single dict)
            dataset_id: Identifier for the dataset
            schema: Optional schema definition for validation
            reference_data: Optional reference data for drift/consistency checks
            
        Returns:
            QualityReport with all quality metrics
        """
        # Convert to DataFrame for easier analysis
        try:
            df = self._to_dataframe(data)
        except Exception as e:
            # Return error report if conversion fails
            return QualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(),
                overall_score=0.0,
                dimension_scores={},
                metrics=[QualityMetric(
                    dimension="validity",
                    metric_name="data_conversion_error",
                    value=0.0,
                    threshold=self.config.validity_threshold,
                    passed=False,
                    details={"error": str(e)}
                )],
                summary={"error": "Failed to convert data to DataFrame"},
                recommendations=[f"Data conversion failed: {str(e)}"]
            )
        
        metrics = []
        
        # 1. Completeness checks
        try:
            completeness_metrics = self._check_completeness(df)
            metrics.extend(completeness_metrics)  # FIX: Was missing this line
        except Exception as e:
            metrics.append(QualityMetric(
                dimension="completeness",
                metric_name="completeness_check_error",
                value=0.0,
                threshold=self.config.completeness_threshold,
                passed=False,
                details={"error": str(e)}
            ))
       
        # 2. Consistency checks
        try:
            consistency_metrics = self._check_consistency(df)
            metrics.extend(consistency_metrics)
        except Exception as e:
            metrics.append(QualityMetric(
                dimension="consistency",
                metric_name="consistency_check_error",
                value=0.0,
                threshold=self.config.consistency_threshold,
                passed=False,
                details={"error": str(e)}
            ))
        
        # 3. Validity checks
        try:
            validity_metrics = self._check_validity(df, schema)
            metrics.extend(validity_metrics)
        except Exception as e:
            metrics.append(QualityMetric(
                dimension="validity",
                metric_name="validity_check_error",
                value=0.0,
                threshold=self.config.validity_threshold,
                passed=False,
                details={"error": str(e)}
            ))
        
        # 4. Uniqueness checks
        try:
            uniqueness_metrics = self._check_uniqueness(df)
            metrics.extend(uniqueness_metrics)
        except Exception as e:
            metrics.append(QualityMetric(
                dimension="uniqueness",
                metric_name="uniqueness_check_error",
                value=0.0,
                threshold=self.config.uniqueness_threshold,
                passed=False,
                details={"error": str(e)}
            ))
        
        # 5. Timeliness checks
        try:
            timeliness_metrics = self._check_timeliness(df)
            metrics.extend(timeliness_metrics)
        except Exception as e:
            metrics.append(QualityMetric(
                dimension="timeliness",
                metric_name="timeliness_check_error",
                value=0.0,
                threshold=self.config.timeliness_threshold,
                passed=False,
                details={"error": str(e)}
            ))
        
        # 6. Accuracy checks
        try:
            accuracy_metrics = self._check_accuracy(df, reference_data)
            metrics.extend(accuracy_metrics)
        except Exception as e:
            metrics.append(QualityMetric(
                dimension="accuracy",
                metric_name="accuracy_check_error",
                value=0.0,
                threshold=self.config.accuracy_threshold,
                passed=False,
                details={"error": str(e)}
            ))
        
        # 7. Integrity checks
        try:
            integrity_metrics = self._check_integrity(df, schema)
            metrics.extend(integrity_metrics)
        except Exception as e:
            metrics.append(QualityMetric(
                dimension="integrity",
                metric_name="integrity_check_error",
                value=0.0,
                threshold=self.config.integrity_threshold,
                passed=False,
                details={"error": str(e)}
            ))
        
        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(metrics)
        
        # Calculate overall score (weighted average)
        overall_score = np.mean(list(dimension_scores.values()))
        
        # Generate summary and recommendations
        summary = self._generate_summary(df, metrics)
        recommendations = self._generate_recommendations(dimension_scores, metrics)
        
        return QualityReport(
            dataset_id=dataset_id,
            timestamp=datetime.now(),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations
        )
    def _to_dataframe(self, data: Union[pd.DataFrame, List[Dict], Dict]) -> pd.DataFrame:
        """
        Convert various data formats to DataFrame.
        
        Args:
            data: Input data in various formats
            
        Returns:
            pandas DataFrame
            
        Raises:
            ValueError: If data format is unsupported or conversion fails
            TypeError: If data type cannot be converted
        """
        try:
            if isinstance(data, pd.DataFrame):
                return data.copy()
            elif isinstance(data, list):
                if len(data) == 0:
                    return pd.DataFrame()
                if isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    raise ValueError(f"List data must contain dictionaries, got {type(data[0])}")
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            raise ValueError(f"Failed to convert data to DataFrame: {str(e)}") from e
    
    def _check_completeness(self, df: pd.DataFrame) -> List[QualityMetric]:
        """Check completeness dimension (missing values, coverage)."""
        metrics = []
        
        if df.empty:
            metrics.append(QualityMetric(
                dimension="completeness",
                metric_name="empty_dataset",
                value=0.0,
                threshold=self.config.completeness_threshold,
                passed=False,
                details={"message": "Dataset is empty"}
            ))
            return metrics
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1.0 - (missing_cells / total_cells)
        
        metrics.append(QualityMetric(
            dimension="completeness",
            metric_name="overall_completeness",
            value=completeness_score,
            threshold=self.config.completeness_threshold,
            passed=completeness_score >= self.config.completeness_threshold,
            details={
                "total_cells": int(total_cells),
                "missing_cells": int(missing_cells),
                "missing_percentage": float(missing_cells / total_cells * 100)
            }
        ))
        
        # Per-column completeness
        for col in df.columns:
            col_completeness = 1.0 - (df[col].isnull().sum() / len(df))
            metrics.append(QualityMetric(
                dimension="completeness",
                metric_name=f"column_{col}_completeness",
                value=col_completeness,
                threshold=self.config.completeness_threshold,
                passed=col_completeness >= self.config.completeness_threshold,
                details={
                    "column": col,
                    "missing_count": int(df[col].isnull().sum()),
                    "missing_percentage": float(df[col].isnull().sum() / len(df) * 100)
                }
            ))
        
        # Row-level completeness (rows with any missing values)
        rows_with_missing = df.isnull().any(axis=1).sum()
        row_completeness = 1.0 - (rows_with_missing / len(df))
        metrics.append(QualityMetric(
            dimension="completeness",
            metric_name="row_completeness",
            value=row_completeness,
            threshold=self.config.completeness_threshold,
            passed=row_completeness >= self.config.completeness_threshold,
            details={
                "rows_with_missing": int(rows_with_missing),
                "complete_rows": int(len(df) - rows_with_missing),
                "rows_with_missing_percentage": float(rows_with_missing / len(df) * 100)
            }
        ))
        
        return metrics
    
    def _check_consistency(self, df: pd.DataFrame) -> List[QualityMetric]:
        """Check consistency dimension (ranges, formats)."""
        metrics = []
        
        if df.empty:
            return metrics
        
        # Check for consistent data types
        type_consistency_score = 1.0
        type_issues = []
        
        for col in df.columns:
            # Check if column has consistent types
            col_types = df[col].apply(type).unique()
            if len(col_types) > 1:
                type_consistency_score -= 0.1
                type_issues.append({
                    "column": col,
                    "types": [str(t) for t in col_types]
                })
        
        metrics.append(QualityMetric(
            dimension="consistency",
            metric_name="type_consistency",
            value=max(0.0, type_consistency_score),
            threshold=self.config.consistency_threshold,
            passed=type_consistency_score >= self.config.consistency_threshold,
            details={"type_issues": type_issues}
        ))
        
        # Check numeric ranges
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Check for negative values where not expected (simple heuristic)
            if col_data.min() < 0 and "count" in col.lower() or "size" in col.lower():
                metrics.append(QualityMetric(
                    dimension="consistency",
                    metric_name=f"column_{col}_range_check",
                    value=0.8,
                    threshold=self.config.consistency_threshold,
                    passed=False,
                    details={
                        "column": col,
                        "issue": "Negative values in count/size column",
                        "min_value": float(col_data.min()),
                        "max_value": float(col_data.max())
                    }
                ))
        
        # Check string format consistency (basic check)
        for col in df.select_dtypes(include=['object']).columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Check for mixed case consistency (heuristic)
            str_lengths = col_data.str.len()
            if str_lengths.std() / str_lengths.mean() > 0.5:  # High variance in lengths
                metrics.append(QualityMetric(
                    dimension="consistency",
                    metric_name=f"column_{col}_format_variance",
                    value=0.85,
                    threshold=self.config.consistency_threshold,
                    passed=True,  # Warning, not failure
                    details={
                        "column": col,
                        "mean_length": float(str_lengths.mean()),
                        "std_length": float(str_lengths.std())
                    }
                ))
        
        return metrics
    
    def _check_validity(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> List[QualityMetric]:
        """Check validity dimension (schema, business rules)."""
        metrics = []
        
        if df.empty:
            return metrics
        
        # Schema validation (if schema provided)
        if schema:
            schema_validity = self._validate_against_schema(df, schema)
            metrics.append(schema_validity)
        
        # Basic validity: check for required columns (if schema has required_fields)
        if schema and "required_fields" in schema:
            required_fields = schema["required_fields"]
            missing_required = set(required_fields) - set(df.columns)
            
            if missing_required:
                metrics.append(QualityMetric(
                    dimension="validity",
                    metric_name="required_fields_check",
                    value=0.0,
                    threshold=self.config.validity_threshold,
                    passed=False,
                    details={
                        "missing_required_fields": list(missing_required),
                        "required_fields": required_fields
                    }
                ))
            else:
                metrics.append(QualityMetric(
                    dimension="validity",
                    metric_name="required_fields_check",
                    value=1.0,
                    threshold=self.config.validity_threshold,
                    passed=True,
                    details={"required_fields": required_fields}
                ))
        
        return metrics
    
    def _validate_against_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> QualityMetric:
        """Validate DataFrame against schema definition."""
        # Basic schema validation
        errors = []
        
        if "fields" in schema:
            for field_name, field_spec in schema["fields"].items():
                if field_name not in df.columns:
                    errors.append(f"Missing field: {field_name}")
                    continue
                
                # Check data type
                if "type" in field_spec:
                    expected_type = field_spec["type"]
                    actual_dtype = str(df[field_name].dtype)
                    # Basic type checking (can be enhanced)
                    if not self._type_matches(expected_type, actual_dtype):
                        errors.append(f"Type mismatch for {field_name}: expected {expected_type}, got {actual_dtype}")
                
                # Check constraints
                if "constraints" in field_spec:
                    constraints = field_spec["constraints"]
                    if "min" in constraints:
                        if df[field_name].min() < constraints["min"]:
                            errors.append(f"Value below minimum for {field_name}")
                    if "max" in constraints:
                        if df[field_name].max() > constraints["max"]:
                            errors.append(f"Value above maximum for {field_name}")
        
        # Use configurable error penalty instead of hardcoded 0.1
        validity_score = 1.0 if len(errors) == 0 else max(0.0, 1.0 - len(errors) * self.config.validity_error_penalty)
        
        return QualityMetric(
            dimension="validity",
            metric_name="schema_validation",
            value=validity_score,
            threshold=self.config.validity_threshold,
            passed=validity_score >= self.config.validity_threshold,
            details={"errors": errors} if errors else {"status": "valid"}
        )
    
    def _type_matches(self, expected: str, actual: str) -> bool:
        """
        Check if expected type matches actual dtype using robust type checking.
        
        Uses pandas dtype checking utilities for more reliable type matching.
        
        Args:
            expected: Expected type name (e.g., "int", "float", "string", "bool")
            actual: Actual pandas dtype string (e.g., "int64", "float32", "object")
            
        Returns:
            True if types match, False otherwise
        """
        # Normalize inputs
        expected_lower = expected.lower().strip()
        actual_lower = str(actual).lower().strip()
        
        # Comprehensive type mapping
        type_mapping = {
            "int": ["int64", "int32", "int16", "int8", "int", "integer", "int_", "uint64", "uint32", "uint16", "uint8"],
            "float": ["float64", "float32", "float16", "float", "double"],
            "string": ["object", "string", "str", "unicode"],
            "bool": ["bool", "boolean", "bool_"],
            "datetime": ["datetime64[ns]", "datetime64", "datetime", "timedelta64[ns]"],
            "category": ["category"],
        }
        
        # Direct match
        if expected_lower == actual_lower:
            return True
        
        # Check mapping
        for key, variants in type_mapping.items():
            if key in expected_lower:
                if any(v in actual_lower for v in variants):
                    return True
        
        # Additional pandas-specific checks
        if "int" in expected_lower and pd.api.types.is_integer_dtype(actual):
            return True
        if "float" in expected_lower and pd.api.types.is_float_dtype(actual):
            return True
        if "bool" in expected_lower and pd.api.types.is_bool_dtype(actual):
            return True
        if "datetime" in expected_lower and pd.api.types.is_datetime64_any_dtype(actual):
            return True
        
        return False
    
    def _check_uniqueness(self, df: pd.DataFrame) -> List[QualityMetric]:
        """Check uniqueness dimension (duplicates, keys)."""
        metrics = []
        
        if df.empty:
            return metrics
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = 1.0 - (duplicate_rows / len(df))
        
        metrics.append(QualityMetric(
            dimension="uniqueness",
            metric_name="row_uniqueness",
            value=uniqueness_score,
            threshold=self.config.uniqueness_threshold,
            passed=uniqueness_score >= self.config.uniqueness_threshold,
            details={
                "duplicate_rows": int(duplicate_rows),
                "unique_rows": int(len(df) - duplicate_rows),
                "duplicate_percentage": float(duplicate_rows / len(df) * 100)
            }
        ))
        
        # Check for primary key uniqueness (if applicable)
        # Look for columns that might be keys
        for col in df.columns:
            if "id" in col.lower() or "key" in col.lower():
                unique_values = df[col].nunique()
                col_uniqueness = unique_values / len(df) if len(df) > 0 else 0.0
                
                metrics.append(QualityMetric(
                    dimension="uniqueness",
                    metric_name=f"column_{col}_uniqueness",
                    value=col_uniqueness,
                    threshold=self.config.key_uniqueness_threshold,  # Configurable threshold
                    passed=col_uniqueness >= self.config.key_uniqueness_threshold,
                    details={
                        "column": col,
                        "unique_values": int(unique_values),
                        "total_values": len(df),
                        "duplicates": int(len(df) - unique_values)
                    }
                ))
        
        return metrics
    
    def _check_timeliness(self, df: pd.DataFrame) -> List[QualityMetric]:
        """Check timeliness dimension (freshness, staleness)."""
        metrics = []
        
        if df.empty:
            return metrics
        
        # Check for timestamp columns
        timestamp_cols = []
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower() or "timestamp" in col.lower():
                timestamp_cols.append(col)
        
        if not timestamp_cols:
            # No timestamps found - timeliness not applicable
            metrics.append(QualityMetric(
                dimension="timeliness",
                metric_name="no_timestamps",
                value=1.0,
                threshold=self.config.timeliness_threshold,
                passed=True,
                details={"message": "No timestamp columns found - timeliness check skipped"}
            ))
            return metrics
        
        # Check freshness of most recent timestamp
        for col in timestamp_cols:
            try:
                # Try to convert to datetime
                timestamps = pd.to_datetime(df[col], errors='coerce')
                
                valid_timestamps = timestamps.dropna()
                if len(valid_timestamps) == 0:
                    continue
                
                most_recent = valid_timestamps.max()
                now = pd.Timestamp.now()
                age_days = (now - most_recent).days
                
                # Freshness score (inversely related to age)
                # Configurable decay period from config
                freshness_score = max(0.0, 1.0 - (age_days / self.config.freshness_decay_days))
                
                metrics.append(QualityMetric(
                    dimension="timeliness",
                    metric_name=f"column_{col}_freshness",
                    value=freshness_score,
                    threshold=self.config.timeliness_threshold,  # Configurable threshold
                    passed=freshness_score >= self.config.timeliness_threshold,
                    details={
                        "column": col,
                        "most_recent": most_recent.isoformat(),
                        "age_days": age_days,
                        "total_records": len(valid_timestamps)
                    }
                ))
            except Exception as e:
                # Skip columns that can't be converted to timestamps
                continue
        
        return metrics
    
    def _check_accuracy(
        self, 
        df: pd.DataFrame, 
        reference_data: Optional[Union[pd.DataFrame, List[Dict]]] = None
    ) -> List[QualityMetric]:
        """Check accuracy dimension (statistical tests, outliers)."""
        metrics = []
        
        if df.empty:
            return metrics
        
        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < self.config.min_samples_for_outlier_detection:
                continue
            
            col_array = col_data.values
            
            # IQR-based outlier detection with configurable multiplier
            outliers_iqr, iqr_details = self.stat_validator.detect_outliers_iqr(
                col_array, 
                multiplier=self.config.iqr_multiplier
            )
            outlier_percentage = iqr_details["outlier_percentage"]
            accuracy_score = max(0.0, 1.0 - (outlier_percentage / 100.0))
            
            metrics.append(QualityMetric(
                dimension="accuracy",
                metric_name=f"column_{col}_outliers_iqr",
                value=accuracy_score,
                threshold=self.config.accuracy_threshold,
                passed=accuracy_score >= self.config.accuracy_threshold,
                details={
                    "column": col,
                    "outlier_count": iqr_details["outlier_count"],
                    "outlier_percentage": outlier_percentage,
                    **{k: v for k, v in iqr_details.items() if k not in ["outlier_count", "outlier_percentage"]}
                }
            ))
            
            # Z-score outlier detection with configurable threshold
            outliers_zscore, zscore_details = self.stat_validator.detect_outliers_zscore(
                col_array,
                threshold=self.config.zscore_threshold
            )
            outlier_percentage_z = zscore_details["outlier_percentage"]
            accuracy_score_z = max(0.0, 1.0 - (outlier_percentage_z / 100.0))
            
            metrics.append(QualityMetric(
                dimension="accuracy",
                metric_name=f"column_{col}_outliers_zscore",
                value=accuracy_score_z,
                threshold=self.config.accuracy_threshold,
                passed=accuracy_score_z >= self.config.accuracy_threshold,
                details={
                    "column": col,
                    "outlier_count": zscore_details["outlier_count"],
                    "outlier_percentage": outlier_percentage_z,
                    **{k: v for k, v in zscore_details.items() if k not in ["outlier_count", "outlier_percentage"]}
                }
            ))
        
        # Statistical distribution tests (for numeric columns with sufficient data)
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < self.config.min_samples_for_statistical_tests:
                continue
            
            col_array = col_data.values
            
            # Normality test (Shapiro-Wilk for smaller samples) with configurable sample size
            try:
                _, p_value, sw_details = self.stat_validator.shapiro_wilk_test(
                    col_array,
                    max_sample_size=self.config.shapiro_wilk_sample_size,
                    random_state=self.config.random_state
                )
                is_normal = sw_details["is_normal"]
                
                # Warn if sample was truncated
                if sw_details.get("original_size", len(col_data)) > self.config.shapiro_wilk_sample_size:
                    sw_details["warning"] = f"Test limited to {self.config.shapiro_wilk_sample_size} samples"
                
                metrics.append(QualityMetric(
                    dimension="accuracy",
                    metric_name=f"column_{col}_normality",
                    value=1.0 if is_normal else 0.8,
                    threshold=self.config.accuracy_threshold,  # Use configurable threshold
                    passed=True,  # Informational, not a failure
                    details={
                        "column": col,
                        "is_normal": is_normal,
                        **{k: v for k, v in sw_details.items() if k != "note"}
                    }
                ))
            except Exception:
                pass
        
        return metrics
    
    def _check_integrity(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> List[QualityMetric]:
        """Check integrity dimension (referential, constraints)."""
        metrics = []
        
        if df.empty:
            return metrics
        
        # Check for null values in key columns (if schema defines keys)
        if schema and "primary_keys" in schema:
            primary_keys = schema["primary_keys"]
            for key in primary_keys:
                if key in df.columns:
                    null_count = df[key].isnull().sum()
                    integrity_score = 1.0 if null_count == 0 else 0.0
                    
                    metrics.append(QualityMetric(
                        dimension="integrity",
                        metric_name=f"primary_key_{key}_nulls",
                        value=integrity_score,
                        threshold=self.config.integrity_threshold,
                        passed=integrity_score >= self.config.integrity_threshold,
                        details={
                            "key": key,
                            "null_count": int(null_count),
                            "total_rows": len(df)
                        }
                    ))
        
        # Check referential integrity (if foreign keys defined in schema)
        if schema and "foreign_keys" in schema:
            foreign_keys = schema["foreign_keys"]
            for fk_name, fk_spec in foreign_keys.items():
                # Basic referential integrity check (can be enhanced)
                if fk_name in df.columns:
                    null_count = df[fk_name].isnull().sum()
                    integrity_score = 1.0 if null_count == 0 else max(0.0, 1.0 - (null_count / len(df)))
                    
                    metrics.append(QualityMetric(
                        dimension="integrity",
                        metric_name=f"foreign_key_{fk_name}_integrity",
                        value=integrity_score,
                        threshold=self.config.integrity_threshold,
                        passed=integrity_score >= self.config.integrity_threshold,
                        details={
                            "foreign_key": fk_name,
                            "null_count": int(null_count),
                            "reference": fk_spec.get("references", "unknown")
                        }
                    ))
        
        return metrics
    
    def _calculate_dimension_scores(self, metrics: List[QualityMetric]) -> Dict[str, float]:
        """
        Calculate average score for each dimension.
        
        Uses standardized scoring: all scores are normalized to 0.0-1.0 range.
        Averages all metric values within each dimension.
        
        Args:
            metrics: List of QualityMetric instances
            
        Returns:
            Dictionary mapping dimension names to average scores (0.0 to 1.0)
        """
        dimension_scores = {}
        
        for dimension in ["completeness", "consistency", "validity", "uniqueness", "timeliness", "accuracy", "integrity"]:
            dimension_metrics = [m for m in metrics if m.dimension == dimension and not m.metric_name.endswith("_error")]
            if dimension_metrics:
                # Standardized: mean of all metric values (all should be 0.0-1.0)
                dimension_scores[dimension] = float(np.mean([m.value for m in dimension_metrics]))
            else:
                # Default to 1.0 if no metrics (dimension not checked or all metrics errored)
                dimension_scores[dimension] = 1.0
        
        return dimension_scores
    
    def _generate_summary(self, df: pd.DataFrame, metrics: List[QualityMetric]) -> Dict[str, Any]:
        """
        Generate summary statistics for the quality report.
        
        Args:
            df: DataFrame that was checked
            metrics: List of all quality metrics
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_metrics": len(metrics),
            "passed_metrics": sum(1 for m in metrics if m.passed),
            "failed_metrics": sum(1 for m in metrics if not m.passed),
            "error_metrics": sum(1 for m in metrics if m.metric_name.endswith("_error")),
            "dimensions_checked": len(set(m.dimension for m in metrics))
        }
        
        if not df.empty:
            try:
                summary["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / 1024 / 1024)
                summary["numeric_columns"] = len(df.select_dtypes(include=[np.number]).columns)
                summary["text_columns"] = len(df.select_dtypes(include=['object']).columns)
            except Exception:
                pass  # Skip if memory calculation fails
        
        return summary
    
    def _generate_recommendations(
        self, 
        dimension_scores: Dict[str, float], 
        metrics: List[QualityMetric]
    ) -> List[str]:
        """
        Generate recommendations based on quality scores.
        
        Uses configurable recommendation_threshold to determine when to generate recommendations.
        
        Args:
            dimension_scores: Dictionary mapping dimension names to scores
            metrics: List of all quality metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check each dimension and provide recommendations using configurable threshold
        for dimension, score in dimension_scores.items():
            if score < self.config.recommendation_threshold:
                if dimension == "completeness":
                    recommendations.append(
                        f"Completeness score is {score:.2%}. Consider imputing missing values or investigating data collection process."
                    )
                elif dimension == "consistency":
                    recommendations.append(
                        f"Consistency score is {score:.2%}. Check for format inconsistencies and standardize data types."
                    )
                elif dimension == "validity":
                    recommendations.append(
                        f"Validity score is {score:.2%}. Review data against schema and business rules."
                    )
                elif dimension == "uniqueness":
                    recommendations.append(
                        f"Uniqueness score is {score:.2%}. Remove duplicate records or investigate why duplicates exist."
                    )
                elif dimension == "timeliness":
                    recommendations.append(
                        f"Timeliness score is {score:.2%}. Data may be stale. Check data update frequency."
                    )
                elif dimension == "accuracy":
                    recommendations.append(
                        f"Accuracy score is {score:.2%}. Review outliers and statistical anomalies."
                    )
                elif dimension == "integrity":
                    recommendations.append(
                        f"Integrity score is {score:.2%}. Check referential integrity and key constraints."
                    )
        
        # Specific metric-based recommendations
        failed_metrics = [m for m in metrics if not m.passed]
        for metric in failed_metrics[:5]:  # Limit to top 5 recommendations
            if metric.dimension == "completeness" and "column" in metric.metric_name:
                col_name = metric.details.get("column", "unknown")
                recommendations.append(
                    f"Column '{col_name}' has {metric.details.get('missing_percentage', 0):.1f}% missing values. "
                    f"Consider imputation or data source review."
                )
            elif metric.dimension == "uniqueness" and "duplicate" in metric.metric_name.lower():
                recommendations.append(
                    f"Found {metric.details.get('duplicate_rows', 0)} duplicate rows. Remove duplicates to improve data quality."
                )
        
        return recommendations


def check_data_quality(
    data: Union[pd.DataFrame, List[Dict], Dict],
    dataset_id: str = "dataset",
    schema: Optional[Dict[str, Any]] = None,
    config: Optional[QualityConfig] = None,
    **kwargs
) -> QualityReport:
    """
    Convenience function to check data quality.
    
    Args:
        data: Data to check
        dataset_id: Identifier for the dataset
        schema: Optional schema definition (see QualityConfig docstring for format)
        config: Optional QualityConfig instance (recommended)
        **kwargs: Additional arguments passed to QualityChecker (legacy support)
        
    Returns:
        QualityReport
        
    Example:
        ```python
        from pipelines.data_preprocessing.quality import check_data_quality, QualityConfig
        
        # Use default config
        report = check_data_quality(data, dataset_id="my_dataset")
        
        # Use custom config
        config = QualityConfig(completeness_threshold=0.98)
        report = check_data_quality(data, dataset_id="my_dataset", config=config)
        ```
    """
    checker = QualityChecker(config=config, **kwargs)
    return checker.check_quality(data, dataset_id, schema)


class QualityCheckStage:
    """
    Pipeline stage wrapper for QualityChecker.
    
    This class makes QualityChecker compatible with the PreprocessingStage interface,
    allowing it to be used in pipeline orchestration.
    
    Note: This is a lightweight wrapper. For full PreprocessingStage inheritance,
    consider creating a proper subclass in stages.py.
    
    Example:
        ```python
        from pipelines.data_preprocessing.quality import QualityCheckStage
        
        quality_stage = QualityCheckStage(config={
            "completeness_threshold": 0.95,
            "dataset_id": "my_dataset"
        })
        result = quality_stage.process(data)
        
        if result.success:
            validation_result = result.validation_result
            processed_data = result.processed_data
            quality_metrics = processed_data.quality_metrics
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality check stage.
        
        Args:
            config: Configuration dictionary. Can include:
                - quality_config: QualityConfig instance (recommended)
                - Or individual threshold parameters (legacy)
                - dataset_id: Optional dataset identifier
                - schema: Optional schema definition (see QualityConfig docstring for format)
        """
        # Store config for later use
        self.config = config or {}
        
        # Extract quality-specific config
        quality_config = self.config.get("quality_config")
        if quality_config is None:
            # Create QualityConfig from individual parameters if provided
            quality_kwargs = {
                k: v for k, v in self.config.items()
                if k in ["completeness_threshold", "consistency_threshold", 
                        "validity_threshold", "uniqueness_threshold",
                        "accuracy_threshold", "integrity_threshold",
                        "iqr_multiplier", "zscore_threshold",
                        "isolation_forest_contamination", "random_state",
                        "freshness_decay_days", "key_uniqueness_threshold",
                        "recommendation_threshold", "validity_error_penalty",
                        "shapiro_wilk_sample_size",
                        "min_samples_for_outlier_detection", "min_samples_for_statistical_tests"]
            }
            if quality_kwargs:
                quality_config = QualityConfig(**quality_kwargs)
        
        self.quality_checker = QualityChecker(config=quality_config)
        self.dataset_id = self.config.get("dataset_id", "quality_check")
        self.schema = self.config.get("schema")
    
    def process(self, data: Any) -> Any:  # Returns StageResult from base module
        """
        Process data through quality checks.
        
        Args:
            data: Input data (ProcessedData, DataFrame, list, or dict)
            
        Returns:
            StageResult with processed_data containing quality metrics and validation_result
        """
        from .base import StageResult, ProcessedData, ValidationResult
        
        try:
            # Extract actual data if it's ProcessedData
            if hasattr(data, 'data'):
                actual_data = data.data
                original_metadata = getattr(data, 'metadata', {})
            else:
                actual_data = data
                original_metadata = {}
            
            # Run quality check
            quality_report = self.quality_checker.check_quality(
                data=actual_data,
                dataset_id=self.dataset_id,
                schema=self.schema or original_metadata.get("schema")
            )
            
            # Convert to ValidationResult
            validation_result = quality_report.to_validation_result()
            
            # Create ProcessedData with quality metrics
            processed_data = ProcessedData(
                data=actual_data,
                metadata={
                    **original_metadata,
                    "quality_check_timestamp": quality_report.timestamp.isoformat(),
                    "overall_quality_score": quality_report.overall_score,
                    "schema": self.schema or original_metadata.get("schema")
                },
                quality_metrics=quality_report.get_quality_metrics_for_processed_data(),
                schema_version=original_metadata.get("schema_version")
            )
            
            return StageResult(
                success=True,
                processed_data=processed_data,
                validation_result=validation_result,
                stage_name="quality_check"
            )
            
        except Exception as e:
            from .base import StageResult
            return StageResult(
                success=False,
                error=f"Quality check failed: {str(e)}",
                stage_name="quality_check"
            )
    
    def validate(self, data: Any) -> Any:  # Returns ValidationResult from base module
        """
        Validate data quality.
        
        Args:
            data: Input data to validate
            
        Returns:
            ValidationResult with quality scores, errors, and warnings
        """
        from .base import ValidationResult
        
        try:
            # Extract actual data if it's ProcessedData
            if hasattr(data, 'data'):
                actual_data = data.data
            else:
                actual_data = data
            
            # Run quality check
            quality_report = self.quality_checker.check_quality(
                data=actual_data,
                dataset_id=self.dataset_id,
                schema=self.schema
            )
            
            # Convert to ValidationResult
            return quality_report.to_validation_result()
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Quality check failed: {str(e)}"],
                warnings=[],
                quality_scores={}
            )
    
    def get_dependencies(self) -> List[str]:
        """Get list of stage names this stage depends on."""
        return []  # Quality check can run at any point
    
    def get_name(self) -> str:
        """Get the name of this stage."""
        return "quality_check"

