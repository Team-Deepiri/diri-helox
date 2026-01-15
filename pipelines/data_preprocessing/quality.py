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
    """Comprehensive quality report for a dataset."""
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


class StatisticalValidator:
    """
    Statistical validation methods for data quality checks.
    
    Provides statistical tests and calculations for:
    - Outlier detection
    - Distribution tests
    - Correlation analysis
    - Hypothesis testing
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
        random_state: int = 42
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
    def shapiro_wilk_test(data: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
        """
        Perform Shapiro-Wilk test for normality (limited to 5000 samples).
        
        Args:
            data: Numeric data array
            
        Returns:
            Tuple of (statistic, p_value, details_dict)
        """
        # Limit sample size for Shapiro-Wilk
        if len(data) > 5000:
            sample_data = np.random.choice(data, 5000, replace=False)
            details = {"note": "Test performed on 5000 sample subset"}
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
    """
    
    def __init__(
        self,
        completeness_threshold: float = 0.95,
        consistency_threshold: float = 0.90,
        validity_threshold: float = 1.0,
        uniqueness_threshold: float = 0.98,
        accuracy_threshold: float = 0.90,
        integrity_threshold: float = 1.0
    ):
        """
        Initialize quality checker with thresholds.
        
        Args:
            completeness_threshold: Minimum acceptable completeness score (0-1)
            consistency_threshold: Minimum acceptable consistency score (0-1)
            validity_threshold: Minimum acceptable validity score (0-1)
            uniqueness_threshold: Minimum acceptable uniqueness score (0-1)
            accuracy_threshold: Minimum acceptable accuracy score (0-1)
            integrity_threshold: Minimum acceptable integrity score (0-1)
        """
        self.completeness_threshold = completeness_threshold
        self.consistency_threshold = consistency_threshold
        self.validity_threshold = validity_threshold
        self.uniqueness_threshold = uniqueness_threshold
        self.accuracy_threshold = accuracy_threshold
        self.integrity_threshold = integrity_threshold
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
        df = self._to_dataframe(data)
        
        metrics = []
        
        # 1. Completeness checks
        completeness_metrics = self._check_completeness(df)
       
        # 2. Consistency checks
        consistency_metrics = self._check_consistency(df)
        metrics.extend(consistency_metrics)
        
        # 3. Validity checks
        validity_metrics = self._check_validity(df, schema)
        metrics.extend(validity_metrics)
        
        # 4. Uniqueness checks
        uniqueness_metrics = self._check_uniqueness(df)
        metrics.extend(uniqueness_metrics)
        
        # 5. Timeliness checks
        timeliness_metrics = self._check_timeliness(df)
        metrics.extend(timeliness_metrics)
        
        # 6. Accuracy checks
        accuracy_metrics = self._check_accuracy(df, reference_data)
        metrics.extend(accuracy_metrics)
        
        # 7. Integrity checks
        integrity_metrics = self._check_integrity(df, schema)
        metrics.extend(integrity_metrics)
        
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
   []
    def _to_dataframe(self, data: Union[pd.DataFrame, List[Dict], Dict]) -> pd.DataFrame:
        """Convert various data formats to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            if len(data) == 0:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                raise ValueError("List data must contain dictionaries")
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _check_completeness(self, df: pd.DataFrame) -> List[QualityMetric]:
        """Check completeness dimension (missing values, coverage)."""
        metrics = []
        
        if df.empty:
            metrics.append(QualityMetric(
                dimension="completeness",
                metric_name="empty_dataset",
                value=0.0,
                threshold=self.completeness_threshold,
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
            threshold=self.completeness_threshold,
            passed=completeness_score >= self.completeness_threshold,
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
                threshold=self.completeness_threshold,
                passed=col_completeness >= self.completeness_threshold,
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
            threshold=self.completeness_threshold,
            passed=row_completeness >= self.completeness_threshold,
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
            threshold=self.consistency_threshold,
            passed=type_consistency_score >= self.consistency_threshold,
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
                    threshold=self.consistency_threshold,
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
                    threshold=self.consistency_threshold,
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
                    threshold=self.validity_threshold,
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
                    threshold=self.validity_threshold,
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
        
        validity_score = 1.0 if len(errors) == 0 else max(0.0, 1.0 - len(errors) * 0.1)
        
        return QualityMetric(
            dimension="validity",
            metric_name="schema_validation",
            value=validity_score,
            threshold=self.validity_threshold,
            passed=validity_score >= self.validity_threshold,
            details={"errors": errors} if errors else {"status": "valid"}
        )
    
    def _type_matches(self, expected: str, actual: str) -> bool:
        """Check if expected type matches actual dtype."""
        type_mapping = {
            "int": ["int64", "int32", "int"],
            "float": ["float64", "float32", "float"],
            "string": ["object", "string"],
            "bool": ["bool", "boolean"]
        }
        
        for key, variants in type_mapping.items():
            if expected.lower() in key and any(v in actual.lower() for v in variants):
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
            threshold=self.uniqueness_threshold,
            passed=uniqueness_score >= self.uniqueness_threshold,
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
                    threshold=0.95,  # Keys should be highly unique
                    passed=col_uniqueness >= 0.95,
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
                threshold=1.0,
                passed=True,
                details={"message": "No timestamp columns found - timeliness check skipped"}
            ))
            return metrics
        
        # Check freshness of most recent timestamp
        for col in timestamp_cols:
            try:
                # Try to convert to datetime
                if df[col].dtype == 'object':
                    timestamps = pd.to_datetime(df[col], errors='coerce')
                else:
                    timestamps = pd.to_datetime(df[col], errors='coerce')
                
                valid_timestamps = timestamps.dropna()
                if len(valid_timestamps) == 0:
                    continue
                
                most_recent = valid_timestamps.max()
                now = pd.Timestamp.now()
                age_days = (now - most_recent).days
                
                # Freshness score (inversely related to age)
                # Consider data "fresh" if less than 7 days old
                freshness_score = max(0.0, 1.0 - (age_days / 30.0))  # Decay over 30 days
                
                metrics.append(QualityMetric(
                    dimension="timeliness",
                    metric_name=f"column_{col}_freshness",
                    value=freshness_score,
                    threshold=0.7,  # Data should be less than 7 days old
                    passed=freshness_score >= 0.7,
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
            if len(col_data) < 4:  # Need at least 4 points for outlier detection
                continue
            
            col_array = col_data.values
            
            # IQR-based outlier detection
            outliers_iqr, iqr_details = self.stat_validator.detect_outliers_iqr(col_array)
            outlier_percentage = iqr_details["outlier_percentage"]
            accuracy_score = max(0.0, 1.0 - (outlier_percentage / 100.0))
            
            metrics.append(QualityMetric(
                dimension="accuracy",
                metric_name=f"column_{col}_outliers_iqr",
                value=accuracy_score,
                threshold=self.accuracy_threshold,
                passed=accuracy_score >= self.accuracy_threshold,
                details={
                    "column": col,
                    "outlier_count": iqr_details["outlier_count"],
                    "outlier_percentage": outlier_percentage,
                    **{k: v for k, v in iqr_details.items() if k not in ["outlier_count", "outlier_percentage"]}
                }
            ))
            
            # Z-score outlier detection
            outliers_zscore, zscore_details = self.stat_validator.detect_outliers_zscore(col_array)
            outlier_percentage_z = zscore_details["outlier_percentage"]
            accuracy_score_z = max(0.0, 1.0 - (outlier_percentage_z / 100.0))
            
            metrics.append(QualityMetric(
                dimension="accuracy",
                metric_name=f"column_{col}_outliers_zscore",
                value=accuracy_score_z,
                threshold=self.accuracy_threshold,
                passed=accuracy_score_z >= self.accuracy_threshold,
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
            if len(col_data) < 30:  # Need sufficient data for statistical tests
                continue
            
            col_array = col_data.values
            
            # Normality test (Shapiro-Wilk for smaller samples)
            try:
                _, p_value, sw_details = self.stat_validator.shapiro_wilk_test(col_array)
                is_normal = sw_details["is_normal"]
                
                metrics.append(QualityMetric(
                    dimension="accuracy",
                    metric_name=f"column_{col}_normality",
                    value=1.0 if is_normal else 0.8,
                    threshold=0.7,
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
                        threshold=self.integrity_threshold,
                        passed=integrity_score >= self.integrity_threshold,
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
                        threshold=self.integrity_threshold,
                        passed=integrity_score >= self.integrity_threshold,
                        details={
                            "foreign_key": fk_name,
                            "null_count": int(null_count),
                            "reference": fk_spec.get("references", "unknown")
                        }
                    ))
        
        return metrics
    
    def _calculate_dimension_scores(self, metrics: List[QualityMetric]) -> Dict[str, float]:
        """Calculate average score for each dimension."""
        dimension_scores = {}
        
        for dimension in ["completeness", "consistency", "validity", "uniqueness", "timeliness", "accuracy", "integrity"]:
            dimension_metrics = [m for m in metrics if m.dimension == dimension]
            if dimension_metrics:
                # Weighted average (can be customized)
                dimension_scores[dimension] = np.mean([m.value for m in dimension_metrics])
            else:
                dimension_scores[dimension] = 1.0  # Default if no metrics
        
        return dimension_scores
    
    def _generate_summary(self, df: pd.DataFrame, metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_metrics": len(metrics),
            "passed_metrics": sum(1 for m in metrics if m.passed),
            "failed_metrics": sum(1 for m in metrics if not m.passed),
            "dimensions_checked": len(set(m.dimension for m in metrics))
        }
        
        if not df.empty:
            summary["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / 1024 / 1024)
            summary["numeric_columns"] = len(df.select_dtypes(include=[np.number]).columns)
            summary["text_columns"] = len(df.select_dtypes(include=['object']).columns)
        
        return summary
    
    def _generate_recommendations(
        self, 
        dimension_scores: Dict[str, float], 
        metrics: List[QualityMetric]
    ) -> List[str]:
        """Generate recommendations based on quality scores."""
        recommendations = []
        
        # Check each dimension and provide recommendations
        for dimension, score in dimension_scores.items():
            if score < 0.8:
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
    **kwargs
) -> QualityReport:
    """
    Convenience function to check data quality.
    
    Args:
        data: Data to check
        dataset_id: Identifier for the dataset
        schema: Optional schema definition
        **kwargs: Additional arguments passed to QualityChecker
        
    Returns:
        QualityReport
    """
    checker = QualityChecker(**kwargs)
    return checker.check_quality(data, dataset_id, schema)

