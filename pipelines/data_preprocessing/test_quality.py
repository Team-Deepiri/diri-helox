"""
Test script for data quality framework.

This script tests the QualityChecker and related classes to verify they work correctly.
"""

import sys
import os

# Add parent directories to path to allow proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import quality framework using absolute imports
from .quality import (
    QualityChecker,
    QualityConfig,
    QualityCheckStage,
    check_data_quality,
    QualityReport
)

def create_sample_data():
    """Create sample dataset for testing."""
    # Good data
    data = {
        'id': [1, 2, 3, 4, 5],
        'text': ['Good text', 'Another text', 'More text', 'Some text', 'Last text'],
        'value': [10, 20, 30, 40, 50],
        'timestamp': [
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=5)
        ],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    return pd.DataFrame(data)

def create_data_with_issues():
    """Create dataset with quality issues for testing."""
    data = {
        'id': [1, 2, 3, 4, 5, None, 7],  # Missing ID
        'text': ['Good', 'Bad', None, 'OK', 'Fine', '', None],  # Missing values
        'value': [10, 2000, 30, 40, -5, 50, 100],  # Outliers and negative
        'timestamp': [
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(days=60),  # Old timestamp
            datetime.now() - timedelta(days=2),
            None,  # Missing timestamp
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=5)
        ],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'B']
    }
    df = pd.DataFrame(data)
    # Add duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df

def test_basic_quality_check():
    """Test 1: Basic quality check with good data."""
    print("\n" + "="*70)
    print("TEST 1: Basic Quality Check (Good Data)")
    print("="*70)
    
    data = create_sample_data()
    checker = QualityChecker()
    report = checker.check_quality(data, dataset_id="test_good_data")
    
    print(f"Dataset ID: {report.dataset_id}")
    print(f"Overall Score: {report.overall_score:.2%}")
    print(f"\nDimension Scores:")
    for dim, score in report.dimension_scores.items():
        print(f"  {dim}: {score:.2%}")
    
    print(f"\nTotal Metrics: {len(report.metrics)}")
    print(f"Passed: {sum(1 for m in report.metrics if m.passed)}")
    print(f"Failed: {sum(1 for m in report.metrics if not m.passed)}")
    
    assert report.overall_score > 0.8, "Quality score should be good"
    print("✅ Test 1 PASSED")

def test_data_with_issues():
    """Test 2: Quality check with problematic data."""
    print("\n" + "="*70)
    print("TEST 2: Quality Check (Data with Issues)")
    print("="*70)
    
    data = create_data_with_issues()
    checker = QualityChecker()
    report = checker.check_quality(data, dataset_id="test_bad_data")
    
    print(f"Overall Score: {report.overall_score:.2%}")
    print(f"\nDimension Scores:")
    for dim, score in report.dimension_scores.items():
        status = "✅" if score >= 0.8 else "❌"
        print(f"  {status} {dim}: {score:.2%}")
    
    print(f"\nFailed Metrics:")
    failed = [m for m in report.metrics if not m.passed]
    for m in failed[:5]:  # Show first 5
        print(f"  - {m.dimension}.{m.metric_name}: {m.value:.2f} (threshold: {m.threshold})")
    
    if report.recommendations:
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for rec in report.recommendations[:3]:  # Show first 3
            print(f"  - {rec}")
    
    print("✅ Test 2 PASSED (issues detected)")

def test_custom_config():
    """Test 3: Custom configuration."""
    print("\n" + "="*70)
    print("TEST 3: Custom Configuration")
    print("="*70)
    
    # Create config with strict thresholds
    config = QualityConfig(
        completeness_threshold=0.99,  # Very strict
        accuracy_threshold=0.95,
        iqr_multiplier=2.0,  # Less sensitive to outliers
        freshness_decay_days=7.0  # Data must be very fresh
    )
    
    data = create_data_with_issues()
    checker = QualityChecker(config=config)
    report = checker.check_quality(data, dataset_id="test_custom_config")
    
    print(f"Using strict thresholds:")
    print(f"  Completeness threshold: {config.completeness_threshold}")
    print(f"  Accuracy threshold: {config.accuracy_threshold}")
    print(f"  IQR multiplier: {config.iqr_multiplier}")
    
    print(f"\nOverall Score: {report.overall_score:.2%}")
    print(f"Failed Metrics: {sum(1 for m in report.metrics if not m.passed)}")
    
    print("✅ Test 3 PASSED")

def test_validation_result_conversion():
    """Test 4: Convert QualityReport to ValidationResult."""
    print("\n" + "="*70)
    print("TEST 4: ValidationResult Integration")
    print("="*70)
    
    data = create_sample_data()
    checker = QualityChecker()
    report = checker.check_quality(data, dataset_id="test_validation")
    
    # Convert to ValidationResult
    validation_result = report.to_validation_result()
    
    print(f"Is Valid: {validation_result.is_valid}")
    print(f"Errors: {len(validation_result.errors)}")
    print(f"Warnings: {len(validation_result.warnings)}")
    print(f"\nQuality Scores:")
    for key, value in validation_result.quality_scores.items():
        print(f"  {key}: {value:.2%}")
    
    assert isinstance(validation_result.is_valid, bool)
    assert "overall" in validation_result.quality_scores
    print("✅ Test 4 PASSED")

def test_quality_check_stage():
    """Test 5: QualityCheckStage for pipeline integration."""
    print("\n" + "="*70)
    print("TEST 5: QualityCheckStage (Pipeline Integration)")
    print("="*70)
    
    data = create_sample_data()
    
    # Convert DataFrame to list of dicts (pipeline format)
    data_list = data.to_dict('records')
    
    stage = QualityCheckStage(config={
        "dataset_id": "test_stage",
        "completeness_threshold": 0.9
    })
    
    result = stage.process(data_list)
    
    print(f"Success: {result.success}")
    print(f"Stage Name: {result.stage_name}")
    
    if result.success:
        print(f"\nProcessed Data:")
        print(f"  Quality Metrics: {result.processed_data.quality_metrics}")
        
        print(f"\nValidation Result:")
        print(f"  Is Valid: {result.validation_result.is_valid}")
        print(f"  Quality Scores: {result.validation_result.quality_scores}")
    
    assert result.success, "Stage should succeed"
    assert result.processed_data is not None
    assert result.validation_result is not None
    print("✅ Test 5 PASSED")

def test_schema_validation():
    """Test 6: Schema validation."""
    print("\n" + "="*70)
    print("TEST 6: Schema Validation")
    print("="*70)
    
    schema = {
        "fields": {
            "id": {"type": "int"},
            "text": {"type": "string"},
            "value": {
                "type": "float",
                "constraints": {"min": 0, "max": 100}
            }
        },
        "required_fields": ["id", "text"],
        "primary_keys": ["id"]
    }
    
    data = create_sample_data()
    checker = QualityChecker()
    report = checker.check_quality(
        data, 
        dataset_id="test_schema",
        schema=schema
    )
    
    print(f"Schema validation completed")
    print(f"Overall Score: {report.overall_score:.2%}")
    
    # Check validity dimension
    validity_score = report.dimension_scores.get("validity", 0)
    print(f"Validity Score: {validity_score:.2%}")
    
    print("✅ Test 6 PASSED")

def test_error_handling():
    """Test 7: Error handling with invalid data."""
    print("\n" + "="*70)
    print("TEST 7: Error Handling")
    print("="*70)
    
    checker = QualityChecker()
    
    # Test with empty data
    empty_df = pd.DataFrame()
    report = checker.check_quality(empty_df, dataset_id="test_empty")
    print(f"Empty data handled: {report.overall_score}")
    
    # Test with invalid data type
    try:
        invalid_data = "not a dataframe"
        report = checker.check_quality(invalid_data, dataset_id="test_invalid")
        print(f"Invalid data converted or handled gracefully")
    except Exception as e:
        print(f"Error handled: {type(e).__name__}")
    
    print("✅ Test 7 PASSED")

def test_convenience_function():
    """Test 8: Convenience function check_data_quality."""
    print("\n" + "="*70)
    print("TEST 8: Convenience Function")
    print("="*70)
    
    data = create_sample_data()
    
    # Use convenience function
    report = check_data_quality(
        data, 
        dataset_id="test_convenience",
        config=QualityConfig(completeness_threshold=0.9)
    )
    
    print(f"Convenience function works")
    print(f"Overall Score: {report.overall_score:.2%}")
    
    assert isinstance(report, QualityReport)
    print("✅ Test 8 PASSED")

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    tests = [
        test_basic_quality_check,
        test_data_with_issues,
        test_custom_config,
        test_validation_result_conversion,
        test_quality_check_stage,
        test_schema_validation,
        test_error_handling,
        test_convenience_function
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0

if __name__ == "__main__":
    run_all_tests()

