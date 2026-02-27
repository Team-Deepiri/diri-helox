"""
Data Preprocessing Pipeline - Comprehensive Examples and Usage Guide

This module provides comprehensive examples demonstrating the complete data preprocessing
pipeline framework. It showcases:

1. Pipeline Orchestration: Creating and configuring all preprocessing stages
2. Data Loading: Loading data from various sources (in-memory, CSV, JSON, etc.)
3. Data Quality: Comprehensive quality checks with configurable thresholds
4. Error Handling: Robust error handling and validation
5. Configuration: Advanced configuration options for all stages
6. Production Patterns: Best practices for production use

Usage:
    # Run all examples
    python example_pipeline.py
    
    # Run with CSV file
    python example_pipeline.py path/to/data.csv
    
    # Run with environment variable
    CSV_FILE=path/to/data.csv python example_pipeline.py

Examples:
    - Standard pipeline execution with good data
    - Error handling with problematic data
    - Custom quality configurations
    - File-based data loading (CSV, JSON)
    - Advanced stage configurations
"""

import sys
import os
import argparse
from typing import Any, Dict, List, Optional
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


from pipelines.data_preprocessing.stages import (
    DataLoadingStage,
    DataCleaningStage,
    DataValidationStage,
    DataRoutingStage,
    LabelValidationStage,
    DataTransformationStage,
)
from pipelines.data_preprocessing.quality import QualityConfig



# Import PipelineOrchestrator directly (may fail if networkx is not installed)
try:
    from pipelines.data_preprocessing.orchestrator import PipelineOrchestrator
except ImportError as e:
    raise ImportError(
        "PipelineOrchestrator requires networkx. Install it with: pip install networkx"
    ) from e


def create_sample_data():
    """Create sample data for testing the pipeline."""
    return [
        {
            "text": "  Fix the bug in the authentication module  ",
            "label": "debugging",
            "priority": "high"
        },
        {
            "text": "Refactor the user service to use dependency injection",
            "label": "refactoring",
            "priority": "medium"
        },
        {
            "text": "Add unit tests for the payment processor",
            "label": "testing",
            "priority": "high"
        },
        {
            "text": "  Update documentation for API endpoints  ",
            "label": "documentation",
            "priority": "low"
        },
    ]

def create_problematic_data():
    """Create data with issues for testing error handling."""
    return [
        {
            "text": "  Missing label  ",
            # Missing label field
        },
        {
            "text": "",
            "label": "debugging"  # Empty text
        },
        {
            "text": "Valid text",
            "label": "unknown_label"  # Unknown label
        },
    ]


def create_label_mapping():
    """Create label mapping for routing stage."""
    return {
        "debugging": 0,
        "refactoring": 1,
        "testing": 2,
        "documentation": 3,
        "feature": 4,
        "bug": 5,
        # ... add more labels as needed
    }


def execute_data_preprocessing_pipeline(
    data: Optional[List[Dict[str, Any]]] = None,
    pipeline_name: str = "Data Preprocessing Pipeline",
    config_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Any:
    """
    Execute a complete data preprocessing pipeline with comprehensive configuration.
    
    This function orchestrates the entire data preprocessing workflow, including:
    - Data loading (from source or in-memory)
    - Data cleaning and normalization
    - Data validation and quality checks
    - Label routing and validation
    - Data transformation
    
    Args:
        data: Optional input data as list of dictionaries. If None, data will be
              loaded from source specified in config_overrides.
        pipeline_name: Descriptive name for this pipeline execution (for logging/display).
        config_overrides: Optional dictionary with configuration overrides for stages:
            - loading: Configuration for DataLoadingStage
            - cleaning: Configuration for DataCleaningStage
            - validation: Configuration for DataValidationStage
            - routing: Configuration for DataRoutingStage
            - label_validation: Configuration for LabelValidationStage
            - transformation: Configuration for DataTransformationStage
        verbose: If True, print detailed execution information.
    
    Returns:
        StageResult: Result object containing:
            - success: Boolean indicating if pipeline succeeded
            - processed_data: Final processed data with quality metrics
            - validation_result: Validation results and quality scores
            - execution_time: Total execution time in seconds
            - error: Error message if pipeline failed
            - stage_name: Name of stage where failure occurred (if any)
    
    Example:
        >>> data = [{"text": "Sample text", "label": "category"}]
        >>> result = execute_data_preprocessing_pipeline(
        ...     data=data,
        ...     pipeline_name="Production Pipeline",
        ...     config_overrides={
        ...         "validation": {
        ...             "completeness_threshold": 0.95,
        ...             "quality_threshold": 0.85
        ...         }
        ...     }
        ... )
        >>> if result.success:
        ...     print(f"Processed {len(result.processed_data.data)} items")
    """
    if verbose:
        print("\n" + "="*70)
        print(f"PIPELINE EXECUTION: {pipeline_name}")
        print("="*70)
    
    # 1. Determine data source and load data
    sample_data = data
    if sample_data is not None:
        if verbose:
            print(f"\n📥 Input Data: {len(sample_data)} items")
            for i, item in enumerate(sample_data[:2], 1):
                print(f"   Item {i}: {item}")
            if len(sample_data) > 2:
                print(f"   ... and {len(sample_data) - 2} more items")
    else:
        # Loading from source (CSV, JSON, etc.)
        loading_config = config_overrides.get("loading", {}) if config_overrides else {}
        source = loading_config.get("source", "unknown")
        format_type = loading_config.get("format", "unknown")
        if verbose:
            print(f"\n📥 Loading from source: {source}")
            print(f"   Format: {format_type}")
    
    # 2. Build comprehensive default configuration
    config_overrides = config_overrides or {}
    
    # Default validation configuration with production-ready thresholds
    default_validation_config = {
        "required_fields": ["text", "label"],
        "enable_quality_check": True,
        "completeness_threshold": 0.9,
        "consistency_threshold": 0.85,
        "validity_threshold": 0.9,
        "uniqueness_threshold": 0.95,
        "accuracy_threshold": 0.85,
        "timeliness_threshold": 0.8,
        "integrity_threshold": 0.9,
        "quality_threshold": 0.7,
        "fail_on_low_quality": False,
    }
    
    # Merge user overrides with defaults (user config takes precedence)
    validation_config = {**default_validation_config, **config_overrides.get("validation", {})}
    
    # Create all stages with configuration
    stages = [
        DataLoadingStage(config=config_overrides.get("loading", {"source": "manual", "format": "jsonl"})),
        DataCleaningStage(config=config_overrides.get("cleaning", {})),
        DataValidationStage(config=validation_config),
        DataRoutingStage(config=config_overrides.get("routing", {"label_mapping": create_label_mapping()})),
        LabelValidationStage(config=config_overrides.get("label_validation", {"min_label_id": 0, "max_label_id": 30})),
        DataTransformationStage(config=config_overrides.get("transformation", {"normalize_lowercase": True, "normalize_trim": True})),
    ]
    
    # 3. Initialize orchestrator and register all stages
    orchestrator = PipelineOrchestrator()
    for stage in stages:
        orchestrator.add_stage(stage)
        if verbose:
            print(f"   ✅ Registered stage: {stage.get_name()}")
    
    # 4. Build the dependency graph (DAG) and validate dependencies
    if verbose:
        print("\n🔧 Building pipeline dependency graph...")
    try:
        orchestrator.build_dag()
        if verbose:
            print(f"   ✅ Execution order: {' → '.join(orchestrator.execution_order)}")
            print(f"   ✅ Total stages: {len(orchestrator.execution_order)}")
    except ValueError as e:
        error_msg = f"Failed to build pipeline DAG: {e}"
        if verbose:
            print(f"\n❌ {error_msg}")
        from pipelines.data_preprocessing.base import StageResult
        return StageResult(
            success=False,
            error=error_msg,
            stage_name="orchestrator"
        )
    
    # 5. Execute the pipeline with error handling
    if verbose:
        print("\n🚀 Executing pipeline...")
    try:
        result = orchestrator.execute(initial_data=sample_data)
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        if verbose:
            print(f"\n❌ {error_msg}")
        from pipelines.data_preprocessing.base import StageResult
        import traceback
        if verbose:
            traceback.print_exc()
        return StageResult(
            success=False,
            error=error_msg,
            stage_name="orchestrator"
        )
    
    # 6. Process and display results
    if result.success:
        if verbose:
            print("\n✅ Pipeline completed successfully!")
        
        # Extract and display final processed data
        if result.processed_data:
            final_data = result.processed_data.data
            if verbose:
                print(f"\n📤 Output Data: {len(final_data)} items")
                for i, item in enumerate(final_data[:2], 1):
                    print(f"   Item {i}: {item}")
                if len(final_data) > 2:
                    print(f"   ... and {len(final_data) - 2} more items")
            
            # Display comprehensive quality metrics
            if result.processed_data.quality_metrics:
                if verbose:
                    print(f"\n📊 Quality Metrics (7 Dimensions):")
                    for dimension, score in sorted(result.processed_data.quality_metrics.items()):
                        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
                        print(f"   {status} {dimension.capitalize()}: {score:.2%}")
            
            # Display validation results with detailed information
            if result.validation_result:
                if verbose:
                    print(f"\n✔️  Validation Results:")
                    print(f"   Status: {'✅ Valid' if result.validation_result.is_valid else '❌ Invalid'}")
                    if result.validation_result.errors:
                        print(f"   Errors: {len(result.validation_result.errors)}")
                        for error in result.validation_result.errors[:3]:
                            print(f"     - {error}")
                    if result.validation_result.warnings:
                        print(f"   Warnings: {len(result.validation_result.warnings)}")
                        for warning in result.validation_result.warnings[:3]:
                            print(f"     - {warning}")
                    if result.validation_result.quality_scores:
                        print(f"   Quality Scores:")
                        for key, value in result.validation_result.quality_scores.items():
                            print(f"     {key}: {value:.2%}")
        
        # Display performance metrics
        if verbose and result.execution_time:
            print(f"\n⏱️  Performance Metrics:")
            print(f"   Total Execution Time: {result.execution_time:.3f}s")
            if result.processed_data and len(result.processed_data.data) > 0:
                items_per_second = len(result.processed_data.data) / result.execution_time
                print(f"   Throughput: {items_per_second:.2f} items/second")
        
    else:
        if verbose:
            print(f"\n❌ Pipeline execution failed")
            print(f"   Error: {result.error}")
            if result.stage_name:
                print(f"   Failed at stage: {result.stage_name}")
            if hasattr(result, 'processed_data') and result.processed_data:
                print(f"   Processed {len(result.processed_data.data)} items before failure")
    
    return result


def execute_pipeline_from_csv_file(csv_path: str, verbose: bool = True) -> Any:
    """
    Execute data preprocessing pipeline by loading data from a CSV file.
    
    This function demonstrates how to use the pipeline with file-based data sources.
    It handles file validation, loading, and pipeline execution.
    
    Args:
        csv_path: Path to the CSV file to load and process.
        verbose: If True, print detailed execution information.
    
    Returns:
        StageResult: Result object from pipeline execution.
    
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file path is invalid.
    
    Example:
        >>> result = execute_pipeline_from_csv_file("data/sample.csv")
        >>> if result.success:
        ...     print(f"Processed {len(result.processed_data.data)} records")
    """
    from pathlib import Path
    
    # Validate and resolve file path
    csv_file = Path(csv_path)
    if not csv_file.exists():
        error_msg = f"CSV file not found: {csv_path}"
        if verbose:
            print(f"\n❌ {error_msg}")
            print(f"   Current directory: {os.getcwd()}")
        from pipelines.data_preprocessing.base import StageResult
        raise FileNotFoundError(error_msg)
    
    if verbose:
        print("\n" + "="*70)
        print(f"PIPELINE EXECUTION: CSV File Processing")
        print("="*70)
        print(f"\n📥 CSV File: {csv_file.absolute()}")
        print(f"   File size: {csv_file.stat().st_size / 1024:.2f} KB")
    
    # Configure pipeline for CSV loading
    config_overrides = {
        "loading": {
            "source": str(csv_file.absolute()),
            "format": "csv"
        }
    }
    
    # Execute pipeline with CSV source (no initial_data - will load from CSV)
    result = execute_data_preprocessing_pipeline(
        data=None,  # No initial data - load from CSV
        pipeline_name="CSV File Processing Pipeline",
        config_overrides=config_overrides,
        verbose=verbose
    )
    
    return result


def create_cli_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all options.
    """
    parser = argparse.ArgumentParser(
        description="Data Preprocessing Pipeline - Comprehensive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demonstration examples
  %(prog)s --examples
  
  # Process a CSV file
  %(prog)s --file data.csv
  
  # Process a CSV file with custom quality thresholds
  %(prog)s --file data.csv --completeness-threshold 0.95 --quality-threshold 0.85
  
  # Process data with verbose output disabled
  %(prog)s --file data.csv --quiet
  
  # Process data with strict quality requirements
  %(prog)s --file data.csv --strict
  
  # Show help
  %(prog)s --help
        """
    )
    
    # Input source options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        metavar='PATH',
        help='Path to input data file (CSV, JSON, or JSONL format)'
    )
    input_group.add_argument(
        '--examples', '-e',
        action='store_true',
        help='Run all demonstration examples (default if no file specified)'
    )
    
    # Quality configuration options
    quality_group = parser.add_argument_group('Quality Configuration')
    quality_group.add_argument(
        '--completeness-threshold',
        type=float,
        metavar='FLOAT',
        default=0.9,
        help='Completeness threshold (0.0-1.0, default: 0.9)'
    )
    quality_group.add_argument(
        '--quality-threshold',
        type=float,
        metavar='FLOAT',
        default=0.7,
        help='Overall quality threshold (0.0-1.0, default: 0.7)'
    )
    quality_group.add_argument(
        '--strict',
        action='store_true',
        help='Use strict quality thresholds (completeness=0.98, quality=0.85)'
    )
    quality_group.add_argument(
        '--fail-on-low-quality',
        action='store_true',
        help='Fail pipeline execution if quality threshold is not met'
    )
    
    # Output and behavior options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output (only show errors and final results)'
    )
    output_group.add_argument(
        '--output', '-o',
        type=str,
        metavar='PATH',
        help='Save processed data to output file (JSON format)'
    )
    
    return parser


def parse_cli_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return parsed namespace.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Default to examples if no file specified
    if not args.file and not args.examples:
        args.examples = True
    
    # Validate thresholds
    if args.completeness_threshold < 0 or args.completeness_threshold > 1:
        parser.error("--completeness-threshold must be between 0.0 and 1.0")
    if args.quality_threshold < 0 or args.quality_threshold > 1:
        parser.error("--quality-threshold must be between 0.0 and 1.0")
    
    # Apply strict mode if requested
    if args.strict:
        args.completeness_threshold = 0.98
        args.quality_threshold = 0.85
    
    return args


def run_examples(verbose: bool = True) -> None:
    """
    Run all demonstration examples.
    
    Args:
        verbose: If True, print detailed execution information.
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING PIPELINE - COMPREHENSIVE EXAMPLES")
    print("="*70)
    if verbose:
        print("\nThis demonstration showcases:")
        print("  • Standard pipeline execution with production-ready configuration")
        print("  • Robust error handling and validation")
        print("  • Custom quality thresholds and configurations")
        print("  • File-based data loading (CSV, JSON)")
        print("  • Performance metrics and quality reporting")
    
    try:
        # Example 1: Standard pipeline with good quality data
        if verbose:
            print("\n" + "="*70)
            print("EXAMPLE 1: Standard Pipeline - Good Quality Data")
            print("="*70)
        execute_data_preprocessing_pipeline(
            data=create_sample_data(),
            pipeline_name="Standard Production Pipeline",
            config_overrides=None,
            verbose=verbose
        )
        
        # Example 2: Error handling with problematic data
        if verbose:
            print("\n" + "="*70)
            print("EXAMPLE 2: Error Handling - Problematic Data")
            print("="*70)
        execute_data_preprocessing_pipeline(
            data=create_problematic_data(),
            pipeline_name="Error Handling Demonstration",
            config_overrides=None,
            verbose=verbose
        )
        
        # Example 3: Custom quality configuration with strict thresholds
        if verbose:
            print("\n" + "="*70)
            print("EXAMPLE 3: Custom Quality Configuration - Strict Thresholds")
            print("="*70)
        quality_config = QualityConfig(
            completeness_threshold=0.98,
            consistency_threshold=0.95,
            validity_threshold=0.95,
            uniqueness_threshold=0.98,
            accuracy_threshold=0.90,
            timeliness_threshold=0.85,
            integrity_threshold=0.95,
            iqr_multiplier=2.0,
            zscore_threshold=2.5,
            freshness_decay_days=7.0,
        )
        execute_data_preprocessing_pipeline(
            data=create_sample_data(),
            pipeline_name="Strict Quality Thresholds Pipeline",
            config_overrides={
                "validation": {
                    "quality_config": quality_config,
                    "quality_threshold": 0.85,
                    "fail_on_low_quality": False,
                }
            },
            verbose=verbose
        )
        
        if verbose:
            print("\n" + "="*70)
            print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
            print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def process_file_from_cli(
    file_path: str,
    completeness_threshold: float = 0.9,
    quality_threshold: float = 0.7,
    fail_on_low_quality: bool = False,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Any:
    """
    Process a file using the pipeline with CLI-provided configuration.
    
    Args:
        file_path: Path to input file.
        completeness_threshold: Completeness threshold for quality checks.
        quality_threshold: Overall quality threshold.
        fail_on_low_quality: Whether to fail on low quality.
        output_path: Optional path to save processed data.
        verbose: If True, print detailed execution information.
    
    Returns:
        StageResult: Result from pipeline execution.
    """
    # Validate file exists
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file format from extension
    file_format = file.suffix.lower().lstrip('.')
    if file_format not in ['csv', 'json', 'jsonl']:
        print(f"⚠️  Warning: Unknown file format '{file_format}', assuming CSV")
        file_format = 'csv'
    
    # Create quality config
    quality_config = QualityConfig(
        completeness_threshold=completeness_threshold,
    )
    
    # Configure pipeline
    config_overrides = {
        "loading": {
            "source": str(file.absolute()),
            "format": file_format
        },
        "validation": {
            "quality_config": quality_config,
            "quality_threshold": quality_threshold,
            "fail_on_low_quality": fail_on_low_quality,
        }
    }
    
    # Execute pipeline
    result = execute_data_preprocessing_pipeline(
        data=None,  # Load from file
        pipeline_name=f"File Processing: {file.name}",
        config_overrides=config_overrides,
        verbose=verbose
    )
    
    # Save output if requested
    if output_path and result.success and result.processed_data:
        import json
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(result.processed_data.data, f, indent=2, default=str)
        if verbose:
            print(f"\n💾 Processed data saved to: {output_file.absolute()}")
    
    return result


def main():
    """
    Main entry point with comprehensive CLI support.
    
    This function provides a formal command-line interface for the data preprocessing
    pipeline with argument parsing, help messages, and multiple execution modes.
    """
    try:
        args = parse_cli_arguments()
        
        if args.examples:
            # Run demonstration examples
            run_examples(verbose=not args.quiet)
        elif args.file:
            # Process file
            result = process_file_from_cli(
                file_path=args.file,
                completeness_threshold=args.completeness_threshold,
                quality_threshold=args.quality_threshold,
                fail_on_low_quality=args.fail_on_low_quality,
                output_path=args.output,
                verbose=not args.quiet
            )
            
            # Exit with appropriate code
            if not result.success:
                sys.exit(1)
        else:
            # Should not reach here due to argparse, but handle gracefully
            print("Error: No input source specified. Use --file or --examples")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Please provide a valid file path.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

