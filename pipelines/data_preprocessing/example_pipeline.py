"""
End-to-End Pipeline Example

This script demonstrates how to use the complete data preprocessing pipeline:
1. Create and configure all stages
2. Set up the orchestrator
3. Process data through the entire pipeline
4. Handle results and errors

Run this script to see the pipeline in action!
"""

import sys
import os

# Add parent directories to path
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


def basic_pipeline(data, example_name="Basic Pipeline", config_overrides=None):
    """Run pipeline with given data and optional config overrides."""
    print("\n" + "="*70)
    print(f"EXAMPLE: {example_name}")
    print("="*70)
    
    # 1. Use provided data
    sample_data = data
    print(f"\n📥 Input Data: {len(sample_data)} items")
    for i, item in enumerate(sample_data[:2], 1):
        print(f"   Item {i}: {item}")
    
    # 2. Default configuration
    config_overrides = config_overrides or {}
    default_validation_config = {
        "required_fields": ["text", "label"],
        "enable_quality_check": True,
        "completeness_threshold": 0.9,
        "quality_threshold": 0.7,
        "fail_on_low_quality": False,
    }
    
    # Merge with overrides
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
    
    # 3. Create orchestrator and add stages
    orchestrator = PipelineOrchestrator()
    for stage in stages:
        orchestrator.add_stage(stage)
        print(f"   ✅ Added stage: {stage.get_name()}")
    
    # 4. Build the DAG (dependency graph)
    print("\n🔧 Building pipeline DAG...")
    orchestrator.build_dag()
    print(f"   ✅ Execution order: {' → '.join(orchestrator.execution_order)}")
    
    # 5. Execute the pipeline
    print("\n🚀 Executing pipeline...")
    result = orchestrator.execute(initial_data=sample_data)
    
    # 6. Handle results
    if result.success:
        print("\n✅ Pipeline completed successfully!")
        
        # Get final processed data
        if result.processed_data:
            final_data = result.processed_data.data
            print(f"\n📤 Output Data: {len(final_data)} items")
            for i, item in enumerate(final_data[:2], 1):
                print(f"   Item {i}: {item}")
            
            # Show quality metrics if available
            if result.processed_data.quality_metrics:
                print(f"\n📊 Quality Metrics:")
                for dimension, score in result.processed_data.quality_metrics.items():
                    print(f"   {dimension}: {score:.2%}")
            
            # Show validation results
            if result.validation_result:
                print(f"\n✔️  Validation:")
                print(f"   Valid: {result.validation_result.is_valid}")
                if result.validation_result.warnings:
                    print(f"   Warnings: {len(result.validation_result.warnings)}")
                    for warning in result.validation_result.warnings[:3]:
                        print(f"     - {warning}")
        
        # Show execution times
        print(f"\n⏱️  Execution Time: {result.execution_time:.3f}s")
        
    else:
        print(f"\n❌ Pipeline failed: {result.error}")
        if result.stage_name:
            print(f"   Failed at stage: {result.stage_name}")
    
    return result


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DATA PREPROCESSING PIPELINE - END-TO-END EXAMPLES")
    print("="*70)
    
    try:
        # Run examples using the reusable basic_pipeline function
        basic_pipeline(create_sample_data(), "Good Data - Standard Config")
        basic_pipeline(create_problematic_data(), "Problematic Data - Error Handling")
        
        # Custom quality config example
        quality_config = QualityConfig(
            completeness_threshold=0.98,
            consistency_threshold=0.95,
            accuracy_threshold=0.90,
            iqr_multiplier=2.0,
        )
        basic_pipeline(
            create_sample_data(), 
            "Custom Quality Config - Strict Thresholds",
            config_overrides={
                "validation": {
                    "quality_config": quality_config,
                    "quality_threshold": 0.8,
                }
            }
        )
        
        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

