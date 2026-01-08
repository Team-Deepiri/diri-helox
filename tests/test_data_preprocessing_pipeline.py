#!/usr/bin/env python3
"""
Test script for the data preprocessing pipeline.

This script demonstrates how to:
1. Create and configure pipeline stages
2. Build and execute the pipeline
3. Validate results
4. Test individual stages
"""

import sys
import os

# Add parent directory to path to import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipelines.data_preprocessing import (
    PipelineOrchestrator,
    DataLoadingStage,
    DataCleaningStage,
    DataValidationStage,
    DataRoutingStage,
    LabelValidationStage,
    DataTransformationStage,
)


def create_sample_data():
    """Create sample test data."""
    return [
        {
            "text": "  Debug this code   issue  ",
            "label": "debugging",
            "meta": None,
            "empty": "",
        },
        {
            "text": "Refactor this function",
            "label": "refactoring",
        },
        {
            "text": "Write tests for this module",
            "label": "testing",
        },
        # Add a duplicate to test deduplication
        {
            "text": "Write tests for this module",
            "label": "testing",
        },
        # Test with numeric label
        {
            "text": "Create new feature",
            "label": 5,
        },
    ]


def create_label_mapping():
    """Create label mapping for testing."""
    return {
        "debugging": 0,
        "refactoring": 1,
        "testing": 2,
        "documentation": 3,
        "optimization": 4,
        "feature": 5,
    }


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def create_default_orchestrator(label_mapping=None, include_all_stages=True):
    """Create a pipeline orchestrator with standard stages.
    
    Args:
        label_mapping: Optional label mapping for routing stage
        include_all_stages: If True, includes all 6 stages; if False, includes basic stages only
    
    Returns:
        Configured PipelineOrchestrator instance
    """
    if label_mapping is None:
        label_mapping = create_label_mapping()
    
    orchestrator = PipelineOrchestrator()
    orchestrator.add_stage(DataLoadingStage(config={"source": "test", "format": "jsonl"}))
    orchestrator.add_stage(DataCleaningStage())
    orchestrator.add_stage(DataValidationStage(config={"required_fields": ["text", "label"]}))
    
    if include_all_stages:
        orchestrator.add_stage(DataRoutingStage(config={"label_mapping": label_mapping}))
        orchestrator.add_stage(LabelValidationStage(config={"min_label_id": 0, "max_label_id": 30}))
        orchestrator.add_stage(DataTransformationStage(config={"normalize_lowercase": True, "normalize_trim": True}))
    
    return orchestrator


def print_validation_result(validation_result, prefix="   "):
    """Print validation result in a consistent format."""
    print(f"{prefix}Is valid: {validation_result.is_valid}")
    if validation_result.errors:
        print(f"{prefix}Errors: {validation_result.errors}")
    if validation_result.warnings:
        print(f"{prefix}Warnings: {validation_result.warnings}")


def print_stage_result(result, stage_name="Stage", prefix="   "):
    """Print stage processing result in a consistent format."""
    print(f"{prefix}Success: {result.success}")
    if result.processed_data:
        print(f"{prefix}Data items: {len(result.processed_data.data)}")
    if not result.success and result.error:
        print(f"{prefix}Error: {result.error}")


def execute_and_print_result(orchestrator, initial_data, description="Pipeline"):
    """Execute orchestrator and print results.
    
    Args:
        orchestrator: PipelineOrchestrator instance (must have stages already added)
        initial_data: Data to pass to the pipeline
        description: Optional description for error messages
    
    Returns:
        StageResult or None if execution failed
    """
    try:
        orchestrator.build_dag()
        result = orchestrator.execute(initial_data=initial_data)
        
        print(f"   Success: {result.success}")
        if not result.success and result.stage_name:
            print(f"   Error at stage '{result.stage_name}': {result.error}")
        
        return result
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_processed_data(result):
    """Safely extract processed data from a result.
    
    Returns:
        List of data items or None
    """
    if result and result.processed_data:
        return result.processed_data.data
    return None


def test_individual_stages():
    """Test each stage individually."""
    print_section_header("TESTING INDIVIDUAL STAGES")
    
    sample_data = create_sample_data()
    label_mapping = create_label_mapping()
    result = None
    
    # Test DataLoadingStage
    print("\n1. Testing DataLoadingStage...")
    result = DataLoadingStage(config={"source": "test", "format": "jsonl"}).process(sample_data)
    print_stage_result(result)
    
    # Test DataCleaningStage
    print("\n2. Testing DataCleaningStage...")
    input_data = get_processed_data(result) or sample_data
    result = DataCleaningStage().process(input_data)
    print_stage_result(result)
    data = get_processed_data(result)
    if data:
        print(f"   First item text: '{data[0]['text']}'")
        print(f"   Removed fields: 'meta', 'empty' should be gone")
    
    # Test DataValidationStage
    print("\n3. Testing DataValidationStage...")
    result = DataValidationStage(config={"required_fields": ["text", "label"]}).process(
        result.processed_data if result and result.processed_data else None
    )
    print_stage_result(result)
    
    # Test DataRoutingStage
    print("\n4. Testing DataRoutingStage...")
    result = DataRoutingStage(config={"label_mapping": label_mapping}).process(
        result.processed_data if result and result.processed_data else None
    )
    print_stage_result(result)
    data = get_processed_data(result)
    if data:
        print(f"   First item label_id: {data[0].get('label_id')}")
    
    # Test LabelValidationStage
    print("\n5. Testing LabelValidationStage...")
    result = LabelValidationStage(config={"min_label_id": 0, "max_label_id": 30}).process(
        result.processed_data if result and result.processed_data else None
    )
    print_stage_result(result)
    
    # Test DataTransformationStage
    print("\n6. Testing DataTransformationStage...")
    result = DataTransformationStage(config={"normalize_lowercase": True, "normalize_trim": True}).process(
        result.processed_data if result and result.processed_data else None
    )
    print_stage_result(result)
    data = get_processed_data(result)
    if data:
        print(f"   First item text (normalized): '{data[0]['text']}'")
    
    return result


def test_full_pipeline():
    """Test the full pipeline with orchestrator."""
    print_section_header("TESTING FULL PIPELINE")
    
    sample_data = create_sample_data()
    label_mapping = create_label_mapping()
    
    print(f"\nInput data: {len(sample_data)} items")
    print(f"Sample item: {sample_data[0]}")
    
    # Create pipeline orchestrator with all stages
    print("\n1. Adding stages to pipeline...")
    orchestrator = create_default_orchestrator(label_mapping, include_all_stages=True)
    print(f"   Added {len(orchestrator.stages)} stages")
    
    # Build DAG
    print("\n2. Building dependency graph (DAG)...")
    try:
        orchestrator.build_dag()
        print(f"   Execution order: {orchestrator.execution_order}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return None
    
    # Execute pipeline
    print("\n3. Executing pipeline...")
    result = orchestrator.execute(initial_data=sample_data)
    
    if result.success:
        print(f"   Final result success: {result.success}")
        print(f"   Final stage: {result.stage_name}")
        if result.execution_time:
            print(f"   Execution time: {result.execution_time:.4f} seconds")
        
        if result.processed_data:
            final_data = result.processed_data.data
            print(f"   Output data: {len(final_data)} items")
            print(f"\n   Sample output item:")
            print(f"   {final_data[0]}")
            
            if result.processed_data.metadata:
                print(f"\n   Metadata: {result.processed_data.metadata}")
    else:
        print(f"   ERROR: {result.error}")
    
    # Show checkpoints
    print("\n4. Pipeline checkpoints:")
    for stage_name, checkpoint in orchestrator.checkpoints.items():
        status = "✓" if checkpoint.success else "✗"
        time_str = f"{checkpoint.execution_time:.4f}s" if checkpoint.execution_time else "N/A"
        print(f"   {status} {stage_name}: {time_str}")
    
    return result


def test_validation():
    """Test validation methods."""
    print_section_header("TESTING VALIDATION")
    
    sample_data = create_sample_data()
    label_mapping = create_label_mapping()
    
    # Test DataCleaningStage validation
    print("\n1. Validating data for cleaning stage...")
    cleaning_stage = DataCleaningStage()
    validation_result = cleaning_stage.validate(sample_data)
    print_validation_result(validation_result)
    
    # Test DataRoutingStage validation
    print("\n2. Validating data for routing stage...")
    routing_stage = DataRoutingStage(config={"label_mapping": label_mapping})
    validation_result = routing_stage.validate(sample_data)
    print_validation_result(validation_result)


def test_error_handling():
    """Test error handling with invalid data."""
    print_section_header("TESTING ERROR HANDLING")
    
    label_mapping = create_label_mapping()
    
    # Test with invalid label
    print("\n1. Testing with invalid label...")
    invalid_data = [{"text": "Test with invalid label", "label": "invalid_label"}]
    orchestrator = create_default_orchestrator(label_mapping, include_all_stages=True)
    result = execute_and_print_result(orchestrator, invalid_data)
    
    # Test with missing required field
    print("\n2. Testing with missing required field...")
    missing_field_data = [{"text": "Missing label field"}]
    orchestrator2 = create_default_orchestrator(label_mapping, include_all_stages=False)
    result = execute_and_print_result(orchestrator2, missing_field_data)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE TEST SUITE")
    print("="*60)
    
    try:
        # Test individual stages
        test_individual_stages()
        
        # Test full pipeline
        test_full_pipeline()
        
        # Test validation
        test_validation()
        
        # Test error handling
        test_error_handling()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
