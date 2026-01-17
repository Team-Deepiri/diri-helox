# Testing the Data Preprocessing Pipeline

This guide explains how to test the data preprocessing pipeline.

## Quick Start

Run the comprehensive test suite:

```bash
cd /Users/mahlaka/Projects/deepiri-platform/diri-helox
python tests/test_data_preprocessing_pipeline.py
```

Or make it executable and run directly:

```bash
chmod +x tests/test_data_preprocessing_pipeline.py
./tests/test_data_preprocessing_pipeline.py
```

## What the Test Suite Does

The test suite includes:

1. **Individual Stage Testing**: Tests each pipeline stage in isolation
   - DataLoadingStage
   - DataCleaningStage
   - DataValidationStage
   - DataRoutingStage
   - LabelValidationStage
   - DataTransformationStage

2. **Full Pipeline Testing**: Tests the complete pipeline with orchestrator
   - Creates all stages
   - Builds dependency graph (DAG)
   - Executes pipeline end-to-end
   - Validates final output

3. **Validation Testing**: Tests validation methods for each stage

4. **Error Handling**: Tests how the pipeline handles invalid data
   - Invalid labels
   - Missing required fields

## Manual Testing

You can also test the pipeline manually in Python:

```python
import sys
sys.path.insert(0, 'diri-helox')

from pipelines.data_preprocessing import (
    PipelineOrchestrator,
    DataLoadingStage,
    DataCleaningStage,
    DataValidationStage,
    DataRoutingStage,
    LabelValidationStage,
    DataTransformationStage,
)

# Create sample data
sample_data = [
    {
        "text": "Debug this code issue",
        "label": "debugging",
    },
    {
        "text": "Refactor this function",
        "label": "refactoring",
    },
]

# Create label mapping
label_mapping = {
    "debugging": 0,
    "refactoring": 1,
    "testing": 2,
}

# Create and configure pipeline
orchestrator = PipelineOrchestrator()
orchestrator.add_stage(DataLoadingStage(config={"source": "test", "format": "jsonl"}))
orchestrator.add_stage(DataCleaningStage())
orchestrator.add_stage(DataValidationStage(config={"required_fields": ["text", "label"]}))
orchestrator.add_stage(DataRoutingStage(config={"label_mapping": label_mapping}))
orchestrator.add_stage(LabelValidationStage(config={"min_label_id": 0, "max_label_id": 30}))
orchestrator.add_stage(DataTransformationStage(config={"normalize_lowercase": True, "normalize_trim": True}))

# Build dependency graph
orchestrator.build_dag()

# Execute pipeline
result = orchestrator.execute(initial_data=sample_data)

# Check results
if result.success:
    print("Pipeline executed successfully!")
    print(f"Final data: {result.processed_data.data}")
else:
    print(f"Pipeline failed: {result.error}")
```

## Testing Individual Stages

You can test individual stages without the orchestrator:

```python
from pipelines.data_preprocessing import DataCleaningStage

# Create stage
cleaning_stage = DataCleaningStage()

# Process data
data = [{"text": "  Hello   World  ", "label": "test"}]
result = cleaning_stage.process(data)

if result.success:
    print(f"Cleaned data: {result.processed_data.data}")
else:
    print(f"Error: {result.error}")

# Validate data before processing
validation_result = cleaning_stage.validate(data)
print(f"Is valid: {validation_result.is_valid}")
print(f"Warnings: {validation_result.warnings}")
```

## Expected Output

When running the test suite, you should see:

```
============================================================
DATA PREPROCESSING PIPELINE TEST SUITE
============================================================

============================================================
TESTING INDIVIDUAL STAGES
============================================================

1. Testing DataLoadingStage...
   Success: True
   Data items: 5

2. Testing DataCleaningStage...
   Success: True
   Data items after cleaning: 4
   First item text: 'Debug this code issue'
   Removed fields: 'meta', 'empty' should be gone

...

============================================================
TESTING FULL PIPELINE
============================================================

Input data: 5 items
...

============================================================
ALL TESTS COMPLETED
============================================================
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the correct directory:

```bash
cd /Users/mahlaka/Projects/deepiri-platform/diri-helox
```

### Missing Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```

Required dependencies for the pipeline:
- `networkx>=3.0.0` (for DAG processing)

### Pipeline Errors

If the pipeline fails:

1. Check the error message - it will indicate which stage failed
2. Verify your input data has the required fields (`text` and `label`)
3. Make sure label mapping includes all labels in your data
4. Check that label IDs are in the valid range [0, 30]

## Next Steps

After testing:

1. **Customize Configuration**: Modify stage configs for your use case
2. **Add Custom Stages**: Create new stages by inheriting from `PreprocessingStage`
3. **Integrate with Training**: Use the processed data output for model training
4. **Add Monitoring**: Add logging and metrics collection for production use



