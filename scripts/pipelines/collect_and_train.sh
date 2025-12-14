#!/bin/bash
# Complete data collection and training pipeline

echo "Starting Data Collection and Training Pipeline"
echo "==============================================="

# Step 1: Collect data from recent usage
echo "Step 1: Collecting training data..."
python -c "
from pipelines.training.data_collection_pipeline import get_data_collector
collector = get_data_collector()
collector.export_for_training('data/datasets/raw/collected_classifications.jsonl', 'classification')
collector.export_for_training('data/datasets/raw/collected_challenges.jsonl', 'challenge')
print('Data collection complete')
"

# Step 2: Prepare datasets
echo "Step 2: Preparing datasets..."
python data/processors/prepare_dataset.py

# Step 3: Train models
echo "Step 3: Training models..."

if [ "$1" == "classifier" ] || [ "$1" == "all" ]; then
    echo "Training task classifier..."
    python pipelines/training/full_training_pipeline.py \
        --config configs/training_config.json
fi

if [ "$1" == "generator" ] || [ "$1" == "all" ]; then
    echo "Training challenge generator..."
    python pipelines/training/full_training_pipeline.py \
        --config configs/challenge_generator_config.json
fi

if [ "$1" == "bandit" ] || [ "$1" == "all" ]; then
    echo "Training bandit..."
    python pipelines/training/bandit_training.py
fi

# Step 4: Evaluate models
echo "Step 4: Evaluating models..."
python scripts/evaluation/evaluate_model.py \
    --model_path models/task_classifier \
    --test_dataset data/datasets/raw/task_classification_test.jsonl

echo "Pipeline complete!"

