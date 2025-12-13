#!/bin/bash
# Quick training script - does everything in one go

set -e

echo "🚀 Deepiri Training Quick Start"
echo "================================"
echo ""

# Step 1: Export data
echo "Step 1: Exporting collected data..."
python3 -c "
from app.train.pipelines.data_collection_pipeline import get_data_collector
from pathlib import Path

collector = get_data_collector()
output_dir = Path('app/train/data/exported')
output_dir.mkdir(parents=True, exist_ok=True)

try:
    collector.export_for_training(
        str(output_dir / 'classification_training.jsonl'),
        'classification'
    )
    print('✓ Data exported')
except Exception as e:
    print(f'⚠ Could not export: {e}')
    print('  This is okay if you have no collected data yet')
    print('  Will use synthetic data if available')
"

# Step 2: Check if we have data, if not generate synthetic
if [ ! -f "app/train/data/exported/classification_training.jsonl" ]; then
    echo ""
    echo "No collected data found. Generating synthetic data..."
    if [ -f "app/train/scripts/generate_synthetic_data.py" ]; then
        python3 app/train/scripts/generate_synthetic_data.py
    else
        echo "⚠ Synthetic data generator not found"
        echo "  Create some training data first!"
        exit 1
    fi
fi

# Step 3: Prepare data
echo ""
echo "Step 2: Preparing training data..."
python3 app/train/scripts/prepare_training_data.py

# Step 4: Train
echo ""
echo "Step 3: Training intent classifier..."
echo "  This may take 10-30 minutes depending on your hardware..."
echo ""

python3 app/train/scripts/train_intent_classifier.py \
    --epochs 3 \
    --batch-size 16

echo ""
echo "✅ Training complete!"
echo ""
echo "Model saved to: app/train/models/intent_classifier"
echo ""
echo "To use in production:"
echo "  from app.services.command_router import get_command_router"
echo "  router = get_command_router(model_path='app/train/models/intent_classifier')"

