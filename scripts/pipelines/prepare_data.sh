#!/bin/bash
# Data preparation script

echo "Preparing training datasets..."

python data/processors/prepare_dataset.py \
    --input data/raw/tasks.json \
    --output train/data/task_classification.jsonl \
    --type classification

python data/processors/prepare_dataset.py \
    --input data/raw/challenges.json \
    --output train/data/challenge_generation.jsonl \
    --type generation

echo "Data preparation complete!"

