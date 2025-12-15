# Training Pipeline Quick Start

Get your DeBERTa task classifier trained **right now** with synthetic data.

## 🚀 One-Command Training

Run the complete pipeline (generates data, prepares it, and trains):

```bash
cd deepiri/diri-cyrex
python app/train/scripts/run_training_pipeline.py
```

Or use the bash script:

```bash
cd deepiri/diri-cyrex
./app/train/scripts/quick_train.sh
```

## 📊 What Gets Created

### 1. Synthetic Dataset (5000 examples by default)
- **8 categories**: coding, writing, fitness, cleaning, learning, creative, administrative, social
- **70/15/15 split**: train/validation/test
- **Location**: `app/train/data/`

### 2. Trained Model
- **Model**: DeBERTa-v3-base fine-tuned for task classification
- **Output**: `app/train/models/intent_classifier`
- **Categories**: 8 task categories (0-7)

## 📝 Step-by-Step (Manual)

### Step 1: Generate Synthetic Data

```bash
python app/train/scripts/generate_synthetic_data.py
```

Options:
- `--total-examples 5000` (default)
- `--examples-per-class 625` (625 per class = 5000 total)
- `--output-dir app/train/data`

### Step 2: Prepare Training Data

```bash
python app/train/scripts/prepare_training_data.py
```

This formats the data for training (handles both synthetic and collected data).

### Step 3: Train the Model

```bash
python app/train/scripts/train_intent_classifier.py \
    --epochs 3 \
    --batch-size 16 \
    --learning-rate 2e-5
```

## 🎯 Using the Trained Model

```python
from app.services.command_router import get_command_router

router = get_command_router(
    model_path="app/train/models/intent_classifier"
)

# Classify a task
result = router.classify("Write unit tests for my API")
print(result)  # {"category": "coding", "confidence": 0.95}
```

## 📋 Category Mapping

| ID | Category | Examples |
|----|----------|----------|
| 0 | coding | "Write unit tests", "Debug API", "Refactor code" |
| 1 | writing | "Write blog post", "Draft email", "Create docs" |
| 2 | fitness | "Go for a run", "Do yoga", "Workout" |
| 3 | cleaning | "Clean desk", "Do laundry", "Organize room" |
| 4 | learning | "Read paper", "Study for exam", "Take course" |
| 5 | creative | "Design logo", "Write story", "Create video" |
| 6 | administrative | "Schedule meeting", "Pay bills", "File taxes" |
| 7 | social | "Call friend", "Plan event", "Write thank you" |

## ⚙️ Customization

### Generate More Data

```bash
python app/train/scripts/generate_synthetic_data.py --total-examples 10000
```

### Train with Different Parameters

```bash
python app/train/scripts/train_intent_classifier.py \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 3e-5
```

### Use Different Base Model

```bash
python app/train/scripts/train_intent_classifier.py \
    --model microsoft/deberta-v3-small  # Faster, smaller
    # or
    --model microsoft/deberta-v3-large  # Slower, better accuracy
```

## 📁 File Structure

```
app/train/
├── data/
│   ├── classification_train.jsonl      # Training data
│   ├── classification_val.jsonl        # Validation data
│   ├── classification_test.jsonl       # Test data
│   ├── label_mapping.json              # Category mappings
│   └── dataset_metadata.json            # Dataset stats
├── models/
│   └── intent_classifier/              # Trained model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── category_map.json
│       └── training_info.json
└── scripts/
    ├── generate_synthetic_data.py       # Generate data
    ├── prepare_training_data.py         # Prepare data
    ├── train_intent_classifier.py      # Train model
    └── run_training_pipeline.py         # Run everything
```

## 🔍 Verify Training

After training, check the results:

```bash
# Check training info
cat app/train/models/intent_classifier/training_info.json

# Check category mapping
cat app/train/models/intent_classifier/category_map.json
```

## 🐛 Troubleshooting

### "No training data found"
- Run: `python app/train/scripts/generate_synthetic_data.py`

### "CUDA out of memory"
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model microsoft/deberta-v3-small`

### "Model not found"
- Check: `pip install transformers torch`
- Verify model name is correct

### Training is slow
- Use GPU if available (automatically detected)
- Reduce epochs: `--epochs 2`
- Use smaller model: `--model microsoft/deberta-v3-small`

## 📈 Next Steps

1. **Collect real data** from user interactions
2. **Fine-tune** with your collected data
3. **Evaluate** on test set
4. **Deploy** to production

## 💡 Tips

- Start with synthetic data to get baseline model
- Collect real user data over time
- Retrain periodically with new data
- Monitor model performance in production
- Use A/B testing to compare model versions

