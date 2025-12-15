# Training Pipeline - Ready to Use! 🚀

## ✅ What's Been Set Up

Your complete training pipeline is ready. You can generate synthetic data and train your DeBERTa classifier **right now**.

## 🎯 Quick Start (3 Options)

### Option 1: One Command (Python)
```bash
cd deepiri/diri-cyrex
python app/train/scripts/run_training_pipeline.py
```

### Option 2: One Command (Bash)
```bash
cd deepiri/diri-cyrex
./app/train/scripts/quick_train.sh
```

### Option 3: Step by Step
```bash
# Step 1: Generate data
python app/train/scripts/generate_synthetic_data.py

# Step 2: Prepare data
python app/train/scripts/prepare_training_data.py

# Step 3: Train model
python app/train/scripts/train_intent_classifier.py
```

## 📦 What You Get

### 1. Synthetic Dataset
- **5000 examples** (default, customizable)
- **8 categories**: coding, writing, fitness, cleaning, learning, creative, administrative, social
- **70/15/15 split**: train/validation/test
- **Location**: `app/train/data/`

### 2. Trained Model
- **Base Model**: DeBERTa-v3-base
- **Task**: Multi-class classification (8 categories)
- **Output**: `app/train/models/intent_classifier`
- **Ready for production use**

## 📋 Category System

| ID | Category | Example Tasks |
|----|----------|---------------|
| 0 | **coding** | "Write unit tests", "Debug API", "Refactor code" |
| 1 | **writing** | "Write blog post", "Draft email", "Create docs" |
| 2 | **fitness** | "Go for a run", "Do yoga", "Workout" |
| 3 | **cleaning** | "Clean desk", "Do laundry", "Organize room" |
| 4 | **learning** | "Read paper", "Study for exam", "Take course" |
| 5 | **creative** | "Design logo", "Write story", "Create video" |
| 6 | **administrative** | "Schedule meeting", "Pay bills", "File taxes" |
| 7 | **social** | "Call friend", "Plan event", "Write thank you" |

## 🔧 Customization Options

### Generate More/Less Data
```bash
# 10,000 examples
python app/train/scripts/generate_synthetic_data.py --total-examples 10000

# 625 examples per class (5000 total)
python app/train/scripts/generate_synthetic_data.py --examples-per-class 625
```

### Training Parameters
```bash
python app/train/scripts/train_intent_classifier.py \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 3e-5
```

### Different Models
```bash
# Smaller, faster
python app/train/scripts/train_intent_classifier.py \
    --model microsoft/deberta-v3-small

# Larger, more accurate
python app/train/scripts/train_intent_classifier.py \
    --model microsoft/deberta-v3-large
```

## 📁 Files Created

### Data Files
- `app/train/data/classification_train.jsonl` - Training set
- `app/train/data/classification_val.jsonl` - Validation set
- `app/train/data/classification_test.jsonl` - Test set
- `app/train/data/label_mapping.json` - Category mappings
- `app/train/data/dataset_metadata.json` - Dataset statistics

### Model Files
- `app/train/models/intent_classifier/` - Complete trained model
  - `pytorch_model.bin` - Model weights
  - `config.json` - Model configuration
  - `tokenizer_config.json` - Tokenizer settings
  - `category_map.json` - Category ID mappings
  - `training_info.json` - Training metrics and info

## 💻 Using the Model

```python
from app.services.command_router import get_command_router

# Load the trained model
router = get_command_router(
    model_path="app/train/models/intent_classifier"
)

# Classify a task
result = router.classify("Write unit tests for my authentication API")
# Returns: {"category": "coding", "confidence": 0.95, "label_id": 0}
```

## 📊 Expected Results

With 5000 examples and 3 epochs:
- **Training time**: 10-30 minutes (depending on hardware)
- **Expected accuracy**: 85-95% on validation set
- **Model size**: ~500MB (DeBERTa-v3-base)

## 🎓 Next Steps

1. **Test the model** on your use cases
2. **Collect real data** from user interactions
3. **Fine-tune** with collected data for better performance
4. **Deploy** to production
5. **Monitor** and retrain periodically

## 📚 Documentation

- **Quick Start Guide**: `app/train/TRAINING_QUICK_START.md`
- **Data Collection**: `HOW_TO_COLLECT_TRAINING_DATA.md`
- **Training Scripts**: `app/train/scripts/`

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install transformers torch datasets scikit-learn
```

### "CUDA out of memory"
- Reduce batch size: `--batch-size 8`
- Use CPU: Training will automatically use CPU if GPU unavailable

### "No data found"
- Run: `python app/train/scripts/generate_synthetic_data.py`

## ✨ Features

- ✅ **8 task categories** (coding, writing, fitness, etc.)
- ✅ **Synthetic data generation** (5000+ examples)
- ✅ **Automatic train/val/test split** (70/15/15)
- ✅ **DeBERTa fine-tuning** (state-of-the-art classifier)
- ✅ **Production-ready** model output
- ✅ **Complete pipeline** (one command to run everything)
- ✅ **Customizable** (data size, training params, model size)

---

**You're all set! Run the pipeline and start training your models right now.** 🎉

