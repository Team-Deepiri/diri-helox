# LLM Training from Scratch - Complete Guide

This guide covers training your own Large Language Model (LLM) from zero to a usable model using the Helox training infrastructure.

## Overview

The training pipeline consists of several stages:

1. **Data Collection & Cleaning** - Gather and clean training data
2. **Tokenizer Training** - Train a custom SentencePiece tokenizer
3. **Dataset Preparation** - Convert text to tokenized format
4. **Pretraining** - Train the base language model
5. **Instruction Fine-Tuning** - Fine-tune for instruction following (optional)

## Prerequisites

### Hardware Requirements

**Minimum (realistic):**
- GPU: RTX 4090 (24GB) or A100 40GB
- RAM: 64GB
- Disk: 2-4TB NVMe
- OS: Ubuntu 22.04 (or Windows with WSL2)

**CPU Training:**
- The system automatically detects CPU/GPU and adjusts accordingly
- CPU training is significantly slower but functional

### Software Requirements

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- PyTorch 2.0+

## Installation

### 1. Create Virtual Environment

```bash
python3.10 -m venv llm-env
source llm-env/bin/activate  # On Windows: llm-env\Scripts\activate
```

### 2. Install Dependencies

```bash
cd deepiri/diri-helox
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Step 1: Prepare Your Data

Place your raw training data in `data/datasets/raw/`. Supported formats:
- Plain text files (`.txt`, `.md`)
- Code files (`.py`, `.js`, `.ts`, etc.)
- JSONL files with `text` field

Example structure:
```
data/datasets/raw/
├── books/
│   ├── book1.txt
│   └── book2.txt
├── code/
│   └── project/
│       └── *.py
└── articles/
    └── *.md
```

### Step 2: Configure Training

Edit configuration files in `configs/`:

- `model_config.json` - Model architecture settings
- `data_config.json` - Data processing settings
- `training_config.json` - Training hyperparameters

### Step 3: Run Training Pipeline

```bash
python scripts/train_llm_from_scratch.py
```

This will:
1. Collect and clean data from `data/datasets/raw/`
2. Train a custom tokenizer
3. Prepare tokenized datasets
4. Train the language model

### Step 4: Monitor Training

Training progress is logged to:
- Console output
- `logs/` directory
- Weights & Biases (if enabled in config)

Checkpoints are saved to `models/checkpoints/step_*/`

## Detailed Workflow

### 1. Data Collection

The `DataCollector` class automatically:
- Crawls directories for text files
- Cleans and filters low-quality content
- Removes duplicates
- Saves to JSONL format

**Manual data collection:**
```python
from data_processing.data_collector import DataCollector
from data_processing.text_cleaner import TextCleaner

cleaner = TextCleaner(min_length=50, max_urls=5)
collector = DataCollector(output_dir="data/processed", cleaner=cleaner)

# Collect from directory
output_file = collector.collect_from_directory(
    source_dir="data/raw",
    recursive=True,
)
```

### 2. Tokenizer Training

Train a custom SentencePiece tokenizer optimized for your data:

```python
from tokenization.tokenizer_trainer import TokenizerTrainer

trainer = TokenizerTrainer(
    vocab_size=50000,
    model_type="bpe",
)

model_path, vocab_path = trainer.train_from_jsonl(
    jsonl_file="data/processed/collected_texts.jsonl",
    output_prefix="deepiri_tokenizer",
    output_dir="tokenizers",
)
```

### 3. Dataset Preparation

Convert cleaned text to tokenized datasets:

```python
from data_processing.dataset_builder import DatasetBuilder

builder = DatasetBuilder(
    tokenizer_model_path="tokenizers/deepiri_tokenizer.model",
    max_length=8192,
)

dataset = builder.build_from_jsonl(
    jsonl_file="data/processed/collected_texts.jsonl",
    train_split=0.9,
    val_split=0.05,
    test_split=0.05,
)

builder.save_dataset(dataset, "data/tokenized", format="arrow")
```

### 4. Pretraining

Train the base language model:

```python
from core.device_manager import DeviceManager
from core.training_config import TrainingConfig, ModelConfig
from models.transformer_lm import create_model_from_config
from training.pretraining_trainer import PretrainingTrainer

# Setup
device_manager = DeviceManager()  # Auto-detects CPU/GPU
model_config = ModelConfig.from_file("configs/model_config.json")
training_config = TrainingConfig.from_file("configs/training_config.json")

# Create model
model = create_model_from_config(model_config)

# Create trainer
trainer = PretrainingTrainer(
    model=model,
    config=training_config,
    model_config=model_config,
    device_manager=device_manager,
)

# Train
trainer.train(train_loader, val_loader)
```

### 5. Instruction Fine-Tuning

Fine-tune for instruction following:

```bash
python scripts/train_instruction_finetuning.py \
    --pretrained-model models/checkpoints/step_300000 \
    --instruction-data data/instruction_data.jsonl \
    --tokenizer-model tokenizers/deepiri_tokenizer.model
```

Instruction data format (JSONL):
```json
{"instruction": "Explain TCP vs UDP", "response": "TCP is a connection-oriented protocol..."}
{"instruction": "Write a Python function to sort a list", "response": "def sort_list(items):\n    return sorted(items)"}
```

## Configuration

### Model Configuration (`model_config.json`)

Key parameters:
- `hidden_size`: Model dimension (4096 for 7B-class model)
- `num_layers`: Number of transformer layers (32 for 7B-class)
- `num_attention_heads`: Attention heads (32)
- `vocab_size`: Vocabulary size (50000)
- `max_position_embeddings`: Maximum sequence length (8192)
- `use_rms_norm`: Use RMSNorm instead of LayerNorm
- `use_swiglu`: Use SwiGLU activation
- `use_rope_embeddings`: Use RoPE position embeddings

### Training Configuration (`training_config.json`)

Key parameters:
- `batch_size`: Batch size per device (2)
- `gradient_accumulation_steps`: Gradient accumulation (16)
- `learning_rate`: Learning rate (3e-4)
- `total_steps`: Total training steps (300000)
- `warmup_steps`: Warmup steps (2000)
- `save_steps`: Checkpoint frequency (10000)
- `mixed_precision`: Use mixed precision training (true)
- `device`: Force device or null for auto-detection

### Data Configuration (`data_config.json`)

Key parameters:
- `min_text_length`: Minimum text length (50)
- `max_text_length`: Maximum text length (8192)
- `tokenizer_vocab_size`: Tokenizer vocabulary size (50000)
- `train_split`: Training split ratio (0.9)

## Device Management

The system automatically detects and uses available hardware:

```python
from core.device_manager import DeviceManager

# Auto-detect
device_manager = DeviceManager()
print(device_manager.get_device_info())

# Force device
device_manager = DeviceManager(force_device="cpu")
device_manager = DeviceManager(force_device="cuda")
```

**Device Priority:**
1. CUDA (if available)
2. MPS (Apple Silicon, if available)
3. CPU (fallback)

## Monitoring & Evaluation

### Training Metrics

- **Loss**: Cross-entropy loss
- **Perplexity**: exp(loss)
- **Learning Rate**: Current learning rate

### Evaluation

Evaluation runs automatically during training:
- Every `eval_steps` steps
- Computes perplexity on validation set
- Logs to console and W&B

### Checkpointing

Checkpoints are saved:
- Every `save_steps` steps
- Contains: model, optimizer, scheduler, training state
- Old checkpoints automatically cleaned (keeps `save_total_limit`)

## Advanced Features

### Mixed Precision Training

Automatically enabled for GPU training. Reduces memory usage and speeds up training.

### Gradient Checkpointing

Enabled by default. Trades compute for memory, allowing larger models.

### Resume Training

```python
trainer.train(
    train_loader,
    resume_from_checkpoint=Path("models/checkpoints/step_10000"),
)
```

### Custom Data Sources

Add custom data collectors by extending `DataCollector`:

```python
class CustomDataCollector(DataCollector):
    def collect_from_api(self, api_endpoint):
        # Custom collection logic
        pass
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_sequence_length`
- Enable `gradient_checkpointing`

### Slow Training

- Use GPU (CUDA)
- Enable `mixed_precision`
- Increase `batch_size` (if memory allows)
- Reduce `data_loader_num_workers` for CPU

### Tokenizer Issues

- Ensure sufficient training data (millions of tokens)
- Adjust `vocab_size` based on data size
- Try different `model_type` (bpe, unigram)

## Best Practices

1. **Data Quality**: Clean, diverse, high-quality data is critical
2. **Tokenizer**: Train on representative data
3. **Monitoring**: Monitor loss and perplexity closely
4. **Checkpoints**: Save frequently, keep multiple checkpoints
5. **Evaluation**: Regular evaluation on held-out data
6. **Hyperparameters**: Start with defaults, tune gradually

## Next Steps

After pretraining:
1. Evaluate on downstream tasks
2. Fine-tune for specific domains
3. Instruction fine-tuning for chat
4. Alignment (RLHF/DPO) - advanced

## Support

For issues or questions:
- Check logs in `logs/`
- Review configuration files
- Consult PyTorch documentation
- Check device compatibility

## License

See main repository LICENSE file.

