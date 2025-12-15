# Helox Integration Guide

## Complete Integration with Cyrex and Synapse

This guide shows how to use Helox with all 38 features integrated and connected to Cyrex (RAG) and Synapse (events).

## Quick Start

### 1. Basic Training (All Features Enabled)

```python
import asyncio
from pathlib import Path
from core.training_config import TrainingConfig, ModelConfig, DataConfig
from training.unified_training_orchestrator import UnifiedTrainingOrchestrator
from integrations.cyrex_rag_bridge import create_cyrex_rag_bridge
from tokenization.tokenizer_manager import TokenizerManager

async def train():
    # Load configs
    model_config = ModelConfig.from_file("configs/model_config.json")
    data_config = DataConfig.from_file("configs/data_config.json")
    training_config = TrainingConfig.from_file("configs/training_config.json")
    
    # Initialize RAG bridge (optional)
    rag_bridge = create_cyrex_rag_bridge(auto_discover=True)
    rag_pipeline = rag_bridge if rag_bridge.is_available() else None
    
    # Create orchestrator
    orchestrator = UnifiedTrainingOrchestrator(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        rag_pipeline=rag_pipeline,
        seed=1337,
    )
    
    # Initialize
    await orchestrator.initialize()
    
    # Load tokenizer
    orchestrator.tokenizer_manager = TokenizerManager("tokenizers/deepiri_tokenizer.model")
    
    # Create model
    orchestrator.create_model()
    orchestrator.create_optimizer_and_scheduler()
    
    # Create data loaders
    train_loader, val_loader = orchestrator.create_data_loaders(
        train_dataset_path=Path("data/tokenized/pretraining_dataset/train"),
        val_dataset_path=Path("data/tokenized/pretraining_dataset/validation"),
    )
    
    # Train
    await orchestrator.train(train_loader, val_loader)
    
    # Cleanup
    await orchestrator.cleanup()

asyncio.run(train())
```

### 2. Using the Command Line Script

```bash
python scripts/train_with_full_features.py \
    --config-dir configs \
    --train-data data/tokenized/pretraining_dataset/train \
    --val-data data/tokenized/pretraining_dataset/validation \
    --tokenizer tokenizers/deepiri_tokenizer.model \
    --seed 1337 \
    --enable-rag
```

## Integration Points

### Synapse Integration

Helox automatically publishes training events to Synapse:

- **Training Started**: When training begins
- **Training Progress**: Every 1000 steps
- **Checkpoint Saved**: When checkpoints are created
- **Model Ready**: When training completes and model is exported
- **Training Completed**: Final completion event

Cyrex subscribes to these events via Synapse and automatically loads models when `model-ready` events are published.

### Cyrex RAG Integration

When `--enable-rag` is used, Helox:

1. Auto-discovers Cyrex RAG pipeline
2. Uses RAG for context-aware training
3. Packs retrieved context into training sequences
4. Creates retrieval-conditioned prompts

The RAG bridge handles:
- Connection to Cyrex RAG service
- Fallback when RAG unavailable
- Error handling and retries

## Feature Usage

### Reproducibility

All training runs are fully reproducible:
- Global seed control
- Deterministic operations
- Training fingerprints for tracking

### Dataset Versioning

```python
from data_management.dataset_versioning_system import DatasetVersioningSystem

versioning = DatasetVersioningSystem()
version_record = versioning.create_dataset_version(
    dataset_path=Path("data/processed/collected_texts.jsonl"),
    dataset_id="corpus_v3.2",
    tokenizer_manager=tokenizer_manager,
)
```

### Streaming Datasets

```python
from data_management.streaming_dataset_manager import ShardedDatasetManager

manager = ShardedDatasetManager()
train_loader = manager.create_streaming_dataloader(
    data_paths=[Path("data/processed/large_corpus.jsonl")],
    tokenizer_manager=tokenizer_manager,
    batch_size=2,
    num_shards=4,  # For distributed training
    shard_id=0,
)
```

### RAG-Aware Training

```python
from integrations.rag_aware_training_integration import RAGAwareTrainingIntegrator

rag_integrator = RAGAwareTrainingIntegrator(
    rag_pipeline=cyrex_rag_pipeline,
    max_context_length=8192,
)

# Pack context with retrieval
packed = rag_integrator.pack_context_with_retrieval(
    query="Python programming",
    base_text="def hello(): print('world')",
    tokenizer_manager=tokenizer_manager,
)
```

### Health Monitoring

```python
from observability.training_health_monitor import TrainingHealthMonitor

health_monitor = TrainingHealthMonitor()
health_status = health_monitor.check_loss(loss, step)

if not health_status["healthy"]:
    # Handle alerts
    for alert in health_status["alerts"]:
        logger.error(f"Alert: {alert}")
```

### Evaluation

```python
from evaluation.automatic_evaluation_harness import AutomaticEvaluationHarness

eval_harness = AutomaticEvaluationHarness()
eval_harness.load_test_suite("code_generation", "evaluation/code_tests.jsonl")

results = eval_harness.evaluate_model(
    model=model,
    tokenizer_manager=tokenizer_manager,
    suite_name="code_generation",
)
```

## Configuration

All features are configured via JSON config files:

- `configs/model_config.json` - Model architecture
- `configs/training_config.json` - Training hyperparameters
- `configs/data_config.json` - Data processing settings

## Event Flow

```
Helox Training
    ↓
Publishes events to Synapse
    ↓
Synapse (Redis Streams)
    ↓
Cyrex subscribes
    ↓
Auto-loads model when ready
```

## Error Handling

All components include comprehensive error handling:

- **RAG unavailable**: Falls back gracefully
- **Synapse unavailable**: Logs warning, continues training
- **Checkpoint failures**: Saves recovery state
- **Gradient overflow**: Automatically adjusts loss scale
- **Training divergence**: Health monitor alerts

## Next Steps

1. **Configure**: Edit config files for your use case
2. **Prepare Data**: Place data in `data/datasets/raw/`
3. **Train Tokenizer**: Run tokenizer training
4. **Start Training**: Use `train_with_full_features.py`
5. **Monitor**: Check logs and Synapse events
6. **Deploy**: Models auto-export and publish to Synapse

## Troubleshooting

### RAG Not Available
- Check Cyrex is running
- Verify MILVUS connection
- Check environment variables

### Synapse Connection Failed
- Verify Redis is running
- Check REDIS_URL environment variable
- Training continues without events

### Out of Memory
- Reduce batch size in config
- Enable gradient checkpointing
- Use optimizer state sharding

## Support

See `IMPLEMENTATION_COMPLETE.md` for complete feature list and `README_LLM_TRAINING.md` for detailed documentation.

