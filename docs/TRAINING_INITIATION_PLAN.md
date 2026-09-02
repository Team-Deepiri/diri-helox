# Training Initiation Plan — Cyrex Runtime + Control Plane AI

**Date**: 2026-09-02
**Status**: READY TO EXECUTE
**Owner**: Deepiri ML Engineering

---

## Objective

Activate the full Helox training pipeline to produce production models for:
1. **Cyrex Runtime AI** — inference, RAG, and agent services (`diri-cyrex`)
2. **Control Plane AI** — intent classification, routing, and orchestration intelligence

All four core subsystems — **Helox**, **Training Orchestrator**, **Dataset Processor**, and **LIS** — must operate as a unified pipeline.

---

## Current State Assessment

| Component | Status | Readiness |
|-----------|--------|-----------|
| `UnifiedTrainingOrchestrator` | 38 features implemented | Ready |
| `deepiri-training-orchestrator` v0.4.0 | Installed as dep | Ready |
| `deepiri-dataset-processor` v0.3.1 | Installed as dep | Ready |
| LIS (Language Intelligence Service) | Produces training samples to Redis | Ready |
| Cyrex RAG Bridge | Auto-discovery configured | Ready |
| Synapse Event Publisher | Redis Streams connected | Ready |
| DeepSpeed ZeRO Stage 2 | Config present | Ready |
| LoRA / QLoRA | Infrastructure in `mlops/infrastructure/` | Ready |
| MLflow / W&B | Experiment tracking configured | Ready |

**Gap**: No active training jobs have been launched. The pipeline is built but idle.

---

## Execution Plan

### Phase 1: Data Pipeline Activation

**Goal**: Ensure LIS → Redis → Helox data flow is operational.

- [ ] **1.1** Verify LIS is publishing to `pipeline.helox-training.raw` and `pipeline.helox-training.structured` Redis streams
- [ ] **1.2** Run `scripts/generate_synthetic_data.py` to seed initial training corpus (5000+ examples across 8 task categories)
- [ ] **1.3** Run dataset versioning via `scripts/dataset_versioning_cli.py` to snapshot initial dataset as `v1.0.0`
- [ ] **1.4** Validate data quality: null checks, deduplication via `data_management/semantic_deduplication_engine.py`, token distribution via `data_management/token_distribution_monitor.py`
- [ ] **1.5** Confirm Postgres mirror table `cyrex.helox_training_samples` is receiving durable copies

### Phase 2: Control Plane AI Training

**Goal**: Train the intent classifier / routing model that powers Cyrex's request understanding.

- [ ] **2.1** Execute `scripts/training/train_task_classifier.py` — DeBERTa-v3-base fine-tuning for 8-class intent classification
- [ ] **2.2** Target domains: Coding, Writing, Fitness, Cleaning, Learning, Creative, Administrative, Social
- [ ] **2.3** Run HPO via `scripts/run_hpo.py` (Optuna) to optimize learning rate, batch size, epochs
- [ ] **2.4** Evaluate with `helox_sdk` post-training evaluation harness
- [ ] **2.5** Export model to ONNX via `model_export/` for Cyrex runtime deployment
- [ ] **2.6** Register in model registry via `model_management/model_provenance_system.py`
- [ ] **2.7** Publish `model-ready` event to Redis Stream for Cyrex `AutoModelLoader` pickup

### Phase 3: Cyrex Runtime Model Training

**Goal**: Fine-tune the language model for Cyrex's RAG and agent capabilities.

- [ ] **3.1** Configure `configs/training_config.json` for LoRA/QLoRA instruction fine-tuning
- [ ] **3.2** Run `scripts/train_instruction_finetuning.py` using LIS-produced structured training pairs
- [ ] **3.3** Enable RAG-aware training via `integrations/rag_aware_training_integration.py` — context window packing with retrieved content
- [ ] **3.4** Enable domain weighting via `data_management/domain_weighting_engine.py` for lease abstraction, contract intelligence, obligation dependency, regulatory language
- [ ] **3.5** Run full pipeline via `pipelines/training/full_training_pipeline.py` with MLflow + W&B tracking
- [ ] **3.6** Monitor via `observability/` — perplexity, loss curves, gradient health, attention entropy
- [ ] **3.7** Export final model (PyTorch checkpoint + ONNX)
- [ ] **3.8** Register and publish `model-ready` event

### Phase 4: Advanced Training Capabilities

**Goal**: Activate dynamic, versioned, and distributed training modes.

- [ ] **4.1** **Versioned Training**: Pin dataset versions via `pipelines/training/versioned_training_pipeline.py`
- [ ] **4.2** **Dynamic Pipeline**: Switch to config-driven training via `configs/dynamic_pipeline_config.json`
- [ ] **4.3** **Live Stream Training**: Enable real-time learning from LIS via `configs/dynamic_pipeline_stream_live_config.json`
- [ ] **4.4** **Distributed Training**: Multi-GPU via `pipelines/training/distributed_training.py` + DeepSpeed ZeRO Stage 2
- [ ] **4.5** **Bandit Training**: Multi-armed bandit for challenge selection via `pipelines/training/bandit_training.py`
- [ ] **4.6** **Fraud Detection**: Train vendor risk / fraud models via `pipelines/training/fraud_detection_training.py`
- [ ] **4.7** **Personalization**: Train personalization model via `scripts/training/train_personalization_model.py`

### Phase 5: Integration Validation

**Goal**: Prove end-to-end pipeline works Helox → Cyrex.

- [ ] **5.1** Run integration probe: `skills/integration-probe` against live Cyrex instance
- [ ] **5.2** Verify `CyrexRAGBridge` connects and retrieves context
- [ ] **5.3** Verify Synapse events flow: training-started → progress → checkpoint → completed → model-ready
- [ ] **5.4** Verify Cyrex `AutoModelLoader` picks up new model from `model-ready` event
- [ ] **5.5** Run inference parity test via `evaluation/inference_parity_tester.py`
- [ ] **5.6** End-to-end smoke: LIS processes document → Helox trains → Cyrex serves inference

---

## Key Commands

```bash
# Activate virtual environment
poetry shell

# Seed synthetic data
python scripts/generate_synthetic_data.py --count 5000 --output data/seed/

# Start training worker (consumes Redis stream jobs)
python scripts/run_training_worker.py

# Train intent classifier (Control Plane AI)
python scripts/training/train_task_classifier.py

# Instruction fine-tuning (Cyrex Runtime AI)
python scripts/train_instruction_finetuning.py --config configs/training_config.json

# Full pipeline with all features
python scripts/train_with_full_features.py --orchestrator unified

# HPO
python scripts/run_hpo.py --trials 50

# Dataset versioning
python scripts/dataset_versioning_cli.py create --name "initial-seed" --version 1.0.0

# Export model
python -m model_export.export --format onnx --output models/exported/

# Run tests
pytest tests/ -v
```

---

## Redis Stream Topology

```
pipeline.helox-training.raw          ← LIS publishes raw text
pipeline.helox-training.structured   ← LIS publishes instruction triples
training-jobs                        ← Cyrex submits training jobs
training-events                      ← Helox publishes lifecycle events
model-ready                          ← Helox signals new model available
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Insufficient training data | Seed with synthetic generator (5000+ examples), backfill from Postgres mirror |
| GPU OOM | DeepSpeed ZeRO Stage 2 with CPU offload, gradient checkpointing enabled |
| Model quality | Curriculum learning scheduler, gradient monitoring, numerical stability manager |
| Data staleness | Live stream training mode consumes fresh LIS samples in real-time |
| Integration drift | Inference parity tester validates Helox export matches Cyrex expectations |
| Experiment reproducibility | `ReproducibilityController` seeds all RNGs, MLflow logs all params |

---

## Success Criteria

1. **Control Plane classifier** achieves ≥90% accuracy on 8-class intent task
2. **Cyrex runtime model** shows improved RAG answer quality on lease/contract/obligation/regulatory domains
3. **End-to-end pipeline** flows: LIS → Redis → Helox training → model export → Cyrex inference
4. **All experiments** tracked in MLflow with full param/metric provenance
5. **Model-ready events** consumed by Cyrex AutoModelLoader without errors

---

## Next Steps After Training

- Deploy Control Plane classifier to Cyrex routing layer
- Deploy finetuned runtime model to Cyrex Ollama / inference endpoints
- Set up continuous training loop: LIS samples → periodic Helox retraining → hot-reload via Cyrex DynamicLoRAService
- Monitor production metrics and iterate on training data quality
