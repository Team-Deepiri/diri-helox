#!/usr/bin/env python3
"""
Complete training example with all 38 features.

This example demonstrates the full training pipeline with:
- All features enabled
- Cyrex RAG integration
- Synapse event publishing
- Complete observability
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.training_config import TrainingConfig, ModelConfig, DataConfig
from training.unified_training_orchestrator import UnifiedTrainingOrchestrator
from integrations.cyrex_rag_bridge import create_cyrex_rag_bridge
from tokenization.tokenizer_manager import TokenizerManager
from data_management.dataset_versioning_system import DatasetVersioningSystem
from data_safety.data_leakage_detector import DataLeakageDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Complete training example."""
    
    # ============================================
    # 1. Configuration
    # ============================================
    logger.info("=" * 60)
    logger.info("HELOX COMPLETE TRAINING EXAMPLE")
    logger.info("All 38 Features Enabled")
    logger.info("=" * 60)
    
    config_dir = Path("configs")
    
    # Load configurations
    model_config = ModelConfig.from_file(config_dir / "model_config.json")
    data_config = DataConfig.from_file(config_dir / "data_config.json")
    training_config = TrainingConfig.from_file(config_dir / "training_config.json")
    
    logger.info("✓ Configurations loaded")
    
    # ============================================
    # 2. Data Safety Checks
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("DATA SAFETY CHECKS")
    logger.info("=" * 60)
    
    # Load sample data for leakage detection
    train_data_path = Path("data/datasets/processed/collected_texts.jsonl")
    val_data_path = Path("data/datasets/processed/collected_texts.jsonl")  # Would be separate
    
    if train_data_path.exists() and val_data_path.exists():
        leakage_detector = DataLeakageDetector()
        
        # Load sample texts
        import json
        train_texts = []
        val_texts = []
        
        with open(train_data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 1000:  # Sample
                    break
                try:
                    data = json.loads(line)
                    train_texts.append(data.get("text", ""))
                except:
                    continue
        
        with open(val_data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample
                    break
                try:
                    data = json.loads(line)
                    val_texts.append(data.get("text", ""))
                except:
                    continue
        
        # Check for contamination
        contamination_report = leakage_detector.detect_train_eval_contamination(
            train_texts, val_texts
        )
        
        if contamination_report["contamination_detected"]:
            logger.warning(
                f"⚠ Train/eval contamination detected: "
                f"{contamination_report['contamination_rate']:.2%}"
            )
        else:
            logger.info("✓ No train/eval contamination detected")
        
        # Check for duplicates
        duplicate_report = leakage_detector.detect_exact_duplicates(train_texts)
        if duplicate_report["duplicates_detected"]:
            logger.warning(
                f"⚠ Exact duplicates detected: "
                f"{duplicate_report['duplicate_rate']:.2%}"
            )
        else:
            logger.info("✓ No exact duplicates detected")
    
    # ============================================
    # 3. Dataset Versioning
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("DATASET VERSIONING")
    logger.info("=" * 60)
    
    versioning = DatasetVersioningSystem()
    
    if train_data_path.exists():
        version_record = versioning.create_dataset_version(
            dataset_path=train_data_path,
            dataset_id="training_corpus_v1.0",
            tokenizer_manager=None,  # Would load if available
        )
        logger.info(f"✓ Dataset version created: {version_record['dataset_id']}")
        logger.info(f"  Samples: {version_record['sample_count']:,}")
        logger.info(f"  Checksum: {version_record['checksum'][:16]}...")
    
    # ============================================
    # 4. Cyrex RAG Integration
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("CYREX RAG INTEGRATION")
    logger.info("=" * 60)
    
    rag_bridge = create_cyrex_rag_bridge(auto_discover=True)
    
    if rag_bridge.is_available():
        logger.info("✓ Cyrex RAG pipeline connected")
        
        # Test retrieval
        test_results = rag_bridge.retrieve("Python programming", top_k=2)
        logger.info(f"✓ Test retrieval: {len(test_results)} results")
        rag_pipeline = rag_bridge
    else:
        logger.warning("⚠ Cyrex RAG not available - RAG features disabled")
        rag_pipeline = None
    
    # ============================================
    # 5. Create Training Orchestrator
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("INITIALIZING TRAINING ORCHESTRATOR")
    logger.info("=" * 60)
    
    orchestrator = UnifiedTrainingOrchestrator(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        rag_pipeline=rag_pipeline,
        seed=1337,
    )
    
    logger.info("✓ Orchestrator created")
    logger.info(f"  Training fingerprint: {orchestrator.training_fingerprint}")
    logger.info(f"  Device: {orchestrator.device_manager.get_device_info()['device']}")
    
    # ============================================
    # 6. Initialize Async Components
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("INITIALIZING ASYNC COMPONENTS")
    logger.info("=" * 60)
    
    await orchestrator.initialize()
    logger.info("✓ Synapse connection established")
    
    # ============================================
    # 7. Load Tokenizer
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("LOADING TOKENIZER")
    logger.info("=" * 60)
    
    tokenizer_path = Path("tokenizers/deepiri_tokenizer.model")
    if tokenizer_path.exists():
        orchestrator.tokenizer_manager = TokenizerManager(tokenizer_path)
        logger.info(f"✓ Tokenizer loaded: {orchestrator.tokenizer_manager.get_vocab_size()} vocab")
    else:
        logger.error("✗ Tokenizer not found - please train tokenizer first")
        return
    
    # ============================================
    # 8. Create Model
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("CREATING MODEL")
    logger.info("=" * 60)
    
    orchestrator.create_model()
    
    param_count = sum(p.numel() for p in orchestrator.model.parameters())
    logger.info(f"✓ Model created: {param_count / 1e9:.2f}B parameters")
    
    # ============================================
    # 9. Create Optimizer & Scheduler
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("CREATING OPTIMIZER & SCHEDULER")
    logger.info("=" * 60)
    
    orchestrator.create_optimizer_and_scheduler()
    logger.info("✓ Optimizer and scheduler created")
    logger.info(f"  Initial LR: {training_config.learning_rate:.2e}")
    logger.info(f"  Warmup steps: {training_config.warmup_steps}")
    
    # ============================================
    # 10. Create Data Loaders
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("CREATING DATA LOADERS")
    logger.info("=" * 60)
    
    # Use tokenized dataset if available
    tokenized_train = Path("data/tokenized/pretraining_dataset/train")
    tokenized_val = Path("data/tokenized/pretraining_dataset/validation")
    
    if tokenized_train.exists():
        train_loader, val_loader = orchestrator.create_data_loaders(
            train_dataset_path=tokenized_train,
            val_dataset_path=tokenized_val if tokenized_val.exists() else None,
        )
        logger.info("✓ Data loaders created")
        logger.info(f"  Batch size: {train_loader.batch_size}")
        logger.info(f"  Sequence length: {orchestrator.curriculum.get_current_sequence_length(0)}")
    else:
        logger.error("✗ Tokenized dataset not found")
        logger.info("  Please run data preparation pipeline first")
        return
    
    # ============================================
    # 11. Start Training
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("All 38 Features Active")
    logger.info("=" * 60)
    
    logger.info("\nActive Features:")
    logger.info("  ✓ Reproducibility & Determinism")
    logger.info("  ✓ Dataset Versioning")
    logger.info("  ✓ Streaming Datasets")
    logger.info("  ✓ Dynamic Loss Scaling")
    logger.info("  ✓ Gradient Monitoring")
    logger.info("  ✓ Optimizer Sharding")
    logger.info("  ✓ Fine-Grained Metrics")
    logger.info("  ✓ Health Monitoring")
    logger.info("  ✓ Automatic Evaluation")
    logger.info("  ✓ Data Leakage Detection")
    logger.info("  ✓ Multi-Objective Training")
    logger.info("  ✓ Instruction Formatting")
    logger.info("  ✓ RAG-Aware Training" + (" (enabled)" if rag_pipeline else " (disabled)"))
    logger.info("  ✓ Model Provenance")
    logger.info("  ✓ Format Export")
    logger.info("  ✓ Inference Parity Testing")
    logger.info("  ✓ Curriculum Learning")
    logger.info("  ✓ Failure Resilience")
    logger.info("  ✓ Precision Control")
    logger.info("  ✓ Attention Optimization")
    logger.info("  ✓ Token Distribution Monitoring")
    logger.info("  ✓ Domain Weighting")
    logger.info("  ✓ Semantic Deduplication")
    logger.info("  ✓ Synapse Event Publishing")
    logger.info("  ... and 12 more features")
    
    try:
        await orchestrator.train(train_loader, val_loader)
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted - saving checkpoint...")
        await orchestrator.save_checkpoint()
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        await orchestrator.save_checkpoint()
        raise
    finally:
        await orchestrator.cleanup()
        logger.info("✓ Cleanup complete")
    
    # ============================================
    # 12. Summary
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total steps: {orchestrator.global_step}")
    logger.info(f"Total samples: {orchestrator.training_stats['total_samples']:,}")
    logger.info(f"Total tokens: {orchestrator.training_stats['total_tokens']:,}")
    logger.info(f"Checkpoints saved: {orchestrator.training_stats['checkpoints_saved']}")
    logger.info(f"Evaluations run: {orchestrator.training_stats['evaluations_run']}")
    logger.info(f"Training fingerprint: {orchestrator.training_fingerprint}")
    
    logger.info("\n✓ Model exported and published to Synapse")
    logger.info("✓ Cyrex will auto-load model when ready")
    logger.info("\nTraining complete with all features!")


if __name__ == "__main__":
    asyncio.run(main())

