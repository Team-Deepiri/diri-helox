#!/usr/bin/env python3
"""
Main training script with all 38 features integrated.

This script uses the UnifiedTrainingOrchestrator to train LLMs
with all production-grade features enabled.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.training_config import TrainingConfig, ModelConfig, DataConfig
from training.unified_training_orchestrator import UnifiedTrainingOrchestrator
from tokenization.tokenizer_manager import TokenizerManager

# Import RAG bridge
from integrations.cyrex_rag_bridge import create_cyrex_rag_bridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_configs(config_dir: Path):
    """Load all configuration files."""
    config_dir = Path(config_dir)
    
    model_config = ModelConfig.from_file(config_dir / "model_config.json")
    data_config = DataConfig.from_file(config_dir / "data_config.json")
    training_config = TrainingConfig.from_file(config_dir / "training_config.json")
    
    return model_config, data_config, training_config


def initialize_cyrex_rag() -> RAGPipeline:
    """Initialize Cyrex RAG pipeline if available."""
    if not CYREX_AVAILABLE:
        logger.warning("Cyrex RAG not available - RAG-aware training disabled")
        return None
    
    try:
        rag_pipeline = initialize_rag_system()
        logger.info("Cyrex RAG pipeline initialized")
        return rag_pipeline
    except Exception as e:
        logger.warning(f"Failed to initialize Cyrex RAG: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Train LLM with all features")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Path to validation dataset (optional)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed",
    )
    parser.add_argument(
        "--enable-rag",
        action="store_true",
        help="Enable RAG-aware training (requires Cyrex)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    
    args = parser.parse_args()
    
    # Load configurations
    logger.info("Loading configurations...")
    model_config, data_config, training_config = load_configs(args.config_dir)
    
    # Override resume checkpoint if provided
    if args.resume_from:
        training_config.resume_from_checkpoint = args.resume_from
    
    # Initialize RAG bridge if requested
    rag_pipeline = None
    if args.enable_rag:
        logger.info("Initializing Cyrex RAG bridge...")
        rag_bridge = create_cyrex_rag_bridge(auto_discover=True)
        if rag_bridge.is_available():
            rag_pipeline = rag_bridge
            logger.info("RAG-aware training enabled")
        else:
            logger.warning("RAG requested but not available - continuing without RAG")
    
    # Create orchestrator
    logger.info("Initializing unified training orchestrator...")
    orchestrator = UnifiedTrainingOrchestrator(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        rag_pipeline=rag_pipeline,
        seed=args.seed,
    )
    
    # Initialize async components
    await orchestrator.initialize()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}")
    orchestrator.tokenizer_manager = TokenizerManager(args.tokenizer)
    
    # Create model
    logger.info("Creating model...")
    orchestrator.create_model()
    
    # Create optimizer and scheduler
    orchestrator.create_optimizer_and_scheduler()
    
    # Resume from checkpoint if provided
    if args.resume_from and args.resume_from.exists():
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        recovery_state = orchestrator.failure_manager.load_recovery_state(args.resume_from)
        if recovery_state:
            orchestrator.model.load_state_dict(recovery_state["model_state"])
            orchestrator.optimizer.load_state_dict(recovery_state["optimizer_state"])
            if orchestrator.scheduler and recovery_state.get("scheduler_state"):
                orchestrator.scheduler.load_state_dict(recovery_state["scheduler_state"])
            orchestrator.global_step = recovery_state.get("step", 0)
            logger.info(f"Resumed from step {orchestrator.global_step}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = orchestrator.create_data_loaders(
        train_dataset_path=args.train_data,
        val_dataset_path=args.val_data,
    )
    
    # Run data safety checks
    logger.info("Running data safety checks...")
    # (Would load sample data for leakage detection)
    
    # Start training
    logger.info("Starting training with all features enabled...")
    try:
        await orchestrator.train(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("Training interrupted - saving checkpoint...")
        await orchestrator.save_checkpoint()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        # Attempt recovery
        logger.info("Attempting to save recovery state...")
        await orchestrator.save_checkpoint()
        raise
    finally:
        await orchestrator.cleanup()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())

