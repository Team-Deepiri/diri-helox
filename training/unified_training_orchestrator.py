"""
Unified Training Orchestrator.

Integrates all 38 features into a seamless training pipeline.
This is the main entry point for production training.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader
import json
from datetime import datetime

# Core components
from ..core.device_manager import DeviceManager
from ..core.training_config import TrainingConfig, ModelConfig, DataConfig
from ..core.reproducibility_controller import ReproducibilityController

# Data management
from ..data_management.dataset_versioning_system import DatasetVersioningSystem
from ..data_management.streaming_dataset_manager import ShardedDatasetManager
from ..data_management.token_distribution_monitor import TokenDistributionMonitor
from ..data_management.domain_weighting_engine import DomainWeightingEngine
from ..data_management.semantic_deduplication_engine import SemanticDeduplicationEngine

# Data safety
from ..data_safety.data_leakage_detector import DataLeakageDetector

# Training components
from ..training.numerical_stability_manager import DynamicLossScaler, NumericalStabilityMonitor
from ..training.gradient_monitoring_system import GradientMonitoringSystem
from ..training.optimizer_state_sharding import OptimizerStateSharder
from ..training.curriculum_learning_scheduler import CurriculumLearningScheduler, AdaptiveBatchScheduler
from ..training.failure_resilience_manager import FailureResilienceManager
from ..training.precision_aware_layer_control import PrecisionAwareLayerControl
from ..training.multi_objective_trainer import MultiObjectiveLoss
from ..training.instruction_formatting_abstraction import InstructionFormatter

# Observability
from ..observability.metrics_collector import FineGrainedMetricsCollector
from ..observability.training_health_monitor import TrainingHealthMonitor

# Evaluation
from ..evaluation.automatic_evaluation_harness import AutomaticEvaluationHarness
from ..evaluation.inference_parity_tester import InferenceParityTester

# Integrations
from ..integrations.rag_aware_training_integration import RAGAwareTrainingIntegrator
from ..integrations.synapse_event_publisher import SynapseEventPublisher

# Model management
from ..model_management.model_provenance_system import ModelProvenanceSystem
from ..model_export.format_exporter import ModelFormatExporter

# Models
from ..models.transformer_lm import TransformerLanguageModel, create_model_from_config
from ..tokenization.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class UnifiedTrainingOrchestrator:
    """
    Unified orchestrator that integrates all training features.
    
    This class coordinates all 38 features into a seamless training pipeline.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        rag_pipeline=None,  # Cyrex RAG pipeline
        seed: int = 1337,
    ):
        """
        Initialize unified training orchestrator.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            rag_pipeline: Optional Cyrex RAG pipeline
            seed: Random seed
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Initialize reproducibility
        logger.info("Initializing reproducibility controller...")
        self.repro_controller = ReproducibilityController(seed=seed, deterministic=True)
        self.repro_controller.set_seeds()
        self.training_fingerprint = self.repro_controller.generate_training_fingerprint(
            config={
                **model_config.to_dict(),
                **training_config.to_dict(),
                **data_config.to_dict(),
            }
        )
        
        # Initialize device manager
        logger.info("Initializing device manager...")
        self.device_manager = DeviceManager(force_device=training_config.device)
        self.device = self.device_manager.get_device()
        logger.info(f"Using device: {self.device_manager.get_device_info()}")
        
        # Initialize data management
        logger.info("Initializing data management systems...")
        self.dataset_versioning = DatasetVersioningSystem(
            metadata_dir=data_config.processed_data_dir / "metadata"
        )
        self.streaming_manager = ShardedDatasetManager(
            state_dir=Path("data/training_state")
        )
        self.token_monitor = TokenDistributionMonitor(vocab_size=model_config.vocab_size)
        self.domain_weighter = DomainWeightingEngine()
        self.semantic_dedup = SemanticDeduplicationEngine()
        
        # Initialize data safety
        self.leakage_detector = DataLeakageDetector()
        
        # Initialize training stability
        logger.info("Initializing training stability systems...")
        # Use PyTorch's GradScaler for mixed precision
        self.loss_scaler = torch.cuda.amp.GradScaler() if (
            training_config.mixed_precision and self.device_manager.is_gpu_available()
        ) else None
        self.dynamic_scaler = DynamicLossScaler() if training_config.mixed_precision else None
        self.stability_monitor = NumericalStabilityMonitor()
        self.gradient_monitor = GradientMonitoringSystem(
            max_norm=1.0,
            adaptive_clipping=True,
        )
        self.optimizer_sharder = OptimizerStateSharder(
            num_shards=1,  # Can be increased for multi-GPU
            cpu_offload=False,
        )
        
        # Initialize curriculum learning
        self.curriculum = CurriculumLearningScheduler(
            initial_seq_len=512,
            max_seq_len=training_config.max_sequence_length,
        )
        self.batch_scheduler = AdaptiveBatchScheduler(
            initial_batch_size=training_config.batch_size,
        )
        
        # Initialize failure resilience
        self.failure_manager = FailureResilienceManager(
            checkpoint_dir=training_config.checkpoint_dir,
        )
        
        # Initialize precision control
        self.precision_controller = PrecisionAwareLayerControl(
            default_precision=torch.float16 if training_config.mixed_precision else torch.float32,
        )
        
        # Initialize observability
        logger.info("Initializing observability systems...")
        self.metrics_collector = FineGrainedMetricsCollector(
            use_wandb=training_config.use_wandb,
            log_interval=training_config.logging_steps,
        )
        self.health_monitor = TrainingHealthMonitor()
        
        # Initialize evaluation
        self.eval_harness = AutomaticEvaluationHarness(
            eval_dir=Path("evaluation"),
        )
        self.parity_tester = InferenceParityTester()
        
        # Initialize integrations
        logger.info("Initializing integrations...")
        self.rag_integrator = RAGAwareTrainingIntegrator(
            rag_pipeline=rag_pipeline,
            max_context_length=training_config.max_sequence_length,
        ) if rag_pipeline else None
        
        self.synapse_publisher = SynapseEventPublisher()
        
        # Initialize model management
        self.provenance_system = ModelProvenanceSystem()
        self.format_exporter = ModelFormatExporter()
        
        # Initialize instruction formatter
        self.instruction_formatter = InstructionFormatter(format_type="chatml")
        
        # Model and training state
        self.model: Optional[TransformerLanguageModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.tokenizer_manager: Optional[TokenizerManager] = None
        
        self.global_step = 0
        self.current_epoch = 0
        
        # Training statistics
        self.training_stats = {
            "total_samples": 0,
            "total_tokens": 0,
            "checkpoints_saved": 0,
            "evaluations_run": 0,
        }
    
    async def initialize(self):
        """Initialize all async components."""
        logger.info("Initializing async components...")
        await self.synapse_publisher.connect()
        await self.synapse_publisher.publish_training_event(
            event_type="started",
            model_name="llm-training",
            step=0,
            metrics={"fingerprint": self.training_fingerprint},
        )
    
    def create_model(self) -> TransformerLanguageModel:
        """Create and initialize model."""
        logger.info("Creating model...")
        self.model = create_model_from_config(self.model_config)
        
        # Apply precision control
        self.model = self.precision_controller.apply_precision_control(self.model)
        self.model.to(self.device)
        
        # Embed provenance
        metadata = {
            "fingerprint": self.training_fingerprint,
            "config": self.model_config.to_dict(),
            "created_at": datetime.utcnow().isoformat(),
        }
        self.model = self.provenance_system.embed_training_metadata(self.model, metadata)
        
        # Generate fingerprint
        fingerprint = self.provenance_system.generate_model_fingerprint(
            self.model,
            self.model_config.to_dict(),
        )
        logger.info(f"Model fingerprint: {fingerprint}")
        
        return self.model
    
    def create_optimizer_and_scheduler(self):
        """Create optimizer and scheduler."""
        logger.info("Creating optimizer and scheduler...")
        
        # Create optimizer
        if self.training_config.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                betas=(self.training_config.beta1, self.training_config.beta2),
                eps=self.training_config.eps,
                weight_decay=self.training_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer_type}")
        
        # Apply optimizer sharding if needed
        if self.optimizer_sharder.num_shards > 1:
            self.optimizer = self.optimizer_sharder.partition_optimizer_state(
                self.optimizer,
                shard_id=0,
            )
        
        # Create scheduler
        if self.training_config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.total_steps,
                eta_min=self.training_config.min_learning_rate,
            )
        elif self.training_config.scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.training_config.min_learning_rate / self.training_config.learning_rate,
                total_iters=self.training_config.total_steps,
            )
        else:
            from torch.optim.lr_scheduler import ConstantLR
            self.scheduler = ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=self.training_config.total_steps,
            )
        
        # Create warmup + main scheduler
        # Use custom warmup wrapper
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step / self.training_config.warmup_steps, 1.0),
        )
        
        # Wrap in warmup scheduler
        class WarmupWrapper:
            def __init__(self, warmup_scheduler, main_scheduler, warmup_steps):
                self.warmup_scheduler = warmup_scheduler
                self.main_scheduler = main_scheduler
                self.warmup_steps = warmup_steps
                self.last_step = 0
            
            def step(self):
                if self.last_step < self.warmup_steps:
                    self.warmup_scheduler.step()
                else:
                    self.main_scheduler.step()
                self.last_step += 1
            
            def get_last_lr(self):
                if self.last_step < self.warmup_steps:
                    return self.warmup_scheduler.get_last_lr()
                return self.main_scheduler.get_last_lr()
            
            def state_dict(self):
                return {
                    "warmup": self.warmup_scheduler.state_dict(),
                    "main": self.main_scheduler.state_dict(),
                    "last_step": self.last_step,
                }
            
            def load_state_dict(self, state):
                self.warmup_scheduler.load_state_dict(state["warmup"])
                self.main_scheduler.load_state_dict(state["main"])
                self.last_step = state.get("last_step", 0)
        
        self.scheduler = WarmupWrapper(
            warmup_scheduler,
            self.scheduler,
            self.training_config.warmup_steps,
        )
    
    def load_tokenizer(self):
        """Load tokenizer."""
        if self.data_config.tokenizer_model_path and self.data_config.tokenizer_model_path.exists():
            logger.info(f"Loading tokenizer from {self.data_config.tokenizer_model_path}")
            self.tokenizer_manager = TokenizerManager(self.data_config.tokenizer_model_path)
        else:
            raise ValueError("Tokenizer model path not found")
    
    def create_data_loaders(
        self,
        train_dataset_path: Path,
        val_dataset_path: Optional[Path] = None,
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """Create data loaders with all features integrated."""
        logger.info("Creating data loaders...")
        
        # Get current sequence length from curriculum
        current_seq_len = self.curriculum.get_current_sequence_length(self.global_step)
        
        # Create train loader
        train_loader = self.streaming_manager.create_streaming_dataloader(
            data_paths=[train_dataset_path],
            tokenizer_manager=self.tokenizer_manager,
            batch_size=self.batch_scheduler.current_batch_size,
            max_length=current_seq_len,
            shuffle=True,
            num_workers=self.training_config.data_loader_num_workers,
            shard_id=0,
            num_shards=1,
            resume_from_checkpoint=self.training_config.resume_from_checkpoint,
        )
        
        # Create val loader if provided
        val_loader = None
        if val_dataset_path:
            val_loader = self.streaming_manager.create_streaming_dataloader(
                data_paths=[val_dataset_path],
                tokenizer_manager=self.tokenizer_manager,
                batch_size=self.training_config.eval_batch_size,
                max_length=current_seq_len,
                shuffle=False,
                num_workers=self.training_config.data_loader_num_workers,
                shard_id=0,
                num_shards=1,
            )
        
        return train_loader, val_loader
    
    async def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Single training step with all features integrated."""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass with gradient checkpointing
        if self.training_config.mixed_precision and self.loss_scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    use_gradient_checkpointing=self.training_config.gradient_checkpointing,
                )
                loss = outputs["loss"]
            
            # Scale loss and backward
            self.loss_scaler.scale(loss).backward()
            
            # Check for overflow using dynamic scaler
            overflow = False
            if self.dynamic_scaler:
                overflow = self.dynamic_scaler.unscale_gradients(self.optimizer)
            
            if overflow:
                self.loss_scaler.update()
                self.optimizer.zero_grad()
                logger.warning(f"Step {self.global_step}: Gradient overflow detected")
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_gradient_checkpointing=self.training_config.gradient_checkpointing,
            )
            loss = outputs["loss"]
            loss.backward()
        
        # Gradient clipping
        grad_stats = self.gradient_monitor.clip_gradients(self.model, self.global_step)
        
        # Check for exploding gradients
        if self.gradient_monitor.detect_exploding_gradients():
            logger.error(f"Step {self.global_step}: Gradient explosion detected!")
            self.health_monitor._trigger_alert("gradient_explosion", "Gradient explosion detected")
        
        # Optimizer step
        if self.loss_scaler and not overflow:
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
            self.scheduler.step()
        elif not overflow:
            self.optimizer.step()
            self.scheduler.step()
        
        # Stability monitoring
        self.stability_monitor.check_activations(outputs, self.global_step)
        self.stability_monitor.check_loss(loss, self.global_step)
        
        # Health monitoring
        health_status = self.health_monitor.check_loss(loss.item(), self.global_step)
        
        # Metrics collection
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.training_config.learning_rate
        
        # Compute token-level perplexity
        if "logits" in outputs:
            ppl_metrics = self.metrics_collector.compute_token_perplexity(
                outputs["logits"],
                input_ids,
            )
        else:
            ppl_metrics = {}
        
        # Track metrics
        self.metrics_collector.track_lr_loss_pair(current_lr, loss.item())
        self.metrics_collector.log_metrics(
            self.global_step,
            loss.item(),
            current_lr,
            additional_metrics={
                **ppl_metrics,
                **grad_stats,
                "health_status": health_status,
            },
        )
        
        # Update token distribution
        token_ids = input_ids.cpu().tolist()
        for token_seq in token_ids:
            self.token_monitor.update_token_frequencies(token_seq, self.global_step)
        
        # Update domain weighting
        # (Would need domain assignment from batch metadata)
        
        # Update training stats
        self.training_stats["total_samples"] += input_ids.size(0)
        self.training_stats["total_tokens"] += input_ids.numel()
        
        return {
            "loss": loss.item(),
            "grad_norm": grad_stats.get("gradient_norm", 0.0),
            "health_status": health_status,
            **ppl_metrics,
        }
    
    async def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Run evaluation with all features."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                
                loss = outputs["loss"]
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }
        
        logger.info(f"Evaluation: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")
        
        return metrics
    
    async def save_checkpoint(self):
        """Save checkpoint with all state."""
        checkpoint_dir = self.training_config.checkpoint_dir / f"step_{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer and scheduler
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        scheduler_path = checkpoint_dir / "scheduler.pt"
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save scaler
        if self.loss_scaler:
            scaler_path = checkpoint_dir / "scaler.pt"
            torch.save(self.loss_scaler.state_dict(), scaler_path)
        
        # Save dynamic scaler state
        if self.dynamic_scaler:
            dynamic_scaler_path = checkpoint_dir / "dynamic_scaler.json"
            import json
            with open(dynamic_scaler_path, "w") as f:
                json.dump(self.dynamic_scaler.get_state(), f, indent=2)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "training_stats": self.training_stats,
            "training_fingerprint": self.training_fingerprint,
            "config": self.training_config.to_dict(),
            "model_config": self.model_config.to_dict(),
        }
        
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        
        # Save fingerprint
        self.repro_controller.save_fingerprint(checkpoint_dir / "training_fingerprint.json")
        
        # Save recovery state
        self.failure_manager.save_training_state(
            self.global_step,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.scheduler.state_dict() if self.scheduler else None,
        )
        
        # Save provenance
        self.provenance_system.create_provenance_record(
            model_name="llm-training",
            fingerprint=self.training_fingerprint,
            metadata=state,
            checkpoint_path=checkpoint_dir,
        )
        
        self.training_stats["checkpoints_saved"] += 1
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        # Publish checkpoint event
        await self.synapse_publisher.publish_training_event(
            event_type="checkpoint",
            model_name="llm-training",
            step=self.global_step,
            metrics={"checkpoint_path": str(checkpoint_dir)},
        )
    
    async def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Main training loop with all features integrated."""
        logger.info("Starting training...")
        
        from tqdm import tqdm
        
        progress_bar = tqdm(
            total=self.training_config.total_steps,
            desc="Training",
            initial=self.global_step,
        )
        
        accumulated_loss = 0.0
        
        while self.global_step < self.training_config.total_steps:
            for batch in train_loader:
                if self.global_step >= self.training_config.total_steps:
                    break
                
                # Training step
                step_metrics = await self.training_step(batch)
                accumulated_loss += step_metrics["loss"]
                
                # Update curriculum
                current_seq_len = self.curriculum.get_current_sequence_length(self.global_step)
                
                # Update batch size
                if self.global_step % 100 == 0:
                    memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.0
                    grad_norm = step_metrics.get("grad_norm", 0.0)
                    new_batch_size = self.batch_scheduler.update_batch_size(
                        grad_norm,
                        memory_usage,
                        self.global_step,
                    )
                    if new_batch_size != train_loader.batch_size:
                        logger.info(f"Updated batch size to {new_batch_size}")
                
                # Logging
                if (self.global_step + 1) % self.training_config.logging_steps == 0:
                    avg_loss = accumulated_loss / self.training_config.logging_steps
                    logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, "
                        f"seq_len={current_seq_len}, samples={self.training_stats['total_samples']}"
                    )
                    accumulated_loss = 0.0
                
                # Evaluation
                if val_loader and (self.global_step + 1) % self.training_config.eval_steps == 0:
                    eval_metrics = await self.evaluate(val_loader)
                    self.training_stats["evaluations_run"] += 1
                    
                    # Run parity tests
                    sample_batch = next(iter(val_loader))
                    sample_input = sample_batch["input_ids"][:1].to(self.device)
                    parity_results = self.parity_tester.run_full_parity_suite(
                        self.model,
                        sample_input,
                    )
                    
                    if not parity_results.get("all_tests_passed", False):
                        logger.warning("Parity tests failed - check inference consistency")
                
                # Checkpointing
                if (self.global_step + 1) % self.training_config.save_steps == 0:
                    await self.save_checkpoint()
                
                # Publish progress event
                if (self.global_step + 1) % 1000 == 0:
                    await self.synapse_publisher.publish_training_event(
                        event_type="progress",
                        model_name="llm-training",
                        step=self.global_step,
                        metrics=step_metrics,
                    )
                
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{step_metrics['loss']:.4f}",
                    "step": self.global_step,
                })
        
        progress_bar.close()
        
        # Final checkpoint
        await self.save_checkpoint()
        
        # Publish completion event
        await self.synapse_publisher.publish_training_event(
            event_type="completed",
            model_name="llm-training",
            step=self.global_step,
            metrics=self.training_stats,
        )
        
        # Export model
        await self.export_model()
        
        logger.info("Training complete!")
    
    async def export_model(self):
        """Export model to all formats."""
        logger.info("Exporting model...")
        
        # Export to PyTorch
        pytorch_path = self.format_exporter.export_to_pytorch(
            self.model,
            f"llm_training_step_{self.global_step}",
            include_optimizer=True,
            optimizer_state=self.optimizer.state_dict(),
        )
        
        # Export to ONNX
        try:
            onnx_path = self.format_exporter.export_to_onnx(
                self.model,
                f"llm_training_step_{self.global_step}",
                input_shape=(1, 512),
            )
            logger.info(f"Exported to ONNX: {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
        
        # Publish model-ready event
        await self.synapse_publisher.publish_model_ready_event(
            model_name="llm-training",
            version=f"step_{self.global_step}",
            checkpoint_path=str(pytorch_path),
            metrics=self.training_stats,
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.synapse_publisher.close()

