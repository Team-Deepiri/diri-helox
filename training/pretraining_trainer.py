"""
Pretraining trainer for language models.

Implements the core pretraining loop with checkpointing, evaluation, and logging.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, get_constant_schedule_with_warmup
from tqdm import tqdm
import wandb

from ..core.device_manager import DeviceManager
from ..core.training_config import TrainingConfig, ModelConfig
from ..models.transformer_lm import TransformerLanguageModel

logger = logging.getLogger(__name__)


class PretrainingTrainer:
    """
    Trainer for pretraining language models.
    
    Handles:
    - Training loop
    - Checkpointing
    - Evaluation
    - Logging
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: TransformerLanguageModel,
        config: TrainingConfig,
        model_config: ModelConfig,
        device_manager: DeviceManager,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            model_config: Model configuration
            device_manager: Device manager
        """
        self.model = model
        self.config = config
        self.model_config = model_config
        self.device_manager = device_manager
        
        self.device = device_manager.get_device()
        self.model.to(self.device)
        
        # Mixed precision
        self.use_amp = config.mixed_precision and device_manager.is_gpu_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Gradient checkpointing
        self.use_gradient_checkpointing = config.gradient_checkpointing
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Checkpointing
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project or "llm-pretraining",
                name=config.wandb_run_name,
                config={
                    **config.to_dict(),
                    **model_config.to_dict(),
                },
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer_type == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_steps,
                eta_min=self.config.min_learning_rate,
            )
        elif self.config.scheduler_type == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_learning_rate / self.config.learning_rate,
                total_iters=self.config.total_steps,
            )
        elif self.config.scheduler_type == "constant":
            return get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from_checkpoint: Optional[Path] = None,
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            resume_from_checkpoint: Optional checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Training loop
        self.model.train()
        accumulated_loss = 0.0
        
        progress_bar = tqdm(
            total=self.config.total_steps,
            desc="Training",
            initial=self.global_step,
        )
        
        while self.global_step < self.config.total_steps:
            for batch in train_loader:
                if self.global_step >= self.config.total_steps:
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                loss = self._training_step(input_ids, attention_mask)
                
                # Accumulate loss
                accumulated_loss += loss.item()
                
                # Backward pass (with gradient accumulation)
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if (self.global_step + 1) % self.config.logging_steps == 0:
                    avg_loss = accumulated_loss / self.config.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/step": self.global_step,
                    }
                    
                    if self.config.use_wandb:
                        wandb.log(log_dict, step=self.global_step)
                    
                    logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                    )
                    
                    accumulated_loss = 0.0
                
                # Evaluation
                if val_loader and (self.global_step + 1) % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(val_loader)
                    self.model.train()
                    
                    if self.config.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)
                
                # Checkpointing
                if (self.global_step + 1) % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": self.global_step,
                })
        
        progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint()
        
        logger.info("Training complete!")
    
    def _training_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Single training step."""
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        return loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids,
                            use_gradient_checkpointing=False,  # No checkpointing during eval
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                        use_gradient_checkpointing=False,  # No checkpointing during eval
                    )
                
                loss = outputs["loss"]
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        metrics = {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }
        
        logger.info(f"Evaluation: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")
        
        return metrics
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = self.config.checkpoint_dir / f"step_{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer and scheduler
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        scheduler_path = checkpoint_dir / "scheduler.pt"
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save scaler if using AMP
        if self.scaler:
            scaler_path = checkpoint_dir / "scaler.pt"
            torch.save(self.scaler.state_dict(), scaler_path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "model_config": self.model_config.to_dict(),
        }
        
        import json
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        
        # Load scheduler
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path))
        
        # Load scaler
        if self.scaler:
            scaler_path = checkpoint_path / "scaler.pt"
            if scaler_path.exists():
                self.scaler.load_state_dict(torch.load(scaler_path))
        
        # Load training state
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            import json
            with open(state_path, "r") as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.current_epoch = state.get("current_epoch", 0)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted(
            self.config.checkpoint_dir.glob("step_*"),
            key=lambda x: int(x.name.split("_")[1]),
        )
        
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")

