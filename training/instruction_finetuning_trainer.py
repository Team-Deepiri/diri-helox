"""
Instruction fine-tuning trainer.

Fine-tunes pretrained models for instruction following and chat capabilities.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..core.device_manager import DeviceManager
from ..core.training_config import TrainingConfig, ModelConfig
from ..models.transformer_lm import TransformerLanguageModel
from .pretraining_trainer import PretrainingTrainer

logger = logging.getLogger(__name__)


class InstructionFinetuningTrainer(PretrainingTrainer):
    """
    Trainer for instruction fine-tuning.
    
    Extends pretraining trainer with instruction-specific loss computation.
    Only computes loss on response tokens, not instruction tokens.
    """
    
    def __init__(
        self,
        model: TransformerLanguageModel,
        config: TrainingConfig,
        model_config: ModelConfig,
        device_manager: DeviceManager,
    ):
        """Initialize instruction fine-tuning trainer."""
        super().__init__(model, config, model_config, device_manager)
    
    def _training_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training step with instruction masking.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask
            instruction_mask: Mask indicating instruction tokens (0) vs response tokens (1)
            
        Returns:
            Loss tensor
        """
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                )
                
                # Compute loss only on response tokens
                loss = self._compute_instruction_loss(
                    outputs["logits"],
                    input_ids,
                    instruction_mask,
                )
                loss = loss / self.config.gradient_accumulation_steps
            
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
            
            loss = self._compute_instruction_loss(
                outputs["logits"],
                input_ids,
                instruction_mask,
            )
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        return loss
    
    def _compute_instruction_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        instruction_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss only on response tokens.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            instruction_mask: Mask for instruction (0) vs response (1) tokens
            
        Returns:
            Loss tensor
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Create loss mask (only compute loss on response tokens)
        if instruction_mask is not None:
            # Shift mask to align with predictions
            flat_mask = instruction_mask[..., 1:].contiguous().view(-1)
            # Only compute loss where mask is 1 (response tokens)
            flat_mask = flat_mask.bool()
        else:
            # If no mask provided, compute loss on all tokens
            flat_mask = torch.ones_like(flat_labels, dtype=torch.bool)
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(flat_logits, flat_labels)
        
        # Apply mask
        masked_losses = losses * flat_mask.float()
        
        # Average over non-masked tokens
        if flat_mask.sum() > 0:
            loss = masked_losses.sum() / flat_mask.sum()
        else:
            loss = masked_losses.mean()
        
        return loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from_checkpoint: Optional[Path] = None,
    ):
        """
        Training loop for instruction fine-tuning.
        
        Expects batches with 'input_ids', 'attention_mask', and 'instruction_mask'.
        """
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        self.model.train()
        accumulated_loss = 0.0
        
        progress_bar = tqdm(
            total=self.config.total_steps,
            desc="Instruction Fine-tuning",
            initial=self.global_step,
        )
        
        while self.global_step < self.config.total_steps:
            for batch in train_loader:
                if self.global_step >= self.config.total_steps:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                instruction_mask = batch.get("instruction_mask")
                if instruction_mask is not None:
                    instruction_mask = instruction_mask.to(self.device)
                
                loss = self._training_step(input_ids, attention_mask, instruction_mask)
                accumulated_loss += loss.item()
                
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
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
                
                if val_loader and (self.global_step + 1) % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(val_loader)
                    self.model.train()
                    
                    if self.config.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)
                
                if (self.global_step + 1) % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": self.global_step,
                })
        
        progress_bar.close()
        self.save_checkpoint()
        logger.info("Instruction fine-tuning complete!")

