"""
Fine-grained metrics collection system.

Collects token-level perplexity, attention entropy, activation norm histograms,
and learning rate vs loss curves for comprehensive training observability.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import wandb

logger = logging.getLogger(__name__)


class FineGrainedMetricsCollector:
    """
    Collects detailed training metrics beyond simple loss.
    
    Tracks:
    - Token-level perplexity
    - Attention entropy
    - Activation norm histograms
    - Learning rate vs loss curves
    - Gradient statistics
    """
    
    def __init__(
        self,
        use_wandb: bool = False,
        log_interval: int = 100,
        histogram_bins: int = 50,
    ):
        """
        Initialize metrics collector.
        
        Args:
            use_wandb: Enable Weights & Biases logging
            log_interval: Logging interval in steps
            histogram_bins: Number of bins for histograms
        """
        self.use_wandb = use_wandb
        self.log_interval = log_interval
        self.histogram_bins = histogram_bins
        
        # Metrics storage
        self.token_perplexities: deque = deque(maxlen=1000)
        self.attention_entropies: deque = deque(maxlen=1000)
        self.activation_norms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lr_loss_pairs: deque = deque(maxlen=10000)
        
        self.step = 0
    
    def compute_token_perplexity(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> Dict[str, float]:
        """
        Compute token-level perplexity.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            ignore_index: Index to ignore in loss computation
            
        Returns:
            Dictionary with perplexity metrics
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Compute per-token losses
        log_probs = F.log_softmax(flat_logits, dim=-1)
        token_losses = F.nll_loss(
            log_probs,
            flat_labels,
            reduction="none",
            ignore_index=ignore_index,
        )
        
        # Filter out ignored tokens
        valid_mask = flat_labels != ignore_index
        valid_losses = token_losses[valid_mask]
        
        if len(valid_losses) == 0:
            return {"perplexity": float("inf"), "mean_loss": float("inf")}
        
        # Compute perplexity
        mean_loss = valid_losses.mean().item()
        perplexity = np.exp(mean_loss)
        
        # Store
        self.token_perplexities.append(perplexity)
        
        return {
            "perplexity": perplexity,
            "mean_loss": mean_loss,
            "std_loss": valid_losses.std().item(),
            "min_loss": valid_losses.min().item(),
            "max_loss": valid_losses.max().item(),
        }
    
    def compute_attention_entropy(
        self,
        attention_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute attention entropy.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            
        Returns:
            Dictionary with entropy metrics
        """
        # Normalize to probabilities
        probs = F.softmax(attention_weights, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # Average over heads and sequence
        mean_entropy = entropy.mean().item()
        std_entropy = entropy.std().item()
        
        # Store
        self.attention_entropies.append(mean_entropy)
        
        return {
            "mean_entropy": mean_entropy,
            "std_entropy": std_entropy,
            "min_entropy": entropy.min().item(),
            "max_entropy": entropy.max().item(),
        }
    
    def track_activation_norms(
        self,
        activations: Dict[str, torch.Tensor],
    ):
        """
        Track activation norms for each layer.
        
        Args:
            activations: Dictionary of layer names to activation tensors
        """
        for layer_name, activation in activations.items():
            if torch.is_tensor(activation):
                # Compute L2 norm
                norm = activation.norm(2).item()
                self.activation_norms[layer_name].append(norm)
    
    def track_lr_loss_pair(
        self,
        learning_rate: float,
        loss: float,
    ):
        """Track learning rate vs loss pairs."""
        self.lr_loss_pairs.append((learning_rate, loss))
    
    def log_metrics(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log all metrics.
        
        Args:
            step: Current training step
            loss: Current loss
            learning_rate: Current learning rate
            additional_metrics: Additional metrics to log
        """
        self.step = step
        
        if step % self.log_interval != 0:
            return
        
        # Track LR vs loss
        self.track_lr_loss_pair(learning_rate, loss)
        
        # Build metrics dictionary
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
        }
        
        # Token perplexity
        if self.token_perplexities:
            recent_ppl = list(self.token_perplexities)[-100:]
            metrics["metrics/mean_perplexity"] = np.mean(recent_ppl)
            metrics["metrics/std_perplexity"] = np.std(recent_ppl)
        
        # Attention entropy
        if self.attention_entropies:
            recent_entropy = list(self.attention_entropies)[-100:]
            metrics["metrics/mean_attention_entropy"] = np.mean(recent_entropy)
            metrics["metrics/std_attention_entropy"] = np.std(recent_entropy)
        
        # Activation norms (histograms)
        for layer_name, norms in self.activation_norms.items():
            if norms:
                recent_norms = list(norms)[-100:]
                metrics[f"activations/{layer_name}_norm"] = np.mean(recent_norms)
                
                if self.use_wandb:
                    wandb.log({
                        f"activations/{layer_name}_histogram": wandb.Histogram(recent_norms),
                    }, step=step)
        
        # Add additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # Log to console
        logger.info(
            f"Step {step}: loss={loss:.4f}, lr={learning_rate:.2e}, "
            f"ppl={metrics.get('metrics/mean_perplexity', 0):.2f}"
        )
    
    def get_lr_loss_curve(self) -> tuple:
        """Get learning rate vs loss curve data."""
        if not self.lr_loss_pairs:
            return ([], [])
        
        lrs, losses = zip(*self.lr_loss_pairs)
        return (list(lrs), list(losses))

