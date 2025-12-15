"""
Attention path optimizations.

Provides FlashAttention fallback logic, sliding-window attention,
and local + global attention mix for extended context.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FlashAttentionFallback:
    """
    FlashAttention with fallback to standard attention.
    
    Automatically falls back if FlashAttention is unavailable or fails.
    """
    
    def __init__(self, use_flash: bool = True):
        """
        Initialize FlashAttention fallback.
        
        Args:
            use_flash: Whether to attempt FlashAttention
        """
        self.use_flash = use_flash
        self.flash_available = self._check_flash_availability()
    
    def _check_flash_availability(self) -> bool:
        """Check if FlashAttention is available."""
        try:
            # Check if flash_attn is installed
            import flash_attn
            return True
        except ImportError:
            logger.warning("FlashAttention not available, using standard attention")
            return False
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with FlashAttention fallback.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output
        """
        if self.use_flash and self.flash_available:
            try:
                from flash_attn import flash_attn_func
                return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None)
            except Exception as e:
                logger.warning(f"FlashAttention failed, falling back: {e}")
                return self._standard_attention(q, k, v, mask)
        else:
            return self._standard_attention(q, k, v, mask)
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for efficient long sequences.
    
    Only attends to local window + global tokens.
    """
    
    def __init__(
        self,
        window_size: int = 512,
        num_global_tokens: int = 32,
    ):
        """
        Initialize sliding window attention.
        
        Args:
            window_size: Size of sliding window
            num_global_tokens: Number of global tokens to attend to
        """
        super().__init__()
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sliding window.
        
        Args:
            q: Query tensor [batch, heads, seq_len, dim]
            k: Key tensor
            v: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Split into windows
        outputs = []
        
        for i in range(0, seq_len, self.window_size):
            window_start = max(0, i - self.window_size // 2)
            window_end = min(seq_len, i + self.window_size // 2)
            
            # Local window
            q_window = q[:, :, window_start:window_end, :]
            k_window = k[:, :, window_start:window_end, :]
            v_window = v[:, :, window_start:window_end, :]
            
            # Global tokens (first N tokens)
            k_global = k[:, :, :self.num_global_tokens, :]
            v_global = v[:, :, :self.num_global_tokens, :]
            
            # Concatenate
            k_combined = torch.cat([k_global, k_window], dim=2)
            v_combined = torch.cat([v_global, v_window], dim=2)
            
            # Compute attention
            scores = torch.matmul(q_window, k_combined.transpose(-2, -1)) / (head_dim ** 0.5)
            
            if mask is not None:
                mask_window = mask[:, window_start:window_end]
                mask_combined = torch.cat([
                    torch.ones(batch_size, self.num_global_tokens, device=mask.device),
                    mask_window,
                ], dim=1)
                scores = scores.masked_fill(mask_combined.unsqueeze(1) == 0, float("-inf"))
            
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v_combined)
            
            outputs.append(output)
        
        return torch.cat(outputs, dim=2)

