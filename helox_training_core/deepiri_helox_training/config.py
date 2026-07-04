"""Training configuration contract.

``BaseTrainingConfig`` collects the hyperparameters every Deepiri training loop needs.
Services subclass it to add framework- or model-specific fields without redefining the
shared concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict


@dataclass
class BaseTrainingConfig:
    """Common, framework-agnostic training hyperparameters."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    eval_interval: int = 1
    checkpoint_interval: int = 5
    keep_last_checkpoints: int = 3
    early_stopping_patience: int = 0  # 0 disables early stopping

    # Name of the validation metric used for "best"/early-stopping decisions and whether
    # lower values are better (e.g. loss) or higher (e.g. accuracy).
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    def __post_init__(self) -> None:
        if self.monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {self.monitor_mode!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all dataclass fields (including subclass fields)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
