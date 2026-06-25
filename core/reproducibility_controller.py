"""Re-export from deepiri-training-orchestrator (compat shim for Helox imports)."""
from deepiri_training_orchestrator import (
    ReproducibilityController,
    initialize_deterministic_training,
)

__all__ = ["ReproducibilityController", "initialize_deterministic_training"]
