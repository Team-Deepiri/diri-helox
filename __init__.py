"""
Helox - Production LLM Training System

Complete training infrastructure with 38 production-grade features.
"""

__version__ = "1.0.0"

# Core exports — guarded so the file can be imported as a standalone module
# (e.g. by pytest) without breaking when relative imports aren't available.
try:
    from .core.device_manager import DeviceManager
    from .core.training_config import TrainingConfig, ModelConfig, DataConfig
    from .core.reproducibility_controller import ReproducibilityController
    from .training.unified_training_orchestrator import UnifiedTrainingOrchestrator

    __all__ = [
        "DeviceManager",
        "TrainingConfig",
        "ModelConfig",
        "DataConfig",
        "ReproducibilityController",
        "UnifiedTrainingOrchestrator",
    ]
except ImportError:
    __all__ = []

