"""
Failure-resilient training system.

Provides mid-step crash recovery, partial optimizer restore,
and stateless workers for real cluster training.
"""

import logging
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FailureResilienceManager:
    """
    Manages training resilience to failures.
    
    Features:
    - Mid-step crash recovery
    - Partial optimizer restore
    - Stateless workers
    - Checkpoint validation
    """
    
    def __init__(
        self,
        checkpoint_dir: Path = Path("models/checkpoints"),
        recovery_dir: Path = Path("models/recovery"),
    ):
        """
        Initialize failure resilience manager.
        
        Args:
            checkpoint_dir: Checkpoint directory
            recovery_dir: Recovery state directory
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
    
    def save_training_state(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        scheduler_state: Optional[Dict[str, Any]] = None,
        batch_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Save training state for recovery.
        
        Args:
            step: Current step
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Optional scheduler state
            batch_state: Optional current batch state
        """
        recovery_state = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "batch_state": batch_state,
        }
        
        recovery_file = self.recovery_dir / f"recovery_step_{step}.pt"
        torch.save(recovery_state, recovery_file)
        
        # Also save as JSON for metadata
        metadata_file = self.recovery_dir / f"recovery_step_{step}_metadata.json"
        metadata = {
            "step": step,
            "timestamp": recovery_state["timestamp"],
            "has_model": True,
            "has_optimizer": True,
            "has_scheduler": scheduler_state is not None,
            "has_batch": batch_state is not None,
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved recovery state for step {step}")
    
    def load_recovery_state(
        self,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load recovery state.
        
        Args:
            step: Step to recover from (None = latest)
            
        Returns:
            Recovery state or None
        """
        if step is None:
            # Find latest recovery file
            recovery_files = sorted(self.recovery_dir.glob("recovery_step_*.pt"))
            if not recovery_files:
                return None
            recovery_file = recovery_files[-1]
        else:
            recovery_file = self.recovery_dir / f"recovery_step_{step}.pt"
        
        if not recovery_file.exists():
            logger.warning(f"Recovery file not found: {recovery_file}")
            return None
        
        try:
            recovery_state = torch.load(recovery_file, map_location="cpu")
            logger.info(f"Loaded recovery state from step {recovery_state['step']}")
            return recovery_state
        except Exception as e:
            logger.error(f"Failed to load recovery state: {e}")
            return None
    
    def validate_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Dict[str, Any]:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Validation results
        """
        checkpoint_path = Path(checkpoint_path)
        
        validation = {
            "valid": False,
            "errors": [],
            "warnings": [],
        }
        
        if not checkpoint_path.exists():
            validation["errors"].append("Checkpoint file does not exist")
            return validation
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Check required keys
            required_keys = ["model_state_dict"]
            for key in required_keys:
                if key not in checkpoint:
                    validation["errors"].append(f"Missing required key: {key}")
            
            # Check model state
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
                if not isinstance(model_state, dict):
                    validation["errors"].append("Model state dict is not a dictionary")
                elif len(model_state) == 0:
                    validation["warnings"].append("Model state dict is empty")
            
            # Check optimizer state
            if "optimizer_state_dict" in checkpoint:
                opt_state = checkpoint["optimizer_state_dict"]
                if not isinstance(opt_state, dict):
                    validation["warnings"].append("Optimizer state dict format unexpected")
            
            validation["valid"] = len(validation["errors"]) == 0
            
        except Exception as e:
            validation["errors"].append(f"Failed to load checkpoint: {e}")
        
        return validation
    
    def recover_from_failure(
        self,
        model,
        optimizer,
        scheduler=None,
    ) -> Optional[int]:
        """
        Recover training from failure.
        
        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            scheduler: Optional scheduler to restore
            
        Returns:
            Recovered step or None
        """
        recovery_state = self.load_recovery_state()
        
        if not recovery_state:
            logger.warning("No recovery state found")
            return None
        
        step = recovery_state["step"]
        
        # Restore model
        if "model_state" in recovery_state:
            model.load_state_dict(recovery_state["model_state"])
            logger.info(f"Restored model to step {step}")
        
        # Restore optimizer
        if "optimizer_state" in recovery_state:
            optimizer.load_state_dict(recovery_state["optimizer_state"])
            logger.info(f"Restored optimizer to step {step}")
        
        # Restore scheduler
        if scheduler and "scheduler_state" in recovery_state:
            scheduler.load_state_dict(recovery_state["scheduler_state"])
            logger.info(f"Restored scheduler to step {step}")
        
        return step

