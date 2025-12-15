"""
Deterministic training and reproducibility controller.

Provides global seed control, deterministic dataloader ordering,
and training run fingerprinting for complete reproducibility.
"""

import random
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)


class ReproducibilityController:
    """
    Controls all sources of randomness for deterministic training.
    
    Ensures:
    - Python random seed
    - NumPy random seed
    - PyTorch random seed
    - CUDA deterministic operations
    - Dataloader worker seeding
    """
    
    def __init__(
        self,
        seed: int = 1337,
        deterministic: bool = True,
        benchmark: bool = False,
    ):
        """
        Initialize reproducibility controller.
        
        Args:
            seed: Global seed value
            deterministic: Enable deterministic CUDA operations
            benchmark: Disable for reproducibility (True = faster but non-deterministic)
        """
        self.seed = seed
        self.deterministic = deterministic
        self.benchmark = benchmark
        self.fingerprint: Optional[str] = None
    
    def set_seeds(self):
        """Set all random seeds."""
        # Python random
        random.seed(self.seed)
        
        # NumPy
        np.random.seed(self.seed)
        
        # PyTorch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # CUDA deterministic operations
        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            cudnn.deterministic = True
            cudnn.benchmark = self.benchmark
        
        logger.info(f"Seeds set to {self.seed} (deterministic={self.deterministic})")
    
    def get_dataloader_worker_init_fn(self):
        """
        Get worker initialization function for DataLoader.
        
        Ensures each worker process has a unique but deterministic seed.
        """
        def worker_init_fn(worker_id: int):
            worker_seed = self.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        return worker_init_fn
    
    def generate_training_fingerprint(
        self,
        config: Dict[str, Any],
        code_hash: Optional[str] = None,
    ) -> str:
        """
        Generate training run fingerprint.
        
        Creates a hash of config + code to uniquely identify training runs.
        
        Args:
            config: Training configuration dictionary
            code_hash: Optional hash of training code
            
        Returns:
            Fingerprint string
        """
        # Create fingerprint from config
        config_str = json.dumps(config, sort_keys=True)
        fingerprint_data = {
            "seed": self.seed,
            "config": config_str,
            "code_hash": code_hash or "unknown",
        }
        
        fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint = hashlib.sha256(fingerprint_json.encode()).hexdigest()[:16]
        
        self.fingerprint = fingerprint
        logger.info(f"Training fingerprint: {fingerprint}")
        
        return fingerprint
    
    def save_fingerprint(self, output_path: Path):
        """Save fingerprint to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fingerprint_data = {
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "deterministic": self.deterministic,
        }
        
        with open(output_path, "w") as f:
            json.dump(fingerprint_data, f, indent=2)
        
        logger.info(f"Fingerprint saved to {output_path}")
    
    def verify_reproducibility(
        self,
        checkpoint_path: Path,
        expected_fingerprint: Optional[str] = None,
    ) -> bool:
        """
        Verify checkpoint matches expected fingerprint.
        
        Args:
            checkpoint_path: Path to checkpoint
            expected_fingerprint: Expected fingerprint (if None, loads from checkpoint)
            
        Returns:
            True if fingerprint matches
        """
        fingerprint_file = checkpoint_path / "training_fingerprint.json"
        
        if not fingerprint_file.exists():
            logger.warning(f"Fingerprint file not found: {fingerprint_file}")
            return False
        
        with open(fingerprint_file, "r") as f:
            saved_data = json.load(f)
        
        saved_fingerprint = saved_data.get("fingerprint")
        
        if expected_fingerprint:
            matches = saved_fingerprint == expected_fingerprint
            if not matches:
                logger.error(
                    f"Fingerprint mismatch: expected {expected_fingerprint}, "
                    f"got {saved_fingerprint}"
                )
            return matches
        
        return True


def initialize_deterministic_training(
    seed: int = 1337,
    deterministic: bool = True,
) -> ReproducibilityController:
    """
    Convenience function to initialize deterministic training.
    
    Args:
        seed: Global seed
        deterministic: Enable deterministic operations
        
    Returns:
        ReproducibilityController instance
    """
    controller = ReproducibilityController(seed=seed, deterministic=deterministic)
    controller.set_seeds()
    return controller

