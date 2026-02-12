#!/usr/bin/env python3
"""
Setup script for LLM training environment.

Creates necessary directories and validates the environment.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.device_manager import DeviceManager
from core.training_config import DataConfig, ModelConfig, TrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories."""
    dirs = [
        "data/datasets/raw",
        "data/datasets/processed",
        "data/datasets/tokenized",
        "models/checkpoints",
        "models/exports",
        "tokenizers",
        "logs",
        "configs",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def validate_environment():
    """Validate the training environment."""
    logger.info("Validating environment...")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("⚠ CUDA not available (will use CPU)")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("✓ MPS (Apple Silicon) available")
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False
    
    # Check required packages
    required_packages = [
        "transformers",
        "datasets",
        "sentencepiece",
        "einops",
        "accelerate",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ {package} not installed")
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install with: poetry install")
        return False
    
    return True


def test_device_detection():
    """Test device detection."""
    logger.info("Testing device detection...")
    
    device_manager = DeviceManager()
    device_info = device_manager.get_device_info()
    
    logger.info(f"Detected device: {device_info['device']}")
    logger.info(f"Device type: {device_info['device_type']}")
    
    if device_info.get("device_name"):
        logger.info(f"GPU: {device_info['device_name']}")
        if device_info.get("memory_total_gb"):
            logger.info(f"GPU Memory: {device_info['memory_total_gb']:.2f} GB")
    
    return True


def create_default_configs():
    """Create default configuration files if they don't exist."""
    config_dir = Path("configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Model config
    model_config_path = config_dir / "model_config.json"
    if not model_config_path.exists():
        model_config = ModelConfig()
        model_config.save(model_config_path)
        logger.info(f"Created default model config: {model_config_path}")
    
    # Data config
    data_config_path = config_dir / "data_config.json"
    if not data_config_path.exists():
        data_config = DataConfig()
        data_config.save(data_config_path)
        logger.info(f"Created default data config: {data_config_path}")
    
    # Training config
    training_config_path = config_dir / "training_config.json"
    if not training_config_path.exists():
        training_config = TrainingConfig()
        training_config.save(training_config_path)
        logger.info(f"Created default training config: {training_config_path}")


def main():
    """Main setup function."""
    logger.info("Setting up LLM training environment...")
    
    # Setup directories
    setup_directories()
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Please install missing packages.")
        sys.exit(1)
    
    # Test device detection
    test_device_detection()
    
    # Create default configs
    create_default_configs()
    
    logger.info("✓ Setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Add your training data to data/datasets/raw/")
    logger.info("2. Review and adjust configs in configs/")
    logger.info("3. Run: python scripts/train_llm_from_scratch.py")


if __name__ == "__main__":
    main()

