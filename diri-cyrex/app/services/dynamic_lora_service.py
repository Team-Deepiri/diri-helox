"""
Dynamic LoRA Adapter Service
Manages per-user LoRA adapters for personalization
"""
import os
import json
from typing import Dict, Optional, List
from pathlib import Path
import torch
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..logging_config import get_logger

logger = get_logger("cyrex.dynamic_lora")


class DynamicLoRAService:
    """
    Manages dynamic LoRA adapters per user for personalized model fine-tuning.
    """
    
    def __init__(self, base_model_path: Optional[str] = None):
        self.base_model_path = base_model_path or os.getenv("BASE_MODEL_PATH", "microsoft/DeBERTa-v3-base")
        self.base_model = None
        self.base_tokenizer = None
        self.user_adapters = {}  # userId -> adapter_path
        self.adapter_configs = {}
        self.adapter_dir = Path("adapters")
        self.adapter_dir.mkdir(exist_ok=True)
        
        self._load_base_model()
    
    def _load_base_model(self):
        """Load base model and tokenizer."""
        try:
            logger.info("Loading base model", path=self.base_model_path)
            self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info("Base model loaded")
        except Exception as e:
            logger.warning("Base model loading failed, using placeholder", error=str(e))
            self.base_model = None
            self.base_tokenizer = None
    
    async def get_user_model(self, userId: str) -> Optional[PeftModel]:
        """
        Get personalized model for user with their LoRA adapter loaded.
        
        Args:
            userId: User ID
            
        Returns:
            PeftModel with user's adapter, or base model if no adapter
        """
        try:
            if not self.base_model:
                logger.warning("Base model not loaded")
                return None
            
            adapter_path = self.user_adapters.get(userId)
            
            if adapter_path and os.path.exists(adapter_path):
                # Load user's adapter
                model = PeftModel.from_pretrained(
                    self.base_model,
                    adapter_path,
                    adapter_name=userId
                )
                logger.debug("User adapter loaded", userId=userId)
                return model
            else:
                # Return base model
                return self.base_model
        except Exception as e:
            logger.error("Error getting user model", userId=userId, error=str(e))
            return self.base_model
    
    async def create_user_adapter(
        self,
        userId: str,
        training_data: List[Dict],
        config: Optional[Dict] = None
    ) -> str:
        """
        Create and train LoRA adapter for user.
        
        Args:
            userId: User ID
            training_data: List of training examples
            config: LoRA configuration
            
        Returns:
            Path to saved adapter
        """
        try:
            if not self.base_model:
                raise ValueError("Base model not loaded")
            
            # Default LoRA config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.get('r', 8) if config else 8,
                lora_alpha=config.get('lora_alpha', 32) if config else 32,
                lora_dropout=config.get('lora_dropout', 0.1) if config else 0.1,
                target_modules=config.get('target_modules', ['query', 'value']) if config else ['query', 'value']
            )
            
            # Create PEFT model
            model = get_peft_model(self.base_model, lora_config)
            
            # Train adapter (simplified - would need full training loop)
            # For now, just save the initialized adapter
            adapter_path = self.adapter_dir / f"user_{userId}"
            adapter_path.mkdir(exist_ok=True)
            
            model.save_pretrained(str(adapter_path))
            
            # Save config
            config_path = adapter_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'userId': userId,
                    'lora_config': lora_config.__dict__,
                    'training_samples': len(training_data)
                }, f, indent=2)
            
            self.user_adapters[userId] = str(adapter_path)
            self.adapter_configs[userId] = lora_config
            
            logger.info("User adapter created", userId=userId, path=str(adapter_path))
            return str(adapter_path)
            
        except Exception as e:
            logger.error("Error creating user adapter", userId=userId, error=str(e))
            raise
    
    async def update_user_adapter(
        self,
        userId: str,
        new_training_data: List[Dict],
        incremental: bool = True
    ):
        """
        Update user's adapter with new training data.
        
        Args:
            userId: User ID
            new_training_data: New training examples
            incremental: Whether to do incremental learning
        """
        try:
            adapter_path = self.user_adapters.get(userId)
            if not adapter_path or not os.path.exists(adapter_path):
                # Create new adapter
                await self.create_user_adapter(userId, new_training_data)
                return
            
            # Load existing adapter
            model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path
            )
            
            # Fine-tune with new data (simplified)
            # In production, would run full training loop
            
            # Save updated adapter
            model.save_pretrained(adapter_path)
            
            logger.info("User adapter updated", userId=userId, new_samples=len(new_training_data))
            
        except Exception as e:
            logger.error("Error updating user adapter", userId=userId, error=str(e))
            raise
    
    async def delete_user_adapter(self, userId: str):
        """Delete user's adapter."""
        try:
            adapter_path = self.user_adapters.get(userId)
            if adapter_path and os.path.exists(adapter_path):
                import shutil
                shutil.rmtree(adapter_path)
                del self.user_adapters[userId]
                if userId in self.adapter_configs:
                    del self.adapter_configs[userId]
                logger.info("User adapter deleted", userId=userId)
        except Exception as e:
            logger.error("Error deleting user adapter", userId=userId, error=str(e))
    
    async def list_user_adapters(self) -> List[str]:
        """List all users with adapters."""
        return list(self.user_adapters.keys())
    
    async def get_adapter_info(self, userId: str) -> Optional[Dict]:
        """Get information about user's adapter."""
        adapter_path = self.user_adapters.get(userId)
        if not adapter_path or not os.path.exists(adapter_path):
            return None
        
        config_path = Path(adapter_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None


# Singleton instance
_lora_service = None

def get_dynamic_lora_service() -> DynamicLoRAService:
    """Get singleton DynamicLoRAService instance."""
    global _lora_service
    if _lora_service is None:
        _lora_service = DynamicLoRAService()
    return _lora_service


