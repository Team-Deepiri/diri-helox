# diri-helox/mlops/infrastructure/model_adaptation_layers.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

from torch.nn import ModuleDict
from peft import PeftModel, LoraConfig, get_peft_model  # type: ignore

try:
    from deepiri_modelkit.logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class LayerType(str, Enum):
    DOMAIN = "domain"
    TASK = "task"
    USER = "user"
    TENANT = "tenant"


@dataclass(frozen=True)
class LayerConfig:
    layer_type: LayerType
    name: str  # e.g. "b2b_productivity", "classification", "user_123"
    rank: int
    alpha: int
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None

    def adapter_name(self) -> str:
        # Avoid ":" because PEFT parameter names / matching can behave unexpectedly
        return f"{self.layer_type.value}__{self.name}"


class LayeredModelAdapter:
    """
    Minimal v0: load/attach multiple adapters, activate one, freeze all but one, and save active adapter.
    """

    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.model: Any = base_model
        self.loaded_adapters: List[str] = []
        self.active_adapter: Optional[str] = None

        # Maps our external adapter name -> PEFT internal adapter key (for the first adapter created by get_peft_model)
        self._adapter_key_map: dict[str, str] = {}

    def _resolve_adapter_key(self, adapter_name: str) -> str:
        return self._adapter_key_map.get(adapter_name, adapter_name)

    def load_adapter(self, adapter_path: Union[str, Path], adapter_name: str) -> None:
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        logger.info(f"Loading adapter: {adapter_name} from {adapter_path}")

        if isinstance(self.model, PeftModel):
            self.model.load_adapter(str(adapter_path), adapter_name=adapter_name)
        else:
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), adapter_name=adapter_name)

        if adapter_name not in self.loaded_adapters:
            self.loaded_adapters.append(adapter_name)

        # For loaded adapters, PEFT key == name
        self._adapter_key_map[adapter_name] = adapter_name

    def attach_new_lora_layer(self, layer_cfg: LayerConfig, task_type: str = "CAUSAL_LM") -> str:
        if layer_cfg.target_modules is None:
            raise ValueError("LayerConfig.target_modules must be provided (model-specific).")

        adapter_name = layer_cfg.adapter_name()
        lora_cfg = LoraConfig(
            r=layer_cfg.rank,
            lora_alpha=layer_cfg.alpha,
            lora_dropout=layer_cfg.dropout,
            bias=layer_cfg.bias,
            task_type=task_type,
            target_modules=layer_cfg.target_modules,
        )

        logger.info(f"Attaching new adapter: {adapter_name}")

        if isinstance(self.model, PeftModel):
            # Adding on top of an existing PeftModel is easy: key == adapter_name
            self.model.add_adapter(adapter_name, lora_cfg)
            self._adapter_key_map[adapter_name] = adapter_name
        else:
            # First adapter wraps base model, but PEFT picks an internal name (often "default")
            self.model = get_peft_model(self.model, lora_cfg)

            internal_key = next(iter(getattr(self.model, "peft_config", {}).keys()), None)
            if internal_key is None:
                raise RuntimeError("PEFT did not create any adapter config on the model.")

            # We'll keep PEFT's internal key, but allow the rest of the system to refer to adapter_name.
            # This avoids fragile renames across many internal dicts in different PEFT versions.
            self._adapter_key_map[adapter_name] = internal_key

            # Still record it as "loaded" under our external name.
            # (PEFT internal key stays as-is)
            logger.info(f"First adapter internal PEFT key: {internal_key} (external name: {adapter_name})")

        if adapter_name not in self.loaded_adapters:
            self.loaded_adapters.append(adapter_name)

        # Activate the adapter right away
        self.set_active_adapter(adapter_name)
        return adapter_name

    def set_active_adapter(self, adapter_name: str) -> None:
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model is not a PeftModel yet (no adapters loaded/attached).")
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"Adapter not loaded: {adapter_name}. Loaded: {self.loaded_adapters}")

        key = self._resolve_adapter_key(adapter_name)
        self.model.set_adapter(key)
        self.active_adapter = adapter_name

        logger.info(f"Active adapter set: {adapter_name} (peft_key={key})")

    def freeze_all_adapters_except(self, adapter_name: str) -> None:
        """
        Freeze all parameters, then unfreeze ONLY the LoRA weights for the requested adapter.
        This is the key isolation rule for layered training.
        """
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model is not a PeftModel yet (no adapters loaded/attached).")
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"Adapter not loaded: {adapter_name}. Loaded: {self.loaded_adapters}")

        key = self._resolve_adapter_key(adapter_name)

        # 1) Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # 2) Unfreeze LoRA weights for ONLY this adapter (use PEFT internal key)
        unfrozen = 0

        def _unfreeze_moduledict(md: ModuleDict, adapter_key: str) -> None:
            nonlocal unfrozen
            if adapter_key in md:
                for p in md[adapter_key].parameters():
                    p.requires_grad = True
                    unfrozen += p.numel()

        for module in self.model.modules():
            for attr in (
                "lora_A",
                "lora_B",
                "lora_dropout",
                "lora_embedding_A",
                "lora_embedding_B",
                "lora_magnitude_vector",
            ):
                md = getattr(module, attr, None)
                if isinstance(md, ModuleDict):
                    _unfreeze_moduledict(md, key)

        # 3) Activate adapter
        self.set_active_adapter(adapter_name)

        logger.info(f"Freeze complete. Unfrozen params for {adapter_name}: {unfrozen}")

    def save_active_adapter(self, output_dir: Union[str, Path]) -> Path:
        """
        Save only the currently active adapter to output_dir.
        Returns the Path to the saved adapter directory.
        """
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model is not a PeftModel yet (no adapters loaded/attached).")
        if not self.active_adapter:
            raise RuntimeError("No active adapter set. Call set_active_adapter() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter_name = self.active_adapter
        key = self._resolve_adapter_key(adapter_name)

        logger.info(f"Saving active adapter: {adapter_name} (peft_key={key}) to {output_dir}")

        # IMPORTANT: save_pretrained expects PEFT internal key, not our external label
        self.model.save_pretrained(str(output_dir), adapter_name=key)
        return output_dir
    
