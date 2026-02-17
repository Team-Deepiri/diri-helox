# diri-helox/mlops/infrastructure/model_adaptation_layers.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

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
    """
    Configuration for a single LoRA adapter layer.

    Adapter naming convention:
        <layer_type>__<name>

    Example:
        domain__b2b
        task__classification
        user__123
    """

    layer_type: LayerType
    name: str
    rank: int
    alpha: int
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None

    def adapter_name(self) -> str:
        # Avoid ":" because PEFT parameter names can behave unexpectedly
        return f"{self.layer_type.value}__{self.name}"


class LayeredModelAdapter:
    """
    Minimal v0 layered adapter manager.

    Supports:
    - loading adapters
    - attaching new LoRA adapters
    - activating adapters
    - freezing all except one adapter (training isolation)
    - saving active adapter only
    """

    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.model: Any = base_model
        self.loaded_adapters: List[str] = []
        self.active_adapter: Optional[str] = None

        # external adapter name -> internal PEFT key
        self._adapter_key_map: dict[str, str] = {}

    def _resolve_adapter_key(self, adapter_name: str) -> str:
        return self._adapter_key_map.get(adapter_name, adapter_name)

    # -------------------------
    # LOAD EXISTING ADAPTER
    # -------------------------
    def load_adapter(self, adapter_path: Union[str, Path], adapter_name: str) -> None:

        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        logger.info("Loading adapter: %s from %s", adapter_name, adapter_path)

        if isinstance(self.model, PeftModel):
            self.model.load_adapter(str(adapter_path), adapter_name=adapter_name)
        else:
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), adapter_name=adapter_name
            )

        if adapter_name not in self.loaded_adapters:
            self.loaded_adapters.append(adapter_name)

        self._adapter_key_map[adapter_name] = adapter_name

    # -------------------------
    # ATTACH NEW LORA
    # -------------------------
    def attach_new_lora_layer(
        self, layer_cfg: LayerConfig, task_type: str = "CAUSAL_LM"
    ) -> str:

        if layer_cfg.target_modules is None:
            raise ValueError("LayerConfig.target_modules must be provided.")

        adapter_name = layer_cfg.adapter_name()

        lora_cfg = LoraConfig(
            r=layer_cfg.rank,
            lora_alpha=layer_cfg.alpha,
            lora_dropout=layer_cfg.dropout,
            bias=layer_cfg.bias,
            task_type=task_type,
            target_modules=layer_cfg.target_modules,
        )

        logger.info("Attaching new adapter: %s", adapter_name)

        if isinstance(self.model, PeftModel):
            self.model.add_adapter(adapter_name, lora_cfg)
            self._adapter_key_map[adapter_name] = adapter_name
        else:
            self.model = get_peft_model(self.model, lora_cfg)

            internal_key = next(
                iter(getattr(self.model, "peft_config", {}).keys()), None
            )

            if internal_key is None:
                raise RuntimeError("PEFT did not create adapter config.")

            self._adapter_key_map[adapter_name] = internal_key

            logger.info(
                "First adapter internal PEFT key: %s (external name: %s)",
                internal_key,
                adapter_name,
            )

        if adapter_name not in self.loaded_adapters:
            self.loaded_adapters.append(adapter_name)

        self.set_active_adapter(adapter_name)
        return adapter_name

    # -------------------------
    # ACTIVATE ADAPTER
    # -------------------------
    def set_active_adapter(self, adapter_name: str) -> None:

        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model is not a PeftModel yet.")

        if adapter_name not in self.loaded_adapters:
            raise ValueError(
                f"Adapter not loaded: {adapter_name}. Loaded: {self.loaded_adapters}"
            )

        key = self._resolve_adapter_key(adapter_name)

        self.model.set_adapter(key)
        self.active_adapter = adapter_name

        logger.info("Active adapter set: %s (peft_key=%s)", adapter_name, key)

    # -------------------------
    # FREEZE ALL EXCEPT ONE
    # -------------------------
    def freeze_all_adapters_except(self, adapter_name: str) -> None:
        """
        Freeze all parameters, then unfreeze ONLY LoRA weights
        belonging to the requested adapter.

        Uses parameter-name matching instead of scanning modules.
        """

        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model is not a PeftModel yet.")

        if adapter_name not in self.loaded_adapters:
            raise ValueError(
                f"Adapter not loaded: {adapter_name}. Loaded: {self.loaded_adapters}"
            )

        key = self._resolve_adapter_key(adapter_name)

        # Freeze everything
        total_params = 0
        for p in self.model.parameters():
            total_params += p.numel()
            p.requires_grad = False

        # Unfreeze ONLY LoRA params for this adapter
        unfrozen = 0
        tensors = 0

        for name, p in self.model.named_parameters():

            if f".{key}." in name and "lora_" in name:
                p.requires_grad = True
                unfrozen += p.numel()
                tensors += 1

        self.set_active_adapter(adapter_name)

        logger.info(
            "Freeze complete. adapter=%s peft_key=%s unfrozen_params=%s tensors=%s total=%s",
            adapter_name,
            key,
            unfrozen,
            tensors,
            total_params,
        )

    # -------------------------
    # SAVE ACTIVE ADAPTER
    # -------------------------
    def save_active_adapter(self, output_dir: Union[str, Path]) -> Path:

        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model is not a PeftModel yet.")

        if not self.active_adapter:
            raise RuntimeError("No active adapter set.")

        adapter_name = self.active_adapter
        key = self._resolve_adapter_key(adapter_name)

        # validate adapter exists in PEFT config
        peft_cfg = getattr(self.model, "peft_config", {})
        if key not in peft_cfg:
            raise RuntimeError(
                f"Active adapter '{adapter_name}' resolved to '{key}', "
                f"but not found in model.peft_config keys={list(peft_cfg.keys())}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Saving active adapter: %s (peft_key=%s) to %s",
            adapter_name,
            key,
            output_dir,
        )

        self.model.save_pretrained(str(output_dir), adapter_name=key)

        return output_dir
