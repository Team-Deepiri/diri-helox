"""
Model export and format conversion system.

Supports ONNX, GGUF, TensorRT, and other deployment formats
for CPU, edge, mobile, and server deployments.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelFormatExporter:
    """
    Exports models to various deployment formats.
    
    Supports:
    - ONNX (CPU/edge)
    - GGUF (quantized)
    - TensorRT (NVIDIA)
    - PyTorch (standard)
    """
    
    def __init__(self, output_dir: Path = Path("models/exports")):
        """
        Initialize format exporter.
        
        Args:
            output_dir: Output directory for exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_onnx(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: tuple = (1, 512),
        dynamic_axes: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            model_name: Model name
            input_shape: Input shape (batch, seq_len)
            dynamic_axes: Dynamic axes for variable-length inputs
            
        Returns:
            Path to exported ONNX file
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            logger.error("ONNX not installed. Install with: pip install onnx onnxruntime")
            raise
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(0, 50000, input_shape, dtype=torch.long)
        
        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size", 1: "sequence_length"},
            }
        
        output_path = self.output_dir / f"{model_name}.onnx"
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
        
        logger.info(f"Exported model to ONNX: {output_path}")
        
        return output_path
    
    def export_to_gguf(
        self,
        model: nn.Module,
        model_name: str,
        quantization: str = "Q4_K_M",
    ) -> Path:
        """
        Export model to GGUF format (quantized).
        
        Args:
            model: Model to export
            model_name: Model name
            quantization: Quantization type
            
        Returns:
            Path to exported GGUF file
        """
        try:
            # GGUF export typically requires llama.cpp or similar
            # This is a placeholder for the integration
            logger.warning("GGUF export requires llama.cpp integration")
            
            output_path = self.output_dir / f"{model_name}.gguf"
            
            # For now, save as PyTorch and note that conversion needed
            torch.save(model.state_dict(), output_path.with_suffix(".pth"))
            
            logger.info(f"Model saved for GGUF conversion: {output_path}")
            logger.info("Use llama.cpp tools to convert to GGUF format")
            
            return output_path
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            raise
    
    def export_to_tensorrt(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: tuple = (1, 512),
    ) -> Path:
        """
        Export model to TensorRT format.
        
        Args:
            model: Model to export
            model_name: Model name
            input_shape: Input shape
            
        Returns:
            Path to exported TensorRT file
        """
        try:
            import tensorrt as trt
        except ImportError:
            logger.error("TensorRT not installed")
            raise
        
        # TensorRT export is complex and requires ONNX as intermediate
        # Export to ONNX first
        onnx_path = self.export_to_onnx(model, f"{model_name}_intermediate", input_shape)
        
        # Convert ONNX to TensorRT (simplified)
        output_path = self.output_dir / f"{model_name}.trt"
        
        logger.info(f"TensorRT export requires ONNX-to-TRT conversion: {onnx_path}")
        logger.info("Use TensorRT tools to convert ONNX to TensorRT")
        
        return output_path
    
    def export_to_pytorch(
        self,
        model: nn.Module,
        model_name: str,
        include_optimizer: bool = False,
        optimizer_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export model to PyTorch format.
        
        Args:
            model: Model to export
            model_name: Model name
            include_optimizer: Include optimizer state
            optimizer_state: Optional optimizer state
            
        Returns:
            Path to exported PyTorch file
        """
        output_path = self.output_dir / f"{model_name}.pth"
        
        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_config": getattr(model, "config", {}),
        }
        
        if include_optimizer and optimizer_state:
            save_dict["optimizer_state_dict"] = optimizer_state
        
        torch.save(save_dict, output_path)
        
        logger.info(f"Exported model to PyTorch: {output_path}")
        
        return output_path

