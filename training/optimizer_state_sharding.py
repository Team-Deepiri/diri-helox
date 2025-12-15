"""
Optimizer state sharding system.

Provides ZeRO-style optimizer partitioning and CPU offloading hooks
for scaling beyond 1B parameters.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizerStateSharder:
    """
    Shards optimizer state across devices/CPU for memory efficiency.
    
    Implements ZeRO-style partitioning:
    - Partition optimizer states across devices
    - CPU offloading for large models
    - State gathering for checkpointing
    """
    
    def __init__(
        self,
        num_shards: int = 1,
        cpu_offload: bool = False,
        shard_strategy: str = "partition",  # partition, replicate
    ):
        """
        Initialize optimizer state sharder.
        
        Args:
            num_shards: Number of shards
            cpu_offload: Enable CPU offloading
            shard_strategy: Sharding strategy
        """
        self.num_shards = num_shards
        self.cpu_offload = cpu_offload
        self.shard_strategy = shard_strategy
        
        self.shard_assignments: Dict[str, int] = {}
        self.offloaded_states: Dict[str, torch.Tensor] = {}
    
    def partition_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        shard_id: int,
    ) -> torch.optim.Optimizer:
        """
        Partition optimizer state across shards.
        
        Args:
            optimizer: Optimizer to partition
            shard_id: Shard ID for this process
            
        Returns:
            Optimizer with partitioned state
        """
        if self.num_shards == 1:
            return optimizer
        
        # Partition state dict
        state_dict = optimizer.state_dict()
        partitioned_state = {}
        
        param_groups = []
        for group_idx, group in enumerate(optimizer.param_groups):
            partitioned_group = group.copy()
            partitioned_group["params"] = []
            
            for param_idx, param in enumerate(group["params"]):
                param_id = f"{group_idx}_{param_idx}"
                
                # Assign to shard
                assigned_shard = hash(param_id) % self.num_shards
                
                if assigned_shard == shard_id:
                    partitioned_group["params"].append(param)
                    
                    # Partition state
                    if param in state_dict["state"]:
                        partitioned_state[param] = state_dict["state"][param]
            
            if partitioned_group["params"]:
                param_groups.append(partitioned_group)
        
        # Create new optimizer with partitioned state
        optimizer_type = type(optimizer)
        partitioned_optimizer = optimizer_type(param_groups)
        partitioned_optimizer.load_state_dict({
            "state": partitioned_state,
            "param_groups": param_groups,
        })
        
        logger.info(
            f"Optimizer partitioned: shard {shard_id}/{self.num_shards}, "
            f"{len(partitioned_group['params'])} params"
        )
        
        return partitioned_optimizer
    
    def offload_state_to_cpu(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Offload optimizer state to CPU.
        
        Args:
            optimizer: Optimizer to offload
        """
        if not self.cpu_offload:
            return
        
        state_dict = optimizer.state_dict()
        
        for param, state in state_dict["state"].items():
            param_id = id(param)
            
            # Offload to CPU
            cpu_state = {}
            for key, value in state.items():
                if torch.is_tensor(value):
                    cpu_state[key] = value.cpu()
                else:
                    cpu_state[key] = value
            
            self.offloaded_states[param_id] = cpu_state
        
        logger.info(f"Offloaded {len(self.offloaded_states)} states to CPU")
    
    def load_state_from_cpu(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        """
        Load optimizer state from CPU to device.
        
        Args:
            optimizer: Optimizer to load state into
            device: Target device
        """
        if not self.offloaded_states:
            return
        
        state_dict = optimizer.state_dict()
        
        for param, state in state_dict["state"].items():
            param_id = id(param)
            
            if param_id in self.offloaded_states:
                cpu_state = self.offloaded_states[param_id]
                
                # Load to device
                for key, value in cpu_state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
                    else:
                        state[key] = value
        
        logger.info(f"Loaded {len(self.offloaded_states)} states from CPU")
    
    def gather_state_for_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        shard_id: int,
    ) -> Dict[str, Any]:
        """
        Gather optimizer state from all shards for checkpointing.
        
        Args:
            optimizer: Optimizer
            shard_id: Shard ID
            
        Returns:
            Complete state dict (for this shard)
        """
        state_dict = optimizer.state_dict()
        
        # Include shard metadata
        state_dict["shard_metadata"] = {
            "shard_id": shard_id,
            "num_shards": self.num_shards,
            "cpu_offload": self.cpu_offload,
        }
        
        return state_dict

