"""
Streaming and sharded dataset manager.

Provides IterableDataset support, shard-aware sampling, and
resume mid-epoch capabilities for multi-TB corpora.
"""

import logging
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
import json
import torch
from torch.utils.data import IterableDataset, DataLoader
import random

logger = logging.getLogger(__name__)


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large corpora.
    
    Supports:
    - IterableDataset for memory efficiency
    - Shard-aware sampling
    - Resume mid-epoch
    """
    
    def __init__(
        self,
        data_paths: List[Path],
        tokenizer_manager,
        max_length: int = 8192,
        shuffle: bool = True,
        shard_id: int = 0,
        num_shards: int = 1,
        resume_from_position: Optional[int] = None,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_paths: List of paths to data files
            tokenizer_manager: Tokenizer manager instance
            max_length: Maximum sequence length
            shuffle: Whether to shuffle
            shard_id: Shard ID for this worker
            num_shards: Total number of shards
            resume_from_position: Resume from specific position
        """
        self.data_paths = [Path(p) for p in data_paths]
        self.tokenizer_manager = tokenizer_manager
        self.max_length = max_length
        self.shuffle = shuffle
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.resume_from_position = resume_from_position
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset."""
        # Get shard-specific files
        shard_files = self._get_shard_files()
        
        # Resume from position if specified
        start_position = self.resume_from_position or 0
        
        for file_path in shard_files:
            with open(file_path, "r", encoding="utf-8") as f:
                # Skip to resume position
                for _ in range(start_position):
                    try:
                        next(f)
                    except StopIteration:
                        break
                
                # Read and yield samples
                position = start_position
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        
                        if not text:
                            continue
                        
                        # Tokenize
                        token_ids = self.tokenizer_manager.encode(
                            text,
                            add_bos=True,
                            add_eos=True,
                        )
                        
                        # Truncate if necessary
                        if len(token_ids) > self.max_length:
                            token_ids = token_ids[:self.max_length]
                        
                        yield {
                            "input_ids": token_ids,
                            "text": text,
                            "position": position,
                            "file": str(file_path),
                        }
                        
                        position += 1
                    except json.JSONDecodeError:
                        continue
    
    def _get_shard_files(self) -> List[Path]:
        """Get files for this shard."""
        if self.num_shards == 1:
            return self.data_paths
        
        # Distribute files across shards
        shard_files = []
        for i, file_path in enumerate(self.data_paths):
            if i % self.num_shards == self.shard_id:
                shard_files.append(file_path)
        
        if self.shuffle:
            random.shuffle(shard_files)
        
        return shard_files


class ShardedDatasetManager:
    """
    Manages sharded datasets for distributed training.
    
    Handles:
    - Shard distribution
    - Resume state tracking
    - Progress persistence
    """
    
    def __init__(self, state_dir: Path = Path("data/training_state")):
        """
        Initialize sharded dataset manager.
        
        Args:
            state_dir: Directory for storing resume state
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def create_streaming_dataloader(
        self,
        data_paths: List[Path],
        tokenizer_manager,
        batch_size: int = 2,
        max_length: int = 8192,
        shuffle: bool = True,
        num_workers: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        resume_from_checkpoint: Optional[Path] = None,
    ) -> DataLoader:
        """
        Create streaming dataloader with shard support.
        
        Args:
            data_paths: List of data file paths
            tokenizer_manager: Tokenizer manager
            batch_size: Batch size
            max_length: Maximum sequence length
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            shard_id: Shard ID for this worker
            num_shards: Total number of shards
            resume_from_checkpoint: Optional checkpoint to resume from
            
        Returns:
            DataLoader instance
        """
        # Load resume position if checkpoint provided
        resume_position = None
        if resume_from_checkpoint:
            resume_position = self._load_resume_position(
                resume_from_checkpoint,
                shard_id,
            )
        
        dataset = StreamingTextDataset(
            data_paths=data_paths,
            tokenizer_manager=tokenizer_manager,
            max_length=max_length,
            shuffle=shuffle,
            shard_id=shard_id,
            num_shards=num_shards,
            resume_from_position=resume_position,
        )
        
        def collate_fn(batch):
            """Collate function for batching."""
            input_ids = [item["input_ids"] for item in batch]
            
            # Pad sequences
            max_len = max(len(ids) for ids in input_ids)
            max_len = min(max_len, max_length)
            
            padded_ids = []
            attention_masks = []
            
            for ids in input_ids:
                if len(ids) > max_len:
                    ids = ids[:max_len]
                
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [0] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
            
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    
    def save_resume_state(
        self,
        checkpoint_dir: Path,
        shard_id: int,
        position: int,
    ):
        """Save resume state for shard."""
        state_file = checkpoint_dir / f"shard_{shard_id}_resume_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "shard_id": shard_id,
            "position": position,
        }
        
        import json
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_resume_position(
        self,
        checkpoint_dir: Path,
        shard_id: int,
    ) -> Optional[int]:
        """Load resume position for shard."""
        state_file = checkpoint_dir / f"shard_{shard_id}_resume_state.json"
        
        if not state_file.exists():
            return None
        
        import json
        with open(state_file, "r") as f:
            state = json.load(f)
        
        return state.get("position")

