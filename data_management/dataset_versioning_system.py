"""
Dataset versioning and lineage tracking system.

Tracks dataset checksums, sample counts, token counts, and version IDs
for complete dataset provenance and auditability.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import tqdm

logger = logging.getLogger(__name__)


class DatasetVersioningSystem:
    """
    Manages dataset versioning, checksums, and lineage.
    
    Tracks:
    - Dataset checksums
    - Sample/document counts
    - Token counts
    - Version IDs
    - Lineage (parent datasets)
    """
    
    def __init__(self, metadata_dir: Path = Path("data/metadata")):
        """
        Initialize versioning system.
        
        Args:
            metadata_dir: Directory for storing metadata
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_dataset_checksum(
        self,
        dataset_path: Path,
        chunk_size: int = 8192,
    ) -> str:
        """
        Compute SHA256 checksum of dataset.
        
        Args:
            dataset_path: Path to dataset file or directory
            chunk_size: Chunk size for reading
            
        Returns:
            Hex digest of checksum
        """
        dataset_path = Path(dataset_path)
        sha256 = hashlib.sha256()
        
        if dataset_path.is_file():
            with open(dataset_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
        elif dataset_path.is_dir():
            # Hash all files in directory
            for file_path in sorted(dataset_path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        while chunk := f.read(chunk_size):
                            sha256.update(chunk)
        else:
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        return sha256.hexdigest()
    
    def count_samples_and_tokens(
        self,
        dataset_path: Path,
        tokenizer_manager=None,
    ) -> Dict[str, int]:
        """
        Count samples and tokens in dataset.
        
        Args:
            dataset_path: Path to dataset file
            tokenizer_manager: Optional tokenizer for token counting
            
        Returns:
            Dictionary with counts
        """
        dataset_path = Path(dataset_path)
        sample_count = 0
        token_count = 0
        
        if dataset_path.suffix == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in tqdm.tqdm(f, desc="Counting samples"):
                    try:
                        data = json.loads(line)
                        sample_count += 1
                        
                        if tokenizer_manager and "text" in data:
                            tokens = tokenizer_manager.encode(data["text"])
                            token_count += len(tokens)
                    except json.JSONDecodeError:
                        continue
        else:
            # Assume directory with multiple files
            for file_path in dataset_path.rglob("*.jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            sample_count += 1
                            
                            if tokenizer_manager and "text" in data:
                                tokens = tokenizer_manager.encode(data["text"])
                                token_count += len(tokens)
                        except json.JSONDecodeError:
                            continue
        
        return {
            "sample_count": sample_count,
            "token_count": token_count,
        }
    
    def create_dataset_version(
        self,
        dataset_path: Path,
        dataset_id: str,
        parent_versions: Optional[List[str]] = None,
        tokenizer_manager=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create version record for dataset.
        
        Args:
            dataset_path: Path to dataset
            dataset_id: Unique dataset identifier
            parent_versions: List of parent dataset version IDs
            tokenizer_manager: Optional tokenizer for token counting
            metadata: Additional metadata
            
        Returns:
            Version record dictionary
        """
        dataset_path = Path(dataset_path)
        
        logger.info(f"Creating version for dataset: {dataset_id}")
        
        # Compute checksum
        checksum = self.compute_dataset_checksum(dataset_path)
        logger.info(f"Checksum: {checksum[:16]}...")
        
        # Count samples and tokens
        counts = self.count_samples_and_tokens(dataset_path, tokenizer_manager)
        logger.info(f"Samples: {counts['sample_count']:,}, Tokens: {counts['token_count']:,}")
        
        # Create version record
        version_record = {
            "dataset_id": dataset_id,
            "version": datetime.utcnow().isoformat(),
            "path": str(dataset_path.absolute()),
            "checksum": checksum,
            "sample_count": counts["sample_count"],
            "token_count": counts["token_count"],
            "parent_versions": parent_versions or [],
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        
        # Save version record
        version_file = self.metadata_dir / f"{dataset_id}_version.json"
        with open(version_file, "w") as f:
            json.dump(version_record, f, indent=2)
        
        # Update lineage
        self._update_lineage(dataset_id, version_record)
        
        logger.info(f"Version record saved: {version_file}")
        
        return version_record
    
    def _update_lineage(self, dataset_id: str, version_record: Dict[str, Any]):
        """Update dataset lineage tracking."""
        lineage_file = self.metadata_dir / "dataset_lineage.json"
        
        if lineage_file.exists():
            with open(lineage_file, "r") as f:
                lineage = json.load(f)
        else:
            lineage = {}
        
        if dataset_id not in lineage:
            lineage[dataset_id] = []
        
        lineage[dataset_id].append({
            "version": version_record["version"],
            "checksum": version_record["checksum"],
            "sample_count": version_record["sample_count"],
            "token_count": version_record["token_count"],
            "parent_versions": version_record["parent_versions"],
            "created_at": version_record["created_at"],
        })
        
        with open(lineage_file, "w") as f:
            json.dump(lineage, f, indent=2)
    
    def verify_dataset_integrity(
        self,
        dataset_path: Path,
        expected_checksum: str,
    ) -> bool:
        """
        Verify dataset integrity against expected checksum.
        
        Args:
            dataset_path: Path to dataset
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.compute_dataset_checksum(dataset_path)
        matches = actual_checksum == expected_checksum
        
        if not matches:
            logger.error(
                f"Checksum mismatch: expected {expected_checksum[:16]}..., "
                f"got {actual_checksum[:16]}..."
            )
        else:
            logger.info("Dataset integrity verified")
        
        return matches
    
    def get_dataset_lineage(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get lineage for a dataset."""
        lineage_file = self.metadata_dir / "dataset_lineage.json"
        
        if not lineage_file.exists():
            return []
        
        with open(lineage_file, "r") as f:
            lineage = json.load(f)
        
        return lineage.get(dataset_id, [])

