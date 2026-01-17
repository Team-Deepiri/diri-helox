# Dataset Versioning System for Language Intelligence

## Overview

This guide explains how to implement and use a dataset versioning system for language intelligence training pipelines in Helox. Dataset versioning is critical for tracking training data evolution, ensuring reproducibility, and managing model training across different data versions.

## Table of Contents

1. [Why Dataset Versioning?](#why-dataset-versioning)
2. [System Architecture](#system-architecture)
3. [Implementation Guide](#implementation-guide)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Integration with Training Pipelines](#integration-with-training-pipelines)

---

## Why Dataset Versioning?

### Benefits for Language Intelligence

1. **Reproducibility**: Track exactly which data version was used to train each model
2. **Data Lineage**: Understand how training datasets evolved over time
3. **A/B Testing**: Compare model performance across different dataset versions
4. **Compliance**: Maintain audit trails for regulatory requirements
5. **Rollback Capability**: Revert to previous dataset versions if needed
6. **Collaboration**: Multiple team members can work with consistent data versions

### Use Cases in Language Intelligence

- **Lease Abstraction Training**: Version lease document datasets as new leases are added
- **Contract Intelligence Training**: Track contract clause extraction datasets across versions
- **Obligation Dependency Training**: Version dependency graph training data
- **Regulatory Language Training**: Track regulation document datasets as regulations evolve

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│           Dataset Versioning System                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Version      │    │  Metadata   │                   │
│  │  Manager      │───▶│  Store      │                   │
│  └──────────────┘    └──────────────┘                   │
│         │                    │                           │
│         ▼                    ▼                           │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Storage     │    │  Validation  │                   │
│  │  Backend     │    │  Engine      │                   │
│  └──────────────┘    └──────────────┘                   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Dataset Creation/Update** → Version Manager creates new version
2. **Validation** → Validation Engine checks data quality
3. **Storage** → Data stored in versioned format
4. **Metadata** → Version metadata stored in database
5. **Training** → Training pipelines reference specific versions

---

## Implementation Guide

### Step 1: Install Dependencies

Add to `requirements.txt`:

```txt
dvc>=3.0.0
dvc-s3>=3.0.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
alembic>=1.12.0
```

### Step 2: Create Version Manager

**File**: `diri-helox/utils/dataset_versioning.py`

```python
"""
Dataset Versioning Manager
Manages dataset versions for language intelligence training
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import hashlib
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class DatasetType(str, Enum):
    """Types of datasets for language intelligence"""
    LEASE_ABSTRACTION = "lease_abstraction"
    CONTRACT_INTELLIGENCE = "contract_intelligence"
    OBLIGATION_DEPENDENCY = "obligation_dependency"
    REGULATORY_LANGUAGE = "regulatory_language"
    CLAUSE_EVOLUTION = "clause_evolution"


class DatasetVersion(Base):
    """Database model for dataset versions"""
    __tablename__ = "dataset_versions"
    
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    dataset_type = Column(String(50), nullable=False)
    
    # Storage information
    storage_path = Column(String(500), nullable=False)
    storage_backend = Column(String(50), default="s3")  # s3, local, minio
    
    # Data statistics
    total_samples = Column(Integer, nullable=False)
    file_count = Column(Integer, nullable=False)
    total_size_bytes = Column(Integer, nullable=False)
    
    # Checksums for integrity
    data_checksum = Column(String(64), nullable=False)
    metadata_checksum = Column(String(64), nullable=False)
    
    # Version metadata
    parent_version = Column(String(50), nullable=True)  # Previous version
    change_summary = Column(Text, nullable=True)
    change_type = Column(String(50))  # MAJOR, MINOR, PATCH
    
    # Quality metrics
    quality_score = Column(String(20), nullable=True)
    validation_status = Column(String(50), default="PENDING")
    validation_errors = Column(JSON, nullable=True)
    
    # Metadata
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    
    # Relations
    training_runs = []  # Will be linked to training runs


class DatasetVersionMetadata(BaseModel):
    """Pydantic model for dataset version metadata"""
    dataset_name: str
    version: str
    dataset_type: DatasetType
    storage_path: str
    storage_backend: str = "s3"
    
    total_samples: int
    file_count: int
    total_size_bytes: int
    
    data_checksum: str
    metadata_checksum: str
    
    parent_version: Optional[str] = None
    change_summary: Optional[str] = None
    change_type: str = "PATCH"  # MAJOR, MINOR, PATCH
    
    quality_score: Optional[float] = None
    validation_status: str = "PENDING"
    validation_errors: Optional[List[Dict[str, Any]]] = None
    
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None


class DatasetVersionManager:
    """
    Manages dataset versions for language intelligence training
    
    Features:
    - Create new dataset versions
    - Track version lineage
    - Validate dataset integrity
    - Query version history
    - Compare versions
    """
    
    def __init__(
        self,
        db_url: str,
        storage_backend: str = "s3",
        storage_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize version manager
        
        Args:
            db_url: Database URL for version metadata
            storage_backend: Storage backend (s3, local, minio)
            storage_config: Configuration for storage backend
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}
    
    def create_version(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        data_path: Path,
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
        change_summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> DatasetVersionMetadata:
        """
        Create a new dataset version
        
        Args:
            dataset_name: Name of the dataset
            dataset_type: Type of dataset (lease, contract, etc.)
            data_path: Path to dataset files
            version: Version string (e.g., "1.0.0"). Auto-increments if None
            parent_version: Previous version this is based on
            change_summary: Description of changes
            tags: Tags for organization
            metadata: Additional metadata
            created_by: User/process that created this version
            
        Returns:
            DatasetVersionMetadata object
        """
        # Determine version number
        if version is None:
            version = self._get_next_version(dataset_name, dataset_type)
        
        # Calculate statistics
        stats = self._calculate_statistics(data_path)
        
        # Calculate checksums
        data_checksum = self._calculate_data_checksum(data_path)
        metadata_checksum = self._calculate_metadata_checksum(
            dataset_name, version, stats, metadata or {}
        )
        
        # Determine change type
        change_type = self._determine_change_type(
            dataset_name, dataset_type, parent_version, stats
        )
        
        # Store dataset
        storage_path = self._store_dataset(
            dataset_name, version, data_path
        )
        
        # Create version record
        version_metadata = DatasetVersionMetadata(
            dataset_name=dataset_name,
            version=version,
            dataset_type=dataset_type,
            storage_path=storage_path,
            storage_backend=self.storage_backend,
            total_samples=stats["total_samples"],
            file_count=stats["file_count"],
            total_size_bytes=stats["total_size_bytes"],
            data_checksum=data_checksum,
            metadata_checksum=metadata_checksum,
            parent_version=parent_version,
            change_summary=change_summary,
            change_type=change_type,
            tags=tags or [],
            metadata=metadata or {},
            created_by=created_by,
        )
        
        # Save to database
        self._save_version_to_db(version_metadata)
        
        return version_metadata
    
    def get_version(
        self,
        dataset_name: str,
        version: str,
        dataset_type: Optional[DatasetType] = None
    ) -> Optional[DatasetVersionMetadata]:
        """Get version metadata"""
        session = self.Session()
        try:
            query = session.query(DatasetVersion).filter(
                DatasetVersion.dataset_name == dataset_name,
                DatasetVersion.version == version
            )
            
            if dataset_type:
                query = query.filter(DatasetVersion.dataset_type == dataset_type.value)
            
            version_record = query.first()
            
            if not version_record:
                return None
            
            return self._db_to_metadata(version_record)
        finally:
            session.close()
    
    def list_versions(
        self,
        dataset_name: Optional[str] = None,
        dataset_type: Optional[DatasetType] = None,
        limit: int = 100
    ) -> List[DatasetVersionMetadata]:
        """List all versions, optionally filtered"""
        session = self.Session()
        try:
            query = session.query(DatasetVersion)
            
            if dataset_name:
                query = query.filter(DatasetVersion.dataset_name == dataset_name)
            
            if dataset_type:
                query = query.filter(DatasetVersion.dataset_type == dataset_type.value)
            
            versions = query.order_by(DatasetVersion.created_at.desc()).limit(limit).all()
            
            return [self._db_to_metadata(v) for v in versions]
        finally:
            session.close()
    
    def get_latest_version(
        self,
        dataset_name: str,
        dataset_type: Optional[DatasetType] = None
    ) -> Optional[DatasetVersionMetadata]:
        """Get latest version of a dataset"""
        versions = self.list_versions(dataset_name, dataset_type, limit=1)
        return versions[0] if versions else None
    
    def compare_versions(
        self,
        dataset_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a dataset
        
        Returns:
            Dictionary with comparison results
        """
        v1 = self.get_version(dataset_name, version1)
        v2 = self.get_version(dataset_name, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        return {
            "version1": v1.version,
            "version2": v2.version,
            "sample_difference": v2.total_samples - v1.total_samples,
            "file_difference": v2.file_count - v1.file_count,
            "size_difference_bytes": v2.total_size_bytes - v1.total_size_bytes,
            "change_type": v2.change_type,
            "change_summary": v2.change_summary,
        }
    
    def validate_version(
        self,
        dataset_name: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Validate dataset version integrity
        
        Returns:
            Dictionary with validation results
        """
        version_meta = self.get_version(dataset_name, version)
        if not version_meta:
            raise ValueError(f"Version {version} not found")
        
        # Recalculate checksum
        data_path = self._retrieve_dataset(version_meta.storage_path)
        current_checksum = self._calculate_data_checksum(data_path)
        
        is_valid = current_checksum == version_meta.data_checksum
        
        return {
            "is_valid": is_valid,
            "expected_checksum": version_meta.data_checksum,
            "actual_checksum": current_checksum,
            "validation_timestamp": datetime.utcnow().isoformat(),
        }
    
    # Private helper methods
    
    def _get_next_version(
        self,
        dataset_name: str,
        dataset_type: DatasetType
    ) -> str:
        """Get next version number (semantic versioning)"""
        latest = self.get_latest_version(dataset_name, dataset_type)
        
        if not latest:
            return "1.0.0"
        
        # Parse current version
        parts = latest.version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Increment patch version
        return f"{major}.{minor}.{patch + 1}"
    
    def _calculate_statistics(self, data_path: Path) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        total_samples = 0
        file_count = 0
        total_size = 0
        
        # Count files and samples
        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size
                
                # Count samples based on file type
                if file_path.suffix == ".jsonl":
                    with open(file_path, "r") as f:
                        total_samples += sum(1 for _ in f)
                elif file_path.suffix == ".json":
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_samples += len(data)
                        else:
                            total_samples += 1
        
        return {
            "total_samples": total_samples,
            "file_count": file_count,
            "total_size_bytes": total_size,
        }
    
    def _calculate_data_checksum(self, data_path: Path) -> str:
        """Calculate SHA256 checksum of dataset"""
        hasher = hashlib.sha256()
        
        # Hash all files in sorted order
        for file_path in sorted(data_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _calculate_metadata_checksum(
        self,
        dataset_name: str,
        version: str,
        stats: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Calculate checksum of metadata"""
        metadata_str = json.dumps({
            "dataset_name": dataset_name,
            "version": version,
            "stats": stats,
            "metadata": metadata,
        }, sort_keys=True)
        
        return hashlib.sha256(metadata_str.encode()).hexdigest()
    
    def _determine_change_type(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        parent_version: Optional[str],
        stats: Dict[str, Any]
    ) -> str:
        """Determine semantic version change type"""
        if not parent_version:
            return "MAJOR"  # First version
        
        parent = self.get_version(dataset_name, parent_version, dataset_type)
        if not parent:
            return "MAJOR"
        
        # Compare statistics
        sample_diff = stats["total_samples"] - parent.total_samples
        sample_change_pct = abs(sample_diff) / parent.total_samples if parent.total_samples > 0 else 0
        
        if sample_change_pct > 0.5:  # >50% change
            return "MAJOR"
        elif sample_change_pct > 0.1:  # >10% change
            return "MINOR"
        else:
            return "PATCH"
    
    def _store_dataset(
        self,
        dataset_name: str,
        version: str,
        data_path: Path
    ) -> str:
        """Store dataset in versioned location"""
        if self.storage_backend == "local":
            storage_path = Path(f"./datasets/{dataset_name}/{version}")
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            import shutil
            shutil.copytree(data_path, storage_path, dirs_exist_ok=True)
            
            return str(storage_path.absolute())
        
        elif self.storage_backend == "s3":
            # Implement S3 storage
            s3_path = f"s3://{self.storage_config.get('bucket')}/{dataset_name}/{version}/"
            # Use boto3 to upload
            # ... implementation ...
            return s3_path
        
        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")
    
    def _retrieve_dataset(self, storage_path: str) -> Path:
        """Retrieve dataset from storage"""
        if storage_path.startswith("s3://"):
            # Download from S3
            # ... implementation ...
            pass
        else:
            return Path(storage_path)
    
    def _save_version_to_db(self, metadata: DatasetVersionMetadata):
        """Save version metadata to database"""
        session = self.Session()
        try:
            version_record = DatasetVersion(
                dataset_name=metadata.dataset_name,
                version=metadata.version,
                dataset_type=metadata.dataset_type.value,
                storage_path=metadata.storage_path,
                storage_backend=metadata.storage_backend,
                total_samples=metadata.total_samples,
                file_count=metadata.file_count,
                total_size_bytes=metadata.total_size_bytes,
                data_checksum=metadata.data_checksum,
                metadata_checksum=metadata.metadata_checksum,
                parent_version=metadata.parent_version,
                change_summary=metadata.change_summary,
                change_type=metadata.change_type,
                quality_score=str(metadata.quality_score) if metadata.quality_score else None,
                validation_status=metadata.validation_status,
                validation_errors=metadata.validation_errors,
                tags=metadata.tags,
                metadata=metadata.metadata,
                created_by=metadata.created_by,
            )
            
            session.add(version_record)
            session.commit()
        finally:
            session.close()
    
    def _db_to_metadata(self, db_record: DatasetVersion) -> DatasetVersionMetadata:
        """Convert database record to metadata model"""
        return DatasetVersionMetadata(
            dataset_name=db_record.dataset_name,
            version=db_record.version,
            dataset_type=DatasetType(db_record.dataset_type),
            storage_path=db_record.storage_path,
            storage_backend=db_record.storage_backend,
            total_samples=db_record.total_samples,
            file_count=db_record.file_count,
            total_size_bytes=db_record.total_size_bytes,
            data_checksum=db_record.data_checksum,
            metadata_checksum=db_record.metadata_checksum,
            parent_version=db_record.parent_version,
            change_summary=db_record.change_summary,
            change_type=db_record.change_type,
            quality_score=float(db_record.quality_score) if db_record.quality_score else None,
            validation_status=db_record.validation_status,
            validation_errors=db_record.validation_errors,
            tags=db_record.tags or [],
            metadata=db_record.metadata or {},
            created_at=db_record.created_at,
            created_by=db_record.created_by,
        )
```

### Step 3: Create CLI Interface

**File**: `diri-helox/scripts/dataset_versioning_cli.py`

```python
"""
CLI for dataset versioning operations
"""
import click
from pathlib import Path
from utils.dataset_versioning import (
    DatasetVersionManager,
    DatasetType,
)

@click.group()
def dataset():
    """Dataset versioning commands"""
    pass

@dataset.command()
@click.option("--name", required=True, help="Dataset name")
@click.option("--type", type=click.Choice([dt.value for dt in DatasetType]), required=True)
@click.option("--path", type=click.Path(exists=True), required=True)
@click.option("--version", help="Version string (auto-increments if not provided)")
@click.option("--parent", help="Parent version")
@click.option("--summary", help="Change summary")
@click.option("--tags", help="Comma-separated tags")
def create(name, type, path, version, parent, summary, tags):
    """Create a new dataset version"""
    manager = DatasetVersionManager(
        db_url="sqlite:///dataset_versions.db",
        storage_backend="local"
    )
    
    tag_list = tags.split(",") if tags else None
    
    metadata = manager.create_version(
        dataset_name=name,
        dataset_type=DatasetType(type),
        data_path=Path(path),
        version=version,
        parent_version=parent,
        change_summary=summary,
        tags=tag_list,
    )
    
    click.echo(f"Created version {metadata.version} of {name}")
    click.echo(f"  Storage: {metadata.storage_path}")
    click.echo(f"  Samples: {metadata.total_samples}")
    click.echo(f"  Checksum: {metadata.data_checksum[:16]}...")

@dataset.command()
@click.option("--name", required=True)
@click.option("--version", help="Specific version (default: latest)")
def list(name, version):
    """List dataset versions"""
    manager = DatasetVersionManager(
        db_url="sqlite:///dataset_versions.db"
    )
    
    if version:
        metadata = manager.get_version(name, version)
        if metadata:
            click.echo(f"Version {metadata.version}:")
            click.echo(f"  Created: {metadata.created_at}")
            click.echo(f"  Samples: {metadata.total_samples}")
            click.echo(f"  Change: {metadata.change_type}")
        else:
            click.echo(f"Version {version} not found")
    else:
        versions = manager.list_versions(dataset_name=name)
        click.echo(f"Versions for {name}:")
        for v in versions:
            click.echo(f"  {v.version} - {v.created_at} - {v.total_samples} samples")

@dataset.command()
@click.option("--name", required=True)
@click.option("--version1", required=True)
@click.option("--version2", required=True)
def compare(name, version1, version2):
    """Compare two dataset versions"""
    manager = DatasetVersionManager(
        db_url="sqlite:///dataset_versions.db"
    )
    
    comparison = manager.compare_versions(name, version1, version2)
    
    click.echo(f"Comparison: {version1} vs {version2}")
    click.echo(f"  Sample difference: {comparison['sample_difference']:+d}")
    click.echo(f"  File difference: {comparison['file_difference']:+d}")
    click.echo(f"  Size difference: {comparison['size_difference_bytes']:+d} bytes")
    click.echo(f"  Change type: {comparison['change_type']}")

if __name__ == "__main__":
    dataset()
```

---

## Usage Examples

### Example 1: Creating a Lease Abstraction Dataset Version

```python
from pathlib import Path
from utils.dataset_versioning import DatasetVersionManager, DatasetType

# Initialize manager
manager = DatasetVersionManager(
    db_url="postgresql://user:pass@localhost/datasets",
    storage_backend="s3",
    storage_config={"bucket": "helox-datasets"}
)

# Create version 1.0.0
version_1 = manager.create_version(
    dataset_name="lease_abstraction_training",
    dataset_type=DatasetType.LEASE_ABSTRACTION,
    data_path=Path("./data/leases_v1"),
    version="1.0.0",
    change_summary="Initial dataset with 5,000 lease documents",
    tags=["production", "lease_abstraction"],
    metadata={
        "source": "SEC filings",
        "annotation_method": "manual",
        "quality_score": 0.95
    },
    created_by="data_team"
)

print(f"Created version {version_1.version}")
print(f"  Samples: {version_1.total_samples}")
print(f"  Checksum: {version_1.data_checksum}")
```

### Example 2: Creating an Updated Version

```python
# Create version 1.1.0 (adds more data)
version_2 = manager.create_version(
    dataset_name="lease_abstraction_training",
    dataset_type=DatasetType.LEASE_ABSTRACTION,
    data_path=Path("./data/leases_v2"),
    parent_version="1.0.0",
    change_summary="Added 2,000 new lease documents from customer data",
    tags=["production", "lease_abstraction", "customer_data"],
    created_by="data_team"
)

# Compare versions
comparison = manager.compare_versions(
    "lease_abstraction_training",
    "1.0.0",
    "1.1.0"
)

print(f"Added {comparison['sample_difference']} samples")
print(f"Change type: {comparison['change_type']}")
```

### Example 3: Using Version in Training Pipeline

```python
from pipelines.lease_abstraction_training_pipeline import LeaseAbstractionTrainingPipeline
from utils.dataset_versioning import DatasetVersionManager, DatasetType

# Get specific version
manager = DatasetVersionManager(...)
version_meta = manager.get_version(
    "lease_abstraction_training",
    "1.1.0",
    DatasetType.LEASE_ABSTRACTION
)

# Load dataset from versioned storage
dataset_path = manager._retrieve_dataset(version_meta.storage_path)

# Use in training
pipeline = LeaseAbstractionTrainingPipeline()
pipeline.train(
    dataset_path=dataset_path,
    dataset_version=version_meta.version,
    output_dir=f"./models/lease_abstraction_v{version_meta.version}"
)
```

### Example 4: Querying Version History

```python
# List all versions
all_versions = manager.list_versions(
    dataset_name="lease_abstraction_training"
)

for v in all_versions:
    print(f"{v.version}: {v.total_samples} samples, {v.created_at}")

# Get latest version
latest = manager.get_latest_version(
    "lease_abstraction_training",
    DatasetType.LEASE_ABSTRACTION
)

print(f"Latest: {latest.version} with {latest.total_samples} samples")
```

---

## Best Practices

### 1. Semantic Versioning

Use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Significant dataset changes (>50% samples changed, schema changes)
- **MINOR**: Moderate changes (10-50% samples added/modified)
- **PATCH**: Small changes (<10% samples, bug fixes)

### 2. Change Summaries

Always provide clear change summaries:
```python
change_summary="Added 2,000 retail lease documents. Fixed annotation errors in 50 samples."
```

### 3. Tagging Strategy

Use consistent tags:
- `production`, `staging`, `experimental`
- Dataset type: `lease_abstraction`, `contract_intelligence`
- Source: `sec_filings`, `customer_data`, `synthetic`

### 4. Validation

Validate datasets before creating versions:
```python
# Validate data quality
validation_result = validate_dataset_quality(data_path)
if validation_result["score"] < 0.8:
    raise ValueError("Dataset quality too low")
```

### 5. Metadata

Include comprehensive metadata:
```python
metadata={
    "source": "SEC filings",
    "annotation_method": "manual",
    "annotators": ["annotator1", "annotator2"],
    "inter_annotator_agreement": 0.95,
    "quality_score": 0.92,
    "regulations_applicable": ["ASC 842"],
}
```

### 6. Storage Backend

Choose appropriate storage:
- **Local**: Development, small datasets
- **S3/MinIO**: Production, large datasets, team collaboration

---

## Integration with Training Pipelines

### Modify Training Pipelines to Use Versioning

**File**: `diri-helox/pipelines/lease_abstraction_training_pipeline.py`

```python
from utils.dataset_versioning import DatasetVersionManager, DatasetType

class LeaseAbstractionTrainingPipeline:
    def __init__(self, version_manager: DatasetVersionManager):
        self.version_manager = version_manager
    
    def train(
        self,
        dataset_name: str,
        dataset_version: str,
        output_dir: str,
        **kwargs
    ):
        # Get dataset version
        version_meta = self.version_manager.get_version(
            dataset_name,
            dataset_version,
            DatasetType.LEASE_ABSTRACTION
        )
        
        if not version_meta:
            raise ValueError(f"Version {dataset_version} not found")
        
        # Load dataset
        dataset_path = self.version_manager._retrieve_dataset(
            version_meta.storage_path
        )
        
        # Train model
        model = self._train_model(dataset_path, **kwargs)
        
        # Save model with version info
        model_metadata = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "dataset_checksum": version_meta.data_checksum,
            "training_timestamp": datetime.utcnow().isoformat(),
        }
        
        self._save_model(model, output_dir, model_metadata)
        
        return model
```

---

## Database Schema

Create migration file:

**File**: `diri-helox/migrations/001_create_dataset_versions.sql`

```sql
CREATE TABLE dataset_versions (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    dataset_type VARCHAR(50) NOT NULL,
    
    storage_path VARCHAR(500) NOT NULL,
    storage_backend VARCHAR(50) DEFAULT 's3',
    
    total_samples INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    total_size_bytes INTEGER NOT NULL,
    
    data_checksum VARCHAR(64) NOT NULL,
    metadata_checksum VARCHAR(64) NOT NULL,
    
    parent_version VARCHAR(50),
    change_summary TEXT,
    change_type VARCHAR(50),
    
    quality_score VARCHAR(20),
    validation_status VARCHAR(50) DEFAULT 'PENDING',
    validation_errors JSONB,
    
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    
    UNIQUE(dataset_name, version, dataset_type)
);

CREATE INDEX idx_dataset_versions_name ON dataset_versions(dataset_name);
CREATE INDEX idx_dataset_versions_type ON dataset_versions(dataset_type);
CREATE INDEX idx_dataset_versions_created ON dataset_versions(created_at);
```

---

## CLI Usage

```bash
# Create new version
python -m scripts.dataset_versioning_cli dataset create \
    --name lease_abstraction_training \
    --type lease_abstraction \
    --path ./data/leases_v1 \
    --summary "Initial dataset with 5,000 leases"

# List versions
python -m scripts.dataset_versioning_cli dataset list \
    --name lease_abstraction_training

# Compare versions
python -m scripts.dataset_versioning_cli dataset compare \
    --name lease_abstraction_training \
    --version1 1.0.0 \
    --version2 1.1.0
```

---

## Next Steps

1. **Set up database**: Create database and run migrations
2. **Configure storage**: Set up S3/MinIO storage backend
3. **Create first version**: Version your initial training datasets
4. **Integrate with pipelines**: Update training pipelines to use versioning
5. **Set up monitoring**: Track dataset usage in training runs

---

## Troubleshooting

### Issue: Checksum mismatch
**Solution**: Dataset files may have been modified. Re-validate and recreate version if needed.

### Issue: Storage backend errors
**Solution**: Check storage credentials and permissions. Verify storage backend configuration.

### Issue: Version conflicts
**Solution**: Use semantic versioning consistently. Check for existing versions before creating new ones.

---

## References

- [Semantic Versioning](https://semver.org/)
- [DVC (Data Version Control)](https://dvc.org/)
- [MLflow Data Versioning](https://mlflow.org/docs/latest/tracking.html)

