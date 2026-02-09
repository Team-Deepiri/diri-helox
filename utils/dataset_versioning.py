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
    dataset_metadata = Column(JSON, default={})

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
        # Add connection parameters to avoid locking issues
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args = {
                "check_same_thread": False,
                "timeout": 60
            }

        self.engine = create_engine(
            db_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=3600
        )

        # Create tables with error handling
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            print(f"Warning: Could not create tables automatically: {e}")
            print("Tables may already exist or database is locked.")

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

        # Parse current version (allow "1.0.0", "1.0", "v1.0.0")
        raw = latest.version.strip().lstrip("v")
        parts = raw.split(".")
        try:
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except (ValueError, IndexError):
            return "1.0.0"
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
            raise NotImplementedError(
                "S3 storage is not implemented yet. Use storage_backend='local' for now, "
                "or implement upload via boto3 to the path: "
                f"s3://{self.storage_config.get('bucket', 'your-bucket')}/{dataset_name}/{version}/"
            )

        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")

    def _retrieve_dataset(self, storage_path: str) -> Path:
        """Retrieve dataset from storage. Returns a Path to the data (local only)."""
        if storage_path.startswith("s3://"):
            raise NotImplementedError(
                "Retrieving from S3 is not implemented yet. Use storage_backend='local' "
                "or implement download via boto3."
            )
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
                dataset_metadata=metadata.metadata,
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
            metadata=db_record.dataset_metadata or {},
            created_at=db_record.created_at,
            created_by=db_record.created_by,
        )
