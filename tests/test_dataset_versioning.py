#!/usr/bin/env python3
"""
Self-contained tests for the dataset versioning system.
Uses temporary directories and a temp SQLite DB — no existing data required.

Required: pydantic, sqlalchemy (or install full deps with pip install -r requirements.txt).

Run from repo root (diri-helox):
  pytest tests/test_dataset_versioning.py -v
  python tests/test_dataset_versioning.py
"""
import os
import sys
import tempfile
import json
from pathlib import Path

# Ensure diri-helox is on path when run as script or from project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load dataset_versioning without importing the rest of utils (e.g. avoids numpy from confidence_classes)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dataset_versioning",
    _REPO_ROOT / "utils" / "dataset_versioning.py",
)
_dataset_versioning = importlib.util.module_from_spec(_spec)
# dataset_versioning.py needs pydantic/sqlalchemy; it does not need numpy
_spec.loader.exec_module(_dataset_versioning)
DatasetVersionManager = _dataset_versioning.DatasetVersionManager
DatasetType = _dataset_versioning.DatasetType


def _make_sample_dataset(base_dir: Path, name: str, num_lines: int) -> Path:
    """Create a minimal dataset dir with a .jsonl file. Returns path to that dir."""
    d = base_dir / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "samples.jsonl"
    with open(path, "w") as f:
        for i in range(num_lines):
            f.write(json.dumps({"id": i, "text": f"sample {i}"}) + "\n")
    return d


def test_dataset_versioning_e2e():
    """
    End-to-end test: create two versions, list, get, compare, validate.
    Runs in a temp directory so ./datasets/ is isolated.
    """
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            # Use a DB file in temp dir so it's cleaned up
            db_path = Path(tmp) / "test_versions.db"
            manager = DatasetVersionManager(
                db_url=f"sqlite:///{db_path}",
                storage_backend="local",
            )

            # Create minimal datasets (v1: 3 samples, v2: 5 samples)
            data_v1 = _make_sample_dataset(Path(tmp), "data_v1", 3)
            data_v2 = _make_sample_dataset(Path(tmp), "data_v2", 5)

            # --- Create version 1.0.0 ---
            v1 = manager.create_version(
                dataset_name="test_dataset",
                dataset_type=DatasetType.LEASE_ABSTRACTION,
                data_path=data_v1,
                version="1.0.0",
                change_summary="Initial",
                tags=["test"],
            )
            assert v1.version == "1.0.0"
            assert v1.total_samples == 3
            assert v1.file_count == 1
            assert len(v1.data_checksum) == 64
            assert Path(v1.storage_path).exists()

            # --- Create version 1.0.1 (auto or explicit) ---
            v2 = manager.create_version(
                dataset_name="test_dataset",
                dataset_type=DatasetType.LEASE_ABSTRACTION,
                data_path=data_v2,
                parent_version="1.0.0",
                change_summary="Added samples",
                tags=["test"],
            )
            assert v2.version == "1.0.1"
            assert v2.total_samples == 5
            assert v2.parent_version == "1.0.0"

            # --- List versions ---
            versions = manager.list_versions(dataset_name="test_dataset")
            assert len(versions) >= 2
            assert {v.version for v in versions}.issuperset({"1.0.0", "1.0.1"})

            # --- Get specific version ---
            got = manager.get_version("test_dataset", "1.0.0")
            assert got is not None
            assert got.version == "1.0.0" and got.total_samples == 3

            # --- Latest version ---
            latest = manager.get_latest_version("test_dataset")
            assert latest is not None
            assert latest.version == "1.0.1"

            # --- Compare versions ---
            comparison = manager.compare_versions("test_dataset", "1.0.0", "1.0.1")
            assert comparison["sample_difference"] == 2
            assert comparison["version1"] == "1.0.0" and comparison["version2"] == "1.0.1"

            # --- Validate integrity (local storage) ---
            result = manager.validate_version("test_dataset", "1.0.0")
            assert result["is_valid"] is True
            assert result["expected_checksum"] == result["actual_checksum"]

        finally:
            os.chdir(original_cwd)


def test_auto_version_increment():
    """Test that version auto-increments when not provided."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            db_path = Path(tmp) / "test_auto.db"
            manager = DatasetVersionManager(
                db_url=f"sqlite:///{db_path}",
                storage_backend="local",
            )
            data = _make_sample_dataset(Path(tmp), "data", 2)

            v1 = manager.create_version(
                dataset_name="auto_test",
                dataset_type=DatasetType.CONTRACT_INTELLIGENCE,
                data_path=data,
                change_summary="First",
            )
            assert v1.version == "1.0.0"

            v2 = manager.create_version(
                dataset_name="auto_test",
                dataset_type=DatasetType.CONTRACT_INTELLIGENCE,
                data_path=data,
                change_summary="Second",
            )
            assert v2.version == "1.0.1"

        finally:
            os.chdir(original_cwd)


def test_s3_raises_not_implemented():
    """S3 storage and retrieve must raise NotImplementedError until implemented."""
    manager = DatasetVersionManager(
        db_url="sqlite:///:memory:",
        storage_backend="s3",
        storage_config={"bucket": "test-bucket"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        data_path = _make_sample_dataset(Path(tmp), "data", 1)
        try:
            manager.create_version(
                dataset_name="s3_test",
                dataset_type=DatasetType.LEASE_ABSTRACTION,
                data_path=data_path,
                version="1.0.0",
            )
            assert False, "Expected NotImplementedError"
        except NotImplementedError as e:
            assert "S3" in str(e) or "s3" in str(e).lower()

    manager_local = DatasetVersionManager(
        db_url="sqlite:///:memory:",
        storage_backend="local",
    )
    try:
        manager_local._retrieve_dataset("s3://bucket/key/")
        assert False, "Expected NotImplementedError for S3 retrieve"
    except NotImplementedError as e:
        assert "S3" in str(e) or "s3" in str(e).lower()


if __name__ == "__main__":
    print("Running dataset versioning tests...")
    test_dataset_versioning_e2e()
    print("  test_dataset_versioning_e2e passed")
    test_auto_version_increment()
    print("  test_auto_version_increment passed")
    test_s3_raises_not_implemented()
    print("  test_s3_raises_not_implemented passed")
    print("All tests passed.")