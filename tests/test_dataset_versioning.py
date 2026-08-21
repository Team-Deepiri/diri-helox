#!/usr/bin/env python3
"""
Tests for the dataset versioning system against the real platform Postgres.

No throwaway databases: these tests run against the platform dev database
(``postgres-core`` from deepiri-platform ``docker-compose.dev.yml`` —
localhost:5433, db ``deepiri``). The connection is overridable via env var so
other environments (CI, staging) can point at their own instance without code
changes:

    export HELOX_DATASET_VERSIONING_DB_URL=postgresql://user:pass@host:5432/db

The manager creates the ``dataset_versions`` table via SQLAlchemy if it does
not exist (matching migrations/001), and each test uses a unique dataset name
with guaranteed row cleanup, so the shared database is never polluted.

Storage is always local under a temp dir; only the metadata lives in Postgres.

Run from repo root (diri-helox):
  docker compose -f ../docker-compose.dev.yml up -d postgres-core   # once
  pytest tests/test_dataset_versioning.py -v
"""

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

pytest.importorskip(
    "deepiri_dataset_processor",
    reason='Install sibling: poetry run pip install -e "../../deepiri-dataset-processor[all]" (see pyproject.toml)',
)

# Ensure diri-helox is on path when run as script or from project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load dataset_versioning without importing the rest of utils (e.g. avoids numpy from confidence_classes)
from utils.dataset_versioning import DatasetType, DatasetVersionManager  # noqa: E402

#: Platform dev database (deepiri-platform/docker-compose.dev.yml -> postgres-core).
DEFAULT_DB_URL = "postgresql://deepiri:deepiripassword@localhost:5433/deepiri"
DB_URL = os.environ.get("HELOX_DATASET_VERSIONING_DB_URL", DEFAULT_DB_URL)


def _make_sample_dataset(base_dir: Path, name: str, num_lines: int) -> Path:
    """Create a minimal dataset dir with a .jsonl file. Returns path to that dir."""
    d = base_dir / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "samples.jsonl"
    with open(path, "w") as f:
        for i in range(num_lines):
            f.write(json.dumps({"id": i, "text": f"sample {i}"}) + "\n")
    return d


@pytest.fixture(scope="session")
def db_url():
    """Resolve the platform Postgres URL, skipping cleanly when unreachable."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(DB_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
    except Exception as e:
        pytest.skip(
            f"Platform Postgres not reachable at {DB_URL!r} ({e.__class__.__name__}). "
            "Start it with: docker compose -f docker-compose.dev.yml up -d postgres-core "
            "(or set HELOX_DATASET_VERSIONING_DB_URL)."
        )
    yield DB_URL


@pytest.fixture()
def manager(tmp_path, db_url):
    """DatasetVersionManager backed by the shared platform Postgres."""
    return DatasetVersionManager(db_url=db_url, storage_backend="local")


@pytest.fixture()
def dataset_name(db_url):
    """Unique per-test dataset name; removes its rows afterwards."""
    name = f"test_dv_{uuid.uuid4().hex[:8]}"
    yield name
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("DELETE FROM dataset_versions WHERE dataset_name = :n"),
            {"n": name},
        )


def test_dataset_versioning_e2e(manager, dataset_name, tmp_path):
    """
    End-to-end test: create two versions, list, get, compare, validate.
    Metadata goes to the shared platform Postgres; files stay in a temp dir.
    """
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)

            # Create minimal datasets (v1: 3 samples, v2: 5 samples)
            data_v1 = _make_sample_dataset(Path(tmp), "data_v1", 3)
            data_v2 = _make_sample_dataset(Path(tmp), "data_v2", 5)

            # --- Create version 1.0.0 ---
            v1 = manager.create_version(
                dataset_name=dataset_name,
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
                dataset_name=dataset_name,
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
            versions = manager.list_versions(dataset_name=dataset_name)
            assert {v.version for v in versions}.issuperset({"1.0.0", "1.0.1"})

            # --- Get specific version ---
            got = manager.get_version(dataset_name, "1.0.0")
            assert got is not None
            assert got.version == "1.0.0" and got.total_samples == 3

            # --- Latest version ---
            latest = manager.get_latest_version(dataset_name)
            assert latest is not None
            assert latest.version == "1.0.1"

            # --- Compare versions ---
            comparison = manager.compare_versions(dataset_name, "1.0.0", "1.0.1")
            assert comparison["sample_difference"] == 2
            assert comparison["version1"] == "1.0.0" and comparison["version2"] == "1.0.1"

            # --- Validate integrity (local storage) ---
            result = manager.validate_version(dataset_name, "1.0.0")
            assert result["is_valid"] is True
            assert result["expected_checksum"] == result["actual_checksum"]
        finally:
            os.chdir(original_cwd)


def test_auto_version_increment(manager, dataset_name, tmp_path):
    """Test that version auto-increments when not provided."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            data = _make_sample_dataset(Path(tmp), "data", 2)

            v1 = manager.create_version(
                dataset_name=dataset_name,
                dataset_type=DatasetType.CONTRACT_INTELLIGENCE,
                data_path=data,
                change_summary="First",
            )
            assert v1.version == "1.0.0"

            v2 = manager.create_version(
                dataset_name=dataset_name,
                dataset_type=DatasetType.CONTRACT_INTELLIGENCE,
                data_path=data,
                change_summary="Second",
            )
            assert v2.version == "1.0.1"
        finally:
            os.chdir(original_cwd)


def test_rows_persist_in_postgres(manager, dataset_name, tmp_path, db_url):
    """Rows land in the shared platform Postgres, not a local file."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            data = _make_sample_dataset(Path(tmp), "data", 1)
            manager.create_version(
                dataset_name=dataset_name,
                dataset_type=DatasetType.REGULATORY_LANGUAGE,
                data_path=data,
                change_summary="persist check",
            )
            sqlalchemy = pytest.importorskip("sqlalchemy")
            engine = sqlalchemy.create_engine(db_url)
            with engine.connect() as conn:
                count = conn.execute(
                    sqlalchemy.text(
                        "SELECT COUNT(*) FROM dataset_versions WHERE dataset_name = :n"
                    ),
                    {"n": dataset_name},
                ).scalar()
            assert count == 1
        finally:
            os.chdir(original_cwd)


def test_s3_raises_not_implemented():
    """S3 backend must fail fast with NotImplementedError until S3 upload is implemented."""
    try:
        DatasetVersionManager(
            db_url="postgresql://deepiri:deepiripassword@localhost:5433/deepiri",
            storage_backend="s3",
            storage_config={"bucket": "test-bucket"},
        )
        assert False, "Expected NotImplementedError"
    except NotImplementedError as e:
        assert "s3" in str(e).lower() or "S3" in str(e)


if __name__ == "__main__":
    # These tests require pytest fixtures (DB session, cleanup); no plain runner.
    print(f"DB_URL: {os.environ.get('HELOX_DATASET_VERSIONING_DB_URL', DEFAULT_DB_URL)}")
    raise SystemExit("Run via: pytest tests/test_dataset_versioning.py -v")
