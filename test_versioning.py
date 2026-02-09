#!/usr/bin/env python3
"""
Simple test script for dataset versioning system
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset_versioning import DatasetVersionManager, DatasetType

def test_versioning():
    """Test the dataset versioning functionality"""

    print("🔄 Testing Dataset Versioning System")
    print("=" * 50)

    # Initialize manager with in-memory database for testing
    # Use a unique database name to avoid conflicts
    import uuid
    db_name = f"test_versions_{uuid.uuid4().hex[:8]}.db"

    manager = DatasetVersionManager(
        db_url=f"sqlite:///{db_name}",
        storage_backend="local"
    )

    print("✅ Manager initialized successfully!")

    # Test data paths
    data_v1_path = Path("./data/samples/lease_abstraction_v1")
    data_v2_path = Path("./data/samples/lease_abstraction_v2")

    print(f"📁 Testing with data paths:")
    print(f"   V1: {data_v1_path}")
    print(f"   V2: {data_v2_path}")

    # Create first version
    print("\n📦 Creating version 1.0.0...")
    try:
        version_1 = manager.create_version(
            dataset_name="lease_abstraction_training",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=data_v1_path,
            version="1.0.0",
            change_summary="Initial dataset with 5 lease documents",
            tags=["production", "lease_abstraction"],
            metadata={
                "source": "SEC filings",
                "annotation_method": "manual",
                "quality_score": 0.95
            }
        )
        print("✅ Version 1.0.0 created successfully!")
        print(f"   Samples: {version_1.total_samples}")
        print(f"   Checksum: {version_1.data_checksum[:16]}...")
        print(f"   Storage: {version_1.storage_path}")

    except Exception as e:
        print(f"❌ Failed to create version 1.0.0: {e}")
        return

    # Create second version
    print("\n📦 Creating version 1.0.1...")
    try:
        version_2 = manager.create_version(
            dataset_name="lease_abstraction_training",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=data_v2_path,
            parent_version="1.0.0",
            change_summary="Added 3 more lease documents",
            tags=["production", "lease_abstraction", "updated"]
        )
        print("✅ Version 1.0.1 created successfully!")
        print(f"   Samples: {version_2.total_samples}")
        print(f"   Checksum: {version_2.data_checksum[:16]}...")
        print(f"   Change type: {version_2.change_type}")

    except Exception as e:
        print(f"❌ Failed to create version 1.0.1: {e}")
        return

    # List versions
    print("\n📋 Listing all versions...")
    try:
        versions = manager.list_versions(dataset_name="lease_abstraction_training")
        print(f"Found {len(versions)} versions:")
        for v in versions:
            print(f"   {v.version} - {v.total_samples} samples - {v.created_at}")

    except Exception as e:
        print(f"❌ Failed to list versions: {e}")

    # Compare versions
    print("\n⚖️  Comparing versions 1.0.0 vs 1.0.1...")
    try:
        comparison = manager.compare_versions(
            "lease_abstraction_training",
            "1.0.0",
            "1.0.1"
        )
        print("Comparison results:")
        print(f"   Sample difference: {comparison['sample_difference']:+d}")
        print(f"   File difference: {comparison['file_difference']:+d}")
        print(f"   Size difference: {comparison['size_difference_bytes']:+d} bytes")
        print(f"   Change type: {comparison['change_type']}")

    except Exception as e:
        print(f"❌ Failed to compare versions: {e}")

    # Test retrieval
    print("\n🔍 Testing version retrieval...")
    try:
        retrieved = manager.get_version("lease_abstraction_training", "1.0.0")
        if retrieved:
            print("✅ Successfully retrieved version 1.0.0")
            print(f"   Samples: {retrieved.total_samples}")
            print(f"   Tags: {retrieved.tags}")
        else:
            print("❌ Version not found")

    except Exception as e:
        print(f"❌ Failed to retrieve version: {e}")

    print("\n🎉 Dataset versioning system test completed!")
    print("\n💡 Next steps:")
    print("   1. Run: python scripts/dataset_versioning_cli.py --help")
    print("   2. Try creating versions with the CLI")
    print("   3. Integrate with your training pipelines")

if __name__ == "__main__":
    test_versioning()
