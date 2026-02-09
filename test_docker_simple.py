#!/usr/bin/env python3
"""
Simple Docker test for dataset versioning system
"""
import sys
import os
sys.path.insert(0, '/app/diri-helox')

print("🐳 Testing Dataset Versioning in Docker Container")
print("=" * 50)

try:
    # Test imports
    from utils.dataset_versioning import DatasetVersionManager, DatasetType
    print("✅ Successfully imported DatasetVersionManager")

    from utils.dataset_validation import validate_dataset_quality
    print("✅ Successfully imported dataset validation")

    from utils.dataset_monitoring import DatasetMonitor
    print("✅ Successfully imported monitoring")

    # Test basic functionality
    manager = DatasetVersionManager(
        db_url="sqlite:////tmp/docker_test.db",
        storage_backend="local"
    )
    print("✅ DatasetVersionManager initialized")

    # Test dataset types
    print(f"✅ Available dataset types: {[dt.value for dt in DatasetType]}")

    print("\n🎉 All imports and basic functionality working!")
    print("💡 The dataset versioning system is ready to use in Docker!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n📝 Next steps to test fully:")
print("1. Run: python /app/diri-helox/docker_test_versioning.py")
print("2. Or use CLI: python scripts/dataset_versioning_cli.py --help")
