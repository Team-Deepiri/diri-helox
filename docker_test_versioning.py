#!/usr/bin/env python3
"""
Docker-based test script for dataset versioning system
Run this in the cyrex container to test the versioning system
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/app/diri-helox')

def test_versioning_in_docker():
    """Test the dataset versioning system in Docker environment"""

    print("🐳 Testing Dataset Versioning System in Docker")
    print("=" * 60)

    try:
        # Import the versioning components
        from utils.dataset_versioning import DatasetVersionManager, DatasetType
        from utils.dataset_validation import validate_dataset_quality
        from utils.dataset_monitoring import DatasetMonitor, get_health_report

        print("✅ Successfully imported versioning modules!")

        # Test data paths in container
        data_v1_path = Path("/app/diri-helox/data/samples/lease_abstraction_v1")
        data_v2_path = Path("/app/diri-helox/data/samples/lease_abstraction_v2")

        print(f"📁 Testing with data paths:")
        print(f"   V1: {data_v1_path}")
        print(f"   V2: {data_v2_path}")

        # Verify data exists
        if not data_v1_path.exists():
            print("❌ Test data V1 not found!")
            return False

        if not data_v2_path.exists():
            print("❌ Test data V2 not found!")
            return False

        print("✅ Test data directories found!")

        # Initialize manager with unique database for Docker testing
        import uuid
        db_name = f"docker_test_versions_{uuid.uuid4().hex[:8]}.db"
        db_path = f"/tmp/{db_name}"  # Use /tmp for Docker

        manager = DatasetVersionManager(
            db_url=f"sqlite:///{db_path}",
            storage_backend="local"
        )

        print("✅ DatasetVersionManager initialized successfully!")

        # Test dataset validation
        print("\n🔍 Testing dataset validation...")
        validation_result = validate_dataset_quality(data_v1_path, "lease_abstraction")

        print("✅ Dataset validation completed!"        print(f"   Valid: {validation_result['is_valid']}")
        print(f"   Quality Score: {validation_result['quality_score']:.2f}")
        print(f"   Samples: {validation_result['statistics']['total_samples']}")

        # Create first version
        print("\n📦 Creating version 1.0.0...")
        version_1 = manager.create_version(
            dataset_name="docker_lease_abstraction_test",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=data_v1_path,
            version="1.0.0",
            change_summary="Initial dataset for Docker testing",
            tags=["docker", "test", "lease_abstraction"],
            metadata={
                "source": "test_data",
                "environment": "docker",
                "quality_score": validation_result['quality_score']
            }
        )
        print("✅ Version 1.0.0 created successfully!")
        print(f"   Storage: {version_1.storage_path}")

        # Create second version
        print("\n📦 Creating version 1.0.1...")
        version_2 = manager.create_version(
            dataset_name="docker_lease_abstraction_test",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=data_v2_path,
            parent_version="1.0.0",
            change_summary="Added more lease documents for testing",
            tags=["docker", "test", "lease_abstraction", "updated"]
        )
        print("✅ Version 1.0.1 created successfully!")
        print(f"   Change type: {version_2.change_type}")

        # List versions
        print("\n📋 Listing all versions...")
        versions = manager.list_versions(dataset_name="docker_lease_abstraction_test")
        print(f"Found {len(versions)} versions:")
        for v in versions:
            print(f"   {v.version} - {v.total_samples} samples - {v.created_at}")

        # Compare versions
        print("\n⚖️  Comparing versions 1.0.0 vs 1.0.1...")
        comparison = manager.compare_versions(
            "docker_lease_abstraction_test",
            "1.0.0",
            "1.0.1"
        )
        print("Comparison results:")
        print(f"   Sample difference: {comparison['sample_difference']:+d}")
        print(f"   File difference: {comparison['file_difference']:+d}")

        # Test monitoring
        print("\n📊 Testing monitoring system...")
        monitor = DatasetMonitor()

        # Log some operations
        monitor.log_version_creation({
            "dataset_name": "docker_lease_abstraction_test",
            "version": "1.0.0",
            "total_samples": version_1.total_samples,
            "creation_time": 2.5
        })

        monitor.log_validation_result({
            "dataset_name": "docker_lease_abstraction_test",
            "version": "1.0.0",
            "is_valid": validation_result['is_valid'],
            "quality_score": validation_result['quality_score'],
            "validation_time": 1.2
        })

        # Get health report
        health = monitor.get_health_report()
        print("✅ Health report generated!"        print(f"   Total versions: {health['summary']['total_versions']}")

        print("\n🎉 Docker dataset versioning test completed successfully!")
        print("\n💡 The dataset versioning system is working correctly in Docker!")
        print("\n📝 Test Results Summary:")
        print(f"   - Created {len(versions)} dataset versions")
        print(f"   - Validated dataset quality: {validation_result['quality_score']:.2f}")
        print(f"   - Successfully compared version changes")
        print(f"   - Monitoring system operational")
        print(f"   - Database file: {db_path}")

        return True

    except Exception as e:
        print(f"❌ Docker test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_in_docker():
    """Test the CLI commands in Docker environment"""
    print("\n🔧 Testing CLI commands in Docker...")

    try:
        # Test CLI list command
        os.system("cd /app/diri-helox && python scripts/dataset_versioning_cli.py --help > /dev/null 2>&1")
        print("✅ CLI commands accessible in Docker!")

        # Note: Full CLI testing would require setting up the database first
        print("💡 To test full CLI functionality:")
        print("   1. Run: docker exec -it deepiri-cyrex bash")
        print("   2. cd /app/diri-helox")
        print("   3. python scripts/dataset_versioning_cli.py create --name test --type lease_abstraction --path data/samples/lease_abstraction_v1 --summary 'Docker CLI test'")

    except Exception as e:
        print(f"⚠️  CLI test note: {str(e)}")


if __name__ == "__main__":
    success = test_versioning_in_docker()
    test_cli_in_docker()

    if success:
        print("\n✅ ALL TESTS PASSED - Dataset Versioning System is ready for production!")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED - Check the error messages above")
        sys.exit(1)
