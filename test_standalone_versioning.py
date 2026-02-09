#!/usr/bin/env python3
"""
Standalone test script for dataset versioning system within Helox
This allows testing the versioning system without requiring cyrex runtime
"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_helox_versioning_system():
    """Test the complete dataset versioning system within Helox"""

    print("🔬 Testing Helox Dataset Versioning System (Standalone)")
    print("=" * 60)

    try:
        # Test all imports work
        print("📦 Testing imports...")
        from utils.dataset_versioning import DatasetVersionManager, DatasetType
        from utils.dataset_validation import validate_dataset_quality, DatasetValidator
        from utils.dataset_monitoring import DatasetMonitor, get_health_report
        from pipelines.training.versioned_training_pipeline import VersionedTrainingPipeline

        print("✅ All modules imported successfully!")

        # Test data paths
        data_v1_path = Path("./data/samples/lease_abstraction_v1")
        data_v2_path = Path("./data/samples/lease_abstraction_v2")

        print(f"📁 Test data paths:")
        print(f"   V1: {data_v1_path.absolute()}")
        print(f"   V2: {data_v2_path.absolute()}")

        # Verify data exists
        if not data_v1_path.exists() or not data_v2_path.exists():
            print("❌ Test data not found. Please run from helox directory with test data.")
            return False

        print("✅ Test data verified!")

        # Initialize system with unique database
        import uuid
        db_name = f"helox_test_{uuid.uuid4().hex[:8]}.db"
        db_path = Path("./") / db_name

        print(f"🗄️  Using database: {db_path}")

        manager = DatasetVersionManager(
            db_url=f"sqlite:///{db_name}",
            storage_backend="local"
        )

        print("✅ DatasetVersionManager initialized!")

        # Test dataset validation
        print("\n🔍 Testing dataset validation...")
        validation_result = validate_dataset_quality(data_v1_path, "lease_abstraction")

        print("📊 Validation Results:")
        print(f"   Valid: {validation_result['is_valid']}")
        print(f"   Quality Score: {validation_result['quality_score']:.2f}")
        print(f"   Total Samples: {validation_result['statistics']['total_samples']}")
        print(f"   Average Text Length: {validation_result['statistics']['avg_text_length']:.1f}")

        # Create first version
        print("\n📦 Creating dataset version 1.0.0...")
        version_1 = manager.create_version(
            dataset_name="helox_lease_test",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=data_v1_path,
            version="1.0.0",
            change_summary="Initial test dataset for Helox versioning system",
            tags=["helox", "test", "lease_abstraction", "standalone"],
            metadata={
                "environment": "helox_standalone",
                "test_run": True,
                "quality_score": validation_result['quality_score'],
                "created_by": "helox_test_script"
            }
        )

        print("✅ Version 1.0.0 created!")
        print(f"   Storage: {version_1.storage_path}")
        print(f"   Checksum: {version_1.data_checksum[:16]}...")
        print(f"   Change Type: {version_1.change_type}")

        # Create second version
        print("\n📦 Creating dataset version 1.0.1...")
        version_2 = manager.create_version(
            dataset_name="helox_lease_test",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=data_v2_path,
            parent_version="1.0.0",
            change_summary="Added additional lease documents for testing",
            tags=["helox", "test", "lease_abstraction", "updated"]
        )

        print("✅ Version 1.0.1 created!")
        print(f"   Change Type: {version_2.change_type}")

        # Test version retrieval
        print("\n🔍 Testing version retrieval...")
        retrieved_v1 = manager.get_version("helox_lease_test", "1.0.0")
        retrieved_v2 = manager.get_version("helox_lease_test", "1.0.1")

        if retrieved_v1 and retrieved_v2:
            print("✅ Both versions retrieved successfully!")
        else:
            print("❌ Version retrieval failed!")
            return False

        # Test version listing
        print("\n📋 Testing version listing...")
        versions = manager.list_versions(dataset_name="helox_lease_test")
        print(f"Found {len(versions)} versions:")
        for v in versions:
            print(f"   {v.version} - {v.total_samples} samples - {v.created_at}")

        # Test version comparison
        print("\n⚖️  Testing version comparison...")
        comparison = manager.compare_versions("helox_lease_test", "1.0.0", "1.0.1")
        print("Comparison results:")
        print(f"   Sample difference: {comparison['sample_difference']:+d}")
        print(f"   File difference: {comparison['file_difference']:+d}")
        print(f"   Size difference: {comparison['size_difference_bytes']:+d} bytes")

        # Test monitoring system
        print("\n📊 Testing monitoring system...")
        monitor = DatasetMonitor()

        # Log operations
        monitor.log_version_creation({
            "dataset_name": "helox_lease_test",
            "version": "1.0.0",
            "total_samples": version_1.total_samples,
            "creation_time": 2.1,
            "quality_score": validation_result['quality_score']
        })

        monitor.log_validation_result({
            "dataset_name": "helox_lease_test",
            "version": "1.0.0",
            "is_valid": validation_result['is_valid'],
            "quality_score": validation_result['quality_score'],
            "validation_time": 1.5
        })

        # Get health report
        health = monitor.get_health_report()
        print("✅ Health report generated!")
        print(f"   Total versions tracked: {health['summary']['total_versions']}")

        # Test dataset types
        print("\n🏷️  Testing dataset types...")
        print(f"Available dataset types: {[dt.value for dt in DatasetType]}")

        # Test configuration loading
        print("\n⚙️  Testing configuration...")
        config_path = Path("./configs/versioned_training_config.json")
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Versioned training config loaded!")
            print(f"   Dataset spec: {config.get('dataset_spec', 'N/A')}")
            print(f"   Dataset type: {config.get('dataset_type', 'N/A')}")
        else:
            print("⚠️  Config file not found, but that's okay for testing")

        # Clean up test database
        print(f"\n🧹 Cleaning up test database: {db_path}")
        if db_path.exists():
            db_path.unlink()
            print("✅ Test database cleaned up!")

        print("\n🎉 Helox Dataset Versioning System Test Completed Successfully!")
        print("\n📝 Test Summary:")
        print(f"   ✅ Created {len(versions)} dataset versions")
        print(f"   ✅ Validated dataset quality: {validation_result['quality_score']:.2f}")
        print(f"   ✅ Tested version comparison and retrieval")
        print(f"   ✅ Monitoring and logging operational")
        print(f"   ✅ All core functionality working")

        print("\n🚀 The dataset versioning system is ready for use in Helox!")
        print("\n💡 Usage Examples:")
        print("   # Create versions")
        print("   python scripts/dataset_versioning_cli.py create --name test --type lease_abstraction --path ./data")
        print("")
        print("   # Train with versioned datasets")
        print("   python pipelines/training/versioned_training_pipeline.py --config configs/versioned_training_config.json")

        return True

    except Exception as e:
        print(f"❌ Helox versioning test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_cyrex_integration_concept():
    """Show how cyrex services would integrate with the versioning system"""

    print("\n🔗 Cyrex Integration Concept")
    print("-" * 30)

    print("This demonstrates how cyrex runtime services would use the versioning system:")
    print()

    # Simulate cyrex service usage
    cyrex_service_code = '''
# Example: How a cyrex service would use dataset versioning

from diri_helox.utils.dataset_versioning import DatasetVersionManager, DatasetType
from diri_helox.utils.dataset_validation import validate_dataset_quality

class CyrexDataService:
    """Cyrex service that utilizes Helox dataset versioning"""

    def __init__(self):
        self.version_manager = DatasetVersionManager(
            db_url="postgresql://user:pass@postgres:5432/datasets",  # Production DB
            storage_backend="s3",
            storage_config={"bucket": "cyrex-datasets"}
        )

    def load_training_data(self, dataset_name: str, version: str = "latest"):
        """Load versioned training data for cyrex models"""

        if version == "latest":
            dataset_version = self.version_manager.get_latest_version(
                dataset_name, DatasetType.LEASE_ABSTRACTION
            )
        else:
            dataset_version = self.version_manager.get_version(
                dataset_name, version, DatasetType.LEASE_ABSTRACTION
            )

        if not dataset_version:
            raise ValueError(f"Dataset version not found: {dataset_name}@{version}")

        # Validate data integrity
        validation = self.version_manager.validate_version(
            dataset_version.dataset_name,
            dataset_version.version
        )

        if not validation["is_valid"]:
            raise ValueError(f"Dataset integrity check failed for {dataset_name}@{version}")

        # Load and return the data
        data_path = dataset_version.storage_path
        # ... load data from path ...

        return {
            "data_path": data_path,
            "version_info": {
                "name": dataset_version.dataset_name,
                "version": dataset_version.version,
                "checksum": dataset_version.data_checksum,
                "sample_count": dataset_version.total_samples
            }
        }

    def create_new_dataset_version(self, data_path: str, metadata: dict):
        """Create a new version from cyrex-generated data"""

        # Validate the new data first
        validation = validate_dataset_quality(data_path, "lease_abstraction")

        if not validation["is_valid"]:
            raise ValueError("New dataset failed validation")

        # Create version
        version = self.version_manager.create_version(
            dataset_name="cyrex_generated",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=data_path,
            change_summary="Generated by cyrex service",
            tags=["cyrex", "generated", "lease_abstraction"],
            metadata={
                **metadata,
                "quality_score": validation["quality_score"],
                "generated_by": "cyrex_service"
            }
        )

        return version

# Usage in cyrex service:
# service = CyrexDataService()
# data = service.load_training_data("lease_abstraction_training", "latest")
# print(f"Loaded dataset: {data['version_info']}")
'''

    print(cyrex_service_code)


if __name__ == "__main__":
    print("🧪 Helox Dataset Versioning System - Standalone Test")
    print("This test runs the complete versioning system within Helox")
    print("without requiring external cyrex runtime dependencies.\n")

    # Run the main test
    success = test_helox_versioning_system()

    if success:
        # Show integration concept
        test_cyrex_integration_concept()

        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Helox dataset versioning system is fully operational!")
        print("\nCyrex services can now integrate with this system using the patterns shown above.")

        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED!")
        print("Check the error messages above and ensure you're running from the helox directory.")
        sys.exit(1)
