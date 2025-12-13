#!/usr/bin/env python3
"""
Setup Training Environment
Checks dependencies and prepares everything for training
"""
import sys
import subprocess
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, you have {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_requirements():
    """Install requirements from requirements.txt"""
    # Try root requirements.txt first, fallback to train-specific if needed
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        # Fallback to train-specific requirements if root doesn't exist
        requirements_file = Path("app/train/requirements.txt")
        if not requirements_file.exists():
            print(f"❌ Requirements file not found: {requirements_file}")
            return False
    
    print(f"\nInstalling dependencies from {requirements_file}...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("\n✓ All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install dependencies: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        if check_package(package_name, import_name):
            print(f"✓ {package_name}")
        else:
            print(f"❌ {package_name} (missing)")
            missing.append(package_name)
    
    return missing

def check_cuda():
    """Check if CUDA is available"""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  CUDA not available (will use CPU)")
            print("   Training will be slower but still works")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "app/train/data",
        "app/train/data/exported",
        "app/train/models",
        "app/train/models/intent_classifier",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")
    
    return True

def main():
    """Main setup function"""
    print("=" * 80)
    print("🔧 TRAINING ENVIRONMENT SETUP")
    print("=" * 80)
    print()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        return False
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} required package(s)")
        response = input("\nInstall missing dependencies? (y/n): ").strip().lower()
        
        if response == 'y':
            if not install_requirements():
                print("\n❌ Setup failed: Could not install dependencies")
                return False
            
            # Re-check
            missing = check_dependencies()
            if missing:
                print(f"\n❌ Still missing packages: {', '.join(missing)}")
                return False
        else:
            print("\n❌ Setup cancelled: Dependencies required")
            print(f"   Run: pip install -r requirements.txt")
            return False
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        return False
    
    # Success
    print("\n" + "=" * 80)
    print("✅ SETUP COMPLETE!")
    print("=" * 80)
    print()
    print("You're ready to train! Run:")
    print("  python3 app/train/scripts/run_training_pipeline.py")
    print()
    print("Or step by step:")
    print("  1. python3 app/train/scripts/generate_synthetic_data.py")
    print("  2. python3 app/train/scripts/prepare_training_data.py")
    print("  3. python3 app/train/scripts/train_intent_classifier.py")
    print("  4. python3 app/train/scripts/evaluate_trained_model.py")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

