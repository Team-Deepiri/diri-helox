#!/bin/bash
# Setup script to create and configure a virtual environment

set -e  # Exit on error

echo "Setting up virtual environment for diri-helox..."

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if venv already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists at ./venv"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment."
        echo "To activate it, run: source venv/bin/activate"
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install minimal requirements (for data preprocessing pipeline only)
echo "Installing minimal dependencies from requirements-minimal.txt..."
echo "Note: PyTorch and ML dependencies excluded for Python 3.13 compatibility"
pip install -r requirements-minimal.txt

# Ask if user wants to install full requirements
read -p "Do you want to install full requirements.txt? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing full requirements (may fail on Python 3.13)..."
    pip install -r requirements.txt || echo "⚠️  Some packages may not be available for Python 3.13"
fi

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "To run the tests, activate the venv and then:"
echo "  python3 tests/test_data_preprocessing_pipeline.py"

