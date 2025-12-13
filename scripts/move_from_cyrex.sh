#!/bin/bash
# Script to move training files from diri-cyrex/app/train to diri-helox
# Run from deepiri root directory

set -e

CYREX_TRAIN_DIR="diri-cyrex/app/train"
HELOX_ROOT="diri-helox"

echo "📦 Moving training files from Cyrex to Helox..."

# Move pipelines
if [ -d "$CYREX_TRAIN_DIR/pipelines" ]; then
    echo "  Moving pipelines..."
    cp -r "$CYREX_TRAIN_DIR/pipelines"/* "$HELOX_ROOT/pipelines/training/" 2>/dev/null || true
fi

# Move scripts
if [ -d "$CYREX_TRAIN_DIR/scripts" ]; then
    echo "  Moving scripts..."
    cp -r "$CYREX_TRAIN_DIR/scripts"/* "$HELOX_ROOT/scripts/" 2>/dev/null || true
fi

# Move experiments
if [ -d "$CYREX_TRAIN_DIR/experiments" ]; then
    echo "  Moving experiments..."
    cp -r "$CYREX_TRAIN_DIR/experiments"/* "$HELOX_ROOT/experiments/notebooks/" 2>/dev/null || true
fi

# Move configs
if [ -d "$CYREX_TRAIN_DIR/configs" ]; then
    echo "  Moving configs..."
    cp -r "$CYREX_TRAIN_DIR/configs"/* "$HELOX_ROOT/experiments/configs/" 2>/dev/null || true
fi

# Move data preparation
if [ -d "$CYREX_TRAIN_DIR/data" ]; then
    echo "  Moving data preparation..."
    cp -r "$CYREX_TRAIN_DIR/data"/* "$HELOX_ROOT/pipelines/data_preprocessing/" 2>/dev/null || true
fi

# Move infrastructure
if [ -d "$CYREX_TRAIN_DIR/infrastructure" ]; then
    echo "  Moving infrastructure..."
    cp -r "$CYREX_TRAIN_DIR/infrastructure"/* "$HELOX_ROOT/mlops/" 2>/dev/null || true
fi

# Move utils
if [ -d "$CYREX_TRAIN_DIR/utils" ]; then
    echo "  Moving utils..."
    cp -r "$CYREX_TRAIN_DIR/utils"/* "$HELOX_ROOT/utils/" 2>/dev/null || true
fi

echo "✅ Files moved! Review and update imports as needed."

