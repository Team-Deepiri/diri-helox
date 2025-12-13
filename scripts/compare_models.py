#!/usr/bin/env python3
"""
Model Comparison Tool
Compare multiple trained models side-by-side
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def load_training_info(model_path: str) -> Dict:
    """Load training info from model directory"""
    info_file = Path(model_path) / "training_info.json"
    if info_file.exists():
        with open(info_file) as f:
            return json.load(f)
    return {}

def load_evaluation_report(model_path: str) -> Dict:
    """Load evaluation report from model directory"""
    report_file = Path(model_path) / "evaluation_report.json"
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    return {}

def compare_models(model_paths: List[str]):
    """Compare multiple models"""
    
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print()
    
    models_data = []
    for path in model_paths:
        model_path = Path(path)
        if not model_path.exists():
            print(f"⚠️  Model not found: {path}")
            continue
        
        training_info = load_training_info(path)
        eval_report = load_evaluation_report(path)
        
        models_data.append({
            "path": path,
            "name": model_path.name,
            "training_info": training_info,
            "eval_report": eval_report
        })
    
    if not models_data:
        print("❌ No valid models found")
        return
    
    # Print comparison table
    print(f"{'Model':<30} {'Accuracy':<12} {'F1 Score':<12} {'Epochs':<8} {'Examples':<10}")
    print("-" * 100)
    
    for model in models_data:
        name = model['name'][:28]
        
        # Get metrics
        accuracy = "N/A"
        f1 = "N/A"
        epochs = "N/A"
        examples = "N/A"
        
        if model['training_info']:
            epochs = model['training_info'].get('num_epochs', 'N/A')
            examples = model['training_info'].get('train_examples', 'N/A')
        
        if model['eval_report'] and 'metrics' in model['eval_report']:
            metrics = model['eval_report']['metrics']['overall']
            accuracy = f"{metrics['accuracy']:.4f}"
            f1 = f"{metrics['f1']:.4f}"
        
        print(f"{name:<30} {accuracy:<12} {f1:<12} {epochs!s:<8} {examples!s:<10}")
    
    # Detailed comparison
    print("\n" + "=" * 100)
    print("DETAILED METRICS")
    print("=" * 100)
    
    for i, model in enumerate(models_data, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Path: {model['path']}")
        
        if model['eval_report'] and 'metrics' in model['eval_report']:
            overall = model['eval_report']['metrics']['overall']
            print(f"   Accuracy:  {overall['accuracy']:.4f}")
            print(f"   Precision: {overall['precision']:.4f}")
            print(f"   Recall:    {overall['recall']:.4f}")
            print(f"   F1 Score:  {overall['f1']:.4f}")
            print(f"   Avg Confidence: {overall['avg_confidence']:.4f}")
        else:
            print("   (No evaluation report available)")
    
    # Best model
    if all('eval_report' in m and 'metrics' in m['eval_report'] for m in models_data):
        best_model = max(
            models_data,
            key=lambda m: m['eval_report']['metrics']['overall']['accuracy']
        )
        
        print("\n" + "=" * 100)
        print("🏆 BEST MODEL")
        print("=" * 100)
        print(f"Name: {best_model['name']}")
        print(f"Path: {best_model['path']}")
        print(f"Accuracy: {best_model['eval_report']['metrics']['overall']['accuracy']:.4f}")
        print()

def create_model_snapshot(model_path: str, version: str = None):
    """Create a versioned snapshot of the model"""
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Generate version name
    if version is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"
    
    # Create snapshot directory
    snapshot_dir = model_path.parent / f"{model_path.name}_{version}"
    
    print(f"Creating model snapshot: {snapshot_dir}")
    
    # Copy model files
    import shutil
    shutil.copytree(model_path, snapshot_dir, dirs_exist_ok=True)
    
    # Add snapshot metadata
    snapshot_info = {
        "original_path": str(model_path),
        "snapshot_version": version,
        "created_at": datetime.datetime.now().isoformat(),
    }
    
    with open(snapshot_dir / "snapshot_info.json", 'w') as f:
        json.dump(snapshot_info, f, indent=2)
    
    print(f"✓ Snapshot created: {snapshot_dir}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["app/train/models/intent_classifier"],
        help="Paths to models to compare"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a versioned snapshot of the model"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version name for snapshot"
    )
    
    args = parser.parse_args()
    
    if args.snapshot:
        # Create snapshot
        for model_path in args.models:
            create_model_snapshot(model_path, args.version)
    else:
        # Compare models
        compare_models(args.models)

