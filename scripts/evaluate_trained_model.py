#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Tests the trained model and generates detailed performance reports
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter, defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Missing dependencies: {e}")
    print("Install: pip install transformers torch scikit-learn")
    IMPORTS_AVAILABLE = False

# Category mapping
CATEGORIES = {
    0: "coding",
    1: "writing",
    2: "fitness",
    3: "cleaning",
    4: "learning",
    5: "creative",
    6: "administrative",
    7: "social"
}

def load_test_data(test_file: str) -> List[Dict]:
    """Load test data"""
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def load_model(model_path: str):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")
    
    return model, tokenizer, device

def predict_batch(model, tokenizer, device, texts: List[str], batch_size: int = 32):
    """Predict on batch of texts"""
    predictions = []
    confidences = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
            batch_confs = torch.max(probs, dim=-1)[0].cpu().numpy()
            
            predictions.extend(batch_preds)
            confidences.extend(batch_confs)
    
    return predictions, confidences

def calculate_metrics(y_true, y_pred, y_conf):
    """Calculate comprehensive metrics"""
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Confidence statistics
    avg_confidence = np.mean(y_conf)
    correct_mask = np.array(y_true) == np.array(y_pred)
    avg_confidence_correct = np.mean(np.array(y_conf)[correct_mask])
    avg_confidence_incorrect = np.mean(np.array(y_conf)[~correct_mask]) if np.any(~correct_mask) else 0
    
    return {
        "overall": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_confidence": float(avg_confidence),
            "avg_confidence_correct": float(avg_confidence_correct),
            "avg_confidence_incorrect": float(avg_confidence_incorrect)
        },
        "per_class": {
            CATEGORIES[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support_per_class[i])
            }
            for i in range(len(CATEGORIES))
        },
        "confusion_matrix": cm.tolist()
    }

def print_confusion_matrix(cm, categories):
    """Print confusion matrix in readable format"""
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    
    # Header
    header = "Actual \\ Pred |"
    for cat in categories.values():
        header += f" {cat[:6]:>6} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, cat_name in categories.items():
        row = f"{cat_name:>13} |"
        for j in range(len(categories)):
            row += f" {cm[i][j]:>6} |"
        print(row)
    print()

def find_misclassifications(test_data, predictions, confidences, top_n=10):
    """Find most confident misclassifications"""
    misclassifications = []
    
    for i, (item, pred, conf) in enumerate(zip(test_data, predictions, confidences)):
        true_label = item['label']
        if pred != true_label:
            misclassifications.append({
                "text": item['text'],
                "true_label": CATEGORIES[true_label],
                "predicted_label": CATEGORIES[pred],
                "confidence": float(conf),
                "index": i
            })
    
    # Sort by confidence (most confident mistakes first)
    misclassifications.sort(key=lambda x: x['confidence'], reverse=True)
    
    return misclassifications[:top_n]

def evaluate_model(
    model_path: str = "app/train/models/intent_classifier",
    test_file: str = "app/train/data/classification_test.jsonl",
    output_file: str = "app/train/models/intent_classifier/evaluation_report.json"
):
    """Complete model evaluation"""
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot evaluate: missing dependencies")
        return False
    
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # Check files exist
    model_path_obj = Path(model_path)
    test_file_obj = Path(test_file)
    
    if not model_path_obj.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train a model first: python3 app/train/scripts/train_intent_classifier.py")
        return False
    
    if not test_file_obj.exists():
        print(f"❌ Test data not found: {test_file}")
        print("   Generate data: python3 app/train/scripts/generate_synthetic_data.py")
        return False
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    test_data = load_test_data(test_file)
    print(f"  ✓ {len(test_data)} test examples")
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    print("  ✓ Model loaded")
    
    # Extract texts and labels
    texts = [item['text'] for item in test_data]
    true_labels = [item['label'] for item in test_data]
    
    # Predict
    print("\nRunning predictions...")
    predictions, confidences = predict_batch(model, tokenizer, device, texts)
    print("  ✓ Predictions complete")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(true_labels, predictions, confidences)
    
    # Print results
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"Accuracy:              {metrics['overall']['accuracy']:.4f}")
    print(f"Precision (weighted):  {metrics['overall']['precision']:.4f}")
    print(f"Recall (weighted):     {metrics['overall']['recall']:.4f}")
    print(f"F1 Score (weighted):   {metrics['overall']['f1']:.4f}")
    print(f"\nAvg Confidence:        {metrics['overall']['avg_confidence']:.4f}")
    print(f"  Correct predictions: {metrics['overall']['avg_confidence_correct']:.4f}")
    print(f"  Incorrect predictions: {metrics['overall']['avg_confidence_incorrect']:.4f}")
    
    print("\n" + "=" * 80)
    print("PER-CLASS METRICS")
    print("=" * 80)
    print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<10}")
    print("-" * 80)
    for cat_name, metrics_dict in metrics['per_class'].items():
        print(f"{cat_name:<15} "
              f"{metrics_dict['precision']:<12.4f} "
              f"{metrics_dict['recall']:<12.4f} "
              f"{metrics_dict['f1']:<12.4f} "
              f"{metrics_dict['support']:<10}")
    
    # Confusion matrix
    print_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        CATEGORIES
    )
    
    # Find misclassifications
    print("=" * 80)
    print("TOP 10 CONFIDENT MISCLASSIFICATIONS")
    print("=" * 80)
    misclass = find_misclassifications(test_data, predictions, confidences, top_n=10)
    
    for i, mc in enumerate(misclass, 1):
        print(f"\n{i}. Confidence: {mc['confidence']:.4f}")
        print(f"   Text: \"{mc['text']}\"")
        print(f"   True: {mc['true_label']} | Predicted: {mc['predicted_label']}")
    
    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "model_path": str(model_path),
        "test_file": str(test_file),
        "num_test_examples": len(test_data),
        "metrics": metrics,
        "top_misclassifications": misclass
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nReport saved to: {output_file}")
    print(f"\n🎯 Model Performance Summary:")
    print(f"   Accuracy: {metrics['overall']['accuracy']:.2%}")
    print(f"   F1 Score: {metrics['overall']['f1']:.4f}")
    print(f"   Confidence: {metrics['overall']['avg_confidence']:.2%}")
    
    if metrics['overall']['accuracy'] >= 0.90:
        print("\n🔥 EXCELLENT! Model is performing very well!")
    elif metrics['overall']['accuracy'] >= 0.80:
        print("\n✅ GOOD! Model performance is solid.")
    elif metrics['overall']['accuracy'] >= 0.70:
        print("\n⚠️  DECENT. Consider training longer or with more data.")
    else:
        print("\n❌ NEEDS IMPROVEMENT. Review data quality and training params.")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model-path",
        default="app/train/models/intent_classifier",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test-file",
        default="app/train/data/classification_test.jsonl",
        help="Path to test data"
    )
    parser.add_argument(
        "--output-file",
        default="app/train/models/intent_classifier/evaluation_report.json",
        help="Path to save evaluation report"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output_file
    )

