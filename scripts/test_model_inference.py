#!/usr/bin/env python3
"""
Interactive Model Testing Script
Test your trained model with custom inputs
"""
import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Missing dependencies: {e}")
    print("Install: pip install transformers torch")
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

class TaskClassifier:
    """Simple classifier wrapper"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✓ Model loaded on {self.device}")
    
    def predict(self, text: str):
        """Predict category for a single text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs).item()
            
            # Get all probabilities
            all_probs = probs[0].cpu().numpy()
        
        return {
            "category": CATEGORIES[pred_id],
            "category_id": pred_id,
            "confidence": confidence,
            "all_probabilities": {
                CATEGORIES[i]: float(all_probs[i])
                for i in range(len(CATEGORIES))
            }
        }

def interactive_test(model_path: str = "app/train/models/intent_classifier"):
    """Interactive testing mode"""
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot test: missing dependencies")
        return
    
    # Check model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train a model first: python app/train/scripts/train_intent_classifier.py")
        return
    
    # Load classifier
    classifier = TaskClassifier(model_path)
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODEL TESTING")
    print("=" * 80)
    print("\nTest your model with custom inputs!")
    print("Type 'quit' or 'exit' to stop\n")
    
    # Example inputs
    examples = [
        "Write unit tests for my authentication API",
        "Go for a 5K run in the park",
        "Clean my desk and organize files",
        "Read a research paper on machine learning",
        "Design a logo for the startup",
        "Schedule a meeting with the team",
        "Call my friend to catch up"
    ]
    
    print("Example tasks you can try:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex}")
    print()
    
    while True:
        try:
            # Get input
            user_input = input("Enter task (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Predict
            result = classifier.predict(user_input)
            
            # Display results
            print("\n" + "-" * 80)
            print(f"Task: \"{user_input}\"")
            print(f"\n🎯 Prediction: {result['category'].upper()}")
            print(f"   Confidence: {result['confidence']:.2%}")
            
            # Show top 3 probabilities
            probs_sorted = sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            print(f"\nTop 3 predictions:")
            for i, (cat, prob) in enumerate(probs_sorted[:3], 1):
                bar = "█" * int(prob * 40)
                print(f"  {i}. {cat:<15} {prob:>6.2%} {bar}")
            print("-" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

def batch_test(model_path: str, test_cases: list):
    """Test on batch of predefined cases"""
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot test: missing dependencies")
        return
    
    # Check model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load classifier
    classifier = TaskClassifier(model_path)
    
    print("\n" + "=" * 80)
    print("BATCH MODEL TESTING")
    print("=" * 80)
    print(f"\nTesting {len(test_cases)} examples...\n")
    
    results = []
    for i, text in enumerate(test_cases, 1):
        result = classifier.predict(text)
        results.append(result)
        
        print(f"{i}. \"{text}\"")
        print(f"   → {result['category'].upper()} ({result['confidence']:.2%})")
        print()
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained model")
    parser.add_argument(
        "--model-path",
        default="app/train/models/intent_classifier",
        help="Path to trained model"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode with example tasks"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode with examples
        test_cases = [
            "Write unit tests for my API endpoints",
            "Debug the payment processing bug",
            "Draft a blog post about AI trends",
            "Write documentation for the new feature",
            "Go for a morning jog",
            "Do 30 minutes of yoga",
            "Clean the kitchen and do dishes",
            "Organize my workspace",
            "Study for the certification exam",
            "Read a paper on neural networks",
            "Design a poster for the event",
            "Create a video tutorial",
            "Schedule team meeting",
            "File my taxes",
            "Call mom to wish happy birthday",
            "Plan a dinner party for friends"
        ]
        batch_test(args.model_path, test_cases)
    else:
        # Interactive mode
        interactive_test(args.model_path)

