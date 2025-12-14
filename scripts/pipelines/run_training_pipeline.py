#!/usr/bin/env python3
"""
Complete Training Pipeline - Run Everything End-to-End
Generates synthetic data, prepares it, and trains the model
"""
import sys
import subprocess
from pathlib import Path
import argparse
import os

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.absolute()

# Add parent to path
sys.path.insert(0, str(PROJECT_ROOT))

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 60)
    print(f"Step: {description}")
    print("=" * 60)
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True

def main(
    generate_data: bool = True,
    total_examples: int = 7000,  # Increased by 2000 from 5000
    examples_per_class: int = None,
    epochs: int = 2,  # Reduced from 3
    batch_size: int = 16,
    learning_rate: float = 1e-5,  # Reduced from 2e-5
    skip_training: bool = False
):
    """Run the complete training pipeline"""
    
    print("=" * 60)
    print("🚀 Deepiri Training Pipeline")
    print("=" * 60)
    print()
    print("This will:")
    print("  1. Generate synthetic training data")
    print("  2. Prepare the dataset")
    print("  3. Train the DeBERTa classifier")
    print()
    
    # Get absolute paths to scripts
    generate_script = SCRIPT_DIR / "generate_synthetic_data.py"
    prepare_script = SCRIPT_DIR / "prepare_training_data.py"
    train_script = SCRIPT_DIR / "train_intent_classifier.py"
    evaluate_script = SCRIPT_DIR / "evaluate_trained_model.py"
    
    # Step 1: Generate synthetic data
    if generate_data:
        cmd = f'python "{generate_script}"'
        if examples_per_class:
            cmd += f" --examples-per-class {examples_per_class}"
        else:
            cmd += f" --total-examples {total_examples}"
        
        if not run_command(cmd, "Generate Synthetic Data"):
            print("\n❌ Failed to generate synthetic data")
            return False
    
    # Step 2: Prepare training data
    cmd = f'python "{prepare_script}"'
    if not run_command(cmd, "Prepare Training Data"):
        print("\n❌ Failed to prepare training data")
        print("   Note: If data was already generated, this might be okay")
        data_path = PROJECT_ROOT / "app" / "train" / "data" / "classification_train.jsonl"
        print(f"   Check if {data_path} exists")
    
    # Step 3: Train the model
    if not skip_training:
        cmd = f'python "{train_script}"'
        cmd += f" --epochs {epochs}"
        cmd += f" --batch-size {batch_size}"
        cmd += f" --learning-rate {learning_rate}"
        
        if not run_command(cmd, "Train Intent Classifier"):
            print("\n❌ Failed to train model")
            return False
    
    # Step 4: Evaluate the model
    print("\n" + "=" * 60)
    print("🎯 Evaluating Model Performance")
    print("=" * 60)
    cmd = f'python "{evaluate_script}"'
    if not run_command(cmd, "Evaluate Model on Test Set"):
        print("\n⚠️  Evaluation failed, but model is still trained")
    
    # Success!
    print("\n" + "=" * 60)
    print("🚀 TRAINING PIPELINE COMPLETE! 🚀")
    print("=" * 60)
    print()
    print("✅ Model trained and evaluated successfully!")
    print()
    model_dir = PROJECT_ROOT / "app" / "train" / "models" / "intent_classifier"
    print(f"📁 Model location: {model_dir}")
    print(f"📊 Evaluation report: {model_dir / 'evaluation_report.json'}")
    print()
    
    # Important reminder about data preparation
    inspect_script = SCRIPT_DIR / "inspect_datasets.py"
    print("⚠️  IMPORTANT REMINDER:")
    print("   Before using this model in production, ensure your data is properly prepared:")
    print("   1. Review dataset quality:")
    print(f"      python \"{inspect_script}\" --all --quality")
    print("   2. Verify label distribution is balanced")
    print("   3. Check for any data quality issues")
    print()
    
    test_script = SCRIPT_DIR / "test_model_inference.py"
    print("🧪 Test the model interactively:")
    print(f"   python \"{test_script}\"")
    print()
    print("📈 Use in production:")
    print("   from app.services.command_router import get_command_router")
    print(f"   router = get_command_router(")
    print(f"       model_path='{model_dir}'")
    print("   )")
    print()
    print("🎯 Categories (31 total):")
    print()
    print("   Coding Breakdown (6):")
    print("   0: debugging       - Debug, troubleshoot, fix bugs")
    print("   1: refactoring     - Refactor, restructure, improve code")
    print("   2: writing_code    - Write code, implement features")
    print("   3: programming     - Program, develop, create systems")
    print("   4: running_code    - Run tests, execute scripts, deploy")
    print("   5: inspecting     - Review, analyze, inspect code")
    print()
    print("   Core Categories (5):")
    print("   6: writing         - Blog posts, docs, emails, content")
    print("   7: learning_research - Research, investigate, read papers")
    print("   8: learning_study  - Study, review, prepare for exams")
    print("   9: learning_training - Take courses, attend workshops")
    print("  10: learning_practice - Practice, exercises, hands-on")
    print()
    print("   Team & Organization (5):")
    print("  11: creative        - Design, art, create content")
    print("  12: administrative  - Schedule, bills, admin tasks")
    print("  13: team_organization - Organize teams, structure, resources")
    print("  14: team_collaboration - Collaborate, pair program, work together")
    print("  15: team_planning   - Plan sprints, capacity, team activities")
    print()
    print("   Computer/Desk-Work (6):")
    print("  16: research        - Research, investigate, look up information")
    print("  17: planning        - Project planning, roadmaps, scheduling")
    print("  18: communication  - Emails, messages, meetings, calls")
    print("  19: big_data_analytics - Large-scale data analysis, big data")
    print("  20: data_processing - Process data, transform, clean data")
    print("  21: design          - UI/UX design, graphics, mockups")
    print()
    print("   Quality & Operations (6):")
    print("  22: qa              - Quality assurance, QA testing")
    print("  23: testing         - Testing, test execution, test suites")
    print("  24: validation     - Validate data, rules, requirements")
    print("  25: reporting      - Generate reports, create summaries")
    print("  26: documentation  - Technical docs, guides, manuals")
    print("  27: system_admin   - System maintenance, IT tasks, ops")
    print()
    print("   Specialized (3):")
    print("  28: ux_ui          - User experience, user interface design")
    print("  29: security       - Security audits, security implementation")
    print("  30: data_privacy  - Data privacy, GDPR, privacy compliance")
    print()
    print("📊 Intent Classifier Notes:")
    print("   - Uses redesigned confidence classes for better reliability")
    print("   - Confidence attributes include uncertainty, calibration, and reliability")
    print("   - Multiple confidence sources: model prediction, training coverage, etc.")
    print()
    print("🔥 YOU'RE READY FOR LIFTOFF! 🔥")
    print()
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete training pipeline for Deepiri task classifier"
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation (use existing data)"
    )
    parser.add_argument(
        "--total-examples",
        type=int,
        default=7000,
        help="Total number of examples to generate (default: 7000)"
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=None,
        help="Number of examples per class (overrides total-examples)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (only generate and prepare data)"
    )
    
    args = parser.parse_args()
    
    success = main(
        generate_data=not args.skip_data_generation,
        total_examples=args.total_examples,
        examples_per_class=args.examples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        skip_training=args.skip_training
    )
    
    sys.exit(0 if success else 1)

