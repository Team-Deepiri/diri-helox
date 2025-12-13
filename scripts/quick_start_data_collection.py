#!/usr/bin/env python3
"""
Quick Start Data Collection Script
Run this to immediately start collecting training data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.train.pipelines.data_collection_pipeline import get_data_collector
import json
import sqlite3

def check_data_collection_setup():
    """Check if data collection is set up correctly"""
    print("Checking Deepiri data collection setup...")
    
    try:
        collector = get_data_collector()
        print("✓ Data collector initialized")
        
        # Check database
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Deepiri-specific tables
        required_tables = [
            'task_classifications',  # Tier 1
            'ability_generations',  # Tier 2
            'prompt_to_tasks',  # Main differentiator
            'rl_training_sequences',  # Tier 3
            'productivity_recommendations',  # Tier 3
            'objective_data',  # Gamification
            'odyssey_data',  # Gamification
            'season_data',  # Gamification
            'momentum_events',  # Gamification
            'streak_events',  # Gamification
            'boost_usage',  # Gamification
            'user_interactions'  # General
        ]
        
        print("\nChecking tables:")
        for table in required_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  ✓ {table}: {count} records")
            else:
                print(f"  ✗ {table}: MISSING")
        
        # Summary counts
        print("\nData Summary:")
        cursor.execute("SELECT COUNT(*) FROM task_classifications")
        print(f"  - Intent Classifications (Tier 1): {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM ability_generations")
        print(f"  - Ability Generations (Tier 2): {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM prompt_to_tasks")
        print(f"  - Prompt-to-Tasks: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM rl_training_sequences")
        print(f"  - RL Sequences (Tier 3): {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM objective_data")
        print(f"  - Objectives: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM momentum_events")
        print(f"  - Momentum Events: {cursor.fetchone()[0]}")
        
        cursor.execute("SELECT COUNT(*) FROM boost_usage")
        print(f"  - Boost Usage: {cursor.fetchone()[0]}")
        
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_initial_synthetic_data():
    """Generate initial synthetic data to get started"""
    print("\nGenerating initial synthetic data...")
    
    # Simple synthetic data generator
    abilities = [
        {"id": "summarize_text", "commands": [
            "Can you summarize this?",
            "Give me a summary",
            "What are the key points?",
            "Make this shorter"
        ]},
        {"id": "create_objective", "commands": [
            "Create a task to refactor auth.ts",
            "I need to complete the login feature",
            "Add an objective for testing",
            "Create a goal to improve performance"
        ]},
        {"id": "activate_focus_boost", "commands": [
            "I need to focus",
            "Activate focus mode",
            "Help me concentrate",
            "Boost my focus"
        ]},
        {"id": "generate_code_review", "commands": [
            "Review this code for security issues",
            "Check my code for bugs",
            "Can you review this pull request?",
            "Analyze this code"
        ]},
    ]
    
    collector = get_data_collector()
    
    # Generate classification examples
    for ability in abilities:
        for command in ability["commands"]:
            collector.collect_classification(
                task_text=command,
                description=None,
                prediction={
                    'type': ability["id"],
                    'complexity': 'medium',
                    'estimated_duration': 30
                },
                actual={
                    'type': ability["id"],  # Assume prediction is correct for synthetic
                    'complexity': 'medium',
                    'estimated_duration': 30
                },
                feedback=4.5  # High feedback for synthetic data
            )
    
    print(f"✓ Generated {len(abilities) * 4} synthetic classification examples")

def show_next_steps():
    """Show next steps for data collection"""
    print("\n" + "="*60)
    print("DEEPIRI DATA COLLECTION - NEXT STEPS:")
    print("="*60)
    
    print("\n🎯 PRIORITY 1: PROMPT-TO-TASKS ENGINE (Main Differentiator)")
    print("   - Instrument: /ai/prompt-to-tasks endpoint")
    print("   - Collect: Every prompt → tasks conversion")
    print("   - Track: User acceptance, task completion")
    print("   - See: app/train/examples/deepiri_integration_example.py")
    
    print("\n🎯 PRIORITY 2: THREE-TIER AI SYSTEM")
    print("   Tier 1 (Classification):")
    print("   - Instrument: /intelligence/route-command")
    print("   - Collect: Command → Ability ID predictions")
    print("   - Goal: 100+ examples per ability (50 abilities = 5,000+)")
    print("   ")
    print("   Tier 2 (Generation):")
    print("   - Instrument: /intelligence/generate-ability")
    print("   - Collect: Role-based ability generation")
    print("   - Track: Ability usage, engagement, completion")
    print("   ")
    print("   Tier 3 (RL):")
    print("   - Instrument: /intelligence/productivity-recommendation")
    print("   - Collect: State-action-reward sequences")
    print("   - Track: Recommendation acceptance, actual benefits")
    
    print("\n🎯 PRIORITY 3: GAMIFICATION SYSTEM")
    print("   - Instrument: Engagement service endpoints")
    print("   - Collect: Momentum, Streaks, Boosts, Objectives, Odysseys, Seasons")
    print("   - Use: For reward signal generation in RL")
    
    print("\n📋 QUICK INTEGRATION:")
    print("   1. See examples: app/train/examples/deepiri_integration_example.py")
    print("   2. Add feedback endpoints for user labeling")
    print("   3. Export data weekly: python3 app/train/scripts/export_training_data.py")
    print("   4. Generate synthetic data: python3 app/train/scripts/generate_synthetic_data.py")
    
    print("\n" + "="*60)
    print("DATA COLLECTION IS NOW ACTIVE FOR DEEPIRI PLATFORM!")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("DIRI-CYREX DATA COLLECTION QUICK START")
    print("="*60)
    
    # Check setup
    if not check_data_collection_setup():
        print("\n✗ Setup check failed. Please fix errors above.")
        sys.exit(1)
    
    # Generate initial data
    try:
        generate_initial_synthetic_data()
    except Exception as e:
        print(f"⚠ Warning: Could not generate synthetic data: {e}")
        print("  This is okay - you can generate it later")
    
    # Show next steps
    show_next_steps()

