#!/usr/bin/env python3
"""
Test script to validate sentence generation fixes
"""
import sys
from pathlib import Path

# Add helox root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_synthetic_data import is_valid_sentence, generate_variations

def test_validation():
    """Test the sentence validation function"""
    print("=" * 60)
    print("Testing Sentence Validation")
    print("=" * 60)
    
    # Valid sentences (should pass)
    valid_sentences = [
        "Create a white paper on industry trends",
        "Review the database configuration",
        "Debug the API endpoint",
        "Write documentation for the feature",
        "Test the authentication flow",
        "I need to write a report",
        "Please create the design document"
    ]
    
    # Invalid sentences (should fail)
    invalid_sentences = [
        "A white paper on industry trends write",
        "Inventory data process",
        "The database configuration review",
        "API endpoint debug",
        "Documentation for the feature write",
        "The authentication flow test"
    ]
    
    print("\n✓ Testing VALID sentences:")
    all_valid_passed = True
    for sentence in valid_sentences:
        result = is_valid_sentence(sentence)
        status = "✓ PASS" if result else "✗ FAIL"
        if not result:
            all_valid_passed = False
        print(f"  {status}: '{sentence}'")
    
    print("\n✗ Testing INVALID sentences (should be rejected):")
    all_invalid_passed = True
    for sentence in invalid_sentences:
        result = is_valid_sentence(sentence)
        status = "✓ PASS (rejected)" if not result else "✗ FAIL (accepted)"
        if result:
            all_invalid_passed = False
        print(f"  {status}: '{sentence}'")
    
    print("\n" + "=" * 60)
    if all_valid_passed and all_invalid_passed:
        print("✅ All validation tests passed!")
    else:
        print("❌ Some tests failed")
        if not all_valid_passed:
            print("   - Some valid sentences were incorrectly rejected")
        if not all_invalid_passed:
            print("   - Some invalid sentences were incorrectly accepted")
    print("=" * 60)
    
    return all_valid_passed and all_invalid_passed

def test_variations():
    """Test that generate_variations produces only valid sentences"""
    print("\n" + "=" * 60)
    print("Testing Variation Generation")
    print("=" * 60)
    
    test_templates = [
        "Write a white paper on industry trends",
        "Process inventory data",
        "Review the code changes",
        "Create a new feature"
    ]
    
    all_passed = True
    for template in test_templates:
        print(f"\nTemplate: '{template}'")
        variations = generate_variations(
            template, 
            category="writing_code", 
            num_variations=5,
            use_ollama=False,
            semantic_analyzer=None
        )
        
        print(f"Generated {len(variations)} variations:")
        for i, variation in enumerate(variations, 1):
            is_valid = is_valid_sentence(variation)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {i}. '{variation}'")
            if not is_valid:
                all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All generated variations are valid!")
    else:
        print("❌ Some generated variations are invalid")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    validation_passed = test_validation()
    variations_passed = test_variations()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Validation Tests: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    print(f"Variation Tests: {'✅ PASSED' if variations_passed else '❌ FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if (validation_passed and variations_passed) else 1)
