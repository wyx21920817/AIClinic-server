#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test disease mapping functionality
"""

import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from extract_and_predict_ecg import get_disease_mapping, map_diseases_to_categories

def test_disease_mapping():
    """Testing disease mapping capabilities"""
    print("🧪 Testing disease mapping capabilities")
    print("=" * 50)
    
    # Get disease mapping
    mapping = get_disease_mapping()
    print("Disease Mapping Dictionary:")
    for original, target in mapping.items():
        print(f"  {original:20s} -> {target}")
    
    print("\n" + "=" * 50)
    
    # Test case 1: Normal disease mapping
    print("Test case 1: Normal disease mapping")
    test_predictions = {
        'NORM': 0.8,
        'ILBBB': 0.6,
        'CLBBB': 0.4,
        'AFIB': 0.3,
        'PAC': 0.2
    }
    
    print("Original prediction results:")
    for disease, prob in test_predictions.items():
        print(f"  {disease}: {prob:.3f}")
    
    mapped_result = map_diseases_to_categories(test_predictions, mapping)
    
    print("\nMapped results:")
    for category, prob in mapped_result.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")
    
    print("\n" + "=" * 50)
    
    # Test case 2: Mapping with unknown diseases
    print("Test case 2: Mapping with unknown diseases")
    test_predictions2 = {
        'NORM': 0.7,
        'UNKNOWN_DISEASE': 0.5,
        'RBBB': 0.4,
        'IRBBB': 0.3,
        'STACH': 0.2
    }
    
    print("Original prediction results:")
    for disease, prob in test_predictions2.items():
        print(f"  {disease}: {prob:.3f}")
    
    mapped_result2 = map_diseases_to_categories(test_predictions2, mapping)
    
    print("\nMapped results:")
    for category, prob in mapped_result2.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")
    
    print("\n" + "=" * 50)
    
    # Test case 3: LBBB merge test
    print("Test case 3: LBBB merge test")
    test_predictions3 = {
        'ILBBB': 0.6,
        'CLBBB': 0.4,
        'LBBB': 0.3,
        'NORM': 0.2
    }
    
    print("Original prediction results:")
    for disease, prob in test_predictions3.items():
        print(f"  {disease}: {prob:.3f}")
    
    mapped_result3 = map_diseases_to_categories(test_predictions3, mapping)
    
    print("\nMapped results (LBBB should be merged):")
    for category, prob in mapped_result3.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")
    
    print("\n" + "=" * 50)
    
    # Verify target categories
    print("Target disease category list:")
    target_categories = [
        'NORM', '1dAVB', '2dAVB', '3dAVB', 'AFIB', 'AFLT', 
        'LBBB', 'RBBB', 'STACH', 'PAC', 'PVC'
    ]
    
    for i, category in enumerate(target_categories, 1):
        print(f"  {i:2d}. {category}")
    
    print(f"\nTotal: {len(target_categories)} target disease categories")

def test_edge_cases():
    """Test edge cases"""
    print("\n🧪 Testing edge cases")
    print("=" * 50)
    
    mapping = get_disease_mapping()
    
    # Test empty prediction results
    print("Testing empty prediction results:")
    empty_predictions = {}
    result = map_diseases_to_categories(empty_predictions, mapping)
    print(f"Empty prediction results after mapping: {result}")
    
    # Test all probabilities are 0
    print("\nTesting all probabilities are 0:")
    zero_predictions = {'NORM': 0.0, 'LBBB': 0.0}
    result = map_diseases_to_categories(zero_predictions, mapping)
    print(f"Zero probability prediction results after mapping: {result}")
    
    # Test probability accumulation
    print("\nTesting probability accumulation:")
    cumulative_predictions = {
        'ILBBB': 0.3,
        'CLBBB': 0.4,
        'LBBB': 0.2
    }
    result = map_diseases_to_categories(cumulative_predictions, mapping)
    print(f"LBBB related prediction results:")
    for category, prob in result.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")

if __name__ == "__main__":
    try:
        test_disease_mapping()
        test_edge_cases()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
