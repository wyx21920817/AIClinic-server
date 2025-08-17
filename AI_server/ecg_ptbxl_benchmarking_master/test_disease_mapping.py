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
    
    # 测试用例1：正常的疾病映射
    print("测试用例1: 正常疾病映射")
    test_predictions = {
        'NORM': 0.8,
        'ILBBB': 0.6,
        'CLBBB': 0.4,
        'AFIB': 0.3,
        'PAC': 0.2
    }
    
    print("原始预测结果:")
    for disease, prob in test_predictions.items():
        print(f"  {disease}: {prob:.3f}")
    
    mapped_result = map_diseases_to_categories(test_predictions, mapping)
    
    print("\n映射后的结果:")
    for category, prob in mapped_result.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")
    
    print("\n" + "=" * 50)
    
    # 测试用例2：包含未知疾病的映射
    print("测试用例2: 包含未知疾病的映射")
    test_predictions2 = {
        'NORM': 0.7,
        'UNKNOWN_DISEASE': 0.5,
        'RBBB': 0.4,
        'IRBBB': 0.3,
        'STACH': 0.2
    }
    
    print("原始预测结果:")
    for disease, prob in test_predictions2.items():
        print(f"  {disease}: {prob:.3f}")
    
    mapped_result2 = map_diseases_to_categories(test_predictions2, mapping)
    
    print("\n映射后的结果:")
    for category, prob in mapped_result2.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")
    
    print("\n" + "=" * 50)
    
    # 测试用例3：LBBB 合并测试
    print("测试用例3: LBBB 合并测试")
    test_predictions3 = {
        'ILBBB': 0.6,
        'CLBBB': 0.4,
        'LBBB': 0.3,
        'NORM': 0.2
    }
    
    print("原始预测结果:")
    for disease, prob in test_predictions3.items():
        print(f"  {disease}: {prob:.3f}")
    
    mapped_result3 = map_diseases_to_categories(test_predictions3, mapping)
    
    print("\n映射后的结果 (LBBB 应该合并):")
    for category, prob in mapped_result3.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")
    
    print("\n" + "=" * 50)
    
    # 验证目标类别
    print("目标疾病类别列表:")
    target_categories = [
        'NORM', '1dAVB', '2dAVB', '3dAVB', 'AFIB', 'AFLT', 
        'LBBB', 'RBBB', 'STACH', 'PAC', 'PVC'
    ]
    
    for i, category in enumerate(target_categories, 1):
        print(f"  {i:2d}. {category}")
    
    print(f"\n总计: {len(target_categories)} 种目标疾病类别")

def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 测试边界情况")
    print("=" * 50)
    
    mapping = get_disease_mapping()
    
    # 测试空预测结果
    print("测试空预测结果:")
    empty_predictions = {}
    result = map_diseases_to_categories(empty_predictions, mapping)
    print(f"空预测结果映射后: {result}")
    
    # 测试所有概率为0的情况
    print("\n测试所有概率为0的情况:")
    zero_predictions = {'NORM': 0.0, 'LBBB': 0.0}
    result = map_diseases_to_categories(zero_predictions, mapping)
    print(f"零概率预测结果映射后: {result}")
    
    # 测试概率累加
    print("\n测试概率累加:")
    cumulative_predictions = {
        'ILBBB': 0.3,
        'CLBBB': 0.4,
        'LBBB': 0.2
    }
    result = map_diseases_to_categories(cumulative_predictions, mapping)
    print(f"LBBB 相关预测结果:")
    for category, prob in result.items():
        if prob > 0:
            print(f"  {category}: {prob:.3f}")

if __name__ == "__main__":
    try:
        test_disease_mapping()
        test_edge_cases()
        print("\n✅ 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
