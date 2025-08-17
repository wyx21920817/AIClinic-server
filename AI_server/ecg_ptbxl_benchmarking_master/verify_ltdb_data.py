#!/usr/bin/env python3
"""
验证LTDB处理后数据的质量
检查数据形状、统计信息、可视化等
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def verify_ltdb_data():
    """验证LTDB处理后的数据"""
    data_dir = Path("data/ltdb_processed")
    
    print("🔍 验证LTDB处理后的数据...")
    
    # 加载元数据
    with open(data_dir / "ltdb_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 元数据:")
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    # 加载训练、验证、测试数据
    print(f"\n📁 加载数据集...")
    train_data = np.load(data_dir / "ltdb_train.npy")
    val_data = np.load(data_dir / "ltdb_val.npy")
    test_data = np.load(data_dir / "ltdb_test.npy")
    
    print(f"📊 数据形状:")
    print(f"   训练集: {train_data.shape}")
    print(f"   验证集: {val_data.shape}")
    print(f"   测试集: {test_data.shape}")
    
    # 检查数据统计信息
    print(f"\n📈 数据统计信息:")
    for name, data in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        print(f"   {name}:")
        print(f"     均值: {data.mean():.4f}")
        print(f"     标准差: {data.std():.4f}")
        print(f"     最小值: {data.min():.4f}")
        print(f"     最大值: {data.max():.4f}")
    
    # 检查数据范围（标准化后应该在合理范围内）
    print(f"\n✅ 数据质量检查:")
    
    # 检查是否包含NaN或无穷大
    has_nan = np.isnan(train_data).any() or np.isnan(val_data).any() or np.isnan(test_data).any()
    has_inf = np.isinf(train_data).any() or np.isinf(val_data).any() or np.isinf(test_data).any()
    
    print(f"   包含NaN: {'❌' if has_nan else '✅'}")
    print(f"   包含无穷大: {'❌' if has_inf else '✅'}")
    
    # 检查标准化效果（均值应该接近0，标准差接近1）
    train_mean = train_data.mean()
    train_std = train_data.std()
    
    mean_ok = abs(train_mean) < 0.1
    std_ok = 0.8 < train_std < 1.2
    
    print(f"   均值接近0 (|{train_mean:.4f}| < 0.1): {'✅' if mean_ok else '❌'}")
    print(f"   标准差接近1 ({train_std:.4f}): {'✅' if std_ok else '❌'}")
    
    # 检查采样频率
    expected_samples = metadata['target_fs'] * metadata['win_sec']
    actual_samples = train_data.shape[2]
    
    samples_ok = actual_samples == expected_samples
    print(f"   采样点数正确 ({actual_samples} = {expected_samples}): {'✅' if samples_ok else '❌'}")
    
    # 检查导联数
    expected_leads = 2  # 我们统一为2导联
    actual_leads = train_data.shape[1]
    
    leads_ok = actual_leads == expected_leads
    print(f"   导联数正确 ({actual_leads} = {expected_leads}): {'✅' if leads_ok else '❌'}")
    
    # 可视化几个样本
    print(f"\n🎨 可视化样本...")
    try:
        # 选择几个样本进行可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('LTDB处理后的ECG样本 (2导联)', fontsize=16)
        
        # 训练集样本
        for i in range(3):
            sample_idx = i * 1000  # 选择不同的样本
            if sample_idx < train_data.shape[0]:
                axes[0, i].plot(train_data[sample_idx, 0, :], label='ECG1', alpha=0.8)
                axes[0, i].plot(train_data[sample_idx, 1, :], label='ECG2', alpha=0.8)
                axes[0, i].set_title(f'训练集样本 {sample_idx}')
                axes[0, i].set_xlabel('样本点')
                axes[0, i].set_ylabel('幅值')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
        
        # 验证集样本
        for i in range(3):
            sample_idx = i * 1000
            if sample_idx < val_data.shape[0]:
                axes[1, i].plot(val_data[sample_idx, 0, :], label='ECG1', alpha=0.8)
                axes[1, i].plot(val_data[sample_idx, 1, :], label='ECG2', alpha=0.8)
                axes[1, i].set_title(f'验证集样本 {sample_idx}')
                axes[1, i].set_xlabel('样本点')
                axes[1, i].set_ylabel('幅值')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ltdb_samples_visualization.png', dpi=150, bbox_inches='tight')
        print(f"   💾 可视化图已保存为: ltdb_samples_visualization.png")
        
    except Exception as e:
        print(f"   ⚠️ 可视化失败: {e}")
    
    print(f"\n🎉 数据验证完成！")
    
    # 总结
    all_checks = [not has_nan, not has_inf, mean_ok, std_ok, samples_ok, leads_ok]
    passed_checks = sum(all_checks)
    total_checks = len(all_checks)
    
    print(f"\n📊 验证结果: {passed_checks}/{total_checks} 项检查通过")
    
    if passed_checks == total_checks:
        print("✅ 所有检查都通过！数据质量良好，可以进入下一步。")
    else:
        print("⚠️ 部分检查未通过，请检查数据处理过程。")

if __name__ == "__main__":
    verify_ltdb_data() 