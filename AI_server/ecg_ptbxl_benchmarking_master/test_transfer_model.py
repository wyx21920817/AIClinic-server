#!/usr/bin/env python3
"""
测试迁移学习模型
验证导联适配器和模型组合是否工作正常
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加代码路径
sys.path.append('code')

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试迁移学习模型基本功能...")
    
    try:
        # 导入模块
        from code.transfer_model import create_transfer_model
        from code.lead_adapter import LeadAdapter
        
        print("✅ 模块导入成功")
        
        # 检查预训练模型
        pretrained_path = "output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth"
        
        if not os.path.exists(pretrained_path):
            print(f"⚠️ 预训练模型文件不存在: {pretrained_path}")
            print("请确保模型文件存在后再运行测试")
            return False
        
        print("✅ 预训练模型文件存在")
        
        # 创建模型
        model = create_transfer_model(
            backbone_path=pretrained_path,
            num_classes=5,
            adapter_type='basic',
            freeze_backbone=True,
            freeze_stages=[0, 1]
        )
        
        print("✅ 迁移学习模型创建成功")
        
        # 测试前向传播
        batch_size = 4
        input_tensor = torch.randn(batch_size, 2, 1000)
        
        print(f"📥 输入形状: {input_tensor.shape}")
        
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"📤 输出形状: {output.shape}")
        
        # 统计参数
        param_counts = model.count_parameters()
        print(f"📊 参数统计:")
        print(f"   总参数数: {param_counts['total']:,}")
        print(f"   可训练参数数: {param_counts['trainable']:,}")
        print(f"   冻结参数数: {param_counts['frozen']:,}")
        print(f"   适配器参数数: {param_counts['adapter']:,}")
        print(f"   主干网络参数数: {param_counts['backbone']:,}")
        
        print("✅ 迁移学习模型测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n📊 测试LTDB数据加载...")
    
    try:
        # 检查数据文件
        data_dir = Path("data/ltdb_processed")
        
        if not data_dir.exists():
            print(f"❌ 数据目录不存在: {data_dir}")
            return False
        
        # 检查训练数据
        train_data_path = data_dir / "ltdb_train.npy"
        if train_data_path.exists():
            train_data = np.load(str(train_data_path))
            print(f"✅ 训练数据加载成功: {train_data.shape}")
        else:
            print(f"❌ 训练数据文件不存在: {train_data_path}")
            return False
        
        # 检查验证数据
        val_data_path = data_dir / "ltdb_val.npy"
        if val_data_path.exists():
            val_data = np.load(str(val_data_path))
            print(f"✅ 验证数据加载成功: {val_data.shape}")
        else:
            print(f"❌ 验证数据文件不存在: {val_data_path}")
            return False
        
        # 检查测试数据
        test_data_path = data_dir / "ltdb_test.npy"
        if test_data_path.exists():
            test_data = np.load(str(test_data_path))
            print(f"✅ 测试数据加载成功: {test_data.shape}")
        else:
            print(f"❌ 测试数据文件不存在: {test_data_path}")
            return False
        
        print("✅ 所有数据文件加载成功！")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False


def test_adapter_only():
    """测试单独的导联适配器"""
    print("\n🔌 测试导联适配器...")
    
    try:
        from code.lead_adapter import LeadAdapter
        
        # 创建适配器
        adapter = LeadAdapter(in_channels=2, out_channels=12)
        
        # 测试输入
        batch_size = 4
        input_tensor = torch.randn(batch_size, 2, 1000)
        
        print(f"📥 输入形状: {input_tensor.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = adapter(input_tensor)
        
        print(f"📤 输出形状: {output.shape}")
        
        # 测试参数数量
        total_params = sum(p.numel() for p in adapter.parameters())
        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        
        print(f"📊 适配器参数数: {total_params:,}")
        print(f"📊 可训练参数数: {trainable_params:,}")
        
        print("✅ 导联适配器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 导联适配器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始迁移学习模型测试...")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("导联适配器", test_adapter_only),
        ("数据加载", test_data_loading),
        ("迁移学习模型", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 测试: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果总结:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！迁移学习模型准备就绪。")
        print("\n🚀 下一步：运行LTDB自监督学习训练")
        print("   命令: python scripts/train_ltdb_ssl.py")
    else:
        print("⚠️ 部分测试失败，请检查错误信息并修复问题。")
    
    return passed == total


if __name__ == "__main__":
    main() 