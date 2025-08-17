#!/usr/bin/env python3
"""
迁移学习配置文件

包含不同的训练策略和超参数设置：
1. 基础迁移学习配置
2. 渐进式解冻配置
3. 注意力机制配置
4. 多任务学习配置
"""

from typing import Dict, Any, List

# 基础迁移学习配置
BASIC_TRANSFER_CONFIG = {
    'name': 'basic_transfer',
    'description': 'Basic transfer learning: freeze backbone network, only train adapter',
    
    # 模型配置
    'adapter_type': 'basic',
    'freeze_backbone': True,
    'freeze_stages': [0, 1, 2, 3],  # 冻结所有stage
    
    # 训练配置
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-4,  # 降低学习率，更适合迁移学习
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'use_scheduler': True,
    'patience': 15,
    'min_delta': 0.001,
    
    # 数据配置
    'num_classes': 5,
    'input_length': 1000,
    'augmentation': True,
    
    # 保存配置
    'save_dir': 'output/transfer_learning/basic',
    'save_freq': 5,
    'save_best_only': True
}

# 渐进式解冻配置
PROGRESSIVE_UNFREEZING_CONFIG = {
    'name': 'progressive_unfreezing',
    'description': 'Progressive unfreezing: gradually unfreeze different layers of backbone network',
    
    # 模型配置
    'adapter_type': 'basic',
    'freeze_backbone': False,
    'freeze_stages': [0, 1],  # 初始冻结前两个stage
    
    # 训练配置
    'batch_size': 16,  # 较小批次，因为更多参数可训练
    'epochs': 100,
    'lr': 5e-5,  # 较低学习率，因为训练更多参数
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'use_scheduler': True,
    'patience': 20,
    'min_delta': 0.001,
    
    # 渐进式解冻配置
    'unfreezing_schedule': [
        {'epoch': 0, 'freeze_stages': [0, 1, 2, 3]},      # 前10个epoch只训练适配器
        {'epoch': 10, 'freeze_stages': [0, 1, 2]},         # 10-20个epoch解冻最后一个stage
        {'epoch': 20, 'freeze_stages': [0, 1]},             # 20-30个epoch解冻倒数第二个stage
        {'epoch': 30, 'freeze_stages': [0]},                # 30-40个epoch解冻倒数第三个stage
        {'epoch': 40, 'freeze_stages': []},                 # 40个epoch后解冻所有层
    ],
    
    # 数据配置
    'num_classes': 5,
    'input_length': 1000,
    'augmentation': True,
    
    # 保存配置
    'save_dir': 'output/transfer_learning/progressive',
    'save_freq': 5,
    'save_best_only': True
}

# 注意力机制配置
ATTENTION_CONFIG = {
    'name': 'attention_adapter',
    'description': 'Lead adapter with attention mechanism',
    
    # 模型配置
    'adapter_type': 'attention',
    'freeze_backbone': True,
    'freeze_stages': [0, 1, 2, 3],
    
    # 注意力配置
    'attention_heads': 4,
    'attention_dropout': 0.1,
    'use_positional_encoding': True,
    
    # 训练配置
    'batch_size': 32,
    'epochs': 60,
    'lr': 1e-3,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'use_scheduler': True,
    'patience': 20,
    'min_delta': 0.001,
    
    # 数据配置
    'num_classes': 5,
    'input_length': 1000,
    'augmentation': True,
    
    # 保存配置
    'save_dir': 'output/transfer_learning/attention',
    'save_freq': 5,
    'save_best_only': True
}

# 多任务学习配置
MULTITASK_CONFIG = {
    'name': 'multitask_learning',
    'description': 'Multi-task learning: simultaneously predict multiple ECG abnormalities',
    
    # 模型配置
    'adapter_type': 'basic',
    'freeze_backbone': True,
    'freeze_stages': [0, 1, 2, 3],
    
    # 多任务配置
    'tasks': ['rhythm', 'morphology', 'conduction'],
    'task_weights': [1.0, 1.0, 1.0],  # 任务权重
    'use_task_specific_heads': True,
    
    # 训练配置
    'batch_size': 32,
    'epochs': 80,
    'lr': 1e-3,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'use_scheduler': True,
    'patience': 25,
    'min_delta': 0.001,
    
    # 数据配置
    'num_classes': 5,
    'input_length': 1000,
    'augmentation': True,
    
    # 保存配置
    'save_dir': 'output/transfer_learning/multitask',
    'save_freq': 5,
    'save_best_only': True
}

# 高级配置：组合策略
ADVANCED_CONFIG = {
    'name': 'advanced_combined',
    'description': 'Advanced combined strategy: attention mechanism + progressive unfreezing + multi-task learning',
    
    # 模型配置
    'adapter_type': 'attention',
    'freeze_backbone': False,
    'freeze_stages': [0, 1],
    
    # 注意力配置
    'attention_heads': 8,
    'attention_dropout': 0.2,
    'use_positional_encoding': True,
    
    # 渐进式解冻配置
    'unfreezing_schedule': [
        {'epoch': 0, 'freeze_stages': [0, 1, 2, 3], 'lr': 1e-3},
        {'epoch': 15, 'freeze_stages': [0, 1, 2], 'lr': 5e-4},
        {'epoch': 30, 'freeze_stages': [0, 1], 'lr': 2e-4},
        {'epoch': 45, 'freeze_stages': [0], 'lr': 1e-4},
        {'epoch': 60, 'freeze_stages': [], 'lr': 5e-5},
    ],
    
    # 多任务配置
    'tasks': ['rhythm', 'morphology', 'conduction', 'noise'],
    'task_weights': [1.0, 0.8, 0.8, 0.5],
    'use_task_specific_heads': True,
    
    # 训练配置
    'batch_size': 16,
    'epochs': 120,
    'lr': 1e-3,  # 添加缺失的lr键
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'use_scheduler': True,
    'patience': 30,
    'min_delta': 0.001,
    
    # 数据配置
    'num_classes': 5,
    'input_length': 1000,
    'augmentation': True,
    'mixup_alpha': 0.2,
    'cutmix_prob': 0.3,
    
    # 保存配置
    'save_dir': 'output/transfer_learning/advanced',
    'save_freq': 3,
    'save_best_only': True
}

# 快速测试配置
QUICK_TEST_CONFIG = {
    'name': 'quick_test',
    'description': 'Quick test configuration for code functionality verification',
    
    # 模型配置
    'adapter_type': 'basic',
    'freeze_backbone': True,
    'freeze_stages': [0, 1, 2, 3],
    
    # 训练配置
    'batch_size': 8,
    'epochs': 5,
    'lr': 1e-3,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'use_scheduler': False,
    'patience': 3,
    'min_delta': 0.01,
    
    # 数据配置
    'num_classes': 5,
    'input_length': 500,  # 较短序列，快速测试
    'augmentation': False,
    
    # 保存配置
    'save_dir': 'output/transfer_learning/quick_test',
    'save_freq': 1,
    'save_best_only': False
}

# 配置字典
TRANSFER_LEARNING_CONFIGS = {
    'basic': BASIC_TRANSFER_CONFIG,
    'progressive': PROGRESSIVE_UNFREEZING_CONFIG,
    'attention': ATTENTION_CONFIG,
    'multitask': MULTITASK_CONFIG,
    'advanced': ADVANCED_CONFIG,
    'quick_test': QUICK_TEST_CONFIG
}

def get_config(config_name: str) -> Dict[str, Any]:
    """
    获取指定的配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        配置字典
    """
    if config_name not in TRANSFER_LEARNING_CONFIGS:
        raise ValueError(f"未知的配置名称: {config_name}. "
                       f"可用配置: {list(TRANSFER_LEARNING_CONFIGS.keys())}")
    
    return TRANSFER_LEARNING_CONFIGS[config_name].copy()

def list_available_configs() -> List[str]:
    """列出所有可用的配置名称"""
    return list(TRANSFER_LEARNING_CONFIGS.keys())

def print_config_summary(config_name: str):
    """打印配置摘要"""
    config = get_config(config_name)
    
    print(f"\n📋 Configuration Summary: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Adapter type: {config['adapter_type']}")
    print(f"Freezing strategy: {config['freeze_stages']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Training epochs: {config['epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Save directory: {config['save_dir']}")

if __name__ == "__main__":
    # 打印所有可用配置
    print("🚀 可用的迁移学习配置:")
    for name in list_available_configs():
        print_config_summary(name)
        print() 