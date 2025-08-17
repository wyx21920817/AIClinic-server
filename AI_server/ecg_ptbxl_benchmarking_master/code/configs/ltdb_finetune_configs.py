#!/usr/bin/env python3
"""
LTDB微调配置文件
定义各种微调场景的参数设置
"""

# 基础微调配置
conf_ltdb_basic = {
    'modelname': 'fastai_xresnet1d50',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 30,
        'gradual_unfreezing': True,
        'discriminative_lrs': True,
        'early_stopping': 'valid_loss',
        'bs': 32,  # 较小的批次大小，适合微调
        'lr': 1e-4,  # 较小的学习率
        'wd': 1e-3,  # 权重衰减
        'ps_head': 0.3,  # 头部dropout
        'lin_ftrs_head': [64]  # 头部线性层
    }
}

# 快速微调配置（用于快速验证）
conf_ltdb_quick = {
    'modelname': 'fastai_xresnet1d50',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 10,
        'gradual_unfreezing': False,  # 不进行渐进式解冻
        'discriminative_lrs': False,  # 不使用判别式学习率
        'early_stopping': 'valid_loss',
        'bs': 64,
        'lr': 1e-3,
        'wd': 1e-2
    }
}

# 深度微调配置（用于最佳性能）
conf_ltdb_deep = {
    'modelname': 'fastai_xresnet1d101',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 50,
        'gradual_unfreezing': True,
        'discriminative_lrs': True,
        'early_stopping': 'macro_auc',  # 使用AUC作为早停指标
        'bs': 16,  # 更小的批次大小
        'lr': 5e-5,  # 更小的学习率
        'wd': 5e-4,  # 更小的权重衰减
        'ps_head': 0.5,
        'lin_ftrs_head': [128, 64]
    }
}

# 轻量级模型微调配置
conf_ltdb_lightweight = {
    'modelname': 'fastai_xresnet1d18',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 25,
        'gradual_unfreezing': True,
        'discriminative_lrs': True,
        'early_stopping': 'valid_loss',
        'bs': 64,
        'lr': 2e-4,
        'wd': 1e-3,
        'ps_head': 0.2,
        'lin_ftrs_head': [32]
    }
}

# 高精度微调配置
conf_ltdb_high_precision = {
    'modelname': 'fastai_xresnet1d152',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 100,
        'gradual_unfreezing': True,
        'discriminative_lrs': True,
        'early_stopping': 'fmax',  # 使用Fmax作为早停指标
        'bs': 8,  # 非常小的批次大小
        'lr': 1e-5,  # 非常小的学习率
        'wd': 1e-4,
        'ps_head': 0.6,
        'lin_ftrs_head': [256, 128, 64]
    }
}

# 二分类微调配置
conf_ltdb_binary = {
    'modelname': 'fastai_xresnet1d50',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 20,
        'gradual_unfreezing': True,
        'discriminative_lrs': True,
        'early_stopping': 'valid_loss',
        'bs': 32,
        'lr': 1e-4,
        'wd': 1e-3,
        'ps_head': 0.3,
        'lin_ftrs_head': [64],
        'loss': 'binary_cross_entropy'
    }
}

# 多分类微调配置
conf_ltdb_multiclass = {
    'modelname': 'fastai_xresnet1d50',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 40,
        'gradual_unfreezing': True,
        'discriminative_lrs': True,
        'early_stopping': 'macro_auc',
        'bs': 24,
        'lr': 8e-5,
        'wd': 8e-4,
        'ps_head': 0.4,
        'lin_ftrs_head': [128, 64]
    }
}

# 数据增强微调配置
conf_ltdb_augmented = {
    'modelname': 'fastai_xresnet1d50',
    'modeltype': 'fastai_model',
    'parameters': {
        'epochs_finetuning': 35,
        'gradual_unfreezing': True,
        'discriminative_lrs': True,
        'early_stopping': 'valid_loss',
        'bs': 28,
        'lr': 1e-4,
        'wd': 1e-3,
        'ps_head': 0.3,
        'lin_ftrs_head': [64],
        'chunkify_train': True,  # 启用训练数据分块
        'chunkify_valid': True   # 启用验证数据分块
    }
}

# 所有配置的字典
ltdb_finetune_configs = {
    'basic': conf_ltdb_basic,
    'quick': conf_ltdb_quick,
    'deep': conf_ltdb_deep,
    'lightweight': conf_ltdb_lightweight,
    'high_precision': conf_ltdb_high_precision,
    'binary': conf_ltdb_binary,
    'multiclass': conf_ltdb_multiclass,
    'augmented': conf_ltdb_augmented
}

def get_config(config_name: str):
    """
    获取指定的配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        config: 配置字典
    """
    if config_name not in ltdb_finetune_configs:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(ltdb_finetune_configs.keys())}")
    
    return ltdb_finetune_configs[config_name]

def list_available_configs():
    """
    列出所有可用的配置
    
    Returns:
        config_names: 配置名称列表
    """
    return list(ltdb_finetune_configs.keys())

def print_config_info(config_name: str):
    """
    打印配置信息
    
    Args:
        config_name: 配置名称
    """
    config = get_config(config_name)
    
    print(f"📋 配置: {config_name}")
    print(f"   模型: {config['modelname']}")
    print(f"   类型: {config['modeltype']}")
    print(f"   参数:")
    for param, value in config['parameters'].items():
        print(f"     {param}: {value}")
    print()

if __name__ == "__main__":
    print("🔧 LTDB微调配置")
    print("=" * 40)
    
    # 列出所有可用配置
    print("📋 可用配置:")
    for config_name in list_available_configs():
        print(f"   - {config_name}")
    
    print("\n📊 配置详情:")
    for config_name in list_available_configs():
        print_config_info(config_name) 