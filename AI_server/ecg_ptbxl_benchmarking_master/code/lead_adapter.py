#!/usr/bin/env python3
"""
导联适配器模块
将2导联LTDB数据转换为12导联，以便复用现有的XResNet1d101模型

主要功能：
1. 2导联 -> 12导联投影
2. 保持现有模型权重不变
3. 支持冻结/解冻策略
"""

"""
Lead Adapter Module
Converts 2-lead LTDB data to 12-lead data for reuse with existing XResNet1d101 models.

Main Features:
1. 2-lead -> 12-lead projection
2. Maintains existing model weights
3. Supports freezing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LeadAdapter(nn.Module):
    """
    Lead Adapter: Converts 2-lead to 12-lead

    Design Principle:
    - Use 1x1 convolution to expand the number of leads
    - Add residual connections and normalization
    - Support different initialization strategies
    """
    
    def __init__(self, 
                 in_channels: int = 2, 
                 out_channels: int = 12,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 init_method: str = 'kaiming'):
        """
            Args:
            in_channels: Number of input channels (LTDB: 2)
            out_channels: Number of output channels (PTB-XL: 12)
            use_residual: Whether to use residual connections
            use_batch_norm: Whether to use batch normalization
            init_method: Initialization method ('kaiming', 'xavier', 'normal')
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        # 主要的1x1卷积层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        
        # 批归一化层
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None
        
        # 激活函数
        self.activation = nn.ReLU(inplace=True)
        
        # 残差连接的投影层（如果输入输出导联数不同）
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = None
        
        # 初始化权重
        self._init_weights(init_method)
    
    def _init_weights(self, method: str):
        """初始化权重"""
        if method == 'kaiming':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        elif method == 'xavier':
            nn.init.xavier_normal_(self.conv.weight)
        elif method == 'normal':
            nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        
        if self.residual_proj is not None:
            nn.init.kaiming_normal_(self.residual_proj.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, 2, 1000)
        
        Returns:
            output: 输出张量 (batch_size, 12, 1000)
        """
        # 主路径
        out = self.conv(x)
        
        if self.bn is not None:
            out = self.bn(out)
        
        out = self.activation(out)
        
        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            out = out + residual
        
        return out
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return list(self.parameters())
    
    def freeze_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True


class LeadAdapterWithAttention(LeadAdapter):
    """
    带注意力机制的导联适配器
    
    增强功能：
    - 添加通道注意力机制
    - 更好的导联间关系建模
    """
    
    def __init__(self, 
                 in_channels: int = 2, 
                 out_channels: int = 12,
                 reduction: int = 4):
        super().__init__(in_channels, out_channels)
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础转换
        out = super().forward(x)
        
        # 应用通道注意力
        attention_weights = self.channel_attention(out)
        out = out * attention_weights
        
        return out


def create_lead_adapter(adapter_type: str = 'basic',
                       in_channels: int = 2,
                       out_channels: int = 12,
                       **kwargs) -> LeadAdapter:
    """
    工厂函数：创建导联适配器
    
    Args:
        adapter_type: 适配器类型 ('basic', 'attention')
        in_channels: 输入导联数
        out_channels: 输出导联数
        **kwargs: 其他参数
    
    Returns:
        LeadAdapter实例
    """
    if adapter_type == 'basic':
        return LeadAdapter(in_channels, out_channels, **kwargs)
    elif adapter_type == 'attention':
        return LeadAdapterWithAttention(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"不支持的适配器类型: {adapter_type}")


def test_lead_adapter():
    """测试导联适配器"""
    print("🧪 测试导联适配器...")
    
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
    print(f"✅ 导联适配器测试通过！")
    
    # 测试参数数量
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    
    print(f"📊 总参数数: {total_params:,}")
    print(f"📊 可训练参数数: {trainable_params:,}")
    
    return adapter


if __name__ == "__main__":
    # 运行测试
    adapter = test_lead_adapter() 