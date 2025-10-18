#!/usr/bin/env python3
"""
Lead Adapter Module
Converts 2-lead LTDB data to 12-lead data for reuse with existing XResNet1d101 models

Main Features:
1. 2-lead -> 12-lead projection
2. Maintains existing model weights
3. Supports freezing/unfreezing strategies
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
        
        # Main 1x1 convolutional layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        
        # Batch normalization layer
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None
        
        # Activation function
        self.activation = nn.ReLU(inplace=True)
        
        # Residual connection projection layer (if input and output lead numbers differ)
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = None
        
        # Initialize weights
        self._init_weights(init_method)
    
    def _init_weights(self, method: str):
        """Initialize weights"""
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
            x: Input tensor (batch_size, 2, 1000)
        
        Returns:
            output: Output tensor (batch_size, 12, 1000)
        """
        # Main path
        out = self.conv(x)
        
        if self.bn is not None:
            out = self.bn(out)
        
        out = self.activation(out)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            out = out + residual
        
        return out
    
    def get_trainable_parameters(self):
        """Get trainable parameters"""
        return list(self.parameters())
    
    def freeze_parameters(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


class LeadAdapterWithAttention(LeadAdapter):
    """
    Lead adapter with attention mechanism
    
    Enhanced features:
    - Add channel attention mechanism
    - Better modeling of inter-lead relationships
    """
    
    def __init__(self, 
                 in_channels: int = 2, 
                 out_channels: int = 12,
                 reduction: int = 4):
        super().__init__(in_channels, out_channels)
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Basic transformation
        out = super().forward(x)
        
        # Apply channel attention
        attention_weights = self.channel_attention(out)
        out = out * attention_weights
        
        return out


def create_lead_adapter(adapter_type: str = 'basic',
                       in_channels: int = 2,
                       out_channels: int = 12,
                       **kwargs) -> LeadAdapter:
    """
    Factory function: Create lead adapter
    
    Args:
        adapter_type: Adapter type ('basic', 'attention')
        in_channels: Number of input leads
        out_channels: Number of output leads
        **kwargs: Other parameters
    
    Returns:
        LeadAdapter instance
    """
    if adapter_type == 'basic':
        return LeadAdapter(in_channels, out_channels, **kwargs)
    elif adapter_type == 'attention':
        return LeadAdapterWithAttention(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")


def test_lead_adapter():
    """Test lead adapter"""
    print("🧪 Testing lead adapter...")
    
    # Create adapter
    adapter = LeadAdapter(in_channels=2, out_channels=12)
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 2, 1000)
    
    print(f"📥 Input shape: {input_tensor.shape}")
    
    # Forward propagation
    with torch.no_grad():
        output = adapter(input_tensor)
    
    print(f"📤 Output shape: {output.shape}")
    print(f"✅ Lead adapter test passed!")
    
    # Test parameter count
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    return adapter


if __name__ == "__main__":
    # Run test
    adapter = test_lead_adapter() 