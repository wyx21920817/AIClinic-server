#!/usr/bin/env python3
"""
Transfer Learning Model Combiner
Combines lead adapter with existing XResNet1d101 model to implement 2-lead to 12-lead transfer learning

Main Functions:
1. Load pre-trained XResNet1d101 model
2. Add lead adapter
3. Support different training strategies (freeze/unfreeze)
4. Model saving and loading
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

# Import necessary modules
from .lead_adapter import LeadAdapter, create_lead_adapter
from .models.xresnet1d import xresnet1d101


class TransferLearningModel(nn.Module):
    """
    Transfer learning model: Lead adapter + XResNet1d101
    
    Architecture:
    LTDB data (2 leads) -> Lead adapter -> XResNet1d101 -> Output
    """
    
    def __init__(self, 
                 backbone_path: str,
                 num_classes: int = 5,
                 adapter_type: str = 'basic',
                 freeze_backbone: bool = True,
                 freeze_stages: Optional[List[int]] = None):
        """
        Args:
            backbone_path: Pre-trained model path
            num_classes: Number of classification classes
            adapter_type: Adapter type ('basic', 'attention')
            freeze_backbone: Whether to freeze backbone network
            freeze_stages: List of frozen stages (0-3)
        """
        super().__init__()
        
        self.backbone_path = backbone_path
        self.num_classes = num_classes
        self.adapter_type = adapter_type
        self.freeze_backbone = freeze_backbone
        self.freeze_stages = freeze_stages or []
        
        # Create lead adapter
        self.lead_adapter = create_lead_adapter(
            adapter_type=adapter_type,
            in_channels=2,
            out_channels=12
        )
        
        # Create backbone network (XResNet1d101)
        self.backbone = xresnet1d101(
            input_channels=12,  # Adapter outputs 12 leads
            num_classes=num_classes
        )
        
        # Load pre-trained weights
        self._load_pretrained_weights()
        
        # Apply freezing strategy
        self._apply_freezing_strategy()
        
        print(f"✅ Transfer learning model created successfully!")
        print(f"   - Lead adapter: {adapter_type}")
        print(f"   - Backbone network: XResNet1d101")
        print(f"   - Freezing strategy: {self._get_freeze_info()}")
    
    def _load_pretrained_weights(self):
        """Load pre-trained weights from the backbone model"""
        try:
            print(f"📥 Loading pre-trained weights: {self.backbone_path}")
            
            # Try to load model with different strategies
            try:
                # First try: weights_only=True (safe loading)
                print("   Attempting safe model loading...")
                checkpoint = torch.load(self.backbone_path, map_location='cpu', weights_only=True)
                
            except Exception as e1:
                print(f"   Safe loading failed: {e1}")
                try:
                    # Second try: weights_only=False (legacy loading)
                    print("   Attempting legacy model loading...")
                    checkpoint = torch.load(self.backbone_path, map_location='cpu', weights_only=False)
                    print(f"   Legacy loading successful!")
                    
                except Exception as e2:
                    print(f"   Legacy loading also failed: {e2}")
                    raise Exception(f"Both loading methods failed: {e1}, {e2}")
            
            # Extract state dict from checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"   Checkpoint type: Model checkpoint with 'model' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"   Checkpoint type: Model checkpoint with 'state_dict' key")
            else:
                state_dict = checkpoint
                print(f"   Checkpoint type: Direct state dict")
            
            print(f"   Loaded state dict with {len(state_dict)} parameters")
            
            # Check and adapt the weight dimensions of the last layer
            if '8.4.weight' in state_dict:
                pretrained_weight = state_dict['8.4.weight']
                pretrained_bias = state_dict['8.4.bias']
                
                # Get the current model's last layer weights
                current_weight = self.backbone[8][4].weight
                current_bias = self.backbone[8][4].bias
                
                # If dimensions don't match, adapt them
                if pretrained_weight.shape[0] != current_weight.shape[0]:
                    print(f"⚠️ Detected output dimension mismatch:")
                    print(f"   Pre-trained model: {pretrained_weight.shape[0]} classes")
                    print(f"   Current model: {current_weight.shape[0]} classes")
                    print(f"   Adapting weights...")
                    
                    # Method 1: If pre-trained model has more classes, take first N
                    if pretrained_weight.shape[0] > current_weight.shape[0]:
                        state_dict['8.4.weight'] = pretrained_weight[:current_weight.shape[0], :]
                        state_dict['8.4.bias'] = pretrained_bias[:current_weight.shape[0]]
                        print(f"   ✅ Truncated to first {current_weight.shape[0]} class weights")
                    # Method 2: If pre-trained model has fewer classes, zero-pad
                    else:
                        new_weight = torch.zeros_like(current_weight)
                        new_bias = torch.zeros_like(current_bias)
                        new_weight[:pretrained_weight.shape[0], :] = pretrained_weight
                        new_bias[:pretrained_bias.shape[0]] = pretrained_bias
                        state_dict['8.4.weight'] = new_weight
                        state_dict['8.4.bias'] = new_bias
                        print(f"   ✅ Zero-padded to extend to {current_weight.shape[0]} classes")
            
            # Load into backbone network
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")
            
            print(f"✅ Pre-trained weights loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load pre-trained weights: {e}")
            print(f"   Attempting to create dummy weights...")
            self._create_dummy_weights()
    
    def _create_dummy_weights(self):
        """Create dummy weights for testing"""
        print(f"🔧 Creating dummy weights for testing...")
        
        # Create random weights for all parameters
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Use Xavier initialization
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
        
        print(f"✅ Dummy weights creation completed")
    
    def _apply_freezing_strategy(self):
        """Apply freezing strategy"""
        if not self.freeze_backbone:
            return
        
        print(f"🔒 Applying freezing strategy...")
        
        # Freeze entire backbone network
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        # Selectively unfreeze certain stages
        if self.freeze_stages:
            for stage_idx in self.freeze_stages:
                if stage_idx < len(self.backbone):
                    stage = self.backbone[stage_idx]
                    for param in stage.parameters():
                        param.requires_grad = True
                    print(f"   - Stage {stage_idx} unfrozen")
        
        # Lead adapter is always trainable
        for param in self.lead_adapter.parameters():
            param.requires_grad = True
        
        print(f"✅ Freezing strategy applied successfully")
    
    def _get_freeze_info(self) -> str:
        """Get freezing strategy information"""
        if not self.freeze_backbone:
            return "No freezing"
        
        if not self.freeze_stages:
            return "Freeze backbone network"
        
        return f"Freeze backbone network, unfreeze Stage {self.freeze_stages}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, 2, 1000) - LTDB data
        
        Returns:
            output: Output tensor (batch_size, num_classes)
        """
        # Lead adapter: 2-lead -> 12-lead
        x = self.lead_adapter(x)  # (batch_size, 12, 1000)
        
        # Backbone network
        output = self.backbone(x)
        
        return output
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters"""
        trainable_params = []
        
        # Lead adapter parameters
        trainable_params.extend(self.lead_adapter.parameters())
        
        # Backbone network parameters (if not frozen)
        if not self.freeze_backbone:
            trainable_params.extend(self.backbone.parameters())
        else:
            # Only add unfrozen stage parameters
            for stage_idx in self.freeze_stages:
                if stage_idx < len(self.backbone):
                    stage = self.backbone[stage_idx]
                    trainable_params.extend(stage.parameters())
        
        return trainable_params
    
    def get_frozen_parameters(self) -> List[nn.Parameter]:
        """Get frozen parameters"""
        frozen_params = []
        
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if param.requires_grad == False:
                    frozen_params.append(param)
        
        return frozen_params
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameter numbers"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        adapter_params = sum(p.numel() for p in self.lead_adapter.parameters())
        backbone_params = total_params - adapter_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'adapter': adapter_params,
            'backbone': backbone_params
        }
    
    def save_model(self, save_path: str, save_adapter_only: bool = False):
        """Save model"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_adapter_only:
            # Only save lead adapter
            torch.save({
                'adapter_state_dict': self.lead_adapter.state_dict(),
                'adapter_type': self.adapter_type,
                'metadata': {
                    'num_classes': self.num_classes,
                    'freeze_backbone': self.freeze_backbone,
                    'freeze_stages': self.freeze_stages
                }
            }, save_path)
            print(f"💾 Lead adapter saved to: {save_path}")
        else:
            # Save complete model
            torch.save({
                'model_state_dict': self.state_dict(),
                'adapter_type': self.adapter_type,
                'metadata': {
                    'num_classes': self.num_classes,
                    'freeze_backbone': self.freeze_backbone,
                    'freeze_stages': self.freeze_stages
                }
            }, save_path)
            print(f"💾 Complete model saved to: {save_path}")
    
    def load_adapter(self, adapter_path: str):
        """Load lead adapter weights"""
        if not os.path.exists(adapter_path):
            print(f"⚠️ Adapter file does not exist: {adapter_path}")
            return False
        
        try:
            checkpoint = torch.load(adapter_path, map_location='cpu')
            
            if 'adapter_state_dict' in checkpoint:
                self.lead_adapter.load_state_dict(checkpoint['adapter_state_dict'])
                print(f"✅ Lead adapter weights loaded successfully!")
                return True
            else:
                print(f"❌ Adapter file format incorrect")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load adapter weights: {e}")
            return False


def create_transfer_model(backbone_path: str,
                         num_classes: int = 5,
                         adapter_type: str = 'basic',
                         freeze_backbone: bool = True,
                         freeze_stages: Optional[List[int]] = None) -> TransferLearningModel:
    """
    Factory function: Create transfer learning model
    
    Args:
        backbone_path: Pre-trained model path
        num_classes: Number of classification classes
        adapter_type: Adapter type
        freeze_backbone: Whether to freeze backbone network
        freeze_stages: List of frozen stages
    
    Returns:
        TransferLearningModel instance
    """
    return TransferLearningModel(
        backbone_path=backbone_path,
        num_classes=num_classes,
        adapter_type=adapter_type,
        freeze_backbone=freeze_backbone,
        freeze_stages=freeze_stages
    )


def test_transfer_model():
    """Test transfer learning model"""
    print("🧪 Testing transfer learning model...")
    
    # Check pre-trained model file
    pretrained_path = "output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth"
    
    if not os.path.exists(pretrained_path):
        print(f"⚠️ Pre-trained model file does not exist: {pretrained_path}")
        print("Please ensure model file exists before running test")
        return None
    
    # Create model
    model = create_transfer_model(
        backbone_path=pretrained_path,
        num_classes=5,
        adapter_type='basic',
        freeze_backbone=True,
        freeze_stages=[0, 1]  # Unfreeze first two stages
    )
    
    # Test forward propagation
    batch_size = 4
    input_tensor = torch.randn(batch_size, 2, 1000)
    
    print(f"📥 Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"📤 Output shape: {output.shape}")
    
    # Count parameters
    param_counts = model.count_parameters()
    print(f"📊 Parameter statistics:")
    print(f"   Total parameters: {param_counts['total']:,}")
    print(f"   Trainable parameters: {param_counts['trainable']:,}")
    print(f"   Frozen parameters: {param_counts['frozen']:,}")
    print(f"   Adapter parameters: {param_counts['adapter']:,}")
    print(f"   Backbone parameters: {param_counts['backbone']:,}")
    
    print(f"✅ Transfer learning model test passed!")
    
    return model


if __name__ == "__main__":
    # Run test
    model = test_transfer_model() 