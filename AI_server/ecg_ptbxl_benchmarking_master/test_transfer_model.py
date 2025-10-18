#!/usr/bin/env python3
"""
Test transfer learning model
Verify if lead adapter and model combination work properly
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add code path
sys.path.append('code')

def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Testing transfer learning model basic functionality...")
    
    try:
        # Import modules
        from code.transfer_model import create_transfer_model
        from code.lead_adapter import LeadAdapter
        
        print("✅ Modules imported successfully")
        
        # Check pretrained model
        pretrained_path = "output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth"
        
        if not os.path.exists(pretrained_path):
            print(f"⚠️ Pretrained model file does not exist: {pretrained_path}")
            print("Please ensure the model file exists before running the test")
            return False
        
        print("✅ Pretrained model file exists")
        
        # Create model
        model = create_transfer_model(
            backbone_path=pretrained_path,
            num_classes=5,
            adapter_type='basic',
            freeze_backbone=True,
            freeze_stages=[0, 1]
        )
        
        print("✅ Transfer learning model created successfully")
        
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
        
        print("✅ Transfer learning model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading"""
    print("\n📊 Testing LTDB data loading...")
    
    try:
        # Check data files
        data_dir = Path("data/ltdb_processed")
        
        if not data_dir.exists():
            print(f"❌ Data directory does not exist: {data_dir}")
            return False
        
        # Check training data
        train_data_path = data_dir / "ltdb_train.npy"
        if train_data_path.exists():
            train_data = np.load(str(train_data_path))
            print(f"✅ Training data loaded successfully: {train_data.shape}")
        else:
            print(f"❌ Training data file does not exist: {train_data_path}")
            return False
        
        # Check validation data
        val_data_path = data_dir / "ltdb_val.npy"
        if val_data_path.exists():
            val_data = np.load(str(val_data_path))
            print(f"✅ Validation data loaded successfully: {val_data.shape}")
        else:
            print(f"❌ Validation data file does not exist: {val_data_path}")
            return False
        
        # Check test data
        test_data_path = data_dir / "ltdb_test.npy"
        if test_data_path.exists():
            test_data = np.load(str(test_data_path))
            print(f"✅ Test data loaded successfully: {test_data.shape}")
        else:
            print(f"❌ Test data file does not exist: {test_data_path}")
            return False
        
        print("✅ All data files loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_adapter_only():
    """Test standalone lead adapter"""
    print("\n🔌 Testing lead adapter...")
    
    try:
        from code.lead_adapter import LeadAdapter
        
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
        
        # Test parameter count
        total_params = sum(p.numel() for p in adapter.parameters())
        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        
        print(f"📊 Adapter parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        print("✅ Lead adapter test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Lead adapter test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Starting transfer learning model testing...")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Lead Adapter", test_adapter_only),
        ("Data Loading", test_data_loading),
        ("Transfer Learning Model", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} encountered exception: {e}")
            results.append((test_name, False))
    
    # Summarize test results
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Transfer learning model is ready.")
        print("\n🚀 Next step: Run LTDB self-supervised learning training")
        print("   Command: python scripts/train_ltdb_ssl.py")
    else:
        print("⚠️ Some tests failed, please check error messages and fix issues.")
    
    return passed == total


if __name__ == "__main__":
    main() 