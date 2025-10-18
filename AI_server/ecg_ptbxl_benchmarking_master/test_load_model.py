#!/usr/bin/env python3
"""
Test load_model_128 function
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    print("🔍 Testing load_model_128 function...")
    
    # Import function
    from extract_and_predict_ecg import load_model_128
    
    print("✅ load_model_128 function imported successfully")
    
    # Test model path
    model_path = os.path.join('output', 'exp0', 'models', 'fastai_xresnet1d101', 'models', 'fastai_xresnet1d101.pth')
    print(f"🔍 Model path: {model_path}")
    
    if os.path.exists(model_path):
        print("✅ Model file exists")
        
        # Test model loading
        print("\n🧪 Testing model loading...")
        model = load_model_128(model_path, n_in=12, n_out=71, device='cpu')
        
        if model is not None:
            print("✅ Model loaded successfully!")
            print(f"📊 Model type: {type(model)}")
        else:
            print("❌ Model loading failed")
    else:
        print("❌ Model file does not exist")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()






