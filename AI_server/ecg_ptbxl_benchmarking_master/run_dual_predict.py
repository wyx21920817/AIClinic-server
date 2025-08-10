#!/usr/bin/env python3
"""
双模型预测脚本：支持超类（5分类）和子类（71分类）预测
Dual Model Prediction Script: Supports superclass (5-class) and subclass (71-class) prediction
"""

import sys
import os
import pickle
import torch
import numpy as np
import pydicom
import torch.nn.functional as F

if len(sys.argv) < 2:
    print("Usage: python3 run_dual_predict.py <dicom_path> [class_type]")
    print("class_type: superclass (5-class) or subclass (71-class), default is superclass")
    print("\nExamples:")
    print("  python3 run_dual_predict.py test_00045_lr.dcm superclass  # Predict 5 superclasses")
    print("  python3 run_dual_predict.py test_00045_lr.dcm subclass    # Predict 71 subclasses")
    print("  python3 run_dual_predict.py test_00045_lr.dcm             # Default superclass prediction")
    sys.exit(1)

dicom_path = sys.argv[1]
class_type = sys.argv[2] if len(sys.argv) > 2 else "superclass"

print(f"🚀 Starting prediction for DICOM: {dicom_path}")
print(f"🎯 Prediction type: {class_type}")

try:
    # Read classify_single_ecg_ft.py content (superclass prediction)
    with open('classify_single_ecg_ft.py', 'r', encoding='utf-8') as f:
        ft_code = f.read()

    # Replace relative imports
    ft_code = ft_code.replace('from .code.models.xresnet1d import', 'from code.models.xresnet1d import')

    # Execute classify_single_ecg_ft.py code
    ft_globals = {
        '__file__': 'classify_single_ecg_ft.py',
        '__name__': 'classify_single_ecg_ft_temp'
    }
    exec(ft_code, ft_globals)
    print("✅ Superclass prediction module loaded successfully")

    # Read classify_single_ecg128.py content (subclass prediction)
    with open('classify_single_ecg128.py', 'r', encoding='utf-8') as f:
        code_128 = f.read()

    # Replace relative imports
    code_128 = code_128.replace('from .code.models.xresnet1d import', 'from code.models.xresnet1d import')

    # Execute classify_single_ecg128.py code
    globals_128 = {
        '__file__': 'classify_single_ecg128.py',
        '__name__': 'classify_single_ecg128_temp'
    }
    exec(code_128, globals_128)
    print("✅ Subclass prediction module loaded successfully")

    # Now directly call the corresponding functions
    if class_type == "superclass":
        print("\n🔍 Using superclass prediction model...")
        
        # Load data required for superclass prediction
        ft_data_dir = os.path.join('output', 'exp0_ft', 'data')
        scaler_path = os.path.join(ft_data_dir, 'standard_scaler.pkl')
        mlb_path = os.path.join(ft_data_dir, 'mlb.pkl')
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        
        print(f"🔢 Loaded {len(mlb.classes_)} superclasses: {mlb.classes_}")
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join('output', 'exp0_ft', 'models', 'fastai_xresnet1d101_finetuned_manual.pth')
        model = ft_globals['load_model'](model_path, n_in=12, n_out=len(mlb.classes_), device=device)
        
        if model is None:
            raise RuntimeError("Superclass model loading failed")
        
        # Extract ECG signal
        ds = pydicom.dcmread(dicom_path, force=True)
        if not hasattr(ds, 'WaveformSequence'):
            raise ValueError("WaveformSequence not found in DICOM file")
        
        wf = ds.WaveformSequence[0]
        data = wf.WaveformData
        n_s = wf.NumberOfWaveformSamples
        n_c = wf.NumberOfWaveformChannels
        print(f"📊 ECG signal: {n_s} samples, {n_c} channels")
        
        arr = np.frombuffer(data, dtype=np.int16)
        if arr.size != n_s * n_c:
            arr = np.pad(arr, (0, n_s*n_c - arr.size))[:n_s*n_c]
        
        ecg = arr.reshape((n_s, n_c))
        sig = ecg.astype(np.float32).T / 1000.0
        sig, _ = ft_globals['preprocess_signal'](sig, scaler)
        x = torch.from_numpy(sig).unsqueeze(0).to(device)
        
        # Prediction
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # Output results
        idxs = np.argsort(probs)[::-1][:5]
        result = {}
        print(f"\n🏥 Superclass prediction results:")
        for i in idxs:
            result[mlb.classes_[i]] = float(probs[i])
            print(f"  {mlb.classes_[i]:15s} prob={probs[i]:.3f}")
        
    else:
        print("\n🔍 Using subclass prediction model...")
        
        # Load data required for subclass prediction
        orig_data_dir = os.path.join('output', 'exp0', 'data')
        scaler_path = os.path.join(orig_data_dir, 'standard_scaler.pkl')
        mlb_path = os.path.join(orig_data_dir, 'mlb.pkl')
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        print(f"🔢 Loaded {len(mlb.classes_)} subclasses")
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join('output', 'exp0', 'models', 'fastai_xresnet1d101', 'models', 'fastai_xresnet1d101.pth')
        model = globals_128['load_model'](model_path, n_in=12, n_out=len(mlb.classes_), device=device)
        
        if model is None:
            raise RuntimeError("Subclass model loading failed")
        
        # Extract ECG signal
        ds = pydicom.dcmread(dicom_path, force=True)
        if not hasattr(ds, 'WaveformSequence'):
            raise ValueError("WaveformSequence not found in DICOM file")
        
        wf = ds.WaveformSequence[0]
        data = wf.WaveformData
        n_s = wf.NumberOfWaveformSamples
        n_c = wf.NumberOfWaveformChannels
        print(f"📊 ECG signal: {n_s} samples, {n_c} channels")
        
        arr = np.frombuffer(data, dtype=np.int16)
        if arr.size != n_s * n_c:
            arr = np.pad(arr, (0, n_s*n_c - arr.size))[:n_s*n_c]
        
        ecg = arr.reshape((n_s, n_c))
        sig = ecg.astype(np.float32).T / 1000.0
        sig, _ = globals_128['preprocess_signal'](sig, scaler)
        x = torch.from_numpy(sig).unsqueeze(0).to(device)
        
        # Prediction
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # Output results
        idxs = np.argsort(probs)[::-1][:5]
        result = {}
        print(f"\n🏥 Subclass prediction results:")
        for i in idxs:
            result[mlb.classes_[i]] = float(probs[i])
            print(f"  {mlb.classes_[i]:15s} prob={probs[i]:.3f}")
    
    print(f"\n✅ Prediction completed!")
    print(f"📊 Prediction results ({class_type}):")
    for label, prob in result.items():
        print(f"  {label}: {prob:.3f}")

except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc() 