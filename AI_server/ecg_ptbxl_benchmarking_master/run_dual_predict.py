#!/usr/bin/env python3
"""
Configurable multi-model prediction script:
- Read the same ECG DICOM, run each provided model, and compare side-by-side
- Model dir layout: output/.../models/<model_name>
  - Weights: <model_dir>/models/<model_name>.pth
  - Data dir inferred as <exp_root>/data (two levels up from model dir)
  - Data dir must contain standard_scaler.pkl and mlb.pkl

Example:
python run_dual_predict.py \
  --dicom data/ICBEB/dcm_records100/2.dcm \
  --model_dir output/exp0/models/fastai_xresnet1d101 \
  --model_dir output/exp_ptbxl12_from_icbeb/models/fastai_xresnet1d101_ptbxl12_from_icbeb \
  --topk 10
"""




import sys
import os
import pickle
import argparse
from pathlib import Path
import torch
import numpy as np
import pydicom
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Dual/multi model DICOM prediction and comparison")
parser.add_argument('--dicom', required=True, help='Path to DICOM file')
parser.add_argument('--model_dir', action='append', default=[], help='Model directory (can repeat)')
parser.add_argument('--topk', type=int, default=10, help='Show Top-K classes')
args = parser.parse_args()

dicom_path = args.dicom
model_dirs = args.model_dir
topk = args.topk

print(f"🚀 Starting prediction for DICOM: {dicom_path}")
print(f"🎯 Prediction type: subclass (Table by model class) Comparing the number of models: {len(model_dirs) if model_dirs else 2}")

try:
    # Load subclass prediction utilities (shared for all models)
    with open('classify_single_ecg128.py', 'r', encoding='utf-8') as f:
        code_128 = f.read()

    # Replace relative imports
    code_128 = code_128.replace('from .code.models.xresnet1d import', 'from code.models.xresnet1d import')
    # Adapt for PyTorch 2.6+ default weights_only=True
    code_128 = code_128.replace('torch.load(weights_path, map_location=device)',
                                'torch.load(weights_path, map_location=device, weights_only=False)')

    # Execute classify_single_ecg128.py code
    globals_128 = {
        '__file__': 'classify_single_ecg128.py',
        '__name__': 'classify_single_ecg128_temp'
    }
    exec(code_128, globals_128)
    print("✅ Subclass prediction module loaded successfully")

    # Ensure set_seed_128 is available
    if 'set_seed' in globals_128:
        globals_128['set_seed_128'] = globals_128['set_seed']
    else:
        # If missing, create a default set_seed_128
        def set_seed_128(seed=42):
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        globals_128['set_seed_128'] = set_seed_128

    # Read DICOM once as 12 leads, then preprocess by each model's scaler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    base_sig = ecg.astype(np.float32).T / 1000.0

    # Default models if none provided
    if not model_dirs:
        model_dirs = [
            'output/exp0/models/fastai_xresnet1d101',
            'output/exp_ptbxl12_from_icbeb/models/fastai_xresnet1d101_ptbxl12_from_icbeb'
        ]

    def infer_weights_and_data_dirs(model_dir: str):
        """
        Robust parsing:
        - Accept: exp root / models dir / specific model dir / direct .pth
        - Data dir search: <path>/data, <path>/../data, <path>/../../data
        - Weights search: given .pth, <path>/models/<basename>.pth, any .pth
        - model_name fallback: checkpoint stem
        """
        p = Path(model_dir)
        candidates_data = [
            p / 'data',
            p.parent / 'data',
            p.parent.parent / 'data'
        ]

        # Resolve weights
        weights_path = None
        model_name = None
        if p.suffix == '.pth' and p.is_file():
            weights_path = p
            model_name = p.stem
        else:
            # Prefer convention: .../models/<model_name>/<model_name>.pth
            if (p / 'models' / f"{p.name}.pth").exists():
                weights_path = p / 'models' / f"{p.name}.pth"
                model_name = p.name
            # Else: any .pth under models
            elif (p / 'models').exists():
                pths = sorted((p / 'models').glob('*.pth'))
                if pths:
                    weights_path = pths[0]
                    model_name = weights_path.stem
            # Else: any .pth under current path
            if weights_path is None:
                pths = sorted(p.glob('*.pth'))
                if pths:
                    weights_path = pths[0]
                    model_name = weights_path.stem
            # Finally: specific model dir heuristic
            if weights_path is None and p.parent.name == 'models':
                cand = p / 'models' / f"{p.name}.pth"
                if cand.exists():
                    weights_path = cand
                    model_name = p.name

        # Recursive fallback search
        if weights_path is None:
            pths = sorted(p.rglob('*.pth')) if p.exists() and p.is_dir() else []
            if pths:
                # 优先选择 models 子目录下的 .pth 文件
                models_pths = [pt for pt in pths if 'models' in str(pt)]
                if models_pths:
                    weights_path = models_pths[0]
                else:
                    weights_path = pths[0]
                model_name = weights_path.stem
        if weights_path is None:
            raise FileNotFoundError(f"Cannot locate weights under: {p}. Provide a dir with checkpoints or a .pth file")

        # Resolve data directory
        data_dir = None
        for d in candidates_data:
            if (d / 'standard_scaler.pkl').exists() and (d / 'mlb.pkl').exists():
                data_dir = d
                break
        # Fallback: accept dir with scaler only
        if data_dir is None:
            for d in candidates_data:
                if (d / 'standard_scaler.pkl').exists():
                    data_dir = d
                    break
        if data_dir is None:
            raise FileNotFoundError(
                "standard_scaler.pkl / mlb.pkl not found. Tried: " + ", ".join(str(d) for d in candidates_data)
            )

        return (model_name or p.name), str(weights_path), str(data_dir)

    models_cfg = []
    for md in model_dirs:
        name, wpath, ddir = infer_weights_and_data_dirs(md)
        # Try classes.json near the model (also check models subdir)
        model_dir_path = Path(md)
        classes_json = None
        for cand in [
            model_dir_path / 'classes.json',
            model_dir_path / 'models' / 'classes.json',
            Path(wpath).parent / 'classes.json'
        ]:
            if cand.exists():
                classes_json = str(cand)
                break
        models_cfg.append({'name': name, 'model_path': wpath, 'data_dir': ddir, 'classes_json': classes_json})

    all_results = {}
    for cfg in models_cfg:
        print(f"\n🔍 Using model: {cfg['name']}")
        # 加载 scaler 与 mlb
        with open(os.path.join(cfg['data_dir'], 'standard_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(cfg['data_dir'], 'mlb.pkl'), 'rb') as f:
            mlb = pickle.load(f)

        # 尝试用模型目录下的 classes.json 覆盖类别列表（若存在）
        classes_list = None
        if cfg.get('classes_json') and os.path.exists(cfg['classes_json']):
            try:
                import json
                with open(cfg['classes_json'], 'r', encoding='utf-8') as jf:
                    classes_list = json.load(jf)
                if isinstance(classes_list, dict) and 'classes' in classes_list:
                    classes_list = classes_list['classes']
            except Exception:
                classes_list = None
        if classes_list is None:
            classes_list = list(getattr(mlb, 'classes_', []))
        print(f"🔢 Classes: {len(classes_list)}")

        # Load weights and preprocess
        model = globals_128['load_model'](cfg['model_path'], n_in=12, n_out=len(classes_list), device=device)
        if model is None:
            raise RuntimeError(f"Model loading failed: {cfg['model_path']}")
        sig, _ = globals_128['preprocess_signal'](base_sig, scaler)
        x = torch.from_numpy(sig).unsqueeze(0).to(device)

        # Inference
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        idxs = np.argsort(probs)[::-1][:topk]
        result = {classes_list[i]: float(probs[i]) for i in idxs}
        all_results[cfg['name']] = result
        print(f"Top-{min(5, topk)}:")
        for i in idxs[:min(5, topk)]:
            print(f"  {classes_list[i]:20s} prob={probs[i]:.3f}")
    
    # Comparison output
    print(f"\n✅ Prediction completed!\n")
    print(f"=== Side-by-side (Top-{topk}) ===")
    # 收集所有模型的top类
    all_keys = set()
    for name, res in all_results.items():
        all_keys |= set(res.keys())
    # 按所有模型中的最大概率排序
    def max_score(k):
        return max([all_results[m].get(k, 0.0) for m in all_results])
    sorted_keys = sorted(list(all_keys), key=max_score, reverse=True)[:topk]
    for k in sorted_keys:
        line = [f"{k:20s}"]
        base = None
        for mi, (mname, res) in enumerate(all_results.items()):
            p = res.get(k, 0.0)
            if mi == 0:
                base = p
                line.append(f"{mname}={p:.3f}")
            else:
                line.append(f"{mname}={p:.3f} Δ={p-base:+.3f}")
        print(" | ".join(line))

except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc() 