# extract_and_predict_ecg.py

import sys
import os
import torch
import pydicom
import numpy as np
import pickle
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# 修复相对导入问题
try:
    from .classify_single_ecg_ft import load_model, preprocess_signal, set_seed
    from .classify_single_ecg128 import load_model as load_model_128, preprocess_signal as preprocess_signal_128, set_seed as set_seed_128
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from classify_single_ecg_ft import load_model, preprocess_signal, set_seed
        from classify_single_ecg128 import load_model as load_model_128, preprocess_signal as preprocess_signal_128, set_seed as set_seed_128
    except ImportError:
        print("Warning: Unable to import necessary modules, please ensure the files are in the correct directory")

# 1. 配置微调模型路径（超类预测）
HERE            = os.path.dirname(os.path.realpath(__file__))
BASE_PATH_FT    = os.path.join(HERE, 'output', 'exp0_ft')
MODELS_DIR_FT   = os.path.join(BASE_PATH_FT, 'models')
WEIGHTS_FILE_FT = os.path.join(
    MODELS_DIR_FT,
    'fastai_xresnet1d101_finetuned_manual.pth'
)
DATA_DIR_FT     = os.path.join(BASE_PATH_FT, 'data')  # standard_scaler.pkl & mlb.pkl

# 2. 配置原始模型路径（子类预测）
BASE_PATH_ORIG  = os.path.join(HERE, 'output', 'exp0')
MODELS_DIR_ORIG = os.path.join(BASE_PATH_ORIG, 'models')
WEIGHTS_FILE_ORIG = os.path.join(
    MODELS_DIR_ORIG,
    'fastai_xresnet1d101', 'models', 'fastai_xresnet1d101.pth'
)
DATA_DIR_ORIG   = os.path.join(BASE_PATH_ORIG, 'data')  # standard_scaler.pkl & mlb.pkl

def extract_ecg_from_dicom(dicom_file):
    """
    从 DICOM 的 WaveformSequence 提取 ECG 信号，返回 shape=(samples, channels) 的 np.array
    """
    ds = pydicom.dcmread(dicom_file, force=True)
    if not hasattr(ds, 'WaveformSequence'):
        raise ValueError("DICOM 文件中未找到 WaveformSequence")
    wf = ds.WaveformSequence[0]
    data = wf.WaveformData
    n_s = wf.NumberOfWaveformSamples
    n_c = wf.NumberOfWaveformChannels
    print(f"Number of Samples: {n_s}, Number of Channels: {n_c}")
    arr = np.frombuffer(data, dtype=np.int16)
    if arr.size != n_s * n_c:
        # 截取或填充
        arr = np.pad(arr, (0, n_s*n_c - arr.size))[:n_s*n_c]
    return arr.reshape((n_s, n_c))

def load_scaler_and_mlb(data_dir, class_type="superclass"):
    """
    加载标准化器和 MultiLabelBinarizer
    class_type: "superclass" 或 "subclass"
    """
    sp = os.path.join(data_dir, 'standard_scaler.pkl')
    mp = os.path.join(data_dir, 'mlb.pkl')
    print(f"Scaler ({class_type}):", sp)
    print(f"MLB ({class_type}):   ", mp)
    
    try:
        with open(sp,'rb') as f: 
            scaler = pickle.load(f)
        with open(mp,'rb') as f: 
            mlb = pickle.load(f)
        print(f"🔢 Loaded {len(mlb.classes_)} classes: {mlb.classes_}")
        return scaler, mlb
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        raise
    except Exception as e:
        print(f"❌ 加载文件时出错: {e}")
        raise

def build_class_map(mlb):
    """
    返回 {idx: label_name} 映射
    """
    return {i:cls for i,cls in enumerate(mlb.classes_)}

def process_and_predict(dicom_file, class_type="subclass"):
    """
    处理DICOM文件并进行ECG预测
    
    参数:
        dicom_file: DICOM文件路径
        class_type: 预测类型，"superclass"（超类，5分类）或 "subclass"（子类，71分类），默认为"subclass"
    
    返回:
        预测结果字典 {label: probability}
    """
    if class_type not in ["superclass", "subclass"]:
        raise ValueError('class_type 必须是 "superclass" 或 "subclass"')
    
    print(f"🎯 Prediction type: {class_type}")
    
    if class_type == "superclass":
        # 使用微调模型预测5个超类
        return _predict_superclass(dicom_file)
    else:
        # 使用原始模型预测71个子类
        return _predict_subclass(dicom_file)

def _predict_superclass(dicom_file):
    """使用微调模型预测超类（5分类）"""
    set_seed()  # 固定随机

    # -- 1. 加载 scaler 与 mlb
    scaler, mlb = load_scaler_and_mlb(DATA_DIR_FT, "superclass")
    idx2name   = build_class_map(mlb)
    n_out      = len(mlb.classes_)

    # -- 2. 提取 ECG
    ecg = extract_ecg_from_dicom(dicom_file)         # shape (samples, channels)
    
    # -- 3. 预处理：转成 (channels, samples)，再 normalize
    sig = ecg.astype(np.float32).T   
    # 归一化DICOM数据到[-1, 1]范围
    sig = sig / 1000.0                # (12, N)
    sig, scaler = preprocess_signal(sig, scaler)     # reuse classify_single_ecg_ft 里的逻辑
    x   = torch.from_numpy(sig).unsqueeze(0)         # (1,12,N)

    # -- 4. 加载微调模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(WEIGHTS_FILE_FT, n_in=12, n_out=n_out, device=device)
    if model is None:
        raise RuntimeError("微调模型加载失败")

    # -- 5. 推理
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)                            # (1, n_out)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()
    idxs = np.argsort(probs)[::-1][:5]

    # -- 6. 输出 top-5 并以字典形式返回
    result = {}
    print(f"\nDICOM: {dicom_file}")
    for i in idxs:
        result[idx2name[i]] = float(probs[i])
        print(f"  {idx2name[i]:15s} prob={probs[i]:.3f}")

    return result

def _predict_subclass(dicom_file):
    """使用原始模型预测子类（71分类）"""
    set_seed_128()  # 固定随机

    # -- 1. 加载 scaler 与 mlb
    scaler, mlb = load_scaler_and_mlb(DATA_DIR_ORIG, "subclass")
    idx2name   = build_class_map(mlb)
    n_out      = len(mlb.classes_)

    # -- 2. 提取 ECG
    ecg = extract_ecg_from_dicom(dicom_file)         # shape (samples, channels)
    
    # -- 3. 预处理：转成 (channels, samples)，再 normalize
    sig = ecg.astype(np.float32).T   
    # 归一化DICOM数据到[-1, 1]范围
    sig = sig / 1000.0                # (12, N)
    sig, scaler = preprocess_signal_128(sig, scaler)     # 使用子类预测的预处理逻辑
    x   = torch.from_numpy(sig).unsqueeze(0)         # (1,12,N)

    # -- 4. 加载原始模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model_128(WEIGHTS_FILE_ORIG, n_in=12, n_out=n_out, device=device)
    if model is None:
        raise RuntimeError("原始模型加载失败")

    # -- 5. 推理
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)                            # (1, n_out)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()
    idxs = np.argsort(probs)[::-1][:5]

    # -- 6. 输出 top-5 并以字典形式返回
    result = {}
    print(f"\nDICOM: {dicom_file}")
    for i in idxs:
        result[idx2name[i]] = float(probs[i])
        print(f"  {idx2name[i]:15s} prob={probs[i]:.3f}")

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python extract_and_predict_ecg.py <dicom_path> [class_type]")
        print("class_type: superclass (超类，5分类) 或 subclass (子类，71分类)，默认为 subclass")
        sys.exit(1)

    dicom_path = sys.argv[1]
    class_type = sys.argv[2] if len(sys.argv) > 2 else "subclass"
    
    # 处理 DICOM 文件并获取预测结果（字典类型）
    try:
        predictions = process_and_predict(dicom_path, class_type)
        
        # 打印预测结果（字典形式）
        print(f"\n Pridiction outcomes ({class_type}):")
        for label, prob in predictions.items():
            print(f"{label}: {prob:.3f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

