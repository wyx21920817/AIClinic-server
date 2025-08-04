# extract_and_predict_ecg.py

import sys
import os
import torch
import pydicom
import numpy as np
import pickle
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from classify_single_ecg_ft import load_model, preprocess_signal, set_seed

# 1. 配置微调模型路径
BASE_PATH_FT    = 'output/exp0_ft'
WEIGHTS_FILE_FT = os.path.join(BASE_PATH_FT, 'models', 'fastai_xresnet1d101_finetuned_manual.pth')
DATA_DIR_FT     = os.path.join(BASE_PATH_FT, 'data')  # standard_scaler.pkl & mlb.pkl

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

def load_scaler_and_mlb(data_dir=DATA_DIR_FT):
    """
    加载标准化器和 MultiLabelBinarizer
    """
    sp = os.path.join(data_dir, 'standard_scaler.pkl')
    mp = os.path.join(data_dir, 'mlb.pkl')
    print("Scaler:", sp)
    print("MLB:   ", mp)
    with open(sp,'rb') as f: scaler = pickle.load(f)
    with open(mp,'rb') as f: mlb    = pickle.load(f)
    print(f"🔢 {len(mlb.classes_)} classes loaded")
    return scaler, mlb

def build_class_map(mlb):
    """
    返回 {idx: label_name} 映射
    """
    return {i:cls for i,cls in enumerate(mlb.classes_)}

def process_and_predict(dicom_file):
    set_seed()  # 固定随机

    # -- 1. 加载 scaler 与 mlb
    scaler, mlb = load_scaler_and_mlb()
    idx2name   = build_class_map(mlb)
    n_out      = len(mlb.classes_)

    # -- 2. 提取 ECG
    ecg = extract_ecg_from_dicom(dicom_file)         # shape (samples, channels)
    
    # -- 3. 预处理：转成 (channels, samples)，再 normalize
    sig = ecg.astype(np.float32).T                   # (12, N)
    # 归一化DICOM数据到[-1, 1]范围
    sig = sig / 1000.0
    sig, scaler = preprocess_signal(sig, scaler)     # reuse classify_single_ecg_ft 里的逻辑
    x   = torch.from_numpy(sig).unsqueeze(0)         # (1,12,N)

    # -- 4. 加载微调模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(WEIGHTS_FILE_FT, n_in=12, n_out=n_out, device=device)
    if model is None:
        raise RuntimeError("模型加载失败")

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
    if len(sys.argv) != 2:
        print("用法: python extract_and_predict_ecg.py <dicom_path>")
        sys.exit(1)

    dicom_path = sys.argv[1]
    # 处理 DICOM 文件并获取预测结果（字典类型）
    predictions = process_and_predict(dicom_path)
    
    # 打印预测结果（字典形式）
    print("Predictions:")
    for label, prob in predictions.items():
        print(f"{label}: {prob:.3f}")

