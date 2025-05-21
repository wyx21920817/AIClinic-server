# python3 classify_single_ecg_ft.py data/ptbxl/records100/00000/00005_lr

import sys, os
import wfdb
import torch
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from .code.models.xresnet1d import xresnet1d101
import pickle
import torch.nn.functional as F

# -------------------- 配置 --------------------
# 微调实验的输出根目录
BASE_PATH_FT     = 'output/exp0_ft'
DATA_DIR_FT      = os.path.join(BASE_PATH_FT, 'data')
MODELS_DIR_FT    = os.path.join(BASE_PATH_FT, 'models')
WEIGHTS_FILE_FT  = os.path.join(MODELS_DIR_FT, 'fastai_xresnet1d101_finetuned_manual.pth')

# 随机种子
SEED = 42

# ----------------------------------------------
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(weights_path, n_in, n_out, device='cpu'):
    # 1. 载入 checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get('model', ckpt)

    # 2. 从 state 推断分类头 hidden_size
    hidden_size = None
    for k,v in state.items():
        if v.ndim==2 and v.shape[0]==n_out:
            hidden_size = v.shape[1]
            print(f"🔍 hidden_size={hidden_size} (from `{k}`)")
            break
    if hidden_size is None:
        print(f"❌ 无法推断 hidden_size (n_out={n_out})")
        return None

    # 3. 重建模型
    model = xresnet1d101(
        input_channels=n_in,
        num_classes=n_out,
        lin_ftrs_head=[hidden_size]
    ).to(device)

    # 4. 严格加载权重
    try:
        model.load_state_dict(state, strict=True)
        print("✅ finetuned 模型权重加载成功")
    except Exception as e:
        print("❌ 加载权重出错:", e)
        return None

    model.eval()
    return model

def preprocess_signal(sig, scaler=None):
    flat = sig.flatten()[:,None]
    if scaler is not None:
        scaled = scaler.transform(flat).reshape(sig.shape)
    else:
        scaler = StandardScaler().fit(flat)
        scaled = scaler.transform(flat).reshape(sig.shape)
    return scaled, scaler

def predict_ecg(record_base, model, device, idx2name, scaler, top_k=5):
    rec = wfdb.rdrecord(record_base)
    sig = rec.p_signal.astype(np.float32).T
    sig, scaler = preprocess_signal(sig, scaler)
    x = torch.from_numpy(sig).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, -1)[0].cpu().numpy()
    idxs = np.argsort(probs)[::-1][:top_k]
    print(f"\nRecord: {record_base}")
    for i in idxs:
        print(f"  {idx2name[i]:15s} prob={probs[i]:.3f}")

def load_data_and_scaler(base_path=BASE_PATH_FT):
    # 从微调目录加载 scaler 和 mlb
    sp = os.path.join(base_path, 'data', 'standard_scaler.pkl')
    mp = os.path.join(base_path, 'data', 'mlb.pkl')
    print("scaler:", sp)
    print("mlb:   ", mp)
    with open(sp,'rb') as f: scaler = pickle.load(f)
    with open(mp,'rb') as f: mlb    = pickle.load(f)
    print(f"🔢 Loaded {len(mlb.classes_)} classes:", mlb.classes_)
    return scaler, mlb

def build_class_map(mlb):
    return {i:cls for i,cls in enumerate(mlb.classes_)}

# -------------------- 主流程 --------------------
if __name__=='__main__':
    set_seed()
    if len(sys.argv)!=2:
        print("Usage: python classify_single_ecg_ft.py <record_base_path>")
        sys.exit(1)

    rec_base = sys.argv[1]
    # 1) 加载 scaler 和 mlb（微调版，5 类）
    scaler, mlb = load_data_and_scaler()
    idx2name    = build_class_map(mlb)
    n_out       = len(mlb.classes_)      # 应为 5

    # 2) 载入微调模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(WEIGHTS_FILE_FT, n_in=12, n_out=n_out, device=device)
    if model is None: sys.exit(1)

    # 3) 推理并打印 top-5
    predict_ecg(rec_base, model, device, idx2name, scaler, top_k=5)
