import sys, os
import wfdb
import torch
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from code.models.xresnet1d import xresnet1d101
import pickle
import torch.nn.functional as F

# 设置随机种子以确保每次预测的一致性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(weights_path, n_in=12, n_out=5, device='cpu'):
    # 1. 构造模型（5 分类）
    model = xresnet1d101(input_channels=n_in, num_classes=n_out).to(device)

    # 2. 读取保存的模型权重（包含额外的优化器信息等）
    state = torch.load(weights_path, map_location=device)

    # 3. 提取模型的实际权重（忽略"opt" 和 "model"等不需要的键）
    if 'model' in state:
        state = state['model']  # 如果包含 "model" 字段，提取模型部分

    # 4. 加载模型权重，并处理最后一层大小不匹配的情况
    try:
        model.load_state_dict(state, strict=False)  # 使用 strict=False 忽略不匹配的键
        print("模型权重加载成功")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None

    # 5. 切换到评估模式
    model.eval()
    return model

def preprocess_signal(sig, scaler=None):
    """
    对 ECG 信号进行标准化处理。
    使用提供的 scaler 进行标准化。
    """
    if scaler is not None:
        scaled_sig = scaler.transform(sig.flatten()[:, np.newaxis]).reshape(sig.shape)
    else:
        # 如果没有提供 scaler 参数，初始化一个新的 StandardScaler
        scaler = StandardScaler()
        scaler.fit(sig.flatten()[:, np.newaxis])
        scaled_sig = scaler.transform(sig.flatten()[:, np.newaxis]).reshape(sig.shape)
    return scaled_sig, scaler

def predict_ecg(record_base, model, device='cpu', class_names=None, scaler=None):
    rec = wfdb.rdrecord(record_base)
    sig = rec.p_signal.astype(np.float32).T  # 转置以适应模型输入
    sig, scaler = preprocess_signal(sig, scaler)
    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # 将信号转化为 tensor，并添加 batch 维度

    model.eval()
    with torch.no_grad():
        logits = model(x)  # (1, 5)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()  # 计算每个类的概率
        pred = probs.argmax()  # 预测类别

    print(f"\nRecord: {record_base}")
    for i, p in enumerate(probs):
        print(f"  {class_names[i]:15s} prob={p:.3f}")

    print(f"→ 预测类别: {class_names[pred]} (概率 {probs[pred]:.3f})")
    return probs, pred, scaler

def load_data_and_scaler():
    """
    加载数据和标准化器，用于预测时的预处理
    """
    # 将路径指向正确的位置
    scaler_path = os.path.join('output', 'exp0', 'data', 'standard_scaler.pkl')
    mlb_path = os.path.join('output', 'exp0', 'data', 'mlb.pkl')
    print(f'scaler_path:',scaler_path)
    print(f'mlb_path:',mlb_path)

    # 1. 加载标准化器
    with open(scaler_path, 'rb') as ss_file:
        scaler = pickle.load(ss_file)

    # 2. 加载标签的 MultiLabelBinarizer（需要与训练时一致）
    with open(mlb_path, 'rb') as mlb_file:
        label_binarizer = pickle.load(mlb_file)

    return scaler, label_binarizer

if __name__ == '__main__':
    # 设置随机种子确保可重复性
    set_seed(seed=42)

    if len(sys.argv) != 2:
        print("用法: python classify_single_ecg.py <record_base>")
        print("示例: python classify_single_ecg.py data/ptbxl/records100/00000/00002_lr")
        sys.exit(1)

    # 用户输入的记录前缀
    record_base = sys.argv[1]

    # 模型权重路径（相对于项目根目录）
    weights_path = os.path.join('output', 'exp0', 'models', 'fastai_xresnet1d101', 'models', 'fastai_xresnet1d101.pth')

    # 加载数据和标准化器
    scaler, label_binarizer = load_data_and_scaler()

    # —— 请根据您训练时的设置修改 —— 
    NUM_CLASSES = 5
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']  # 诊断性五分类示例

    # 1. 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(weights_path, n_in=12, n_out=NUM_CLASSES, device=device)

    if model is None:
        print("模型加载失败！")
        sys.exit(1)

    # 2. 执行预测
    predict_ecg(record_base, model, device=device, class_names=CLASS_NAMES, scaler=scaler)
