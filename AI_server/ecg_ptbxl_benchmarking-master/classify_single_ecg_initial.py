import sys, os
import wfdb
import torch
import numpy as np
import random
from code.models.xresnet1d import xresnet1d101

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
        # 去掉最后一层的权重，重新初始化为需要的尺寸
        if '8.4.weight' in state:
            del state['8.4.weight']
            del state['8.4.bias']
        model.load_state_dict(state, strict=False)  # 使用 strict=False 忽略不匹配的键

        print("Model weights loaded successfully")
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return None

    # 5. 切换到评估模式
    model.eval()
    return model

def predict_ecg(record_base, model, device='cpu', thresh=0.7, class_names=None):
    """
    对单条 WFDB 记录执行预测并打印结果
    - record_base: 记录文件前缀，不带扩展名，例如 'data/ptbxl/records100/00000/00001_lr'
    - model: 已加载的 xresnet1d101 模型
    - thresh: 概率二值化阈值（多标签分类）
    - class_names: 类别名称列表，长度需等于模型输出通道数
    """
    # 1. 读信号
    rec = wfdb.rdrecord(record_base)
    sig = rec.p_signal.astype(np.float32).T  # 转为 (n_leads, n_samples)
    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # shape=(1, n_leads, n_samples)

    # 2. 前向推断
    with torch.no_grad():
        logits = model(x)                # shape=(1, n_out)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

    # 3. 二值化
    preds = (probs >= thresh).astype(int)

    # 4. 打印
    print(f"\nRecord: {record_base}")
    for i, p in enumerate(probs):
        name = class_names[i] if class_names else f"Class_{i}"
        flag = '✔' if preds[i] == 1 else '✘'
        print(f"  {name:15s} prob={p:.3f} → pred={preds[i]} {flag}")

    return probs, preds


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
    predict_ecg(record_base, model, device=device, thresh=0.7, class_names=CLASS_NAMES)
