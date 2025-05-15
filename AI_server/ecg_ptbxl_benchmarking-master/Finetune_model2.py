import os
import sys
import torch
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1) 把 code 目录加入 path
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))

# 项目内的模块
from utils.utils import load_dataset, compute_label_aggregations, select_data, preprocess_signals
from models.fastai_model import fastai_model

# ------------------- 配置 -------------------
DATA_PATH            = 'data/ptbxl/'                               # 数据根目录
SAMPLING_RATE        = 100                                         # 采样率
CTYPE                = 'superdiagnostic'                           # 任务类型
MIN_SAMPLES          = 0                                           # 每类最少样本数
# 微调专用输出目录，不再使用 output/exp0/data
FT_EXPERIMENT        = 'exp0_ft'
OUTPUT_DATA_FT       = f'output/{FT_EXPERIMENT}/data/'              # 微调输出目录
OUTPUT_MODELS_FT     = f'output/{FT_EXPERIMENT}/models/'            # 微调模型存放目录

PRETRAINED_NAME      = 'fastai_xresnet1d101'                       # 预训练模型标识
PRETRAINED_FOLDER    = 'output/exp0/models/fastai_xresnet1d101/'   # 预训练权重目录
EPOCHS_FT            = 10                                          # 微调轮数
SEED                 = 42
DEVICE               = 'cuda' if torch.cuda.is_available() else 'cpu'

# 假设所有信号长度都是 1000，导联数是 12
INPUT_SHAPE = (1000, 12)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DATA_FT, exist_ok=True)
    os.makedirs(OUTPUT_MODELS_FT, exist_ok=True)

    # 1. 加载并预处理标签
    data, raw_labels = load_dataset(DATA_PATH, SAMPLING_RATE)
    labels_df       = compute_label_aggregations(raw_labels, DATA_PATH, CTYPE)
    data, labels_df, Y, _ = select_data(
        data, labels_df, CTYPE,
        min_samples=MIN_SAMPLES,
        outputfolder=OUTPUT_DATA_FT    # 改为微调目录
    )

    print(f"Total samples: {len(data)}, label matrix shape: {Y.shape}")

    # 2. 使用官方 strat_fold 划分训练/验证
    X_train = data[labels_df.strat_fold < 10]
    y_train = Y[labels_df.strat_fold < 10]
    X_val   = data[labels_df.strat_fold == 10]
    y_val   = Y[labels_df.strat_fold == 10]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # 3. 标准化 —— 与预训练时完全一致
    #    preprocess_signals 会保存到 OUTPUT_DATA_FT/standard_scaler.pkl
    X_train, X_val, _ = preprocess_signals(X_train, X_val, [], OUTPUT_DATA_FT)

    # 转成 list of arrays 供 fastai_model.fit 调用
    X_train_fin = [x for x in X_train]
    X_val_fin   = [x for x in X_val]
    y_train_fin = [y for y in y_train]
    y_val_fin   = [y for y in y_val]

    # 4. 实例化并微调 fastai_model
    model = fastai_model(
        name                = PRETRAINED_NAME,
        n_classes           = y_train.shape[1],   # 例如 5
        freq                = SAMPLING_RATE,
        outputfolder        = OUTPUT_MODELS_FT,   # 微调权重存放目录
        input_shape         = INPUT_SHAPE,
        pretrained          = True,
        pretrainedfolder    = PRETRAINED_FOLDER,
        n_classes_pretrained= 71,                 # 预训练用的类别数
        epochs_finetuning   = EPOCHS_FT,
        chunkify_train      = False,
    )

    # 5. 运行微调（内部包含 lr_find、earlystop 等）
    model.fit(X_train_fin, y_train_fin, X_val_fin, y_val_fin)

    # fit 结束后，fastai_model 已在内部调用 learn.save(self.name)
    # 并通过下面这行暴露了 learner
    learn = model.learn

    # 6. 额外手动保存一次 state_dict（可选）
    torch.save(
        {'model': learn.model.state_dict()},
        os.path.join(OUTPUT_MODELS_FT, f'{PRETRAINED_NAME}_finetuned_manual.pth')
    )

    # 7. 绘制并保存 LR Finder 曲线
    learn.recorder.plot()
    plt.savefig(os.path.join(OUTPUT_MODELS_FT, 'lr_find.png'))

if __name__ == '__main__':
    main()
