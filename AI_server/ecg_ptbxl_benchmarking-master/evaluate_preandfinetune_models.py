#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_models.py

批量评估预训练模型和微调模型在 PTB-XL 测试集上的性能。
"""
import os, sys, pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    multilabel_confusion_matrix,
    classification_report,
    roc_curve
)

def unwrap_wrapper_pth(folder, name):
    pth = os.path.join(folder, 'models', f'{name}.pth')
    if not os.path.exists(pth): return
    raw = torch.load(pth, map_location=DEVICE)
    if isinstance(raw, dict) and 'model' in raw:
        torch.save(raw['model'], pth)

# 将 code 目录加入 import 路径
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))

from utils.utils import load_dataset, compute_label_aggregations, select_data
from models.fastai_model import fastai_model

# ----------------- 配置 -----------------
DATA_PATH          = 'data/ptbxl/'                               # 原始数据目录
SAMPLING_RATE      = 100
OUTPUT_DATA_PRE    = 'output/exp0/data/'                          # scaler & mlb for pretrained
OUTPUT_DATA_FT     = 'output/exp0_ft/data/'                          # scaler & mlb for finetuned (同上)
PRETRAINED_FOLDER  = 'output/exp0/models/fastai_xresnet1d101'     # pretrained learner 保存目录
FINETUNE_FOLDER    = 'output/exp0_ft'                         # finetuned .pth 所在目录
PRETRAINED_NAME    = 'fastai_xresnet1d101'
FINETUNED_NAME     = 'fastai_xresnet1d101_finetuned_manual'
INPUT_SHAPE        = (1000, 12)
N_SUPER            = 5
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD          = 0.5
SUPER_LABELS       = ['NORM','MI','STTC','CD','HYP']

# ----------------- 加载测试数据 -----------------
def load_test_data_71():
    data, raw_labels = load_dataset(DATA_PATH, SAMPLING_RATE)
    labels_df_71 = compute_label_aggregations(raw_labels, DATA_PATH, 'diagnostic')
    data_71, labels_df_71, Y_71, _ = select_data(
        data, labels_df_71, 'diagnostic',
        min_samples=0, outputfolder=OUTPUT_DATA_PRE
    )
    mask = labels_df_71.strat_fold == 10
    X = [x for x,m in zip(data_71, mask) if m]
    return X, Y_71[mask]

def load_test_data_5():
    data, raw_labels = load_dataset(DATA_PATH, SAMPLING_RATE)
    labels_df_5 = compute_label_aggregations(raw_labels, DATA_PATH, 'superdiagnostic')
    data_5, labels_df_5, Y_5, _ = select_data(
        data, labels_df_5, 'superdiagnostic',
        min_samples=0, outputfolder=OUTPUT_DATA_PRE
    )
    mask = labels_df_5.strat_fold == 10
    X = [x for x,m in zip(data_5, mask) if m]
    return X, Y_5[mask]

def load_scaler(dir_path):
    with open(os.path.join(dir_path, 'standard_scaler.pkl'),'rb') as f:
        return pickle.load(f)

def preprocess_batch(X, scaler):
    out = []
    for rec in X:
        sig = rec.T.astype(np.float32)
        flat = sig.flatten()[:,None]
        scaled = scaler.transform(flat).reshape(sig.shape)
        out.append(scaled.T)
    return out

# ----------------- 聚合到 5 类 -----------------
def aggregate_to_super(probs_71):
    # 读取 mlb 获取 71 类顺序
    mlb_path = os.path.join(OUTPUT_DATA_PRE, 'mlb.pkl')
    with open(mlb_path,'rb') as f:
        mlb = pickle.load(f)
    classes_71 = list(mlb.classes_)

    # 读取 scp 声明文件
    scp = pd.read_csv(os.path.join(DATA_PATH,'scp_statements.csv'))
    desc_col = 'description' if 'description' in scp.columns else 'scp_statement'
    desc2sup = dict(zip(scp[desc_col], scp['diagnostic_class']))
    sup_idx  = {s:i for i,s in enumerate(SUPER_LABELS)}

    n = probs_71.shape[0]
    agg = np.zeros((n,N_SUPER),dtype=np.float32)
    for j,desc in enumerate(classes_71):
        sup = desc2sup.get(desc)
        if sup in sup_idx:
            agg[:, sup_idx[sup]] += probs_71[:, j]
    return agg

# ----------------- 评估 5 类 -----------------
def evaluate_5(y_true, y_pred):
    macro = roc_auc_score(y_true,y_pred,average='macro')
    micro = roc_auc_score(y_true,y_pred,average='micro')
    per_cls = [roc_auc_score(y_true[:,i],y_pred[:,i]) for i in range(N_SUPER)]
    y_bin = (y_pred>=THRESHOLD).astype(int)
    mcm = multilabel_confusion_matrix(y_true, y_bin)
    report = classification_report(y_true, y_bin, zero_division=0, output_dict=True)
    return {
        'macro_auc': macro,
        'micro_auc': micro,
        'per_class_auc': per_cls,
        'confusion_matrices': mcm,
        'classification_report': report,
        'probs': y_pred
    }

def plot_roc(y_true, y_pred, title, out_png):
    plt.figure()
    for i,cls in enumerate(SUPER_LABELS):
        fpr,tpr,_ = roc_curve(y_true[:,i], y_pred[:,i])
        plt.plot(fpr,tpr,label=cls)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# ----------------- 主流程 -----------------
def main():
    # 1. 测试集
    X5, y5     = load_test_data_5()
    X71, _     = load_test_data_71()

    # 2. Scaler
    scaler_pre = load_scaler(OUTPUT_DATA_PRE)
    scaler_ft  = load_scaler(OUTPUT_DATA_FT)

    # 3. 预处理
    X_pre71 = preprocess_batch(X71, scaler_pre)
    X_ft5   = preprocess_batch(X5,  scaler_ft)

    # 3.5 拆箱微调权重（fix wrapper .pth）
    unwrap_wrapper_pth(FINETUNE_FOLDER, FINETUNED_NAME)

    # 4. 模型实例化
    model_pre = fastai_model(
        name=PRETRAINED_NAME,
        n_classes=71,
        freq=SAMPLING_RATE,
        outputfolder=PRETRAINED_FOLDER,
        input_shape=INPUT_SHAPE,
        pretrained=True,
        pretrainedfolder=PRETRAINED_FOLDER,
        n_classes_pretrained=71
    )
    model_ft = fastai_model(
        name=FINETUNED_NAME,
        n_classes=N_SUPER,
        freq=SAMPLING_RATE,
        outputfolder=FINETUNE_FOLDER,
        input_shape=INPUT_SHAPE,
        pretrained=True,
        pretrainedfolder=FINETUNE_FOLDER,
        n_classes_pretrained=71
    )

    # 5. 预测 & 聚合
    probs71    = model_pre.predict(X_pre71)       # (n,71)
    probs5_pre = aggregate_to_super(probs71)      # (n,5)
    probs5_ft  = model_ft.predict(X_ft5)          # (n,5)

    # 6. 评估
    res_pre = evaluate_5(y5, probs5_pre)
    res_ft  = evaluate_5(y5, probs5_ft)

    # 7. 保存报表
    pd.DataFrame([
        {'model':'pre', 'macro_auc':res_pre['macro_auc'], 'micro_auc':res_pre['micro_auc']},
        {'model':'ft',  'macro_auc':res_ft['macro_auc'],  'micro_auc':res_ft['micro_auc']}
    ]).to_csv('evaluation_summary.csv', index=False)

    pd.DataFrame({
        'class':SUPER_LABELS,
        'pre_auc':res_pre['per_class_auc'],
        'ft_auc': res_ft['per_class_auc']
    }).to_csv('evaluation_per_class_auc.csv', index=False)

    # 8. 混淆矩阵 & 报告
    np.savez('confusion_matrices.npz',
             pre=res_pre['confusion_matrices'], ft=res_ft['confusion_matrices'])
    pd.DataFrame(res_pre['classification_report']).T.to_csv('class_report_pre.csv')
    pd.DataFrame(res_ft['classification_report']).T.to_csv('class_report_ft.csv')

    # 9. ROC 曲线
    plot_roc(y5, res_pre['probs'], 'ROC - Pretrained', 'roc_pretrained.png')
    plot_roc(y5, res_ft['probs'],  'ROC - Finetuned',  'roc_finetuned.png')

    # 10. 打印
    print("Summary:\n", pd.read_csv('evaluation_summary.csv'))
    print("\nPer-class AUC:\n", pd.read_csv('evaluation_per_class_auc.csv'))

if __name__=='__main__':
    main()
