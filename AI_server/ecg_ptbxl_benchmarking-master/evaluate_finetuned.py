#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_finetuned.py

仅对微调模型进行评估，并生成以下图表：
1. metrics_table.png：各超类的 Accuracy/Precision/Recall/F1-Score/ROC AUC 表格
2. confusion_matrix.png：5×5 混淆矩阵热力图（带数值标注，仅对单标签样本）
3. roc_curves.png：5 条超类的 ROC 曲线
"""
import os, sys, pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)

# 将 code 目录加入 import 路径
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))

from utils.utils import load_dataset, compute_label_aggregations, select_data
from models.fastai_model import fastai_model

# ----------------- 配置 -----------------
DATA_PATH       = 'data/ptbxl/'
SAMPLING_RATE   = 100
OUTPUT_DATA_FT  = 'output/exp0_ft/data/'
FINETUNE_FOLDER = 'output/exp0_ft'
FINETUNED_NAME  = 'fastai_xresnet1d101_finetuned_manual'
INPUT_SHAPE     = (1000, 12)
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD       = 0.5
SUPER_LABELS    = ['NORM','MI','STTC','CD','HYP']

# ----------------- 加载测试数据 -----------------
def load_test_data_5():
    data, raw_labels = load_dataset(DATA_PATH, SAMPLING_RATE)
    labels_df = compute_label_aggregations(raw_labels, DATA_PATH, 'superdiagnostic')
    data_sel, labels_df, Y, _ = select_data(
        data, labels_df, 'superdiagnostic',
        min_samples=0, outputfolder=OUTPUT_DATA_FT
    )
    mask = labels_df.strat_fold == 10
    X = [x for x, m in zip(data_sel, mask) if m]
    y = Y[mask]
    return X, y

# ----------------- Scaler & 预处理 -----------------
def load_scaler(path):
    pth = os.path.join(path, 'standard_scaler.pkl')
    with open(pth, 'rb') as f:
        return pickle.load(f)

def preprocess_batch(X, scaler):
    out = []
    for rec in X:
        sig = rec.T.astype(np.float32)
        flat = sig.flatten()[:, None]
        scaled = scaler.transform(flat).reshape(sig.shape)
        out.append(scaled.T)
    return out

# ----------------- 主流程 -----------------
def main():
    # 1. 加载并预处理测试集
    X, y_true = load_test_data_5()
    scaler = load_scaler(OUTPUT_DATA_FT)
    X_proc = preprocess_batch(X, scaler)

    # 2. 实例化微调模型并预测
    model = fastai_model(
        name=FINETUNED_NAME,
        n_classes=len(SUPER_LABELS),
        freq=SAMPLING_RATE,
        outputfolder=FINETUNE_FOLDER,
        input_shape=INPUT_SHAPE,
        pretrained=True,
        pretrainedfolder=FINETUNE_FOLDER,
        n_classes_pretrained=71
    )
    probs = model.predict(X_proc)  # shape (n_samples, 5)

    # 3. 计算二分类指标（多标签）
    y_pred_bin = (probs >= THRESHOLD).astype(int)
    acc_list  = [accuracy_score(y_true[:,i], y_pred_bin[:,i]) for i in range(len(SUPER_LABELS))]
    prec_list = precision_score(y_true, y_pred_bin, average=None, zero_division=0)
    rec_list  = recall_score(y_true, y_pred_bin, average=None, zero_division=0)
    f1_list   = f1_score(y_true, y_pred_bin, average=None)

    # 4. 计算 ROC AUC
    fpr, tpr, roc_auc = {}, {}, []
    for i in range(len(SUPER_LABELS)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], probs[:,i])
        roc_auc.append(auc(fpr[i], tpr[i]))

    # 5. 生成 Metrics 表格图
    metrics_df = pd.DataFrame({
        'Class': SUPER_LABELS,
        'Accuracy': np.round(acc_list,3),
        'Precision': np.round(prec_list,3),
        'Recall': np.round(rec_list,3),
        'F1-Score': np.round(f1_list,3),
        'ROC AUC': np.round(roc_auc,3)
    })
    fig, ax = plt.subplots(figsize=(8,2))
    ax.axis('off')
    table = ax.table(
        cellText=metrics_df.drop(columns='Class').values,
        rowLabels=metrics_df['Class'],
        colLabels=metrics_df.columns[1:],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.tight_layout()
    plt.savefig('metrics_table.png', dpi=300)
    plt.close()

    # 6. 绘制带数值的混淆矩阵（仅对单标签样本）
    # 过滤出真实标签唯一的样本，避免多标签导致的分布偏差
    single_mask = y_true.sum(axis=1) == 1
    true_idx = np.argmax(y_true[single_mask], axis=1)
    pred_idx = np.argmax(probs[single_mask], axis=1)
    cm = confusion_matrix(true_idx, pred_idx)
    fig, ax = plt.subplots(figsize=(5,5))
    cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(range(len(SUPER_LABELS))); ax.set_xticklabels(SUPER_LABELS)
    ax.set_yticks(range(len(SUPER_LABELS))); ax.set_yticklabels(SUPER_LABELS)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color=color)
    plt.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

    # 7. 绘制 ROC 曲线
    fig, ax = plt.subplots(figsize=(6,5))
    for i, cls in enumerate(SUPER_LABELS):
        ax.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.3f})")
    ax.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Super Classes')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.close()

    print('已生成: metrics_table.png, confusion_matrix.png, roc_curves.png')

if __name__ == '__main__':
    main()
