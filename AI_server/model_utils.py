# model_utils.py
import torch
import numpy as np
import h5py
from tensorflow.keras.models import load_model

# 模型路径
DNN_MODEL_PATH = 'models/model.hdf5'

# 全局模型缓存
_model_dnn = None

def load_models():
    """
    加载 DNN 模型
    """
    global _model_dnn
    _model_dnn = load_model(DNN_MODEL_PATH)
    print("Model expects:", _model_dnn.input_shape)

def predict_dnn(ecg_array: np.ndarray) -> dict:
    """
    使用 DNN 模型预测 RBBB 和 LBBB
    Args:
        ecg_array: ndarray, shape (12, N)
    Returns:
        dict: {'RBBB': prob, 'LBBB': prob}
    """
    x = np.expand_dims(ecg_array, axis=0)  # shape: (1, 12, N)
    print("Feeding tensor shape:", x.shape)
    preds = _model_dnn.predict(x)
    return {
        '1dAVb': float(preds[0][0]),
        'RBBB': float(preds[0][1]),
        'LBBB': float(preds[0][2]),
        'SB': float(preds[0][3]),
        'AF': float(preds[0][4]),
        'ST': float(preds[0][5])
    }

def predict_ecg(ecg_array: np.ndarray) -> dict:
    """
    使用 DNN 模型对 ECG 数据进行预测
    """
    if _model_dnn is None:
        raise RuntimeError("DNN model is not loaded. Call load_models() first.")
    return predict_dnn(ecg_array)
