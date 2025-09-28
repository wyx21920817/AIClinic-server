# model_utils.py
import torch
import numpy as np
import h5py
from tensorflow.keras.models import load_model

DNN_MODEL_PATH = 'models/model.hdf5'

_model_dnn = None

def load_models():
    global _model_dnn
    _model_dnn = load_model(DNN_MODEL_PATH)
    print("Model expects:", _model_dnn.input_shape)

def predict_dnn(ecg_array: np.ndarray) -> dict:
    """
    Args:
        ecg_array: ndarray, shape (12, N)
    Returns:
        dict: {'RBBB': prob, 'LBBB': prob}
    """
    x = np.expand_dims(ecg_array, axis=0)  # shape: (1, 12, N)
    print("Feeding tensor shape:", x.shape)
    preds = _model_dnn.predict(x)
    return {
        '1dAVB': float(preds[0][0]),
        'RBBB': float(preds[0][1]),
        'LBBB': float(preds[0][2]),
        'SBRAD': float(preds[0][3]),
        'AFIB': float(preds[0][4]),
        'STACH': float(preds[0][5])
    }

def predict_ecg(ecg_array: np.ndarray) -> dict:
    if _model_dnn is None:
        raise RuntimeError("DNN model is not loaded. Call load_models() first.")
    return predict_dnn(ecg_array)
