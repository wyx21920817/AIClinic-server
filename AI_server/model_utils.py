# model_utils.py
import torch
import numpy as np
import h5py
from tensorflow.keras.models import load_model

from transformers import AutoModel
import torch.nn as nn
from typing import Dict, List, Optional

from preprocess import extract_waveform_raw, preprocess_12lead_to_flat

DNN_MODEL_PATH = 'models/model.hdf5'
HUBERT_MODEL_ID = "Edoardo-BS/hubert-ecg-base"
CLF_HEAD_PATH = 'models/clf_head.pt'
CLF_LABELS: List[str] = ["1AVB","CRBBB","CLBBB","AFIB","STACH","2AVB","3AVB","PAC","PVC","AFLT"]

_model_dnn = None
_hubert_model: Optional[AutoModel] = None
_clf_head: Optional[nn.Module] = None
_clf_hidden_dim: Optional[int] = None
_device: str = "cpu"


class MLPHeadStrong(nn.Module):
    def __init__(self, hidden_dim: int, n_classes: int, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(512, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_models():
    global _model_dnn, _hubert_model, _device
    _model_dnn = load_model(DNN_MODEL_PATH)
    print("Model expects:", _model_dnn.input_shape)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _hubert_model = AutoModel.from_pretrained(HUBERT_MODEL_ID, trust_remote_code=True).to(_device)
        _hubert_model.eval()
        print(f"Loaded HuBERT backbone: {HUBERT_MODEL_ID} on {_device}")
    except Exception as e:
        print(f"Warning: failed to load HuBERT backbone: {e}")
        _hubert_model = None


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


@torch.no_grad()
def _hubert_embed(flat_6000: np.ndarray, fs: int, pool: str = "mean") -> torch.Tensor:
    if _hubert_model is None:
        raise RuntimeError("HuBERT model is not loaded. Call load_models() first or check backbone availability.")
    x = torch.tensor(flat_6000, dtype=torch.float32, device=_device).unsqueeze(0)  # (1, 6000)
    out = _hubert_model(input_values=x)
    if hasattr(out, "last_hidden_state"):
        hs = out.last_hidden_state  # (1, T, D)
    elif isinstance(out, (list, tuple)):
        hs = out[0]
    else:
        hs = getattr(out, "embeddings", None)
        if hs is None:
            raise RuntimeError("Unexpected HuBERT outputs; cannot find hidden states.")
    if pool == "mean":
        emb = hs.mean(dim=1)  # (1, D)
    elif pool == "max":
        emb, _ = hs.max(dim=1)  # (1, D)
    else:
        raise ValueError("pool must be 'mean' or 'max'")
    return emb.squeeze(0).detach().cpu()


def _ensure_head_loaded(hidden_dim: int):
    global _clf_head, _clf_hidden_dim
    if _clf_head is not None and _clf_hidden_dim == hidden_dim:
        return
    head = MLPHeadStrong(hidden_dim=hidden_dim, n_classes=len(CLF_LABELS), p=0.2).to(_device)
    state = torch.load(CLF_HEAD_PATH, map_location=_device)
    head.load_state_dict(state, strict=True)
    head.eval()
    _clf_head = head
    _clf_hidden_dim = hidden_dim


@torch.no_grad()
def predict_hubert_head_from_ds(ds) -> Dict[str, float]:
    sig_mv, fs, _ = extract_waveform_raw(ds)
    flat, fs100 = preprocess_12lead_to_flat(sig_mv, fs)
    if fs100 != 100:
        raise RuntimeError(f"Expected 100Hz after preprocessing, got {fs100}")
    if flat.shape[0] != 6000:
        raise RuntimeError(f"Expected flattened length 6000, got {flat.shape[0]}")
    emb = _hubert_embed(flat, fs100, pool="mean")  # (D,) on CPU
    hidden_dim = int(emb.shape[-1])
    _ensure_head_loaded(hidden_dim)
    logits = _clf_head(emb.unsqueeze(0).to(_device))  # (1, C)
    probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    return {name: float(p) for name, p in zip(CLF_LABELS, probs)}
