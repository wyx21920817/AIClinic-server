import sys, os
import wfdb
import torch
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from code.models.xresnet1d import xresnet1d101
import pickle
import torch.nn.functional as F

# Set random seed for reproducible predictions
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(weights_path, n_in, n_out, device='cpu'):
    """
    Load model and set to eval mode.
    n_in: number of input channels (typically 12 leads)
    n_out: number of classes (e.g., len(label_binarizer.classes_), 71)
    """
    # 1. 读取 checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get('model', ckpt)

    # 2. Infer hidden_size from the first linear layer of the head in checkpoint
    hidden_size = None
    for k, v in state.items():
        if v.ndim == 2 and v.shape[0] == n_out:
            hidden_size = v.shape[1]
            print(f"Inferred hidden_size from `{k}`: {hidden_size}")
            break
    if hidden_size is None:
        print(f"❌ Unable to infer hidden_size from checkpoint (n_out={n_out})")
        return None

    # 3. Build identical model: 512 -> hidden_size -> n_out
    model = xresnet1d101(
        input_channels=n_in,
        num_classes=n_out,
        lin_ftrs_head=[hidden_size],
    ).to(device)

    # 4. Strictly load all weights (including head)
    try:
        model.load_state_dict(state, strict=True)
        print("✅ finetuned model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None

    model.eval()
    return model


def preprocess_signal(sig, scaler=None):
    """
    Standardize ECG signal
    """
    flat = sig.flatten()[:, None]
    if scaler is not None:
        scaled = scaler.transform(flat).reshape(sig.shape)
    else:
        scaler = StandardScaler().fit(flat)
        scaled = scaler.transform(flat).reshape(sig.shape)
    return scaled, scaler


def predict_ecg(record_base, model, device, class_index_to_name, scaler, top_k=5):
    """
    Inference on a single ECG record, output top_k classes and probabilities
    """
    rec = wfdb.rdrecord(record_base)
    sig = rec.p_signal.astype(np.float32).T  # transpose to (channels, time)
    sig, scaler = preprocess_signal(sig, scaler)
    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # (1, channels, time)

    with torch.no_grad():
        logits = model(x)               # (1, n_out)
        probs  = F.softmax(logits, -1)[0].cpu().numpy()
        top_idx = np.argsort(probs)[::-1][:top_k]
        top_ps  = probs[top_idx]

    print(f"\nRecord: {record_base}")
    for idx, p in zip(top_idx, top_ps):
        name = class_index_to_name.get(idx, f"Unknown_{idx}")
        print(f"  {name:15s} prob={p:.3f}")


def load_data_and_scaler(base_path='output/exp0'):
    """
    Load StandardScaler and MultiLabelBinarizer; print normalized paths
    """
    scaler_p = os.path.normpath(os.path.join(base_path, 'data', 'standard_scaler.pkl'))
    mlb_p    = os.path.normpath(os.path.join(base_path, 'data', 'mlb.pkl'))
    print("scaler_path:", scaler_p)
    print("mlb_path:   ", mlb_p)

    with open(scaler_p, 'rb') as f:
        scaler = pickle.load(f)
    with open(mlb_p, 'rb') as f:
        mlb = pickle.load(f)

    print(f"🔢 Loaded {len(mlb.classes_)} labels:", mlb.classes_)
    return scaler, mlb


def build_class_map(mlb):
    """
    Build mapping from class index to name
    """
    return {i: cls for i, cls in enumerate(mlb.classes_)}


if __name__ == '__main__':
    set_seed(42)

    if len(sys.argv) != 2:
        print("Usage: python classify_single_ecg.py <record_base>")
        print("Example: python classify_single_ecg.py data/ptbxl/records100/00000/00002_lr")
        sys.exit(1)

    record_base = os.path.normpath(sys.argv[1])
    weights_path = os.path.normpath(os.path.join(
        'output', 'exp0', 'models',
        'fastai_xresnet1d101', 'models',
        'fastai_xresnet1d101.pth'
    ))

    # 1. Load scaler and label binarizer
    scaler, mlb = load_data_and_scaler('output/exp0')
    class_map = build_class_map(mlb)

    # 2. Set class count to actual labels length
    NUM_CLASSES = len(mlb.classes_)  # 71

    # 3. Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(weights_path, n_in=12, n_out=NUM_CLASSES, device=device)
    if model is None:
        sys.exit(1)

    # 4. Inference and print top-5 labels
    predict_ecg(record_base, model, device, class_map, scaler, top_k=5)
