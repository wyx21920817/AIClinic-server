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

def load_model(weights_path, n_in=12, n_out=5, device='cpu'):
    # 1. Build model (5 classes)
    model = xresnet1d101(input_channels=n_in, num_classes=n_out).to(device)

    # 2. Load saved checkpoint (may include optimizer etc.)
    state = torch.load(weights_path, map_location=device)

    # 3. Extract actual model weights (handle "model" key)
    if 'model' in state:
        state = state['model']  # If checkpoint has "model" key

    # 4. Load weights; allow key mismatch for last layer
    try:
        model.load_state_dict(state, strict=False)  # strict=False to ignore mismatched keys
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return None

    # 5. Switch to eval mode
    model.eval()
    return model

def preprocess_signal(sig, scaler=None):
    """
    Standardize ECG signal using the provided scaler (or fit a new one)
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
    sig = rec.p_signal.astype(np.float32).T  # transpose for model input
    sig, scaler = preprocess_signal(sig, scaler)
    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # to tensor with batch dim

    model.eval()
    with torch.no_grad():
        logits = model(x)  # (1, 5)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()  # probabilities
        pred = probs.argmax()  # predicted class

    print(f"\nRecord: {record_base}")
    for i, p in enumerate(probs):
        print(f"  {class_names[i]:15s} prob={p:.3f}")

    print(f"→ Predicted: {class_names[pred]} (prob {probs[pred]:.3f})")
    return probs, pred, scaler

def load_data_and_scaler():
    """
    Load data utilities: StandardScaler and MultiLabelBinarizer for preprocessing
    """
    # 将路径指向正确的位置
    scaler_path = os.path.join('output', 'exp0', 'data', 'standard_scaler.pkl')
    mlb_path = os.path.join('output', 'exp0', 'data', 'mlb.pkl')
    print(f'scaler_path:',scaler_path)
    print(f'mlb_path:',mlb_path)

    # 1. Load StandardScaler
    with open(scaler_path, 'rb') as ss_file:
        scaler = pickle.load(ss_file)

    # 2. Load MultiLabelBinarizer (must match training)
    with open(mlb_path, 'rb') as mlb_file:
        label_binarizer = pickle.load(mlb_file)

    return scaler, label_binarizer

if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(seed=42)

    if len(sys.argv) != 2:
        print("Usage: python classify_single_ecg.py <record_base>")
        print("Example: python classify_single_ecg.py data/ptbxl/records100/00000/00002_lr")
        sys.exit(1)

    # User-provided record base
    record_base = sys.argv[1]

    # Weights path (relative to project root)
    weights_path = os.path.join('output', 'exp0', 'models', 'fastai_xresnet1d101', 'models', 'fastai_xresnet1d101.pth')

    # Load data and scaler
    scaler, label_binarizer = load_data_and_scaler()

    # Adjust according to your training setup
    NUM_CLASSES = 5
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']  # example 5-class setup

    # 1. Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(weights_path, n_in=12, n_out=NUM_CLASSES, device=device)

    if model is None:
        print("Model loading failed!")
        sys.exit(1)

    # 2. Run prediction
    predict_ecg(record_base, model, device=device, class_names=CLASS_NAMES, scaler=scaler)
