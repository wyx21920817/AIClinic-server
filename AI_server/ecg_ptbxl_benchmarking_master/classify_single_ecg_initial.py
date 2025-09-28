import sys, os
import wfdb
import torch
import numpy as np
import random
from code.models.xresnet1d import xresnet1d101

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

    # 3. Extract actual model weights (ignore keys like "opt" / wrap "model")
    if 'model' in state:
        state = state['model']  # If checkpoint has "model" key, take that

    # 4. Load weights; handle mismatch on final layer
    try:
        # Drop final layer weights to reinitialize with desired size
        if '8.4.weight' in state:
            del state['8.4.weight']
            del state['8.4.bias']
        model.load_state_dict(state, strict=False)  # strict=False to ignore mismatched keys

        print("Model weights loaded successfully")
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return None

    # 5. Switch to eval mode
    model.eval()
    return model

def predict_ecg(record_base, model, device='cpu', thresh=0.7, class_names=None):
    """
    Run prediction on a single WFDB record and print results
    - record_base: path prefix without extension, e.g. 'data/ptbxl/records100/00000/00001_lr'
    - model: loaded xresnet1d101 model
    - thresh: binarization threshold for multi-label
    - class_names: list of class names with length == model outputs
    """
    # 1. Read signal
    rec = wfdb.rdrecord(record_base)
    sig = rec.p_signal.astype(np.float32).T  # to (n_leads, n_samples)
    x = torch.from_numpy(sig).unsqueeze(0).to(device)  # shape=(1, n_leads, n_samples)

    # 2. Inference
    with torch.no_grad():
        logits = model(x)                # shape=(1, n_out)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

    # 3. Binarize
    preds = (probs >= thresh).astype(int)

    # 4. Print
    print(f"\nRecord: {record_base}")
    for i, p in enumerate(probs):
        name = class_names[i] if class_names else f"Class_{i}"
        flag = '✔' if preds[i] == 1 else '✘'
        print(f"  {name:15s} prob={p:.3f} → pred={preds[i]} {flag}")

    return probs, preds


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

    # Adjust according to your training setup
    NUM_CLASSES = 5
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']  # example 5 diagnostic classes

    # 1. Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(weights_path, n_in=12, n_out=NUM_CLASSES, device=device)

    if model is None:
        print("Model loading failed!")
        sys.exit(1)

    # 2. Run prediction
    predict_ecg(record_base, model, device=device, thresh=0.7, class_names=CLASS_NAMES)
