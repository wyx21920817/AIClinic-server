# extract_and_predict_ecg.py

import sys
import os
import torch
import pydicom
import numpy as np
import pickle
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Fix import path issues - set at file top
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
#1
# Fix relative import issues
try:
    from .classify_single_ecg_ft import load_model, preprocess_signal, set_seed
    from .classify_single_ecg128 import load_model as load_model_128, preprocess_signal as preprocess_signal_128, set_seed as set_seed_128
except ImportError:
    # If relative import fails, try absolute import
    try:
        from classify_single_ecg_ft import load_model, preprocess_signal, set_seed
        from classify_single_ecg128 import load_model as load_model_128, preprocess_signal as preprocess_signal_128, set_seed as set_seed_128
    except ImportError:
        print("Warning: Unable to import necessary modules, please ensure the files are in the correct directory")
        # Fallback: create default set_seed_128 function
        def set_seed_128(seed=42):
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Fallback: create default load_model and preprocess_signal functions
        def load_model_128(weights_path, n_in, n_out, device='cpu'):
            print("Warning: Using default load_model_128 function")
            # Try to import the xresnet1d model directly
            try:
                from code.models.xresnet1d import xresnet1d101
                
                # 1. Load checkpoint
                ckpt = torch.load(weights_path, map_location=device, weights_only=False)
                state = ckpt.get('model', ckpt)

                # 2. Infer the hidden_size of the classification head from the state
                hidden_size = None
                for k,v in state.items():
                    if v.ndim==2 and v.shape[0]==n_out:
                        hidden_size = v.shape[1]
                        print(f"Hidden size inferred from `{k}`: {hidden_size}")
                        break
                if hidden_size is None:
                    print(f"Failed to infer hidden_size (n_out={n_out})")
                    return None

                # 3. Rebuild the model
                model = xresnet1d101(
                    input_channels=n_in,
                    num_classes=n_out,
                    lin_ftrs_head=[hidden_size]
                ).to(device)

                # 4. Strictly load the weights
                try:
                    model.load_state_dict(state, strict=True)
                    print("Finetuned model weights loaded successfully")
                except Exception as e:
                    print("Error loading weights:", e)
                    return None

                model.eval()
                return model
                
            except ImportError as e:
                print(f"Cannot import xresnet1d101 model: {e}")
                # Try other import methods
                try:
                    # Method 1: Directly add path
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    code_dir = os.path.join(current_dir, 'code')
                    sys.path.insert(0, code_dir)
                    
                    from models.xresnet1d import xresnet1d101
                    print("Successfully imported xresnet1d101 using alternative path")
                    
                    # Continue loading model...
                    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
                    state = ckpt.get('model', ckpt)

                    hidden_size = None
                    for k,v in state.items():
                        if v.ndim==2 and v.shape[0]==n_out:
                            hidden_size = v.shape[1]
                            print(f"Hidden size inferred from `{k}`: {hidden_size}")
                            break
                    if hidden_size is None:
                        print(f"Failed to infer hidden_size (n_out={n_out})")
                        return None

                    model = xresnet1d101(
                        input_channels=n_in,
                        num_classes=n_out,
                        lin_ftrs_head=[hidden_size]
                    ).to(device)

                    try:
                        model.load_state_dict(state, strict=True)
                        print("Finetuned model weights loaded successfully")
                    except Exception as e:
                        print("Error loading weights:", e)
                        return None

                    model.eval()
                    return model
                    
                except ImportError as e2:
                    print(f"Alternative import also failed: {e2}")
                    return None
        
        def preprocess_signal_128(sig, scaler=None):
            print("Warning: Using default preprocess_signal_128 function")
            # Use the same preprocessing logic as classify_single_ecg128.py
            flat = sig.flatten()[:, None]
            if scaler is not None:
                scaled = scaler.transform(flat).reshape(sig.shape)
            else:
                scaler = StandardScaler().fit(flat)
                scaled = scaler.transform(flat).reshape(sig.shape)
            return scaled, scaler

# Define disease category mapping of interest
def get_disease_mapping():
    """
    Return mapping dict that maps 71 subclass labels to target categories of interest
    
    Returns:
        dict: {original_label: target_category}
    """
    mapping = {
        # Normal cases
        'NORM': 'NORM',
        
        # AV Block cases
        '1AVB': '1dAVB',
        '2AVB': '2dAVB', 
        '3AVB': '3dAVB',
        # Also support with 'd' for backward compatibility
        '1dAVB': '1dAVB',
        '2dAVB': '2dAVB', 
        '3dAVB': '3dAVB',
        
        # Atrial Fibrillation/Flutter
        'AFIB': 'AFIB',
        'AFLT': 'AFLT',
        
        # Bundle Branch Block cases - Merge ILBBB and CLBBB into LBBB
        'LBBB': 'LBBB',
        'ILBBB': 'LBBB',  # Incomplete LBBB -> LBBB
        'CLBBB': 'LBBB',  # Complete LBBB -> LBBB
        
        # Right Bundle Branch Block
        'RBBB': 'RBBB',
        'IRBBB': 'RBBB',  # Incomplete RBBB -> RBBB
        'CRBBB': 'RBBB',  # Complete RBBB -> RBBB
        
        # Sinus Tachycardia
        'STACH': 'STACH',
        # 'SVTAC': 'STACH',     # Supraventricular tachycardia -> STACH
        # 'PSVT': 'STACH',      # Paroxysmal SVT -> STACH
        
        # Premature Atrial Complex/Contraction
        'PAC': 'PAC',
        # 'PRC(S)': 'PAC',      # Premature complex(es) -> PAC
        # 'BIGU': 'PAC',        # Bigeminal pattern -> PAC 
        # 'TRIGU': 'PAC',       # Trigeminal pattern -> PAC 
        
        # Premature Ventricular Complex/Contraction  
        'PVC': 'PVC',

    }
    return mapping

def map_diseases_to_categories(predictions, disease_mapping):
    """
    Map original predictions to target disease categories
    
    Args:
        predictions: original prediction dict {disease: probability}
        disease_mapping: mapping dict
    
    Returns:
        mapped prediction dict {target_category: probability}
    """
    # Define target categories
    target_categories = [
        'NORM', '1dAVB', '2dAVB', '3dAVB', 'AFIB', 'AFLT', 
        'LBBB', 'RBBB', 'STACH', 'PAC', 'PVC'
    ]
    
    # Initialize result dict
    mapped_predictions = {category: 0.0 for category in target_categories}
    
    # Map diseases and accumulate probabilities
    for disease, prob in predictions.items():
        # Find mapped target category
        target_category = None
        
        # Direct match
        if disease in disease_mapping:
            target_category = disease_mapping[disease]
        else:
            # Fuzzy match (handle possible variants)
            for key, value in disease_mapping.items():
                if key in disease.upper() or disease.upper() in key:
                    target_category = value
                    break
        
        # If mapped, accumulate; otherwise ignore (teacher-focused subset)
        if target_category and target_category in mapped_predictions:
            mapped_predictions[target_category] += prob
        else:
            # Ignore unmapped categories
            pass
    
    # Sort by probability and return top-5 for display
    sorted_predictions = sorted(mapped_predictions.items(), key=lambda x: x[1], reverse=True)
    top_5_predictions = dict(sorted_predictions[:5])
    
    return top_5_predictions

# 1. Configure fine-tuned model paths (superclass prediction)
HERE            = os.path.dirname(os.path.realpath(__file__))
BASE_PATH_FT    = os.path.join(HERE, 'output', 'exp0_ft')
MODELS_DIR_FT   = os.path.join(BASE_PATH_FT, 'models')
WEIGHTS_FILE_FT = os.path.join(
    MODELS_DIR_FT,
    'fastai_xresnet1d101_finetuned_manual.pth'
)
DATA_DIR_FT     = os.path.join(BASE_PATH_FT, 'data')  # standard_scaler.pkl & mlb.pkl

# 2. Configure original model paths (subclass prediction)
BASE_PATH_ORIG  = os.path.join(HERE, 'output', 'exp0')
MODELS_DIR_ORIG = os.path.join(BASE_PATH_ORIG, 'models')
WEIGHTS_FILE_ORIG = os.path.join(
    MODELS_DIR_ORIG,
    'fastai_xresnet1d101', 'models', 'fastai_xresnet1d101.pth'
)
DATA_DIR_ORIG   = os.path.join(BASE_PATH_ORIG, 'data')  # standard_scaler.pkl & mlb.pkl

# 3. Configure 13-class model paths
BASE_PATH_13   = os.path.join(HERE, 'output', 'exp_13class_smart_multilabel')
MODELS_DIR_13  = os.path.join(BASE_PATH_13, 'models')
WEIGHTS_FILE_13 = os.path.join(MODELS_DIR_13, 'fastai_xresnet1d101_13class_smart.pth')
DATA_DIR_13    = os.path.join(BASE_PATH_13, 'data')  # standard_scaler.pkl & mlb.pkl

def extract_ecg_from_dicom(dicom_file):
    """
    Extract ECG signal from DICOM WaveformSequence
    Returns np.ndarray with shape (samples, channels)
    """
    ds = pydicom.dcmread(dicom_file, force=True)
    if not hasattr(ds, 'WaveformSequence'):
        raise ValueError("WaveformSequence not found in DICOM")
    wf = ds.WaveformSequence[0]
    data = wf.WaveformData
    n_s = wf.NumberOfWaveformSamples
    n_c = wf.NumberOfWaveformChannels
    print(f"Number of Samples: {n_s}, Number of Channels: {n_c}")
    arr = np.frombuffer(data, dtype=np.int16)
    if arr.size != n_s * n_c:
        # Pad or truncate to expected size
        arr = np.pad(arr, (0, n_s*n_c - arr.size))[:n_s*n_c]
    return arr.reshape((n_s, n_c))

def load_scaler_and_mlb(data_dir, class_type="superclass"):
    """
    Load StandardScaler and MultiLabelBinarizer
    class_type: "superclass" | "subclass" | "13class"
    """
    sp = os.path.join(data_dir, 'standard_scaler.pkl')
    mp = os.path.join(data_dir, 'mlb.pkl')
    print(f"Scaler ({class_type}):", sp)
    print(f"MLB ({class_type}):   ", mp)
    
    try:
        with open(sp,'rb') as f: 
            scaler = pickle.load(f)
        with open(mp,'rb') as f: 
            mlb = pickle.load(f)
        print(f"Loaded {len(mlb.classes_)} classes: {mlb.classes_}")
        return scaler, mlb
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

def build_class_map(mlb):
    """
    Return mapping {index: label_name}
    """
    return {i:cls for i,cls in enumerate(mlb.classes_)}

def process_and_predict(dicom_file, class_type="13class"):
    """
    Process DICOM file and run ECG prediction
    
    Args:
        dicom_file: path to DICOM file
        class_type: "superclass" (5-class), "subclass" (71-class), or "13class"
    
    Returns:
        dict {label: probability}
    """
    if class_type not in ["superclass", "subclass", "13class"]:
        raise ValueError('class_type must be "superclass", "subclass" or "13class"')
    
    print(f"Prediction type: {class_type}")
    
    if class_type == "superclass":
        # Use fine-tuned model to predict 5 superclasses
        return _predict_superclass(dicom_file)
    elif class_type == "subclass":
        # Use original model to predict 71 subclasses, then map to target categories
        return _predict_subclass(dicom_file)
    else:
        # Default 13-class model prediction
        return _predict_13class(dicom_file)

def _predict_superclass(dicom_file):
    """Use fine-tuned model to predict superclasses (5 classes)"""
    set_seed()  # Fix random seed

    # -- 1. Load scaler and mlb
    scaler, mlb = load_scaler_and_mlb(DATA_DIR_FT, "superclass")
    idx2name   = build_class_map(mlb)
    n_out      = len(mlb.classes_)

    # -- 2. Extract ECG
    ecg = extract_ecg_from_dicom(dicom_file)         # shape (samples, channels)
    
    # -- 3. Preprocess: convert to (channels, samples), then normalize
    sig = ecg.astype(np.float32).T   
    # Normalize DICOM data to [-1, 1] range
    sig = sig / 1000.0                # (12, N)
    sig, scaler = preprocess_signal(sig, scaler)     # reuse logic from classify_single_ecg_ft
    x   = torch.from_numpy(sig).unsqueeze(0)         # (1,12,N)

    # -- 4. Load fine-tuned model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(WEIGHTS_FILE_FT, n_in=12, n_out=n_out, device=device)
    if model is None:
        raise RuntimeError("Failed to load fine-tuned model")

    # -- 5. Inference
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)                            # (1, n_out)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()
    idxs = np.argsort(probs)[::-1][:5]

    # -- 6. Output top-5 and return as dictionary
    result = {}
    print(f"\nDICOM: {dicom_file}")
    for i in idxs:
        result[idx2name[i]] = float(probs[i])
        print(f"  {idx2name[i]:15s} prob={probs[i]:.3f}")

    return result

def _predict_subclass(dicom_file):
    """Use original model to predict subclasses (71 classes), then map to target disease categories"""
    set_seed_128()  # Fix random seed

    # -- 1. Load scaler and mlb
    scaler, mlb = load_scaler_and_mlb(DATA_DIR_ORIG, "subclass")
    idx2name   = build_class_map(mlb)
    n_out      = len(mlb.classes_)

    # -- 2. Extract ECG
    ecg = extract_ecg_from_dicom(dicom_file)         # shape (samples, channels)
    
    # -- 3. Preprocess: convert to (channels, samples), then normalize
    sig = ecg.astype(np.float32).T   
    # Normalize DICOM data to [-1, 1] range
    sig = sig / 1000.0                # (12, N)
    sig, scaler = preprocess_signal_128(sig, scaler)     # Use preprocessing logic for subclass prediction
    x   = torch.from_numpy(sig).unsqueeze(0)         # (1,12,N)

    # -- 4. Load original model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model_128(WEIGHTS_FILE_ORIG, n_in=12, n_out=n_out, device=device)
    if model is None:
        raise RuntimeError("Failed to load original model")

    # -- 5. Inference
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)                            # (1, n_out)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()
    idxs = np.argsort(probs)[::-1][:5]

    # -- 6. Get original prediction results
    original_result = {}
    print(f"\nDICOM: {dicom_file}")
    print("Original prediction results (71 classes):")
    for i in idxs:
        original_result[idx2name[i]] = float(probs[i])
        print(f"  {idx2name[i]:15s} prob={probs[i]:.3f}")

    # -- 7. Map to target disease categories
    disease_mapping = get_disease_mapping()
    mapped_result = map_diseases_to_categories(original_result, disease_mapping)
    
    print("\nMapped prediction results (target disease categories):")
    for category, prob in mapped_result.items():
        if prob > 0:
            print(f"  {category:15s} prob={prob:.3f}")

    return mapped_result


def _predict_13class(dicom_file):
    """Use 13-class model to predict target classes directly (13 classes)."""
    # Reuse subclass utilities for robustness
    set_seed_128()

    # -- 1. Load scaler and mlb (13-class)
    scaler, mlb = load_scaler_and_mlb(DATA_DIR_13, "13class")
    idx2name   = build_class_map(mlb)
    n_out      = len(mlb.classes_)

    # -- 2. Extract ECG
    ecg = extract_ecg_from_dicom(dicom_file)

    # -- 3. Preprocess to (12, N) normalized
    sig = ecg.astype(np.float32).T
    sig = sig / 1000.0
    sig, scaler = preprocess_signal_128(sig, scaler)
    x   = torch.from_numpy(sig).unsqueeze(0)

    # -- 4. Load 13-class model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model_128(WEIGHTS_FILE_13, n_in=12, n_out=n_out, device=device)
    if model is None:
        raise RuntimeError("Failed to load 13-class model")

    # -- 5. Inference (multi-label probabilities with sigmoid)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

    # -- 6. Build original prediction dict (top-5 for display)
    idxs = np.argsort(probs)[::-1]
    original_result = {}
    print(f"\nDICOM: {dicom_file}")
    print("Original prediction results (13 classes):")
    for i in idxs[:5]:
        original_result[idx2name[i]] = float(probs[i])
        print(f"  {idx2name[i]:15s} prob={probs[i]:.3f}")

    # -- 7. Map to teacher-focused categories (merge LBBB/RBBB, rename AVB)
    disease_mapping = get_disease_mapping()
    mapped_result = map_diseases_to_categories(original_result, disease_mapping)

    print("\nMapped prediction results (target disease categories):")
    for category, prob in mapped_result.items():
        if prob > 0:
            print(f"  {category:15s} prob={prob:.3f}")

    return mapped_result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_and_predict_ecg.py <dicom_path> [class_type]")
        print("class_type: superclass | subclass | 13class (default: 13class)")
        sys.exit(1)

    dicom_path = sys.argv[1]
    class_type = sys.argv[2] if len(sys.argv) > 2 else "13class"
    
    # Process DICOM file and get prediction results (dictionary type)
    try:
        predictions = process_and_predict(dicom_path, class_type)
        
        # Print prediction results (dictionary format)
        print(f"\nFinal prediction results ({class_type}):")
        for label, prob in predictions.items():
            if prob > 0:  # Only show categories with probability > 0
                print(f"{label}: {prob:.3f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

