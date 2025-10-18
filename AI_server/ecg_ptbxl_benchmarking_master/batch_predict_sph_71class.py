#!/usr/bin/env python3
"""
Ultra-Fast Batch SPH Prediction Script for 71-Class Models
SPH evaluation script using H5 format for 71-class models

Usage:
    python batch_predict_sph_71class.py [exp_folder]
    
Examples:
    python batch_predict_sph_71class.py                           # Use default model
    python batch_predict_sph_71class.py output/exp0              # Use specified experiment folder
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
import h5py
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import signal
import warnings
import argparse
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# SPH to our class mapping (only map classes that directly correspond to our 13 target classes)
SPH_TO_OUR_MAPPING = {
    # Direct category mapping (based on actual SPH code.csv descriptions)
    '1': 'NORM',           # Normal ECG (13907 samples)
    
    # Atrial fibrillation and flutter - direct mapping
    '50': 'AFIB',          # Atrial fibrillation (676 samples)
    '51': 'AFLT',          # Atrial flutter
    
    # AV blocks - direct mapping
    '82': '1AVB',          # Prolonged PR interval (first-degree AV block) (238 samples)
    '83': '2AVB',          # Second-degree AV block, Mobitz type I (Wenckebach)
    '84': '2AVB',          # Second-degree AV block, Mobitz type II
    '85': '2AVB',          # 2:1 AV block
    '86': '2AVB',          # AV block, varying conduction
    '87': '2AVB',          # AV block, advanced (high-grade)
    '88': '3AVB',          # AV block, complete (third-degree)
    
    # Bundle branch blocks - direct mapping
    '104': 'CLBBB',        # Left bundle-branch block
    '105': 'IRBBB',        # Incomplete right bundle-branch block (1259 samples)
    '106': 'CRBBB',        # Right bundle-branch block (710 samples)
    '101': 'ILBBB',        # Left anterior fascicular block -> ILBBB (incomplete LBBB)
    
    # Premature beats - direct mapping
    '30': 'PAC',           # Atrial premature complex(es) (556 samples)
    '31': 'PAC',           # Atrial premature complexes, nonconducted
    '60': 'PVC',           # Ventricular premature complex(es) (1089 samples)
    
    # ST-T changes - direct mapping
    '145': 'STACH',        # ST deviation (1831 samples)
    '146': 'STACH',        # ST deviation with T-wave change (1063 samples)
    '147': 'STACH',        # T-wave abnormality (2220 samples)
    
    # Note: Do not map other categories, maintain original category distribution of SPH dataset
    # If SPH dataset does not have corresponding categories, do not force mapping creation
    # For example: 21(Sinus tachycardia), 22(Sinus bradycardia), 23(Sinus arrhythmia), etc.
    # These may be normal variants but are not any of our 13 target categories
}

# Display name mapping (for output display)
DISPLAY_NAME_MAPPING = {
    'NORM': 'NORM',
    'AFIB': 'AFIB', 
    'AFLT': 'AFLT',
    '1AVB': '1AVB',
    '2AVB': '2AVB',
    '3AVB': '3AVB',
    'CLBBB': 'CLBBB',
    'CRBBB': 'CRBBB',
    'IRBBB': 'IRBBB',
    'ILBBB': 'ILBBB',
    'PAC': 'PAC',
    'PVC': 'PVC',
    'STACH': 'STACH'
}

# 71-class model class list will be dynamically loaded from mlb.pkl at runtime
MODEL_71_CLASSES = None  # Will be dynamically loaded in load_model_and_preprocessing

# Global output directory (will be set in main function)
OUTPUT_DIR = None

# Global flag for graceful shutdown
_shutdown_requested = False

def extract_model_name_from_path(model_path):
    """Extract model name from path for output directory naming"""
    from pathlib import Path
    # Use the actual model file stem as identifier; sanitize and limit length
    model_file = Path(model_path).stem
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_file)
    return safe[:50]

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    global _shutdown_requested
    print("\n🛑 Received interrupt signal (Ctrl+C), gracefully stopping program...")
    print("⏳ Waiting for current tasks to complete...")
    _shutdown_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Ultra-Fast Batch SPH Prediction Script for 71-Class Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_predict_sph_71class.py                           # Use default model
    python batch_predict_sph_71class.py output/exp0              # Use specified experiment folder
        """,
        add_help=True
    )
    
    parser.add_argument(
        'exp_folder', 
        nargs='?',  # Optional positional argument
        default="output/exp0",  # Default use exp0 (71-class model)
        help='Experiment folder path containing models/ and data/ directories (default: output/exp0)'
    )
    
    return parser.parse_args()

def load_model_and_preprocessing(exp_folder):
    """Load 71-class model and preprocessing components"""
    try:
        print(f"🔧 Loading 71-class model and preprocessing components from: {exp_folder}")
        
        # Build paths
        exp_path = Path(exp_folder)
        models_dir = exp_path / "models"
        data_dir = exp_path / "data"
        
        print(f"   📁 Models directory: {models_dir}")
        print(f"   📁 Data directory: {data_dir}")
        
        # Check if directories exist
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find model files (recursive search)
        model_files = list(models_dir.rglob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in: {models_dir}")
        
        # Filter temporary files, prioritize main model files
        main_models = [f for f in model_files if not f.name.startswith('tmp')]
        if main_models:
            model_path = main_models[0]
        else:
            model_path = model_files[0]
        
        print(f"   📁 Using model: {model_path}")
        
        # Load scaler and mlb
        scaler_path = data_dir / "standard_scaler.pkl"
        mlb_path = data_dir / "mlb.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not mlb_path.exists():
            raise FileNotFoundError(f"MLB file not found: {mlb_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        
        # Dynamically set 71-class model class list
        global MODEL_71_CLASSES
        MODEL_71_CLASSES = list(mlb.classes_)
        print(f"   ✅ Loaded scaler and mlb from: {data_dir}")
        print(f"   📋 Model has {len(MODEL_71_CLASSES)} classes")
        
        # Verify SPH mapped classes exist in 71-class model
        print("   🔍 Verifying SPH class mappings:")
        for sph_code, our_class in SPH_TO_OUR_MAPPING.items():
            if our_class in MODEL_71_CLASSES:
                idx = MODEL_71_CLASSES.index(our_class)
                print(f"      {sph_code:3s} -> {our_class:6s} (index: {idx:2d}) ✅")
            else:
                print(f"      {sph_code:3s} -> {our_class:6s} (not found) ❌")
        
        # Import prediction module (avoid command line argument conflicts)
        import subprocess
        import sys
        
        # Directly import classify_single_ecg128.py content
        with open('classify_single_ecg128.py', 'r', encoding='utf-8') as f:
            code_128 = f.read()
        
        # Replace relative imports
        code_128 = code_128.replace('from .code.models.xresnet1d import', 'from code.models.xresnet1d import')
        code_128 = code_128.replace('from .code.data.ptb_xl import', 'from code.data.ptb_xl import')
        
        # Execute code and get globals_128
        exec_globals = {}
        exec(code_128, exec_globals)
        globals_128 = exec_globals
        
        # Detect model output class count
        print("   🔍 Detecting model output classes...")
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model_weights = checkpoint['model']
            # Find the last layer (classification head)
            last_key = list(model_weights.keys())[-1]
            if last_key.endswith('.bias'):
                n_out = model_weights[last_key].shape[0]
                print(f"   📊 Detected model output classes: {n_out}")
            else:
                n_out = 71  # Default 71 classes
                print(f"   ⚠️  Could not detect output classes, using default: {n_out}")
        else:
            n_out = 71  # Default 71 classes
            print(f"   ⚠️  Could not detect output classes, using default: {n_out}")
        
        # Pre-load model and set to evaluation mode (same as ICBEB script)
        print("   🔄 Pre-loading 71-class model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = globals_128['load_model'](
            str(model_path), 
            n_in=12, 
            n_out=n_out, 
            device=device
        )
        if model is not None:
            model.eval()  # Set to evaluation mode
            print("   ✅ 71-class model pre-loaded and set to eval mode")
            globals_128['model'] = model
            globals_128['device'] = device
            globals_128['n_out'] = n_out  # Store actual output size
        else:
            raise ValueError("Failed to load model")
        
        pred_module = globals_128
        
        print("   ✅ Model and preprocessing components loaded successfully!")
        return pred_module, scaler, mlb, n_out
        
    except Exception as e:
        print(f"   ❌ Error loading model and preprocessing: {e}")
        raise

def load_sph_dataset(data_path, debug_mode=False, max_samples=None):
    """Load SPH dataset from H5 files"""
    print(f"📂 Loading SPH dataset from: {data_path}")
    
    # Load metadata
    metadata_path = Path(data_path) / "metadata.csv"
    code_path = Path(data_path) / "code.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not code_path.exists():
        raise FileNotFoundError(f"Code file not found: {code_path}")
    
    # Load metadata and code mappings
    metadata_df = pd.read_csv(metadata_path)
    code_df = pd.read_csv(code_path)
    
    print(f"   📊 Loaded metadata: {len(metadata_df)} records")
    print(f"   📊 Loaded code mappings: {len(code_df)} codes")
    
    # Create code to description mapping
    code_to_desc = dict(zip(code_df['Code'].astype(str), code_df['Description']))
    
    # Define target classes (13 classes we want to evaluate)
    TARGET_ECG_CLASSES = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']
    
    # Load ECG data and labels
    records_dir = Path(data_path) / "records"
    if not records_dir.exists():
        raise FileNotFoundError(f"Records directory not found: {records_dir}")
    
    # Get all H5 files
    h5_files = list(records_dir.glob("*.h5"))
    print(f"   📁 Found {len(h5_files)} H5 files")
    
    if debug_mode and max_samples:
        h5_files = h5_files[:max_samples]
        print(f"   🔍 Debug mode: Processing {len(h5_files)} files")
    
    X_data = []
    y_data = []
    record_ids = []
    successful_loads = 0
    
    print("   📥 Loading ECG data...")
    
    for i, h5_file in enumerate(h5_files):
        if _shutdown_requested:
            print("   🛑 Shutdown requested, stopping data loading...")
            break
            
        try:
            # Extract ECG ID from filename
            ecg_id = h5_file.stem
            
            # Find corresponding metadata
            record_meta = metadata_df[metadata_df['ECG_ID'] == ecg_id]
            if record_meta.empty:
                if i < 10:  # Show first 10 errors
                    print(f"   ⚠️ No metadata found for {ecg_id}")
                continue
            
            # Get AHA codes for this record
            aha_codes = record_meta.iloc[0]['AHA_Code']
            if pd.isna(aha_codes):
                if i < 10:  # Show first 10 errors
                    print(f"   ⚠️ No AHA codes for {ecg_id}")
                continue
            
            # Parse AHA codes (format: "22;23", "30+310", or single code)
            if isinstance(aha_codes, str):
                # First split by semicolon, then by plus sign to handle modifiers
                codes = []
                for code_group in aha_codes.split(';'):
                    code_group = code_group.strip()
                    if '+' in code_group:
                        # Handle modifiers (e.g., "30+310" -> ["30", "310"])
                        base_code, modifier = code_group.split('+')
                        codes.extend([base_code.strip(), modifier.strip()])
                    else:
                        codes.append(code_group)
            else:
                codes = [str(aha_codes)]
            
            # Load ECG data from H5 file
            with h5py.File(h5_file, 'r') as f:
                ecg_data = f['ecg'][:]  # Shape: (12, variable_length) - (channels, timesteps)
            
            # SPH dataset has variable lengths, standardize to 1000 samples (100Hz)
            # Most are 500Hz, some are other lengths, we need to standardize
            original_length = ecg_data.shape[1]
            
            if original_length >= 5000:  # 500Hz or higher frequency data
                # Downsample from 500Hz to 100Hz (take every 5th sample)
                ecg_data = ecg_data[:, ::5]  # Shape: (12, original_length/5)
                # If still longer than 1000, truncate to 1000
                if ecg_data.shape[1] > 1000:
                    ecg_data = ecg_data[:, :1000]
                # If shorter than 1000, pad with zeros
                elif ecg_data.shape[1] < 1000:
                    padding = np.zeros((12, 1000 - ecg_data.shape[1]), dtype=ecg_data.dtype)
                    ecg_data = np.concatenate([ecg_data, padding], axis=1)
            elif original_length >= 1000:  # Already close to 100Hz or higher
                # Truncate or pad to exactly 1000 samples
                if original_length > 1000:
                    ecg_data = ecg_data[:, :1000]
                else:
                    padding = np.zeros((12, 1000 - original_length), dtype=ecg_data.dtype)
                    ecg_data = np.concatenate([ecg_data, padding], axis=1)
            else:
                # Very short signals, pad to 1000
                padding = np.zeros((12, 1000 - original_length), dtype=ecg_data.dtype)
                ecg_data = np.concatenate([ecg_data, padding], axis=1)
            
            # Ensure final shape is (12, 1000)
            if ecg_data.shape != (12, 1000):
                if i < 10:  # Show first 10 errors
                    print(f"   ⚠️ Failed to standardize ECG length for {ecg_id}: {ecg_data.shape}")
                continue
            
            # Create label vector for target classes
            label_vector = np.zeros(len(TARGET_ECG_CLASSES), dtype=np.int32)
            
            # Map SPH codes to our target classes
            for code in codes:
                if code in SPH_TO_OUR_MAPPING:
                    mapped_class = SPH_TO_OUR_MAPPING[code]
                    if mapped_class in TARGET_ECG_CLASSES:
                        target_idx = TARGET_ECG_CLASSES.index(mapped_class)
                        label_vector[target_idx] = 1
            
            # Only include samples that have at least one recognized target class
            if np.sum(label_vector) > 0:
                X_data.append(ecg_data)  # Keep as (samples, channels) format to match extract_and_predict_ecg.py
                y_data.append(label_vector)
                record_ids.append(ecg_id)
                successful_loads += 1
                
                if debug_mode and i < 10:  # Show first 10 successful loads in debug mode
                    mapped_classes = [TARGET_ECG_CLASSES[j] for j in range(len(TARGET_ECG_CLASSES)) if label_vector[j] == 1]
                    # Show code descriptions
                    code_descriptions = []
                    for code in codes:
                        if code in code_to_desc:
                            code_descriptions.append(f"{code}({code_to_desc[code]})")
                        else:
                            code_descriptions.append(f"{code}(unknown)")
                    print(f"   ✅ {ecg_id}: {aha_codes} -> codes: {codes} -> descriptions: {code_descriptions} -> mapped: {mapped_classes}")
            
            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"   📊 Progress: {i + 1}/{len(h5_files)} ({100 * (i + 1) / len(h5_files):.1f}%) - Loaded: {successful_loads}")
                
        except Exception as e:
            if debug_mode and i < 10:  # Show first 10 errors in debug mode
                print(f"   ⚠️ Failed to load {h5_file.name}: {e}")
            continue
    
    if not X_data:
        raise ValueError("No valid ECG data loaded from SPH dataset")
    
    # Convert to numpy arrays
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int32)
    
    print(f"   ✅ Successfully loaded {len(X_data)} ECG records")
    print(f"   📊 ECG data shape: {X_data.shape}")
    print(f"   📊 Label data shape: {y_data.shape}")
    
    # Print class distribution
    print("   📈 Class distribution in loaded data:")
    for i, class_name in enumerate(TARGET_ECG_CLASSES):
        count = np.sum(y_data[:, i])
        if count > 0:
            print(f"      {class_name}: {count} samples ({100 * count / len(y_data):.1f}%)")
    
    return X_data, y_data, TARGET_ECG_CLASSES, record_ids

def extract_ecg_from_sph(h5_file_path):
    """Extract ECG data from SPH H5 file and standardize to 1000 samples"""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            ecg_data = f['ecg'][:]  # Shape: (12, variable_length) - (channels, timesteps)
        
        # SPH dataset has variable lengths, standardize to 1000 samples (100Hz)
        original_length = ecg_data.shape[1]
        
        if original_length >= 5000:  # 500Hz or higher frequency data
            # Downsample from 500Hz to 100Hz (take every 5th sample)
            ecg_data = ecg_data[:, ::5]  # Shape: (12, original_length/5)
            # If still longer than 1000, truncate to 1000
            if ecg_data.shape[1] > 1000:
                ecg_data = ecg_data[:, :1000]
            # If shorter than 1000, pad with zeros
            elif ecg_data.shape[1] < 1000:
                padding = np.zeros((12, 1000 - ecg_data.shape[1]), dtype=ecg_data.dtype)
                ecg_data = np.concatenate([ecg_data, padding], axis=1)
        elif original_length >= 1000:  # Already close to 100Hz or higher
            # Truncate or pad to exactly 1000 samples
            if original_length > 1000:
                ecg_data = ecg_data[:, :1000]
            else:
                padding = np.zeros((12, 1000 - original_length), dtype=ecg_data.dtype)
                ecg_data = np.concatenate([ecg_data, padding], axis=1)
        else:
            # Very short signals, pad to 1000
            padding = np.zeros((12, 1000 - original_length), dtype=ecg_data.dtype)
            ecg_data = np.concatenate([ecg_data, padding], axis=1)
        
        # Ensure final shape is (12, 1000)
        if ecg_data.shape != (12, 1000):
            raise ValueError(f"Failed to standardize ECG length: {ecg_data.shape}")
        
        return ecg_data.T  # Transpose to (timesteps, channels) format
    except Exception as e:
        raise ValueError(f"Failed to extract ECG from {h5_file_path}: {e}")

def predict_batch_sph(record_ids, pred_module, scaler, mlb, n_out, actual_classes=None, debug_mode=False, batch_size=100):
    """Predict batch of SPH records with memory-efficient processing"""
    print(f"🔮 Predicting {len(record_ids)} SPH records...")
    
    if actual_classes is None:
        actual_classes = list(range(len(record_ids)))
    
    # Get target classes
    TARGET_ECG_CLASSES = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']
    
    # Get model and device from pred_module
    model = pred_module['model']
    device = pred_module['device']
    
    predictions = []
    successful_predictions = 0
    total_samples = len(record_ids)
    
    # Progress tracking (reduced frequency for better performance)
    progress_interval = max(1, total_samples // 20)  # Update every 5%
    
    # Process in chunks for memory efficiency (same as Chapman script)
    for chunk_start in range(0, total_samples, batch_size):
        if _shutdown_requested:
            print("   🛑 Shutdown requested, stopping prediction...")
            break
            
        chunk_end = min(chunk_start + batch_size, total_samples)
        chunk_ids = record_ids[chunk_start:chunk_end]
        
        print(f"   📦 Processing chunk {chunk_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} ({len(chunk_ids)} samples)")
        
        for i, record_id in enumerate(chunk_ids):
            global_index = chunk_start + i
            
            # Progress update (reduced frequency)
            if global_index % progress_interval == 0 or global_index == total_samples - 1:
                progress = (global_index + 1) / total_samples * 100
                print(f"   📊 Progress: {global_index+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
            
            try:
                # Load ECG data
                h5_file_path = Path("data/SPH/records") / f"{record_id}.h5"
                ecg_data = extract_ecg_from_sph(h5_file_path)  # Returns (1000, 12) - (samples, channels)
                
                # Apply same preprocessing as extract_and_predict_ecg.py
                # 1. Convert to float32 and transpose to (channels, samples)
                sig = ecg_data.astype(np.float32).T  # Transpose from (1000, 12) to (12, 1000)
                
                # 2. Apply scaler using the same method as extract_and_predict_ecg.py
                # Flatten ECG data to 1D, apply scaler, then reshape back to original shape
                x_shape = sig.shape
                x_flat = sig.flatten()[:, np.newaxis]  # Flatten to column vector
                x_scaled = scaler.transform(x_flat).reshape(x_shape)  # Apply scaler and reshape
                
                # 3. Convert to PyTorch tensor and add batch dimension
                x = torch.from_numpy(x_scaled).unsqueeze(0)  # Add batch dimension -> (1, 12, 1000)
                x = x.to(device)
                
                # 4. Predict using model (same as ICBEB script)
                with torch.no_grad():
                    logits = model(x)
                    
                    # Check for NaN/Inf in logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        if global_index < 10:
                            print(f"   ⚠️ NaN/Inf detected in logits at sample {global_index}")
                        continue
                    
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                    
                    # Check for NaN/Inf in probabilities
                    if np.isnan(probs).any() or np.isinf(probs).any():
                        if global_index < 10:
                            print(f"   ⚠️ NaN/Inf detected in probabilities at sample {global_index}")
                        continue
                
                # Debug: print prediction info (only in debug mode)
                if debug_mode and global_index < 3:
                    print(f"   🔍 Prediction Debug:")
                    print(f"      Input shape: {x.shape}")
                    print(f"      Logits shape: {logits.shape}")
                    print(f"      Probabilities shape: {probs.shape}")
                    print(f"      Max probability: {probs.max():.6f}")
                    print(f"      Min probability: {probs.min():.6f}")
                    print(f"      Mean probability: {probs.mean():.6f}")
                
                # Map 71-class predictions to target classes
                pred_probs = np.zeros(len(TARGET_ECG_CLASSES))
                
                # Map each target class to corresponding class in 71-class model
                for j, target_class in enumerate(TARGET_ECG_CLASSES):
                    # Find corresponding class index in 71-class model
                    if target_class in MODEL_71_CLASSES:
                        class_idx = MODEL_71_CLASSES.index(target_class)
                        if class_idx < len(probs):
                            pred_probs[j] = probs[class_idx]
                        else:
                            pred_probs[j] = 0.0
                    else:
                        pred_probs[j] = 0.0
                
                predictions.append(pred_probs)
                successful_predictions += 1
                
                # Debug: print mapped predictions (only in debug mode)
                if debug_mode and global_index < 3:
                    print(f"   🔍 Mapped Predictions:")
                    for k, cls in enumerate(TARGET_ECG_CLASSES):
                        print(f"      {cls}: {pred_probs[k]:.6f}")
                
            except Exception as e:
                if debug_mode and global_index < 10:
                    print(f"   ⚠️ Failed to predict {record_id}: {e}")
                continue
        
        # Clear GPU cache after each chunk (same as Chapman script)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Additional memory management for large datasets
        if chunk_start % (batch_size * 5) == 0:  # Every 5 chunks
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
    
    if not predictions:
        raise ValueError("No successful predictions generated")
    
    predictions = np.array(predictions)
    print(f"   ✅ Successfully predicted {len(predictions)} samples")
    print(f"   📊 Predictions shape: {predictions.shape}")
    
    # Print prediction statistics
    print(f"   🔍 Prediction Statistics:")
    print(f"      Min probability: {np.min(predictions):.6f}")
    print(f"      Max probability: {np.max(predictions):.6f}")
    print(f"      Mean probability: {np.mean(predictions):.6f}")
    print(f"      Std probability: {np.std(predictions):.6f}")
    
    return predictions

def extract_target_classes(y_data, target_class_names):
    """Extract target classes from y_data"""
    print("🎯 Extracting target classes...")
    
    # y_data is already in target format, just return it
    y_target = y_data
    
    print(f"   📊 Target classes shape: {y_target.shape}")
    print(f"   📊 Target class names: {target_class_names}")
    
    return y_target, target_class_names

def evaluate_performance(y_true, y_pred, target_class_names):
    """Evaluate model performance"""
    print("📊 Evaluating performance...")
    
    results = []
    
    for i, class_name in enumerate(target_class_names):
        if i >= y_true.shape[1] or i >= y_pred.shape[1]:
            continue
            
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        # Skip if no positive samples
        if np.sum(y_true_class) == 0:
            continue
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_true_class, y_pred_class)
        except ValueError:
            auc = 0.5
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_class > 0.5).astype(int)
        
        # Calculate precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_class, y_pred_binary, average='binary', zero_division=0
        )
        
        # Calculate support manually if it's None
        if support is None:
            support = np.sum(y_true_class)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_binary).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'Class': class_name,
            'Support': int(support),
            'AUC': round(auc, 4),
            'F1-Score': round(f1, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'Specificity': round(specificity, 4),
            'True Positives': int(tp),
            'False Positives': int(fp),
            'False Negatives': int(fn),
            'True Negatives': int(tn)
        })
    
    return results

def create_visualizations(results, output_dir):
    """Create confusion matrices and ROC curves"""
    print("📊 Creating visualizations...")
    
    # Prepare data for visualization
    class_names = [r['Class'] for r in results]
    n_classes = len(class_names)
    
    if n_classes == 0:
        print("   ⚠️ No classes to visualize")
        return
    
    # Create subplots
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_classes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, n_cols)
    
    for i, result in enumerate(results):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            ax = axes[0]
        elif n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Create confusion matrix
        tp = result['True Positives']
        fp = result['False Positives']
        fn = result['False Negatives']
        tn = result['True Negatives']
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_title(f"{result['Class']}\nF1: {result['F1-Score']:.3f}, AUC: {result['AUC']:.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_classes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save confusion matrices
    confusion_matrix_path = output_dir / "confusion_matrices.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Confusion matrices saved: {confusion_matrix_path}")
    
    plt.close()
    print("   ✅ Visualizations created")

def save_results(results, predictions, y_target, target_class_names, record_ids, output_dir):
    """Save evaluation results"""
    print("💾 Saving results...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.index.name = 'Class'
    
    # Save summary results
    summary_path = output_dir / "results.csv"
    results_df.to_csv(summary_path, encoding='utf-8')
    print(f"   ✅ Summary results saved: {summary_path}")
    
    # Save detailed results
    detailed_path = output_dir / "detailed_results.csv"
    results_df.to_csv(detailed_path, encoding='utf-8')
    print(f"   ✅ Detailed results saved: {detailed_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions, columns=target_class_names)
    predictions_df['Record_ID'] = record_ids[:len(predictions)]
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding='utf-8')
    print(f"   ✅ Predictions saved: {predictions_path}")
    
    # Save true labels
    true_labels_df = pd.DataFrame(y_target, columns=target_class_names)
    true_labels_df['Record_ID'] = record_ids[:len(y_target)]
    true_labels_path = output_dir / "true_labels.csv"
    true_labels_df.to_csv(true_labels_path, index=False, encoding='utf-8')
    print(f"   ✅ True labels saved: {true_labels_path}")
    
    print(f"   📁 Files created:")
    print(f"      - {summary_path.name}")
    print(f"      - {detailed_path.name}")
    print(f"      - {predictions_path.name}")
    print(f"      - {true_labels_path.name}")

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        print("🔬 SPH Dataset Evaluation with 71-Class Model")
        print("=" * 60)
        print(f"📁 Experiment folder: {args.exp_folder}")
        print(f"📁 SPH data folder: data/SPH")
        
        # Load model and preprocessing
        pred_module, scaler, mlb, n_out = load_model_and_preprocessing(args.exp_folder)
        
        # Load SPH dataset
        X_data, y_data, target_class_names, record_ids = load_sph_dataset("data/SPH", debug_mode=False, max_samples=None)
        
        # Extract target classes
        y_target, actual_classes = extract_target_classes(y_data, target_class_names)
        
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = extract_model_name_from_path("fastai_xresnet1d101")
        output_dir = Path("output/prediction/SPH_test/batch_sph_71class_results") / f"run_{timestamp}_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        global OUTPUT_DIR
        OUTPUT_DIR = output_dir
        
        print(f"📁 Output directory: {output_dir}")
        
        # Predict
        predictions = predict_batch_sph(record_ids, pred_module, scaler, mlb, n_out, actual_classes, debug_mode=False, batch_size=100)
        
        # Check for sample count mismatch and fix if needed
        if len(predictions) != len(y_target):
            print(f"⚠️ Warning: Sample count mismatch - predictions: {len(predictions)}, targets: {len(y_target)}")
            # Use the minimum count to ensure consistency
            min_count = min(len(predictions), len(y_target))
            predictions = predictions[:min_count]
            y_target = y_target[:min_count]
            print(f"   📊 Adjusted to use {min_count} samples for evaluation")
        
        # Evaluate performance
        results = evaluate_performance(y_target, predictions, target_class_names)
        
        # Print results
        print("\n" + "=" * 70)
        print("📊 71-Class Model SPH Dataset Evaluation Results:")
        print("=" * 70)
        
        # Create results table
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            print(results_df.to_string(index=False, float_format='%.4f'))
        else:
            print("No results to display")
        
        # Save results
        save_results(results, predictions, y_target, target_class_names, record_ids, output_dir)
        
        # Create visualizations
        create_visualizations(results, output_dir)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
