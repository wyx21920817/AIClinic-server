#!/usr/bin/env python3
"""
Ultra-Fast Batch Chapman Shaoxing Prediction Script for 13-Class Models
Chapman Shaoxing evaluation script using WFDB format for 13-class models

Usage:
    python batch_predict_chapman_13class.py [exp_folder]
    
Examples:
    python batch_predict_chapman_13class.py                                    # Use default model
    python batch_predict_chapman_13class.py output/exp_13class_smart_multilabel  # Use specified experiment folder
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
import wfdb
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import signal
import warnings
import argparse
import csv
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Chapman to our class mapping (only map classes that directly correspond to our 13 target classes)
CHAPMAN_TO_OUR_MAPPING = {
    # Direct category mapping (based on actual Chapman dataset analysis)
    'AFIB': 'AFIB',          # Atrial fibrillation (164889003) - 1780次
    '1AVB': '1AVB',          # First degree atrioventricular block (270492004) - 1140次
    '2AVB': '2AVB',          # Second degree atrioventricular block (195042002) - 66次
    '2AVB1': '2AVB',         # 2 degree AVB Type one
    '2AVB2': '2AVB',         # 2 degree AVB Type two
    '3AVB': '3AVB',          # Third degree atrioventricular block (27885002) - 76次
    'AF': 'AFLT',            # Atrial Flutter (164890007) - 8060次
    'LBBB': 'CLBBB',         # Left bundle branch block (164909002) -> CLBBB
    'RBBB': 'CRBBB',         # Right bundle branch block (59118001) - 649次
    'LBBBB': 'CLBBB',        # Left back bundle branch block -> CLBBB
    'LFBBB': 'ILBBB',        # Left front bundle branch block -> ILBBB
    'APB': 'PAC',            # Atrial premature beats (284470004) - 1312次
    'VPB': 'PVC',            # Ventricular premature beat -> PVC
    'SR': 'NORM',            # Sinus Rhythm (426783006) - 8125次 -> NORM (正常窦性心律)
    'ST': 'STACH',           # Sinus Tachycardia (427084000) - 7255次 -> STACH
    'STTC': 'STACH',         # ST-T Change (428750005) - 1158次
    'STDD': 'STACH',         # ST drop down (429622005) - 1668次
    'STE': 'STACH',          # ST extension (164930006) - 801次
    'STTU': 'STACH',         # ST tilt up -> STACH
}

# Target ECG classes we want to evaluate (same as other scripts)
TARGET_ECG_CLASSES = [
    'NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'ILBBB', 'CLBBB', 
    'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC'
]

# Display name mapping (for output display)
DISPLAY_NAME_MAPPING = {
    'NORM': 'NORM',
    'AFIB': 'AFIB', 
    '1AVB': '1AVB',
    '2AVB': '2AVB',
    '3AVB': '3AVB',
    'CLBBB': 'CLBBB',
    'CRBBB': 'CRBBB',
    'ILBBB': 'ILBBB',
    'IRBBB': 'IRBBB',
    'STACH': 'STACH',
    'PAC': 'PAC',
    'PVC': 'PVC'
}

# Global variable to track shutdown request
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signal gracefully"""
    global _shutdown_requested
    print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Ultra-Fast Batch Chapman Shaoxing Prediction Script for 13-Class Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_predict_chapman_13class.py                                    # Use default model
    python batch_predict_chapman_13class.py output/exp_13class_smart_multilabel  # Use specified experiment folder
        """,
        add_help=True
    )
    
    parser.add_argument(
        'exp_folder', 
        nargs='?',  # Optional positional argument
        default="output/exp_13class_smart_multilabel",  # Default use 13-class model
        help='Experiment folder path containing models/ and data/ directories (default: output/exp_13class_smart_multilabel)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode with verbose output'
    )
    
    parser.add_argument(
        '--max-samples', 
        type=int,
        help='Maximum number of samples to process (for debugging)'
    )
    
    return parser.parse_args()

def load_model_and_preprocessing(exp_folder):
    """Load 13-class model and preprocessing components"""
    try:
        print(f"🔧 Loading 13-class model and preprocessing components from: {exp_folder}")
        
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
        
        # Load class names from model
        class_names_path = data_dir / "class_names.txt"
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            print(f"   ⚠️  Class names file not found, using default 13 classes")
            class_names = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']
        
        print(f"   ✅ Loaded scaler and mlb from: {data_dir}")
        print(f"   📋 Model has {len(class_names)} classes")
        
        # Verify Chapman mapped classes exist in 13-class model
        print("   🔍 Verifying Chapman class mappings:")
        for chapman_class, our_class in CHAPMAN_TO_OUR_MAPPING.items():
            if our_class in class_names:
                idx = class_names.index(our_class)
                print(f"      {chapman_class:10s} -> {our_class:6s} (index: {idx:2d}) ✅")
            else:
                print(f"      {chapman_class:10s} -> {our_class:6s} (not found) ❌")
        
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
            else:
                # Try to find weight layer
                weight_key = last_key.replace('.bias', '.weight')
                if weight_key in model_weights:
                    n_out = model_weights[weight_key].shape[0]
                else:
                    n_out = 13  # Default 13 classes
        else:
            n_out = 13  # Default 13 classes
        
        print(f"   📊 Detected model output classes: {n_out}")
        
        # Pre-load model and set to evaluation mode
        print("   🔄 Pre-loading 13-class model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = globals_128['load_model'](
            str(model_path), 
            n_in=12, 
            n_out=n_out, 
            device=device
        )
        if model is not None:
            model.eval()  # Set to evaluation mode
            print("   ✅ 13-class model pre-loaded and set to eval mode")
            globals_128['model'] = model
            globals_128['device'] = device
            globals_128['n_out'] = n_out  # Store actual output size
        else:
            raise RuntimeError("Model loading returned None")
        
        # Store model path in globals for later use
        globals_128['model_path'] = str(model_path)
        
        print("✅ 13-class model and preprocessing components loaded")
        return globals_128, scaler, mlb, n_out, class_names
        
    except Exception as e:
        print(f"❌ Failed to load 13-class model: {e}")
        return None, None, None, None, None

def extract_ecg_from_wfdb(wfdb_path, debug_mode=False):
    """Extract ECG data from WFDB file"""
    try:
        # Read WFDB record
        record = wfdb.rdrecord(wfdb_path)
        
        if debug_mode:
            print(f"   🔍 WFDB Debug - Record info:")
            print(f"      Signal shape: {record.p_signal.shape}")
            print(f"      Signal dtype: {record.p_signal.dtype}")
            print(f"      Signal range: [{record.p_signal.min():.3f}, {record.p_signal.max():.3f}]")
            print(f"      Sampling rate: {record.fs} Hz")
        
        # Get ECG signal (p_signal is already in float format)
        ecg_data = record.p_signal.astype(np.float32)
        
        # Transpose to (channels, samples) format
        ecg_data = ecg_data.T
        
        # Chapman dataset is 500Hz, need to downsample to 100Hz to match model expectations
        if record.fs == 500:
            # Downsample from 500Hz to 100Hz (take every 5th sample)
            ecg_data = ecg_data[:, ::5]
            if debug_mode:
                print(f"      Downsampled from 500Hz to 100Hz: {ecg_data.shape}")
        elif record.fs != 100:
            if debug_mode:
                print(f"      Warning: Unexpected sampling rate {record.fs}Hz, expected 100Hz or 500Hz")
        
        if debug_mode:
            print(f"      After transpose: {ecg_data.shape}")
            print(f"      WFDB data already normalized: range [{ecg_data.min():.6f}, {ecg_data.max():.6f}]")
        
        return ecg_data
        
    except Exception as e:
        print(f"❌ Failed to extract ECG from {wfdb_path}: {e}")
        return None

def load_chapman_dataset(data_path, debug_mode=False, max_samples=None):
    """Load Chapman dataset and extract target classes"""
    print("📊 Loading Chapman dataset...")
    
    # Load condition names mapping
    condition_df = pd.read_csv(data_path / 'ConditionNames_SNOMED-CT.csv')
    print(f"   📋 Loaded {len(condition_df)} condition mappings")
    
    # Create SNOMED-CT code to acronym mapping
    snomed_to_acronym = {}
    for _, row in condition_df.iterrows():
        snomed_code = str(row['Snomed_CT']).strip()
        acronym = row['Acronym Name'].strip()
        snomed_to_acronym[snomed_code] = acronym
    
    print(f"   🔍 Created SNOMED-CT mapping for {len(snomed_to_acronym)} codes")
    
    # Load RECORDS file
    records_path = data_path / 'RECORDS'
    if not records_path.exists():
        raise FileNotFoundError(f"RECORDS file not found: {records_path}")
    
    with open(records_path, 'r') as f:
        record_lines = f.readlines()
    
    # Parse record paths - Chapman dataset has nested structure
    wfdb_paths = []
    for line in record_lines:
        line = line.strip()
        if line:
            # This is a directory path like "WFDBRecords/01/010/"
            dir_path = data_path / line
            
            if dir_path.exists():
                # Read the RECORDS file in this directory
                records_file = dir_path / 'RECORDS'
                if records_file.exists():
                    with open(records_file, 'r') as f:
                        sub_records = f.readlines()
                    
                    # Add each record in this directory
                    for sub_record in sub_records:
                        sub_record = sub_record.strip()
                        if sub_record:
                            # Full path to the WFDB record (without extension)
                            wfdb_path = dir_path / sub_record
                            wfdb_paths.append(str(wfdb_path))
                else:
                    print(f"   ⚠️ Warning: RECORDS file not found in {dir_path}")
            else:
                print(f"   ⚠️ Warning: Directory not found: {dir_path}")
    
    print(f"   📁 Found {len(wfdb_paths)} valid records")
    
    # Limit samples if specified
    if max_samples and max_samples < len(wfdb_paths):
        wfdb_paths = wfdb_paths[:max_samples]
        print(f"   🔢 Limited to {max_samples} samples for debugging")
    
    # Extract target classes from each record
    X_data = []
    y_data = []
    record_ids = []
    
    # Get Chapman classes that map to our target classes
    chapman_classes = list(CHAPMAN_TO_OUR_MAPPING.keys())
    print(f"   🎯 Target Chapman classes: {chapman_classes}")
    
    successful_loads = 0
    
    for i, wfdb_path in enumerate(wfdb_paths):
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\n🛑 Interrupt request received, processed {i}/{len(wfdb_paths)} samples")
            break
        
        # Progress update
        if i % 50 == 0 or i == len(wfdb_paths) - 1:
            progress = (i + 1) / len(wfdb_paths) * 100
            print(f"   📊 Progress: {i+1}/{len(wfdb_paths)} ({progress:.1f}%) - Loaded: {successful_loads}")
        
        try:
            # Extract ECG data
            ecg_data = extract_ecg_from_wfdb(wfdb_path, debug_mode and i < 3)
            
            if ecg_data is None:
                continue
            
            # Read .hea file to get diagnosis from comments
            record_path = Path(wfdb_path)
            diagnosis_codes = []
            
            try:
                hea_path = record_path.with_suffix('.hea')
                with open(hea_path, 'r') as f:
                    hea_content = f.read()
                
                # Look for #Dx: line in .hea file
                for line in hea_content.split('\n'):
                    if line.strip().startswith('#Dx:'):
                        # Extract SNOMED-CT codes
                        dx_line = line.strip()
                        codes_str = dx_line.replace('#Dx:', '').strip()
                        if codes_str and codes_str != 'Unknown':
                            diagnosis_codes = [code.strip() for code in codes_str.split(',')]
                        break
                        
            except Exception as e:
                if debug_mode and i < 10:
                    print(f"   ⚠️ Failed to read .hea file {hea_path}: {e}")
            
            # Convert SNOMED-CT codes to acronyms
            diagnosis_acronyms = []
            for code in diagnosis_codes:
                if code in snomed_to_acronym:
                    diagnosis_acronyms.append(snomed_to_acronym[code])
                elif debug_mode and i < 10:
                    print(f"   ⚠️ Unknown SNOMED-CT code: {code}")
            
            # Create label vector for target classes (only map classes that have direct correspondence)
            label_vector = np.zeros(len(TARGET_ECG_CLASSES), dtype=np.int32)
            
            # Map diagnosis acronyms to our target classes (only if there's a direct mapping)
            for diag_acronym in diagnosis_acronyms:
                if diag_acronym in CHAPMAN_TO_OUR_MAPPING:
                    target_class = CHAPMAN_TO_OUR_MAPPING[diag_acronym]
                    if target_class in TARGET_ECG_CLASSES:
                        target_idx = TARGET_ECG_CLASSES.index(target_class)
                        label_vector[target_idx] = 1
            
            # Only keep samples that have at least one target class label
            if np.sum(label_vector) > 0:
                X_data.append(ecg_data)
                y_data.append(label_vector)
                record_ids.append(Path(wfdb_path).name)
                successful_loads += 1
                
                if debug_mode and i < 5:
                    print(f"   🔍 Sample {i}: {Path(wfdb_path).name}")
                    print(f"      Diagnoses: {diagnosis_acronyms}")
                    print(f"      Target labels: {[TARGET_ECG_CLASSES[j] for j in range(len(TARGET_ECG_CLASSES)) if label_vector[j] == 1]}")
        
        except Exception as e:
            if debug_mode and i < 10:
                print(f"   ❌ Failed to process {wfdb_path}: {e}")
            continue
    
    if len(X_data) == 0:
        raise ValueError("No valid samples found with target class labels")
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"   ✅ Successfully loaded {len(X_data)} samples with target class labels")
    print(f"   📊 Data shapes: X={X_data.shape}, y={y_data.shape}")
    
    # Print target class distribution
    print(f"   📈 Target class distribution:")
    for i, class_name in enumerate(TARGET_ECG_CLASSES):
        count = np.sum(y_data[:, i])
        print(f"      {class_name}: {count} samples")
    
    return X_data, y_data, TARGET_ECG_CLASSES

def predict_batch_chapman_optimized(X_data, pred_module, scaler, mlb, n_out, class_names, actual_classes=None, debug_mode=False):
    """Fast batch prediction for Chapman dataset"""
    print(f"🚀 Starting optimized batch prediction for {len(X_data)} Chapman samples...")
    
    predictions = []
    successful_predictions = 0
    failed_predictions = 0
    
    # Pre-fetch model and device
    model = pred_module['model']
    device = pred_module['device']
    
    # Set warning flag
    warning_shown = False
    
    # Progress tracking
    total_samples = len(X_data)
    progress_interval = max(1, total_samples // 20)  # Update every 5%
    
    for i, ecg_data in enumerate(X_data):
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\n🛑 Interrupt request received, processed {i}/{total_samples} samples")
            break
        
        # Progress update
        if i % progress_interval == 0 or i == total_samples - 1:
            progress = (i + 1) / total_samples * 100
            print(f"   📊 Progress: {i+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
            
        try:
            # Preprocessing - use same method as 13-class model training
            # ECG data is already in (channels, timesteps) format from extract_ecg_from_wfdb
            
            # Flatten ECG data to 1D, apply scaler, then reshape back to original shape
            x_shape = ecg_data.shape
            x_flat = ecg_data.flatten()[:, np.newaxis]  # Flatten to column vector
            x_scaled = scaler.transform(x_flat).reshape(x_shape)  # Apply scaler and reshape
            
            x = torch.from_numpy(x_scaled).unsqueeze(0).float()  # Add batch dimension and convert to float -> (1, 12, 1000)
            
            x = x.to(device)
            
            # Inference (model already in eval mode, no need to call again)
            with torch.no_grad():
                logits = model(x)
                
                # Check for NaN in logits before sigmoid
                if debug_mode and torch.isnan(logits).any():
                    nan_positions = torch.isnan(logits)
                    print(f"   ⚠️ Warning: NaN values found in logits at positions: {torch.where(nan_positions)}")
                    print(f"      Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
                    
                    # Fix: Replace NaN logits with zeros to prevent propagation
                    print(f"   🔧 Fixing NaN logits by replacing with zeros...")
                    logits = torch.nan_to_num(logits, nan=0.0)
                    print(f"   ✅ NaN logits fixed")
                
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                
                # Check for NaN in probabilities after sigmoid
                if debug_mode and np.isnan(probs).any():
                    nan_positions = np.isnan(probs)
                    print(f"   ⚠️ Warning: NaN values found in probabilities at positions: {np.where(nan_positions)}")
                    print(f"      Probabilities range: [{probs.min():.6f}, {probs.max():.6f}]")
                    
                    # Fix: Replace NaN probabilities with zeros
                    print(f"   🔧 Fixing NaN probabilities by replacing with zeros...")
                    probs = np.nan_to_num(probs, nan=0.0)
                    print(f"   ✅ NaN probabilities fixed")
                
                # Debug: print prediction info (only in debug mode)
                if debug_mode:
                    print(f"   🔍 Prediction Debug:")
                    print(f"      Input shape: {x.shape}")
                    print(f"      Logits shape: {logits.shape}")
                    print(f"      Probabilities shape: {probs.shape}")
                    print(f"      Max probability: {probs.max():.6f}")
                    print(f"      Min probability: {probs.min():.6f}")
                    print(f"      Mean probability: {probs.mean():.6f}")
            
            # Map 13-class predictions to Chapman-related classes
            # Create mapped prediction probability array (only for classes present in Chapman)
            chapman_classes = list(CHAPMAN_TO_OUR_MAPPING.values())
            pred_probs = np.zeros(len(chapman_classes))
            
            # Map each Chapman class to corresponding class in 13-class model
            for j, chapman_class in enumerate(chapman_classes):
                # Find corresponding class index in 13-class model
                if chapman_class in class_names:
                    class_idx = class_names.index(chapman_class)
                    if class_idx < len(probs):
                        pred_probs[j] = probs[class_idx]
                    else:
                        pred_probs[j] = 0.0
                        if not warning_shown:
                            print(f"⚠️ Class {chapman_class} index {class_idx} in 13-class model exceeds prediction probability range")
                else:
                    pred_probs[j] = 0.0
                    if not warning_shown:
                        print(f"⚠️ Class {chapman_class} does not exist in 13-class model")
            
            predictions.append(pred_probs)
            successful_predictions += 1
            
            # Debug: print mapped predictions (only in debug mode)
            if debug_mode:
                print(f"   🔍 Mapped Predictions Debug:")
                for j, (chapman_class, prob) in enumerate(zip(chapman_classes, pred_probs)):
                    print(f"      {chapman_class}: {prob:.6f}")
                print(f"      Max mapped probability: {pred_probs.max():.6f}")
                print(f"      Min mapped probability: {pred_probs.min():.6f}")
            
            # Show warning message (only once)
            if not warning_shown and n_out != len(chapman_classes):
                print(f"   ⚠️  Model has {n_out} classes, mapping to {len(chapman_classes)} Chapman classes.")
                warning_shown = True
            
        except Exception as e:
            print(f"   ❌ Failed to process sample {i}: {e}")
            failed_predictions += 1
            # Add zero prediction to maintain consistent length
            predictions.append(np.zeros(len(CHAPMAN_TO_OUR_MAPPING)))
        
        # Reduce progress update frequency
        if (i + 1) % 200 == 0:
            print(f"   📊 Progress: {i+1}/{len(X_data)} ({(i+1)/len(X_data)*100:.1f}%) - Success: {successful_predictions}")
    
    print(f"✅ Batch prediction completed: {successful_predictions}/{len(X_data)} successful")
    print(f"   📊 Predictions shape: ({successful_predictions}, {len(CHAPMAN_TO_OUR_MAPPING)})")
    
    predictions_array = np.array(predictions)
    
    # Check for NaN values in predictions
    has_nan = np.isnan(predictions_array).any()
    if debug_mode:
        print(f"🔍 Debug - Predictions has NaN: {has_nan}")
    if has_nan:
        nan_count = np.isnan(predictions_array).sum()
        if debug_mode:
            print(f"   NaN count: {nan_count}/{predictions_array.size} ({nan_count/predictions_array.size*100:.2f}%)")
            print(f"   NaN positions: {np.where(np.isnan(predictions_array))}")
        
        # Fix: Replace NaN values with 0.0 to prevent AUC calculation errors
        if debug_mode:
            print(f"   🔧 Fixing NaN values by replacing with 0.0...")
        predictions_array = np.nan_to_num(predictions_array, nan=0.0)
        if debug_mode:
            print(f"   ✅ NaN values fixed")
    
    # Debug: Check prediction statistics
    print(f"   🔍 Prediction Statistics:")
    print(f"      Min probability: {predictions_array.min():.6f}")
    print(f"      Max probability: {predictions_array.max():.6f}")
    print(f"      Mean probability: {predictions_array.mean():.6f}")
    print(f"      Std probability: {predictions_array.std():.6f}")
    
    # Check if all predictions are the same (indicating a problem)
    if predictions_array.std() < 1e-6:
        print(f"   ⚠️ Warning: All predictions are nearly identical (std={predictions_array.std():.8f})")
    elif predictions_array.max() - predictions_array.min() < 0.01:
        print(f"   ⚠️ Warning: Prediction range is very small (max-min={predictions_array.max()-predictions_array.min():.6f})")
    
    # Filter predictions to only include actual classes (like in 71-class script)
    if actual_classes is not None:
        filtered_predictions = []
        chapman_classes = list(CHAPMAN_TO_OUR_MAPPING.values())
        
        for i, target_class in enumerate(actual_classes):
            # Find Chapman classes that map to this target class
            target_predictions = []
            for j, chapman_class in enumerate(chapman_classes):
                if chapman_class == target_class:
                    target_predictions.append(predictions_array[:, j])
            
            if target_predictions:
                # Take maximum probability among all Chapman classes that map to this target class
                target_pred = np.maximum.reduce(target_predictions)
                filtered_predictions.append(target_pred)
            else:
                # If no Chapman class maps to this target class, use zeros
                filtered_predictions.append(np.zeros(predictions_array.shape[0]))
        
        if filtered_predictions:
            predictions_array = np.column_stack(filtered_predictions)
            print(f"   🎯 Filtered predictions to {len(actual_classes)} classes: {actual_classes}")
            print(f"   📊 Filtered predictions shape: {predictions_array.shape}")
    
    return predictions_array

def evaluate_performance(y_true, y_pred, actual_classes, use_threshold=True, threshold=0.5):
    """Evaluate performance with flexible threshold options"""
    print("📊 Evaluating performance...")
    print(f"🔧 Using {'fixed threshold' if use_threshold else 'top-k'} approach")
    
    # Check if there are prediction data
    if y_pred.size == 0 or y_true.size == 0:
        print("⚠️  No predictions or labels available for evaluation")
        return {}
    
    # Convert to binary predictions
    if use_threshold:
        binary_preds = (y_pred >= threshold).astype(int)
        print(f"   📊 Fixed threshold approach: threshold = {threshold}")
    else:
        # Top-k approach: for each sample, set top-k predictions to 1
        binary_preds = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            # Get indices of top predictions for this sample
            top_indices = np.argsort(y_pred[i])[-1:]  # Top-1 for now
            binary_preds[i, top_indices] = 1
        print(f"   📊 Top-k approach: highest probability class per sample")
    
    # Calculate metrics for each class
    results = {}
    
    print(f"   📊 Evaluating {len(actual_classes)} actual classes: {actual_classes}")
    
    for i, class_name in enumerate(actual_classes):
        try:
            y_true_class = y_true[:, i]
            y_pred_proba = y_pred[:, i]
            y_pred_binary = binary_preds[:, i]
            
            # Skip if no positive samples in true labels
            if np.sum(y_true_class) == 0:
                print(f"   {class_name}: No positive samples (support = 0)")
                results[class_name] = {
                    'support': 0,
                    'auc': 0.5,  # Default AUC for no positive samples
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'specificity': 1.0,
                    'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
                }
                continue
            
            # Calculate AUC
            try:
                # Check for NaN values in predictions before AUC calculation
                if np.isnan(y_pred_proba).any():
                    if debug_mode:
                        print(f"   ⚠️ Warning: NaN values found in predictions for {class_name}, setting AUC to 0.5")
                    auc = 0.5
                else:
                    auc = roc_auc_score(y_true_class, y_pred_proba)
            except ValueError:
                auc = 0.5  # Default AUC if calculation fails
            
            # Calculate other metrics
            precision = precision_score(y_true_class, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_class, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_class, y_pred_binary, zero_division=0)
            
            # Calculate specificity
            tn = np.sum((y_true_class == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_class == 0) & (y_pred_binary == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate confusion matrix elements
            tp = np.sum((y_true_class == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true_class == 1) & (y_pred_binary == 0))
            
            results[class_name] = {
                'support': np.sum(y_true_class),
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
            
            print(f"   {class_name}: AUC={auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {class_name}: {e}")
            results[class_name] = {
                'support': 0, 'auc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'specificity': 0,
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
            }
    
    return results

def save_results(results, y_true, y_pred, output_dir, actual_classes):
    """Save evaluation results"""
    print(f"💾 Saving results...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_results = []
    for class_name in actual_classes:
        if class_name in results:
            result = results[class_name]
            detailed_results.append({
                'Class': class_name,
                'Support': result['support'],
                'AUC': result['auc'],
                'F1-Score': result['f1'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'Specificity': result['specificity'],
                'True Positives': result['tp'],
                'False Positives': result['fp'],
                'False Negatives': result['fn'],
                'True Negatives': result['tn']
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(output_dir / "detailed_results.csv", index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame(y_pred, columns=actual_classes)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save true labels
    true_labels_df = pd.DataFrame(y_true, columns=actual_classes)
    true_labels_df.to_csv(output_dir / "true_labels.csv", index=False)
    
    print(f"💾 Results saved to: {output_dir}")

def plot_confusion_matrix(y_true, y_pred, actual_classes, output_dir):
    """Create and save confusion matrix"""
    print(f"📊 Creating confusion matrix...")
    
    # Calculate confusion matrix for each class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, class_name in enumerate(actual_classes):
        if i < len(axes):
            y_true_class = y_true[:, i]
            y_pred_class = (y_pred[:, i] > 0.5).astype(int)
            
            cm = confusion_matrix(y_true_class, y_pred_class)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{class_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(len(actual_classes), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Confusion matrix saved")

def plot_roc_curves(y_true, y_pred, actual_classes, output_dir):
    """Create and save ROC curves"""
    print(f"📊 Creating ROC curves...")
    
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(actual_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        except Exception as e:
            print(f"   ⚠️  Could not plot ROC for {class_name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Chapman 13-Class Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ ROC curves saved")

def main():
    """Main function"""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    args = parse_arguments()
    exp_folder = Path(args.exp_folder)
    debug_mode = args.debug
    max_samples = args.max_samples
    
    print(f"🚀 Starting Chapman Dataset Batch Prediction for 13-Class Models")
    print(f"📁 Experiment folder: {exp_folder}")
    print(f"🔧 Debug mode: {debug_mode}")
    if max_samples:
        print(f"🔢 Max samples: {max_samples}")
    
    # Load model first to get model path for output directory naming
    pred_module, scaler, mlb, n_out, class_names = load_model_and_preprocessing(exp_folder)
    if pred_module is None:
        print("❌ Failed to load model, exiting...")
        return
    
    # Extract model name and create timestamped output directory with model info
    model_path = pred_module.get('model_path', 'unknown_model')
    model_name = Path(model_path).stem
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/prediction/chapman_test/batch_chapman_13class_results/run_{timestamp}_{model_name}"
    
    print(f"📁 Model: {model_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load Chapman dataset
    try:
        data_path = Path("data/Chapman")
        if not data_path.exists():
            raise FileNotFoundError(f"Chapman dataset not found at: {data_path}")
        
        X, y, actual_classes = load_chapman_dataset(data_path, debug_mode, max_samples)
    except Exception as e:
        print(f"❌ Failed to load Chapman dataset: {e}")
        return
    
    if len(X) == 0:
        print("❌ No valid samples found, exiting...")
        return
    
    # Run prediction
    start_time = time.time()
    print(f"\n🚀 Starting batch prediction for {len(X)} Chapman samples...")
    
    # Check for shutdown request
    if _shutdown_requested:
        print("🛑 Shutdown requested before prediction")
        return
    
    predictions = predict_batch_chapman_optimized(X, pred_module, scaler, mlb, n_out, class_names, actual_classes, debug_mode)
    
    prediction_time = time.time() - start_time
    print(f"⏱️  Prediction time: {prediction_time:.2f} seconds")
    print(f"⚡ Average time per sample: {prediction_time/len(X):.4f} seconds")
    
    # Evaluate performance
    results = evaluate_performance(y, predictions, actual_classes)
    
    # Save results
    save_results(results, y, predictions, output_dir, actual_classes)
    
    # Create visualizations
    plot_confusion_matrix(y, predictions, actual_classes, Path(output_dir))
    plot_roc_curves(y, predictions, actual_classes, Path(output_dir))
    
    # Final performance statistics
    total_time = time.time() - start_time
    avg_samples_per_second = len(X) / total_time if total_time > 0 else 0
    
    print(f"\n🎉 Chapman dataset batch prediction completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total samples evaluated: {len(X):,}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"⚡ Average processing speed: {avg_samples_per_second:.1f} samples/second")
    print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
