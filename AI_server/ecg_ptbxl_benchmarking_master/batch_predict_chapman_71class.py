#!/usr/bin/env python3
"""
Ultra-Fast Batch Chapman Shaoxing Prediction Script for 71-Class Models
Chapman Shaoxing evaluation script using WFDB format for 71-class models

Usage:
    python batch_predict_chapman_71class.py [exp_folder]
    
Examples:
    python batch_predict_chapman_71class.py                           # Use default model
    python batch_predict_chapman_71class.py output/exp0              # Use specified experiment folder
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
    'SR': 'NORM',            # Sinus Rhythm (426783006) - 8125 samples -> NORM (normal sinus rhythm)
    # 'ST': 'STACH',        # Sinus Tachycardia (427084000) - 7255 samples -> not mapped, ST and STACH are different concepts
    'STTC': 'STACH',         # ST-T Change (428750005) - 1158次
    'STDD': 'STACH',         # ST drop down (429622005) - 1668次
    'STE': 'STACH',          # ST extension (164930006) - 801次
    'STTU': 'STACH',         # ST tilt up -> STACH
    
    # 注意：现在将SR映射到NORM，因为正常窦性心律就是正常心电图
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

# 71-class model class list will be dynamically loaded from mlb.pkl at runtime
MODEL_71_CLASSES = None  # Will be dynamically loaded in load_model_and_preprocessing

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
        description='Ultra-Fast Batch Chapman Shaoxing Prediction Script for 71-Class Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_predict_chapman_71class.py                           # Use default model
    python batch_predict_chapman_71class.py output/exp0              # Use specified experiment folder
        """
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
        print(f"   ✅ Loaded scaler: {scaler_path}")
        
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        print(f"   ✅ Loaded MLB: {mlb_path}")
        
        # Dynamically set 71-class model class list
        global MODEL_71_CLASSES
        MODEL_71_CLASSES = list(mlb.classes_)
        print(f"   📋 Model has {len(MODEL_71_CLASSES)} classes")
        
        # Load model loading code (same as reference script)
        print("   🔄 Loading model loading code...")
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
        
        # Pre-load model and set to evaluation mode
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
            raise RuntimeError("Model loading returned None")
        
        # Store model path in globals for later use
        globals_128['model_path'] = str(model_path)
        
        print("✅ 71-class model and preprocessing components loaded")
        return globals_128, scaler, mlb, n_out
        
    except Exception as e:
        print(f"❌ Failed to load 71-class model: {e}")
        return None, None, None, None

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

def load_chapman_dataset(data_path, debug_mode=False, max_samples=None, memory_efficient=True):
    """Load Chapman dataset and extract target classes with memory optimization"""
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
    wfdb_paths_for_prediction = []  # Store WFDB paths for later prediction
    
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
                # Only map if there's a direct correspondence in our mapping
                if diag_acronym in CHAPMAN_TO_OUR_MAPPING:
                    mapped_class = CHAPMAN_TO_OUR_MAPPING[diag_acronym]
                    
                    # Find the index in our target classes
                    if mapped_class in TARGET_ECG_CLASSES:
                        target_idx = TARGET_ECG_CLASSES.index(mapped_class)
                        label_vector[target_idx] = 1
            
            # Only include samples that have at least one recognized target class
            # This follows the same approach as ICBEB script
            if np.sum(label_vector) > 0:
                X_data.append(ecg_data)
                y_data.append(label_vector)
                record_ids.append(record_path.stem)
                wfdb_paths_for_prediction.append(wfdb_path)  # Save the WFDB path
                successful_loads += 1
            
            # Debug output for first few samples
            if debug_mode and i < 3:
                print(f"   🔍 Sample {i}: {record_path.stem}")
                print(f"      SNOMED-CT codes: {diagnosis_codes}")
                print(f"      Diagnosis acronyms: {diagnosis_acronyms}")
                print(f"      Label vector: {label_vector}")
            
        except Exception as e:
            if debug_mode and i < 10:  # Only show errors for first few samples in debug mode
                print(f"   ⚠️ Failed to load {wfdb_path}: {e}")
            elif i < 50:  # Show first 50 errors even in non-debug mode
                print(f"   ⚠️ Failed to load {wfdb_path}: {e}")
            continue
    
    if not X_data:
        raise RuntimeError("No valid ECG records found in Chapman dataset")
    
    # Memory-efficient array creation
    if memory_efficient:
        # Pre-allocate arrays with exact size
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.int32)
        print(f"   💾 Memory-efficient array creation completed")
    else:
        # Standard approach
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.int32)
    
    print(f"   ✅ Successfully loaded {len(X_data)} samples")
    print(f"   📊 Data shapes: X={X_data.shape}, y={y_data.shape}")
    
    # Print class distribution for target classes
    print(f"   📈 Target class distribution:")
    for i, class_name in enumerate(TARGET_ECG_CLASSES):
        count = np.sum(y_data[:, i])
        if count > 0:
            print(f"      {class_name}: {count} samples")
    
    return X_data, y_data, TARGET_ECG_CLASSES, record_ids, wfdb_paths_for_prediction

def extract_target_classes(X_data, y_data, target_classes):
    """Extract target classes that are present in the dataset (already in target format)"""
    print("🎯 Extracting target classes...")
    
    # Find which target classes are present in the dataset
    target_indices = []
    target_class_names = []
    
    for i, target_class in enumerate(target_classes):
        if np.sum(y_data[:, i]) > 0:  # Only include classes with positive samples
            target_indices.append(i)
            target_class_names.append(target_class)
            print(f"   ✅ Found target class: {target_class} ({np.sum(y_data[:, i])} samples)")
        else:
            print(f"   ⚠️ Target class {target_class} present but no positive samples")
    
    if target_indices:
        y_target = y_data[:, target_indices]
        print(f"   ✅ Extracted {len(target_class_names)} target classes: {target_class_names}")
        print(f"   📊 Target label shape: {y_target.shape}")
        
        # Print target class distribution
        print(f"   📈 Target class distribution:")
        for i, class_name in enumerate(target_class_names):
            count = np.sum(y_target[:, i])
            print(f"      {class_name}: {count} samples")
    else:
        print("   ⚠️ No target classes found in dataset")
        y_target = y_data
        target_class_names = target_classes
    
    return X_data, y_target, target_class_names

def predict_batch_chapman(wfdb_paths, pred_module, scaler, mlb, n_out, actual_classes=None, debug_mode=False, batch_size=1000):
    """Batch prediction for Chapman dataset with memory-efficient processing"""
    print("🔮 Starting batch prediction...")
    
    predictions = []
    successful_predictions = 0
    
    # Pre-fetch model and device
    model = pred_module['model']
    device = pred_module['device']
    
    # Set warning flag
    warning_shown = False
    
    # Progress tracking (reduced frequency for better performance)
    total_samples = len(wfdb_paths)
    progress_interval = max(1, total_samples // 20)  # Update every 5%
    
    # Process in chunks for memory efficiency
    for chunk_start in range(0, total_samples, batch_size):
        chunk_end = min(chunk_start + batch_size, total_samples)
        chunk_paths = wfdb_paths[chunk_start:chunk_end]
        
        print(f"   📦 Processing chunk {chunk_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} ({len(chunk_paths)} samples)")
        
        # Process this chunk
        for i, wfdb_path in enumerate(chunk_paths):
            global_index = chunk_start + i
            # Check for shutdown request
            if _shutdown_requested:
                print(f"\n🛑 Interrupt request received, processed {global_index}/{total_samples} samples")
                break
            
            # Progress update (reduced frequency)
            if global_index % progress_interval == 0 or global_index == total_samples - 1:
                progress = (global_index + 1) / total_samples * 100
                print(f"   📊 Progress: {global_index+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
            
            try:
                # Extract ECG data from WFDB
                ecg_data = extract_ecg_from_wfdb(wfdb_path, debug_mode and global_index < 3)
            
                if ecg_data is None:
                    continue
                
                # Preprocessing - use same method as 71-class model training
                # Flatten ECG data to 1D, apply scaler, then reshape back to original shape
                x_shape = ecg_data.shape
                x_flat = ecg_data.flatten()[:, np.newaxis]  # Flatten to column vector
                x_scaled = scaler.transform(x_flat).reshape(x_shape)  # Apply scaler and reshape
                x = torch.from_numpy(x_scaled).unsqueeze(0)  # Add batch dimension
                
                x = x.to(device)
                
                # Inference (model already in eval mode, no need to call again)
                with torch.no_grad():
                    logits = model(x)
                    
                    # Check for NaN/Inf in logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        if global_index < 10:  # Show first few errors
                            print(f"   ⚠️ NaN/Inf detected in logits at sample {global_index}")
                        continue
                    
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                    
                    # Check for NaN/Inf in probabilities
                    if np.isnan(probs).any() or np.isinf(probs).any():
                        if global_index < 10:  # Show first few errors
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
                # Create mapped prediction probability array for target classes
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
                            if not warning_shown:
                                print(f"⚠️ Class {target_class} index {class_idx} in 71-class model exceeds prediction probability range")
                                warning_shown = True
                    else:
                        pred_probs[j] = 0.0
                        if not warning_shown:
                            print(f"⚠️ Class {target_class} does not exist in 71-class model")
                            warning_shown = True
                
                predictions.append(pred_probs)
                successful_predictions += 1
                
                # Debug: print mapped predictions (only in debug mode)
                if debug_mode and global_index < 3:
                    print(f"   🔍 Mapped Predictions Debug:")
                    for j, class_name in enumerate(TARGET_ECG_CLASSES):
                        print(f"      {class_name}: {pred_probs[j]:.6f}")
                
            except Exception as e:
                if debug_mode and global_index < 10:  # Only show errors for first few samples in debug mode
                    print(f"   ⚠️ Failed to predict {wfdb_path}: {e}")
                continue
        
        # Clear GPU cache after each chunk to prevent memory buildup
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
        raise RuntimeError("No successful predictions made")
    
    predictions = np.array(predictions, dtype=np.float32)
    print(f"   ✅ Successfully predicted {len(predictions)} samples")
    print(f"   📊 Predictions shape: {predictions.shape}")
    
    # Debug: Check prediction statistics
    print(f"   🔍 Prediction Statistics:")
    print(f"      Min probability: {predictions.min():.6f}")
    print(f"      Max probability: {predictions.max():.6f}")
    print(f"      Mean probability: {predictions.mean():.6f}")
    print(f"      Std probability: {predictions.std():.6f}")
    
    # Check if all predictions are the same (indicating a problem)
    if predictions.std() < 1e-6:
        print(f"   ⚠️ Warning: All predictions are nearly identical (std={predictions.std():.8f})")
    elif predictions.max() - predictions.min() < 0.01:
        print(f"   ⚠️ Warning: Prediction range is very small (max-min={predictions.max()-predictions.min():.6f})")
    
    # Filter predictions to only include actual classes if specified
    if actual_classes is not None:
        filtered_predictions = []
        for i, class_name in enumerate(TARGET_ECG_CLASSES):
            if class_name in actual_classes:
                filtered_predictions.append(predictions[:, i])
        if filtered_predictions:
            predictions = np.column_stack(filtered_predictions)
            print(f"   🎯 Filtered predictions to {len(actual_classes)} classes: {actual_classes}")
            print(f"   📊 Filtered predictions shape: {predictions.shape}")
    
    return predictions

def evaluate_performance(y_true, y_pred, class_names, actual_classes=None):
    """Evaluate model performance with detailed metrics"""
    print("📊 Evaluating performance...")
    
    if actual_classes is None:
        actual_classes = class_names
    
    n_classes = len(actual_classes)
    metrics = {
        'Class': [],
        'Support': [],
        'AUC': [],
        'F1-Score': [],
        'Precision': [],
        'Recall': [],
        'Specificity': [],
        'True Positives': [],
        'False Positives': [],
        'False Negatives': [],
        'True Negatives': []
    }
    
    for i, class_name in enumerate(actual_classes):
        if i >= y_true.shape[1]:
            print(f"   ⚠️ Warning: Class {class_name} index {i} exceeds true labels shape {y_true.shape}")
            continue
            
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_true_class, y_pred_class)
        except ValueError:
            auc = 0.5  # Default AUC for constant predictions
        
        # Convert probabilities to binary predictions for classification metrics
        y_pred_binary = (y_pred_class > 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_class, y_pred_binary, average='binary', zero_division=0
        )
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_binary, labels=[0, 1]).ravel()
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Store metrics
        metrics['Class'].append(class_name)
        metrics['Support'].append(int(np.sum(y_true_class)))
        metrics['AUC'].append(auc)
        metrics['F1-Score'].append(f1)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['Specificity'].append(specificity)
        metrics['True Positives'].append(int(tp))
        metrics['False Positives'].append(int(fp))
        metrics['False Negatives'].append(int(fn))
        metrics['True Negatives'].append(int(tn))
    
    return metrics

def save_results(predictions, true_labels, class_names, metrics, output_dir, record_ids):
    """Save results to files"""
    print("💾 Saving results...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions and true labels
    predictions_df = pd.DataFrame(predictions, columns=class_names, index=record_ids[:len(predictions)])
    true_labels_df = pd.DataFrame(true_labels, columns=class_names, index=record_ids[:len(true_labels)])
    
    predictions_df.to_csv(output_dir / 'predictions.csv', index=True)
    true_labels_df.to_csv(output_dir / 'true_labels.csv', index=True)
    
    # Save detailed results using csv module (same as clinical script)
    detailed_results_path = output_dir / 'detailed_results.csv'
    with open(detailed_results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        header = ['Class', 'Support', 'AUC', 'F1-Score', 'Precision', 'Recall', 'Specificity', 
                 'True Positives', 'False Positives', 'False Negatives', 'True Negatives']
        writer.writerow(header)
        
        # Write data
        for i in range(len(metrics['Class'])):
            row = [
                metrics['Class'][i],
                metrics['Support'][i],
                f"{metrics['AUC'][i]:.4f}",
                f"{metrics['F1-Score'][i]:.4f}",
                f"{metrics['Precision'][i]:.4f}",
                f"{metrics['Recall'][i]:.4f}",
                f"{metrics['Specificity'][i]:.4f}",
                metrics['True Positives'][i],
                metrics['False Positives'][i],
                metrics['False Negatives'][i],
                metrics['True Negatives'][i]
            ]
            writer.writerow(row)
    
    # Save summary results using csv module
    summary_results_path = output_dir / 'summary_results.csv'
    with open(summary_results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        header = ['Class', 'Support', 'AUC', 'F1-Score', 'Precision', 'Recall', 'Specificity', 
                 'True Positives', 'False Positives', 'False Negatives', 'True Negatives']
        writer.writerow(header)
        
        # Write data
        for i in range(len(metrics['Class'])):
            row = [
                metrics['Class'][i],
                metrics['Support'][i],
                f"{metrics['AUC'][i]:.4f}",
                f"{metrics['F1-Score'][i]:.4f}",
                f"{metrics['Precision'][i]:.4f}",
                f"{metrics['Recall'][i]:.4f}",
                f"{metrics['Specificity'][i]:.4f}",
                metrics['True Positives'][i],
                metrics['False Positives'][i],
                metrics['False Negatives'][i],
                metrics['True Negatives'][i]
            ]
            writer.writerow(row)
    
    print(f"   ✅ Results saved to: {output_dir}")
    print(f"   📁 Files created:")
    print(f"      - predictions.csv")
    print(f"      - true_labels.csv")
    print(f"      - detailed_results.csv")
    print(f"      - summary_results.csv")

def create_visualizations(metrics, output_dir):
    """Create confusion matrix and ROC curves"""
    print("📊 Creating visualizations...")
    
    output_dir = Path(output_dir)
    
    # Create confusion matrix for each class
    class_names = metrics['Class']
    n_classes = len(class_names)
    
    if n_classes > 0:
        # Create subplots for confusion matrices
        n_cols = min(4, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle different subplot configurations
        if n_classes == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1 and n_rows > 1:
            axes = axes.reshape(-1, 1)
        
        for i, class_name in enumerate(class_names):
            row = i // n_cols
            col = i % n_cols
            
            # Get the correct axes object
            if n_classes == 1:
                ax = axes[0]
            elif n_rows == 1:
                ax = axes[0, col] if n_cols > 1 else axes[col]
            elif n_cols == 1:
                ax = axes[row, 0] if n_rows > 1 else axes[row]
            else:
                ax = axes[row, col]
            
            # Get confusion matrix values
            tp = metrics['True Positives'][i]
            fp = metrics['False Positives'][i]
            fn = metrics['False Negatives'][i]
            tn = metrics['True Negatives'][i]
            
            # Create confusion matrix
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(f'Confusion Matrix - {class_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_classes, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Confusion matrices saved: {output_dir / 'confusion_matrices.png'}")
    
    print("   ✅ Visualizations created")

def main():
    """Main function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Chapman Shaoxing Dataset Evaluation for 71-Class Models")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    exp_folder = args.exp_folder
    
    print(f"📁 Experiment folder: {exp_folder}")
    print(f"📁 Chapman dataset: data/Chapman")
    
    # Load model and preprocessing
    pred_module, scaler, mlb, n_out = load_model_and_preprocessing(exp_folder)
    if pred_module is None:
        print("❌ Failed to load model, exiting...")
        return
    
    # Load Chapman dataset
    data_path = Path("data/Chapman")
    if not data_path.exists():
        print(f"❌ Chapman dataset not found: {data_path}")
        return
    
    try:
        # Load dataset with memory optimization (full dataset with fixes)
        X_data, y_data, chapman_classes, record_ids, wfdb_paths = load_chapman_dataset(
            data_path, debug_mode=False, max_samples=None, memory_efficient=True
        )
        # Load dataset (enable debug mode for testing)
        # X_data, y_data, chapman_classes, record_ids, wfdb_paths = load_chapman_dataset(data_path, debug_mode=True, max_samples=100)
        
        # Extract target classes
        X_target, y_target, target_class_names = extract_target_classes(X_data, y_data, chapman_classes)
        
        # Use the saved WFDB paths (no need to rebuild them)
        if len(wfdb_paths) != len(X_target):
            print(f"⚠️ Warning: Mismatch between loaded samples ({len(X_target)}) and WFDB paths ({len(wfdb_paths)})")
            # Use the minimum length
            min_len = min(len(X_target), len(wfdb_paths))
            X_target = X_target[:min_len]
            y_target = y_target[:min_len]
            wfdb_paths = wfdb_paths[:min_len]
        
        # Predict with batch processing for memory efficiency (smaller batch size)
        predictions = predict_batch_chapman(
            wfdb_paths, pred_module, scaler, mlb, n_out, 
            actual_classes=target_class_names, debug_mode=False, batch_size=100
            # actual_classes=target_class_names, debug_mode=True
        )
        
        # Check for sample count mismatch and fix if needed
        if len(predictions) != len(y_target):
            print(f"⚠️ Warning: Sample count mismatch - predictions: {len(predictions)}, targets: {len(y_target)}")
            # Use the minimum count to ensure consistency
            min_count = min(len(predictions), len(y_target))
            predictions = predictions[:min_count]
            y_target = y_target[:min_count]
            print(f"   📊 Adjusted to use {min_count} samples for evaluation")
        
        # Evaluate performance
        metrics = evaluate_performance(y_target, predictions, target_class_names, target_class_names)
        
        # Print results
        print("\n" + "="*70)
        print("📊 71-Class Model Chapman Dataset Evaluation Results:")
        print("="*70)
        
        # Create results DataFrame for display
        results_df = pd.DataFrame(metrics)
        results_df.index.name = 'Class'
        
        # Print formatted results
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Create output directory with timestamp and model name (same structure as other scripts)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "fastai_xresnet1d101"
        output_dir = Path(f"output/prediction/Chapman_test/batch_chapman_71class_results/run_{timestamp}_{model_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        save_results(predictions, y_target, target_class_names, metrics, output_dir, record_ids)
        
        # Create visualizations
        create_visualizations(metrics, output_dir)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
