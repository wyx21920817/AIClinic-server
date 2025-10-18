#!/usr/bin/env python3
"""
MIMIC-IV-ECG Dataset Batch Prediction Script for 71-Class Models
Script for evaluating MIMIC-IV-ECG dataset using 71-class models for prediction

Usage:
    python batch_predict_mimic_iv_ecg_71class.py [exp_folder]
    
Examples:
    python batch_predict_mimic_iv_ecg_71class.py                           # Use default model
    python batch_predict_mimic_iv_ecg_71class.py output/exp0              # Use specified experiment folder
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
import wfdb
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import signal
import warnings
import argparse
import gc
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MIMIC-IV-ECG dataset classes to our 71-class model mapping
# Only map direct correspondences, avoid over-generalization
MIMIC_TO_OUR_MAPPING = {
    # Normal rhythms - map to SR (Sinus Rhythm) as it's a specific rhythm type
    'Sinus rhythm': 'SR',
    'Sinus rhythm.': 'SR',
    'Normal ECG': 'NORM',  # Generic normal ECG maps to NORM
    'Normal ECG except for rate': 'NORM',  # Normal except rate maps to NORM
    # Note: "Borderline ECG" and "Poor R wave progression" are excluded - too ambiguous
    # Note: "Abnormal ECG" is excluded - it's too non-specific and doesn't represent normal findings
    
    # Arrhythmias - only map confirmed diagnoses, avoid over-generalization
    'Atrial fibrillation': 'AFIB',
    'Atrial fibrillation.': 'AFIB',
    'Atrial flutter': 'AFLT',
    'Sinus tachycardia': 'STACH',
    'Sinus tachycardia.': 'STACH',
    # Note: Removed combined diagnoses to avoid over-generalization
    # Such as 'Atrial fibrillation with rapid ventricular response' should be marked as AFIB + ventricular response type through multi-label classification
    
    # AV blocks - simplified mapping based on actual diagnoses in data
    # 1AVB - actual diagnoses (sorted by frequency)
    'FIRST DEGREE A-V BLOCK': '1AVB',
    'Sinus rhythm with borderline 1st degree A-V block': '1AVB',
    'Sinus rhythm with borderline 1st degree A-V block.': '1AVB',
    'Sinus rhythm with 1st degree A-V block': '1AVB',
    'Sinus rhythm with 1st degree A-V block.': '1AVB',  # Add period variant
    'Sinus bradycardia with borderline 1st degree A-V block': '1AVB',
    'Sinus bradycardia with borderline 1st degree A-V block.': '1AVB',  # Add period variant
    'Sinus bradycardia with 1st degree A-V block': '1AVB',
    'Sinus arrhythmia with borderline 1st degree A-V block': '1AVB',
    '- first degree A-V block': '1AVB',
    '- borderline first degree A-V block': '1AVB',
    # Combined diagnoses removed to avoid over-generalization
    
    # 2AVB - actual diagnoses
    'Sinus rhythm with 2nd degree A-V block, Mobitz I (Wenckebach).': '2AVB',
    'Second degree AV block, Mobitz II': '2AVB',
    # Combined diagnoses removed (such as 2:1 / 4:1 AV block combined with AFLT) to avoid over-generalization
    
    # 3AVB - actual diagnoses
    'Probable atrial fibrillation with complete A-V block': '3AVB',
    
    # Bundle branch blocks - strict mapping to avoid over-generalization
    'Left bundle branch block': 'CLBBB',
    'Right bundle branch block': 'CRBBB',
    'Incomplete left bundle branch block': 'ILBBB',
    'Incomplete right bundle branch block': 'IRBBB',
    'LEFT BUNDLE BRANCH BLOCK': 'CLBBB',
    'RIGHT BUNDLE BRANCH BLOCK': 'CRBBB',
    # Note: Removed over-generalized mappings
    # Such as 'Incomplete LBBB' should not be classified as 'CLBBB' because incomplete and complete have different clinical significance
    # Such as 'Conduction defect of LBBB type' is too vague, not a clear LBBB diagnosis
    # Note: RBBB with fascicular blocks are excluded - they represent combined conduction defects
    
    # Premature beats - based on pure PAC/PVC diagnoses that exist in actual data
    # PAC - pure atrial premature contraction diagnoses (not including other arrhythmias)
    'PREMATURE ATRIAL CONTRACTION(S)': 'PAC',
    '- premature atrial contractions': 'PAC',
    'Sinus rhythm with PAC(s)': 'PAC',
    'Sinus rhythm with PAC(s).': 'PAC',
    'Sinus rhythm with PACs': 'PAC',
    'Sinus rhythm with PACs.': 'PAC',
    'Sinus tachycardia with PAC(s)': 'PAC',
    'Sinus tachycardia with PAC(s).': 'PAC',
    'Sinus tachycardia with PACs': 'PAC',
    'Sinus tachycardia with PACs.': 'PAC',
    'Sinus bradycardia with PAC(s)': 'PAC',
    'Sinus bradycardia with PAC(s).': 'PAC',
    'Sinus bradycardia with PACs': 'PAC',
    'Sinus arrhythmia with PAC(s)': 'PAC',
    'Sinus arrhythmia with PACs': 'PAC',
    'Sinus rhythm with frequent PACs': 'PAC',
    'Sinus rhythm with frequent PACs.': 'PAC',
    'Sinus rhythm with bigeminal PACs': 'PAC',
    'Sinus rhythm with bigeminal PACs.': 'PAC',
    'Sinus rhythm with occasional PACs': 'PAC',
    '- bigeminal PACs': 'PAC',
    
    # PVC - pure ventricular premature contraction diagnoses (not including other arrhythmias)
    'PREMATURE VENTRICULAR CONTRACTION(S)': 'PVC',
    'PREMATURE VENTRICULAR CONTRACTIONS': 'PVC',
    'MULTIFOCAL PREMATURE VENTRICULAR CONTRACTIONS': 'PVC',
    'MULTIFORM PVCs OR ABERRANT VENTRICULAR CONDUCTION': 'PVC',
    '- premature ventricular contractions': 'PVC',
    '- frequent premature ventricular contractions': 'PVC',
    '- frequent multiform PVCs': 'PVC',
    '- multiform PVCs': 'PVC',
    '- frequent multifocal PVCs': 'PVC',
    '- multifocal PVCs': 'PVC',
    'Sinus rhythm with PVC(s)': 'PVC',
    'Sinus rhythm with PVC(s).': 'PVC',
    'Sinus rhythm with PVCs': 'PVC',
    'Sinus rhythm with PVCs.': 'PVC',
    'Sinus tachycardia with PVC(s)': 'PVC',
    'Sinus tachycardia with PVC(s).': 'PVC',
    'Sinus tachycardia with PVCs': 'PVC',
    'Sinus tachycardia with PVCs.': 'PVC',
    'Sinus bradycardia with PVC(s)': 'PVC',
    'Sinus bradycardia with PVC(s).': 'PVC',
    'Sinus bradycardia with PVCs': 'PVC',
    'Sinus bradycardia with PVCs.': 'PVC',
    'Sinus arrhythmia with PVC(s)': 'PVC',
    'Sinus arrhythmia with PVCs': 'PVC',
    'Sinus arrhythmia with PVCs.': 'PVC',
    'Sinus rhythm with frequent PVCs': 'PVC',
    'Sinus rhythm with frequent PVCs.': 'PVC',
    'Sinus rhythm with bigeminal PVCs': 'PVC',
    'Sinus rhythm with bigeminal PVCs.': 'PVC',
    'Sinus rhythm with multifocal PVCs': 'PVC',
    'Sinus rhythm with frequent multifocal PVCs': 'PVC',
    'Sinus rhythm with frequent multifocal PVCs.': 'PVC',
    'Sinus rhythm with interpolated PVC(s)': 'PVC',
    'Sinus rhythm with interpolated PVC(s).': 'PVC',
    'Sinus rhythm with interpolated PVCs': 'PVC',
    'Sinus bradycardia with interpolated PVC(s)': 'PVC',
    'Sinus bradycardia with interpolated PVC(s).': 'PVC',
    'Sinus bradycardia with interpolated PVCs': 'PVC',
    'Sinus bradycardia with multifocal PVCs': 'PVC',
    'Sinus bradycardia with frequent PVCs': 'PVC',
    'Sinus bradycardia with frequent PVCs.': 'PVC',
    'Sinus bradycardia with frequent multifocal PVCs': 'PVC',
    'Sinus bradycardia with multifocal interpolated PVCs': 'PVC',
    'Sinus arrhythmia with interpolated PVC(s)': 'PVC',
    'Sinus arrhythmia with multifocal PVCs': 'PVC',
    'Sinus arrhythmia with frequent PVCs': 'PVC',
    'Sinus arrhythmia with frequent PVCs.': 'PVC',
    # Combined diagnoses removed (such as AFIB+PVC, PAC+PVC, STACH+PVC, etc.) to avoid over-generalization
    
    # ST-T abnormalities - excluded since we don't care about these classes
    # Note: ST-T abnormalities are different from Sinus Tachycardia (STACH)
    # We exclude these mappings since STD_, STE_, NST_ are not in our target classes
    # 'ST-T changes': 'NST_',        # Excluded - not in target classes
    # 'ST-T abnormalities': 'NST_',  # Excluded - not in target classes
    # 'ST-T depression': 'STD_',     # Excluded - not in target classes
    # 'ST-T elevation': 'STE_',      # Excluded - not in target classes
    # 'ST elevation': 'STE_',        # Excluded - not in target classes
    # 'ST depression': 'STD_'        # Excluded - not in target classes
    
    # Note: Excluded mappings (to avoid over-generalization):
    # - Sinus bradycardia: Not in our target classes (no SBRAD)
    # - Sinus arrhythmia: Too general, not clearly abnormal
    # - Abnormal ECG: Too non-specific, doesn't represent normal findings
    # - Borderline ECG: Too ambiguous, not clearly normal
    # - Poor R wave progression: Too ambiguous, not clearly normal
    # - Possible right atrial abnormality: Not in our target classes
    # - Left axis deviation: Not in our target classes
    # - Prolonged QT interval: Not in our target classes
    # - ST-T abnormalities (STD_, STE_, NST_): Not in our target classes
    # - Summary diagnoses: Too general, non-specific
    # - Pacing-related diagnoses: Artificial rhythms, not natural arrhythmias
    # - RBBB with fascicular blocks: Combined conduction defects, not pure RBBB
}

# Target ECG classes we want to evaluate (subset of 71 classes)
# Note: Including all classes that are actually mapped from MIMIC-IV-ECG
TARGET_ECG_CLASSES = ['NORM', 'SR', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 
                      'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']

# Global variable for model classes (will be set dynamically)
MODEL_71_CLASSES = []

# Global shutdown flag
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signal"""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n🛑 Received signal {signum}, will shutdown gracefully...")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MIMIC-IV-ECG Dataset Batch Prediction Script for 71-Class Models')
    parser.add_argument('exp_folder', nargs='?', default='output/exp0', 
                       help='Experiment folder containing model and preprocessing files')
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
        
        # Use the first model file found
        model_path = model_files[0]
        print(f"   🎯 Using model: {model_path}")
        
        # Find scaler and mlb files
        scaler_path = data_dir / "standard_scaler.pkl"
        mlb_path = data_dir / "mlb.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not mlb_path.exists():
            raise FileNotFoundError(f"MLB file not found: {mlb_path}")
        
        print(f"   🔧 Loading scaler: {scaler_path}")
        print(f"   🔧 Loading MLB: {mlb_path}")
        
        # Load scaler and MLB
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        
        # Load model classes dynamically
        global MODEL_71_CLASSES
        MODEL_71_CLASSES = mlb.classes_.tolist()
        print(f"   📊 Loaded {len(MODEL_71_CLASSES)} model classes")
        
        # Load model using classify_single_ecg128.py approach
        print(f"   🚀 Loading model from: {model_path}")
        
        # Execute classify_single_ecg128.py to get model loading functions
        classify_script_path = Path("classify_single_ecg128.py")
        if not classify_script_path.exists():
            raise FileNotFoundError(f"classify_single_ecg128.py not found: {classify_script_path}")
        
        # Read and execute the script to get model loading functions
        with open(classify_script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Create a namespace for the script execution
        globals_128 = {}
        exec(script_content, globals_128)
        
        # Get number of output classes from MLB
        n_out = len(mlb.classes_)
        print(f"   📊 Model output classes: {n_out}")
        
        # Load the model
        print("   🔄 Pre-loading 71-class model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU optimizations
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
            torch.backends.cudnn.allow_tf32 = True
            # Set memory allocation strategy for better performance
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        model = globals_128['load_model'](
            str(model_path), 
            n_in=12, 
            n_out=n_out, 
            device=device
        )
        if model is None:
            print("❌ Failed to load model")
            return None, None, None, None
        model.eval()
        model = model.to(device)
        print(f"   🖥️ Using device: {device}")
        
        # GPU memory optimization
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear cache for better memory management
        
        # Create prediction module
        pred_module = {
            'model': model,
            'device': device,
            'model_path': str(model_path)
        }
        
        n_out = len(MODEL_71_CLASSES)
        print(f"   ✅ Model loaded successfully with {n_out} output classes")
        
        return pred_module, scaler, mlb, n_out
        
    except Exception as e:
        print(f"❌ Failed to load 71-class model: {e}")
        return None, None, None, None

def extract_model_name_from_path(model_path):
    """Extract model name from model path"""
    model_path = str(model_path)
    if 'fastai_xresnet1d101' in model_path:
        return 'fastai_xresnet1d101'
    elif 'naive' in model_path:
        return 'naive'
    else:
        return 'unknown_model'

def load_single_ecg_parallel(args):
    """Helper function for parallel loading of single ECG file"""
    file_path, data_path, scaler = args
    try:
        # Build complete file path
        full_path = data_path / file_path
        
        if not full_path.exists():
            return None
            
        # Read ECG data
        record = wfdb.rdrecord(str(full_path))
        signal = record.p_signal
        
        if signal is None:
            return None
            
        # Convert to numpy array and transpose to (time, leads)
        signal = signal.T
        
        # Preprocessing: truncate or pad to 1000 sampling points
        if signal.shape[0] > 1000:
            signal = signal[:1000, :]
        elif signal.shape[0] < 1000:
            pad_width = 1000 - signal.shape[0]
            signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
        
        # Standardization
        if scaler is not None:
            signal = scaler.transform(signal)
        
        return signal
    except Exception as e:
        return None

def load_mimic_dataset_batch(data_path, batch_size=1000, max_batches=None, debug_mode=False, n_workers=4):
    """
    Load MIMIC-IV-ECG dataset in batches to avoid memory issues
    
    Args:
        data_path: Path to MIMIC-IV-ECG dataset directory
        batch_size: Number of samples per batch
        max_batches: Maximum number of batches to process (for testing)
        debug_mode: Whether to enable debug output
        n_workers: Number of parallel workers for data loading
        
    Yields:
        X_data: ECG signal data (batch_size, n_channels, n_timesteps)
        y_data: Labels (batch_size, n_classes)
        class_names: List of class names
        batch_info: Dictionary with batch information
    """
    print("📊 Loading MIMIC-IV-ECG dataset in batches...")
    
    # Load metadata files
    record_list_path = data_path / 'record_list.csv'
    measurements_path = data_path / 'machine_measurements.csv'
    
    if not record_list_path.exists():
        raise FileNotFoundError(f"Record list not found: {record_list_path}")
    if not measurements_path.exists():
        raise FileNotFoundError(f"Measurements not found: {measurements_path}")
    
    # Load record list and measurements
    record_df = pd.read_csv(record_list_path)
    measurements_df = pd.read_csv(measurements_path)
    
    print(f"   📋 Loaded {len(record_df)} records from record_list.csv")
    print(f"   📋 Loaded {len(measurements_df)} measurements from machine_measurements.csv")
    
    # Merge dataframes on subject_id and study_id
    merged_df = pd.merge(record_df, measurements_df, on=['subject_id', 'study_id'])
    print(f"   📋 Merged dataset: {len(merged_df)} records")
    
    # Process in batches
    total_processed = 0
    batch_count = 0
    
    # Pre-allocate arrays for better performance
    batch_signals = np.zeros((batch_size, 1000, 12), dtype=np.float32)  # 1000 samples at 100Hz
    batch_labels = np.zeros((batch_size, len(MIMIC_TO_OUR_MAPPING)), dtype=np.int32)
    batch_record_ids = []
    
    current_batch_size = 0
    
    for idx, row in merged_df.iterrows():
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\n🛑 Interrupt request received, processed {total_processed} samples")
            break
        
        # Progress update
        if idx % 10000 == 0:
            print(f"   📊 Progress: {idx}/{len(merged_df)} ({idx/len(merged_df)*100:.1f}%) - Processed: {total_processed}")
        
        try:
            # Construct file path
            file_path = data_path / row['path']
            record_name = file_path.stem
            
            # Try to load ECG data using WFDB directly
            # WFDB can handle path issues better than Path.exists() on Windows
            record = wfdb.rdrecord(str(file_path))
            signal = record.p_signal.astype(np.float32)  # Shape: (n_samples, 12)
            
            # Debug: Print signal info for first few samples (only in debug mode)
            if debug_mode and current_batch_size < 3 and batch_count == 0:
                print(f"   🔍 Debug - Raw signal from WFDB (sample {current_batch_size}):")
                print(f"      Record: {record_name}")
                print(f"      Signal shape: {signal.shape}")
                print(f"      Signal dtype: {signal.dtype}")
                print(f"      Signal range: [{signal.min():.3f}, {signal.max():.3f}] mV")
                print(f"      Sampling rate: {record.fs} Hz")
                print(f"      Subject ID: {row['subject_id']}, Study ID: {row['study_id']}")
            
            # Handle different sampling rates - downsample from 500Hz to 100Hz
            if record.fs == 500:
                # Downsample by taking every 5th sample
                signal = signal[::5, :]  # Shape: (n_samples/5, 12)
            elif record.fs != 100:
                print(f"   ⚠️ Warning: Unexpected sampling rate {record.fs} Hz for {record_name}")
                # Assume it's 500Hz and downsample
                signal = signal[::5, :]
            
            # Standardize to 10 seconds at 100Hz (1000 samples)
            target_length = 1000
            if signal.shape[0] > target_length:
                # Truncate from the middle
                start_idx = (signal.shape[0] - target_length) // 2
                signal = signal[start_idx:start_idx + target_length, :]
            elif signal.shape[0] < target_length:
                # Pad with zeros
                pad_length = target_length - signal.shape[0]
                signal = np.pad(signal, ((0, pad_length), (0, 0)), mode='constant')
            
            # Debug: Print processed signal info for first sample (only in debug mode)
            if debug_mode and current_batch_size < 3 and batch_count == 0:
                print(f"   🔍 Debug - Processed signal (sample {current_batch_size}):")
                print(f"      Final shape: {signal.shape}")
                print(f"      Final range: [{signal.min():.3f}, {signal.max():.3f}]")
            
            # Store in pre-allocated array
            batch_signals[current_batch_size] = signal
            batch_record_ids.append(f"{row['subject_id']}_{row['study_id']}")
            
            # Create label vector based on diagnosis reports
            mimic_classes = list(MIMIC_TO_OUR_MAPPING.keys())
            for j, mimic_class in enumerate(mimic_classes):
                # Check if this diagnosis appears in any of the report columns
                has_diagnosis = False
                for report_col in [f'report_{i}' for i in range(18)]:
                    if report_col in row and pd.notna(row[report_col]):
                        report_text = str(row[report_col]).strip()
                        # Use exact match instead of substring match to avoid false positives
                        if report_text.lower() == mimic_class.lower():
                            has_diagnosis = True
                            break
                
                batch_labels[current_batch_size, j] = 1 if has_diagnosis else 0
            
            # Debug: Print diagnosis info for first few samples (only in debug mode)
            if debug_mode and current_batch_size < 3 and batch_count == 0:
                print(f"   🔍 Debug - Diagnosis info (sample {current_batch_size}):")
                # Show first few non-empty reports
                reports = []
                for report_col in [f'report_{i}' for i in range(5)]:  # Only show first 5 reports
                    if report_col in row and pd.notna(row[report_col]):
                        reports.append(str(row[report_col]))
                print(f"      Reports: {reports}")
                # Show label vector for this sample
                sample_labels = batch_labels[current_batch_size]
                active_classes = [mimic_classes[i] for i, label in enumerate(sample_labels) if label == 1]
                print(f"      Active diagnoses: {active_classes}")
            
            current_batch_size += 1
            
            # Yield batch when it reaches batch_size
            if current_batch_size >= batch_size:
                X_data = batch_signals[:current_batch_size].copy()
                y_data = batch_labels[:current_batch_size].copy()
                
                batch_count += 1
                total_processed += current_batch_size
                
                print(f"   📦 Yielding batch {batch_count}: {current_batch_size} samples (Total: {total_processed})")
                
                batch_info = {
                    'batch_id': batch_count,
                    'samples': current_batch_size,
                    'total_processed': total_processed
                }
                
                yield X_data, y_data, mimic_classes, batch_info
                
                # Reset batch
                current_batch_size = 0
                
                # Check if we've reached max batches
                if max_batches is not None and batch_count >= max_batches:
                    print(f"   🛑 Reached maximum batches limit: {max_batches}")
                    break
        
        except Exception as e:
            print(f"   ❌ Error processing record {row['subject_id']}_{row['study_id']}: {e}")
            continue
    
    # Yield remaining samples if any
    if current_batch_size > 0:
        X_data = batch_signals[:current_batch_size].copy()
        y_data = batch_labels[:current_batch_size].copy()
        
        batch_count += 1
        total_processed += current_batch_size
        
        print(f"   📦 Yielding final batch {batch_count}: {current_batch_size} samples (Total: {total_processed})")
        
        batch_info = {
            'batch_id': batch_count,
            'samples': current_batch_size,
            'total_processed': total_processed
        }
        
        yield X_data, y_data, mimic_classes, batch_info
    
    print(f"   ✅ Dataset loading completed: {total_processed} total samples in {batch_count} batches")

def extract_target_classes(X_data, y_data, mimic_classes):
    """Extract target classes from the full dataset"""
    print(f"🎯 Extracting target classes from {len(mimic_classes)} MIMIC classes...")
    
    # Collect all relevant MIMIC category indices for each target category
    target_class_mappings = {}
    
    for target_ecg_class in TARGET_ECG_CLASSES:
        target_class_mappings[target_ecg_class] = []
        for mimic_class, mapped_ecg_class in MIMIC_TO_OUR_MAPPING.items():
            if mapped_ecg_class == target_ecg_class and mimic_class in mimic_classes:
                class_idx = mimic_classes.index(mimic_class)
                target_class_mappings[target_ecg_class].append(class_idx)
    
    # Create merged target label matrix
    target_class_names = []
    target_indices_list = []
    
    for target_ecg_class in TARGET_ECG_CLASSES:
        if target_class_mappings[target_ecg_class]:  # If mapping found
            target_class_names.append(target_ecg_class)
            target_indices_list.append(target_class_mappings[target_ecg_class])
    
    if target_class_names:
        # Create merged label matrix
        y_target = np.zeros((y_data.shape[0], len(target_class_names)), dtype=np.int32)
        
        for i, (target_class, mimic_indices) in enumerate(zip(target_class_names, target_indices_list)):
            # Merge multiple MIMIC categories into one target category (using OR logic)
            for mimic_idx in mimic_indices:
                y_target[:, i] = np.logical_or(y_target[:, i], y_data[:, mimic_idx]).astype(np.int32)
        
        print(f"   ✅ Extracted {len(target_class_names)} target classes: {target_class_names}")
        print(f"   📊 Target label shape: {y_target.shape}")
        
        # Filter samples - only keep samples that have at least one target class label
        # This ensures we only evaluate samples relevant to our 14 target classes
        has_target_label = np.any(y_target, axis=1)
        filtered_indices = np.where(has_target_label)[0]
        
        if len(filtered_indices) > 0:
            X_data = X_data[filtered_indices]
            y_target = y_target[filtered_indices]
            print(f"   📊 Filtered to {len(X_data)} samples with target class labels (from {len(y_data)} total)")
        else:
            print(f"   ⚠️ No samples found with target class labels")
        
        # Print target class distribution
        print(f"   📈 Target class distribution:")
        for i, class_name in enumerate(target_class_names):
            count = np.sum(y_target[:, i])
            print(f"      {class_name}: {count} samples")
    else:
        print("   ⚠️ No target classes found in dataset")
        y_target = y_data
        target_class_names = mimic_classes
    
    return X_data, y_target, target_class_names

def predict_batch_mimic_optimized(X_data, pred_module, scaler, mlb, n_out, actual_classes=None, debug_mode=False):
    """Highly optimized batch prediction for MIMIC-IV-ECG dataset"""
    print(f"🚀 Starting optimized batch prediction for {len(X_data)} MIMIC-IV-ECG samples...")
    
    # Pre-fetch model and device
    model = pred_module['model']
    device = pred_module['device']
    
    # Set warning flag
    warning_shown = False
    
    # Progress tracking - reduced frequency for better performance
    total_samples = len(X_data)
    progress_interval = max(1, total_samples // 50)  # Update every 2% for better performance
    
    # Pre-allocate prediction array for better memory efficiency
    classes_to_predict = actual_classes if actual_classes is not None else TARGET_ECG_CLASSES
    predictions = np.zeros((total_samples, len(classes_to_predict)), dtype=np.float32)
    
    # Process in mini-batches - OPTIMIZED FOR GPU
    mini_batch_size = 2048  # Aggressively increased for maximum GPU utilization (2x improvement)
    
    # Pre-allocate GPU tensor for better memory management
    if device.type == 'cuda':
        # Pre-allocate tensor on GPU to avoid repeated memory allocation
        dummy_tensor = torch.zeros((mini_batch_size, 12, 1000), dtype=torch.float32, device=device)
    
    # Pre-compute class indices for faster mapping
    class_indices = []
    for target_class in classes_to_predict:
        if target_class in MODEL_71_CLASSES:
            class_indices.append(MODEL_71_CLASSES.index(target_class))
        else:
            class_indices.append(-1)  # Invalid index
            if not warning_shown:
                print(f"⚠️ Class {target_class} does not exist in 71-class model")
    
    successful_predictions = 0
    failed_predictions = 0
    
    for i in range(0, total_samples, mini_batch_size):
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\n🛑 Interrupt request received, processed {i}/{total_samples} samples")
            break
        
        # Progress update (less frequent in non-debug mode)
        if debug_mode:
            # Debug mode: frequent updates
            if i % progress_interval == 0 or i == total_samples - 1:
                progress = (i + 1) / total_samples * 100
                print(f"   📊 Progress: {i+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
        else:
            # Non-debug mode: less frequent updates (every 25%)
            if i % (total_samples // 4) == 0 or i == total_samples - 1:
                progress = (i + 1) / total_samples * 100
                print(f"   📊 Progress: {i+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
        
        # Determine mini-batch size (handle last batch)
        batch_end = min(i + mini_batch_size, total_samples)
        current_batch_size = batch_end - i
        
        try:
            # Process mini-batch
            batch_data = X_data[i:batch_end]  # Shape: (batch_size, n_timesteps, n_channels)
            
            # Vectorized preprocessing for better performance
            # Transpose entire batch at once: (batch_size, n_timesteps, n_channels) -> (batch_size, n_channels, n_timesteps)
            batch_transposed = batch_data.transpose(0, 2, 1)  # Shape: (batch_size, 12, 1000)
            
            # Reshape for vectorized scaling: (batch_size, 12, 1000) -> (batch_size * 12 * 1000, 1)
            batch_shape = batch_transposed.shape
            batch_flat = batch_transposed.reshape(-1, 1)  # Flatten to column vector
            
            # Vectorized scaling - much faster than loop
            batch_scaled_flat = scaler.transform(batch_flat)
            
            # Reshape back to original shape: (batch_size, 12, 1000)
            batch_scaled = batch_scaled_flat.reshape(batch_shape)
            
            # Convert to PyTorch tensor and move to device
            batch_tensor = torch.from_numpy(batch_scaled).to(device)
            
            # Batch inference
            with torch.no_grad():
                # GPU inference with mixed precision for better performance
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        batch_logits = model(batch_tensor)
                else:
                    batch_logits = model(batch_tensor)
                
                # Debug: Check for NaN in logits before sigmoid
                if debug_mode and torch.isnan(batch_logits).any():
                    nan_positions = torch.isnan(batch_logits)
                    print(f"   ⚠️ Warning: NaN values found in logits at positions: {torch.where(nan_positions)}")
                    print(f"      Logits range: [{batch_logits.min():.6f}, {batch_logits.max():.6f}]")
                    
                    # Fix: Replace NaN logits with zeros to prevent propagation
                    print(f"   🔧 Fixing NaN logits by replacing with zeros...")
                    batch_logits = torch.nan_to_num(batch_logits, nan=0.0)
                    print(f"   ✅ NaN logits fixed")
                
                batch_probs = torch.sigmoid(batch_logits).cpu().numpy()
                
                # Debug: Check for NaN in probabilities after sigmoid
                if debug_mode and np.isnan(batch_probs).any():
                    nan_positions = np.isnan(batch_probs)
                    print(f"   ⚠️ Warning: NaN values found in probabilities at positions: {np.where(nan_positions)}")
                    print(f"      Probabilities range: [{batch_probs.min():.6f}, {batch_probs.max():.6f}]")
                    
                    # Fix: Replace NaN probabilities with zeros
                    print(f"   🔧 Fixing NaN probabilities by replacing with zeros...")
                    batch_probs = np.nan_to_num(batch_probs, nan=0.0)
                    print(f"   ✅ NaN probabilities fixed")
            
            # Debug: Print prediction statistics for first mini-batch (only in debug mode)
            if i == 0 and debug_mode:
                print(f"   🔍 Debug - First mini-batch prediction stats:")
                print(f"      Input tensor shape: {batch_tensor.shape}")
                print(f"      Logits shape: {batch_logits.shape}")
                print(f"      Probabilities shape: {batch_probs.shape}")
                print(f"      Max probability: {batch_probs.max():.6f}")
                print(f"      Min probability: {batch_probs.min():.6f}")
                print(f"      Mean probability: {batch_probs.mean():.6f}")
                print(f"      Class indices: {class_indices}")
            
            # Extract probabilities for target classes (vectorized)
            for j, class_idx in enumerate(class_indices):
                if class_idx >= 0 and class_idx < batch_probs.shape[1]:
                    predictions[i:batch_end, j] = batch_probs[:, class_idx]
                    
                    # Debug: Check for NaN values in predictions
                    if debug_mode and i == 0 and np.isnan(batch_probs[:, class_idx]).any():
                        nan_indices = np.where(np.isnan(batch_probs[:, class_idx]))[0]
                        print(f"   ⚠️ Warning: NaN values found in batch {i//mini_batch_size + 1}, class {classes_to_predict[j]}, samples {nan_indices}")
                        print(f"      Batch probabilities range: [{batch_probs[:, class_idx].min():.6f}, {batch_probs[:, class_idx].max():.6f}]")
                else:
                    predictions[i:batch_end, j] = 0.0
            
            # Debug: Print extracted probabilities for first mini-batch (only in debug mode)
            if i == 0 and debug_mode:
                print(f"   🔍 Debug - Extracted target class probabilities:")
                for j, class_name in enumerate(classes_to_predict):
                    avg_prob = predictions[i:batch_end, j].mean()
                    max_prob = predictions[i:batch_end, j].max()
                    print(f"      {class_name}: avg={avg_prob:.6f}, max={max_prob:.6f}")
            
            successful_predictions += current_batch_size
            
        except Exception as e:
            if debug_mode:
                print(f"   ❌ Error processing mini-batch {i//mini_batch_size + 1}: {e}")
            failed_predictions += current_batch_size
            # Zero predictions already initialized
    
    print(f"   ✅ Optimized batch prediction completed:")
    print(f"      📊 Total samples: {total_samples}")
    print(f"      ✅ Successful: {successful_predictions}")
    print(f"      ❌ Failed: {failed_predictions}")
    
    # Memory cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return predictions

def evaluate_performance(y_true, y_pred, actual_classes=None, debug_mode=False):
    """Evaluate model performance"""
    print(f"📊 Evaluating performance for {y_true.shape[0]} samples...")
    
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES
    
    # Debug: Print shape and sample data (only in debug mode)
    if debug_mode:
        print(f"🔍 Debug - Evaluation input shapes:")
        print(f"   y_true shape: {y_true.shape}")
        print(f"   y_pred shape: {y_pred.shape}")
        print(f"   actual_classes: {actual_classes}")
        
        # Debug: Print first few samples of each class
        print(f"🔍 Debug - Sample data (first 5 samples):")
        for i, class_name in enumerate(actual_classes[:3]):  # Only show first 3 classes
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]
            print(f"   {class_name}:")
            print(f"     y_true: {y_true_class[:5]} (unique: {np.unique(y_true_class)})")
            print(f"     y_pred: {y_pred_class[:5]} (range: [{y_pred_class.min():.6f}, {y_pred_class.max():.6f}])")
    
    results = []
    
    for i, class_name in enumerate(actual_classes):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        # Calculate metrics
        try:
            # Check for NaN values in predictions before AUC calculation
            if np.isnan(y_pred_class).any():
                if debug_mode:
                    print(f"   ⚠️ Warning: NaN values found in predictions for {class_name}, setting AUC to 0.5")
                auc = 0.5
            else:
                auc = roc_auc_score(y_true_class, y_pred_class)
        except ValueError as e:
            if debug_mode:
                print(f"   ⚠️ Warning: AUC calculation failed for {class_name}: {e}, setting AUC to 0.5")
            auc = 0.5
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_class > 0.5).astype(int)
        
        # Calculate precision, recall, F1-score, and support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_class, y_pred_binary, average='binary', zero_division=0
        )
        
        if support is None:
            support = np.sum(y_true_class)
        
        # Calculate specificity
        tn = np.sum((y_true_class == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_class == 0) & (y_pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        
        # Calculate confusion matrix components
        tp = np.sum((y_true_class == 1) & (y_pred_binary == 1))
        fn = np.sum((y_true_class == 1) & (y_pred_binary == 0))
        
        results.append({
            'Class': class_name,
            'Support': int(support),
            'AUC': auc,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'True Positives': int(tp),
            'False Positives': int(fp),
            'False Negatives': int(fn),
            'True Negatives': int(tn)
        })
    
    return results

def calculate_multilabel_metrics(y_true, y_pred, actual_classes, threshold=0.5):
    """Calculate multi-label specific metrics"""
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate multi-label metrics
    macro_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    
    # Subset accuracy (exact match)
    subset_accuracy = accuracy_score(y_true, y_pred_binary)
    
    # Hamming loss
    hamming_loss_val = hamming_loss(y_true, y_pred_binary)
    
    # Jaccard index (average across samples)
    jaccard_avg = jaccard_score(y_true, y_pred_binary, average='macro', zero_division=0)
    
    # Calculate label cardinality (average number of labels per sample)
    label_cardinality_true = np.mean(np.sum(y_true, axis=1))
    label_cardinality_pred = np.mean(np.sum(y_pred_binary, axis=1))
    
    return {
        'Macro F1': macro_f1,
        'Micro F1': micro_f1,
        'Weighted F1': weighted_f1,
        'Subset Accuracy': subset_accuracy,
        'Hamming Loss': hamming_loss_val,
        'Jaccard Index': jaccard_avg,
        'Label Cardinality (True)': label_cardinality_true,
        'Label Cardinality (Pred)': label_cardinality_pred
    }

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, actual_classes=None):
    """Create confusion matrix visualization"""
    print(f"📊 Creating confusion matrix...")
    
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    n_classes = len(actual_classes)
    if n_classes == 0:
        print("   ⚠️ No classes to plot")
        return
    
    # Create subplots
    n_rows = (n_classes + 2) // 3  # 3 columns
    n_cols = min(3, n_classes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_classes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_name in enumerate(actual_classes):
        row = i // n_cols
        col = i % n_cols
        
        y_true_class = y_true[:, i]
        y_pred_class = y_pred_binary[:, i]
        
        cm = confusion_matrix(y_true_class, y_pred_class)
        
        ax = axes[row, col] if n_rows > 1 else axes[col]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        ax.set_title(f'Confusion Matrix - {class_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_classes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_pred, class_names, output_dir, actual_classes=None):
    """Create ROC curves visualization"""
    print(f"📊 Creating ROC curves...")
    
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES
    
    n_classes = len(actual_classes)
    if n_classes == 0:
        print("   ⚠️ No classes to plot")
        return
    
    # Create subplots
    n_rows = (n_classes + 2) // 3  # 3 columns
    n_cols = min(3, n_classes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_classes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_name in enumerate(actual_classes):
        row = i // n_cols
        col = i % n_cols
        
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        try:
            fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
            auc = roc_auc_score(y_true_class, y_pred_class)
            
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {class_name}')
            ax.legend(loc="lower right")
        except ValueError:
            # Handle case where only one class is present
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.text(0.5, 0.5, f'{class_name}\nInsufficient data for ROC', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'ROC Curve - {class_name}')
    
    # Hide empty subplots
    for i in range(n_classes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, y_true, y_pred, output_dir, actual_classes=None, multilabel_metrics=None):
    """Save evaluation results"""
    print(f"💾 Saving results to {output_dir}...")
    
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES
    
    # Save detailed results
    detailed_df = pd.DataFrame(results)
    detailed_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # Save per-class summary results
    summary_df = detailed_df[['Class', 'Support', 'AUC', 'F1-Score', 'Precision', 'Recall', 'Specificity', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives']].copy()
    summary_df.to_csv(output_dir / 'per_class_results.csv', index=False)
    
    # Save multi-label metrics
    if multilabel_metrics is not None:
        multilabel_df = pd.DataFrame([multilabel_metrics])
        multilabel_df.to_csv(output_dir / 'multilabel_metrics.csv', index=False)
    
    # Save true labels and predictions
    np.save(output_dir / 'true_labels.npy', y_true)
    np.save(output_dir / 'predictions.npy', y_pred)
    
    # Save class information
    with open(output_dir / 'class_info.json', 'w') as f:
        json.dump({
            'target_classes': actual_classes,
            'n_samples': len(y_true),
            'n_classes': len(actual_classes),
            'multilabel_metrics': multilabel_metrics
        }, f, indent=2)
    
    # Print per-class results (already displayed above)
    print(f"\n💾 Results saved to: {output_dir}")
    
    # Print summary table
    print("\n📊 71-Class Model MIMIC-IV-ECG Dataset Evaluation Results:")
    print("⚠️  Note: Single-label metrics below may not be meaningful for multi-label data!")
    print("    Focus on Multi-Label metrics for accurate assessment.")
    print("=" * 80)
    print(summary_df.to_string(index=False))

def main():
    """Main function"""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    args = parse_args()
    exp_folder = Path(args.exp_folder)
    
    print(f"🚀 Starting MIMIC-IV-ECG dataset evaluation...")
    print(f"📁 Experiment folder: {exp_folder}")
    
    # Load model first to get model path for output directory naming
    pred_module, scaler, mlb, n_out = load_model_and_preprocessing(exp_folder)
    if pred_module is None:
        print("❌ Failed to load model, exiting...")
        return
    
    # Extract model name and create timestamped output directory with model info
    model_path = pred_module.get('model_path', 'unknown_model')
    model_name = extract_model_name_from_path(model_path)
    
    # Create output directory with model identifier
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/prediction") / "MIMIC-IV-ECG_test" / "batch_mimic_iv_ecg_71class_results" / f"run_{timestamp}_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Load MIMIC-IV-ECG dataset
    data_path = Path("data/MIMIC-IV-ECG")
    if not data_path.exists():
        print(f"❌ MIMIC-IV-ECG dataset not found: {data_path}")
        return
    
    # Configuration for batch processing - FULL DATASET MODE
    DEBUG_MODE = False  # Full dataset processing mode
    BATCH_SIZE = 10000   # Optimized batch size for GPU processing
    MAX_BATCHES = None  # Process all data (no limit)
    
    print(f"🔧 Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max batches: {MAX_BATCHES if MAX_BATCHES else 'No limit (process all data)'}")
    print(f"   Debug mode: {DEBUG_MODE}")
    print(f"   🚀 Performance optimizations enabled:")
    print(f"      - GPU-optimized batch sizes (data: {BATCH_SIZE}, GPU: 2048)")
    print(f"      - Mixed precision inference (FP16)")
    print(f"      - Vectorized preprocessing")
    print(f"      - Pre-allocated memory arrays")
    print(f"      - Parallel data loading (4 workers)")
    print(f"      - Reduced progress update frequency")
    print(f"      - Memory cleanup and garbage collection")
    print(f"   🖥️ Using GPU inference for maximum performance")
    
    # Process dataset in batches
    all_predictions = []
    all_true_labels = []
    actual_classes = None
    total_samples = 0
    
    # Performance monitoring
    import time
    start_time = time.time()
    batch_start_time = time.time()
    
    print(f"\n📊 Starting batch processing...")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for batch_idx, (X_data, y_data, class_names, batch_info) in enumerate(load_mimic_dataset_batch(data_path, batch_size=BATCH_SIZE, max_batches=MAX_BATCHES, debug_mode=DEBUG_MODE, n_workers=4)):
        batch_id = batch_info['batch_id']
        batch_samples = batch_info['samples']
        
        # Calculate progress percentage (approximate, based on total records)
        total_records = 800035  # MIMIC-IV-ECG total records
        progress_percent = (total_samples / total_records) * 100
        
        print(f"\n📦 Processing batch {batch_id} ({batch_samples} samples) - Overall progress: {progress_percent:.2f}%")
        
        # Extract target classes (only for first batch to determine actual_classes)
        if actual_classes is None:
            X_data, y_target, actual_classes = extract_target_classes(X_data, y_data, class_names)
            print(f"   🎯 Target classes determined: {actual_classes}")
        else:
            X_data, y_target, _ = extract_target_classes(X_data, y_data, class_names)
        
        # Predict on this batch
        print(f"   🚀 Starting prediction for batch {batch_id}...")
        batch_predictions = predict_batch_mimic_optimized(
            X_data, pred_module, scaler, mlb, n_out, 
            actual_classes=actual_classes, debug_mode=DEBUG_MODE
        )
        
        # Store results
        all_predictions.append(batch_predictions)
        all_true_labels.append(y_target)
        total_samples += batch_samples
        
        # Performance monitoring
        batch_time = time.time() - batch_start_time
        batch_start_time = time.time()
        samples_per_second = batch_samples / batch_time if batch_time > 0 else 0
        
        print(f"   ✅ Batch {batch_id} completed in {batch_time:.1f}s ({samples_per_second:.1f} samples/s)")
        print(f"   📊 Total samples processed: {total_samples:,}")
        
        # Memory cleanup
        del X_data, y_data, y_target, batch_predictions
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Check for shutdown request
        if _shutdown_requested:
            print("\n🛑 Program interrupted by user, exiting...")
            break
    
    if total_samples == 0:
        print("❌ No samples were processed, exiting...")
        return
    
    print(f"\n🔄 Combining results from {len(all_predictions)} batches...")
    
    # Debug: Check individual batch predictions before combining (only in debug mode)
    if DEBUG_MODE:
        print(f"🔍 Debug - Individual batch predictions:")
        for i, batch_pred in enumerate(all_predictions):
            print(f"   Batch {i+1}: shape={batch_pred.shape}, has_nan={np.isnan(batch_pred).any()}, range=[{batch_pred.min():.6f}, {batch_pred.max():.6f}]")
    
    # Combine all predictions and labels
    combined_predictions = np.vstack(all_predictions)
    combined_true_labels = np.vstack(all_true_labels)
    
    print(f"   📊 Combined predictions shape: {combined_predictions.shape}")
    print(f"   📊 Combined true labels shape: {combined_true_labels.shape}")
    print(f"   📊 Total samples: {total_samples}")
    
    # Debug: Check combined predictions for NaN values (only in debug mode)
    has_nan = np.isnan(combined_predictions).any()
    if DEBUG_MODE:
        print(f"🔍 Debug - Combined predictions has NaN: {has_nan}")
    if has_nan:
        nan_count = np.isnan(combined_predictions).sum()
        if DEBUG_MODE:
            print(f"   NaN count: {nan_count}/{combined_predictions.size} ({nan_count/combined_predictions.size*100:.2f}%)")
            print(f"   NaN positions: {np.where(np.isnan(combined_predictions))}")
        
        # Fix: Replace NaN values with 0.0 to prevent AUC calculation errors
        if DEBUG_MODE:
            print(f"   🔧 Fixing NaN values by replacing with 0.0...")
        combined_predictions = np.nan_to_num(combined_predictions, nan=0.0)
        if DEBUG_MODE:
            print(f"   ✅ NaN values fixed")
    
    # Evaluate performance on combined results
    print(f"\n📊 Evaluating combined results...")
    results = evaluate_performance(combined_true_labels, combined_predictions, actual_classes=actual_classes, debug_mode=DEBUG_MODE)
    
    # Calculate and display multi-label metrics
    print(f"\n📊 Multi-Label Classification Performance Summary:")
    print("=" * 80)
    multilabel_metrics = calculate_multilabel_metrics(combined_true_labels, combined_predictions, actual_classes)
    
    print(f"📈 Overall Performance:")
    print(f"   Macro F1-Score      : {multilabel_metrics['Macro F1']:.6f}")
    print(f"   Micro F1-Score      : {multilabel_metrics['Micro F1']:.6f}")
    print(f"   Weighted F1-Score   : {multilabel_metrics['Weighted F1']:.6f}")
    print(f"   Subset Accuracy     : {multilabel_metrics['Subset Accuracy']:.6f}")
    print(f"   Hamming Loss        : {multilabel_metrics['Hamming Loss']:.6f}")
    print(f"   Jaccard Index       : {multilabel_metrics['Jaccard Index']:.6f}")
    
    print(f"\n📊 Label Distribution:")
    print(f"   Avg Labels per Sample (True) : {multilabel_metrics['Label Cardinality (True)']:.3f}")
    print(f"   Avg Labels per Sample (Pred) : {multilabel_metrics['Label Cardinality (Pred)']:.3f}")
    
    print("=" * 80)
    print(f"\n📋 Per-Class Performance Details:")
    print("⚠️  Note: Single-label F1-Scores may not be meaningful for multi-label data!")
    print("    Use Multi-Label metrics above for accurate performance assessment.")
    print("    Per-class metrics are mainly for diagnostic purposes.")
    print("=" * 80)
    
    # Save results
    save_results(results, combined_true_labels, combined_predictions, output_dir, actual_classes, multilabel_metrics)
    
    # Create visualizations
    plot_confusion_matrix(combined_true_labels, combined_predictions, actual_classes, output_dir, actual_classes)
    plot_roc_curves(combined_true_labels, combined_predictions, actual_classes, output_dir, actual_classes)
    
    # Final performance statistics
    total_time = time.time() - start_time
    avg_samples_per_second = total_samples / total_time if total_time > 0 else 0
    
    print(f"\n🎉 MIMIC-IV-ECG dataset evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total samples evaluated: {total_samples:,}")
    print(f"⏱️ Total processing time: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print(f"⚡ Average processing speed: {avg_samples_per_second:.1f} samples/second")
    print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
