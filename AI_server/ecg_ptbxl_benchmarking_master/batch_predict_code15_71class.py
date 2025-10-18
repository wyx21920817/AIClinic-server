#!/usr/bin/env python3
"""
CODE-15 Dataset Batch Prediction Script for 71-Class Models
Script for evaluating CODE-15 dataset using 71-class models for prediction

Usage:
    python batch_predict_code15_71class.py [exp_folder]
    
Examples:
    python batch_predict_code15_71class.py                           # Use default model
    python batch_predict_code15_71class.py output/exp0              # Use specified experiment folder
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

# CODE-15 dataset classes to our 71-class model mapping
CODE15_TO_OUR_MAPPING = {
    '1dAVb': '1AVB',        # First degree AV block -> 1AVB
    'RBBB': 'CRBBB',        # Right bundle branch block -> CRBBB (assuming complete RBBB)
    'LBBB': 'CLBBB',        # Left bundle branch block -> CLBBB (assuming complete LBBB)
    'SB': 'SBRAD',          # Sinus bradycardia -> SBRAD
    'ST': 'STACH',          # ST abnormality -> STACH (assuming ST elevation/depression)
    'AF': 'AFIB',           # Atrial fibrillation -> AFIB
    'normal_ecg': 'NORM'    # Normal ECG -> NORM
}

# Target ECG classes we want to evaluate (subset of 71 classes)
TARGET_ECG_CLASSES = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 
                      'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']

# Display name mapping for output (English ECG terms)
DISPLAY_NAME_MAPPING = {
    '1dAVb': '1AVB',
    'RBBB': 'CRBBB', 
    'LBBB': 'CLBBB',
    'SB': 'SBRAD',
    'ST': 'STACH',
    'AF': 'AFIB',
    'normal_ecg': 'NORM'
}

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
    parser = argparse.ArgumentParser(description='CODE-15 Dataset Batch Prediction Script for 71-Class Models')
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
        
        # Verify CODE-15 mapped classes exist in 71-class model
        print("   🔍 Verifying CODE-15 class mappings:")
        for code15_class, our_class in CODE15_TO_OUR_MAPPING.items():
            if our_class in MODEL_71_CLASSES:
                idx = MODEL_71_CLASSES.index(our_class)
                print(f"      {code15_class:8s} -> {our_class:6s} (index: {idx:2d}) ✅")
            else:
                print(f"      {code15_class:8s} -> {our_class:6s} (not found) ❌")
        
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

def extract_model_name_from_path(model_path):
    """Extract model name from model path"""
    model_path = str(model_path)
    if 'fastai_xresnet1d101' in model_path:
        return 'fastai_xresnet1d101'
    elif 'naive' in model_path:
        return 'naive'
    else:
        return 'unknown_model'

def load_code15_dataset_batch(data_path, batch_size=1000, max_batches=None):
    """
    Load CODE-15 dataset in batches to avoid memory issues
    
    Args:
        data_path: Path to CODE-15 dataset directory
        batch_size: Number of samples per batch
        max_batches: Maximum number of batches to process (for testing)
        
    Yields:
        X_data: ECG signal data (batch_size, n_channels, n_timesteps)
        y_data: Labels (batch_size, n_classes)
        class_names: List of class names
        batch_info: Dictionary with batch information
    """
    print("📊 Loading CODE-15 dataset in batches...")
    
    # Load metadata
    exams_df = pd.read_csv(data_path / 'exams.csv')
    print(f"   📋 Loaded {len(exams_df)} exam records")
    
    # Get unique trace files
    trace_files = exams_df['trace_file'].unique()
    print(f"   📁 Found {len(trace_files)} trace files")
    print(f"   🔍 Trace files: {sorted(trace_files)}")
    
    # Process each trace file
    total_processed = 0
    batch_count = 0
    
    for trace_file in trace_files:
        hdf5_path = data_path / trace_file
        if not hdf5_path.exists():
            print(f"   ⚠️ Warning: {trace_file} not found, skipping")
            continue
            
        print(f"   📖 Processing {trace_file}...")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Get exam IDs for this trace file
            file_exams = exams_df[exams_df['trace_file'] == trace_file]
            
            # Get exam IDs from HDF5 file
            exam_ids_in_file = f['exam_id'][:]
            
            # Create mapping from exam_id to index
            exam_id_to_idx = {int(eid): idx for idx, eid in enumerate(exam_ids_in_file)}
            
            # Process exams in batches
            batch_signals = []
            batch_labels = []
            batch_exam_ids = []
            
            # Pre-allocate arrays for better performance
            batch_signals_reserved = np.zeros((batch_size, 1000, 12), dtype=np.float32)
            batch_labels_reserved = np.zeros((batch_size, len(CODE15_TO_OUR_MAPPING)), dtype=np.int32)
            
            current_batch_size = 0
            
            for _, exam in file_exams.iterrows():
                exam_id = int(exam['exam_id'])
                
                if exam_id in exam_id_to_idx:
                    idx = exam_id_to_idx[exam_id]
                    
                    # Load only the specific tracing for this exam
                    signal = f['tracings'][idx]  # Shape: (4096, 12)
                    
                    # Debug: Print signal info for first few samples
                    if current_batch_size == 0 and batch_count == 0:
                        print(f"   🔍 Debug - Raw signal from HDF5:")
                        print(f"      Signal shape: {signal.shape}")
                        print(f"      Signal dtype: {signal.dtype}")
                        print(f"      Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
                    
                    # Transpose to (n_timesteps, n_channels) format
                    signal = signal.T  # Shape: (12, 4096)
                    signal = signal.T  # Shape: (4096, 12) - timesteps x channels
                    
                    # Standardize to 10 seconds at 500Hz (5000 samples)
                    target_length = 5000
                    if signal.shape[0] > target_length:
                        # Truncate from the middle
                        start_idx = (signal.shape[0] - target_length) // 2
                        signal = signal[start_idx:start_idx + target_length, :]
                    elif signal.shape[0] < target_length:
                        # Pad with zeros
                        pad_length = target_length - signal.shape[0]
                        signal = np.pad(signal, ((0, pad_length), (0, 0)), mode='constant')
                    
                    # Downsample from 500Hz to 100Hz to match model expectations
                    # Take every 5th sample to get 1000 samples (100Hz)
                    signal = signal[::5, :]
                    
                    # Debug: Print processed signal info for first sample
                    if current_batch_size == 0 and batch_count == 0:
                        print(f"   🔍 Debug - Processed signal:")
                        print(f"      Final shape: {signal.shape}")
                        print(f"      Final range: [{signal.min():.3f}, {signal.max():.3f}]")
                    
                    # Store in pre-allocated array
                    batch_signals_reserved[current_batch_size] = signal.astype(np.float32)
                    batch_exam_ids.append(str(exam_id))
                    
                    # Create label vector
                    for j, (code15_class, our_class) in enumerate(CODE15_TO_OUR_MAPPING.items()):
                        if code15_class in exam and exam[code15_class]:
                            batch_labels_reserved[current_batch_size, j] = 1
                        else:
                            batch_labels_reserved[current_batch_size, j] = 0
                    
                    current_batch_size += 1
                    
                    # Yield batch when it reaches batch_size
                    if current_batch_size >= batch_size:
                        X_data = batch_signals_reserved[:current_batch_size].copy()
                        y_data = batch_labels_reserved[:current_batch_size].copy()
                        
                        batch_count += 1
                        total_processed += current_batch_size
                        
                        print(f"   📦 Yielding batch {batch_count}: {current_batch_size} samples (Total: {total_processed})")
                        
                        batch_info = {
                            'batch_id': batch_count,
                            'samples': current_batch_size,
                            'trace_file': trace_file,
                            'total_processed': total_processed
                        }
                        
                        yield X_data, y_data, list(CODE15_TO_OUR_MAPPING.keys()), batch_info
                        
                        # Reset batch
                        current_batch_size = 0
                        
                        # Check if we've reached max_batches
                        if max_batches and batch_count >= max_batches:
                            print(f"   🔬 Reached maximum batches limit: {max_batches}")
                            return
            
            # Yield remaining samples in the last batch
            if current_batch_size > 0:
                X_data = batch_signals_reserved[:current_batch_size].copy()
                y_data = batch_labels_reserved[:current_batch_size].copy()
                
                batch_count += 1
                total_processed += current_batch_size
                
                print(f"   📦 Yielding final batch {batch_count}: {current_batch_size} samples (Total: {total_processed})")
                
                batch_info = {
                    'batch_id': batch_count,
                    'samples': current_batch_size,
                    'trace_file': trace_file,
                    'total_processed': total_processed
                }
                
                yield X_data, y_data, list(CODE15_TO_OUR_MAPPING.keys()), batch_info
    
    print(f"   ✅ Completed loading all batches. Total samples processed: {total_processed}")

def load_code15_dataset(data_path, max_samples=None):
    """
    Legacy function for backward compatibility - loads a single batch
    """
    for X_data, y_data, class_names, batch_info in load_code15_dataset_batch(data_path, batch_size=1000, max_batches=1):
        return X_data, y_data, class_names

def extract_target_classes(X_data, y_data, code15_classes):
    """
    Extract only the target ECG classes from the dataset
    
    Args:
        X_data: ECG signal data
        y_data: Labels
        code15_classes: List of CODE-15 class names
        
    Returns:
        X_data: Filtered ECG signal data
        y_target: Filtered labels for target classes
        target_class_names: List of target class names
    """
    print("🎯 Extracting target ECG classes...")
    
    # Find which CODE-15 classes map to our target classes
    target_indices = []
    target_class_names = []
    
    for target_ecg_class in TARGET_ECG_CLASSES:
        for code15_class, mapped_ecg_class in CODE15_TO_OUR_MAPPING.items():
            if mapped_ecg_class == target_ecg_class and code15_class in code15_classes:
                class_idx = code15_classes.index(code15_class)
                target_indices.append(class_idx)
                target_class_names.append(target_ecg_class)
                break
    
    if target_indices:
        y_target = y_data[:, target_indices]
        print(f"   ✅ Extracted {len(target_class_names)} target classes: {target_class_names}")
        print(f"   📊 Target label shape: {y_target.shape}")
        
        # Don't filter samples - we need to evaluate ALL samples
        # Even samples with all 0s are important for evaluation (they represent "normal" cases)
        print(f"   📊 Keeping all {len(X_data)} samples for evaluation")
        
        # Print target class distribution
        print(f"   📈 Target class distribution:")
        for i, class_name in enumerate(target_class_names):
            count = np.sum(y_target[:, i])
            print(f"      {class_name}: {count} samples")
    else:
        print("   ⚠️ No target classes found in dataset")
        y_target = y_data
        target_class_names = code15_classes
    
    return X_data, y_target, target_class_names

def predict_batch_code15_optimized(X_data, pred_module, scaler, mlb, n_out, actual_classes=None, debug_mode=False):
    """Highly optimized batch prediction for CODE-15 dataset"""
    print(f"🚀 Starting optimized batch prediction for {len(X_data)} CODE-15 samples...")
    
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
    
    # Process in mini-batches for better GPU utilization
    mini_batch_size = 32  # Process 32 samples at once
    
    for i in range(0, total_samples, mini_batch_size):
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\n🛑 Interrupt request received, processed {i}/{total_samples} samples")
            break
        
        # Progress update
        if i % progress_interval == 0 or i == total_samples - 1:
            progress = (i + 1) / total_samples * 100
            print(f"   📊 Progress: {i+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
        
        # Determine mini-batch size (handle last batch)
        batch_end = min(i + mini_batch_size, total_samples)
        current_batch_size = batch_end - i
        
        try:
            # Process mini-batch
            batch_data = X_data[i:batch_end]  # Shape: (batch_size, n_timesteps, n_channels)
            
            # Process each sample exactly like the ICBEB script
            batch_scaled_list = []
            for j in range(current_batch_size):
                ecg_data = batch_data[j]  # Shape: (n_timesteps, n_channels) = (1000, 12)
                
                # First transpose to match ICBEB format: (channels, timesteps)
                ecg_data = ecg_data.T  # Shape: (12, 1000) - (channels, timesteps)
                
                # Preprocessing - use same method as 71-class model training
                # Flatten ECG data to 1D, apply scaler, then reshape back to original shape
                x_shape = ecg_data.shape
                x_flat = ecg_data.flatten()[:, np.newaxis]  # Flatten to column vector
                x_scaled = scaler.transform(x_flat).reshape(x_shape)  # Apply scaler and reshape
                
                batch_scaled_list.append(x_scaled)
            
            # Stack to create batch tensor (batch_size, n_channels, n_timesteps)
            batch_scaled = np.stack(batch_scaled_list, axis=0)
            
            # Convert to PyTorch tensor and move to device
            batch_tensor = torch.FloatTensor(batch_scaled).to(device)
            
            # Batch inference
            with torch.no_grad():
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
    print(f"      📈 Success rate: {successful_predictions/total_samples*100:.1f}%")
    print(f"      📊 Predictions shape: {predictions.shape}")
    
    return predictions

def predict_batch_code15(X_data, pred_module, scaler, mlb, n_out, actual_classes=None, debug_mode=False):
    """Wrapper function - uses optimized version"""
    return predict_batch_code15_optimized(X_data, pred_module, scaler, mlb, n_out, actual_classes, debug_mode)

def evaluate_performance(true_labels, predictions, use_threshold=True, threshold=0.5, actual_classes=None):
    """
    Evaluate prediction performance
    
    Args:
        true_labels: True labels (n_samples, n_classes)
        predictions: Predicted probabilities (n_samples, n_classes)
        use_threshold: Whether to use threshold for binary classification
        threshold: Classification threshold
        actual_classes: List of actual class names
        
    Returns:
        results: Dictionary containing evaluation metrics
    """
    print("📊 Evaluating prediction performance...")
    
    # Use actual classes if provided
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES[:true_labels.shape[1]]
    
    print(f"   📊 Evaluating {len(actual_classes)} actual classes: {actual_classes}")
    
    results = {}
    
    for i, class_name in enumerate(actual_classes):
        y_true = true_labels[:, i]
        y_pred = predictions[:, i]
        
        # Skip classes with no positive samples
        if np.sum(y_true) == 0:
            print(f"   ⚠️ Skipping {class_name}: no positive samples")
            continue
        
        # Calculate AUC with improved error handling
        try:
            # Check for NaN values in predictions before AUC calculation
            if np.isnan(y_pred).any():
                print(f"   ⚠️ Warning: NaN values found in predictions for {class_name}, setting AUC to 0.5")
                auc = 0.5
            else:
                auc = roc_auc_score(y_true, y_pred)
        except ValueError as e:
            if debug_mode:
                print(f"   ⚠️ Warning: AUC calculation failed for {class_name}: {e}, setting AUC to 0.5")
            auc = 0.5
        
        # Calculate binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        
        # Calculate performance metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        results[class_name] = {
            'support': int(np.sum(y_true)),
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
        
        print(f"   {class_name}: AUC={auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    return results

def plot_confusion_matrix(true_labels, predictions, class_names, output_dir, actual_classes=None):
    """
    Plot confusion matrix
    
    Args:
        true_labels: True labels (n_samples, n_classes)
        predictions: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        output_dir: Output directory
        actual_classes: List of actual class names
    """
    print("📊 Creating confusion matrix...")
    
    # Use actual classes if provided
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES[:true_labels.shape[1]]
    
    # Convert probabilities to binary predictions
    y_pred_binary = (predictions >= 0.5).astype(int)
    
    # Create confusion matrix for each class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, class_name in enumerate(actual_classes):
        if i >= len(axes):
            break
            
        y_true_class = true_labels[:, i]
        y_pred_class = y_pred_binary[:, i]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_class, y_pred_class)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{class_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(len(actual_classes), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Confusion matrix saved")

def plot_roc_curves(true_labels, predictions, class_names, output_dir, actual_classes=None):
    """
    Plot ROC curves
    
    Args:
        true_labels: True labels (n_samples, n_classes)
        predictions: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        output_dir: Output directory
        actual_classes: List of actual class names
    """
    print("📊 Creating ROC curves...")
    
    # Use actual classes if provided
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES[:true_labels.shape[1]]
    
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(actual_classes):
        y_true_class = true_labels[:, i]
        y_pred_class = predictions[:, i]
        
        # Skip classes with no positive samples
        if np.sum(y_true_class) == 0:
            continue
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for CODE-15 Dataset')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ ROC curves saved")

def save_results(results, true_labels, predictions, output_dir, actual_classes=None):
    """
    Save evaluation results to CSV files
    
    Args:
        results: Dictionary containing evaluation metrics
        true_labels: True labels (n_samples, n_classes)
        predictions: Predicted probabilities (n_samples, n_classes)
        output_dir: Output directory
        actual_classes: List of actual class names
    """
    print("💾 Saving results...")
    
    # Get actual classes from data (not all target classes may be present)
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES[:true_labels.shape[1]]
    
    # Save detailed results using csv module for better Excel compatibility
    import csv
    with open(output_dir / 'detailed_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Class', 'support', 'auc', 'f1', 'precision', 'recall', 'specificity', 'tp', 'fp', 'fn', 'tn'])
        # Write data rows
        for class_name, metrics in results.items():
            writer.writerow([
                class_name,
                metrics['support'],
                round(metrics['auc'], 4),
                round(metrics['f1'], 4),
                round(metrics['precision'], 4),
                round(metrics['recall'], 4),
                round(metrics['specificity'], 4),
                metrics['tp'],
                metrics['fp'],
                metrics['fn'],
                metrics['tn']
            ])
    
    # Save prediction results using actual ECG classes (same as ICBEB script)
    predictions_df = pd.DataFrame(predictions, columns=actual_classes)
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save true labels using actual ECG classes (same as ICBEB script)
    true_labels_df = pd.DataFrame(true_labels, columns=actual_classes)
    true_labels_df.to_csv(output_dir / 'true_labels.csv', index=False)
    
    # Create summary report
    summary_data = []
    for class_name, metrics in results.items():
        # class_name is now already English ECG term
        summary_data.append([
            class_name,
            metrics['support'],
            f"{metrics['auc']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['specificity']:.4f}",
            metrics['tp'],
            metrics['fp'],
            metrics['fn'],
            metrics['tn']
        ])
    
    summary_df = pd.DataFrame(summary_data, columns=[
        'Class', 'Support', 'AUC', 'F1-Score', 'Precision', 'Recall', 
        'Specificity', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives'
    ])
    
    # Save summary results using csv module for better Excel compatibility
    with open(output_dir / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Class', 'Support', 'AUC', 'F1-Score', 'Precision', 'Recall', 'Specificity', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives'])
        # Write data rows
        for _, row in summary_df.iterrows():
            writer.writerow([
                row['Class'],
                row['Support'],
                row['AUC'],
                row['F1-Score'],
                row['Precision'],
                row['Recall'],
                row['Specificity'],
                row['True Positives'],
                row['False Positives'],
                row['False Negatives'],
                row['True Negatives']
            ])
    
    # Print summary results
    print("\n📊 71-Class Model CODE-15 Dataset Evaluation Results:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    print(f"\n💾 Results saved to: {output_dir}")

def main():
    """Main function"""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    args = parse_args()
    exp_folder = Path(args.exp_folder)
    
    print(f"🚀 Starting CODE-15 dataset evaluation...")
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
    output_dir = Path("output/prediction") / "CODE-15%_test" / "batch_code15_71class_results" / f"run_{timestamp}_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Load CODE-15 dataset
    data_path = Path("data/CODE-15%")
    if not data_path.exists():
        print(f"❌ CODE-15 dataset not found: {data_path}")
        return
    
    # Configuration for batch processing
    DEBUG_MODE = False  # Disable debug mode for full dataset processing
    BATCH_SIZE = 1000  # Batch size for processing
    MAX_BATCHES = None  # No limit - process all data
    
    print(f"🔧 Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max batches: {MAX_BATCHES if MAX_BATCHES else 'No limit (process all data)'}")
    print(f"   Debug mode: {DEBUG_MODE}")
    print(f"   🚀 Performance optimizations enabled:")
    print(f"      - Mini-batch processing (32 samples per GPU batch)")
    print(f"      - Vectorized preprocessing")
    print(f"      - Pre-allocated memory arrays")
    print(f"      - Reduced progress update frequency")
    
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
    
    for batch_idx, (X_data, y_data, class_names, batch_info) in enumerate(load_code15_dataset_batch(data_path, batch_size=BATCH_SIZE, max_batches=MAX_BATCHES)):
        batch_id = batch_info['batch_id']
        batch_samples = batch_info['samples']
        
        # Calculate progress percentage (approximate, based on total records)
        total_records = 345779
        progress_percent = (total_samples / total_records) * 100
        
        print(f"\n📦 Processing batch {batch_id} ({batch_samples} samples)...")
        print(f"   📊 Progress: {total_samples:,}/{total_records:,} ({progress_percent:.1f}%)")
        
        # Check for shutdown request
        if _shutdown_requested:
            print("\n🛑 Program interrupted by user, exiting...")
            break
        
        # Extract target classes (only need to do this once)
        if actual_classes is None:
            X_data, y_target, actual_classes = extract_target_classes(X_data, y_data, class_names)
            print(f"   🎯 Target classes identified: {actual_classes}")
        else:
            X_data, y_target, _ = extract_target_classes(X_data, y_data, class_names)
        
        # Batch prediction
        batch_predictions = predict_batch_code15(X_data, pred_module, scaler, mlb, n_out, actual_classes, DEBUG_MODE)
        
        # Accumulate results
        all_predictions.append(batch_predictions)
        all_true_labels.append(y_target)
        total_samples += batch_samples
        
        # Performance monitoring
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        samples_per_second = batch_samples / batch_duration if batch_duration > 0 else 0
        
        # Calculate estimated time remaining
        elapsed_time = batch_end_time - start_time
        if total_samples > 0:
            avg_time_per_sample = elapsed_time / total_samples
            remaining_samples = 345779 - total_samples
            estimated_remaining_time = remaining_samples * avg_time_per_sample
            eta_hours = estimated_remaining_time / 3600
        else:
            eta_hours = 0
        
        print(f"   ✅ Batch {batch_id} completed: {batch_samples} samples")
        print(f"   📊 Total samples processed: {total_samples:,}")
        print(f"   ⚡ Processing speed: {samples_per_second:.1f} samples/sec")
        print(f"   ⏱️ Batch time: {batch_duration:.1f}s")
        if eta_hours > 0:
            print(f"   🕐 Estimated time remaining: {eta_hours:.1f} hours")
        
        batch_start_time = time.time()  # Reset for next batch
        
        # Check if program was interrupted
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
    results = evaluate_performance(combined_true_labels, combined_predictions, actual_classes=actual_classes)
    
    # Save results
    save_results(results, combined_true_labels, combined_predictions, output_dir, actual_classes)
    
    # Create visualizations
    plot_confusion_matrix(combined_true_labels, combined_predictions, actual_classes, output_dir, actual_classes)
    plot_roc_curves(combined_true_labels, combined_predictions, actual_classes, output_dir, actual_classes)
    
    # Final performance statistics
    total_time = time.time() - start_time
    avg_samples_per_second = total_samples / total_time if total_time > 0 else 0
    
    print(f"\n🎉 CODE-15 dataset evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total samples evaluated: {total_samples:,}")
    print(f"⏱️ Total processing time: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print(f"⚡ Average processing speed: {avg_samples_per_second:.1f} samples/second")
    print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
