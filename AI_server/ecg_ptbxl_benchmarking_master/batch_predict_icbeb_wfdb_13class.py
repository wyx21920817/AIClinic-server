#!/usr/bin/env python3
"""
Ultra-Fast Batch ICBEB WFDB Prediction Script for 13-Class Models
ICBEB evaluation script using original WFDB format for 13-class models

Usage:
    python batch_predict_icbeb_wfdb_13class.py [exp_folder]
    
Examples:
    python batch_predict_icbeb_wfdb_13class.py                                    # Use default model
    python batch_predict_icbeb_wfdb_13class.py output/exp_13class_smart_multilabel  # Use specified experiment folder
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
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ICBEB to our class mapping (only evaluate classes present in ICBEB)
ICBEB_TO_OUR_MAPPING = {
    'NORM': 'NORM',
    'AFIB': 'AFIB', 
    '1AVB': '1AVB',
    'CLBBB': 'CLBBB',
    'CRBBB': 'CRBBB',
    'PAC': 'PAC',
    'VPC': 'PVC'  # ICBEB uses VPC, we use PVC
}

# Display name mapping (for output display)
DISPLAY_NAME_MAPPING = {
    'NORM': 'NORM',
    'AFIB': 'AFIB', 
    '1AVB': '1AVB',
    'CLBBB': 'CLBBB',
    'CRBBB': 'CRBBB',
    'PAC': 'PAC',
    'PVC': 'PVC'
}

# Global shutdown flag
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n🛑 Received signal {signum}, requesting graceful shutdown...")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ICBEB WFDB 13-Class Model Evaluation')
    parser.add_argument('exp_folder', nargs='?', 
                       default='output/exp_13class_smart_multilabel',
                       help='Experiment folder containing model and preprocessing files')
    return parser.parse_args()

def load_model_and_preprocessing(exp_folder):
    """Load 13-class model and preprocessing components"""
    try:
        print(f"🔧 Loading 13-class model and preprocessing components from: {exp_folder}")
        
        exp_path = Path(exp_folder)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment folder not found: {exp_folder}")
        
        # Load preprocessing components
        data_dir = exp_path / "data"
        models_dir = exp_path / "models"
        
        print(f"   📁 Models directory: {models_dir}")
        print(f"   📁 Data directory: {data_dir}")
        
        # Find model file
        model_files = list(models_dir.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in {models_dir}")
        
        # Use the first .pth file (should be the main model)
        model_path = model_files[0]
        print(f"   📁 Using model: {model_path}")
        
        # Load scaler and mlb
        scaler_path = data_dir / "standard_scaler.pkl"
        mlb_path = data_dir / "mlb.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not mlb_path.exists():
            raise FileNotFoundError(f"MLB not found: {mlb_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        
        print(f"   ✅ Loaded scaler and mlb from: {data_dir}")
        
        # Load class names
        class_names_path = data_dir / "class_names.txt"
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"   📋 Model has {len(class_names)} classes")
            print(f"   🔍 Verifying ICBEB class mappings:")
            for icbeb_class, our_class in ICBEB_TO_OUR_MAPPING.items():
                if our_class in class_names:
                    class_idx = class_names.index(our_class)
                    print(f"      {icbeb_class}             -> {our_class}   (index: {class_idx:2d}) ✅")
                else:
                    print(f"      {icbeb_class}             -> {our_class}   (index: N/A) ❌")
        else:
            print(f"   ⚠️  Class names file not found, using default 13 classes")
            class_names = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']
        
        # Load prediction module
        try:
            from classify_single_ecg128 import load_model
            globals_128 = {'load_model': load_model}
        except ImportError:
            raise ImportError("Could not import load_model from classify_single_ecg128")
        
        # Detect model output classes by loading and checking
        print(f"   🔍 Detecting model output classes...")
        try:
            # Try to load model temporarily to detect output size
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            temp_model = load_model(str(model_path), n_in=12, n_out=13, device=device)
            if hasattr(temp_model, 'classifier') and hasattr(temp_model.classifier, 'out_features'):
                n_out = temp_model.classifier.out_features
                print(f"   📊 Detected model output classes: {n_out}")
            else:
                n_out = 13  # Default 13 classes
                print(f"   ⚠️  Could not detect output classes, using default: {n_out}")
        except Exception as e:
            n_out = 13  # Default 13 classes
            print(f"   ⚠️  Could not detect output classes, using default: {n_out}")
        
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
        
        # Get ECG signal (p_signal is already in float format)
        ecg_data = record.p_signal.astype(np.float32)
        
        # Transpose to (channels, samples) format
        ecg_data = ecg_data.T
        
        if debug_mode:
            print(f"      After transpose: {ecg_data.shape}")
            
            # WFDB p_signal is already in proper units, no need to divide by 1000
            # DICOM normalization is different because it starts from int16 raw data
            print(f"      WFDB data already normalized: range [{ecg_data.min():.6f}, {ecg_data.max():.6f}]")
        
        return ecg_data
        
    except Exception as e:
        raise Exception(f"Failed to extract ECG from WFDB: {e}")

def predict_batch_fast(wfdb_paths, pred_module, scaler, mlb, n_out, class_names, debug_mode=False):
    """Fast batch prediction using WFDB format"""
    print(f"🚀 Starting optimized batch prediction for {len(wfdb_paths)} WFDB samples...")
    
    predictions = []
    successful_predictions = 0
    failed_predictions = 0
    
    # Pre-fetch model and device
    model = pred_module['model']
    device = pred_module['device']
    
    # Set warning flag
    warning_shown = False
    
    # Progress tracking (reduced frequency for better performance)
    total_samples = len(wfdb_paths)
    progress_interval = max(1, total_samples // 20)  # Update every 5%
    
    for i, wfdb_path in enumerate(wfdb_paths):
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\n🛑 Interrupt request received, processed {i}/{total_samples} samples")
            break
        
        # Progress update (reduced frequency)
        if i % progress_interval == 0 or i == total_samples - 1:
            progress = (i + 1) / total_samples * 100
            print(f"   📊 Progress: {i+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
            
        try:
            # Extract ECG data from WFDB
            ecg_data = extract_ecg_from_wfdb(wfdb_path, debug_mode)
            
            # Preprocessing - use same method as 13-class model training
            # Flatten ECG data to 1D, apply scaler, then reshape back to original shape
            x_shape = ecg_data.shape
            x_flat = ecg_data.flatten()[:, np.newaxis]  # Flatten to column vector
            x_scaled = scaler.transform(x_flat).reshape(x_shape)  # Apply scaler and reshape
            x = torch.from_numpy(x_scaled).unsqueeze(0)  # Add batch dimension
            
            x = x.to(device)
            
            # Inference (model already in eval mode, no need to call again)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                
                # Debug: print prediction info (only in debug mode)
                if debug_mode:
                    print(f"   🔍 Prediction Debug:")
                    print(f"      Input shape: {x.shape}")
                    print(f"      Logits shape: {logits.shape}")
                    print(f"      Probabilities shape: {probs.shape}")
                    print(f"      Max probability: {probs.max():.6f}")
                    print(f"      Min probability: {probs.min():.6f}")
                    print(f"      Mean probability: {probs.mean():.6f}")
            
            # Map 13-class predictions to ICBEB-related 7 classes
            # Create mapped prediction probability array (only for classes present in ICBEB)
            icbeb_classes = list(ICBEB_TO_OUR_MAPPING.keys())
            pred_probs = np.zeros(len(icbeb_classes))
            
            # Map each ICBEB class to corresponding class in 13-class model
            for i, icbeb_class in enumerate(icbeb_classes):
                # Get corresponding class name in 13-class model
                our_class = ICBEB_TO_OUR_MAPPING[icbeb_class]
                
                # Find corresponding class index in 13-class model
                if our_class in class_names:
                    class_idx = class_names.index(our_class)
                    if class_idx < len(probs):
                        pred_probs[i] = probs[class_idx]
                    else:
                        pred_probs[i] = 0.0
                        if not warning_shown:
                            print(f"⚠️ Class {our_class} index {class_idx} in 13-class model exceeds prediction probability range")
                else:
                    pred_probs[i] = 0.0
                    if not warning_shown:
                        print(f"⚠️ Class {our_class} does not exist in 13-class model")
            
            predictions.append(pred_probs)
            successful_predictions += 1
            
            # Debug: print mapped predictions (only in debug mode)
            if debug_mode:
                print(f"   🔍 Mapped Predictions Debug:")
                for j, (icbeb_class, prob) in enumerate(zip(icbeb_classes, pred_probs)):
                    print(f"      {icbeb_class}: {prob:.6f}")
                print(f"      Max mapped probability: {pred_probs.max():.6f}")
                print(f"      Min mapped probability: {pred_probs.min():.6f}")
            
            # Show warning message (only once)
            if not warning_shown and n_out != len(ICBEB_TO_OUR_MAPPING):
                print(f"   ⚠️  Model has {n_out} classes, mapping to {len(ICBEB_TO_OUR_MAPPING)} ICBEB classes.")
                warning_shown = True
            
        except Exception as e:
            print(f"   ❌ Failed to process {wfdb_path}: {e}")
            failed_predictions += 1
            # Add zero prediction to maintain consistent length
            predictions.append(np.zeros(len(ICBEB_TO_OUR_MAPPING)))
        
        # Reduce progress update frequency
        if (i + 1) % 200 == 0:
            print(f"   📊 Progress: {i+1}/{len(wfdb_paths)} ({(i+1)/len(wfdb_paths)*100:.1f}%) - Success: {successful_predictions}")
    
    print(f"✅ Optimized batch prediction completed: {successful_predictions}/{len(wfdb_paths)} successful")
    return np.array(predictions)

def load_icbeb_wfdb_database():
    """Load ICBEB WFDB database"""
    print("📊 Loading ICBEB WFDB database to get all available files...")
    
    # Load ICBEB database
    icbeb_db_path = "data/ICBEB/icbeb_database.csv"
    if not os.path.exists(icbeb_db_path):
        raise FileNotFoundError(f"ICBEB database not found: {icbeb_db_path}")
    
    df = pd.read_csv(icbeb_db_path)
    print(f"   ✅ Loaded {len(df)} records from ICBEB database")
    
    # Filter records with WFDB files and target classes in ICBEB mapping
    relevant_classes = set(ICBEB_TO_OUR_MAPPING.keys())
    
    # Create boolean mask: check if scp_codes column contains relevant classes
    mask = df['scp_codes'].apply(
        lambda x: any(class_name in str(x) for class_name in relevant_classes)
    )
    
    df_filtered = df[mask].copy()
    
    # Build WFDB file paths
    wfdb_paths = []
    valid_indices = []
    
    for idx, row in df_filtered.iterrows():
        # Convert filename from A0001 format to numeric format for WFDB
        filename = row['filename']  # e.g., A0001
        # Extract numeric part and convert to WFDB path
        ecg_id = row['ecg_id']  # This should be the numeric ID
        wfdb_path = os.path.join("data/ICBEB/records100", str(ecg_id))
        
        # Check if WFDB files exist (.dat and .hea files)
        dat_path = wfdb_path + ".dat"
        hea_path = wfdb_path + ".hea"
        
        if os.path.exists(dat_path) and os.path.exists(hea_path):
            wfdb_paths.append(wfdb_path)
            valid_indices.append(idx)
        else:
            print(f"   ⚠️  WFDB files not found: {dat_path} or {hea_path}")
    
    # Keep only valid records
    df_valid = df_filtered.loc[valid_indices].reset_index(drop=True)
    
    print(f"   ✅ Found {len(df_valid)} records with WFDB files and target classes")
    return df_valid, wfdb_paths

def extract_true_labels(df, wfdb_paths):
    """Extract true labels (only for predicted samples)"""
    print("🏷️  Extracting true labels for evaluation...")
    
    true_labels = []
    
    # Create ecg_id to row mapping
    ecg_id_to_row = {}
    for idx, row in df.iterrows():
        ecg_id = row['ecg_id']  # Use ecg_id as key
        ecg_id_to_row[ecg_id] = row
    
    # Only process predicted samples
    for wfdb_path in wfdb_paths:
        # Extract ecg_id from path (filename is the numeric ID)
        ecg_id = int(os.path.basename(wfdb_path))
        
        if ecg_id in ecg_id_to_row:
            row = ecg_id_to_row[ecg_id]
            # Parse scp_codes
            scp_codes_str = str(row['scp_codes'])
            
            # Create binary label vector (only for ICBEB-related classes)
            label_vector = np.zeros(len(ICBEB_TO_OUR_MAPPING))
            
            for i, icbeb_class in enumerate(ICBEB_TO_OUR_MAPPING.keys()):
                if icbeb_class in scp_codes_str:
                    label_vector[i] = 1.0
            
            true_labels.append(label_vector)
        else:
            print(f"⚠️  Warning: No database entry found for {ecg_id}")
            # If no corresponding row found, create zero vector
            true_labels.append(np.zeros(len(ICBEB_TO_OUR_MAPPING)))
    
    return np.array(true_labels)

def evaluate_performance(true_labels, predictions, use_threshold=True, threshold=0.5):
    """Evaluate performance with flexible threshold options"""
    print("📊 Evaluating performance...")
    print(f"🔧 Using {'fixed threshold' if use_threshold else 'top-k'} approach")
    
    # Check if there are prediction data
    if predictions.size == 0 or true_labels.size == 0:
        print("⚠️  No predictions or labels available for evaluation")
        return {}
    
    # Convert to binary predictions
    if use_threshold:
        binary_preds = (predictions >= threshold).astype(int)
        print(f"   📊 Fixed threshold approach: threshold = {threshold}")
    else:
        # Top-k approach: for each sample, set top-k predictions to 1
        binary_preds = np.zeros_like(predictions)
        for i in range(predictions.shape[0]):
            # Get indices of top predictions for this sample
            top_indices = np.argsort(predictions[i])[-1:]  # Top-1 for now
            binary_preds[i, top_indices] = 1
        print(f"   📊 Top-k approach: highest probability class per sample")
    
    # Calculate metrics for each class
    class_names = list(ICBEB_TO_OUR_MAPPING.keys())
    results = {}
    
    print(f"   📊 Evaluating {len(class_names)} actual classes: {class_names}")
    
    for i, class_name in enumerate(class_names):
        try:
            y_true = true_labels[:, i]
            y_pred_proba = predictions[:, i]
            y_pred_binary = binary_preds[:, i]
            
            # Skip if no positive samples in true labels
            if np.sum(y_true) == 0:
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
                auc = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                auc = 0.5  # Default AUC if calculation fails
            
            # Calculate other metrics
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            
            # Calculate specificity
            tn = np.sum((y_true == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true == 0) & (y_pred_binary == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true == 1) & (y_pred_binary == 0))
            
            results[class_name] = {
                'support': np.sum(y_true),
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

def save_results(results, true_labels, predictions, output_dir, class_names):
    """Save evaluation results"""
    print(f"💾 Saving results...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_results = []
    for class_name in class_names:
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
    predictions_df = pd.DataFrame(predictions, columns=class_names)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save true labels
    true_labels_df = pd.DataFrame(true_labels, columns=class_names)
    true_labels_df.to_csv(output_dir / "true_labels.csv", index=False)
    
    print(f"💾 Results saved to: {output_dir}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Create and save confusion matrix"""
    print(f"📊 Creating confusion matrix...")
    
    # Calculate confusion matrix for each class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            y_true_class = y_true[:, i]
            y_pred_class = (y_pred[:, i] > 0.5).astype(int)
            
            cm = confusion_matrix(y_true_class, y_pred_class)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{class_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(len(class_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Confusion matrix saved")

def plot_roc_curves(y_true, y_pred, class_names, output_dir):
    """Create and save ROC curves"""
    print(f"📊 Creating ROC curves...")
    
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        except Exception as e:
            print(f"   ⚠️  Could not plot ROC for {class_name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - ICBEB 13-Class Model')
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
    args = parse_args()
    exp_folder = Path(args.exp_folder)
    
    print(f"🚀 Starting ICBEB WFDB 13-Class Model Evaluation...")
    print(f"📁 Experiment folder: {exp_folder}")
    
    # Load model first to get model path for output directory naming
    pred_module, scaler, mlb, n_out, class_names = load_model_and_preprocessing(exp_folder)
    if pred_module is None:
        print("❌ Failed to load model, exiting...")
        return
    
    # Extract model name and create timestamped output directory with model info
    model_path = pred_module.get('model_path', 'unknown_model')
    model_name = Path(model_path).stem
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/prediction/icbeb_test/batch_icbeb_13class_results/run_{timestamp}_{model_name}"
    
    print(f"📁 Model: {model_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load ICBEB dataset
    try:
        df, wfdb_paths = load_icbeb_wfdb_database()
    except Exception as e:
        print(f"❌ Failed to load ICBEB dataset: {e}")
        return
    
    if len(wfdb_paths) == 0:
        print("❌ No valid samples found, exiting...")
        return
    
    # Extract true labels
    true_labels = extract_true_labels(df, wfdb_paths)
    
    # Run prediction
    start_time = time.time()
    print(f"\n🚀 Starting batch prediction for {len(wfdb_paths)} ICBEB samples...")
    
    # Check for shutdown request
    if _shutdown_requested:
        print("🛑 Shutdown requested before prediction")
        return
    
    predictions = predict_batch_fast(wfdb_paths, pred_module, scaler, mlb, n_out, class_names)
    
    prediction_time = time.time() - start_time
    print(f"⏱️  Prediction time: {prediction_time:.2f} seconds")
    print(f"⚡ Average time per sample: {prediction_time/len(wfdb_paths):.4f} seconds")
    
    # Evaluate performance
    results = evaluate_performance(true_labels, predictions)
    
    # Save results
    class_names_list = list(ICBEB_TO_OUR_MAPPING.keys())
    save_results(results, true_labels, predictions, output_dir, class_names_list)
    
    # Create visualizations
    plot_confusion_matrix(true_labels, predictions, class_names_list, Path(output_dir))
    plot_roc_curves(true_labels, predictions, class_names_list, Path(output_dir))
    
    # Final performance statistics
    total_time = time.time() - start_time
    avg_samples_per_second = len(wfdb_paths) / total_time if total_time > 0 else 0
    
    print(f"\n🎉 ICBEB 13-class model evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total samples evaluated: {len(wfdb_paths):,}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"⚡ Average processing speed: {avg_samples_per_second:.1f} samples/second")
    print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()