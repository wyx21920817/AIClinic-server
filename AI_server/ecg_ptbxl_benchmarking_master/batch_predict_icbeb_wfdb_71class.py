#!/usr/bin/env python3
"""
Ultra-Fast Batch ICBEB WFDB Prediction Script for 71-Class Models
ICBEB evaluation script using original WFDB format for 71-class models

Usage:
    python batch_predict_icbeb_wfdb_71class.py [exp_folder]
    
Examples:
    python batch_predict_icbeb_wfdb_71class.py                           # Use default model
    python batch_predict_icbeb_wfdb_71class.py output/exp0              # Use specified experiment folder
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
    'VPC': 'PVC'  # Display as PVC instead of VPC
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
        description="Ultra-Fast Batch ICBEB WFDB Prediction Script for 71-Class Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_predict_icbeb_wfdb_71class.py                           # Use default model
    python batch_predict_icbeb_wfdb_71class.py output/exp0              # Use specified experiment folder
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
        
        # Verify ICBEB mapped classes exist in 71-class model
        print("   🔍 Verifying ICBEB class mappings:")
        for icbeb_class, our_class in ICBEB_TO_OUR_MAPPING.items():
            if our_class in MODEL_71_CLASSES:
                idx = MODEL_71_CLASSES.index(our_class)
                print(f"      {icbeb_class:6s} -> {our_class:6s} (index: {idx:2d}) ✅")
            else:
                print(f"      {icbeb_class:6s} -> {our_class:6s} (not found) ❌")
        
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

def predict_batch_fast(wfdb_paths, pred_module, scaler, mlb, n_out, debug_mode=False):
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
            
            # Map 71-class predictions to ICBEB-related 7 classes
            # Create mapped prediction probability array (only for classes present in ICBEB)
            icbeb_classes = list(ICBEB_TO_OUR_MAPPING.keys())
            pred_probs = np.zeros(len(icbeb_classes))
            
            # Map each ICBEB class to corresponding class in 71-class model
            for i, icbeb_class in enumerate(icbeb_classes):
                # Get corresponding class name in 71-class model
                our_class = ICBEB_TO_OUR_MAPPING[icbeb_class]
                
                # Find corresponding class index in 71-class model
                if our_class in MODEL_71_CLASSES:
                    class_idx = MODEL_71_CLASSES.index(our_class)
                    if class_idx < len(probs):
                        pred_probs[i] = probs[class_idx]
                    else:
                        pred_probs[i] = 0.0
                        if not warning_shown:
                            print(f"⚠️ Class {our_class} index {class_idx} in 71-class model exceeds prediction probability range")
                else:
                    pred_probs[i] = 0.0
                    if not warning_shown:
                        print(f"⚠️ Class {our_class} does not exist in 71-class model")
            
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
        binary_predictions = (predictions > threshold).astype(float)
        print(f"   📊 Threshold: {threshold}")
    else:
        # Top-k approach: for each sample, select the class with highest probability
        binary_predictions = np.zeros_like(predictions)
        for i in range(predictions.shape[0]):
            max_idx = np.argmax(predictions[i])
            binary_predictions[i, max_idx] = 1.0
        print("   📊 Top-1 approach: highest probability class per sample")
    
    results = {}
    
    # ICBEB class names that actually exist (only evaluate these 7 classes)
    icbeb_classes = list(ICBEB_TO_OUR_MAPPING.keys())
    
    for i, class_name in enumerate(icbeb_classes):
        if i < predictions.shape[1]:
            true_class = true_labels[:, i]
            pred_class = predictions[:, i]
            binary_pred_class = binary_predictions[:, i]
            
            # Calculate support
            support = int(np.sum(true_class))
            
            if support > 0:
                # Calculate AUC
                try:
                    auc = roc_auc_score(true_class, pred_class)
                except ValueError:
                    auc = 0.0
                
                # Calculate other metrics
                if np.sum(binary_pred_class) > 0:
                    precision = precision_score(true_class, binary_pred_class, zero_division=0)
                    recall = recall_score(true_class, binary_pred_class, zero_division=0)
                    f1 = f1_score(true_class, binary_pred_class, zero_division=0)
                else:
                    precision = recall = f1 = 0.0
                
                # Calculate specificity
                tn = np.sum((true_class == 0) & (binary_pred_class == 0))
                fp = np.sum((true_class == 0) & (binary_pred_class == 1))
                fn = np.sum((true_class == 1) & (binary_pred_class == 0))
                tp = np.sum((true_class == 1) & (binary_pred_class == 1))
                
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                results[class_name] = {
                    'support': support,
                    'auc': auc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                }
            else:
                results[class_name] = {
                    'support': 0,
                    'auc': 0.0, 'f1': 0.0, 'precision': 0.0,
                    'recall': 0.0, 'specificity': 0.0,
                    'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
                }
    
    return results

def plot_confusion_matrix(true_labels, predictions, class_names, output_dir):
    """Plot confusion matrix"""
    print("📊 Creating confusion matrix...")
    
    # Convert to binary predictions
    binary_predictions = (predictions > 0.5).astype(float)
    
    # Calculate confusion matrix for each class
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i < len(axes) and i < predictions.shape[1]:
            true_class = true_labels[:, i]
            pred_class = binary_predictions[:, i]
            
            cm = confusion_matrix(true_class, pred_class, labels=[0, 1])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[i].set_title(f'{class_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(len(class_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(true_labels, predictions, class_names, output_dir):
    """Plot ROC curves"""
    print("📊 Creating ROC curves...")
    
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        if i < predictions.shape[1]:
            true_class = true_labels[:, i]
            pred_class = predictions[:, i]
            
            if np.sum(true_class) > 0:  # Only plot if positive samples exist
                fpr, tpr, _ = roc_curve(true_class, pred_class)
                auc = roc_auc_score(true_class, pred_class)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for ICBEB Classes (WFDB Format)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, true_labels, predictions, output_dir):
    """Save results"""
    print("💾 Saving results...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df.to_csv(output_dir / 'detailed_results.csv')
    
    # Save prediction results (only classes present in ICBEB)
    icbeb_classes = list(ICBEB_TO_OUR_MAPPING.keys())
    predictions_df = pd.DataFrame(predictions, columns=icbeb_classes)
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save true labels
    true_labels_df = pd.DataFrame(true_labels, columns=icbeb_classes)
    true_labels_df.to_csv(output_dir / 'true_labels.csv', index=False)
    
    # Create summary report
    summary_data = []
    for class_name, metrics in results.items():
        # Use display name mapping
        display_name = DISPLAY_NAME_MAPPING.get(class_name, class_name)
        summary_data.append([
            display_name,
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
    
    summary_df.to_csv(output_dir / 'summary_results.csv', index=False)
    
    # Print summary results
    print("\n📊 71-Class Model ICBEB WFDB Evaluation Results:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Create visualizations
    # Only evaluate ICBEB-related classes
    icbeb_classes = list(ICBEB_TO_OUR_MAPPING.keys())
    plot_confusion_matrix(true_labels, predictions, icbeb_classes, output_dir)
    plot_roc_curves(true_labels, predictions, icbeb_classes, output_dir)
    
    print(f"\n💾 Results saved to: {output_dir}")

def main():
    """Main function"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    exp_folder = args.exp_folder
    
    # Debug mode - only process first 5 samples
    DEBUG_MODE = False
    if DEBUG_MODE:
        print("🐛 DEBUG MODE: Processing only first 5 samples")
    
    # Evaluation method: True for threshold-based, False for top-k
    USE_THRESHOLD = False  # Set to False to use top-k approach (recommended for ICBEB)
    THRESHOLD = 0.5      # Threshold for binary classification (not used when USE_THRESHOLD=False)
    
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
    global OUTPUT_DIR
    OUTPUT_DIR = Path(f"output/prediction/icbeb_test/batch_icbeb_wfdb_71class_results/run_{timestamp}_{model_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Ultra-Fast Batch ICBEB WFDB Prediction for 71-Class Models")
    print(f"📁 Experiment folder: {exp_folder}")
    print(f"📁 Model: {model_name}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Load ICBEB WFDB database
    df, wfdb_paths = load_icbeb_wfdb_database()
    
    if len(wfdb_paths) == 0:
        print("❌ No valid WFDB files found. Exiting...")
        return
    
    # Debug mode: limit to first 5 samples
    if DEBUG_MODE:
        wfdb_paths = wfdb_paths[:5]
        df = df.head(5)
        print(f"🐛 DEBUG: Limited to {len(wfdb_paths)} samples")
    
    # Extract true labels
    true_labels = extract_true_labels(df, wfdb_paths)
    
    # Batch prediction
    predictions = predict_batch_fast(wfdb_paths, pred_module, scaler, mlb, n_out, DEBUG_MODE)
    
    # Check if program was interrupted
    if _shutdown_requested:
        print("\n🛑 Program interrupted by user, exiting...")
        return
    
    # Evaluate performance
    results = evaluate_performance(true_labels, predictions, USE_THRESHOLD, THRESHOLD)
    
    if not results:
        print("❌ No results to save. Exiting...")
        return
    
    # Save results
    save_results(results, true_labels, predictions, OUTPUT_DIR)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"⚡ Average time per prediction: {execution_time / len(wfdb_paths):.2f} seconds")
    print("✅ Ultra-fast batch WFDB prediction completed!")

if __name__ == "__main__":
    main()
