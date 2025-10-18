#!/usr/bin/env python3
"""
Clinical ECG Dataset Batch Prediction Script for 71-Class Models
Script for evaluating clinical ECG dataset using 71-class models for prediction

Usage:
    python batch_predict_clinical_71class.py [exp_folder]
    
Examples:
    python batch_predict_clinical_71class.py                           # Use default model
    python batch_predict_clinical_71class.py output/exp0              # Use specified experiment folder
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
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

# Clinical dataset to our 71-class model mapping
# Clinical dataset category to 71-class model mapping - only keep 13 target categories
CLINICAL_TO_OUR_MAPPING = {
    'I°房室传导阻滞': '1AVB',      # First degree AV block -> 1AVB
    '完全性右束支传导阻滞': 'CRBBB',  # Complete RBBB -> CRBBB
    '室性早搏': 'PVC',             # Ventricular premature contraction -> PVC
    '心房扑动': 'AFLT',            # Atrial flutter -> AFLT
    '心房颤动': 'AFIB',            # Atrial fibrillation -> AFIB
    '房性早搏': 'PAC',             # Premature atrial contraction -> PAC
    '正常心电图': 'NORM',          # Normal ECG -> NORM
    '窦性心动过速': 'STACH'         # Sinus tachycardia -> STACH
}

# Target ECG class list (13 classes)
TARGET_ECG_CLASSES = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 
                      'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']

# Display name mapping (for output display) - use English ECG terms
DISPLAY_NAME_MAPPING = {
    'I°房室传导阻滞': '1AVB',
    '完全性右束支传导阻滞': 'CRBBB',
    '室性早搏': 'PVC',
    '心房扑动': 'AFLT',
    '心房颤动': 'AFIB',
    '房性早搏': 'PAC',
    '正常心电图': 'NORM',
    '窦性心动过速': 'STACH'
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
        description="Clinical ECG Dataset Batch Prediction Script for 71-Class Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_predict_clinical_71class.py                           # Use default model
    python batch_predict_clinical_71class.py output/exp0              # Use specified experiment folder
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
        
        # Verify clinical mapped classes exist in 71-class model
        print("   🔍 Verifying clinical class mappings:")
        for clinical_class, our_class in CLINICAL_TO_OUR_MAPPING.items():
            if our_class in MODEL_71_CLASSES:
                idx = MODEL_71_CLASSES.index(our_class)
                print(f"      {clinical_class:20s} -> {our_class:6s} (index: {idx:2d}) ✅")
            else:
                print(f"      {clinical_class:20s} -> {our_class:6s} (not found) ❌")
        
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

def load_clinical_dataset():
    """Load clinical dataset"""
    print("📊 Loading clinical dataset...")
    
    # Load clinical dataset
    clinical_data_path = "data/clinical/clinical_dataset"
    if not os.path.exists(clinical_data_path):
        raise FileNotFoundError(f"Clinical dataset not found: {clinical_data_path}")
    
    # Load data
    X = np.load(os.path.join(clinical_data_path, "X_clinical_data_12.npy"), allow_pickle=True)
    y = np.load(os.path.join(clinical_data_path, "y_clinical_data_12.npy"), allow_pickle=True)
    
    with open(os.path.join(clinical_data_path, "mlb12.pkl"), 'rb') as f:
        mlb_clinical = pickle.load(f)
    
    print(f"   ✅ Loaded clinical dataset:")
    print(f"      X shape: {X.shape}")
    print(f"      y shape: {y.shape}")
    print(f"      Classes: {len(mlb_clinical.classes_)}")
    print(f"      Class names: {list(mlb_clinical.classes_)}")
    
    return X, y, mlb_clinical

def extract_target_classes(X_data, y_data, mlb_clinical):
    """Extract only target ECG classes from clinical dataset"""
    print("🎯 Extracting target ECG classes from clinical dataset...")
    
    # Get clinical class names
    clinical_classes = list(mlb_clinical.classes_)
    
    # Create a mapping from clinical class to index
    clinical_to_index = {cls: i for i, cls in enumerate(clinical_classes)}
    
    # Find which target ECG classes have corresponding clinical classes
    target_indices = []
    target_class_names = []
    
    # Iterate through TARGET_ECG_CLASSES in order to maintain consistent ordering
    for target_ecg_class in TARGET_ECG_CLASSES:
        # Find the clinical class that maps to this target ECG class
        for clinical_class, mapped_ecg_class in CLINICAL_TO_OUR_MAPPING.items():
            if mapped_ecg_class == target_ecg_class and clinical_class in clinical_to_index:
                target_indices.append(clinical_to_index[clinical_class])
                target_class_names.append(target_ecg_class)
                print(f"      {clinical_class} -> {target_ecg_class}")
                break
    
    print(f"   📋 Found {len(target_indices)} clinical classes mapping to target ECG classes:")
    
    # Extract corresponding labels
    y_target = y_data[:, target_indices]
    
    print(f"   ✅ Extracted {y_target.shape[1]} target ECG classes from clinical dataset")
    print(f"   📊 Target classes: {target_class_names}")
    
    return X_data, y_target, target_class_names

def predict_batch_clinical(X_data, pred_module, scaler, mlb, n_out, actual_classes=None, debug_mode=False):
    """Fast batch prediction for clinical dataset"""
    print(f"🚀 Starting batch prediction for {len(X_data)} clinical samples...")
    
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
            # Preprocessing - use same method as 71-class model training (consistent with ICBEB)
            # First transpose to match ICBEB format: (channels, timesteps)
            ecg_data = ecg_data.T  # Transpose from (1000, 12) to (12, 1000) - (channels, timesteps)
            
            # Flatten ECG data to 1D, apply scaler, then reshape back to original shape
            x_shape = ecg_data.shape
            x_flat = ecg_data.flatten()[:, np.newaxis]  # Flatten to column vector
            x_scaled = scaler.transform(x_flat).reshape(x_shape)  # Apply scaler and reshape
            
            x = torch.from_numpy(x_scaled).unsqueeze(0).float()  # Add batch dimension and convert to float -> (1, 12, 1000)
            
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
            
            # Map 71-class predictions to actual ECG classes only
            # Use actual_classes if provided, otherwise use all target classes
            classes_to_predict = actual_classes if actual_classes is not None else TARGET_ECG_CLASSES
            
            # Create mapped prediction probability array for actual classes
            pred_probs = np.zeros(len(classes_to_predict))
            
            # Map each actual ECG class to corresponding class in 71-class model
            for j, target_class in enumerate(classes_to_predict):
                # Find corresponding class index in 71-class model
                if target_class in MODEL_71_CLASSES:
                    class_idx = MODEL_71_CLASSES.index(target_class)
                    if class_idx < len(probs):
                        pred_probs[j] = probs[class_idx]
                    else:
                        pred_probs[j] = 0.0
                        if not warning_shown:
                            print(f"⚠️ Class {target_class} index {class_idx} in 71-class model exceeds prediction probability range")
                else:
                    pred_probs[j] = 0.0
                    if not warning_shown:
                        print(f"⚠️ Class {target_class} does not exist in 71-class model")
            
            predictions.append(pred_probs)
            successful_predictions += 1
            
            # Debug: print mapped predictions (only in debug mode)
            if debug_mode:
                print(f"   🔍 Mapped Predictions Debug:")
                for j, (target_class, prob) in enumerate(zip(classes_to_predict, pred_probs)):
                    print(f"      {target_class}: {prob:.6f}")
                print(f"      Max mapped probability: {pred_probs.max():.6f}")
                print(f"      Min mapped probability: {pred_probs.min():.6f}")
            
            # Show warning message (only once)
            if not warning_shown and n_out != len(classes_to_predict):
                print(f"   ⚠️  Model has {n_out} classes, mapping to {len(classes_to_predict)} actual ECG classes.")
                warning_shown = True
            
        except Exception as e:
            print(f"   ❌ Failed to process sample {i}: {e}")
            failed_predictions += 1
            # Add zero prediction to maintain consistent length
            classes_to_predict = actual_classes if actual_classes is not None else TARGET_ECG_CLASSES
            predictions.append(np.zeros(len(classes_to_predict)))
        
        # Reduce progress update frequency
        if (i + 1) % 200 == 0:
            print(f"   📊 Progress: {i+1}/{len(X_data)} ({(i+1)/len(X_data)*100:.1f}%) - Success: {successful_predictions}")
    
    print(f"✅ Batch prediction completed: {successful_predictions}/{len(X_data)} successful")
    return np.array(predictions)

def evaluate_performance(true_labels, predictions, use_threshold=True, threshold=0.5, actual_classes=None):
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
    
    # Get actual classes from data (not all target classes may be present)
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES[:true_labels.shape[1]]
    print(f"   📊 Evaluating {len(actual_classes)} actual classes: {actual_classes}")
    
    for i, class_name in enumerate(actual_classes):
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
                
                # Use target ECG class name as key
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
                # Use target ECG class name as key
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
    n_classes = len(class_names)
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i < predictions.shape[1]:
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
    plt.title('ROC Curves for Clinical Dataset Classes')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, true_labels, predictions, output_dir, actual_classes=None):
    """Save results"""
    print("💾 Saving results...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get actual classes from data (not all target classes may be present)
    if actual_classes is None:
        actual_classes = TARGET_ECG_CLASSES[:true_labels.shape[1]]
    
    # Save detailed results using pandas with proper index handling for Excel compatibility
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df.index.name = 'Class'  # Set index name for proper CSV header
    results_df.to_csv(output_dir / 'detailed_results.csv')
    
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
    
    # Save summary results using pandas (same as ICBEB script)
    summary_df.to_csv(output_dir / 'summary_results.csv', index=False)
    
    # Print summary results
    print("\n📊 71-Class Model Clinical Dataset Evaluation Results:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Create visualizations using actual ECG classes
    plot_confusion_matrix(true_labels, predictions, actual_classes, output_dir)
    plot_roc_curves(true_labels, predictions, actual_classes, output_dir)
    
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
    USE_THRESHOLD = False  # Set to False to use top-k approach (recommended for clinical dataset)
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
    OUTPUT_DIR = Path(f"output/prediction/clinical_test/batch_clinical_71class_results/run_{timestamp}_{model_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Clinical Dataset Batch Prediction for 71-Class Models")
    print(f"📁 Experiment folder: {exp_folder}")
    print(f"📁 Model: {model_name}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Load clinical dataset
    X_data, y_data, mlb_clinical = load_clinical_dataset()
    
    if len(X_data) == 0:
        print("❌ No valid data found. Exiting...")
        return
    
    # Extract only target ECG classes from clinical dataset
    X_data, y_data, extracted_class_names = extract_target_classes(X_data, y_data, mlb_clinical)
    
    # Use the actual extracted class names
    actual_classes = extracted_class_names
    
    # Debug mode: limit to first 5 samples
    if DEBUG_MODE:
        X_data = X_data[:5]
        y_data = y_data[:5]
        print(f"🐛 DEBUG: Limited to {len(X_data)} samples")
    
    # Batch prediction with actual classes
    predictions = predict_batch_clinical(X_data, pred_module, scaler, mlb, n_out, actual_classes, DEBUG_MODE)
    
    # Check if program was interrupted
    if _shutdown_requested:
        print("\n🛑 Program interrupted by user, exiting...")
        return
    
    # Evaluate performance
    results = evaluate_performance(y_data, predictions, USE_THRESHOLD, THRESHOLD, actual_classes)
    
    if not results:
        print("❌ No results to save. Exiting...")
        return
    
    # Save results
    save_results(results, y_data, predictions, OUTPUT_DIR, actual_classes)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"⚡ Average time per prediction: {execution_time / len(X_data):.2f} seconds")
    print("✅ Clinical dataset batch prediction completed!")

if __name__ == "__main__":
    main()
