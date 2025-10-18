#!/usr/bin/env python3
"""
Clinical ECG Dataset Batch Prediction Script for 13-Class Models
Script for evaluating clinical ECG dataset using 13-class models for prediction

Usage:
    python batch_predict_clinical_13class.py [exp_folder]
    
Examples:
    python batch_predict_clinical_13class.py                                    # Use default model
    python batch_predict_clinical_13class.py output/exp_13class_smart_multilabel  # Use specified experiment folder
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

# Clinical dataset to our 13-class model mapping
# Clinical dataset category to 13-class model mapping
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
        description="Clinical ECG Dataset Batch Prediction Script for 13-Class Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_predict_clinical_13class.py                                    # Use default model
    python batch_predict_clinical_13class.py output/exp_13class_smart_multilabel  # Use specified experiment folder
        """,
        add_help=True
    )
    
    parser.add_argument(
        'exp_folder', 
        nargs='?',  # Optional positional argument
        default="output/exp_13class_smart_multilabel",  # Default use 13-class model
        help='Experiment folder path containing models/ and data/ directories (default: output/exp_13class_smart_multilabel)'
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
        
        # Verify clinical mapped classes exist in 13-class model
        print("   🔍 Verifying clinical class mappings:")
        for clinical_class, our_class in CLINICAL_TO_OUR_MAPPING.items():
            if our_class in class_names:
                idx = class_names.index(our_class)
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

def load_clinical_dataset():
    """Load clinical dataset"""
    clinical_data_path = "data/clinical/clinical_dataset"
    if not os.path.exists(clinical_data_path):
        raise FileNotFoundError(f"Clinical dataset not found: {clinical_data_path}")
    
    print(f"📊 Loading clinical dataset...")
    
    # Load clinical data
    X = np.load(os.path.join(clinical_data_path, "X_clinical_data_12.npy"), allow_pickle=True)
    y = np.load(os.path.join(clinical_data_path, "y_clinical_data_12.npy"), allow_pickle=True)
    
    # Load clinical MLB
    with open(os.path.join(clinical_data_path, "mlb12.pkl"), 'rb') as f:
        mlb_clinical = pickle.load(f)
    
    print(f"   ✅ Loaded clinical dataset:")
    print(f"      X shape: {X.shape}")
    print(f"      y shape: {y.shape}")
    print(f"      Classes: {len(mlb_clinical.classes_)}")
    print(f"      Class names: {list(mlb_clinical.classes_)}")
    
    return X, y, mlb_clinical

def extract_target_classes(X_data, y_data, mlb_clinical, class_names):
    """Extract only target ECG classes from clinical dataset"""
    print("🎯 Extracting target ECG classes from clinical dataset...")
    
    # Get clinical class names
    clinical_classes = list(mlb_clinical.classes_)
    
    # Create a mapping from clinical class to index
    clinical_to_index = {cls: i for i, cls in enumerate(clinical_classes)}
    
    # Find which target ECG classes have corresponding clinical classes
    target_indices = []
    target_class_names = []
    
    # Iterate through CLINICAL_TO_OUR_MAPPING to find available classes
    for clinical_class, mapped_ecg_class in CLINICAL_TO_OUR_MAPPING.items():
        if clinical_class in clinical_to_index and mapped_ecg_class in class_names:
            target_indices.append(clinical_to_index[clinical_class])
            target_class_names.append(mapped_ecg_class)
            print(f"      {clinical_class} -> {mapped_ecg_class}")
    
    print(f"   📋 Found {len(target_indices)} clinical classes mapping to target ECG classes:")
    
    # Extract corresponding labels
    y_target = y_data[:, target_indices]
    
    print(f"   ✅ Extracted {y_target.shape[1]} target ECG classes from clinical dataset")
    print(f"   📊 Target classes: {target_class_names}")
    
    return X_data, y_target, target_class_names

def predict_batch_clinical_optimized(X_data, pred_module, scaler, mlb, n_out, class_names, actual_classes=None, debug_mode=False):
    """Fast batch prediction for clinical dataset"""
    print(f"🚀 Starting optimized batch prediction for {len(X_data)} clinical samples...")
    
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
            # First transpose to match format: (channels, timesteps)
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
            
            # Map 13-class predictions to clinical-related classes
            # Create mapped prediction probability array (only for classes present in clinical)
            clinical_classes = list(CLINICAL_TO_OUR_MAPPING.values())
            pred_probs = np.zeros(len(clinical_classes))
            
            # Map each clinical class to corresponding class in 13-class model
            for i, clinical_class in enumerate(clinical_classes):
                # Find corresponding class index in 13-class model
                if clinical_class in class_names:
                    class_idx = class_names.index(clinical_class)
                    if class_idx < len(probs):
                        pred_probs[i] = probs[class_idx]
                    else:
                        pred_probs[i] = 0.0
                        if not warning_shown:
                            print(f"⚠️ Class {clinical_class} index {class_idx} in 13-class model exceeds prediction probability range")
                else:
                    pred_probs[i] = 0.0
                    if not warning_shown:
                        print(f"⚠️ Class {clinical_class} does not exist in 13-class model")
            
            predictions.append(pred_probs)
            successful_predictions += 1
            
            # Debug: print mapped predictions (only in debug mode)
            if debug_mode:
                print(f"   🔍 Mapped Predictions Debug:")
                for j, (clinical_class, prob) in enumerate(zip(clinical_classes, pred_probs)):
                    print(f"      {clinical_class}: {prob:.6f}")
                print(f"      Max mapped probability: {pred_probs.max():.6f}")
                print(f"      Min mapped probability: {pred_probs.min():.6f}")
            
            # Show warning message (only once)
            if not warning_shown and n_out != len(clinical_classes):
                print(f"   ⚠️  Model has {n_out} classes, mapping to {len(clinical_classes)} clinical classes.")
                warning_shown = True
            
        except Exception as e:
            print(f"   ❌ Failed to process sample {i}: {e}")
            failed_predictions += 1
            # Add zero prediction to maintain consistent length
            predictions.append(np.zeros(len(CLINICAL_TO_OUR_MAPPING)))
        
        # Reduce progress update frequency
        if (i + 1) % 200 == 0:
            print(f"   📊 Progress: {i+1}/{len(X_data)} ({(i+1)/len(X_data)*100:.1f}%) - Success: {successful_predictions}")
    
    print(f"✅ Batch prediction completed: {successful_predictions}/{len(X_data)} successful")
    print(f"   📊 Predictions shape: ({successful_predictions}, {len(CLINICAL_TO_OUR_MAPPING)})")
    
    return np.array(predictions)

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
    plt.title('ROC Curves - Clinical 13-Class Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ ROC curves saved")

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    exp_folder = Path(args.exp_folder)
    
    print(f"🚀 Starting Clinical Dataset Batch Prediction for 13-Class Models")
    print(f"📁 Experiment folder: {exp_folder}")
    
    # Load model first to get model path for output directory naming
    pred_module, scaler, mlb, n_out, class_names = load_model_and_preprocessing(exp_folder)
    if pred_module is None:
        print("❌ Failed to load model, exiting...")
        return
    
    # Extract model name and create timestamped output directory with model info
    model_path = pred_module.get('model_path', 'unknown_model')
    model_name = extract_model_name_from_path(model_path)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/prediction/clinical_test/batch_clinical_13class_results/run_{timestamp}_{model_name}"
    
    print(f"📁 Model: {model_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load clinical dataset
    try:
        X, y, mlb_clinical = load_clinical_dataset()
    except Exception as e:
        print(f"❌ Failed to load clinical dataset: {e}")
        return
    
    # Extract target classes
    X_target, y_target, actual_classes = extract_target_classes(X, y, mlb_clinical, class_names)
    
    if len(X_target) == 0:
        print("❌ No valid samples found, exiting...")
        return
    
    # Run prediction
    start_time = time.time()
    print(f"\n🚀 Starting batch prediction for {len(X_target)} clinical samples...")
    
    # Check for shutdown request
    if _shutdown_requested:
        print("🛑 Shutdown requested before prediction")
        return
    
    predictions = predict_batch_clinical_optimized(X_target, pred_module, scaler, mlb, n_out, class_names, actual_classes)
    
    prediction_time = time.time() - start_time
    print(f"⏱️  Prediction time: {prediction_time:.2f} seconds")
    print(f"⚡ Average time per sample: {prediction_time/len(X_target):.4f} seconds")
    
    # Evaluate performance
    results = evaluate_performance(y_target, predictions, actual_classes)
    
    # Save results
    save_results(results, y_target, predictions, output_dir, actual_classes)
    
    # Create visualizations
    plot_confusion_matrix(y_target, predictions, actual_classes, Path(output_dir))
    plot_roc_curves(y_target, predictions, actual_classes, Path(output_dir))
    
    # Final performance statistics
    total_time = time.time() - start_time
    avg_samples_per_second = len(X_target) / total_time if total_time > 0 else 0
    
    print(f"\n🎉 Clinical dataset batch prediction completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total samples evaluated: {len(X_target):,}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"⚡ Average processing speed: {avg_samples_per_second:.1f} samples/second")
    print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
