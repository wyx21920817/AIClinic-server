#!/usr/bin/env python3
"""
PTB-XL ECG Dataset Batch Prediction Script for 71-Class Models
Script for evaluating PTB-XL ECG dataset using 71-class models for prediction

Usage:
    python batch_predict_ptbxl_71class.py [exp_folder]
    
Examples:
    python batch_predict_ptbxl_71class.py                           # Use default model
    python batch_predict_ptbxl_71class.py output/exp0              # Use specified experiment folder
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

# Target ECG class list (13 classes)
TARGET_ECG_CLASSES = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 
                      'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']

# PTB-XL category to our target category mapping
PTBXL_TO_OUR_MAPPING = {
    '1AVB': '1AVB',
    '2AVB': '2AVB', 
    '3AVB': '3AVB',
    'AFIB': 'AFIB',
    'AFLT': 'AFLT',
    'ILBBB': 'ILBBB',
    'CLBBB': 'CLBBB',
    'IRBBB': 'IRBBB',
    'CRBBB': 'CRBBB',
    'STACH': 'STACH',
    'PAC': 'PAC',
    'PVC': 'PVC',
    'NORM': 'NORM'
}

# Global variable for shutdown request
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n🛑 Received interrupt signal. Will finish current batch and exit gracefully...")

def load_model_and_preprocessing(exp_folder):
    """Load model and preprocessing components"""
    print(f"🔧 Loading model and preprocessing components from: {exp_folder}")
    
    # Find model file
    models_dir = Path(exp_folder) / "models"
    data_dir = Path(exp_folder) / "data"
    
    print(f"   📁 Models directory: {models_dir}")
    print(f"   📁 Data directory: {data_dir}")
    
    # Find model file - check multiple possible locations
    model_files = []
    
    # Check direct directory
    model_files = list(models_dir.glob("*.pth"))
    if model_files:
        print(f"   📁 Found model in direct directory: {models_dir}")
    else:
        # Check models subdirectory
        models_subdir = models_dir / "models"
        if models_subdir.exists():
            model_files = list(models_subdir.glob("*.pth"))
            if model_files:
                print(f"   📁 Found model in models subdirectory: {models_subdir}")
        
        # Check fastai_xresnet1d101 subdirectory
        if not model_files:
            fastai_subdir = models_dir / "fastai_xresnet1d101"
            if fastai_subdir.exists():
                model_files = list(fastai_subdir.glob("*.pth"))
                if model_files:
                    print(f"   📁 Found model in fastai_xresnet1d101 directory: {fastai_subdir}")
                
                # Check nested models subdirectory
                if not model_files:
                    nested_models = fastai_subdir / "models"
                    if nested_models.exists():
                        model_files = list(nested_models.glob("*.pth"))
                        if model_files:
                            print(f"   📁 Found model in nested models directory: {nested_models}")
    
    if not model_files:
        raise FileNotFoundError(f"No .pth model file found in {models_dir} or its subdirectories")
    
    model_path = model_files[0]
    print(f"   📁 Using model: {model_path}")
    
    # Load preprocessing components
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
    print(f"   📋 Model has {len(mlb.classes_)} classes")
    
    return model_path, scaler, mlb

def load_ptbxl_test_data(exp_folder):
    """Load PTB-XL test data directly from raw data"""
    print("📊 Loading PTB-XL test data from raw dataset...")
    
    ptbxl_data_dir = Path("data/ptbxl")
    
    # Load raw ECG data from PTB-XL directory
    raw_data_path = ptbxl_data_dir / "raw100.npy"
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw ECG data not found: {raw_data_path}")
    
    # Load PTB-XL database and labels
    db_path = ptbxl_data_dir / "ptbxl_database.csv"
    labels_path = ptbxl_data_dir / "ptbxl_database.csv"
    
    if not db_path.exists():
        raise FileNotFoundError(f"PTB-XL database not found: {db_path}")
    
    import pandas as pd
    
    # Load data
    X_raw = np.load(raw_data_path, allow_pickle=True)
    df = pd.read_csv(db_path)
    
    print(f"   ✅ Loaded raw ECG data: {X_raw.shape}")
    print(f"   ✅ Loaded PTB-XL database: {len(df)} records")
    
    # Use PTB-XL standard fold 10 split for test set
    test_indices = df[df['strat_fold'] == 10].index.values
    print(f"   ✅ Using PTB-XL standard fold 10 split ({len(test_indices)} samples)")
    
    # Extract test set features
    X_test = X_raw[test_indices]
    
    # Generate labels from PTB-XL database
    # PTB-XL uses diagnostic classification (scp_codes column)
    test_df = df.iloc[test_indices]
    
    # Load the MultiLabelBinarizer from experiment to get class mapping
    data_dir = Path(exp_folder) / "data"
    mlb_path = data_dir / "mlb.pkl"
    if not mlb_path.exists():
        raise FileNotFoundError(f"MultiLabelBinarizer not found: {mlb_path}")
    
    import pickle
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    
    # Extract SCP codes from test samples
    scp_codes_list = []
    for idx in test_indices:
        row = df.iloc[idx]
        # Parse scp_codes (format: "['code1', 'code2', ...]")
        scp_codes_str = row['scp_codes']
        if pd.isna(scp_codes_str) or scp_codes_str == '[]':
            scp_codes = []
        else:
            # Parse the string representation of list
            import ast
            try:
                scp_codes = ast.literal_eval(scp_codes_str)
            except:
                scp_codes = []
        scp_codes_list.append(scp_codes)
    
    # Transform to binary labels using the same MLB as the model
    y_test = mlb.transform(scp_codes_list)
    
    print(f"   ✅ Generated test labels from PTB-XL database: {y_test.shape}")
    print(f"   ✅ Extracted test features: {X_test.shape}")
    
    return X_test, y_test

def predict_batch_ptbxl_optimized(X_data, model, scaler, device, batch_size=1024):
    """Optimized batch prediction for PTB-XL dataset"""
    print(f"🚀 Starting batch prediction for {len(X_data)} PTB-XL samples...")
    
    model.eval()
    predictions = []
    
    # Process in batches
    for i in range(0, len(X_data), batch_size):
        batch_end = min(i + batch_size, len(X_data))
        batch_data = X_data[i:batch_end]
        
        # Convert to correct format: (batch_size, 12, 1000)
        batch_transposed = batch_data.transpose(0, 2, 1)  # (batch_size, 1000, 12) -> (batch_size, 12, 1000)
        
        # Vectorized scaling: flatten, scale, reshape back
        batch_shape = batch_transposed.shape
        batch_flat = batch_transposed.reshape(-1, 1)  # Flatten to column vector
        
        # Apply scaling
        batch_scaled_flat = scaler.transform(batch_flat)
        
        # Reshape back to original shape
        batch_scaled = batch_scaled_flat.reshape(batch_shape)
        
        # Convert to tensor
        batch_tensor = torch.from_numpy(batch_scaled).float()
        batch_tensor = batch_tensor.to(device)
        
        # Make predictions
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits)
            batch_preds = probs.cpu().numpy()
        
        predictions.extend(batch_preds)
        
        # Progress update
        if (i // batch_size + 1) % 10 == 0:
            print(f"   📊 Processed {batch_end}/{len(X_data)} samples")
    
    return np.array(predictions)

def extract_target_classes(y_true, y_pred, mlb, target_classes):
    """Extract only target classes from PTB-XL data"""
    print(f"🎯 Extracting target classes: {target_classes}")
    
    # Get indices of target classes in PTB-XL
    target_indices = []
    target_names = []
    
    for target_class in target_classes:
        if target_class in mlb.classes_:
            idx = list(mlb.classes_).index(target_class)
            target_indices.append(idx)
            target_names.append(target_class)
            print(f"   ✅ Found {target_class} at index {idx}")
        else:
            print(f"   ⚠️ Warning: {target_class} not found in PTB-XL classes")
    
    if not target_indices:
        raise ValueError("No target classes found in PTB-XL data")
    
    # Extract target classes
    y_true_target = y_true[:, target_indices]
    y_pred_target = y_pred[:, target_indices]
    
    print(f"   📊 Extracted {len(target_names)} target classes: {target_names}")
    print(f"   📊 Target data shape: {y_true_target.shape}")
    
    return y_true_target, y_pred_target, target_names

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
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
            
            # Calculate confusion matrix components
            tp = np.sum((y_true_class == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true_class == 1) & (y_pred_binary == 0))
            
            results[class_name] = {
                'support': int(np.sum(y_true_class)),
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
            }
            
            print(f"   {class_name}: AUC={auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ Error evaluating {class_name}: {e}")
            results[class_name] = {
                'support': 0,
                'auc': 0.5,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'specificity': 1.0,
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
            }
    
    return results

def save_results(results, y_true, y_pred, output_dir, actual_classes):
    """Save evaluation results"""
    print("💾 Saving results...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_results = []
    for class_name, metrics in results.items():
        detailed_results.append({
            'Class': class_name,
            'Support': metrics['support'],
            'AUC': metrics['auc'],
            'F1-Score': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'True Positives': metrics['tp'],
            'False Positives': metrics['fp'],
            'False Negatives': metrics['fn'],
            'True Negatives': metrics['tn']
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
    
    return output_dir

def create_confusion_matrix(y_true, y_pred, actual_classes, output_dir, threshold=0.5):
    """Create and save confusion matrix"""
    print("📊 Creating confusion matrix...")
    
    try:
        binary_preds = (y_pred >= threshold).astype(int)
        
        # Create confusion matrix for each class
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, class_name in enumerate(actual_classes):
            if i < len(axes):
                y_true_class = y_true[:, i]
                y_pred_class = binary_preds[:, i]
                
                if np.sum(y_true_class) > 0:  # Only plot if there are positive samples
                    cm = confusion_matrix(y_true_class, y_pred_class)
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(f'{class_name} Confusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                else:
                    axes[i].text(0.5, 0.5, f'{class_name}\nNo positive samples', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{class_name} (No Data)')
        
        # Hide unused subplots
        for i in range(len(actual_classes), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Confusion matrix saved")
        
    except Exception as e:
        print(f"   ⚠️ Error creating confusion matrix: {e}")

def create_roc_curves(y_true, y_pred, actual_classes, output_dir):
    """Create and save ROC curves"""
    print("📊 Creating ROC curves...")
    
    try:
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, class_name in enumerate(actual_classes):
            if i < len(axes):
                y_true_class = y_true[:, i]
                y_pred_class = y_pred[:, i]
                
                if np.sum(y_true_class) > 0 and len(np.unique(y_true_class)) > 1:
                    try:
                        fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
                        auc = roc_auc_score(y_true_class, y_pred_class)
                        
                        axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                                   label=f'ROC curve (AUC = {auc:.2f})')
                        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        axes[i].set_xlim([0.0, 1.0])
                        axes[i].set_ylim([0.0, 1.05])
                        axes[i].set_xlabel('False Positive Rate')
                        axes[i].set_ylabel('True Positive Rate')
                        axes[i].set_title(f'{class_name} ROC Curve')
                        axes[i].legend(loc="lower right")
                    except Exception as e:
                        axes[i].text(0.5, 0.5, f'{class_name}\nError: {str(e)}', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f'{class_name} (Error)')
                else:
                    axes[i].text(0.5, 0.5, f'{class_name}\nOnly one class present', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{class_name} (No ROC)')
                    print(f"   ⚠️  Could not plot ROC for {class_name}: Only one class present in y_true. ROC AUC score is not defined in that case.")
        
        # Hide unused subplots
        for i in range(len(actual_classes), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ ROC curves saved")
        
    except Exception as e:
        print(f"   ⚠️ Error creating ROC curves: {e}")

def main():
    """Main function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='PTB-XL ECG Dataset Batch Prediction for 71-Class Models')
    parser.add_argument('exp_folder', nargs='?', default='output/exp0',
                       help='Experiment folder path (default: output/exp0)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (for debugging)')
    
    args = parser.parse_args()
    
    exp_folder = args.exp_folder
    threshold = args.threshold
    max_samples = args.max_samples
    
    print("🚀 Starting PTB-XL Dataset Batch Prediction for 71-Class Models")
    print(f"📁 Experiment folder: {exp_folder}")
    print(f"🎯 Classification threshold: {threshold}")
    if max_samples:
        print(f"🔢 Max samples: {max_samples}")
    
    try:
        # Load model and preprocessing
        model_path, scaler, mlb = load_model_and_preprocessing(exp_folder)
        
        # Load model using the same approach as classify_single_ecg128.py
        from code.models.xresnet1d import xresnet1d101
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt)
        
        # Infer hidden_size and create model
        n_out = len(mlb.classes_)
        n_in = 12  # ECG leads
        hidden_size = None
        
        for k, v in state.items():
            if v.ndim == 2 and v.shape[0] == n_out:
                hidden_size = v.shape[1]
                print(f"   🔍 Inferred hidden_size from `{k}`: {hidden_size}")
                break
        
        if hidden_size is None:
            raise ValueError(f"❌ Unable to infer hidden_size from checkpoint (n_out={n_out})")
        
        # Build model
        model = xresnet1d101(
            input_channels=n_in,
            num_classes=n_out,
            lin_ftrs_head=[hidden_size],
        ).to(device)
        
        # Load weights
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"   📱 Using device: {device}")
        
        # Load dataset
        X_test, y_test = load_ptbxl_test_data(exp_folder)
        
        # Make predictions
        y_test_pred = predict_batch_ptbxl_optimized(X_test, model, scaler, device)
        
        # Debug: Check prediction statistics
        print(f"🔍 Prediction debug info:")
        print(f"   Y test pred shape: {y_test_pred.shape}")
        print(f"   Y test pred range: [{y_test_pred.min():.6f}, {y_test_pred.max():.6f}]")
        print(f"   Y test pred mean: {y_test_pred.mean():.6f}")
        print(f"   Y test pred std: {y_test_pred.std():.6f}")
        print(f"   Non-zero predictions (>0.1): {np.count_nonzero(y_test_pred > 0.1)}")
        print(f"   Non-zero predictions (>0.01): {np.count_nonzero(y_test_pred > 0.01)}")
        
        # Extract target classes
        y_true, y_pred, actual_classes = extract_target_classes(y_test, y_test_pred, mlb, TARGET_ECG_CLASSES)
        
        # Limit samples if specified
        if max_samples and max_samples < len(y_true):
            y_true = y_true[:max_samples]
            y_pred = y_pred[:max_samples]
            print(f"🔢 Limited to {max_samples} samples for debugging")
        
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_path).stem
        output_dir = Path("output/prediction/ptbxl_test") / f"batch_ptbxl_71class_results" / f"run_{timestamp}_{model_name}"
        
        print(f"📁 Output directory: {output_dir}")
        
        # Evaluate performance
        start_time = time.time()
        results = evaluate_performance(y_true, y_pred, actual_classes, threshold=threshold)
        eval_time = time.time() - start_time
        
        print(f"⏱️  Evaluation time: {eval_time:.2f} seconds")
        
        # Save results
        save_results(results, y_true, y_pred, output_dir, actual_classes)
        
        # Create visualizations
        create_confusion_matrix(y_true, y_pred, actual_classes, output_dir, threshold)
        create_roc_curves(y_true, y_pred, actual_classes, output_dir)
        
        print(f"\n🎉 PTB-XL dataset batch prediction completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Total samples evaluated: {len(y_true)}")
        print(f"⏱️  Total processing time: {eval_time:.2f} seconds")
        print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
