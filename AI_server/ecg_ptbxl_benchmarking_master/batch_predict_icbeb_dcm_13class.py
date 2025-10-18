#!/usr/bin/env python3
"""
Ultra-Fast Batch ICBEB Prediction Script
This script uses a single model instance to predict multiple samples without restarting
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
import pydicom
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve
import signal
import warnings
import argparse
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants
OUR_13_CLASSES = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']
ICBEB_TO_OUR_MAPPING = {
    'NORM': 'NORM',
    'AFIB': 'AFIB', 
    '1AVB': '1AVB',
    'CLBBB': 'CLBBB',
    'CRBBB': 'CRBBB',
    'PAC': 'PAC',
    'VPC': 'PVC'  # ICBEB uses VPC, we use PVC
}

# Global output directory (will be set in main function)
OUTPUT_DIR = None

# Global flag for graceful shutdown
_shutdown_requested = False

def extract_model_name_from_path(model_path):
    """Extract model name from path for output directory naming"""
    from pathlib import Path
    # Use actual model filename stem; sanitize and cap length
    model_file = Path(model_path).stem
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_file)
    return safe[:50]

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global _shutdown_requested
    print("\n🛑 Received interrupt signal (Ctrl+C), gracefully stopping program...")
    print("⏳ Waiting for current tasks to complete...")
    _shutdown_requested = True

# Set up signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Ultra-Fast Batch ICBEB Prediction Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model
  python batch_predict_icbeb.py
  
  # Use specific experiment folder
  python batch_predict_icbeb.py output/exp_13class_smart_multilabel
  python batch_predict_icbeb.py output/exp_other_model
        """
    )
    
    parser.add_argument(
        'exp_folder', 
        nargs='?',  # Optional positional argument
        default="output/exp_13class_smart_multilabel",
        help='Experiment folder path containing models/ and data/ directories (default: output/exp_13class_smart_multilabel)'
    )
    
    return parser.parse_args()

def load_model_and_preprocessing(exp_folder):
    """Load model and preprocessing components from experiment folder"""
    print(f"🔧 Loading model and preprocessing components from: {exp_folder}")
    
    try:
        exp_path = Path(exp_folder)
        
        # Check if experiment folder exists
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment folder not found: {exp_path}")
        
        # Set up paths
        models_dir = exp_path / "models"
        data_dir = exp_path / "data"
        
        print(f"   📁 Models directory: {models_dir}")
        print(f"   📁 Data directory: {data_dir}")
        
        # Find model file (recursive search)
        model_files = list(models_dir.rglob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in: {models_dir}")
        
        # Filter out temporary files and prefer main model files
        main_models = [f for f in model_files if not f.name.startswith('tmp')]
        if main_models:
            model_path = main_models[0]
        else:
            model_path = model_files[0]
        
        print(f"   📁 Using model: {model_path}")
        
        # Load prediction module
        with open('classify_single_ecg128.py', 'r', encoding='utf-8') as f:
            code_128 = f.read()

        # Replace relative imports
        code_128 = code_128.replace('from .code.models.xresnet1d import', 'from code.models.xresnet1d import')
        code_128 = code_128.replace('torch.load(weights_path, map_location=device)',
                                    'torch.load(weights_path, map_location=device, weights_only=False)')

        # Execute the code
        globals_128 = {
            '__file__': 'classify_single_ecg128.py',
            '__name__': 'classify_single_ecg128_temp'
        }
        exec(code_128, globals_128)
        
        # Set up seed function
        if 'set_seed' in globals_128:
            globals_128['set_seed_128'] = globals_128['set_seed']
        else:
            def set_seed_128(seed=42):
                torch.manual_seed(seed)
                np.random.seed(seed)
                import random
                random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            globals_128['set_seed_128'] = set_seed_128
        
        # Set seed once
        globals_128['set_seed_128'](42)
        
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
        
        print(f"   ✅ Loaded scaler and mlb from: {data_dir}")
        
        # Detect model output classes from checkpoint
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
                n_out = len(OUR_13_CLASSES)  # Default fallback
                print(f"   ⚠️  Could not detect output classes, using default: {n_out}")
        else:
            n_out = len(OUR_13_CLASSES)  # Default fallback
            print(f"   ⚠️  Could not detect output classes, using default: {n_out}")
        
        # Pre-load the model and set to eval mode
        print("   🔄 Pre-loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = globals_128['load_model'](
            str(model_path), 
            n_in=12, 
            n_out=n_out, 
            device=device
        )
        if model is not None:
            model.eval()  # Set to evaluation mode
            print("   ✅ Model pre-loaded and set to eval mode")
            globals_128['model'] = model
            globals_128['device'] = device
            globals_128['n_out'] = n_out  # Store the actual output size
        else:
            raise RuntimeError("Model loading returned None")
        
        # Store model path in globals for later use
        globals_128['model_path'] = str(model_path)
        
        print("✅ Model and preprocessing components loaded")
        return globals_128, scaler, mlb
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def extract_ecg_from_dicom(dicom_path):
    """Extract ECG data from DICOM file using WaveformSequence (ICBEB format, optimized)"""
    try:
        # Use defer_size for faster DICOM reading
        ds = pydicom.dcmread(dicom_path, force=True, defer_size="1 KB")
        
        # ICBEB DICOM files use WaveformSequence, not pixel_array
        if not hasattr(ds, 'WaveformSequence'):
            raise ValueError("WaveformSequence not found in DICOM file")
        
        wf = ds.WaveformSequence[0]
        data = wf.WaveformData
        n_s = wf.NumberOfWaveformSamples
        n_c = wf.NumberOfWaveformChannels
        
        # Optimized array operations
        arr = np.frombuffer(data, dtype=np.int16)
        expected_size = n_s * n_c
        
        if arr.size != expected_size:
            if arr.size < expected_size:
                arr = np.pad(arr, (0, expected_size - arr.size))
            else:
                arr = arr[:expected_size]
        
        # Reshape and normalize in one step
        base_sig = (arr.reshape((n_s, n_c)).astype(np.float32).T / 1000.0)
        
        return base_sig
        
    except Exception as e:
        print(f"❌ Failed to extract ECG from {dicom_path}: {e}")
        return None

def predict_batch_fast(dcm_paths, pred_module, scaler, mlb):
    """Ultra-fast batch prediction using single model instance (optimized)"""
    print(f"🚀 Starting optimized batch prediction for {len(dcm_paths)} samples...")
    
    predictions = []
    successful_predictions = 0
    
    # Get model and device once (optimization)
    model = pred_module['model']
    device = pred_module['device']
    
    # Progress tracking (reduced frequency for better performance)
    total_samples = len(dcm_paths)
    progress_interval = max(1, total_samples // 20)  # Update every 5%
    
    for i, dcm_path in enumerate(dcm_paths):
        # Check for shutdown request
        if _shutdown_requested:
            print(f"\n🛑 Interrupt request received, processed {i}/{total_samples} samples")
            break
        
        # Progress update (reduced frequency)
        if i % progress_interval == 0 or i == total_samples - 1:
            progress = (i + 1) / total_samples * 100
            print(f"   📊 Progress: {i+1}/{total_samples} ({progress:.1f}%) - Success: {successful_predictions}")
        
        try:
            # Extract ECG (optimized DICOM reading)
            base_sig = extract_ecg_from_dicom(dcm_path)
            if base_sig is None:
                predictions.append(np.zeros(13))
                continue
            
            # Preprocess ECG data (no model.eval() call - already in eval mode)
            sig, _ = pred_module['preprocess_signal'](base_sig, scaler)
            x = torch.from_numpy(sig).unsqueeze(0).to(device)
            
            # Inference (model already in eval mode, no need to call again)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            # Convert to our 13-class format (vectorized operation)
            n_out = pred_module.get('n_out', len(OUR_13_CLASSES))
            pred_probs = np.zeros(13)
            
            # Handle different model output sizes
            if n_out == len(OUR_13_CLASSES):
                # Direct mapping for 13-class models
                pred_probs[:len(probs)] = probs
            else:
                # For other models (like 71-class), we need to map to our 13 classes
                print(f"   ⚠️  Model has {n_out} classes, but we expect 13. Using first 13 classes.")
                pred_probs[:min(13, len(probs))] = probs[:min(13, len(probs))]
            
            predictions.append(pred_probs)
            successful_predictions += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {dcm_path.name}: {e}")
            predictions.append(np.zeros(13))
    
    print(f"✅ Optimized batch prediction completed: {successful_predictions}/{len(dcm_paths)} successful")
    return np.array(predictions)

def load_icbeb_database():
    """Load ICBEB database"""
    try:
        df = pd.read_csv("data/ICBEB/icbeb_database.csv")
        print(f"   ✅ Loaded {len(df)} records from ICBEB database")
        return df
    except Exception as e:
        print(f"   ❌ Failed to load ICBEB database: {e}")
        return pd.DataFrame()

def extract_true_labels(dcm_paths):
    """Extract true labels for the DICOM files"""
    try:
        import ast
        df = pd.read_csv("data/ICBEB/icbeb_database.csv")
        
        # Parse scp_codes from string to dict
        def parse_scp_codes(scp_str):
            try:
                if pd.isna(scp_str):
                    return {}
                if isinstance(scp_str, str):
                    return ast.literal_eval(scp_str)
                return scp_str
            except:
                return {}
        
        df['scp_codes_parsed'] = df['scp_codes'].apply(parse_scp_codes)
        y_true = []
        
        for dcm_path in dcm_paths:
            # Get filename without extension
            filename = dcm_path.stem
            
            # Find corresponding record in database
            if filename.startswith('A'):
                file_num = filename[1:].lstrip('0') or '0'
                ecg_id = int(file_num)
            else:
                ecg_id = int(filename)
            
            # Find the record
            record = df[df['ecg_id'] == ecg_id]
            if record.empty:
                y_true.append(np.zeros(13))
                continue
            
            # Extract labels from parsed scp_codes
            scp_codes = record.iloc[0]['scp_codes_parsed']
            
            # Create binary vector (use 1.0 for any confidence > 0)
            binary_labels = np.zeros(13, dtype=np.float32)
            for icbeb_class, confidence in scp_codes.items():
                if icbeb_class in ICBEB_TO_OUR_MAPPING:
                    our_class = ICBEB_TO_OUR_MAPPING[icbeb_class]
                    class_idx = OUR_13_CLASSES.index(our_class)
                    binary_labels[class_idx] = 1.0  # Binary: 1 if present, 0 if not
            
            y_true.append(binary_labels)
        
        return np.array(y_true)
        
    except Exception as e:
        print(f"   ❌ Failed to extract true labels: {e}")
        return None

def evaluate_performance(y_true, y_pred, class_names, use_threshold=True, threshold=0.5):
    """Evaluate model performance with flexible threshold options"""
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    
    print("📊 Evaluating model performance...")
    print(f"🔧 Using {'fixed threshold' if use_threshold else 'top-k'} approach")
    
    # Apply thresholding or top-k approach
    if use_threshold:
        binary_predictions = (y_pred > threshold).astype(np.float32)
        print(f"   📊 Threshold: {threshold}")
    else:
        # Top-k approach: for each sample, select the class with highest probability
        binary_predictions = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            max_idx = np.argmax(y_pred[i])
            binary_predictions[i, max_idx] = 1.0
        print("   📊 Top-1 approach: highest probability class per sample")
    
    # Use the processed predictions
    y_pred = binary_predictions
    
    # Calculate metrics for each class
    results = {}
    
    for i, class_name in enumerate(class_names):
        if class_name in ICBEB_TO_OUR_MAPPING.values():
            try:
                # Only evaluate classes that exist in ICBEB
                y_true_class = y_true[:, i]
                y_pred_class = y_pred[:, i]
                
                # Skip if no positive samples
                if np.sum(y_true_class) == 0:
                    print(f"   ⚠️  Skipping {class_name} - no positive samples in ICBEB")
                    continue
                
                # Calculate metrics
                try:
                    auc = roc_auc_score(y_true_class, y_pred_class) if len(np.unique(y_true_class)) > 1 else 0.0
                except:
                    auc = 0.0
                
                f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
                
                # Calculate precision and recall
                tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
                fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
                fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
                tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                results[class_name] = {
                    'auc': auc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'support': int(np.sum(y_true_class))
                }
                
                print(f"   ✅ {class_name}: AUC={auc:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, S={specificity:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {class_name}: {e}")
    
    return results

def save_results(predictions, y_true, dcm_paths, execution_time):
    """Save prediction results and performance metrics"""
    try:
        # Save prediction details
        prediction_details = []
        for i, (dcm_path, pred_probs, true_labels) in enumerate(zip(dcm_paths, predictions, y_true)):
            # Convert probabilities to binary predictions (threshold = 0.5)
            pred_binary = (pred_probs > 0.5).astype(np.float32)
            
            prediction_details.append({
                'filename': dcm_path.name,
                'ecg_id': dcm_path.stem,
                'pred_probs': pred_probs.tolist(),
                'pred_binary': pred_binary.tolist(),
                'true_labels': true_labels.tolist()
            })
        
        # Save to JSON
        details_file = OUTPUT_DIR / "batch_prediction_details.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_details, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Prediction details saved to: {details_file}")
        
        # Evaluation method: True for threshold-based, False for top-k
        USE_THRESHOLD = False  # Set to False to use top-k approach (recommended for ICBEB)
        THRESHOLD = 0.5      # Threshold for binary classification (not used when USE_THRESHOLD=False)
        
        # Evaluate performance
        print("\n📊 Evaluating model performance...")
        results = evaluate_performance(y_true, predictions, OUR_13_CLASSES, USE_THRESHOLD, THRESHOLD)
        
        if results:
            # Calculate overall performance
            avg_auc = np.mean([r['auc'] for r in results.values()])
            avg_f1 = np.mean([r['f1'] for r in results.values()])
            avg_precision = np.mean([r['precision'] for r in results.values()])
            avg_recall = np.mean([r['recall'] for r in results.values()])
            
            # Save performance report
            report_data = []
            for class_name, metrics in results.items():
                report_data.append({
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
            
            report_df = pd.DataFrame(report_data)
            report_file = OUTPUT_DIR / "batch_performance_report.csv"
            report_df.to_csv(report_file, index=False)
            print(f"   ✅ Performance report saved to: {report_file}")
            
            # Print summary
            print(f"\n📊 Overall Performance:")
            print(f"   Average AUC: {avg_auc:.4f}")
            print(f"   Average F1-Score: {avg_f1:.4f}")
            print(f"   Average Precision: {avg_precision:.4f}")
            print(f"   Average Recall: {avg_recall:.4f}")
            
            # Save execution statistics
            stats = {
                'total_samples': len(dcm_paths),
                'execution_time_seconds': execution_time,
                'average_time_per_prediction': execution_time / len(dcm_paths),
                'overall_auc': avg_auc,
                'overall_f1': avg_f1,
                'overall_precision': avg_precision,
                'overall_recall': avg_recall
            }
            
            stats_file = OUTPUT_DIR / "batch_execution_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Execution statistics saved to: {stats_file}")
        
        print(f"\n📁 All results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")

def create_confusion_matrix_plot_batch(y_true, y_pred, class_names, output_dir):
    """Create and save confusion matrix plot for batch prediction"""
    try:
        print("📊 Creating confusion matrix plot...")
        
        # Create figure with subplots for each class
        n_classes = len(class_names)
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, class_name in enumerate(class_names):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes
            else:
                ax = axes[row, col] if n_cols > 1 else axes[row]
            
            # Create confusion matrix for this class
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Predicted Negative', 'Predicted Positive'],
                       yticklabels=['Actual Negative', 'Actual Positive'])
            
            ax.set_title(f'{class_name} Confusion Matrix')
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
        
        # Save plot
        plot_file = output_dir / "batch_confusion_matrices.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Confusion matrix plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"   ❌ Error creating confusion matrix plot: {e}")

def create_roc_curves_plot_batch(y_true, y_pred, class_names, output_dir):
    """Create and save ROC curves plot for batch prediction"""
    try:
        print("📊 Creating ROC curves plot...")
        
        plt.figure(figsize=(12, 8))
        
        # Calculate and plot ROC curve for each class
        for i, class_name in enumerate(class_names):
            if np.sum(y_true[:, i]) > 0:  # Only plot if there are positive samples
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{class_name} (AUC = {auc:.3f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for ICBEB Batch Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / "batch_roc_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ ROC curves plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"   ❌ Error creating ROC curves plot: {e}")

def main():
    """Main function for batch prediction"""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load model first to get model path for output directory naming
    pred_module, scaler, mlb = load_model_and_preprocessing(args.exp_folder)
    if pred_module is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Extract model name and create timestamped output directory with model info
    model_path = pred_module.get('model_path', 'unknown_model')
    model_name = extract_model_name_from_path(model_path)
    
    # Create output directory with model identifier
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global OUTPUT_DIR
    OUTPUT_DIR = Path(f"output/prediction/icbeb_test/batch_icbeb_13class_dcm_results/run_{timestamp}_{model_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Ultra-Fast Batch ICBEB Prediction")
    print("=" * 60)
    print(f"📁 Experiment folder: {args.exp_folder}")
    print(f"📁 Model: {model_name}")
    print(f"📁 Results will be saved to: {OUTPUT_DIR}")
    print(f"⏰ Timestamp: {timestamp}")
    print("=" * 60)
    
    # Load ICBEB database to get all available files
    print("📊 Loading ICBEB database to get all available files...")
    try:
        import pandas as pd
        import ast
        
        # Load ICBEB database
        icbeb_db_path = Path("data/ICBEB/icbeb_database.csv")
        df = pd.read_csv(icbeb_db_path)
        print(f"   ✅ Loaded {len(df)} records from ICBEB database")
        
        # Parse scp_codes from string to dict
        def parse_scp_codes(scp_str):
            try:
                if pd.isna(scp_str):
                    return {}
                if isinstance(scp_str, str):
                    return ast.literal_eval(scp_str)
                return scp_str
            except:
                return {}
        
        df['scp_codes_parsed'] = df['scp_codes'].apply(parse_scp_codes)
        
        # Filter records that have DICOM files and contain our target classes
        dcm_dir = Path("data/ICBEB/dcm_records100")
        dcm_paths = []
        available_files = []
        
        for _, row in df.iterrows():
            # Convert filename format (A0001 -> 1.dcm)
            filename = row['filename']
            if filename.startswith('A'):
                file_num = filename[1:].lstrip('0') or '0'
                dcm_path = dcm_dir / f"{file_num}.dcm"
            else:
                dcm_path = dcm_dir / f"{filename}.dcm"
            
            if dcm_path.exists():
                # Check if this record has any of our target classes
                scp_codes = row['scp_codes_parsed']
                has_target_class = any(cls in ICBEB_TO_OUR_MAPPING for cls in scp_codes.keys())
                if has_target_class:
                    dcm_paths.append(dcm_path)
                    available_files.append(row)
        
        print(f"   ✅ Found {len(available_files)} records with DICOM files and target classes")
        
    except Exception as e:
        print(f"❌ Failed to load ICBEB database: {e}")
        # Fallback to original method
        dcm_dir = Path("data/ICBEB/dcm_records100")
        dcm_paths = []
        
        # Get all available DICOM files (not just first 5)
        for dcm_file in dcm_dir.glob("*.dcm"):
            dcm_paths.append(dcm_file)
        
        print(f"   📁 Fallback: Found {len(dcm_paths)} DICOM files for prediction")
    
    if not dcm_paths:
        print("❌ No DICOM files found")
        return
    
    print(f"📊 Total {len(dcm_paths)} DICOM files for prediction")
    
    # Memory optimization info
    print(f"💾 Estimated memory usage: ~{len(dcm_paths) * 0.5:.1f} MB for predictions")
    
    # Extract true labels for evaluation
    print("\n🏷️  Extracting true labels for evaluation...")
    y_true = extract_true_labels(dcm_paths)
    if y_true is None:
        print("❌ Failed to extract true labels")
        return
    
    # Batch prediction
    predictions = predict_batch_fast(dcm_paths, pred_module, scaler, mlb)
    
    # Check if program was interrupted
    if _shutdown_requested:
        print("\n🛑 Program interrupted by user, exiting...")
        return
    
    # Display summary results only
    print("\n📊 Batch prediction completed successfully!")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"⚡ Average time per prediction: {execution_time / len(dcm_paths):.2f} seconds")
    
    # Save results
    print("\n💾 Saving results...")
    save_results(predictions, y_true, dcm_paths, execution_time)
    
    # Create visualizations (use binary predictions)
    print("\n📊 Creating visualizations...")
    y_pred_binary = (predictions > 0.5).astype(np.float32)
    create_confusion_matrix_plot_batch(y_true, y_pred_binary, OUR_13_CLASSES, OUTPUT_DIR)
    create_roc_curves_plot_batch(y_true, predictions, OUR_13_CLASSES, OUTPUT_DIR)  # ROC uses probabilities
    
    print("✅ Ultra-fast batch prediction completed!")

if __name__ == "__main__":
    main()
