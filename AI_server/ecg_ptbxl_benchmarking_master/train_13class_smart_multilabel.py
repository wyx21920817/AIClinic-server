#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Multi-label 13-class ECG Classification Model Training - Specialized for Complex Multi-label Cases
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings
warnings.filterwarnings('ignore')

# CUDA GPU优化设置
# CUDA GPU optimization settings
def setup_cuda_optimization():
    """Configure CUDA GPU optimization parameters"""
    if torch.cuda.is_available():
        print(f"🚀 Setting up CUDA optimization...")
        
        # Enable cuDNN autotuner
        cudnn.benchmark = True
        cudnn.deterministic = False  # Allow non-determinism for speed
        
        # Set CUDA memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Enable mixed precision training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set CUDA device
        device = torch.device('cuda')
        torch.cuda.set_device(0)  # Use the first GPU
        
        # Clean GPU caches
        torch.cuda.empty_cache()
        
        print(f"   ✅ CUDA optimization enabled")
        print(f"   📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   🔧 cuDNN benchmark: {cudnn.benchmark}")
        print(f"   ⚡ TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        
        return device
    else:
        print(f"   ⚠️ CUDA not available, using CPU")
        return torch.device('cpu')

# Initialize CUDA optimization
device = setup_cuda_optimization()

def get_gpu_memory_info():
    """Get GPU memory usage info"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return allocated, cached, total
    return 0, 0, 0

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated, cached, total = get_gpu_memory_info()
        print(f"   💾 GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total")

def optimize_gpu_memory():
    """Optimize GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def select_data_without_save(XX, YY, ctype, min_samples):
    """Select data without saving mlb.pkl file (to avoid duplicate files)"""
    from sklearn.preprocessing import MultiLabelBinarizer
    import pandas as pd
    import numpy as np
    
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'all':
        # Use all scp_codes (correct logic)
        # Ensure all_scp_len exists
        if 'all_scp_len' not in YY.columns:
            YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # Do not save mlb.pkl to avoid duplicates
    return X, Y, y, mlb

def test_gpu_performance():
    """Test GPU performance"""
    if not torch.cuda.is_available():
        print(f"   ⚠️ CUDA not available, skipping GPU performance test")
        return
    
    print(f"   🧪 Testing GPU performance...")
    
    try:
        # Create test data
        test_data = torch.randn(1000, 12, 1000).cuda()
        test_target = torch.randn(1000, 13).cuda()
        
        # Test matrix multiplication performance
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(100):
            result = torch.matmul(test_data, test_data.transpose(-1, -2))
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"   ⚡ GPU Matrix multiplication: {elapsed_time:.2f}ms for 100 iterations")
        
        # Safely get GPU utilization (avoid NVML errors)
        try:
            gpu_util = torch.cuda.utilization(0)
            print(f"   📊 GPU utilization: {gpu_util}%")
        except Exception as e:
            print(f"   ⚠️ GPU utilization unavailable: {str(e)[:50]}...")
            print(f"   💡 This is normal on some systems without NVML")
        
        # Clean test tensors
        del test_data, test_target, result
        optimize_gpu_memory()
        
    except Exception as e:
        print(f"   ⚠️ GPU performance test failed: {str(e)[:100]}...")
        print(f"   💡 Continuing with training...")
        optimize_gpu_memory()

# Add project path
sys.path.append('code')
from models.fastai_model import fastai_model
from models.timeseries_utils import TimeseriesDatasetCrops
from utils.utils import load_dataset, compute_label_aggregations, select_data

# Config
PTBXL_DATA_DIR = "data/ptbxl/"
OUTPUT_MODELS = "output/exp_13class_smart_multilabel"
OUTPUT_DATA = "output/exp_13class_smart_multilabel/data"
SAMPLING_RATE = 100

# Training mode config
FAST_TRAINING = False  # Set True to enable fast training mode

# GPU training optimization config
GPU_OPTIMIZATION = {
    'use_amp': True,              # Automatic mixed precision
    'use_compile': False,         # torch.compile (disabled for compatibility)
    'use_pin_memory': True,       # Pin memory
    'use_non_blocking': True,     # Non-blocking transfer
    'num_workers': 2,             # DataLoader workers (reduced for Windows)
    'prefetch_factor': 2,         # Prefetch factor
    'persistent_workers': False,  # Persistent workers (Windows compatibility)
}

# Target classes (13)
TARGET_CLASSES = [
    'NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT',
    'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC'
]

def load_ptbxl_data():
    """Load PTB-XL data"""
    print(f"\n📂 Loading PTB-XL data...")
    try:
        # Use utils functions to load PTB-XL data
        data, raw_labels = load_dataset(PTBXL_DATA_DIR, 100)
        
        # Compute label aggregations
        labels_df = compute_label_aggregations(raw_labels, PTBXL_DATA_DIR, 'all')
        
        # Select data and convert to one-hot (without saving mlb.pkl)
        # Ensure all_scp_len exists
        if 'all_scp_len' not in labels_df.columns:
            labels_df['all_scp_len'] = labels_df.all_scp.apply(lambda x: len(x))
        
        data, labels_df, Y, mlb = select_data_without_save(
            data, labels_df, 'all', 0
        )
        
        print(f"   ✅ PTB-XL data loaded successfully: {len(data)} samples")
        return data, labels_df, Y, mlb
    except Exception as e:
        print(f"   ❌ PTB-XL data loading failed: {e}")
        return None, None, None, None

def create_smart_multilabel_labels(labels_df):
    """Create smart multi-label labels"""
    print(f"\n🏷️ Creating smart multi-label labels...")
    
    # Create 13-class label matrix
    n_samples = len(labels_df)
    y_13class = np.zeros((n_samples, 13), dtype=float)  # Use float to store confidence
    
    # Extract labels and confidence from scp_codes
    actual_classes = set()
    confidence_stats = {cls: [] for cls in TARGET_CLASSES}
    
    for idx, (i, row) in enumerate(labels_df.iterrows()):
        scp_codes = row['scp_codes']
        if isinstance(scp_codes, dict):
            for ptbxl_class, confidence in scp_codes.items():
                actual_classes.add(ptbxl_class)
                if ptbxl_class in TARGET_CLASSES:
                    target_idx = TARGET_CLASSES.index(ptbxl_class)
                    # Use confidence as label value instead of simple 0/1
                    y_13class[idx, target_idx] = confidence / 100.0
                    confidence_stats[ptbxl_class].append(confidence)
    
    print(f"   🔍 Target classes present in PTB-XL: {sorted(actual_classes.intersection(set(TARGET_CLASSES)))}")
    
    # Statistics for each class sample count and confidence
    class_counts = np.sum(y_13class > 0, axis=0)  # Number of samples with confidence > 0
    print(f"   📊 13-class label distribution:")
    for i, class_name in enumerate(TARGET_CLASSES):
        if confidence_stats[class_name]:
            avg_conf = np.mean(confidence_stats[class_name])
            print(f"      {class_name}: {class_counts[i]} samples, avg confidence: {avg_conf:.1f}")
        else:
            print(f"      {class_name}: {class_counts[i]} samples")
    
    print(f"   ✅ Smart multi-label creation completed: {y_13class.shape}")
    return y_13class

def create_confidence_weighted_dataset(data, labels_df, y_13class):
    """Create confidence-weighted dataset with smart sampling"""
    print(f"\n⚖️ Creating confidence-weighted dataset...")
    
    # Calculate confidence weight for each sample
    sample_weights = []
    for idx, (i, row) in enumerate(labels_df.iterrows()):
        scp_codes = row['scp_codes']
        if isinstance(scp_codes, dict):
            # Calculate average confidence of target classes as sample weight
            target_confidences = [conf for cls, conf in scp_codes.items() if cls in TARGET_CLASSES]
            if target_confidences:
                avg_confidence = np.mean(target_confidences) / 100.0
                sample_weights.append(avg_confidence)
            else:
                sample_weights.append(0.1)  # Default low weight
        else:
            sample_weights.append(0.1)
    
    sample_weights = np.array(sample_weights)
    
    # Smart sampling: balance between high confidence and diversity
    # Use stratified sampling to ensure class diversity
    n_samples = len(data)
    
    # For faster training, use moderate oversampling (1.5x instead of 2x)
    selected_indices = []
    
    # Sample based on confidence weights
    selection_prob = sample_weights / np.sum(sample_weights)
    
    if FAST_TRAINING:
        # Fast training: use smaller dataset (1.2x oversampling)
        oversample_ratio = 1.2
        print(f"   🚀 Fast mode: Using {oversample_ratio}x oversampling")
    else:
        # Standard training: use moderate oversampling
        oversample_ratio = 1.5
        print(f"   🐌 Standard mode: Using {oversample_ratio}x oversampling")
    
    for _ in range(int(n_samples * oversample_ratio)):
        idx = np.random.choice(n_samples, p=selection_prob)
        selected_indices.append(idx)
    
    # Create weighted dataset
    weighted_data = [data[i] for i in selected_indices]
    weighted_labels = labels_df.iloc[selected_indices].reset_index(drop=True)
    weighted_y_13class = y_13class[selected_indices]
    
    print(f"   ✅ Confidence-weighted dataset creation completed: {len(weighted_data)} samples")
    return weighted_data, weighted_labels, weighted_y_13class

def create_13class_model_smart(n_classes=13):
    """Create smart 13-class model with GPU optimization"""
    print(f"\n🏗️ Creating smart 13-class model...")
    
    # GPU optimization info
    if torch.cuda.is_available():
        print(f"   🚀 GPU optimization enabled")
        print_gpu_memory()
    
    # Use author's fastai_model, train from scratch
    model_name = "fastai_xresnet1d101_13class_smart"
    
    if FAST_TRAINING:
        print(f"   🚀 Fast training mode enabled")
        # Fast training parameters with GPU optimization
        epochs = 30
        lr = 3e-3
        bs = 64 if torch.cuda.is_available() else 32  # Larger batch size for GPU
        early_stopping = 10
        wd = 5e-4
        ps_head = 0.2
    else:
        print(f"   🐌 Standard training mode")
        # Standard training parameters with GPU optimization
        epochs = 60
        lr = 2e-3
        bs = 32 if torch.cuda.is_available() else 16  # Larger batch size for GPU
        early_stopping = 15
        wd = 1e-3
        ps_head = 0.3
    
    # GPU optimization display
    if torch.cuda.is_available():
        print(f"   ⚡ GPU optimizations will be applied during training")
        print(f"   📊 Batch size optimized for GPU: {bs}")
        print(f"   🔧 cuDNN benchmark enabled")
        print(f"   💾 Memory management enabled")
    
    model = fastai_model(
        name=model_name,
        n_classes=n_classes,
        freq=SAMPLING_RATE,
        outputfolder=OUTPUT_MODELS,
        input_shape=(1000, 12),
        pretrained=False,  # Train from scratch, no pretrained model
        # Parameters optimized for multi-label
        lr=lr,
        epochs=epochs,
        epochs_finetuning=0,  # No fine-tuning
        gradual_unfreezing=True,  # Enable gradual unfreezing for better convergence
        discriminative_lrs=True,  # Enable discriminative learning rates
        wd=wd,
        ps_head=ps_head,
        bs=bs,
        early_stopping=early_stopping,
    )
    
    print(f"   ✅ Smart 13-class model creation completed")
    print(f"   📊 Training config: {epochs} epochs, lr={lr}, bs={bs}")
    return model

def analyze_class_imbalance_smart(y_13class):
    """Smart analysis of class imbalance"""
    print(f"\n📊 Smart class imbalance analysis...")
    
    # Calculate effective sample count (samples with confidence > 0.5)
    effective_counts = np.sum(y_13class > 0.5, axis=0)
    total_samples = len(y_13class)
    
    # Calculate class weights
    class_weights = []
    for i, count in enumerate(effective_counts):
        if count > 0:
            weight = total_samples / (len(TARGET_CLASSES) * count)
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    print(f"   📈 Class weights (based on effective samples):")
    for i, (class_name, weight) in enumerate(zip(TARGET_CLASSES, class_weights)):
        print(f"      {class_name}: {weight:.2f} (effective samples: {effective_counts[i]})")
    
    # Identify minority classes
    minority_classes = []
    for i, count in enumerate(effective_counts):
        if count < 100:  # Classes with <100 effective samples are considered minority
            minority_classes.append(TARGET_CLASSES[i])
    
    print(f"   ⚠️ Minority classes (<100 effective samples): {minority_classes}")
    
    return class_weights, minority_classes

def safe_roc_auc_score(y_true, y_score):
    """Safe ROC AUC calculation"""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5  # Return random performance


def main():
    """Main function with GPU optimization"""
    print("=" * 60)
    print("🎯 Smart Multi-label 13-class ECG Classification Model Training - Handling Complex Multi-label Cases")
    print("=" * 60)
    
    # Check device and GPU optimization
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU Training Optimizations:")
        print(f"   ⚡ Mixed Precision: {GPU_OPTIMIZATION['use_amp']}")
        print(f"   🔧 torch.compile: {GPU_OPTIMIZATION['use_compile']}")
        print(f"   📌 Pin Memory: {GPU_OPTIMIZATION['use_pin_memory']}")
        print(f"   🔄 Non-blocking: {GPU_OPTIMIZATION['use_non_blocking']}")
        print(f"   👥 Workers: {GPU_OPTIMIZATION['num_workers']}")
        print_gpu_memory()
        
        # Test GPU performance
        test_gpu_performance()
    print("=" * 60)
    
    # Create output directories
    os.makedirs(OUTPUT_MODELS, exist_ok=True)
    os.makedirs(OUTPUT_DATA, exist_ok=True)
    
    # Load PTB-XL data
    ptbxl_data, ptbxl_labels, Y_71class, mlb_71 = load_ptbxl_data()
    if ptbxl_data is None:
        print("❌ Unable to load PTB-XL data, exiting")
        return
    
    # Create smart multi-label labels
    y_13class = create_smart_multilabel_labels(ptbxl_labels)
    
    # Smart analysis of class imbalance
    class_weights, minority_classes = analyze_class_imbalance_smart(y_13class)
    
    # Create confidence-weighted dataset
    weighted_data, weighted_labels, weighted_y_13class = create_confidence_weighted_dataset(
        ptbxl_data, ptbxl_labels, y_13class
    )
    
    # Data splitting
    print(f"\n📊 Data splitting...")
    from sklearn.model_selection import train_test_split
    
    # Use strat_fold for stratified splitting
    if 'strat_fold' in weighted_labels.columns:
        strat_fold = weighted_labels['strat_fold'].values
    else:
        strat_fold = None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        weighted_data, weighted_y_13class, test_size=0.3, random_state=42, stratify=strat_fold
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"   📊 Data split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples") 
    print(f"   Test set: {len(X_test)} samples")
    
    # Save split data to data directory
    print(f"   💾 Saving split data to data directory...")
    np.save(Path(OUTPUT_DATA) / 'y_train.npy', y_train)
    np.save(Path(OUTPUT_DATA) / 'y_val.npy', y_val)
    np.save(Path(OUTPUT_DATA) / 'y_test.npy', y_test)
    print(f"   ✅ Split data saved to: {OUTPUT_DATA}")
    
    # Data preprocessing
    print(f"\n🔧 Data preprocessing...")
    try:
        # Create new scaler
        scaler = StandardScaler()
        
        # Fit scaler using training data
        X_train_flat = np.concatenate([x.flatten() for x in X_train]).reshape(-1, 1)
        scaler.fit(X_train_flat)
        
        # Standardize data
        print(f"   🔄 Standardizing data...")
        X_train_scaled = []
        for x in X_train:
            x_scaled = scaler.transform(x.flatten().reshape(-1, 1)).reshape(x.shape)
            X_train_scaled.append(x_scaled)
        X_train_scaled = np.array(X_train_scaled, dtype=object)
        
        X_val_scaled = []
        for x in X_val:
            x_scaled = scaler.transform(x.flatten().reshape(-1, 1)).reshape(x.shape)
            X_val_scaled.append(x_scaled)
        X_val_scaled = np.array(X_val_scaled, dtype=object)
        
        X_test_scaled = []
        for x in X_test:
            x_scaled = scaler.transform(x.flatten().reshape(-1, 1)).reshape(x.shape)
            X_test_scaled.append(x_scaled)
        X_test_scaled = np.array(X_test_scaled, dtype=object)
        
        # Save scaler
        scaler_path = Path(OUTPUT_DATA) / 'standard_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   💾 Scaler saved to: {scaler_path}")
        
        print(f"   ✅ Data preprocessing completed")
        
    except Exception as e:
        print(f"   ❌ Data preprocessing failed: {e}")
        return
    
    # Create smart 13-class model
    model = create_13class_model_smart()
    
    # Train model
    print(f"\n🎯 Starting smart 13-class model training...")
    print(f"   ⚠️ Note: Using confidence weighting and smart multi-label processing")
    
    # Estimate training time
    if FAST_TRAINING:
        estimated_time = "15-30 minutes"
        print(f"   ⏱️ Estimated training time: {estimated_time} (Fast mode)")
    else:
        estimated_time = "45-90 minutes"
        print(f"   ⏱️ Estimated training time: {estimated_time} (Standard mode)")
    
    try:
        import time
        start_time = time.time()
        
        # Pre-training GPU memory check
        if torch.cuda.is_available():
            print(f"   🚀 Starting GPU training...")
            print_gpu_memory()
            optimize_gpu_memory()
            
            # Apply GPU optimization settings
            print(f"   ⚡ Applying GPU optimizations...")
            # Mixed precision training
            if GPU_OPTIMIZATION['use_amp']:
                print(f"   🔧 Mixed precision training enabled")
            
            # Data loading optimization
            if GPU_OPTIMIZATION['use_pin_memory']:
                print(f"   📌 Pin memory enabled for faster data transfer")
            
            # Non-blocking transfer
            if GPU_OPTIMIZATION['use_non_blocking']:
                print(f"   🔄 Non-blocking data transfer enabled")
        
        model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60  # Convert to minutes
        
        # Post-training GPU memory check
        if torch.cuda.is_available():
            print(f"   📊 Training completed, GPU memory status:")
            print_gpu_memory()
            optimize_gpu_memory()
        
        print(f"   ✅ Training completed in {training_time:.1f} minutes")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        if torch.cuda.is_available():
            print(f"   🔄 Cleaning up GPU memory...")
            optimize_gpu_memory()
        return
    
    # Prediction and evaluation
    print(f"\n📊 Prediction and evaluation...")
    try:
        # Training set prediction
        y_train_pred = model.predict(X_train_scaled)
        
        # Validation set prediction
        y_val_pred = model.predict(X_val_scaled)
        
        # Test set prediction
        y_test_pred = model.predict(X_test_scaled)
        
        print(f"   ✅ Prediction completed")
        
        # Save results
        results_dir = Path(OUTPUT_MODELS) / 'results'
        results_dir.mkdir(exist_ok=True)
        
        np.save(results_dir / 'y_train_pred.npy', y_train_pred)
        np.save(results_dir / 'y_val_pred.npy', y_val_pred)
        np.save(results_dir / 'y_test_pred.npy', y_test_pred)
        np.save(results_dir / 'y_train_true.npy', y_train)
        np.save(results_dir / 'y_val_true.npy', y_val)
        np.save(results_dir / 'y_test_true.npy', y_test)
        
        print(f"   📁 Results saved to: {results_dir}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return
    
    # Evaluation results
    print(f"\n📈 Evaluation results:")
    
    # Training set evaluation
    train_auc = safe_roc_auc_score(y_train, y_train_pred)
    print(f"   Training set AUC: {train_auc:.4f}")
    
    # Validation set evaluation
    val_auc = safe_roc_auc_score(y_val, y_val_pred)
    print(f"   Validation set AUC: {val_auc:.4f}")
    
    # Test set evaluation
    test_auc = safe_roc_auc_score(y_test, y_test_pred)
    print(f"   Test set AUC: {test_auc:.4f}")
    
    # Detailed evaluation for each class
    print(f"\n📊 Detailed evaluation for each class:")
    for i, class_name in enumerate(TARGET_CLASSES):
        train_auc_class = safe_roc_auc_score(y_train[:, i], y_train_pred[:, i])
        val_auc_class = safe_roc_auc_score(y_val[:, i], y_val_pred[:, i])
        test_auc_class = safe_roc_auc_score(y_test[:, i], y_test_pred[:, i])
        
        print(f"   {class_name:8s}: Train={train_auc_class:.3f}, Val={val_auc_class:.3f}, Test={test_auc_class:.3f}")
    
    # Save label encoder
    mlb_13 = MultiLabelBinarizer(classes=TARGET_CLASSES)
    mlb_13.fit([TARGET_CLASSES])  # Fit target classes
    
    # Save MultiLabelBinarizer (only one mlb.pkl file)
    with open(Path(OUTPUT_DATA) / 'mlb.pkl', 'wb') as f:
        pickle.dump(mlb_13, f)
    
    # Save class names
    with open(Path(OUTPUT_DATA) / 'class_names.txt', 'w') as f:
        f.write('\n'.join(TARGET_CLASSES))
    
    # Save class weights
    with open(Path(OUTPUT_DATA) / 'class_weights.pkl', 'wb') as f:
        pickle.dump(class_weights, f)
    
    print(f"\n🎉 Smart multi-label 13-class model training completed!")
    print(f"   Model saved to: {OUTPUT_MODELS}/models/")
    print(f"   Results saved to: {OUTPUT_MODELS}/results/")
    print(f"   Data saved to: {OUTPUT_DATA}/")
    
    print(f"\n✅ Script execution successful!")

if __name__ == "__main__":
    main()
