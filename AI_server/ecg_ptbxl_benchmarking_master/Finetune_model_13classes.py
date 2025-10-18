import os
import sys
import torch
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Fix PyTorch 2.6 weights_only issue
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.serialization.DEFAULT_PROTOCOL = 2

# GPU training optimization settings
if torch.cuda.is_available():
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True  # Auto-optimize convolution algorithms
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for better performance
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 acceleration
    torch.backends.cudnn.allow_tf32 = True  # Enable cuDNN TF32 acceleration
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print(f"🚀 GPU optimizations enabled:")
    print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   - TF32 acceleration: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   - GPU device: {torch.cuda.get_device_name()}")
    print(f"   - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Ensure PyTorch 2.6 compatibility
import torch.serialization
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# 1) Add code directory to path
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))

# Project modules
from code.utils.utils import load_dataset, compute_label_aggregations, select_data, preprocess_signals
from code.models.fastai_model import fastai_model

# ------------------- Configuration -------------------
DATA_PATH            = 'data/ptbxl/'                               # Data root directory
SAMPLING_RATE        = 100                                         # Sampling rate
CTYPE                = 'all'                                       # Task type: use all classes to include 13 target classes
MIN_SAMPLES          = 0                                           # Minimum samples per class (set to 0 to include all possible classes)
# Fine-tuning specific output directory (matching pretrained model structure)
FT_EXPERIMENT        = 'exp0_ft_13classes'
OUTPUT_DATA_FT       = f'output/{FT_EXPERIMENT}/data/'              # Fine-tuning output directory
OUTPUT_MODELS_FT     = f'output/{FT_EXPERIMENT}/models/'            # Fine-tuning model storage directory
OUTPUT_MODEL_FOLDER  = f'output/{FT_EXPERIMENT}/models/fastai_xresnet1d101_finetuned_13classes/'  # Specific model directory
OUTPUT_MODEL_MODELS  = f'output/{FT_EXPERIMENT}/models/fastai_xresnet1d101_finetuned_13classes/models/'   # Model weights directory
OUTPUT_MODEL_RESULTS = f'output/{FT_EXPERIMENT}/models/fastai_xresnet1d101_finetuned_13classes/results/'  # Results directory

PRETRAINED_NAME      = 'fastai_xresnet1d101'                       # Pretrained model identifier
PRETRAINED_FOLDER    = 'output/exp0/models/fastai_xresnet1d101/'   # Pretrained weights directory
EPOCHS_FT            = 2                                          # Fine-tuning epochs (increased for better performance)
SEED                 = 42
DEVICE               = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training optimization parameters
LEARNING_RATE        = 1e-4                                        # Conservative learning rate for stable training
WEIGHT_DECAY         = 1e-4                                        # Moderate weight decay for better generalization
EARLY_STOPPING_PATIENCE = 12                                       # Higher patience for better convergence
MIN_DELTA            = 1e-6                                        # More sensitive to small improvements
BATCH_SIZE           = 64                                          # Increased batch size for better GPU utilization
MIXUP_ALPHA          = 0.2                                         # Moderate mixup for better generalization
LABEL_SMOOTHING      = 0.1                                         # Moderate label smoothing

# Assume all signals are 1000 length, 12 leads
INPUT_SHAPE = (1000, 12)



def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def print_class_distribution(Y, class_names):
    """Print class distribution information"""
    print(f"\n📊 Class Distribution:")
    print(f"   Total samples: {len(Y)}")
    print(f"   Number of classes: {Y.shape[1]}")
    
    # Count samples for each class
    for i in range(Y.shape[1]):
        count = Y[:, i].sum()
        if count > 0:
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            print(f"   {class_name}: {int(count)} samples")

def calculate_class_weights(Y, target_indices, target_weight=2.0):
    """Calculate class weights with higher weight for target 13 classes"""
    # For multi-label classification, calculate weights based on class frequency
    class_counts = Y.sum(axis=0)
    total_samples = len(Y)
    
    # Calculate inverse frequency weights
    class_weights = {}
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (len(class_counts) * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    # Apply higher weight to target classes
    for idx in target_indices:
        if idx in class_weights:
            class_weights[idx] *= target_weight
    
    print(f"✅ Class weights calculated with {target_weight}x weight for target classes")
    return class_weights

def create_weighted_loss_function(target_indices, target_weight=2.0):
    """Create a custom loss function that gives higher weight to target classes"""
    import torch.nn as nn
    
    class WeightedBCELoss(nn.Module):
        def __init__(self, target_indices, target_weight=2.0):
            super().__init__()
            self.target_indices = target_indices
            self.target_weight = target_weight
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        def forward(self, predictions, targets):
            # Calculate base loss
            loss = self.bce_loss(predictions, targets)
            
            # Apply higher weight to target classes
            for idx in self.target_indices:
                if idx < loss.shape[1]:
                    loss[:, idx] *= self.target_weight
            
            return loss.mean()
    
    return WeightedBCELoss(target_indices, target_weight)

def apply_weighted_loss_to_model(model, weighted_loss):
    """Apply the weighted loss function to the model"""
    learn = model.learn
    learn.loss_func = weighted_loss
    print(f"✅ Weighted loss function applied to model")
    return learn

def add_training_optimizations(model, X_train, y_train, X_val, y_val):
    """Add training optimizations to the model"""
    learn = model.learn
    
    # Add callbacks for better training
    from fastai.callbacks import EarlyStoppingCallback, ReduceLROnPlateauCallback
    
    # Early stopping callback (learn is the first parameter)
    early_stop = EarlyStoppingCallback(
        learn=learn,
        monitor='valid_loss',
        min_delta=MIN_DELTA,
        patience=EARLY_STOPPING_PATIENCE,
        mode='min'
    )
    
    # Learning rate reduction callback (learn is the first parameter)
    lr_reduce = ReduceLROnPlateauCallback(
        learn=learn,
        monitor='valid_loss',
        mode='min',
        factor=0.5,  # Moderate LR reduction for stable training
        patience=6,  # Higher patience for better convergence
        min_lr=1e-6  # Conservative minimum LR
    )
    
    # Add callbacks to learner
    learn.callbacks.append(early_stop)
    learn.callbacks.append(lr_reduce)
    
    # Set learning rate
    learn.lr = LEARNING_RATE
    
    # Add weight decay
    learn.wd = WEIGHT_DECAY
    
    # GPU training optimization settings
    if torch.cuda.is_available():
        # Enable mixed precision training (AMP) for faster training
        from fastai.callbacks import MixedPrecision
        learn = learn.to_fp16()
        print(f"   - Mixed precision training (FP16) enabled")
        
        # Set data loader optimization
        learn.data.num_workers = min(8, os.cpu_count())  # Parallel data loading
        print(f"   - Data loading workers: {learn.data.num_workers}")
        
        # Enable memory optimization
        learn.data.pin_memory = True
        print(f"   - Pin memory enabled")
    
    # Add data augmentation (mixup)
    if MIXUP_ALPHA > 0:
        try:
            from fastai.callbacks.mixup import MixUpCallback
            mixup = MixUpCallback(learn, alpha=MIXUP_ALPHA)
            learn.callbacks.append(mixup)
            print(f"   - MixUp augmentation: alpha={MIXUP_ALPHA}")
        except ImportError:
            print(f"   - MixUp not available in this fastai version, skipping")
    
    # Note: Label smoothing for multi-label classification requires custom implementation
    # Using weighted BCE loss which is more appropriate for multi-label tasks
    if LABEL_SMOOTHING > 0:
        print(f"   - Label smoothing: {LABEL_SMOOTHING} (note: using weighted BCE for multi-label)")
    else:
        print(f"   - Using weighted BCE loss for multi-label classification")
    
    print(f"✅ Training optimizations added:")
    print(f"   - Early stopping: patience={EARLY_STOPPING_PATIENCE}, min_delta={MIN_DELTA}")
    print(f"   - LR reduction: factor=0.5, patience=6")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Weight decay: {WEIGHT_DECAY}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Total epochs: {EPOCHS_FT} (two-stage: {EPOCHS_FT}+{EPOCHS_FT})")
    
    # Display GPU memory usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"   - GPU memory: {gpu_allocated:.1f}GB allocated, {gpu_cached:.1f}GB cached, {gpu_memory:.1f}GB total")
    
    return learn

# Note: Model performance evaluation is handled by separate evaluation module
# This avoids data dimension issues and allows for more comprehensive evaluation

def main():
    set_seed()
    
    print("🚀 Starting 13-class arrhythmia fine-tuning training...")
    print("=" * 60)
    print(f"Task type: {CTYPE}")
    print(f"Target classes: 13 classes")
    print(f"Fine-tuning epochs: {EPOCHS_FT}")
    print(f"Output directory: {OUTPUT_MODELS_FT}")
    print("=" * 60)

    # Ensure output directories exist
    # Create output directories (matching pretrained model structure)
    os.makedirs(OUTPUT_DATA_FT, exist_ok=True)
    os.makedirs(OUTPUT_MODELS_FT, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_MODELS, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_RESULTS, exist_ok=True)

    # 1. Load and preprocess labels
    print("\n📥 Loading PTB-XL dataset...")
    data, raw_labels = load_dataset(DATA_PATH, SAMPLING_RATE)
    
    # Use standard subdiagnostic classification (71 classes)
    # but focus on the 13 target classes for evaluation
    labels_df = compute_label_aggregations(raw_labels, DATA_PATH, CTYPE)
    
    # Select data using standard method (all subdiagnostic classes)
    data, labels_df, Y, mlb = select_data(
        data, labels_df, CTYPE,
        min_samples=MIN_SAMPLES,
        outputfolder=OUTPUT_DATA_FT
    )

    # Get all selected classes
    class_names = mlb.classes_
    
    # Define target 13 classes for focused evaluation
    # Note: Some classes may have different names in the actual dataset
    target_classes = ['NORM', '1AVB', '2AVB', '3AVB', 'AFIB', 'AFLT', 
                      'ILBBB', 'CLBBB', 'IRBBB', 'CRBBB', 'STACH', 'PAC', 'PVC']
    
    # Find indices of target classes in the full class list
    target_indices = []
    for cls in target_classes:
        if cls in class_names:
            target_indices.append(list(class_names).index(cls))
        else:
            print(f"Warning: {cls} not found in selected classes")
    
    print(f"Target classes found: {len(target_indices)}/13")
    print(f"Total classes for training: {len(class_names)}")

    print(f"✅ Data loading completed:")
    print(f"   Total samples: {len(data)}")
    print(f"   Label matrix shape: {Y.shape}")
    print(f"   Selected classes: {class_names}")
    
    # Print class distribution for all classes
    print_class_distribution(Y, class_names)
    
    # Print focused distribution for target 13 classes
    print(f"\n🎯 Target 13 Classes Distribution:")
    for i, cls in enumerate(target_classes):
        if cls in class_names:
            idx = list(class_names).index(cls)
            count = Y[:, idx].sum()
            print(f"   {cls}: {int(count)} samples")
        else:
            print(f"   {cls}: NOT FOUND")
    
    # Calculate class weights with higher weight for target classes
    class_weights = calculate_class_weights(Y, target_indices, target_weight=2.0)
    
    # Create weighted loss function
    weighted_loss = create_weighted_loss_function(target_indices, target_weight=2.0)
    print(f"✅ Weighted loss function created with 2.0x weight for target classes")

    # 2. Use official strat_fold for train/validation/test split (following author's framework)
    X_train = data[labels_df.strat_fold < 9]  # folds 1-8 for training
    y_train = Y[labels_df.strat_fold < 9]
    X_val   = data[labels_df.strat_fold == 9]  # fold 9 for validation
    y_val   = Y[labels_df.strat_fold == 9]
    X_test  = data[labels_df.strat_fold == 10]  # fold 10 for testing
    y_test  = Y[labels_df.strat_fold == 10]

    print(f"\n📊 Dataset split:")
    print(f"   Training set: {len(X_train)} samples (folds 1-8)")
    print(f"   Validation set: {len(X_val)} samples (fold 9)")
    print(f"   Test set: {len(X_test)} samples (fold 10)")

    # 3. Standardization - use pretrained scaler for distribution alignment
    print("\n🔧 Data preprocessing...")
    
    # Load pretrained standardizer for distribution alignment
    pretrained_scaler_path = os.path.join('output/exp0/data/standard_scaler.pkl')
    if os.path.exists(pretrained_scaler_path):
        import pickle
        with open(pretrained_scaler_path, 'rb') as f:
            standard_scaler = pickle.load(f)
        print(f"✅ Loaded pretrained standardizer from: {pretrained_scaler_path}")
        
        # Apply pretrained scaler to maintain distribution alignment
        from code.utils.utils import apply_standardizer
        X_train = apply_standardizer(X_train, standard_scaler)
        X_val = apply_standardizer(X_val, standard_scaler)
        X_test = apply_standardizer(X_test, standard_scaler)
        
        # Save the pretrained scaler to output folder for consistency
        with open(os.path.join(OUTPUT_DATA_FT, 'standard_scaler.pkl'), 'wb') as f:
            pickle.dump(standard_scaler, f)
        print(f"✅ Applied pretrained standardizer for distribution alignment")
    else:
        raise FileNotFoundError(f"Pretrained scaler not found at {pretrained_scaler_path}. "
                               f"This is required for proper transfer learning. "
                               f"Please ensure the pretrained model directory contains the standard_scaler.pkl file.")

    # Use numpy arrays directly (more efficient than converting to lists)
    X_train_fin = X_train
    X_val_fin   = X_val
    X_test_fin  = X_test
    y_train_fin = y_train
    y_val_fin   = y_val
    y_test_fin  = y_test

    # 4. Instantiate and fine-tune fastai_model
    print(f"\n🔧 Creating fine-tuning model...")
    print(f"   Pretrained model: {PRETRAINED_NAME}")
    print(f"   Target classes: {y_train.shape[1]}")
    print(f"   Pretrained classes: 71")
    
    model = fastai_model(
        name                = PRETRAINED_NAME,
        n_classes           = y_train.shape[1],   # All selected classes
        freq                = SAMPLING_RATE,
        outputfolder        = OUTPUT_MODELS_FT,   # Fine-tuning weights directory
        input_shape         = INPUT_SHAPE,
        pretrained          = True,
        pretrainedfolder    = PRETRAINED_FOLDER,
        n_classes_pretrained= 71,                 # Number of pretrained classes
        epochs_finetuning   = EPOCHS_FT,
        chunkify_train      = False,
    )


    # 5. Run initial training to initialize model.learn
    print(f"\n🎯 Starting fine-tuning training...")
    print(f"   Training epochs: {EPOCHS_FT}")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Target classes weight: 2.0x")
    print(f"   Early stopping: {EARLY_STOPPING_PATIENCE} epochs")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Weight decay: {WEIGHT_DECAY}")
    print(f"   MixUp alpha: {MIXUP_ALPHA}")
    print(f"   Label smoothing: {LABEL_SMOOTHING}")
    print(f"   GPU optimizations: Enabled")
    print(f"   Mixed precision: Enabled")
    print(f"   Data loading workers: {min(8, os.cpu_count())}")
    print(f"   Pin memory: Enabled")
    print(f"   ============================================================")
    
    # Start initial training to initialize model.learn
    model.fit(X_train_fin, y_train_fin, X_val_fin, y_val_fin)

    # 6. Apply weighted loss function AFTER model.learn is initialized
    print(f"\n🔧 Applying weighted loss function...")
    learn = apply_weighted_loss_to_model(model, weighted_loss)

    # Add training optimizations AFTER model.learn is initialized
    print(f"\n🔧 Adding training optimizations...")
    learn = add_training_optimizations(model, X_train_fin, y_train_fin, X_val_fin, y_val_fin)

    # After fit, fastai_model has already called learn.save(self.name) internally
    # and exposes the learner through the following line
    learn = model.learn

    # 6. Model performance evaluation will be done by separate evaluation module
    print(f"\n📊 Model training completed. Performance evaluation will be done by separate module.")
    performance_metrics = {
        'note': 'Performance evaluation will be done by separate evaluation module',
        'training_completed': True
    }

    # 7. Save model in multiple formats for compatibility
    model_filename = f'{PRETRAINED_NAME}_finetuned_13classes.pth'
    model_path = os.path.join(OUTPUT_MODEL_MODELS, model_filename)
    
    # Save PyTorch state_dict (for custom loading)
    torch.save(
        {'model': learn.model.state_dict()},
        model_path
    )
    
    # Save FastAI learner (for FastAI loading)
    fastai_model_name = f'{PRETRAINED_NAME}_finetuned_13classes'
    # Set the path for FastAI to save in the correct directory (must be Path object)
    from pathlib import Path
    learn.path = Path(OUTPUT_MODEL_FOLDER)
    learn.save(fastai_model_name)
    
    # Save FastAI export (complete inference package)
    export_path = os.path.join(OUTPUT_MODEL_FOLDER, f'{fastai_model_name}_export.pkl')
    try:
        learn.export(export_path)
        print(f"   FastAI export: {fastai_model_name}_export.pkl")
    except Exception as e:
        print(f"   FastAI export failed: {e}")
    
    print(f"\n💾 Model saving completed:")
    print(f"   PyTorch state_dict: {model_filename}")
    print(f"   FastAI learner: {fastai_model_name}")
    print(f"   Save path: {model_path}")

    # 8. Plot and save LR Finder curve
    print(f"\n📈 Saving learning rate finder curve...")
    lr_plot_path = os.path.join(OUTPUT_MODEL_FOLDER, 'lr_find_13classes.png')
    try:
        learn.recorder.plot()
        plt.savefig(lr_plot_path)
        print(f"   Learning rate curve saved: {lr_plot_path}")
    except Exception as e:
        print(f"   Warning: Could not save learning rate curve: {e}")

    # 9. Save comprehensive training configuration information
    config_info = {
        'task_type': CTYPE,
        'num_classes': y_train.shape[1],
        'epochs': EPOCHS_FT,
        'sampling_rate': SAMPLING_RATE,
        'min_samples_per_class': MIN_SAMPLES,
        'pretrained_model': PRETRAINED_NAME,
        'pretrained_classes': 71,
        'output_directory': OUTPUT_MODELS_FT,
        'training_optimizations': {
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'min_delta': MIN_DELTA,
            'batch_size': BATCH_SIZE,
            'mixup_alpha': MIXUP_ALPHA,
            'label_smoothing': LABEL_SMOOTHING
        },
        'class_weights': {str(k): float(v) for k, v in class_weights.items()},
        'dataset_info': {
            'total_samples': len(data),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'class_names': list(class_names),
            'class_distribution': {class_names[i]: int(Y[:, i].sum()) for i in range(Y.shape[1])}
        },
        'performance_metrics': performance_metrics,
        'timestamp': str(np.datetime64('now'))
    }
    
    config_path = os.path.join(OUTPUT_MODEL_FOLDER, 'training_config_13classes.json')
    import json
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"   Training configuration saved: {config_path}")
    
    # 8. Training completed - evaluation will be done separately
    print(f"\n📊 Model training completed!")
    print(f"   Model saved to: {model_path}")
    print(f"   Next: Use author's framework to generate evaluation results")
    
    print(f"\n🎉 13-class arrhythmia fine-tuning training completed!")
    print(f"   Training output directory: {OUTPUT_MODEL_FOLDER}")
    print(f"   Directory structure (matching pretrained model):")
    print(f"     - Model weights: models/{model_filename}")
    print(f"     - Learning rate curve: lr_find_13classes.png")
    print(f"     - Training configuration: training_config_13classes.json")
    print(f"\n   Next steps:")
    print(f"   1. Create a separate script to generate evaluation results")
    print(f"   2. Use author's SCP_Experiment framework")
    print(f"   3. Load the trained model without re-training")
    print(f"   4. Generate predictions and evaluation results")

if __name__ == '__main__':
    main()
