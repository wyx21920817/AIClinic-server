#!/usr/bin/env python3
"""
ECG Transfer Learning Main Script

Usage:
python run_transfer_learning.py --config basic
python run_transfer_learning.py --config progressive --epochs 100
python run_transfer_learning.py --config quick_test --pretrained_model path/to/model.pth
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code.configs.transfer_learning_configs import get_config, list_available_configs, print_config_summary
from code.finetune_with_transfer import TransferLearningTrainer
from code.transfer_model import create_transfer_model
from code.data_ltdb import LTDBDataset, create_data_loaders
from code.utils.utils import set_seed


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ECG Transfer Learning Training')
    
    parser.add_argument('--config', type=str, default='basic',
                       choices=list_available_configs(),
                       help='Training configuration name')
    
    parser.add_argument('--pretrained_model', type=str,
                       default='output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth',
                       help='Path to pretrained subclass model (71-class classification)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs in configuration')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size in configuration')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate in configuration')
    
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Override save directory in configuration')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Computing device')
    
    parser.add_argument('--dry_run', action='store_true',
                       help='Only show configuration, no actual training')
    
    return parser.parse_args()


def check_environment():
    """Check running environment"""
    print("🔍 Checking running environment...")
    
    # Check PyTorch version
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"   CUDA available: Yes")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"   CUDA available: No")
    
    # Check project structure
    required_files = [
        'code/transfer_model.py',
        'code/finetune_with_transfer.py',
        'code/data_ltdb.py',
        'code/configs/transfer_learning_configs.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        return False
    
    print(f"✅ Environment check passed!")
    return True


def check_pretrained_model(model_path):
    """Check pretrained subclass model"""
    print(f"🔍 Checking pretrained subclass model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Pretrained subclass model file does not exist: {model_path}")
        print(f"Please ensure the subclass model file exists before running transfer learning")
        print(f"Expected path: output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth")
        return False
    
    # Try to load model with different strategies
    try:
        # First try: weights_only=True (safe loading)
        print("   Attempting safe model loading...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        print(f"✅ Pretrained subclass model loaded successfully with safe loading!")
        
    except Exception as e1:
        print(f"   Safe loading failed: {e1}")
        try:
            # Second try: weights_only=False (legacy loading)
            print("   Attempting legacy model loading...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✅ Pretrained subclass model loaded successfully with legacy loading!")
            
        except Exception as e2:
            print(f"   Legacy loading also failed: {e2}")
            print(f"❌ Failed to load pretrained subclass model with both methods")
            return False
    
    # Display model information
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"   Checkpoint type: Model checkpoint with 'model' key")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"   Checkpoint type: Model checkpoint with 'state_dict' key")
    else:
        state_dict = checkpoint
        print(f"   Checkpoint type: Direct state dict")
    
    print(f"   Model parameter count: {len(state_dict)}")
    print(f"   Model size: {sum(p.numel() for p in state_dict.values()):,} parameters")
    
    # Check if this is a subclass model (should have 71+1=72 output classes)
    print(f"   Model type: Subclass classification model (71 disease classes + normal)")
    
    return True


def load_data(config):
    """Load data"""
    print("📥 Loading LTDB data...")
    
    try:
        # Set data directory
        data_dir = "data/ltdb_processed"
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"❌ Data directory does not exist: {data_dir}")
            print("Please run data preprocessing first: python code/data_ltdb.py")
            return None, None, None
        
        # Check if required data files exist
        required_files = ['ltdb_train.npy', 'ltdb_val.npy', 'ltdb_test.npy']
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing required data files: {missing_files}")
            print(f"Please ensure all required data files exist in: {data_dir}")
            return None, None, None
        
        print(f"   Data files found: {required_files}")
        
        # Get path to pre-trained model's StandardScaler
        scaler_path = "output/exp0/data/standard_scaler.pkl"
        if os.path.exists(scaler_path):
            print(f"   Using pre-trained model's StandardScaler: {scaler_path}")
        else:
            print(f"⚠️ Pre-trained StandardScaler not found: {scaler_path}")
            print(f"   Will use Z-score normalization instead")
            scaler_path = None
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=config['batch_size'],
            num_workers=0,  # Set to 0 on Windows to avoid issues
            scaler_path=scaler_path
        )
        
        print(f"✅ Data loading completed!")
        print(f"   Training set: {len(train_loader.dataset)} samples")
        print(f"   Validation set: {len(val_loader.dataset)} samples")
        print(f"   Test set: {len(test_loader.dataset)} samples")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_model(config, pretrained_model_path):
    """Create transfer learning model based on subclass model"""
    print("🔧 Creating transfer learning model based on subclass model...")
    
    try:
        model = create_transfer_model(
            backbone_path=pretrained_model_path,
            num_classes=config['num_classes'],
            adapter_type=config['adapter_type'],
            freeze_backbone=config['freeze_backbone'],
            freeze_stages=config['freeze_stages']
        )
        print(f"✅ Transfer learning model created successfully!")
        print(f"   Base model: Subclass classification model (71+1 classes)")
        print(f"   Target classes: {config['num_classes']} classes")
        print(f"   Adapter type: {config['adapter_type']}")
        print(f"   Freeze backbone: {config['freeze_backbone']}")
        print(f"   Freeze stages: {config['freeze_stages']}")
        
        # Verify model has required methods
        if not hasattr(model, 'count_parameters'):
            print(f"❌ Model missing required method: count_parameters")
            return None
            
        # Display model information
        try:
            param_counts = model.count_parameters()
            print(f"📊 Model parameter statistics:")
            print(f"   Total parameters: {param_counts['total']:,}")
            print(f"   Trainable parameters: {param_counts['trainable']:,}")
            print(f"   Frozen parameters: {param_counts['frozen']:,}")
            print(f"   Adapter parameters: {param_counts['adapter']:,}")
            print(f"   Backbone parameters: {param_counts['backbone']:,}")
        except Exception as e:
            print(f"⚠️ Warning: Could not get parameter statistics: {e}")
            print(f"   Continuing without parameter count...")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_model_save_path(config, config_name):
    """Generate clear model save path"""
    # Create timestamp for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base directory
    base_dir = "output/transfer_learning"
    
    # Configuration-specific directory
    config_dir = f"{config_name}_{timestamp}"
    
    # Full save directory
    save_dir = os.path.join(base_dir, config_dir)
    
    # Model file names
    model_files = {
        'best_model': f"best_transfer_model_{config_name}.pth",
        'final_model': f"final_transfer_model_{config_name}.pth",
        'checkpoint': f"checkpoint_{config_name}.pth"
    }
    
    return save_dir, model_files


def main():
    """Main function"""
    print("🚀 Starting ECG Transfer Learning Training...")
    print("=" * 60)
    print("📋 Model Information:")
    print("   Base model: Subclass classification model (71 disease classes + normal)")
    print("   Target: Transfer learning for LTDB data with lead adapter")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check environment
    if not check_environment():
        return
    
    # Get configuration
    try:
        config = get_config(args.config)
        print(f"📋 Using configuration: {config['name']}")
        print_config_summary(args.config)
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return
    
    # Override configuration parameters
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir
    
    # Generate clear save path
    save_dir, model_files = get_model_save_path(config, args.config)
    config['save_dir'] = save_dir
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    print(f"💾 Model save directory: {save_dir}")
    print(f"📁 Model files will be saved as:")
    for key, filename in model_files.items():
        print(f"   {key}: {filename}")
    
    # Check pretrained model
    if not check_pretrained_model(args.pretrained_model):
        return
    
    # If dry run, only show configuration
    if args.dry_run:
        print(f"\n🔍 Dry run mode - Configuration preview:")
        print(f"   Configuration name: {config['name']}")
        print(f"   Pretrained model: {args.pretrained_model}")
        print(f"   Model type: Subclass classification (71+1 classes)")
        print(f"   Training epochs: {config['epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Learning rate: {config['lr']}")
        print(f"   Save directory: {save_dir}")
        print(f"   Device: {device}")
        return
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config)
    if train_loader is None:
        return
    
    # Create model
    model = create_model(config, args.pretrained_model)
    if model is None:
        return
    
    # Create trainer
    print("🎯 Creating trainer...")
    try:
        trainer = TransferLearningTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config
        )
        print(f"✅ Trainer created successfully!")
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate training setup
    print("🔍 Validating training setup...")
    try:
        # Check if model can handle the data
        sample_batch = next(iter(train_loader))
        sample_data, sample_target = sample_batch
        print(f"   Sample data shape: {sample_data.shape}")
        print(f"   Sample target shape: {sample_target.shape}")
        print(f"   Data type: {sample_data.dtype}")
        print(f"   Target type: {sample_target.dtype}")
        
        # Test forward pass
        with torch.no_grad():
            sample_output = model(sample_data)
            print(f"   Model output shape: {sample_output.shape}")
            print(f"   Expected output shape: [batch_size, {config['num_classes']}]")
        
        print(f"✅ Training setup validation passed!")
        
    except Exception as e:
        print(f"❌ Training setup validation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start training
    print(f"\n🎯 Starting transfer learning training...")
    print(f"   Configuration: {config['name']}")
    print(f"   Base model: Subclass classification (71+1 classes)")
    print(f"   Pretrained model: {args.pretrained_model}")
    print(f"   Training epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['lr']}")
    print(f"   Save directory: {save_dir}")
    print("=" * 60)
    
    try:
        trainer.train()
        print(f"\n🎉 Transfer learning training completed!")
        print(f"   Results saved to: {save_dir}")
        print(f"   Best model: {os.path.join(save_dir, model_files['best_model'])}")
        print(f"   Final model: {os.path.join(save_dir, model_files['final_model'])}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 