#!/usr/bin/env python3
"""
Transfer Learning Model Fine-tuning on LTDB Data

Main features:
1. Load pre-trained XResNet1d101 model
2. Add lead adapter for 2-lead to 12-lead conversion
3. Fine-tune on LTDB data
4. Support different training strategies and hyperparameters
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from .transfer_model import create_transfer_model
from .data_ltdb import LTDBDataset, create_data_loaders
from .models.xresnet1d import xresnet1d101
from .utils.utils import set_seed, EarlyStopping


class TransferLearningTrainer:
    """Transfer Learning Trainer"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 config):
        """
        Args:
            model: Transfer learning model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup training components
        self._setup_training_components()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print(f"🚀 Transfer learning trainer initialization completed!")
        print(f"   Device: {self.device}")
        print(f"   Optimizer: {type(self.optimizer).__name__}")
        print(f"   Learning rate: {self.config['lr']}")
        print(f"   Batch size: {self.config['batch_size']}")
    
    def _setup_training_components(self):
        """Setup training components"""
        # Get trainable parameters
        trainable_params = self.model.get_trainable_parameters()
        
        # Setup optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config['lr'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self.config['lr'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # Setup learning rate scheduler
        if self.config.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Setup loss function
        if self.config['num_classes'] == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Calculate loss
            if self.config['num_classes'] == 1:
                target = target.float().unsqueeze(1)
                loss = self.criterion(output, target)
            else:
                loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            if self.config['num_classes'] == 1:
                pred = (torch.sigmoid(output) > 0.5).float()
            else:
                pred = output.argmax(dim=1)
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}, Accuracy={100.*correct/total:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                if self.config['num_classes'] == 1:
                    target = target.float().unsqueeze(1)
                    loss = self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                if self.config['num_classes'] == 1:
                    pred = (torch.sigmoid(output) > 0.5).float()
                else:
                    pred = output.argmax(dim=1)
                
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Complete training process"""
        print(f"🎯 Starting transfer learning training...")
        print(f"   Total epochs: {self.config['epochs']}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
        print(f"   Test samples: {len(self.test_loader.dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            print(f"\n📚 Epoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"📊 Training results:")
            print(f"   Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.2f}%")
            print(f"   Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}%")
            print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_best_model()
                print(f"💾 New best model saved!")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"⏹️ Early stopping triggered, training finished")
                break
        
        print(f"\n🎉 Training completed!")
        self.plot_training_history()
        self.evaluate_on_test_set()
    
    def save_best_model(self):
        """Save best model"""
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full model
        model_path = save_dir / "best_transfer_model.pth"
        self.model.save_model(str(model_path))
        
        # Save training history
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"💾 Model saved to: {model_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(self.history['train_loss'], label='Training loss')
        ax1.plot(self.history['val_loss'], label='Validation loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(self.history['train_acc'], label='Training accuracy')
        ax2.plot(self.history['val_acc'], label='Validation accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save image
        save_dir = Path(self.config['save_dir'])
        plot_path = save_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training history plot saved to: {plot_path}")
    
    def evaluate_on_test_set(self):
        """Evaluate model on test set"""
        print(f"\n🧪 Evaluating model on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                if self.config['num_classes'] == 1:
                    target = target.float().unsqueeze(1)
                    loss = self.criterion(output, target)
                    pred = (torch.sigmoid(output) > 0.5).float()
                else:
                    loss = self.criterion(output, target)
                    pred = output.argmax(dim=1)
                
                total_loss += loss.item()
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        
        if self.config['num_classes'] == 1:
            # Binary classification
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(all_targets, all_predictions)
            ap = average_precision_score(all_targets, all_predictions)
            
            print(f"📊 Test set results:")
            print(f"   Average loss: {avg_loss:.4f}")
            print(f"   AUC: {auc:.4f}")
            print(f"   AP: {ap:.4f}")
        else:
            # Multi-class classification
            accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
            
            print(f"📊 Test set results:")
            print(f"   Average loss: {avg_loss:.4f}")
            print(f"   Accuracy: {accuracy:.2f}%")
            
            # Classification report
            print(f"\n📋 Detailed classification report:")
            print(classification_report(all_targets, all_predictions))
            
            # Confusion matrix
            cm = confusion_matrix(all_targets, all_predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            save_dir = Path(self.config['save_dir'])
            cm_path = save_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Confusion matrix saved to: {cm_path}")


def main():
    """Main function"""
    print("🚀 Starting ECG transfer learning fine-tuning...")
    
    # Set random seed
    set_seed(42)
    
    # Configure parameters
    config = {
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-4,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'use_scheduler': True,
        'patience': 15,
        'min_delta': 0.001,
        'num_classes': 5,
        'save_dir': 'output/transfer_learning',
        'pretrained_model': 'output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth',
        'adapter_type': 'basic',
        'freeze_backbone': True,
        'freeze_stages': [0, 1]  # Unfreeze first two stages
    }
    
    # Check pretrained model
    if not os.path.exists(config['pretrained_model']):
        print(f"❌ Pretrained model file does not exist: {config['pretrained_model']}")
        print("Please run basic model training and fine-tuning first")
        return
    
    # Load data
    print("📥 Loading LTDB data...")
    try:
        train_data, val_data, test_data = create_data_loaders(
            config['batch_size'],
            config['num_classes']
        )
        
        # Create datasets
        train_dataset = LTDBDataset(train_data, mode='train')
        val_dataset = LTDBDataset(val_data, mode='val')
        test_dataset = LTDBDataset(test_data, mode='test')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=4
        )
        
        print(f"✅ Data loading completed!")
        print(f"   Training set: {len(train_dataset)} samples")
        print(f"   Validation set: {len(val_dataset)} samples")
        print(f"   Test set: {len(test_dataset)} samples")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Create transfer learning model
    print("🔧 Creating transfer learning model...")
    try:
        model = create_transfer_model(
            backbone_path=config['pretrained_model'],
            num_classes=config['num_classes'],
            adapter_type=config['adapter_type'],
            freeze_backbone=config['freeze_backbone'],
            freeze_stages=config['freeze_stages']
        )
        print(f"✅ Transfer learning model created successfully!")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = TransferLearningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # Start training
    trainer.train()
    
    print(f"\n🎉 Transfer learning fine-tuning completed!")
    print(f"   Results saved to: {config['save_dir']}")


if __name__ == "__main__":
    main() 