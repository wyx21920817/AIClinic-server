#!/usr/bin/env python3
"""
Verify the quality of processed LTDB data
Check data shape, statistics, visualization, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def verify_ltdb_data():
    """Verify processed LTDB data"""
    data_dir = Path("data/ltdb_processed")
    
    print("🔍 Verifying processed LTDB data...")
    
    # Load metadata
    with open(data_dir / "ltdb_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Metadata:")
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    # Load training, validation, test data
    print(f"\n📁 Loading datasets...")
    train_data = np.load(data_dir / "ltdb_train.npy")
    val_data = np.load(data_dir / "ltdb_val.npy")
    test_data = np.load(data_dir / "ltdb_test.npy")
    
    print(f"📊 Data shapes:")
    print(f"   Training set: {train_data.shape}")
    print(f"   Validation set: {val_data.shape}")
    print(f"   Test set: {test_data.shape}")
    
    # Check data statistics
    print(f"\n📈 Data statistics:")
    for name, data in [("Training set", train_data), ("Validation set", val_data), ("Test set", test_data)]:
        print(f"   {name}:")
        print(f"     Mean: {data.mean():.4f}")
        print(f"     Std: {data.std():.4f}")
        print(f"     Min: {data.min():.4f}")
        print(f"     Max: {data.max():.4f}")
    
    # Check data range (should be within reasonable range after normalization)
    print(f"\n✅ Data quality check:")
    
    # Check for NaN or infinity
    has_nan = np.isnan(train_data).any() or np.isnan(val_data).any() or np.isnan(test_data).any()
    has_inf = np.isinf(train_data).any() or np.isinf(val_data).any() or np.isinf(test_data).any()
    
    print(f"   Contains NaN: {'❌' if has_nan else '✅'}")
    print(f"   Contains infinity: {'❌' if has_inf else '✅'}")
    
    # Check normalization effect (mean should be close to 0, std close to 1)
    train_mean = train_data.mean()
    train_std = train_data.std()
    
    mean_ok = abs(train_mean) < 0.1
    std_ok = 0.8 < train_std < 1.2
    
    print(f"   Mean close to 0 (|{train_mean:.4f}| < 0.1): {'✅' if mean_ok else '❌'}")
    print(f"   Std close to 1 ({train_std:.4f}): {'✅' if std_ok else '❌'}")
    
    # Check sampling frequency
    expected_samples = metadata['target_fs'] * metadata['win_sec']
    actual_samples = train_data.shape[2]
    
    samples_ok = actual_samples == expected_samples
    print(f"   Sample count correct ({actual_samples} = {expected_samples}): {'✅' if samples_ok else '❌'}")
    
    # Check number of leads
    expected_leads = 2  # We standardize to 2 leads
    actual_leads = train_data.shape[1]
    
    leads_ok = actual_leads == expected_leads
    print(f"   Lead count correct ({actual_leads} = {expected_leads}): {'✅' if leads_ok else '❌'}")
    
    # Visualize some samples
    print(f"\n🎨 Visualizing samples...")
    try:
        # Select some samples for visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('LTDB Processed ECG Samples (2 leads)', fontsize=16)
        
        # Training set samples
        for i in range(3):
            sample_idx = i * 1000  # Select different samples
            if sample_idx < train_data.shape[0]:
                axes[0, i].plot(train_data[sample_idx, 0, :], label='ECG1', alpha=0.8)
                axes[0, i].plot(train_data[sample_idx, 1, :], label='ECG2', alpha=0.8)
                axes[0, i].set_title(f'Training Sample {sample_idx}')
                axes[0, i].set_xlabel('Sample Points')
                axes[0, i].set_ylabel('Amplitude')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
        
        # Validation set samples
        for i in range(3):
            sample_idx = i * 1000
            if sample_idx < val_data.shape[0]:
                axes[1, i].plot(val_data[sample_idx, 0, :], label='ECG1', alpha=0.8)
                axes[1, i].plot(val_data[sample_idx, 1, :], label='ECG2', alpha=0.8)
                axes[1, i].set_title(f'Validation Sample {sample_idx}')
                axes[1, i].set_xlabel('Sample Points')
                axes[1, i].set_ylabel('Amplitude')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ltdb_samples_visualization.png', dpi=150, bbox_inches='tight')
        print(f"   💾 Visualization saved as: ltdb_samples_visualization.png")
        
    except Exception as e:
        print(f"   ⚠️ Visualization failed: {e}")
    
    print(f"\n🎉 Data verification completed!")
    
    # Summary
    all_checks = [not has_nan, not has_inf, mean_ok, std_ok, samples_ok, leads_ok]
    passed_checks = sum(all_checks)
    total_checks = len(all_checks)
    
    print(f"\n📊 Verification results: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("✅ All checks passed! Data quality is good, ready for next step.")
    else:
        print("⚠️ Some checks failed, please check data processing procedure.")

if __name__ == "__main__":
    verify_ltdb_data() 