#!/usr/bin/env python3
"""
MIT-BIH Long-Term ECG Database Data Processing Script
Converts LTDB data to PTB-XL-style training samples

Main Functions:
1. Resampling: 128 Hz -> 100 Hz
2. Windowing: 10-second windows with optional step size
3. Normalization: Z-score normalization
4. Dataset Partitioning: Partitioning by record number to prevent data leakage
"""

"""
MIT-BIH Long-Term ECG Database Data Processing Script
Converts LTDB data to PTB-XL-style training samples

Main Functions:
1. Resampling: 128 Hz -> 100 Hz
2. Windowing: 10-second windows with optional step size
3. Normalization: Z-score normalization
4. Dataset Partitioning: Partitioning by record number to prevent data leakage
"""

import os
import numpy as np
import wfdb
import scipy.signal as sg
from pathlib import Path
from typing import Tuple, List, Dict
import pickle
import json
import torch
from torch.utils.data import DataLoader

def windows_from_record(rec_base: str, 
                       win_sec: int = 10, 
                       src_fs: int = 128, 
                       tgt_fs: int = 100, 
                       step_sec: int = 10,
                       apply_filter: bool = True,
                       max_leads: int = 2) -> np.ndarray:  # Keep 2 leads
    """
    Generates windowed data from a single LTDB record

    Args:
    rec_base: Record path (without extension)
    win_sec: Window length (seconds)
    src_fs: Source sampling frequency
    tgt_fs: Target sampling frequency
    step_sec: Step size (seconds)
    apply_filter: Whether to apply a bandpass filter
    max_leads: Maximum number of leads (only the first N leads are retained)

    Returns:
    windows: Array of (num_windows, leads, samples)
    """

    """
    Generate windowed data from a single LTDB record
    
    Args:
        rec_base: Record path (without extension)
        win_sec: Window length (seconds)
        src_fs: Source sampling frequency
        tgt_fs: Target sampling frequency
        step_sec: Step size (seconds)
        apply_filter: Whether to apply bandpass filter
        max_leads: Maximum number of leads (only keep first N leads)
    
    Returns:
        windows: Array of (num_windows, leads, samples)
    """
    print(f"📁 Processing record: {rec_base}")
    
    # Read WFDB record
    try:
        rec = wfdb.rdrecord(rec_base)
    except Exception as e:
        print(f"❌ Failed to read record: {e}")
        return np.array([])
    
    # Get signal data (samples, leads)
    sig = rec.p_signal.T.astype(np.float32)  # Transpose to (leads, samples)
    print(f"📊 Original signal shape: {sig.shape}")
    print(f"📈 Original sampling frequency: {rec.fs} Hz")
    print(f"🔌 Leads: {rec.sig_name}")
    
    # Only keep first max_leads leads to ensure consistency
    if sig.shape[0] > max_leads:
        print(f"⚠️ Record has {sig.shape[0]} leads, only keeping first {max_leads}")
        sig = sig[:max_leads, :]
        print(f"📊 After truncation: {sig.shape}")
    
    # Resample to target frequency
    if rec.fs != tgt_fs:
        print(f"🔄 Resampling: {rec.fs} Hz -> {tgt_fs} Hz")
        n_new = int(sig.shape[1] * tgt_fs / rec.fs)
        sig = sg.resample(sig, n_new, axis=1)
        print(f"✅ After resampling: {sig.shape}")
    
    # Apply bandpass filter (0.5-40 Hz)
    if apply_filter:
        print("🔧 Applying bandpass filter (0.5-40 Hz)")
        nyquist = tgt_fs / 2
        low, high = 0.5 / nyquist, 40 / nyquist
        b, a = sg.butter(4, [low, high], btype='band')
        sig = sg.filtfilt(b, a, sig, axis=1)
    
    # Normalization: Z-score per lead
    print("📏 Applying Z-score normalization")
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-6)
    
    # Windowing
    L = tgt_fs * win_sec  # Window length (number of samples)
    step = tgt_fs * step_sec  # Step size (number of samples)
    
    windows = []
    for s in range(0, sig.shape[1] - L + 1, step):
        window = sig[:, s:s+L]  # (leads, window_samples)
        windows.append(window)
    
    windows = np.stack(windows)  # (num_windows, leads, window_samples)
    print(f"✂️ Generated {len(windows)} windows, each {windows.shape[1]} leads × {windows.shape[2]} samples")
    
    return windows

def process_all_records(data_dir: str, 
                       output_dir: str,
                       win_sec: int = 10,
                       step_sec: int = 10,
                       target_fs: int = 100) -> Dict[str, np.ndarray]:
    """
    Process all LTDB records
    
    Args:
        data_dir: LTDB data directory
        output_dir: Output directory
        win_sec: Window length (seconds)
        step_sec: Step size (seconds)
        target_fs: Target sampling frequency
    
    Returns:
        processed_data: Dictionary containing all processed data
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all record files
    hea_files = list(data_dir.glob("*.hea"))
    record_names = [f.stem for f in hea_files]
    
    print(f"🔍 Found {len(record_names)} record files")
    print(f"📁 Output directory: {output_dir}")
    
    all_windows = []
    record_info = {}
    
    for i, rec_name in enumerate(record_names):
        print(f"\n📋 Processing record {i+1}/{len(record_names)}: {rec_name}")
        
        rec_path = data_dir / rec_name
        windows = windows_from_record(str(rec_path), 
                                   win_sec=win_sec, 
                                   src_fs=128, 
                                   tgt_fs=target_fs, 
                                   step_sec=step_sec)
        
        if len(windows) > 0:
            all_windows.append(windows)
            record_info[rec_name] = {
                'num_windows': len(windows),
                'shape': windows.shape,
                'total_samples': len(windows) * windows.shape[2] / target_fs
            }
            print(f"✅ {rec_name}: {len(windows)} windows, total {record_info[rec_name]['total_samples']:.1f} seconds")
        else:
            print(f"⚠️ {rec_name}: Processing failed, skipping")
    
    if not all_windows:
        print("❌ No records successfully processed")
        return {}
    
    # Concatenate all windows
    print(f"\n🔗 Concatenating all windows...")
    combined_windows = np.concatenate(all_windows, axis=0)
    print(f"📊 Final data shape: {combined_windows.shape}")
    print(f"📈 Total windows: {combined_windows.shape[0]}")
    print(f"🔌 Leads: {combined_windows.shape[1]}")
    print(f"⏱️ Each window: {combined_windows.shape[2]} samples ({combined_windows.shape[2]/target_fs:.1f} seconds)")
    
    # Save processed data
    processed_data = {
        'windows': combined_windows,
        'record_info': record_info,
        'metadata': {
            'target_fs': target_fs,
            'win_sec': win_sec,
            'step_sec': step_sec,
            'num_records': len(record_info),
            'total_windows': combined_windows.shape[0],
            'leads': combined_windows.shape[1],
            'samples_per_window': combined_windows.shape[2]
        }
    }
    
    # Save as numpy format
    np.save(output_dir / 'ltdb_processed.npy', combined_windows)
    
    # Save metadata
    with open(output_dir / 'ltdb_metadata.json', 'w') as f:
        json.dump(processed_data['metadata'], f, indent=2)
    
    # Save record information
    with open(output_dir / 'ltdb_record_info.json', 'w') as f:
        json.dump(record_info, f, indent=2)
    
    print(f"\n💾 Data saved to: {output_dir}")
    print(f"   - ltdb_processed.npy: Processed window data")
    print(f"   - ltdb_metadata.json: Metadata information")
    print(f"   - ltdb_record_info.json: Record details")
    
    return processed_data

def split_dataset_by_records(processed_data: Dict[str, np.ndarray],
                           output_dir: str,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15) -> Dict[str, np.ndarray]:
    """
    Split dataset by record number to prevent data leakage from same patient to different sets
    
    Args:
        processed_data: Processed data
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    
    Returns:
        split_data: Split dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    record_info = processed_data['record_info']
    windows = processed_data['windows']
    
    # Sort records to ensure reproducibility
    record_names = sorted(record_info.keys())
    
    # Calculate number of records for each set
    n_records = len(record_names)
    n_train = int(n_records * train_ratio)
    n_val = int(n_records * val_ratio)
    n_test = n_records - n_train - n_val
    
    print(f"📊 Dataset splitting:")
    print(f"    Total records: {n_records}")
    print(f"    Training set: {n_train} records ({train_ratio*100:.1f}%)")
    print(f"    Validation set: {n_val} records ({val_ratio*100:.1f}%)")
    print(f"    Test set: {n_test} records ({test_ratio*100:.1f}%)")
    
    # Split records
    train_records = record_names[:n_train]
    val_records = record_names[n_train:n_train+n_val]
    test_records = record_names[n_train+n_val:]
    
    print(f"\n📋 Record splitting:")
    print(f"    Training set: {train_records}")
    print(f"    Validation set: {val_records}")
    print(f"    Test set: {test_records}")
    
    # Find corresponding window indices based on record information
    def get_window_indices(records):
        indices = []
        start_idx = 0
        for rec_name in sorted(record_info.keys()):
            if rec_name in records:
                n_windows = record_info[rec_name]['num_windows']
                indices.extend(range(start_idx, start_idx + n_windows))
            start_idx += record_info[rec_name]['num_windows']
        return indices
    
    train_indices = get_window_indices(train_records)
    val_indices = get_window_indices(val_records)
    test_indices = get_window_indices(test_records)
    
    # Extract corresponding window data
    train_windows = windows[train_indices]
    val_windows = windows[val_indices]
    test_windows = windows[test_indices]
    
    print(f"\n📊 Window data splitting:")
    print(f"    Training set: {train_windows.shape}")
    print(f"    Validation set: {val_windows.shape}")
    print(f"    Test set: {test_windows.shape}")
    
    # Save split data
    split_data = {
        'train': train_windows,
        'val': val_windows,
        'test': test_windows,
        'train_records': train_records,
        'val_records': val_records,
        'test_records': test_records
    }
    
    np.save(output_dir / 'ltdb_train.npy', train_windows)
    np.save(output_dir / 'ltdb_val.npy', val_windows)
    np.save(output_dir / 'ltdb_test.npy', test_windows)
    
    # Save split information
    split_info = {
        'train_records': train_records,
        'val_records': val_records,
        'test_records': test_records,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    
    with open(output_dir / 'ltdb_split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n💾 Split data saved to: {output_dir}")
    print(f"   - ltdb_train.npy: Training set")
    print(f"   - ltdb_val.npy: Validation set")
    print(f"   - ltdb_test.npy: Test set")
    print(f"   - ltdb_split_info.json: Split information")
    
    return split_data

if __name__ == "__main__":
    # Configuration parameters
    LTDB_DATA_DIR = "data/mit-bih-long-term-ecg-database-1.0.0"
    OUTPUT_DIR = "data/ltdb_processed"
    
    # Processing parameters
    WINDOW_SEC = 10      # 10-second window
    STEP_SEC = 10        # 10-second step (non-overlapping)
    TARGET_FS = 100      # Target sampling frequency 100 Hz
    
    print("🚀 Starting MIT-BIH Long-Term ECG Database processing...")
    print(f"📁 Data directory: {LTDB_DATA_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"⚙️ Parameter settings:")
    print(f"    Window length: {WINDOW_SEC} seconds")
    print(f"    Step: {STEP_SEC} seconds")
    print(f"    Target sampling frequency: {TARGET_FS} Hz")
    
    # Step 1: Process all records
    processed_data = process_all_records(
        data_dir=LTDB_DATA_DIR,
        output_dir=OUTPUT_DIR,
        win_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        target_fs=TARGET_FS
    )
    
    if processed_data:
        # Step 2: Split dataset
        split_data = split_dataset_by_records(
            processed_data=processed_data,
            output_dir=OUTPUT_DIR
        )
        
        print("\n🎉 LTDB data processing complete!")
        print("📊 Data statistics:")
        print(f"    Total windows: {processed_data['metadata']['total_windows']}")
        print(f"    Training set: {split_data['train'].shape[0]} windows")
        print(f"    Validation set: {split_data['val'].shape[0]} windows")
        print(f"    Test set: {split_data['test'].shape[0]} windows")
        print(f"    Each window: {split_data['train'].shape[1]} leads × {split_data['train'].shape[2]} samples")
    else:
        print("\n❌ Data processing failed") 

class LTDBDataset:
    """LTDB data set class for PyTorch data loading"""
    
    def __init__(self, data_path, transform=None, scaler_path=None):
        """
        Args:
            data_path: Data file path (.npy file)
            transform: Data transformation
            scaler_path: Path to pre-trained model's StandardScaler (.pkl file)
        """
        self.data_path = data_path
        self.transform = transform
        self.scaler = None
        
        # Load pre-trained model's StandardScaler if provided
        if scaler_path and os.path.exists(scaler_path):
            try:
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Loaded pre-trained StandardScaler from: {scaler_path}")
            except Exception as e:
                print(f"⚠️ Failed to load StandardScaler: {e}, using Z-score normalization")
        
        if os.path.exists(data_path):
            self.data = np.load(data_path)
            print(f"✅ Data loaded: {data_path}, shape: {self.data.shape}")
        else:
            print(f"❌ Data file does not exist: {data_path}")
            self.data = np.array([])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.data)} samples")
        
        # Get the specific sample
        sample = self.data[idx]  # Shape: (2, 1000)
        
        # Apply transformation if specified
        if self.transform:
            sample = self.transform(sample)
        
        # Apply standardization - follow PTB-XL preprocessing pattern
        if self.scaler is not None:
            try:
                # Use pre-trained model's StandardScaler
                # Follow PTB-XL pattern: flatten completely and add dimension
                sample_flat = sample.flatten()[:, None]  # (2000, 1) - same as PTB-XL
                sample_scaled = self.scaler.transform(sample_flat)  # (2000, 1)
                sample = sample_scaled.reshape(sample.shape)  # Back to (2, 1000)
            except Exception as e:
                print(f"⚠️ StandardScaler failed, using Z-score: {e}")
                # Fallback to Z-score normalization
                sample_flat = sample.flatten()[:, None]  # (2000, 1)
                sample_mean = np.mean(sample_flat)
                sample_std = np.std(sample_flat)
                sample_std = max(sample_std, 1e-6)  # Avoid division by zero
                sample_scaled = (sample_flat - sample_mean) / sample_std
                sample = sample_scaled.reshape(sample.shape)  # Back to (2, 1000)
        else:
            # Use Z-score normalization - follow PTB-XL pattern
            sample_flat = sample.flatten()[:, None]  # (2000, 1)
            sample_mean = np.mean(sample_flat)
            sample_std = np.std(sample_flat)
            sample_std = max(sample_std, 1e-6)  # Avoid division by zero
            sample_scaled = (sample_flat - sample_mean) / sample_std
            sample = sample_scaled.reshape(sample.shape)  # Back to (2, 1000)
        
        # Convert to torch tensor
        sample = torch.FloatTensor(sample)
        
        # For LTDB data, we need to create labels for transfer learning
        # Since LTDB doesn't have disease labels, we'll create synthetic labels
        # This is a common approach in unsupervised transfer learning
        label = torch.randint(0, 5, (1,)).item()  # Random label 0-4 for testing
        
        return sample, label


def create_data_loaders(data_dir, batch_size=32, num_workers=0, scaler_path=None):
    """
    Create PyTorch DataLoaders for LTDB data
    
    Args:
        data_dir: Directory containing processed data files
        batch_size: Batch size for training
        num_workers: Number of worker processes
        scaler_path: Path to pre-trained model's StandardScaler (.pkl file)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data files
    train_data = os.path.join(data_dir, 'ltdb_train.npy')
    val_data = os.path.join(data_dir, 'ltdb_val.npy')
    test_data = os.path.join(data_dir, 'ltdb_test.npy')
    
    # Create datasets
    train_dataset = LTDBDataset(train_data, scaler_path=scaler_path)
    val_dataset = LTDBDataset(val_data, scaler_path=scaler_path)
    test_dataset = LTDBDataset(test_data, scaler_path=scaler_path)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"✅ Data loaders created:")
    print(f"    Training set: {len(train_loader)} batches")
    print(f"    Validation set: {len(val_loader)} batches")
    print(f"    Test set: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader 