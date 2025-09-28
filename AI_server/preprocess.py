import numpy as np
import h5py
import wfdb
import os
from pydicom import Dataset
from typing import Tuple, List

def extract_waveform(ds: Dataset) -> Tuple[np.ndarray, int, List[str]]:
    ws = ds.WaveformSequence[0]
    n_channels = ws.NumberOfWaveformChannels
    n_samples = ws.NumberOfWaveformSamples
    original_sampling_rate = int(ws.SamplingFrequency)
    raw = np.frombuffer(ws.WaveformData, dtype=np.int16)
    ecg_array = raw.reshape((n_samples, n_channels)).astype(np.float32)

    # Unit calibration ADC value → mV
    if hasattr(ws, 'ChannelDefinitionSequence'):
        # Using calibration information in DICOM
        for ch_idx, ch_def in enumerate(ws.ChannelDefinitionSequence):
            if hasattr(ch_def, 'ChannelSensitivity'):
                sensitivity = float(ch_def.ChannelSensitivity)  # µV/unit
                # Conversion: ADC value × sensitivity(µV/unit) ÷ 1000 = mV
                ecg_array[:, ch_idx] = ecg_array[:, ch_idx] * sensitivity / 1000.0
        print(f"Unit calibration completed: ADC → mV, data range: [{ecg_array.min():.3f}, {ecg_array.max():.3f}] mV")
    else:
        ecg_array = ecg_array * 0.005  # 5µV/unit → mV
        print(f"⚠️ Use default calibration (5µV/unit): [{ecg_array.min():.3f}, {ecg_array.max():.3f}] mV")

    target_sampling_rate = 400
    if original_sampling_rate != target_sampling_rate:
        from scipy import signal
        new_length = int(n_samples * target_sampling_rate / original_sampling_rate)
        
        resampled_ecg = np.zeros((new_length, n_channels))
        for ch in range(n_channels):
            resampled_ecg[:, ch] = signal.resample(ecg_array[:, ch], new_length)
        
        ecg_array = resampled_ecg
        print(f"🔄 resample: {original_sampling_rate}Hz → {target_sampling_rate}Hz")

    target_length = 4096
    if ecg_array.shape[0] > target_length:
        ecg_array = ecg_array[:target_length, :]
        print(f"📏 cropping to {target_length} point")
    elif ecg_array.shape[0] < target_length:
        padding = target_length - ecg_array.shape[0]
        ecg_array = np.pad(ecg_array, ((0, padding), (0, 0)), mode='constant', constant_values=0)
        print(f"📏 padding {target_length} point")

    lead_names = []
    try:
        lead_names = [ch.ChannelSourceCodeSequence[0].CodeMeaning for ch in ws.ChannelDefinitionSequence]
    except AttributeError as e:
        print("Warning: The DICOM file does not have a ChannelSourceCodeSequence attribute")
        standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        lead_names = standard_leads[:n_channels]

    print(f"✅ Preprocessing completed: shape={ecg_array.shape}, Sampling rate={target_sampling_rate}Hz")
    return ecg_array, target_sampling_rate, lead_names


def validate_ecg_data(ecg_array: np.ndarray, sampling_rate: int) -> bool:
    checks = {
        "Correct dimension": ecg_array.shape == (4096, 12),
        "Correct sampling rate": sampling_rate == 400,
        "Correct data type": ecg_array.dtype in [np.float32, np.float64],
        "No outliers": not (np.isnan(ecg_array).any() or np.isinf(ecg_array).any()),
        "Reasonable amplitude": 0.1 <= np.abs(ecg_array).max() <= 10.0,
        "Not all zeros": not (ecg_array == 0).all()
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if not passed:
            print(f"❌ {check_name}: {passed}")
            all_passed = False
    
    return all_passed


def save_as_hdf5(ecg_array: np.ndarray, sampling_rate: int, out_path: str):
    with h5py.File(out_path, 'w') as f:
        f.create_dataset("ecg", data=ecg_array)
        f.attrs["sampling_rate"] = sampling_rate

def save_as_wfdb(ecg_array: np.ndarray, sampling_rate: int, lead_names: List[str], out_dir: str, record_name: str):
    # Transpose to (n_samples, n_leads)
    ecg = ecg_array.T
    wfdb.wrsamp(
        record_name=record_name,
        fs=sampling_rate,
        units=['mV'] * ecg.shape[1],
        sig_name=lead_names,
        p_signal=ecg,
        fmt=['16'] * ecg.shape[1],
        write_dir=out_dir
    )

def dicom_to_formats(ds: Dataset, output_base: str):
    """
    Args:
        ds: DICOM Dataset
        output_base: The path base name (without the extension), such as'output/ecg1'
    """
    ecg, fs, lead_names = extract_waveform(ds)
    save_as_hdf5(ecg, fs, f"{output_base}.h5")
    save_as_wfdb(ecg, fs, lead_names, os.path.dirname(output_base), os.path.basename(output_base))
    return f"{output_base}.h5", f"{output_base}.hea"


def add_predictions_to_dicom(ds: Dataset, preds: dict) -> Dataset:
    """Write the prediction results in DICOM and store them in PatientComments."""
    comment = "; ".join(f"{k}:{v:.3f}" for k,v in preds.items())
    ds.add_new((0x0010,0x4000), 'LT', comment)
    return ds
