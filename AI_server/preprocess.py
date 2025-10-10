# preprocess.py
import numpy as np
import h5py
import wfdb
import os
from pydicom import Dataset
from typing import Tuple, List

from scipy.signal import firwin, filtfilt, resample

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

def extract_waveform_raw(ds: Dataset) -> Tuple[np.ndarray, int, List[str]]:
    """
    return: (T, C) mV, fs, lead_names
    """
    ws = ds.WaveformSequence[0]
    fs = int(ws.SamplingFrequency)
    n_channels = int(ws.NumberOfWaveformChannels)
    n_samples = int(ws.NumberOfWaveformSamples)
    raw = np.frombuffer(ws.WaveformData, dtype=np.int16).reshape(n_samples, n_channels).astype(np.float32)

    if hasattr(ws, "ChannelDefinitionSequence"):
        for ch, ch_def in enumerate(ws.ChannelDefinitionSequence):
            if hasattr(ch_def, "ChannelSensitivity"):
                sens_uv_per_lsb = float(ch_def.ChannelSensitivity)
                raw[:, ch] = raw[:, ch] * sens_uv_per_lsb / 1000.0
    else:
        raw = raw * 0.005  # fallback: 5 μV/LSB

    try:
        lead_names = [ch.ChannelSourceCodeSequence[0].CodeMeaning for ch in ws.ChannelDefinitionSequence]
    except Exception:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"][:n_channels]

    return raw, fs, lead_names


def _fir_bandpass(sig: np.ndarray, fs: int, low: float = 0.05, high: float = 47.0) -> np.ndarray:
    """
    sig: (T, C)
    """
    x = np.asarray(sig, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    T, C = x.shape
    if T < 10:
        return x

    nyq = fs / 2.0
    high = min(high, nyq * 0.98)

    target_taps = int(3 * fs)
    max_taps = max(5, (T // 3) - 1)        # padlen=3*(taps-1) < T
    numtaps = min(target_taps, max_taps)
    if numtaps % 2 == 0:
        numtaps -= 1
    numtaps = max(numtaps, 5)

    b = firwin(numtaps, [low/nyq, high/nyq], pass_zero=False)

    y = np.empty_like(x, dtype=np.float32)
    default_padlen = 3 * (len(b) - 1)
    padlen = min(default_padlen, T - 1) if T > 1 else 0

    for c in range(C):
        if T <= 3:
            y[:, c] = x[:, c]
        else:
            y[:, c] = filtfilt(b, [1.0], x[:, c], padlen=padlen)
    return y


def _resample_to_100hz(sig: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
    target_fs = 100
    if fs == target_fs:
        return sig.astype(np.float32), target_fs
    T, C = sig.shape
    new_len = int(round(T * target_fs / fs))
    out = np.zeros((new_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = resample(sig[:, c], new_len)
    return out, target_fs


def _scale_to_unit(sig: np.ndarray) -> np.ndarray:
    sig = sig - np.median(sig, axis=0, keepdims=True)
    m = np.max(np.abs(sig))
    if m > 0:
        sig = sig / m
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def _crop_or_pad_5s(sig: np.ndarray, fs: int) -> np.ndarray:
    target_len = 5 * fs
    T, C = sig.shape
    if T == target_len:
        return sig
    if T > target_len:
        start = (T - target_len) // 2
        return sig[start:start+target_len, :]
    # pad
    pad_front = (target_len - T) // 2
    pad_back = target_len - T - pad_front
    return np.pad(sig, ((pad_front, pad_back), (0, 0)), mode="constant")


def preprocess_12lead_to_flat(sig_mv: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
    bp = _fir_bandpass(sig_mv, fs, low=0.05, high=47.0)
    r100, fs100 = _resample_to_100hz(bp, fs)
    unit = _scale_to_unit(r100)
    sec5 = _crop_or_pad_5s(unit, fs100)
    flat = sec5.reshape(-1).astype(np.float32)
    return flat, fs100
