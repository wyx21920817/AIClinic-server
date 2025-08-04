import numpy as np
import h5py
import wfdb
import os
from pydicom import Dataset

def extract_waveform(ds: Dataset) -> tuple[np.ndarray, int, list[str]]:
    """
    在您原始代码基础上的改进版本
    """
    ws = ds.WaveformSequence[0]
    n_channels = ws.NumberOfWaveformChannels
    n_samples = ws.NumberOfWaveformSamples
    original_sampling_rate = int(ws.SamplingFrequency)

    # 解析数据 - 这部分您的原始代码是对的
    raw = np.frombuffer(ws.WaveformData, dtype=np.int16)
    ecg_array = raw.reshape((n_samples, n_channels)).astype(np.float32)  # 转换为float32以便校准

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
        # 如果没有校准信息，使用默认值（根据您的数据是5µV/unit）
        ecg_array = ecg_array * 0.005  # 5µV/unit → mV
        print(f"⚠️ 使用默认校准 (5µV/unit): [{ecg_array.min():.3f}, {ecg_array.max():.3f}] mV")

    # ✅ 新增：重采样到400Hz（这是您原始代码缺少的）
    target_sampling_rate = 400
    if original_sampling_rate != target_sampling_rate:
        from scipy import signal
        new_length = int(n_samples * target_sampling_rate / original_sampling_rate)
        
        resampled_ecg = np.zeros((new_length, n_channels))
        for ch in range(n_channels):
            resampled_ecg[:, ch] = signal.resample(ecg_array[:, ch], new_length)
        
        ecg_array = resampled_ecg
        print(f"🔄 重采样: {original_sampling_rate}Hz → {target_sampling_rate}Hz")

    # 填充/裁剪 - 您的原始逻辑基本正确，但有小bug
    target_length = 4096
    if ecg_array.shape[0] > target_length:
        ecg_array = ecg_array[:target_length, :]  # 裁剪
        print(f"📏 裁剪到 {target_length} 点")
    elif ecg_array.shape[0] < target_length:
        padding = target_length - ecg_array.shape[0]
        # 修正：应该是 ((0, padding), (0, 0)) 而不是 ((0, 0), (0, padding))
        ecg_array = np.pad(ecg_array, ((0, padding), (0, 0)), mode='constant', constant_values=0)
        print(f"📏 填充到 {target_length} 点")

    # ✅ 关键：保持原始维度，不转置！
    # 您的原始代码在这里是对的，我之前建议转置是错误的

    # 导联名称处理 - 您的原始代码是对的
    lead_names = []
    try:
        lead_names = [ch.ChannelSourceCodeSequence[0].CodeMeaning for ch in ws.ChannelDefinitionSequence]
    except AttributeError as e:
        print("Warning: 该 DICOM 文件没有 ChannelSourceCodeSequence 属性")
        # 使用标准12导联顺序作为默认
        standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        lead_names = standard_leads[:n_channels]

    # ✅ 最终验证
    print(f"✅ 预处理完成: shape={ecg_array.shape}, 采样率={target_sampling_rate}Hz")
    return ecg_array, target_sampling_rate, lead_names


def validate_ecg_data(ecg_array: np.ndarray, sampling_rate: int) -> bool:
    """验证ECG数据质量"""
    checks = {
        "维度正确": ecg_array.shape == (4096, 12),
        "采样率正确": sampling_rate == 400,
        "数据类型正确": ecg_array.dtype in [np.float32, np.float64],
        "无异常值": not (np.isnan(ecg_array).any() or np.isinf(ecg_array).any()),
        "振幅合理": 0.1 <= np.abs(ecg_array).max() <= 10.0,  # 合理的mV范围
        "非全零": not (ecg_array == 0).all()
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if not passed:
            print(f"❌ {check_name}: {passed}")
            all_passed = False
    
    return all_passed


# 保持您的其他函数不变
def save_as_hdf5(ecg_array: np.ndarray, sampling_rate: int, out_path: str):
    with h5py.File(out_path, 'w') as f:
        f.create_dataset("ecg", data=ecg_array)
        f.attrs["sampling_rate"] = sampling_rate

def save_as_wfdb(ecg_array: np.ndarray, sampling_rate: int, lead_names: list[str], out_dir: str, record_name: str):
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
    提取 DICOM ECG 并分别保存为 HDF5 和 WFDB 格式
    Args:
        ds: DICOM Dataset
        output_base: 路径基础名（不含扩展名），如 'output/ecg1'
    """
    ecg, fs, lead_names = extract_waveform(ds)

    # 保存 HDF5
    save_as_hdf5(ecg, fs, f"{output_base}.h5")

    # 保存 WFDB
    save_as_wfdb(ecg, fs, lead_names, os.path.dirname(output_base), os.path.basename(output_base))

    return f"{output_base}.h5", f"{output_base}.hea"


def add_predictions_to_dicom(ds: Dataset, preds: dict) -> Dataset:
    """在 DICOM 中写入预测结果，存储到 PatientComments."""
    comment = "; ".join(f"{k}:{v:.3f}" for k,v in preds.items())
    ds.add_new((0x0010,0x4000), 'LT', comment)  # PatientComments 标签
    return ds
