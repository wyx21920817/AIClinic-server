import numpy as np
import h5py
import wfdb
import os
from pydicom import Dataset

def extract_waveform(ds: Dataset) -> tuple[np.ndarray, int, list[str]]:
    """
    Extract waveform data, sampling rate and lead name
    Returns:
        - ecg_array: ndarray, shape (n_leads, n_samples)
        - sampling_rate: int
        - lead_names: list of str
    """
    ws = ds.WaveformSequence[0]  # choose 1st WaveformSequence
    n_channels = ws.NumberOfWaveformChannels
    n_samples = ws.NumberOfWaveformSamples
    sampling_rate = int(ws.SamplingFrequency)

    # Assume WaveformData is stored as bytes and needs reshaping
    raw = np.frombuffer(ws.WaveformData, dtype=np.int16)  # 或检查 SampleEncoding
    ecg_array = raw.reshape((n_samples, n_channels))  # shape (n_channels, n_samples)

    # 进行裁剪或者填充，确保时间步数为 4096
    target_length = 4096
    if ecg_array.shape[0] > target_length:
        ecg_array = ecg_array[:target_length, :]  # 裁剪
    elif ecg_array.shape[0] < target_length:
        padding = target_length - ecg_array.shape[0]
        ecg_array = np.pad(ecg_array, ((0, 0), (0, padding)), mode='constant', constant_values=0)  # 填充

    # 现在检查是否存在 ChannelSourceCodeSequence
    lead_names = []
    try:
        # 检查是否存在 ChannelSourceCodeSequence
        lead_names = [ch.ChannelSourceCodeSequence[0].CodeMeaning for ch in ws.ChannelDefinitionSequence]
    except AttributeError as e:
        print("Warning: 该 DICOM 文件没有 ChannelSourceCodeSequence 属性")
        # 如果没有该属性，可以将导联名称设置为空，或根据其他信息进行处理
        lead_names = [f"Lead {i+1}" for i in range(n_channels)]  # 使用默认的 Lead 名称

    return ecg_array, sampling_rate, lead_names



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
