# wfdb2dcm.py
"""
Convert one PTB-XL WFDB record (.dat/.hea) to a single-file DICOM ECG Waveform.
Usage:
    python wfdb2dcm.py <record_basename> <wfdb_dir> <out_dcm>
Example:
    python wfdb2dcm.py 00001_lr ptb-xl/records500/00001/ output/00001_lr.dcm
"""

import sys, uuid, datetime
import numpy as np
import wfdb
import pydicom
import os
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

def wfdb_to_dicom(record_name: str, wfdb_dir: str, out_path: str):
    # --- 1. 读取 WFDB ---
    rec_path = os.path.join(wfdb_dir, record_name)   # → records100/00001/00001_lr
    rec = wfdb.rdrecord(rec_path)   # p_signal shape: (n_samples, n_channels)
    sig   = rec.p_signal.astype("float32")              # 单位：mV, shape (T, C)
    fs    = rec.fs
    leads = rec.sig_name                                # 导联名，如 ["I","II",...,"V6"]
    n_samp, n_chan = sig.shape

    # 这里直接把 float (mV) ×1000 → µV 再转 int16（符合 DICOM 规范，WaveformSampleInterpretation='SS'）
    sig_int16 = np.round(sig * 1000).astype("<i2")      # little-endian int16
    wf_bytes  = sig_int16.tobytes()

    # --- 2. 构建 DICOM 文件级元信息 ---
    sop_instance_uid = pydicom.uid.generate_uid()
    sop_class_uid = "1.2.840.10008.5.1.4.1.1.9.1.1"  # ECG Waveform Storage
    
    file_meta = Dataset()
    file_meta.FileMetaInformationGroupLength = 0        # pydicom 会自动计算
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.1000"

    # --- 3. 创建顶层 Dataset ---
    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # ── 必需的 DICOM 标签 ──
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    
    # ── Patient / Study（可按需替换） ──
    ds.PatientName = "Anon^PTBXL"
    ds.PatientID = "PTBXL"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyDate, ds.StudyTime = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S").split("-")
    ds.Modality = "ECG"
    ds.Manufacturer = "PTBXL-Converter"
    
    # ── 其他必需的标签 ──
    ds.StudyID = "PTBXL_STUDY"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "ECG"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15

    # --- 4. Waveform Sequence (一个 item 即一组 12-lead) ---
    wv_item = Dataset()
    wv_item.WaveformOriginality     = "ORIGINAL"
    wv_item.NumberOfWaveformChannels= n_chan
    wv_item.NumberOfWaveformSamples = n_samp
    wv_item.SamplingFrequency       = fs
    wv_item.MultiplexGroupTimeOffset= 0
    wv_item.TriggerTimeOffset       = 0
    wv_item.WaveformBitsAllocated   = 16
    wv_item.WaveformBitsStored      = 16
    wv_item.WaveformSampleInterpretation = "SS"         # signed short
    wv_item.WaveformData            = wf_bytes

    # ---- ChannelDefinitionSequence ----
    ch_seq = []
    for ch_idx, lead in enumerate(leads, 1):
        ch = Dataset()
        ch.ChannelNumber            = ch_idx
        ch.ChannelLabel             = lead
        # 可选：写 ChannelSourceCodeSequence (SNOMED for leads)
        ch_seq.append(ch)
    wv_item.ChannelDefinitionSequence = Sequence(ch_seq)

    ds.WaveformSequence = Sequence([wv_item])

    # --- 5. 存盘 ---
    # 最终验证和设置必需的标签
    if not hasattr(ds, 'SOPClassUID') or ds.SOPClassUID is None:
        ds.SOPClassUID = sop_class_uid
    if not hasattr(ds, 'SOPInstanceUID') or ds.SOPInstanceUID is None:
        ds.SOPInstanceUID = sop_instance_uid
    
    # 确保file_meta与主数据集一致
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    
    ds.save_as(out_path, write_like_original=False)
    print(f"✅  DICOM ECG saved to: {out_path}")
    print(f"   SOPClassUID: {ds.SOPClassUID}")
    print(f"   SOPInstanceUID: {ds.SOPInstanceUID}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    rec_base, wfdb_dir, out_file = sys.argv[1:]
    wfdb_to_dicom(rec_base, wfdb_dir, out_file)
