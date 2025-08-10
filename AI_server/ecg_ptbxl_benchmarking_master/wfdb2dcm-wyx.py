# wfdb2dcm_correct_official.py
"""
Correct version based on pydicom official documentation
"""
import sys, datetime
import numpy as np
import wfdb
import pydicom
import os
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

def wfdb_to_dicom_correct_official(record_name: str, wfdb_dir: str, out_path: str):
    # --- 1. 读取 WFDB ---
    rec_path = os.path.join(wfdb_dir, record_name)
    rec = wfdb.rdrecord(rec_path)
    sig = rec.p_signal.astype("float32")
    fs = rec.fs
    leads = rec.sig_name
    n_samp, n_chan = sig.shape

    print(f"📊 原始数据信息:")
    print(f"   形状: {sig.shape}")
    print(f"   采样率: {fs} Hz")
    print(f"   导联: {leads}")
    print(f"   数据范围: [{sig.min():.3f}, {sig.max():.3f}] mV")

    # --- 2. 数据转换 ---
    sig_uv = sig * 1000.0  # mV -> µV
    sig_uv = np.clip(sig_uv, -32767, 32767)
    sig_int16 = np.round(sig_uv).astype(np.int16)
    
    print(f"   转换后范围: [{sig_int16.min()}, {sig_int16.max()}] (int16)")

    # --- 3. DICOM文件元信息 ---
    sop_instance_uid = pydicom.uid.generate_uid()
    
    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.1000"

    # --- 4. 主数据集 ---
    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # SOP标签
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    ds.SOPInstanceUID = sop_instance_uid
    
    # Patient/Study信息
    ds.PatientName = "Anon^PTBXL"
    ds.PatientID = "PTBXL"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    
    now = datetime.datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    
    ds.StudyID = "1"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    
    ds.Modality = "ECG"
    ds.Manufacturer = "PTBXL-Converter"

    # --- 5. 波形序列 - 按照官方文档 ---
    wv_item = Dataset()
    
    # 基本波形信息
    wv_item.WaveformOriginality = "ORIGINAL"
    wv_item.NumberOfWaveformChannels = n_chan
    wv_item.NumberOfWaveformSamples = n_samp
    wv_item.SamplingFrequency = float(fs)
    
    # 时间相关参数
    wv_item.MultiplexGroupTimeOffset = 0.0
    wv_item.TriggerTimeOffset = 0.0
    
    # ✅ 关键：完整的波形位定义
    wv_item.WaveformBitsAllocated = 16
    wv_item.WaveformSampleInterpretation = "SS"
    
    # ✅ 关键：交错数据并转换为bytes
    # 将数据重塑为 (n_samp, n_chan) 然后转为bytes
    arr = sig_int16.reshape(n_samp, n_chan)
    wf_bytes = arr.tobytes()
    wv_item.WaveformData = wf_bytes

    # --- 6. 通道定义序列 - 按照官方文档格式 ---
    ch_seq = []
    
    # 标准导联代码映射
    lead_codes = {
        'I': ('5.6.3-9-1', 'Lead I (Einthoven)'),
        'II': ('5.6.3-9-2', 'Lead II'),
        'III': ('5.6.3-9-61', 'Lead III'),
        'AVR': ('5.6.3-9-62', 'Lead aVR'),
        'AVL': ('5.6.3-9-63', 'Lead aVL'),
        'AVF': ('5.6.3-9-64', 'Lead aVF'),
        'V1': ('5.6.3-9-3', 'Lead V1'),
        'V2': ('5.6.3-9-4', 'Lead V2'),
        'V3': ('5.6.3-9-5', 'Lead V3'),
        'V4': ('5.6.3-9-6', 'Lead V4'),
        'V5': ('5.6.3-9-7', 'Lead V5'),
        'V6': ('5.6.3-9-8', 'Lead V6')
    }
    
    for ch_idx, lead in enumerate(leads):
        chdef = Dataset()
        
        # ✅ 关键：按照官方文档，在通道定义中设置WaveformBitsStored
        chdef.ChannelSampleSkew = 0.0
        chdef.WaveformBitsStored = 16  # ✅ 这是关键！
        
        # 导联代码序列
        chdef.ChannelSourceSequence = [Dataset()]
        source = chdef.ChannelSourceSequence[0]
        
        std_lead = lead.upper()
        if std_lead in lead_codes:
            source.CodeValue = lead_codes[std_lead][0]
            source.CodingSchemeDesignator = "SCPECG"
            source.CodingSchemeVersion = "1.3"
            source.CodeMeaning = lead_codes[std_lead][1]
        else:
            source.CodeValue = "1.0"
            source.CodingSchemeDesignator = "PYDICOM"
            source.CodingSchemeVersion = "1.0"
            source.CodeMeaning = lead
        
        # 校准信息 (Type 1C - 可选的)
        chdef.ChannelSensitivity = 1.0
        chdef.ChannelSensitivityUnitsSequence = [Dataset()]
        units_item = chdef.ChannelSensitivityUnitsSequence[0]
        units_item.CodeValue = "uV"
        units_item.CodingSchemeDesignator = "UCUM"
        units_item.CodeMeaning = "microvolt"
        
        ch_seq.append(chdef)
    
    wv_item.ChannelDefinitionSequence = Sequence(ch_seq)
    ds.WaveformSequence = Sequence([wv_item])

    # --- 7. 保存并验证 ---
    try:
        ds.save_as(out_path, write_like_original=False)
        
        # 验证文件
        verify_ds = pydicom.dcmread(out_path)
        ws = verify_ds.WaveformSequence[0]
        
        print(f"✅ DICOM ECG保存成功: {out_path}")
        print(f"   ✅ WaveformBitsAllocated: {ws.WaveformBitsAllocated}")
        print(f"   ✅ WaveformSampleInterpretation: {ws.WaveformSampleInterpretation}")
        
        # 检查通道定义中的WaveformBitsStored
        ch = ws.ChannelDefinitionSequence[0]
        print(f"   ✅ 通道WaveformBitsStored: {ch.WaveformBitsStored}")
        
        # 验证数据
        expected_size = n_samp * n_chan * 2
        actual_size = len(ws.WaveformData)
        print(f"   ✅ 数据大小: {actual_size} bytes (期望: {expected_size})")
        
        if actual_size == expected_size:
            print("   ✅ 数据完整性验证通过")
        
        # 使用pydicom的官方方法验证数据解码
        try:
            decoded_arr = verify_ds.waveform_array(0)
            print(f"   ✅ 数据解码成功: {decoded_arr.shape}")
            print(f"   ✅ 解码后范围: [{decoded_arr.min():.1f}, {decoded_arr.max():.1f}]")
        except Exception as decode_error:
            print(f"   ❌ 数据解码失败: {decode_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python wfdb2dcm_correct_official.py <record_name> <wfdb_dir> <output.dcm>")
        sys.exit(1)
    
    rec_base, wfdb_dir, out_file = sys.argv[1:]
    
    print(f"🔄 官方文档标准版本")
    print(f"   转换: {rec_base} -> {out_file}")
    
    success = wfdb_to_dicom_correct_official(rec_base, wfdb_dir, out_file)
    
    if success:
        print(f"\n🎉 按pydicom官方文档转换成功！")
        print(f"💡 关键修复:")
        print(f"   • 在ChannelDefinitionSequence中设置WaveformBitsStored=16")
        print(f"   • 使用正确的数据交错格式")
        print(f"   • 遵循官方文档的完整结构")
        print(f"\n📋 测试:")
        print(f"   1. 验证ECG查看器是否还报'bits stored definition'错误")
        print(f"   2. 检查波形是否能正常显示")
    else:
        print(f"\n❌ 转换失败")
