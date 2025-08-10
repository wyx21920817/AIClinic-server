#!/usr/bin/env python3
"""
检查DICOM文件结构差异的脚本
"""

import pydicom
import os

def check_dicom_structure(filename):
    """检查DICOM文件的结构"""
    print(f"\n=== {filename} ===")
    
    try:
        ds = pydicom.dcmread(filename, force=True)
        
        # 基本信息
        print(f"File size: {os.path.getsize(filename)} bytes")
        
        # 检查波形序列
        if hasattr(ds, 'WaveformSequence'):
            print(f"Has WaveformSequence: True")
            print(f"WaveformSequence length: {len(ds.WaveformSequence)}")
            
            if len(ds.WaveformSequence) > 0:
                wf = ds.WaveformSequence[0]
                
                # 波形数据元素
                print(f"Has WaveformData: {hasattr(wf, 'WaveformData')}")
                if hasattr(wf, 'WaveformData'):
                    print(f"WaveformData length: {len(wf.WaveformData)} bytes")
                
                # 波形参数
                print(f"NumberOfWaveformSamples: {getattr(wf, 'NumberOfWaveformSamples', 'N/A')}")
                print(f"NumberOfWaveformChannels: {getattr(wf, 'NumberOfWaveformChannels', 'N/A')}")
                
                # 其他重要元素
                print(f"WaveformBitsAllocated: {getattr(wf, 'WaveformBitsAllocated', 'N/A')}")
                print(f"WaveformSampleInterpretation: {getattr(wf, 'WaveformSampleInterpretation', 'N/A')}")
                print(f"ChannelDefinitionSequence length: {len(getattr(wf, 'ChannelDefinitionSequence', []))}")
                
                # 检查通道定义
                if hasattr(wf, 'ChannelDefinitionSequence') and len(wf.ChannelDefinitionSequence) > 0:
                    for i, ch in enumerate(wf.ChannelDefinitionSequence):
                        print(f"  Channel {i}: {getattr(ch, 'ChannelLabel', 'N/A')}")
                        print(f"    ChannelSourceSequence: {getattr(ch, 'ChannelSourceSequence', 'N/A')}")
                
                # 检查波形数据格式
                if hasattr(wf, 'WaveformData'):
                    data = wf.WaveformData
                    print(f"WaveformData type: {type(data)}")
                    if isinstance(data, bytes):
                        print(f"First 20 bytes: {data[:20].hex()}")
                        
                        # 尝试解析为数值
                        try:
                            import numpy as np
                            arr = np.frombuffer(data, dtype=np.int16)
                            print(f"Parsed as int16: {arr.size} samples")
                            if arr.size > 0:
                                print(f"First 10 values: {arr[:10]}")
                                print(f"Value range: {arr.min()} to {arr.max()}")
                        except Exception as e:
                            print(f"Error parsing data: {e}")
                
        else:
            print("Has WaveformSequence: False")
            
        # 检查其他可能相关的元素
        print(f"Has ImageSequence: {hasattr(ds, 'ImageSequence')}")
        print(f"Has CurveSequence: {hasattr(ds, 'CurveSequence')}")
        
        # 检查文件元信息
        if hasattr(ds, 'file_meta'):
            print(f"TransferSyntaxUID: {getattr(ds.file_meta, 'TransferSyntaxUID', 'N/A')}")
            print(f"MediaStorageSOPClassUID: {getattr(ds.file_meta, 'MediaStorageSOPClassUID', 'N/A')}")
        
        # 检查SOP类
        print(f"SOPClassUID: {getattr(ds, 'SOPClassUID', 'N/A')}")
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

def main():
    """主函数"""
    files_to_check = [
        'test_00045_lr.dcm',  # 用户转换的文件
        'waveform_ecg.dcm',   # 标准波形DICOM
        'image-00000.dcm'     # 标准图像DICOM
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            check_dicom_structure(filename)
        else:
            print(f"\n=== {filename} ===")
            print("File not found")

if __name__ == "__main__":
    main() 