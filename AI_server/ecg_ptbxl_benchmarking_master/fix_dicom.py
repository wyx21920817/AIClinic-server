#!/usr/bin/env python3
"""
修复现有DICOM文件的脚本，添加缺失的通道源信息
"""

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

def fix_dicom_channels(input_file, output_file):
    """修复DICOM文件的通道定义"""
    print(f"🔧 修复DICOM文件: {input_file}")
    
    # 读取DICOM文件
    ds = pydicom.dcmread(input_file, force=True)
    
    if not hasattr(ds, 'WaveformSequence') or len(ds.WaveformSequence) == 0:
        print("❌ 文件中没有波形序列")
        return False
    
    wf = ds.WaveformSequence[0]
    
    if not hasattr(wf, 'ChannelDefinitionSequence') or len(wf.ChannelDefinitionSequence) == 0:
        print("❌ 文件中没有通道定义序列")
        return False
    
    print(f"📊 发现 {len(wf.ChannelDefinitionSequence)} 个通道")
    
    # 定义标准导联的SCPECG编码（支持多种标签格式）
    lead_codes = {
        # 标准格式
        'I': ('5.6.3-9-1', 'Lead I (Einthoven)'),
        'II': ('5.6.3-9-2', 'Lead II'),
        'III': ('5.6.3-9-61', 'Lead III'),
        'aVR': ('5.6.3-9-62', 'Lead aVR'),
        'aVL': ('5.6.3-9-63', 'Lead aVL'),
        'aVF': ('5.6.3-9-64', 'Lead aVF'),
        'V1': ('5.6.3-9-3', 'Lead V1'),
        'V2': ('5.6.3-9-4', 'Lead V2'),
        'V3': ('5.6.3-9-5', 'Lead V3'),
        'V4': ('5.6.3-9-6', 'Lead V4'),
        'V5': ('5.6.3-9-7', 'Lead V5'),
        'V6': ('5.6.3-9-8', 'Lead V6'),
        # 支持大写格式（如AVR, AVL, AVF）
        'AVR': ('5.6.3-9-62', 'Lead aVR'),
        'AVL': ('5.6.3-9-63', 'Lead aVL'),
        'AVF': ('5.6.3-9-64', 'Lead aVF')
    }
    
    # 修复每个通道
    fixed_channels = 0
    for i, ch in enumerate(wf.ChannelDefinitionSequence):
        lead_label = getattr(ch, 'ChannelLabel', f'Channel_{i+1}')
        print(f"  Channel {i+1}: {lead_label}")
        
        # 检查是否已经有ChannelSourceSequence
        if not hasattr(ch, 'ChannelSourceSequence') or ch.ChannelSourceSequence is None:
            # 添加通道源信息
            if lead_label in lead_codes:
                code_value, code_meaning = lead_codes[lead_label]
                
                # 创建ChannelSourceSequence
                source_seq = Dataset()
                source_seq.CodeValue = code_value
                source_seq.CodingSchemeDesignator = "SCPECG"
                source_seq.CodingSchemeVersion = "1.3"
                source_seq.CodeMeaning = code_meaning
                
                ch.ChannelSourceSequence = Sequence([source_seq])
                fixed_channels += 1
                print(f"    ✅ 添加了通道源信息: {code_meaning}")
            else:
                print(f"    ⚠️  未知导联标签: {lead_label}")
        else:
            print(f"    ℹ️  已有通道源信息")
    
    print(f"🔧 修复了 {fixed_channels} 个通道")
    
    # 保存修复后的文件
    ds.save_as(output_file, write_like_original=False)
    print(f"💾 修复后的文件已保存到: {output_file}")
    
    return True

def main():
    """主函数"""
    input_file = 'test_00045_lr.dcm'
    output_file = 'test_00045_lr_fixed.dcm'
    
    if fix_dicom_channels(input_file, output_file):
        print("\n✅ 修复完成！")
        print(f"📁 原始文件: {input_file}")
        print(f"📁 修复文件: {output_file}")
        print("\n💡 现在你可以用DICOM查看器打开修复后的文件，应该能正常显示波形了！")
    else:
        print("\n❌ 修复失败！")

if __name__ == "__main__":
    main() 