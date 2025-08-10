# DICOM波形显示问题解决方案

## 问题描述

用户使用 `wfdb2dcm.py` 转换出来的DICOM文件 `test_00045_lr.dcm` 无法在DICOM查看器中正常显示波形，而标准的DICOM文件（如 `waveform_ecg.dcm` 和 `image-00000.dcm`）可以正常显示。

## 问题分析

通过对比分析，发现了以下关键差异：

### 1. **ChannelSourceSequence 缺失**
- **问题文件** (`test_00045_lr.dcm`):
  - `ChannelLabel`: ✅ 有 (I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6)
  - `ChannelSourceSequence`: ❌ **缺失** (显示为 N/A)

- **标准文件** (`waveform_ecg.dcm`, `image-00000.dcm`):
  - `ChannelLabel`: ✅ 有
  - `ChannelSourceSequence`: ✅ **完整** (包含SCPECG编码的通道信息)

### 2. **通道定义不完整**
DICOM查看器需要完整的通道源信息来正确解析和显示波形数据。`ChannelSourceSequence` 包含了：
- `CodeValue`: 标准化的导联编码
- `CodingSchemeDesignator`: 编码方案标识符 (SCPECG)
- `CodingSchemeVersion`: 编码方案版本
- `CodeMeaning`: 导联含义描述

## 解决方案

### 方案1: 修复现有文件
使用 `fix_dicom.py` 脚本修复现有的DICOM文件：

```bash
python fix_dicom.py
```

这个脚本会：
1. 读取原始DICOM文件
2. 为每个通道添加完整的 `ChannelSourceSequence`
3. 使用标准SCPECG编码方案
4. 保存修复后的文件为 `test_00045_lr_fixed.dcm`

### 方案2: 使用修复版转换脚本
使用 `wfdb2dcm_fixed.py` 进行新的转换：

```bash
python wfdb2dcm_fixed.py <record_basename> <wfdb_dir> <out_dcm>
```

这个脚本在转换过程中就会包含完整的通道定义信息。

## 技术细节

### SCPECG编码方案
标准ECG导联的SCPECG编码：
- I: `5.6.3-9-1` - Lead I (Einthoven)
- II: `5.6.3-9-2` - Lead II
- III: `5.6.3-9-61` - Lead III
- aVR: `5.6.3-9-62` - Lead aVR
- aVL: `5.6.3-9-63` - Lead aVL
- aVF: `5.6.3-9-64` - Lead aVF
- V1: `5.6.3-9-3` - Lead V1
- V2: `5.6.3-9-4` - Lead V2
- V3: `5.6.3-9-5` - Lead V3
- V4: `5.6.3-9-6` - Lead V4
- V5: `5.6.3-9-7` - Lead V5
- V6: `5.6.3-9-8` - Lead V6

### 支持多种标签格式
修复脚本支持多种导联标签格式：
- 标准格式：`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`
- 大写格式：`AVR`, `AVL`, `AVF` (自动映射到标准格式)

## 验证结果

修复后的文件 `test_00045_lr_fixed.dcm` 现在包含：
- ✅ 完整的 `ChannelDefinitionSequence`
- ✅ 每个通道都有 `ChannelSourceSequence`
- ✅ 标准的SCPECG编码
- ✅ 正确的导联含义描述

## 使用建议

1. **对于现有文件**：使用 `fix_dicom.py` 进行修复
2. **对于新转换**：使用 `wfdb2dcm_fixed.py` 确保包含完整信息
3. **验证修复**：使用DICOM查看器打开修复后的文件，应该能正常显示波形

## 文件列表

- `fix_dicom.py` - 修复现有DICOM文件的脚本
- `wfdb2dcm_fixed.py` - 修复版WFDB到DICOM转换脚本
- `check_dicom.py` - DICOM文件结构检查脚本
- `test_00045_lr_fixed.dcm` - 修复后的DICOM文件

## 总结

DICOM波形无法显示的根本原因是缺少 `ChannelSourceSequence` 信息。通过添加完整的通道定义和标准编码，DICOM查看器就能正确解析和显示波形数据了。 