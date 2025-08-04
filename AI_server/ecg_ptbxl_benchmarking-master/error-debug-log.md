我来帮您检查这个问题。首先让我查看这两个文件的内容，了解它们是如何处理数据的。

Read file: classify_single_ecg_ft.py
Read file: extract_and_predict_ecg.py
现在让我检查一下DICOM文件的内容，看看数据提取是否正确：

```bash
cd "/d/PTE/CurtinUniversity/CSmaterial/Semester3/COMP6015  computer science project/modeltest/ecg_ptbxl_benchmarking-master" && python -c "import pydicom; ds = pydicom.dcmread('00045_lr.dcm', force=True); print('DICOM tags:'); print([tag for tag in ds.keys() if 'Waveform' in str(tag)]); print('WaveformSequence exists:', hasattr(ds, 'WaveformSequence')); print('Number of waveform sequences:', len(ds.WaveformSequence) if hasattr(ds, 'WaveformSequence') else 0)"
```
```bash
python -c "import pydicom; ds = pydicom.dcmread('00045_lr.dcm', force=True); print('DICOM tags:'); print([tag for tag in ds.keys() if 'Waveform' in str(tag)]); print('WaveformSequence exists:', hasattr(ds, 'WaveformSequence')); print('Number of waveform sequences:', len(ds.WaveformSequence) if hasattr(ds, 'WaveformSequence') else 0)"
```
让我更详细地检查DICOM文件的内容：

```bash
python -c "import pydicom; ds = pydicom.dcmread('00045_lr.dcm', force=True); wf = ds.WaveformSequence[0]; print('WaveformData length:', len(wf.WaveformData)); print('NumberOfWaveformSamples:', wf.NumberOfWaveformSamples); print('NumberOfWaveformChannels:', wf.NumberOfWaveformChannels); print('Expected size:', wf.NumberOfWaveformSamples * wf.NumberOfWaveformChannels); import numpy as np; arr = np.frombuffer(wf.WaveformData, dtype=np.int16); print('Actual array size:', arr.size); print('First 10 values:', arr[:10]); print('Data range:', arr.min(), 'to', arr.max())"
```
现在让我比较一下原始WFDB文件和DICOM文件的数据：

```bash
python -c "import wfdb; rec = wfdb.rdrecord('data/ptbxl/records100/00000/00045_lr'); print('WFDB signal shape:', rec.p_signal.shape); print('WFDB signal dtype:', rec.p_signal.dtype); print('WFDB data range:', rec.p_signal.min(), 'to', rec.p_signal.max()); print('First 10 values of first channel:', rec.p_signal[:10, 0])"
```
我发现了问题！让我创建一个测试脚本来验证数据转换过程：

```bash
python debug_data_conversion.py
```
我发现了问题！DICOM数据的形状是(1000, 12)，而WFDB数据的形状是(12, 1000)。让我修复这个调试脚本：

```bash
python debug_data_conversion.py
```
太好了！我发现数据转换是正确的，差异为0。现在让我检查`extract_and_predict_ecg.py`中的问题。让我创建一个测试脚本来验证模型预测：

```bash
python test_model_prediction.py
```
很有趣！我的测试显示WFDB和DICOM数据应该产生相同的结果。让我检查一下`extract_and_predict_ecg.py`中的具体问题。让我运行它看看实际输出：

```bash
python extract_and_predict_ecg.py 00045_lr.dcm
```
现在我发现了问题！`extract_and_predict_ecg.py`确实返回了错误的结果。让我仔细检查这个文件中的数据处理逻辑。我发现了问题所在：

Read file: extract_and_predict_ecg.py
我发现了问题！在`extract_and_predict_ecg.py`的第60行，代码没有对DICOM数据进行归一化处理。让我修复这个问题：

现在让我测试修复后的代码：

```bash
python extract_and_predict_ecg.py 00045_lr.dcm
```
太好了！现在结果正确了。让我也测试一下原始的WFDB文件来确认结果一致：

```bash
python classify_single_ecg_ft.py data/ptbxl/records100/00000/00045_lr
```
完美！现在两个文件都返回了相同的结果。让我清理一下调试文件：

Ran tool
Ran tool
## 问题总结

我已经找到并修复了您的问题。问题的根本原因是：

### 🔍 **问题分析**

1. **数据格式差异**：
   - WFDB文件：数据已经是浮点数格式，范围在[-1.258, 1.048]
   - DICOM文件：数据是int16格式，范围在[-1258, 1048]

2. **缺失的归一化步骤**：
   - `classify_single_ecg_ft.py`：直接使用WFDB数据，数据已经是正确的浮点格式
   - `extract_and_predict_ecg.py`：从DICOM提取的int16数据没有进行归一化处理

### 🛠️ **解决方案**

在`extract_and_predict_ecg.py`的第60行添加了归一化处理：

```python
# 归一化DICOM数据到[-1, 1]范围
sig = sig / 1000.0
```

### ✅ **验证结果**

修复后，两个文件现在都返回相同的结果：
- **CD**: 0.982
- **HYP**: 0.011  
- **MI**: 0.006
- **NORM**: 0.001
- **STTC**: 0.000

### 📝 **关键发现**

1. **数据转换过程是正确的**：WFDB到DICOM的转换没有问题
2. **问题在于预处理**：DICOM数据需要额外的归一化步骤
3. **模型预测是正确的**：一旦数据格式正确，模型能够给出准确的预测

现在您的`extract_and_predict_ecg.py`应该能够正确处理DICOM文件并给出与原始WFDB文件相同的预测结果了！