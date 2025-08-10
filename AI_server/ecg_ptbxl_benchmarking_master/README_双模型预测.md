# 双模型ECG预测系统使用说明

## 概述

本系统支持两种ECG预测模式：
1. **超类预测（5分类）**：使用微调模型 `fastai_xresnet1d101_finetuned_manual.pth` 预测5个主要疾病类别
2. **子类预测（71分类）**：使用原始模型 `fastai_xresnet1d101.pth` 预测71个详细疾病子类

## 文件结构

```
├── extract_and_predict_ecg.py          # 修改后的主预测脚本（支持双模型）
├── run_dual_predict.py                 # 双模型预测运行脚本
├── classify_single_ecg_ft.py           # 超类预测模块（5分类）
├── classify_single_ecg128.py           # 子类预测模块（71分类）
├── test_simple.py                      # 文件检查测试脚本
└── README_双模型预测.md                 # 本说明文档
```

## 使用方法

### 1. 超类预测（5分类）

```bash
# 预测5个超类：CD, HYP, MI, NORM, STTC
python3 run_dual_predict.py test_00045_lr.dcm superclass

# 或者省略参数，默认为超类预测
python3 run_dual_predict.py test_00045_lr.dcm
```

**输出示例：**
```
🎯 预测类型: superclass
🔢 加载了 5 个超类: ['CD' 'HYP' 'MI' 'NORM' 'STTC']
🏥 超类预测结果:
  CD              prob=0.982
  HYP             prob=0.011
  MI              prob=0.006
  NORM            prob=0.001
  STTC            prob=0.000
```

### 2. 子类预测（71分类）

```bash
# 预测71个子类
python3 run_dual_predict.py test_00045_lr.dcm subclass
```

**输出示例：**
```
🎯 预测类型: subclass
🔢 加载了 71 个子类
🏥 子类预测结果:
  SR              prob=0.501
  IRBBB           prob=0.478
  NORM            prob=0.005
  RVH             prob=0.004
  ABQRS           prob=0.003
```

## 模型文件要求

### 超类预测（5分类）
- **模型文件**: `output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth`
- **标准化器**: `output/exp0_ft/data/standard_scaler.pkl`
- **标签编码器**: `output/exp0_ft/data/mlb.pkl`

### 子类预测（71分类）
- **模型文件**: `output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth`
- **标准化器**: `output/exp0/data/standard_scaler.pkl`
- **标签编码器**: `output/exp0/data/mlb.pkl`

## 技术特点

1. **智能导入处理**：自动处理相对导入问题，无需修改原始代码
2. **双模型支持**：根据参数自动选择相应的模型和预处理函数
3. **错误处理**：完善的异常处理和用户友好的错误提示
4. **性能优化**：支持GPU加速，自动检测CUDA可用性

## 注意事项

1. **文件路径**：确保DICOM文件路径正确，支持相对路径和绝对路径
2. **模型文件**：确保所有必要的模型文件和数据文件都存在
3. **依赖库**：需要安装 `torch`, `numpy`, `pydicom`, `sklearn` 等库
4. **内存要求**：子类预测模型较大，确保有足够的内存

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件完整性

2. **数据文件缺失**
   - 运行 `python3 test_simple.py <dicom_path> <class_type>` 检查文件状态
   - 确认 `output` 目录结构正确

3. **导入错误**
   - 确保在正确的目录中运行脚本
   - 检查Python环境和依赖库

### 调试建议

使用 `test_simple.py` 脚本检查文件状态：
```bash
python3 test_simple.py test_00045_lr.dcm superclass
python3 test_simple.py test_00045_lr.dcm subclass
```

## 扩展功能

如需添加新的预测类型或模型，可以：
1. 在 `extract_and_predict_ecg.py` 中添加新的预测函数
2. 修改 `process_and_predict` 函数支持新的类型
3. 更新 `run_dual_predict.py` 脚本

## 联系支持

如遇到问题，请检查：
1. 文件路径和权限
2. 模型文件完整性
3. Python环境和依赖库版本
4. 系统内存和存储空间 