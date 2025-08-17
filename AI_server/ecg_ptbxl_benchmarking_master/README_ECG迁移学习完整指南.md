# ECG迁移学习完整指南

本指南整合了LTDB数据微调和迁移学习的所有内容，提供完整的ECG模型微调解决方案。

## 📋 目录
- [概述](#概述)
- [系统架构](#系统架构)
- [文件结构](#文件结构)
- [脚本详细介绍](#脚本详细介绍)
- [快速开始](#快速开始)
- [配置选项](#配置选项)
- [高级功能](#高级功能)
- [训练监控](#训练监控)
- [测试调试](#测试调试)
- [故障排除](#故障排除)
- [性能优化](#性能优化)

## 🎯 概述

本系统提供了两种ECG模型微调方法：

### 1. 传统微调方法
- 直接在LTDB数据上微调预训练模型
- 支持渐进式解冻和判别式学习率
- 适合有充足计算资源的场景

### 2. 迁移学习方法（推荐）
- 利用预训练的XResNet1d101模型
- 通过导联适配器处理2导联→12导联转换
- 支持多种冻结策略和注意力机制
- 适合资源受限或需要快速部署的场景

## 🏗️ 系统架构

### 传统微调架构
```
LTDB数据 → 数据预处理 → 预训练模型 → 微调 → 输出
```

### 迁移学习架构
```
LTDB数据 (2导联) → 导联适配器 → XResNet1d101 → 输出
```

**核心组件**：
- **导联适配器**: 将2导联信号转换为12导联
- **主干网络**: 预训练的XResNet1d101模型
- **冻结策略**: 可选择冻结/解冻不同层
- **注意力机制**: 增强导联转换能力

## 📁 文件结构

```
AI_server/ecg_ptbxl_benchmarking_master/
├── code/
│   ├── data_ltdb.py                    # LTDB数据预处理
│   ├── transfer_model.py              # 迁移学习模型定义
│   ├── finetune_with_transfer.py     # 迁移学习训练器
│   ├── lead_adapter.py               # 导联适配器实现
│   ├── configs/
│   │   ├── ltdb_finetune_configs.py   # 传统微调配置
│   │   └── transfer_learning_configs.py  # 迁移学习配置
│   └── models/
│       ├── xresnet1d.py              # XResNet1d模型
│       └── base_model.py             # 基础模型类
├── run_transfer_learning.py          # 迁移学习主运行脚本
├── test_transfer_learning.py         # 迁移学习测试脚本
├── test_updated_mapping.py           # 疾病映射测试脚本
├── extract_and_predict_ecg.py        # ECG预测和疾病映射
├── data/
│   └── ltdb_processed/               # 处理后的LTDB数据
├── output/                            # 训练输出和模型
└── README_ECG迁移学习完整指南.md      # 本指南
```

## 📜 脚本详细介绍

### 1. `data_ltdb.py` - 数据预处理核心
**功能**: LTDB数据的预处理和加载
**主要类**:
- `LTDBDataset`: 自定义数据集类，处理2导联ECG数据
- `load_ltdb_data`: 数据加载函数，返回训练/验证/测试数据加载器

**关键特性**:
- 自动重采样到100Hz
- 10秒窗口切分
- Z-score标准化
- 按记录划分数据集

**使用场景**: 所有微调方法的数据准备阶段

### 2. `transfer_model.py` - 迁移学习模型核心
**功能**: 定义迁移学习模型的完整架构
**主要类**:
- `TransferLearningModel`: 核心迁移学习模型
- `create_transfer_model`: 模型创建工厂函数

**关键特性**:
- 自动加载预训练权重
- 灵活的冻结策略控制
- 支持多种适配器类型
- 参数统计和模型保存

**核心方法**:
```python
def forward(self, x):           # 前向传播
def count_parameters(self):     # 参数统计
def save_model(self):           # 模型保存
def _apply_freezing_strategy(self): # 冻结策略应用
```

### 3. `finetune_with_transfer.py` - 迁移学习训练器
**功能**: 完整的迁移学习训练流程
**主要类**:
- `TransferLearningTrainer`: 迁移学习训练器

**关键特性**:
- 自动训练/验证循环
- 早停和学习率调度
- 最佳模型保存
- 训练历史可视化
- 测试集评估

**核心方法**:
```python
def train_epoch(self):          # 训练一个epoch
def validate_epoch(self):       # 验证一个epoch
def train(self):                # 主训练循环
def evaluate_on_test_set(self): # 测试集评估
```

### 4. `transfer_learning_configs.py` - 配置管理系统
**功能**: 集中管理所有训练配置
**主要配置**:
- `BASIC_TRANSFER_CONFIG`: 基础迁移学习
- `PROGRESSIVE_UNFREEZING_CONFIG`: 渐进式解冻
- `ATTENTION_CONFIG`: 注意力机制
- `MULTITASK_CONFIG`: 多任务学习
- `ADVANCED_CONFIG`: 高级组合策略
- `QUICK_TEST_CONFIG`: 快速测试

**配置结构**:
```python
{
    'name': '配置名称',
    'description': '配置描述',
    'adapter_type': '适配器类型',
    'freeze_backbone': True/False,
    'freeze_stages': [冻结的层],
    'batch_size': 批次大小,
    'epochs': 训练轮数,
    'lr': 学习率,
    # ... 其他参数
}
```

### 5. `run_transfer_learning.py` - 主运行脚本
**功能**: 用户友好的训练启动接口
**主要特性**:
- 命令行参数解析
- 环境检查
- 预训练模型验证
- 配置加载和覆盖
- 训练流程启动

**命令行参数**:
```bash
--config: 选择配置
--pretrained_model: 预训练模型路径
--epochs: 训练轮数
--batch_size: 批次大小
--lr: 学习率
--save_dir: 保存目录
--dry_run: 预览模式
```

### 6. `test_transfer_learning.py` - 测试验证脚本
**功能**: 验证迁移学习系统的各个组件
**测试内容**:
- 模型创建和参数统计
- 前向传播测试
- 模型保存/加载
- 配置系统验证
- 数据兼容性检查

**使用场景**: 代码验证、功能测试、调试

### 7. `extract_and_predict_ecg.py` - ECG预测和疾病映射
**功能**: ECG信号预测和疾病分类映射
**主要特性**:
- 支持超类（5分类）和子类（71分类）预测
- 智能疾病映射到10种目标类别
- 概率累加和排序输出
- 支持DICOM和WFDB格式

**疾病映射类别**:
- NORM, 1dAVB, 2dAVB, 3dAVB, AFIB, AFLT
- LBBB, RBBB, STACH, PAC, PVC

## 🚀 快速开始

### 环境准备

```bash
# 安装基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn
pip install wfdb scipy pydicom

# 检查CUDA可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 数据准备

```bash
# 1. 进入项目目录
cd AI_server/ecg_ptbxl_benchmarking_master

# 2. 检查LTDB数据
ls data/ltdb_processed/

# 3. 如果没有数据，运行预处理
python code/data_ltdb.py
```

### 预训练模型检查

```bash
# 检查预训练模型是否存在
ls output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth

# 如果不存在，需要先运行基础训练
# 参考项目中的其他训练脚本
```

### 运行迁移学习

#### 基础配置（推荐新手）
```bash
python run_transfer_learning.py --config basic
```

#### 渐进式解冻（高级用户）
```bash
python run_transfer_learning.py --config progressive --epochs 100
```

#### 快速测试（验证功能）
```bash
python run_transfer_learning.py --config quick_test --dry_run
```

### 测试疾病映射功能

```bash
# 运行疾病映射测试
python test_updated_mapping.py

# 测试ECG预测和映射
python extract_and_predict_ecg.py test.dcm subclass
```

## ⚙️ 配置选项

### 迁移学习配置对比

| 配置名称 | 冻结策略 | 适配器类型 | 适用场景 | 训练时间 |
|---------|----------|------------|----------|----------|
| `basic` | 完全冻结 | 基础 | 新手用户 | 快 |
| `progressive` | 渐进解冻 | 基础 | 最佳性能 | 中等 |
| `attention` | 完全冻结 | 注意力 | 复杂信号 | 快 |
| `multitask` | 完全冻结 | 基础 | 多标签 | 快 |
| `advanced` | 混合策略 | 注意力 | 研究实验 | 慢 |
| `quick_test` | 完全冻结 | 基础 | 功能验证 | 最快 |

## 🔧 高级功能

### 1. 渐进式解冻策略

```python
'unfreezing_schedule': [
    {'epoch': 0, 'freeze_stages': [0, 1, 2, 3]},   # 前10个epoch只训练适配器
    {'epoch': 10, 'freeze_stages': [0, 1, 2]},      # 解冻最后一个stage
    {'epoch': 20, 'freeze_stages': [0, 1]},          # 解冻倒数第二个stage
    {'epoch': 30, 'freeze_stages': [0]},             # 解冻倒数第三个stage
    {'epoch': 40, 'freeze_stages': []},              # 解冻所有层
]
```

### 2. 注意力机制

```python
'attention_heads': 4,                    # 注意力头数
'attention_dropout': 0.1,               # 注意力dropout
'use_positional_encoding': True,        # 位置编码
'attention_hidden_dim': 128             # 注意力隐藏维度
```

### 3. 多任务学习

```python
'tasks': ['rhythm', 'morphology', 'conduction'],  # 任务列表
'task_weights': [1.0, 1.0, 1.0],                # 任务权重
'loss_type': 'weighted_sum'                      # 损失类型
```

## 📊 训练监控

### 实时指标显示

训练过程中会显示：
- Epoch进度: `Epoch [1/50]`
- 训练指标: `Train Loss: 0.1234, Train Acc: 0.8567`
- 验证指标: `Val Loss: 0.0987, Val Acc: 0.8901`
- 学习率: `LR: 0.001000`
- 早停状态: `EarlyStopping counter: 2/10`

### 自动保存内容

训练完成后自动生成：
- `best_transfer_model.pth`: 最佳模型权重
- `training_history.json`: 训练历史数据
- `training_history.png`: 训练曲线图
- `confusion_matrix.png`: 混淆矩阵
- `classification_report.txt`: 分类报告

## 🧪 测试调试

### 第一步：环境检查

```bash
# 1. 检查Python环境
python --version
pip list | grep torch

# 2. 检查CUDA
python -c "import torch; print(f'CUDA版本: {torch.version.cuda}')"

# 3. 检查项目文件
ls -la code/
ls -la data/
```

### 第二步：功能测试

```bash
# 运行完整的测试套件
python test_transfer_learning.py

# 预期输出：
# ✅ 模型创建测试通过
# ✅ 前向传播测试通过
# ✅ 参数统计测试通过
# ✅ 模型保存测试通过
# ✅ 配置系统测试通过
# ✅ 数据兼容性测试通过
```

### 第三步：数据验证

```bash
# 检查LTDB数据
python -c "
# 检查数据文件是否存在
import os
data_dir = 'data/ltdb_processed'
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f'数据目录 {data_dir} 中的文件:')
    for file in files:
        print(f'  - {file}')
    
    # 检查数据文件大小
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(data_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f'  {file}: {size_mb:.2f} MB')
else:
    print(f'数据目录 {data_dir} 不存在，需要先运行数据预处理')
    print('运行: python code/data_ltdb.py')
"
```

### 第四步：模型验证

```bash
# 检查预训练模型
python -c "
import torch
model_path = 'output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth'
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f'✅ 预训练模型加载成功')
    print(f'模型键: {list(checkpoint.keys())}')
except Exception as e:
    print(f'❌ 预训练模型加载失败: {e}')
"
```

### 第五步：配置验证

```bash
# 列出所有可用配置
python -c "
from code.configs.transfer_learning_configs import list_available_configs
configs = list_available_configs()
print('可用配置:')
for config in configs:
    print(f'  - {config}')
"

# 查看特定配置详情
python -c "
from code.configs.transfer_learning_configs import print_config_summary
print_config_summary('basic')
"
```

### 第六步：干运行测试

```bash
# 使用快速测试配置进行干运行
python run_transfer_learning.py --config quick_test --dry_run

# 预期输出：
# ✅ 环境检查通过
# ✅ 预训练模型检查通过
# ✅ 数据加载成功
# ✅ 模型创建成功
# ✅ 训练器初始化成功
# 🚀 干运行完成，配置验证通过
```

### 第七步：小规模训练测试

```bash
# 使用快速配置进行小规模训练
python run_transfer_learning.py --config quick_test --epochs 2

# 检查输出文件
ls -la output/transfer_learning/quick_test/
```

### 第八步：疾病映射测试

```bash
# 测试疾病映射功能
python test_updated_mapping.py

# 测试ECG预测
python extract_and_predict_ecg.py test.dcm subclass
```

## 🔍 故障排除

### 常见问题及解决方案

#### 1. 预训练模型不存在
```bash
❌ 预训练模型文件不存在: output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth
```

**解决方案**:
```bash
# 方案1: 运行基础训练
python code/finetune_with_transfer.py

# 方案2: 使用测试脚本创建虚拟模型
python test_transfer_learning.py
```

#### 2. CUDA内存不足
```bash
RuntimeError: CUDA out of memory
```

**解决方案**:
```bash
# 减小批次大小
python run_transfer_learning.py --config basic --batch_size 8

# 使用CPU训练
python run_transfer_learning.py --config basic --device cpu
```

#### 3. 数据加载失败
```bash
❌ 数据加载失败: 数据文件不存在
```

**解决方案**:
```bash
# 检查数据目录
ls -la data/ltdb_processed/

# 运行数据预处理
python code/data_ltdb.py
```

#### 4. 依赖包缺失
```bash
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**:
```bash
# 安装缺失的包
pip install xxx

# 或者重新安装所有依赖
pip install -r requirements.txt
```

#### 5. 配置参数错误
```bash
KeyError: 'xxx'
```

**解决方案**:
```bash
# 检查配置是否正确
python -c "
from code.configs.transfer_learning_configs import get_config
config = get_config('basic')
print(config.keys())
"

# 使用标准配置
python run_transfer_learning.py --config basic
```

### 调试技巧

#### 1. 启用详细日志
```bash
# 在Python脚本中添加详细输出
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 使用断点调试
```python
import pdb; pdb.set_trace()  # 在代码中添加断点
```

#### 3. 检查中间结果
```python
# 在训练循环中添加打印
print(f"Batch shape: {batch.shape}")
print(f"Model output shape: {output.shape}")
```

#### 4. 内存使用监控
```bash
# 监控GPU内存使用
nvidia-smi -l 1

# 监控系统内存
htop
```

## 📈 性能优化

### 超参数调优建议

#### 1. 学习率调优
```python
# 推荐的学习率范围
'lr': 1e-4,        # 保守学习率
'lr': 1e-3,        # 标准学习率
'lr': 5e-3,        # 激进学习率
```

#### 2. 批次大小优化
```python
# 根据GPU内存选择
'batch_size': 8,   # 小内存GPU
'batch_size': 16,  # 中等内存GPU
'batch_size': 32,  # 大内存GPU
'batch_size': 64,  # 超大内存GPU
```

#### 3. 冻结策略选择
```python
# 数据量少时
'freeze_stages': [0, 1, 2, 3]  # 完全冻结

# 数据量中等时
'freeze_stages': [0, 1, 2]     # 部分解冻

# 数据量充足时
'freeze_stages': [0, 1]        # 大部分解冻
```

### 训练技巧

#### 1. 早停策略
```python
'early_stopping_patience': 10,     # 早停耐心值
'early_stopping_min_delta': 1e-4,  # 最小改善阈值
```

#### 2. 学习率调度
```python
'scheduler_patience': 5,           # 调度器耐心值
'scheduler_factor': 0.5,           # 学习率衰减因子
'scheduler_min_lr': 1e-6,         # 最小学习率
```

#### 3. 梯度裁剪
```python
'grad_clip': 1.0,                  # 梯度裁剪阈值
```

## 🔬 实验建议

### 对比实验设计

建议运行以下配置进行对比：

```bash
# 1. 基础迁移学习
python run_transfer_learning.py --config basic --epochs 50

# 2. 渐进式解冻
python run_transfer_learning.py --config progressive --epochs 100

# 3. 注意力机制
python run_transfer_learning.py --config attention --epochs 50

# 4. 疾病映射测试
python test_updated_mapping.py
```

### 结果分析维度

比较不同配置的以下指标：

#### 性能指标
- 最终验证准确率
- 训练收敛速度
- 过拟合程度

#### 效率指标
- 训练时间
- 内存使用
- 参数量

#### 实用性指标
- 推理速度
- 模型大小
- 部署难度

### 实验记录模板

```markdown
## 实验记录 - {配置名称}

### 基本信息
- 配置: {配置名称}
- 训练时间: {开始时间} - {结束时间}
- 硬件: {GPU型号} + {内存大小}

### 超参数
- 学习率: {学习率}
- 批次大小: {批次大小}
- 训练轮数: {轮数}
- 冻结策略: {冻结策略}

### 结果
- 最佳验证准确率: {准确率}
- 最终训练损失: {损失}
- 收敛轮数: {轮数}
- 训练时间: {时间}

### 分析
- 优势: {优势分析}
- 劣势: {劣势分析}
- 适用场景: {适用场景}
```

## 📚 扩展阅读

### 技术原理
- [迁移学习原理](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ECG信号处理](https://en.wikipedia.org/wiki/Electrocardiography)
- [XResNet架构](https://arxiv.org/abs/2002.09789)
- [注意力机制](https://arxiv.org/abs/1706.03762)

### 相关论文
- PTB-XL: A Large Publicly Available Electrocardiography Dataset
- Transfer Learning for ECG Classification
- Attention Mechanisms in Medical Signal Processing

### 实用工具
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [ECG信号可视化工具](https://github.com/plotly/plotly.py)
- [医学信号处理库](https://github.com/biosignalsplux/biosignalsnotebooks)

## 🤝 贡献指南

欢迎提交问题和改进建议！

### 贡献流程
1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 发起Pull Request

### 贡献类型
- 🐛 Bug报告
- 💡 功能建议
- 📚 文档改进
- 🔧 代码优化
- 🧪 测试用例

### 代码规范
- 遵循PEP 8 Python代码规范
- 添加适当的注释和文档字符串
- 包含单元测试
- 更新相关文档

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

- MIT-BIH Long-Term ECG Database
- PTB-XL Benchmarking Framework
- FastAI深度学习框架
- PyTorch开发团队
- 开源社区贡献者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者
- 参与项目讨论

---

**最后更新**: 2024年12月
**版本**: 1.0.0
**维护者**: AIClinic开发团队 