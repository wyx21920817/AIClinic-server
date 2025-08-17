# 疾病映射功能说明

## 概述

本模块已修改 `extract_and_predict_ecg.py` 中的 `process_and_predict` 函数，实现了疾病分类的智能映射功能。现在可以将71种疾病分类映射到老师感兴趣的10种疾病类别。

## 目标疾病类别

修改后的系统只输出以下10种疾病类别：

| 编号 | 疾病代码 | 疾病名称 | 说明 |
|------|----------|----------|------|
| 1 | NORM | Normal | 正常心电图 |
| 2 | 1dAVB | First-degree AV Block | 一度房室传导阻滞 |
| 3 | 2dAVB | Second-degree AV Block | 二度房室传导阻滞 |
| 4 | 3dAVB | Third-degree AV Block | 三度房室传导阻滞 |
| 5 | AFIB | Atrial Fibrillation | 心房颤动 |
| 6 | AFLT | Atrial Flutter | 心房扑动 |
| 7 | LBBB | Left Bundle Branch Block | 左束支传导阻滞（包含部分和完全性） |
| 8 | RBBB | Right Bundle Branch Block | 右束支传导阻滞（包含部分和完全性） |
| 9 | STACH | Sinus Tachycardia | 窦性心动过速 |
| 10 | PAC | Premature Atrial Complex | 房性早搏 |
| 11 | PVC | Premature Ventricular Complex | 室性早搏 |

## 疾病映射规则

### 1. 直接映射
- `NORM` → `NORM`
- `1AVB` → `1dAVB` (注意：scp_statements.csv 中是 1AVB，映射到 1dAVB)
- `2AVB` → `2dAVB` (注意：scp_statements.csv 中是 2AVB，映射到 2dAVB)
- `3AVB` → `3dAVB` (注意：scp_statements.csv 中是 3AVB，映射到 3dAVB)
- `AFIB` → `AFIB`
- `AFLT` → `AFLT`

### 2. 合并映射
- `LBBB`, `ILBBB`, `CLBBB` → `LBBB`
  - `ILBBB`: Incomplete Left Bundle Branch Block
  - `CLBBB`: Complete Left Bundle Branch Block
- `RBBB`, `IRBBB`, `CRBBB` → `RBBB`
  - `IRBBB`: Incomplete Right Bundle Branch Block
  - `CRBBB`: Complete Right Bundle Branch Block

### 3. 同义词映射
- `STACH`, `SINUS TACHYCARDIA` → `STACH`
- `PAC`, `APB`, `APC` → `PAC`
  - `APB`: Atrial Premature Beat
  - `APC`: Atrial Premature Complex
- `PVC`, `VPB`, `VPC` → `PVC`
  - `VPB`: Ventricular Premature Beat
  - `VPC`: Ventricular Premature Complex

## 使用方法

### 1. 基本使用

```python
from extract_and_predict_ecg import process_and_predict

# 预测子类（71分类）并自动映射到目标类别
result = process_and_predict('path/to/dicom/file.dcm', 'subclass')

# 输出结果示例
# {
#     'NORM': 0.85,
#     'LBBB': 0.65,  # 合并了 ILBBB 和 CLBBB 的概率
#     'AFIB': 0.45,
#     'PAC': 0.32,
#     'PVC': 0.18
# }
```

### 2. 函数说明

#### `get_disease_mapping()`
返回疾病映射字典，包含所有原始疾病到目标类别的映射关系。

#### `map_diseases_to_categories(predictions, disease_mapping)`
将原始预测结果映射到目标疾病类别。

**参数：**
- `predictions`: 原始预测结果字典 `{disease: probability}`
- `disease_mapping`: 疾病映射字典

**返回：**
- 映射后的预测结果字典 `{target_category: probability}`

### 3. 命令行使用

```bash
# 预测子类并映射到目标类别
python extract_and_predict_ecg.py path/to/dicom/file.dcm subclass

# 预测超类（5分类）
python extract_and_predict_ecg.py path/to/dicom/file.dcm superclass
```

## 测试

运行测试脚本验证功能：

```bash
python test_disease_mapping.py
```

测试内容包括：
- 正常疾病映射
- 未知疾病处理
- LBBB 合并功能
- 边界情况处理

## 输出示例

### 原始输出（71分类）
```
原始预测结果 (71分类):
  NORM              prob=0.456
  ILBBB             prob=0.234
  CLBBB             prob=0.123
  AFIB              prob=0.089
  PAC               prob=0.067
```

### 映射后输出（目标类别）
```
映射后的预测结果 (目标疾病类别):
  NORM              prob=0.456
  LBBB              prob=0.357  # ILBBB + CLBBB 合并
  AFIB              prob=0.089
  PAC               prob=0.067
```

## 注意事项

1. **概率累加**：相同目标类别的疾病概率会被累加
2. **未知疾病**：不在映射表中的疾病会被忽略
3. **排序输出**：最终结果按概率降序排列，只显示概率大于0的类别
4. **向后兼容**：原有的超类预测功能保持不变

## 修改历史

- 添加了 `get_disease_mapping()` 函数
- 添加了 `map_diseases_to_categories()` 函数
- 修改了 `_predict_subclass()` 函数，增加疾病映射功能
- 保持了原有的超类预测功能
- 添加了详细的日志输出，便于调试

## 技术细节

- 使用模糊匹配处理疾病名称变体
- 支持概率累加和排序
- 保持原有的模型推理逻辑不变
- 增加了错误处理和边界情况处理
