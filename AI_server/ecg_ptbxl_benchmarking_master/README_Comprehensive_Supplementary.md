# ECG Transfer Learning System - Comprehensive Supplementary

## 🎯 Core System Functions

### 1. Three-Model ECG Prediction System
- **Superclass Prediction (5-class)**: Uses fine-tuned model `fastai_xresnet1d101_finetuned_manual.pth` to predict 5 major disease categories
- **Subclass Prediction (71-class)**: Uses original model `fastai_xresnet1d101.pth` to predict 71 detailed disease subclasses
- **13-class Prediction**: Uses specialized 13-class model `fastai_xresnet1d101_13class_smart.pth` to directly predict target disease categories
- **Intelligent Disease Mapping**: Maps 71 disease classifications to 11 target disease categories

### 2. Model Training System
- **Pre-trained Model**: 71-class model trained on PTB-XL dataset
- **Fine-tuned Models**: 5-class and 13-class models fine-tuned on specific datasets
- **Multiple Training Strategies**: Supports training from scratch, fine-tuning, transfer learning, and other methods

### 3. Disease Mapping Functionality
- **Target Disease Categories**: NORM, 1dAVB, 2dAVB, 3dAVB, AFIB, AFLT, LBBB, RBBB, STACH, PAC, PVC (11 classes total)
- **Intelligent Mapping Rules**: Direct mapping, merge mapping, synonym mapping
- **Probability Accumulation and Sorting**: Supports multi-label classification probability processing
- **Multi-model Support**: Supports mapping prediction results from 71-class and 13-class models to target categories

## 🚀 Quick Start Guide

### Environment Setup
```bash
# Install basic dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn
pip install wfdb scipy pydicom

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Data Preparation
```bash
# 1. Navigate to project directory
cd AI_server/ecg_ptbxl_benchmarking_master

# 2. Check LTDB data
ls data/ltdb_processed/

# 3. If no data exists, run preprocessing
python code/data_ltdb.py
```

### Model Training
```bash
# Pre-trained model (already exists, no training needed)
# fastai_xresnet1d101.pth - 71-class pre-trained model based on PTB-XL

# Fine-tune superclass model (5-class)
python Finetune_model_super5.py

# Train 13-class model (recommended)
python test_icbeb_simple.py

# Evaluate model performance
python batch_predict_ptbxl_71class.py
```

## 🔧 Configuration Options

### Model Comparison

| Model Type | Classes | Training Script | Purpose | Advantages | Use Cases |
|------------|---------|-----------------|---------|------------|-----------|
| **Superclass Model** | 5 classes | `Finetune_model_super5.py` | Quick screening | Simple and fast | Initial diagnosis |
| **Subclass Model** | 71 classes | Pre-trained model | Detailed diagnosis | Comprehensive and accurate | Precise diagnosis |
| **13-class Model** | 11 classes | `test_icbeb_simple.py` | Target diagnosis | Highly targeted | Specific disease detection |

### Model File Requirements

#### Superclass Prediction (5-class)
- **Model File**: `output/exp0_ft/models/fastai_xresnet1d101_finetuned_manual.pth`
- **Training Script**: `Finetune_model_super5.py`
- **Standardizer**: `output/exp0_ft/data/standard_scaler.pkl`
- **Label Encoder**: `output/exp0_ft/data/mlb.pkl`
- **Prediction Classes**: CD, HYP, MI, NORM, STTC

#### Subclass Prediction (71-class)
- **Model File**: `output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth`
- **Training Info**: Pre-trained model based on PTB-XL dataset
- **Standardizer**: `output/exp0/data/standard_scaler.pkl`
- **Label Encoder**: `output/exp0/data/mlb.pkl`
- **Prediction Classes**: 71 PTB-XL disease subclasses, mapped to target categories

#### 13-class Prediction (Target Categories)
- **Model File**: `output/exp_13class_smart_multilabel/models/fastai_xresnet1d101_13class_smart.pth`
- **Training Script**: `test_icbeb_simple.py`
- **Standardizer**: `output/exp_13class_smart_multilabel/data/standard_scaler.pkl`
- **Label Encoder**: `output/exp_13class_smart_multilabel/data/mlb.pkl`
- **Prediction Classes**: Direct prediction of 11 target disease categories

## 🎯 Disease Mapping Rules

### Direct Mapping
- `NORM` → `NORM`
- `1AVB` → `1dAVB`
- `2AVB` → `2dAVB`
- `3AVB` → `3dAVB`
- `AFIB` → `AFIB`
- `AFLT` → `AFLT`

### Merge Mapping
- `LBBB`, `ILBBB`, `CLBBB` → `LBBB`
- `RBBB`, `IRBBB`, `CRBBB` → `RBBB`

### Synonym Mapping
- `STACH`, `SINUS TACHYCARDIA` → `STACH`
- `PAC`, `APB`, `APC` → `PAC`
- `PVC`, `VPB`, `VPC` → `PVC`

## 📚 Technical Principles

### Model Architecture
```
ECG Signal (12 leads) → XResNet1d101 → Classification Head → Disease Prediction
```

**Core Components**:
- **XResNet1d101**: 1D residual network specifically designed for time series signal processing
- **Classification Head**: Outputs different numbers of classes based on task requirements
- **Multi-label Support**: Supports simultaneous prediction of multiple diseases for a single ECG sample
- **Standardization Processing**: Uses StandardScaler for signal standardization

### Disease Mapping Strategy
- **Direct Mapping**: Diseases with identical names map directly
- **Merge Mapping**: Similar diseases are merged (e.g., ILBBB, CLBBB merged into LBBB)
- **Probability Accumulation**: Probabilities of multiple subclasses under the same target category are added
- **Sorted Output**: Sorted in descending order by probability, showing only categories with probability > 0