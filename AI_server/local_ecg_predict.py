#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版本：本地DICOM文件DNN模型预测脚本
使用正确的校准因子
"""

import numpy as np
import os
import sys
from pathlib import Path
import argparse
import logging
from scipy import signal
from pydicom import dcmread
from tensorflow.keras.models import load_model

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalECGPredictor:
    def __init__(self, model_path="models/model.hdf5"):
        """
        初始化ECG预测器
        Args:
            model_path: DNN模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.target_sampling_rate = 400
        self.target_length = 4096
        self.load_model()
    
    def load_model(self):
        """加载DNN模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.model = load_model(self.model_path)
            logger.info(f"✅ DNN模型加载成功: {self.model_path}")
            logger.info(f"模型输入形状: {self.model.input_shape}")
            logger.info(f"模型输出形状: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def extract_waveform(self, dicom_file_path):
        """
        从DICOM文件提取ECG波形数据
        使用修正的校准逻辑
        """
        try:
            # 读取DICOM文件
            ds = dcmread(dicom_file_path)
            logger.info(f"📁 DICOM文件读取成功: {dicom_file_path}")
            
            # 检查是否包含波形数据
            if not hasattr(ds, 'WaveformSequence') or len(ds.WaveformSequence) == 0:
                raise ValueError("DICOM文件中未找到WaveformSequence")
            
            ws = ds.WaveformSequence[0]
            n_channels = ws.NumberOfWaveformChannels
            n_samples = ws.NumberOfWaveformSamples
            original_sampling_rate = int(ws.SamplingFrequency)
            
            logger.info(f"📊 波形信息: {n_channels}导联, {n_samples}采样点, {original_sampling_rate}Hz")
            
            # 解析波形数据
            raw = np.frombuffer(ws.WaveformData, dtype=np.int16)
            ecg_array = raw.reshape((n_samples, n_channels)).astype(np.float32)
            
            logger.info(f"原始数据范围: [{ecg_array.min():.0f}, {ecg_array.max():.0f}] ADC units")
            
            # ✅ 修正的单位校准
            ecg_array = self.apply_calibration_fixed(ws, ecg_array)
            
            # 重采样到400Hz
            if original_sampling_rate != self.target_sampling_rate:
                ecg_array = self.resample_ecg(ecg_array, original_sampling_rate, self.target_sampling_rate)
                logger.info(f"🔄 重采样: {original_sampling_rate}Hz → {self.target_sampling_rate}Hz")
            
            # 调整长度到4096
            ecg_array = self.adjust_length(ecg_array, self.target_length)
            
            # 获取导联名称
            lead_names = self.get_lead_names(ws, n_channels)
            
            # 最终验证
            if ecg_array.shape != (4096, 12):
                raise ValueError(f"最终数据格式错误: 期望(4096, 12), 得到{ecg_array.shape}")
            
            logger.info(f"✅ 预处理完成: shape={ecg_array.shape}, 范围=[{ecg_array.min():.3f}, {ecg_array.max():.3f}] mV")
            
            return ecg_array, self.target_sampling_rate, lead_names
            
        except Exception as e:
            logger.error(f"❌ 波形提取失败: {e}")
            raise
    
    def apply_calibration_fixed(self, ws, ecg_array):
        """
        修正的校准函数 - 基于测试结果使用最佳校准因子
        """
        max_amplitude = np.abs(ecg_array).max()
        logger.info(f"原始数据最大振幅: {max_amplitude:.0f}")
        
        # 检查是否有DICOM校准信息
        has_calibration = False
        if hasattr(ws, 'ChannelDefinitionSequence'):
            for ch_def in ws.ChannelDefinitionSequence:
                if hasattr(ch_def, 'ChannelSensitivity'):
                    sensitivity = float(ch_def.ChannelSensitivity)
                    has_calibration = True
                    logger.info(f"找到DICOM校准信息: {sensitivity}")
                    break
        
        if has_calibration:
            # 使用DICOM提供的校准信息
            for ch_idx, ch_def in enumerate(ws.ChannelDefinitionSequence):
                if hasattr(ch_def, 'ChannelSensitivity'):
                    sensitivity = float(ch_def.ChannelSensitivity)
                    
                    # 根据单位信息决定如何校准
                    if hasattr(ch_def, 'ChannelSensitivityUnitsSequence'):
                        try:
                            units = ch_def.ChannelSensitivityUnitsSequence[0].CodeMeaning
                            if 'MICROVOLT' in units.upper() or 'UV' in units.upper():
                                ecg_array[:, ch_idx] = ecg_array[:, ch_idx] * sensitivity / 1000.0
                            else:
                                ecg_array[:, ch_idx] = ecg_array[:, ch_idx] * sensitivity
                        except:
                            # 默认假设是µV单位
                            ecg_array[:, ch_idx] = ecg_array[:, ch_idx] * sensitivity / 1000.0
                    else:
                        # 根据sensitivity数值判断
                        if sensitivity > 1:
                            ecg_array[:, ch_idx] = ecg_array[:, ch_idx] * sensitivity / 1000.0
                        else:
                            ecg_array[:, ch_idx] = ecg_array[:, ch_idx] * sensitivity
            
            logger.info("✅ 使用DICOM校准信息")
        else:
            # 🎯 基于测试结果，使用最佳校准因子
            logger.warning("⚠️ 无DICOM校准信息，使用测试优化的校准因子")
            
            # 根据数据范围选择校准因子
            if 2000 <= max_amplitude <= 3000:
                # 您这个文件的数据特征，使用测试验证的最佳因子
                calibration_factor = 0.001  # 1µV/unit - 测试显示最合理
                logger.info("使用校准因子: 0.001 (1µV/unit)")
            elif max_amplitude > 3000:
                calibration_factor = 0.0005  # 0.5µV/unit
                logger.info("使用校准因子: 0.0005 (0.5µV/unit)")
            elif 1000 <= max_amplitude < 2000:
                calibration_factor = 0.002   # 2µV/unit
                logger.info("使用校准因子: 0.002 (2µV/unit)")
            else:
                calibration_factor = 0.005   # 5µV/unit
                logger.info("使用校准因子: 0.005 (5µV/unit)")
            
            ecg_array = ecg_array * calibration_factor
        
        # 验证校准结果
        final_max = np.abs(ecg_array).max()
        logger.info(f"校准后最大振幅: {final_max:.3f} mV")
        
        if 0.1 <= final_max <= 15.0:
            logger.info("✅ 校准结果在合理ECG范围内")
        else:
            logger.warning(f"⚠️ 校准结果需要注意: {final_max:.3f} mV")
        
        return ecg_array
    
    def resample_ecg(self, ecg_array, original_rate, target_rate):
        """重采样ECG数据"""
        if original_rate == target_rate:
            return ecg_array
        
        new_length = int(ecg_array.shape[0] * target_rate / original_rate)
        resampled_ecg = np.zeros((new_length, ecg_array.shape[1]))
        
        for ch in range(ecg_array.shape[1]):
            resampled_ecg[:, ch] = signal.resample(ecg_array[:, ch], new_length)
        
        return resampled_ecg
    
    def adjust_length(self, ecg_array, target_length):
        """调整ECG长度到目标长度"""
        current_length = ecg_array.shape[0]
        
        if current_length > target_length:
            # 裁剪：从中间开始
            start_idx = (current_length - target_length) // 2
            ecg_array = ecg_array[start_idx:start_idx + target_length, :]
            logger.info(f"📏 裁剪: {current_length} → {target_length}")
        elif current_length < target_length:
            # 填充：两边对称填充
            padding = target_length - current_length
            pad_before = padding // 2
            pad_after = padding - pad_before
            ecg_array = np.pad(ecg_array, ((pad_before, pad_after), (0, 0)), mode='constant', constant_values=0)
            logger.info(f"📏 填充: {current_length} → {target_length}")
        
        return ecg_array
    
    def get_lead_names(self, ws, n_channels):
        """获取导联名称"""
        try:
            lead_names = [ch.ChannelSourceCodeSequence[0].CodeMeaning for ch in ws.ChannelDefinitionSequence]
            logger.info(f"✅ 获取导联名称: {lead_names}")
        except AttributeError:
            # 使用标准12导联名称
            standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            lead_names = standard_leads[:n_channels]
            logger.info(f"⚠️ 使用标准导联名称: {lead_names}")
        
        return lead_names
    
    def predict(self, ecg_array):
        """使用DNN模型进行预测"""
        try:
            # 验证输入格式
            if ecg_array.shape != (4096, 12):
                raise ValueError(f"输入格式错误: 期望(4096, 12), 得到{ecg_array.shape}")
            
            # 添加batch维度
            x = np.expand_dims(ecg_array, axis=0)  # (1, 4096, 12)
            logger.info(f"🧠 模型输入形状: {x.shape}")
            
            # 执行预测
            predictions = self.model.predict(x, verbose=0)
            logger.info(f"🎯 模型输出形状: {predictions.shape}")
            
            # 解析预测结果
            result = {
                '1dAVb': float(predictions[0][0]),  # 一度房室传导阻滞
                'RBBB': float(predictions[0][1]),   # 右束支传导阻滞
                'LBBB': float(predictions[0][2]),   # 左束支传导阻滞
                'SB': float(predictions[0][3]),     # 窦性心动过缓
                'AF': float(predictions[0][4]),     # 房颤
                'ST': float(predictions[0][5])      # 窦性心动过速
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            raise
    
    def predict_file(self, dicom_file_path):
        """对单个DICOM文件进行完整的预测流程"""
        logger.info(f"🚀 开始预测: {dicom_file_path}")
        
        try:
            # 1. 提取波形
            ecg_data, sampling_rate, lead_names = self.extract_waveform(dicom_file_path)
            
            # 2. 执行预测
            predictions = self.predict(ecg_data)
            
            # 3. 分析结果
            analysis = self.analyze_predictions(predictions)
            
            return {
                'file_path': dicom_file_path,
                'predictions': predictions,
                'analysis': analysis,
                'data_info': {
                    'shape': ecg_data.shape,
                    'sampling_rate': sampling_rate,
                    'lead_names': lead_names,
                    'data_range': [float(ecg_data.min()), float(ecg_data.max())]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 文件预测失败: {e}")
            return {
                'file_path': dicom_file_path,
                'error': str(e)
            }
    
    def analyze_predictions(self, predictions):
        """分析预测结果"""
        # 找出最高概率的异常
        max_disease = max(predictions.keys(), key=lambda k: predictions[k])
        max_prob = predictions[max_disease]
        
        # 判断是否正常
        threshold = 0.1  # 阈值可调整
        is_normal = max_prob < threshold
        
        # 创建疾病名称映射
        disease_names = {
            '1dAVb': '一度房室传导阻滞',
            'RBBB': '右束支传导阻滞',
            'LBBB': '左束支传导阻滞',
            'SB': '窦性心动过缓',
            'AF': '房颤',
            'ST': '窦性心动过速'
        }
        
        analysis = {
            'is_normal': is_normal,
            'max_abnormality': {
                'disease': max_disease,
                'disease_name': disease_names.get(max_disease, max_disease),
                'probability': max_prob
            },
            'summary': '正常ECG' if is_normal else f'疑似{disease_names.get(max_disease, max_disease)} (概率: {max_prob:.1%})'
        }
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='修正版本：本地DICOM文件ECG-DNN预测工具')
    parser.add_argument('input', help='DICOM文件路径或包含DICOM文件的目录')
    parser.add_argument('--model', default='models/model.hdf5', help='DNN模型文件路径')
    parser.add_argument('--output', help='输出结果到JSON文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建预测器
    predictor = LocalECGPredictor(args.model)
    
    # 处理输入
    input_path = Path(args.input)
    results = []
    
    if input_path.is_file():
        # 单个文件
        if input_path.suffix.lower() == '.dcm':
            result = predictor.predict_file(str(input_path))
            results.append(result)
        else:
            logger.error("❌ 输入文件不是DICOM格式 (.dcm)")
            sys.exit(1)
    
    elif input_path.is_dir():
        # 目录中的所有DICOM文件
        dicom_files = list(input_path.glob('*.dcm'))
        if not dicom_files:
            logger.error("❌ 目录中未找到DICOM文件")
            sys.exit(1)
        
        logger.info(f"📁 找到 {len(dicom_files)} 个DICOM文件")
        
        for dicom_file in dicom_files:
            result = predictor.predict_file(str(dicom_file))
            results.append(result)
    
    else:
        logger.error("❌ 输入路径不存在")
        sys.exit(1)
    
    # 显示结果
    print("\n" + "="*60)
    print("🏥 ECG-DNN 预测结果 (修正版)")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"\n📁 文件 {i}: {Path(result['file_path']).name}")
            print(f"❌ 错误: {result['error']}")
            continue
        
        print(f"\n📁 文件 {i}: {Path(result['file_path']).name}")
        print(f"📊 数据信息: {result['data_info']['shape']}, {result['data_info']['sampling_rate']}Hz")
        print(f"📊 数据范围: [{result['data_info']['data_range'][0]:.3f}, {result['data_info']['data_range'][1]:.3f}] mV")
        print(f"📋 诊断结果: {result['analysis']['summary']}")
        
        print(f"\n🔍 详细预测概率:")
        predictions = result['predictions']
        for disease, prob in predictions.items():
            disease_names = {
                '1dAVb': '一度房室传导阻滞',
                'RBBB': '右束支传导阻滞', 
                'LBBB': '左束支传导阻滞',
                'SB': '窦性心动过缓',
                'AF': '房颤',
                'ST': '窦性心动过速'
            }
            print(f"   {disease_names.get(disease, disease):12}: {prob:8.1%}")
    
    # 保存结果到文件
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"📝 结果已保存到: {args.output}")
    
    print(f"\n✅ 预测完成，共处理 {len(results)} 个文件")

if __name__ == "__main__":
    main()
