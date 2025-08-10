# test_calibration.py
from pydicom import dcmread
import numpy as np

def test_different_calibrations(dicom_file):
    """测试不同的校准方法"""
    
    ds = dcmread(dicom_file)
    ws = ds.WaveformSequence[0]
    
    raw = np.frombuffer(ws.WaveformData, dtype=np.int16)
    ecg_array = raw.reshape((ws.NumberOfWaveformSamples, ws.NumberOfWaveformChannels)).astype(np.float32)
    
    max_original = np.abs(ecg_array).max()
    print(f"原始数据最大振幅: {max_original}")
    
    # 测试不同的校准因子
    test_factors = [0.001, 0.002, 0.005, 0.01, 0.02]
    
    for factor in test_factors:
        calibrated = ecg_array * factor
        max_calibrated = np.abs(calibrated).max()
        
        status = ""
        if 0.1 <= max_calibrated <= 10.0:
            status = "✅ 合理"
        elif max_calibrated > 10.0:
            status = "❌ 过大"
        else:
            status = "❌ 过小"
            
        print(f"校准因子 {factor:5.3f}: {max_calibrated:8.3f} mV {status}")

if __name__ == "__main__":
    test_different_calibrations("00045_lr.dcm")
