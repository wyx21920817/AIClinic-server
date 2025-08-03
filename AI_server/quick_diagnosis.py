# quick_diagnosis.py
import requests
import numpy as np
from dicom_utils import get_instance_dicom

def get_temp_token():
    AUTH_SERVICE_URL = "http://52.63.231.27:8000/api/auth/temp-token"
    try:
        response = requests.get(AUTH_SERVICE_URL)
        response.raise_for_status()
        return response.json().get("token")
    except Exception as e:
        print(f"获取token失败: {e}")
        return None

def check_instance_data(instance_id: str):
    """Check the data of a specific instance"""
    print(f" Check Instance: {instance_id}")
    
    token = get_temp_token()
    if not token:
        return
    
    try:
        ds = get_instance_dicom(instance_id, token)
        ws = ds.WaveformSequence[0]
        
        # Check the raw data
        raw = np.frombuffer(ws.WaveformData, dtype=np.int16)
        ecg_array = raw.reshape((ws.NumberOfWaveformSamples, ws.NumberOfWaveformChannels))
        
        print(f"\n Basic Information:")
        print(f"   Number of leads: {ws.NumberOfWaveformChannels}")
        print(f"   Number of sampling points: {ws.NumberOfWaveformSamples}")
        print(f"   Sampling rate: {ws.SamplingFrequency} Hz")
        print(f"   Original data range: [{ecg_array.min()}, {ecg_array.max()}]")
        
        # Check lead names
        print(f"\n Lead information:")
        try:
            lead_names = [ch.ChannelSourceCodeSequence[0].CodeMeaning for ch in ws.ChannelDefinitionSequence]
            print(f"   Lead Name: {lead_names}")
        except:
            print("   Unable to obtain lead name")
            lead_names = [f"Lead_{i+1}" for i in range(ws.NumberOfWaveformChannels)]
        
        # Check calibration information
        print(f"\n🔧 Calibration information:")
        if hasattr(ws, 'ChannelDefinitionSequence'):
            for i, ch_def in enumerate(ws.ChannelDefinitionSequence):
                if hasattr(ch_def, 'ChannelSensitivity'):
                    sensitivity = ch_def.ChannelSensitivity
                    print(f"   Lead{i+1}({lead_names[i]}): {sensitivity} µV/unit")
                else:
                    print(f"   Lead{i+1}({lead_names[i]}): No calibration information")
        
        # Estimate a reasonable data range
        max_amplitude = np.abs(ecg_array).max()
        print(f"\n Unit Analysis:")
        print(f"   maximum amplitude: {max_amplitude}")
        if max_amplitude > 1000:
            print("    The data appears to be raw ADC values and needs calibration")
        elif max_amplitude > 10:
            print("    Data may be in µV")  
        else:
            print("    The data may already be in mV units")
            
    except Exception as e:
        print(f" Check failed: {e}")

if __name__ == "__main__":
    instance_id = "11afdfd9-fa429bbd-995265c4-6d965eb0-4820f0dd"
    check_instance_data(instance_id)
