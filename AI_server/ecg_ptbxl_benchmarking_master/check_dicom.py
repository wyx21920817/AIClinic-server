#!/usr/bin/env python3
"""
Script to inspect and compare DICOM file structures
"""

import pydicom
import os

def check_dicom_structure(filename):
    """Inspect basic DICOM structure for a given file"""
    print(f"\n=== {filename} ===")
    
    try:
        ds = pydicom.dcmread(filename, force=True)
        
        # Basic info
        print(f"File size: {os.path.getsize(filename)} bytes")
        
        # Check waveform sequence
        if hasattr(ds, 'WaveformSequence'):
            print(f"Has WaveformSequence: True")
            print(f"WaveformSequence length: {len(ds.WaveformSequence)}")
            
            if len(ds.WaveformSequence) > 0:
                wf = ds.WaveformSequence[0]
                
                # Waveform data elements
                print(f"Has WaveformData: {hasattr(wf, 'WaveformData')}")
                if hasattr(wf, 'WaveformData'):
                    print(f"WaveformData length: {len(wf.WaveformData)} bytes")
                
                # Waveform parameters
                print(f"NumberOfWaveformSamples: {getattr(wf, 'NumberOfWaveformSamples', 'N/A')}")
                print(f"NumberOfWaveformChannels: {getattr(wf, 'NumberOfWaveformChannels', 'N/A')}")
                
                # Other important elements
                print(f"WaveformBitsAllocated: {getattr(wf, 'WaveformBitsAllocated', 'N/A')}")
                print(f"WaveformSampleInterpretation: {getattr(wf, 'WaveformSampleInterpretation', 'N/A')}")
                print(f"ChannelDefinitionSequence length: {len(getattr(wf, 'ChannelDefinitionSequence', []))}")
                
                # Inspect channel definitions
                if hasattr(wf, 'ChannelDefinitionSequence') and len(wf.ChannelDefinitionSequence) > 0:
                    for i, ch in enumerate(wf.ChannelDefinitionSequence):
                        print(f"  Channel {i}: {getattr(ch, 'ChannelLabel', 'N/A')}")
                        print(f"    ChannelSourceSequence: {getattr(ch, 'ChannelSourceSequence', 'N/A')}")
                
                # Inspect waveform data format
                if hasattr(wf, 'WaveformData'):
                    data = wf.WaveformData
                    print(f"WaveformData type: {type(data)}")
                    if isinstance(data, bytes):
                        print(f"First 20 bytes: {data[:20].hex()}")
                        
                        # Try to parse into numeric array
                        try:
                            import numpy as np
                            arr = np.frombuffer(data, dtype=np.int16)
                            print(f"Parsed as int16: {arr.size} samples")
                            if arr.size > 0:
                                print(f"First 10 values: {arr[:10]}")
                                print(f"Value range: {arr.min()} to {arr.max()}")
                        except Exception as e:
                            print(f"Error parsing data: {e}")
                
        else:
            print("Has WaveformSequence: False")
            
        # Check other possibly relevant elements
        print(f"Has ImageSequence: {hasattr(ds, 'ImageSequence')}")
        print(f"Has CurveSequence: {hasattr(ds, 'CurveSequence')}")
        
        # Check file meta information
        if hasattr(ds, 'file_meta'):
            print(f"TransferSyntaxUID: {getattr(ds.file_meta, 'TransferSyntaxUID', 'N/A')}")
            print(f"MediaStorageSOPClassUID: {getattr(ds.file_meta, 'MediaStorageSOPClassUID', 'N/A')}")
        
        # Check SOP class
        print(f"SOPClassUID: {getattr(ds, 'SOPClassUID', 'N/A')}")
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

def main():
    """Entry point"""
    files_to_check = [
        'test_00045_lr.dcm',  # user converted file
        'waveform_ecg.dcm',   # reference waveform DICOM
        'image-00000.dcm'     # reference image DICOM
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            check_dicom_structure(filename)
        else:
            print(f"\n=== {filename} ===")
            print("File not found")

if __name__ == "__main__":
    main() 