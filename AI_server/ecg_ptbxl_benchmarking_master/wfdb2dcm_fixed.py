#!/usr/bin/env python3
"""
Convert one PTB-XL WFDB record (.dat/.hea) to a single-file DICOM ECG Waveform.
Fixed version with complete channel definition information for proper waveform display.

Usage:
    python wfdb2dcm_fixed.py <record_basename> <wfdb_dir> <out_dcm>
Example:
    python wfdb2dcm_fixed.py 00048_lr "data/ptbxl/records100/00000" "output/00048_lr.dcm"

Note: This script requires WFDB files (.dat and .hea) to exist in the specified directory.
If you don't have WFDB files, you can use fix_dicom.py to repair existing DICOM files instead.
"""
import sys, uuid, datetime
import numpy as np
import wfdb
import pydicom
import os
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

def wfdb_to_dicom(record_name: str, wfdb_dir: str, out_path: str):
    """Convert WFDB record to DICOM ECG waveform"""
    rec_path = os.path.join(wfdb_dir, record_name)
    
    # Check if WFDB files exist
    dat_file = f"{rec_path}.dat"
    hea_file = f"{rec_path}.hea"
    
    if not os.path.exists(dat_file):
        raise FileNotFoundError(f"WFDB data file not found: {dat_file}")
    if not os.path.exists(hea_file):
        raise FileNotFoundError(f"WFDB header file not found: {hea_file}")
    
    print(f"📁 Reading WFDB record: {rec_path}")
    print(f"   Data file: {dat_file}")
    print(f"   Header file: {hea_file}")
    
    try:
        rec = wfdb.rdrecord(rec_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read WFDB record: {e}")
    
    sig   = rec.p_signal.astype("float32")
    fs    = rec.fs
    leads = rec.sig_name
    n_samp, n_chan = sig.shape
    
    print(f"📊 Signal shape: {sig.shape}")
    print(f"📈 Sampling frequency: {fs} Hz")
    print(f"🔌 Leads: {leads}")
    
    sig_int16 = np.round(sig * 1000).astype("<i2")
    wf_bytes  = sig_int16.tobytes()
    
    # Generate DICOM UIDs
    sop_instance_uid = pydicom.uid.generate_uid()
    sop_class_uid = "1.2.840.10008.5.1.4.1.1.9.1.1"
    
    # Create file meta information
    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.1000"
    
    # Create main dataset
    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    
    # Patient and study information
    ds.PatientName = "Anon^PTBXL"
    ds.PatientID = "PTBXL"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyDate, ds.StudyTime = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S").split("-")
    ds.Modality = "ECG"
    ds.Manufacturer = "PTBXL-Converter"
    ds.StudyID = "PTBXL_STUDY"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "ECG"]
    
    # Image-related attributes (required for ECG waveform)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    
    # Create waveform item
    wv_item = Dataset()
    wv_item.WaveformOriginality     = "ORIGINAL"
    wv_item.NumberOfWaveformChannels= n_chan
    wv_item.NumberOfWaveformSamples = n_samp
    wv_item.SamplingFrequency       = fs
    wv_item.MultiplexGroupTimeOffset= 0
    wv_item.TriggerTimeOffset       = 0
    wv_item.WaveformBitsAllocated   = 16
    wv_item.WaveformSampleInterpretation = "SS"
    wv_item.WaveformData            = wf_bytes
    
    # Create channel definitions with complete source information
    ch_seq = []
    lead_codes = {
        'I': ('5.6.3-9-1', 'Lead I (Einthoven)'),
        'II': ('5.6.3-9-2', 'Lead II'),
        'III': ('5.6.3-9-61', 'Lead III'),
        'aVR': ('5.6.3-9-62', 'Lead aVR'),
        'aVL': ('5.6.3-9-63', 'Lead aVL'),
        'aVF': ('5.6.3-9-64', 'Lead aVF'),
        'AVR': ('5.6.3-9-62', 'Lead aVR'),
        'AVL': ('5.6.3-9-63', 'Lead aVL'),
        'AVF': ('5.6.3-9-64', 'Lead aVF'),
        'V1': ('5.6.3-9-3', 'Lead V1'),
        'V2': ('5.6.3-9-4', 'Lead V2'),
        'V3': ('5.6.3-9-5', 'Lead V3'),
        'V4': ('5.6.3-9-6', 'Lead V4'),
        'V5': ('5.6.3-9-7', 'Lead V5'),
        'V6': ('5.6.3-9-8', 'Lead V6')
    }
    
    for ch_idx, lead in enumerate(leads, 1):
        ch = Dataset()
        ch.ChannelNumber            = ch_idx
        ch.ChannelLabel             = lead
        
        # Add SCPECG coding information
        if lead in lead_codes:
            code_value, code_meaning = lead_codes[lead]
            source_seq = Dataset()
            source_seq.CodeValue = code_value
            source_seq.CodingSchemeDesignator = "SCPECG"
            source_seq.CodingSchemeVersion = "1.3"
            source_seq.CodeMeaning = code_meaning
            ch.ChannelSourceSequence = Sequence([source_seq])
        else:
            print(f"⚠️  Warning: Unknown lead '{lead}' - no SCPECG code assigned")
        
        ch_seq.append(ch)
    
    wv_item.ChannelDefinitionSequence = Sequence(ch_seq)
    ds.WaveformSequence = Sequence([wv_item])
    
    # Ensure UIDs are set correctly
    if not hasattr(ds, 'SOPClassUID') or ds.SOPClassUID is None:
        ds.SOPClassUID = sop_class_uid
    if not hasattr(ds, 'SOPInstanceUID') or ds.SOPInstanceUID is None:
        ds.SOPInstanceUID = sop_instance_uid
    
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    
    # Save DICOM file
    ds.save_as(out_path, write_like_original=False)
    
    print(f"✅ DICOM ECG saved to: {out_path}")
    print(f"   SOPClassUID: {ds.SOPClassUID}")
    print(f"   SOPInstanceUID: {ds.SOPInstanceUID}")
    print(f"   Channels: {len(ch_seq)} with complete source information")

def main():
    """Main function with improved error handling"""
    if len(sys.argv) != 4:
        print(__doc__)
        print("\n🔍 Current directory contents:")
        try:
            files = os.listdir('.')
            wfdb_files = [f for f in files if f.endswith(('.hea', '.dat'))]
            if wfdb_files:
                print("   Found WFDB files:")
                for f in wfdb_files[:10]:  # Show first 10
                    print(f"     {f}")
                if len(wfdb_files) > 10:
                    print(f"     ... and {len(wfdb_files) - 10} more")
            else:
                print("   No WFDB files found in current directory")
                print("\n💡 Alternative solutions:")
                print("   1. Use fix_dicom.py to repair existing DICOM files")
                print("   2. Download PTB-XL dataset and place in ptb-xl/ directory")
                print("   3. Use existing DICOM files for testing")
        except Exception as e:
            print(f"   Error listing directory: {e}")
        sys.exit(1)
    
    rec_base, wfdb_dir, out_file = sys.argv[1:]
    
    # Check if output directory exists
    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
            print(f"📁 Created output directory: {out_dir}")
        except Exception as e:
            print(f"❌ Failed to create output directory {out_dir}: {e}")
            sys.exit(1)
    
    try:
        wfdb_to_dicom(rec_base, wfdb_dir, out_file)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"\n💡 The WFDB files were not found at the specified path.")
        print(f"   Expected location: {os.path.abspath(wfdb_dir)}")
        print(f"   Expected files: {rec_base}.dat and {rec_base}.hea")
        print(f"\n🔍 Please check:")
        print(f"   1. The path '{wfdb_dir}' exists")
        print(f"   2. The record '{rec_base}' exists in that directory")
        print(f"   3. Both .dat and .hea files are present")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 