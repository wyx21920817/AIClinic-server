# wfdb2dcm_correct_official.py
# python wfdb2dcm.py 00556_lr data/ptbxl/records100/00000 00556_lr.dcm

"""
Correct version based on pydicom official documentation
"""
import sys, datetime
import numpy as np
import wfdb
import pydicom
import os
import pandas as pd
from pathlib import Path
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

def load_ptbxl_database():
    """Load PTBXL database and create filename_lr to patient_id mapping"""
    db_path = Path("data/ptbxl/ptbxl_database.csv")
    
    if not db_path.exists():
        print(f"❌ PTBXL database not found: {db_path}")
        print("Please ensure ptbxl_database.csv is in data/ptbxl/ directory")
        return None
    
    try:
        df = pd.read_csv(db_path)
        print(f"📊 Loaded PTBXL database: {len(df)} records")
        
        # Create filename_lr to patient_id mapping
        # Extract filename part from filename_lr (remove path prefix)
        filename_to_patient = {}
        for idx, row in df.iterrows():
            filename_lr = row['filename_lr']
            if pd.notna(filename_lr):
                # Extract filename part, e.g., "records100/00000/00556_lr" -> "00556_lr"
                filename = filename_lr.split('/')[-1]
                filename_to_patient[filename] = row['patient_id']
        
        print(f"   Created mapping for {len(filename_to_patient)} filename records")
        
        return filename_to_patient
    except Exception as e:
        print(f"❌ Error loading PTBXL database: {e}")
        return None

def wfdb_to_dicom_correct_official(record_name: str, wfdb_dir: str, out_path: str, patient_name: str = None):
    # --- 1. Read WFDB ---
    rec_path = os.path.join(wfdb_dir, record_name)
    rec = wfdb.rdrecord(rec_path)
    sig = rec.p_signal.astype("float32")
    fs = rec.fs
    leads = rec.sig_name
    n_samp, n_chan = sig.shape

    # --- 2. Extract filename and get Patient ID from database ---
    try:
        # record_name is filename_lr (e.g., "00556_lr")
        filename = record_name
        
        # Load PTBXL database mapping
        ptbxl_mapping = load_ptbxl_database()
        
        if ptbxl_mapping and filename in ptbxl_mapping:
            patient_id = ptbxl_mapping[filename]
            # Convert patient_id to string, handle possible float values
            if isinstance(patient_id, float) and not pd.isna(patient_id):
                patient_id = str(int(patient_id))
            elif pd.isna(patient_id):
                patient_id = "Unknown"
            else:
                patient_id = str(patient_id)
            
            print(f"📋 Patient Mapping:")
            print(f"   Filename: {filename}")
            print(f"   Patient ID: {patient_id}")
        else:
            print(f"⚠️ Filename '{filename}' not found in PTBXL database, using fallback")
            patient_id = "Unknown"
            
    except Exception as e:
        print(f"⚠️ Error loading PTBXL database: {e}, using fallback")
        patient_id = "Unknown"

    print(f"📊 Original Data Info:")
    print(f"   Shape: {sig.shape}")
    print(f"   Sampling Rate: {fs} Hz")
    print(f"   Leads: {leads}")
    print(f"   Data Range: [{sig.min():.3f}, {sig.max():.3f}] mV")

    # --- 3. Data Conversion ---
    sig_uv = sig * 1000.0  # mV -> µV
    sig_uv = np.clip(sig_uv, -32767, 32767)
    sig_int16 = np.round(sig_uv).astype(np.int16)
    
    print(f"   Converted Range: [{sig_int16.min()}, {sig_int16.max()}] (int16)")

    # --- 4. DICOM File Meta Information ---
    sop_instance_uid = pydicom.uid.generate_uid()
    print(f"SOP Instance UID: {sop_instance_uid}")
    
    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.1000"

    # --- 5. Main Dataset ---
    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # SOP Tags
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    ds.SOPInstanceUID = sop_instance_uid
    
    # Patient/Study Info - Use patient_name from parameter or default
    if patient_name:
        ds.PatientName = patient_name
    else:
        ds.PatientName = "Anon^PTBXL"  # Default fallback
    
    ds.PatientID = patient_id  # Use patient_id mapped from database
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    
    now = datetime.datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    
    ds.StudyID = "1"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    
    ds.Modality = "ECG"
    ds.Manufacturer = "PTBXL-Converter"

    # --- 6. Waveform Sequence - Based on official documentation ---
    wv_item = Dataset()
    
    # Basic Waveform Info
    wv_item.WaveformOriginality = "ORIGINAL"
    wv_item.NumberOfWaveformChannels = n_chan
    wv_item.NumberOfWaveformSamples = n_samp
    wv_item.SamplingFrequency = float(fs)
    
    # Time related parameters
    wv_item.MultiplexGroupTimeOffset = 0.0
    wv_item.TriggerTimeOffset = 0.0
    
    # Complete Waveform bit definitions
    wv_item.WaveformBitsAllocated = 16
    wv_item.WaveformSampleInterpretation = "SS"
    
    # Interleave data and convert to bytes
    # Reshape data to (n_samp, n_chan) then convert to bytes
    arr = sig_int16.reshape(n_samp, n_chan)
    wf_bytes = arr.tobytes()
    wv_item.WaveformData = wf_bytes

    # --- 7. Channel Definition Sequence - Based on official documentation format ---
    ch_seq = []
    
    # Standard lead code mapping
    lead_codes = {
        'I': ('5.6.3-9-1', 'Lead I (Einthoven)'),
        'II': ('5.6.3-9-2', 'Lead II'),
        'III': ('5.6.3-9-61', 'Lead III'),
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
    
    for ch_idx, lead in enumerate(leads):
        chdef = Dataset()
        
        # According to official documentation, set WaveformBitsStored in channel definition
        chdef.ChannelSampleSkew = 0.0
        chdef.WaveformBitsStored = 16
        
        # Lead code sequence
        chdef.ChannelSourceSequence = [Dataset()]
        source = chdef.ChannelSourceSequence[0]
        
        std_lead = lead.upper()
        if std_lead in lead_codes:
            source.CodeValue = lead_codes[std_lead][0]
            source.CodingSchemeDesignator = "SCPECG"
            source.CodingSchemeVersion = "1.3"
            source.CodeMeaning = lead_codes[std_lead][1]
        else:
            source.CodeValue = "1.0"
            source.CodingSchemeDesignator = "PYDICOM"
            source.CodingSchemeVersion = "1.0"
            source.CodeMeaning = lead
        
        # Calibration info (Type 1C - Optional)
        chdef.ChannelSensitivity = 1.0
        chdef.ChannelSensitivityUnitsSequence = [Dataset()]
        units_item = chdef.ChannelSensitivityUnitsSequence[0]
        units_item.CodeValue = "uV"
        units_item.CodingSchemeDesignator = "UCUM"
        units_item.CodeMeaning = "microvolt"
        
        ch_seq.append(chdef)
    
    wv_item.ChannelDefinitionSequence = Sequence(ch_seq)
    ds.WaveformSequence = Sequence([wv_item])

    # --- 8. Save and Verify ---
    try:
        ds.save_as(out_path, write_like_original=False)
        
        # Verify file
        verify_ds = pydicom.dcmread(out_path)
        ws = verify_ds.WaveformSequence[0]
        
        print(f"DICOM ECG saved successfully: {out_path}")
        print(f"   Patient ID: {verify_ds.PatientID}")
        print(f"   WaveformBitsAllocated: {ws.WaveformBitsAllocated}")
        print(f"   WaveformSampleInterpretation: {ws.WaveformSampleInterpretation}")
        
        # Check WaveformBitsStored in channel definition
        ch = ws.ChannelDefinitionSequence[0]
        print(f"   Channel WaveformBitsStored: {ch.WaveformBitsStored}")
        
        # Verify data
        expected_size = n_samp * n_chan * 2
        actual_size = len(ws.WaveformData)
        print(f"   Data size: {actual_size} bytes (Expected: {expected_size})")
        
        if actual_size == expected_size:
            print("   Data integrity verification passed")
        
        # Use pydicom's official method to verify data decoding
        try:
            decoded_arr = verify_ds.waveform_array(0)
            print(f"   Data decoding successful: {decoded_arr.shape}")
            print(f"   Decoded range: [{decoded_arr.min():.1f}, {decoded_arr.max():.1f}]")
        except Exception as decode_error:
            print(f"   Data decoding failed: {decode_error}")
        
        return True
        
    except Exception as e:
        print(f"Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python wfdb2dcm-PTBXL.py <record_name> <wfdb_dir> <output.dcm>")
        print("Example: python wfdb2dcm-PTBXL.py 12767_lr data/ptbxl/records100/12000 STACH_12767_lr.dcm")
        print("")
        print("Note: PatientName will be automatically set to the output filename without extension")
        print("      For example: STACH_12767_lr.dcm -> PatientName = 'STACH_12767_lr'")
        sys.exit(1)
    
    rec_base, wfdb_dir, out_file = sys.argv[1:]
    
    # Extract PatientName from output filename (remove .dcm extension)
    patient_name = Path(out_file).stem  # Gets filename without extension
    
    print(f"PTBXL Specialized Version - Custom PatientName Support")
    print(f"   Converting: {rec_base} -> {out_file}")
    print(f"   PatientName: {patient_name}")
    
    success = wfdb_to_dicom_correct_official(rec_base, wfdb_dir, out_file, patient_name)
    
    if success:
        print(f"\n✅ Conversion successful!")
        print(f"   🔧 Key improvements:")
        print(f"   • Read Patient ID mapping from ptbxl_database.csv")
        print(f"   • Automatically set PatientName from output filename")
        print(f"   • Automatically set correct Patient ID based on filename")
        print(f"   • Maintain all other DICOM standard fields")
        print(f"\n📋 Verification:")
        print(f"   1. Check if PatientName in generated DICOM file is: {patient_name}")
        print(f"   2. Check if Patient ID in generated DICOM file is correct")
        print(f"   3. Confirm ECG data integrity")
    else:
        print(f"\n❌ Conversion failed")