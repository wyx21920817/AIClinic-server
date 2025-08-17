
# ecg_utils.py 

import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import pydicom

# Default data directory: project located in XAI/data/ecg_files (i.e. XAI/data/ecg_files)
# This file is located in XAI/xai/ecg_utils.py, so take the上一层目录
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ecg_files")

def _resolve_path(path_like: str) -> str:
    """Resolve relative paths to default data directory XAI/ECG_File (if it exists).
    - If path_like is an absolute path or already exists, return it as is
    - Otherwise, try to join with DEFAULT_DATA_DIR
    """
    if not path_like:
        return path_like
    if os.path.isabs(path_like) or os.path.exists(path_like):
        return path_like
    candidate = os.path.join(DEFAULT_DATA_DIR, path_like)
    return candidate if os.path.exists(candidate) else path_like

def plot_ecg_with_grid(time, signal, title, ax, sampling_rate, show_grid=True, style='color'):
    """Plot ECG signal with grid."""
    if style == 'bw':
        color_major, color_minor, color_line = (0.4, 0.4, 0.4), (0.75, 0.75, 0.75), (0, 0, 0)
    else:
        color_major, color_minor, color_line = (1, 0, 0), (1, 0.7, 0.7), (0, 0, 0.7)

    ax.plot(time, signal, color=color_line, linewidth=1.0)
    ax.set_title(title, fontsize=14); ax.set_xlabel('Time (s)', fontsize=12); ax.set_ylabel('Amplitude (mV)', fontsize=12)

    if show_grid and time is not None and signal is not None and len(time) > 0 and len(signal) > 0:
        x_min, x_max = np.min(time), np.max(time)
        y_min, y_max = np.min(signal), np.max(signal)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min - 0.5, y_max + 0.5)
        x_min_plot, x_max_plot = ax.get_xlim(); y_min_plot, y_max_plot = ax.get_ylim()

        major_xticks = np.arange(np.floor(x_min_plot/0.2)*0.2, np.ceil(x_max_plot/0.2)*0.2+0.01, 0.2)
        major_yticks = np.arange(np.floor(y_min_plot/0.5)*0.5, np.ceil(y_max_plot/0.5)*0.5+0.01, 0.5)
        ax.set_xticks(major_xticks); ax.set_yticks(major_yticks)

        minor_xticks = np.arange(np.floor(x_min_plot/0.04)*0.04, np.ceil(x_max_plot/0.04)*0.04+0.01, 0.04)
        minor_yticks = np.arange(np.floor(y_min_plot/0.1)*0.1, np.ceil(y_max_plot/0.1)*0.1+0.01, 0.1)
        ax.set_xticks(minor_xticks, minor=True); ax.set_yticks(minor_yticks, minor=True)

        ax.grid(which='major', linestyle='-', linewidth=0.5, color=color_major, alpha=0.7)
        ax.grid(which='minor', linestyle='-', linewidth=0.25, color=color_minor, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)

def load_wfdb_signal(filename, lead_index, db_dir=None):
    """Load specified lead from local WFDB record."""
    try:
        # First parse db_dir; if not provided, default to ECG_File
        base_dir = db_dir or DEFAULT_DATA_DIR
        record_path = os.path.join(base_dir, filename) if base_dir else filename
        record_path = _resolve_path(record_path)
        # Strictly local: only read when .hea/.dat exists locally, avoid any network access
        hea_path = record_path + '.hea'
        dat_path = record_path + '.dat'
        if os.path.exists(hea_path) or os.path.exists(dat_path):
            record = wfdb.rdrecord(record_path)
        else:
            alt_path = filename
            if os.path.isabs(alt_path):
                alt_candidate = alt_path
            else:
                alt_candidate = _resolve_path(alt_path)
            if os.path.exists(alt_candidate + '.hea') or os.path.exists(alt_candidate + '.dat'):
                record = wfdb.rdrecord(alt_candidate)
            else:
                raise FileNotFoundError(f"WFDB files not found locally for '{filename}'. Tried: {hea_path}, {dat_path}")
        ecg_signal = record.p_signal[:, lead_index]
        sampling_rate = record.fs
        sig_name = record.sig_name[lead_index]
        if record.units[lead_index].lower() == 'uv':
            ecg_signal = ecg_signal / 1000.0
        print(f"Loaded WFDB {filename}, Lead: {sig_name}, Fs: {sampling_rate} Hz")
        return ecg_signal, sampling_rate, sig_name
    except Exception as e:
        print(f"Error loading WFDB record {filename} from path '{record_path}': {e}")
        return None, None, None


def load_wfdb_signals(filename, leads="all", db_dir=None, allow_missing=True):
    """Load multiple specified leads from local WFDB record."""
    try:
        # Ensure loading from local path
        base_dir = db_dir or DEFAULT_DATA_DIR
        record_path = os.path.join(base_dir, filename) if base_dir else filename
        record_path = _resolve_path(record_path)
        hea_path = record_path + '.hea'
        dat_path = record_path + '.dat'
        if os.path.exists(hea_path) or os.path.exists(dat_path):
            record = wfdb.rdrecord(record_path)
        else:
            alt_path = filename
            alt_candidate = _resolve_path(alt_path)
            if os.path.exists(alt_candidate + '.hea') or os.path.exists(alt_candidate + '.dat'):
                record = wfdb.rdrecord(alt_candidate)
            else:
                raise FileNotFoundError(f"WFDB files not found locally for '{filename}'. Tried: {hea_path}, {dat_path}")
        
        fs, names = record.fs, record.sig_name
        units = [u.lower() for u in record.units]
        
        leads_to_load = names if leads == "all" else ([leads] if not isinstance(leads, (list, tuple)) else leads)
        
        out = {}
        for lead_item in leads_to_load:
            idx = next((i for i, name in enumerate(names) if str(lead_item).lower() == name.lower()), None)
            if idx is not None:
                sig = record.p_signal[:, idx]
                if units[idx] == "uv":
                    sig = sig / 1000.0
                out[names[idx]] = (sig, fs)
            elif not allow_missing:
                raise ValueError(f"Lead '{lead_item}' not found in record.")
        return out
    except Exception as e:
        print(f"Error loading multi-lead WFDB record {filename}: {e}")
        return {}
        

def load_dicom_signal(filepath, lead_name_or_index):
    """Load specified lead ECG signal from DICOM file, following DICOM Waveform demultiplexing/quantization rules.
    Key points:
    - Demultiplexing: (num_samples, num_channels)
    - Data type: based on WaveformSampleInterpretation/BitsAllocated
    - Quantization: ChannelSensitivity × ChannelSensitivityCorrectionFactor (+ ChannelBaseline)
    - Unit: usually uV → converted to mV
    """
    try:
        ds = pydicom.dcmread(_resolve_path(filepath))
        if "WaveformSequence" not in ds or not ds.WaveformSequence:
            raise ValueError("DICOM does not contain WaveformSequence.")

        wf = ds.WaveformSequence[0]
        fs = float(wf.SamplingFrequency)
        num_channels = int(wf.NumberOfWaveformChannels)
        num_samples = int(wf.NumberOfWaveformSamples)
        channel_defs = wf.ChannelDefinitionSequence

        # 1) Parse lead names
        lead_names = []
        try:
            lead_names = [ch.ChannelSourceSequence[0].CodeMeaning for ch in channel_defs]
        except Exception:
            std = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            lead_names = std[:num_channels] if num_channels <= len(std) else [f"Lead {i+1}" for i in range(num_channels)]

        if not lead_names:
            raise ValueError("Unable to resolve lead names from DICOM.")

        # 2) Parse data type and demultiplex
        interp = getattr(wf, "WaveformSampleInterpretation", "SS").upper()
        bits = int(getattr(wf, "WaveformBitsAllocated", 16))
        little = getattr(ds, "is_little_endian", True)

        dtype_map = {
            ("SS", 16): np.int16,
            ("US", 16): np.uint16,
            ("SB", 8): np.int8,
            ("UB", 8): np.uint8,
            ("SL", 32): np.int32,
            ("UL", 32): np.uint32,
            ("FD", 64): np.float64,
            ("FL", 32): np.float32,
        }
        dtype = dtype_map.get((interp, bits), np.int16)
        dtype = np.dtype(dtype).newbyteorder('<' if little else '>')

        raw = np.frombuffer(wf.WaveformData, dtype=dtype)
        if raw.size != num_channels * num_samples:
            # Some files may contain multiple frames, here try to infer by channels
            if raw.size % num_channels != 0:
                raise ValueError(f"WaveformData size mismatch: {raw.size} vs channels*samples {num_channels*num_samples}")
            num_samples = raw.size // num_channels
        data = raw.reshape(num_samples, num_channels)

        # 3) Match lead index
        lead_idx = None
        if isinstance(lead_name_or_index, int):
            if 0 <= lead_name_or_index < num_channels:
                lead_idx = int(lead_name_or_index)
        else:
            search = str(lead_name_or_index).lower().replace("lead", "").strip()
            for i, nm in enumerate(lead_names):
                if search == nm.lower().replace("lead", "").strip():
                    lead_idx = i; break
        if lead_idx is None:
            raise ValueError(f"Lead '{lead_name_or_index}' not found. Available: {lead_names}")

        # 4) Quantization and unit
        ch_def = channel_defs[lead_idx]
        sens = float(getattr(ch_def, "ChannelSensitivity", 1.0))
        corr = float(getattr(ch_def, "ChannelSensitivityCorrectionFactor", 1.0))
        base = float(getattr(ch_def, "ChannelBaseline", 0.0))
        units = "uV"
        try:
            units = ch_def.ChannelSensitivityUnitsSequence[0].CodeValue
        except Exception:
            pass

        sig = data[:, lead_idx].astype(np.float64)
        sig_uv = (sig * sens * corr) + base
        # Common unit uV → mV
        if isinstance(units, str) and units.lower().find('uv') != -1:
            sig_mv = sig_uv / 1000.0
        else:
            # If unit is unknown, still assume uV to avoid large values
            sig_mv = sig_uv / 1000.0

        lead_name = lead_names[lead_idx]
        print(f"Loaded DICOM {os.path.basename(filepath)}, Lead: {lead_name}, Fs: {fs} Hz")
        return sig_mv, fs, lead_name
    except Exception as e:
        print(f"Error loading DICOM ECG from {filepath}: {e}")
        return None, None, None

def load_dicom_signals_from_dataset(dicom_dataset):
    """Load all ECG leads from a DICOM dataset object.

    Returns a dict mapping lead names to (signal_mV, fs).
    Lead names are resolved from ChannelSourceSequence.CodeMeaning when possible.
    """
    try:
        if "WaveformSequence" not in dicom_dataset or not dicom_dataset.WaveformSequence:
            raise ValueError("DICOM dataset does not contain WaveformSequence.")

        wf = dicom_dataset.WaveformSequence[0]
        fs = float(wf.SamplingFrequency)
        num_channels = int(wf.NumberOfWaveformChannels)
        num_samples = int(wf.NumberOfWaveformSamples)
        channel_defs = wf.ChannelDefinitionSequence

        # Lead names
        lead_names = []
        try:
            lead_names = [ch.ChannelSourceSequence[0].CodeMeaning for ch in channel_defs]
        except Exception:
            std = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            lead_names = std[:num_channels] if num_channels <= len(std) else [f"Lead {i+1}" for i in range(num_channels)]

        if not lead_names:
            raise ValueError("Unable to resolve lead names from DICOM dataset.")

        # Data type and demux
        interp = getattr(wf, "WaveformSampleInterpretation", "SS").upper()
        bits = int(getattr(wf, "WaveformBitsAllocated", 16))
        little = getattr(dicom_dataset, "is_little_endian", True)
        dtype_map = {
            ("SS", 16): np.int16,
            ("US", 16): np.uint16,
            ("SB", 8): np.int8,
            ("UB", 8): np.uint8,
            ("SL", 32): np.int32,
            ("UL", 32): np.uint32,
            ("FD", 64): np.float64,
            ("FL", 32): np.float32,
        }
        dtype = np.dtype(dtype_map.get((interp, bits), np.int16)).newbyteorder('<' if little else '>')

        raw = np.frombuffer(wf.WaveformData, dtype=dtype)
        if raw.size != num_channels * num_samples:
            if raw.size % num_channels != 0:
                raise ValueError(f"WaveformData size mismatch: {raw.size} vs channels*samples {num_channels*num_samples}")
            num_samples = raw.size // num_channels
        data = raw.reshape(num_samples, num_channels)

        out = {}
        for i in range(num_channels):
            ch_def = channel_defs[i]
            sens = float(getattr(ch_def, "ChannelSensitivity", 1.0))
            corr = float(getattr(ch_def, "ChannelSensitivityCorrectionFactor", 1.0))
            base = float(getattr(ch_def, "ChannelBaseline", 0.0))
            units = None
            try:
                units = ch_def.ChannelSensitivityUnitsSequence[0].CodeValue
            except Exception:
                units = "uV"

            sig = data[:, i].astype(np.float64)
            sig_uv = (sig * sens * corr) + base
            sig_mv = sig_uv / 1000.0 if isinstance(units, str) and units.lower().find('uv') != -1 else sig_uv / 1000.0
            out[lead_names[i]] = (sig_mv, fs)

        print(f"Loaded DICOM dataset with {len(out)} channels at {fs} Hz")
        return out
    except Exception as e:
        print(f"Error loading DICOM ECG (multi-lead) from dataset: {e}")
        return {}


def load_dicom_signals(filepath):
    """Load all ECG leads from a DICOM waveform file.

    Returns a dict mapping lead names to (signal_mV, fs).
    Lead names are resolved from ChannelSourceSequence.CodeMeaning when possible.
    """
    try:
        ds = pydicom.dcmread(_resolve_path(filepath))
        if "WaveformSequence" not in ds or not ds.WaveformSequence:
            raise ValueError("DICOM does not contain WaveformSequence.")

        wf = ds.WaveformSequence[0]
        fs = float(wf.SamplingFrequency)
        num_channels = int(wf.NumberOfWaveformChannels)
        num_samples = int(wf.NumberOfWaveformSamples)
        channel_defs = wf.ChannelDefinitionSequence

        # Lead names
        lead_names = []
        try:
            lead_names = [ch.ChannelSourceSequence[0].CodeMeaning for ch in channel_defs]
        except Exception:
            std = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            lead_names = std[:num_channels] if num_channels <= len(std) else [f"Lead {i+1}" for i in range(num_channels)]

        if not lead_names:
            raise ValueError("Unable to resolve lead names from DICOM.")

        # Data type and demux
        interp = getattr(wf, "WaveformSampleInterpretation", "SS").upper()
        bits = int(getattr(wf, "WaveformBitsAllocated", 16))
        little = getattr(ds, "is_little_endian", True)
        dtype_map = {
            ("SS", 16): np.int16,
            ("US", 16): np.uint16,
            ("SB", 8): np.int8,
            ("UB", 8): np.uint8,
            ("SL", 32): np.int32,
            ("UL", 32): np.uint32,
            ("FD", 64): np.float64,
            ("FL", 32): np.float32,
        }
        dtype = np.dtype(dtype_map.get((interp, bits), np.int16)).newbyteorder('<' if little else '>')

        raw = np.frombuffer(wf.WaveformData, dtype=dtype)
        if raw.size != num_channels * num_samples:
            if raw.size % num_channels != 0:
                raise ValueError(f"WaveformData size mismatch: {raw.size} vs channels*samples {num_channels*num_samples}")
            num_samples = raw.size // num_channels
        data = raw.reshape(num_samples, num_channels)

        out = {}
        for i in range(num_channels):
            ch_def = channel_defs[i]
            sens = float(getattr(ch_def, "ChannelSensitivity", 1.0))
            corr = float(getattr(ch_def, "ChannelSensitivityCorrectionFactor", 1.0))
            base = float(getattr(ch_def, "ChannelBaseline", 0.0))
            units = None
            try:
                units = ch_def.ChannelSensitivityUnitsSequence[0].CodeValue
            except Exception:
                units = "uV"

            sig = data[:, i].astype(np.float64)
            sig_uv = (sig * sens * corr) + base
            sig_mv = sig_uv / 1000.0 if isinstance(units, str) and units.lower().find('uv') != -1 else sig_uv / 1000.0
            out[lead_names[i]] = (sig_mv, fs)

        print(f"Loaded DICOM {os.path.basename(filepath)} with {len(out)} channels at {fs} Hz")
        return out
    except Exception as e:
        print(f"Error loading DICOM ECG (multi-lead) from {filepath}: {e}")
        return {}

def load_ecg_data(config_task):
    """Load ECG data per task.

    Returns either:
    - (signal, fs, lead_name) for single-lead workflows, or
    - {lead_name: (signal, fs), ...} for multi-lead workflows when 'require_multi_lead' is truthy.

    Supported fields (by priority):
    - instance_id:
        * If path ends with .dcm/.dicom, treat as DICOM absolute/relative path
        * If dicom_dir is provided, try dicom_dir/<instance_id>.dcm
        * Otherwise, use as WFDB record name (combined with db_dir)
    - filepath: DICOM file path (compatible with old interface)
    - record_name: WFDB record name or DICOM path (compatible with old interface)
    """
    instance_id = config_task.get("instance_id")
    record_name = config_task.get("record_name")
    filepath = config_task.get("filepath")
    lead_info = config_task.get("lead_index", 1)
    db_dir = config_task.get("db_dir") or DEFAULT_DATA_DIR
    dicom_dir = config_task.get("dicom_dir")

    require_multi = bool(config_task.get("require_multi_lead"))

    # 1) Unified instance_id process
    if instance_id:
        # 1.1 Multi-lead required
        if require_multi:
            # DICOM path directly provided
            if isinstance(instance_id, str) and instance_id.lower().endswith((".dcm", ".dicom")):
                return load_dicom_signals(_resolve_path(instance_id))
            # DICOM via dicom_dir
            if dicom_dir:
                candidate = os.path.join(dicom_dir, f"{instance_id}.dcm")
                resolved = _resolve_path(candidate)
                if os.path.exists(resolved):
                    return load_dicom_signals(resolved)
            # WFDB multi-lead
            leads = config_task.get("leads") or "all"
            return load_wfdb_signals(instance_id, leads=leads, db_dir=db_dir)

        # 1.2 Single-lead
        if isinstance(instance_id, str) and instance_id.lower().endswith((".dcm", ".dicom")):
            return load_dicom_signal(_resolve_path(instance_id), lead_info)
        if dicom_dir:
            candidate = os.path.join(dicom_dir, f"{instance_id}.dcm")
            resolved = _resolve_path(candidate)
            if os.path.exists(resolved):
                return load_dicom_signal(resolved, lead_info)
        if isinstance(lead_info, list):
            return load_wfdb_signals(instance_id, leads=lead_info, db_dir=db_dir)
        return load_wfdb_signal(instance_id, lead_info, db_dir=db_dir)

    # 2) Compatible with old fields
    if filepath and filepath.lower().endswith((".dcm", ".dicom")):
        return load_dicom_signals(_resolve_path(filepath)) if require_multi else load_dicom_signal(_resolve_path(filepath), lead_info)
    if record_name and record_name.lower().endswith((".dcm", ".dicom")):
        return load_dicom_signals(_resolve_path(record_name)) if require_multi else load_dicom_signal(_resolve_path(record_name), lead_info)
    if record_name:
        if require_multi:
            leads = config_task.get("leads") or "all"
            return load_wfdb_signals(record_name, leads=leads, db_dir=db_dir)
        if isinstance(lead_info, list):
            return load_wfdb_signals(record_name, leads=lead_info, db_dir=db_dir)
        return load_wfdb_signal(record_name, lead_info, db_dir=db_dir)

    print("Task config requires 'instance_id' (preferred), or 'filepath' (DICOM) / 'record_name' (WFDB).")
    return None, None, None