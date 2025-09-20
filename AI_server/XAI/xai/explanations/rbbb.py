# explanation_rbbb.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks
from ..ecg_utils import plot_ecg_with_grid, load_wfdb_signals, load_dicom_signals


def _refined_qrs_regions(signal, fs, r_idx, is_late_terminal=False):
    """Estimate QRS start/end around each R-peak using prominence-based minima search.
    Returns list of (start_idx, end_idx)."""
    q_search_sec = 0.10
    s_search_sec = 0.15
    q_fallback_sec = 0.05
    # RBBB terminal part is generally shorter than LBBB, here slightly shorter
    s_fallback_sec = 0.08 if not is_late_terminal else 0.09

    ptp = float(np.max(signal) - np.min(signal))
    prom = (0.05 * ptp) if ptp > 0 else 0.01

    regions = []
    for r in r_idx:
        left_win = int(q_search_sec * fs)
        start_l = max(0, r - left_win)
        seg_l = signal[start_l:r]
        q_start = max(0, r - int(q_fallback_sec * fs))
        if seg_l.size > 2:
            inv_l = -seg_l
            pks_l, _ = find_peaks(inv_l, prominence=prom, width=1)
            if pks_l.size > 0:
                q_start = start_l + pks_l[-1]

        right_win = int(s_search_sec * fs)
        r1 = min(r + 1, len(signal) - 1)
        end_r = min(len(signal), r1 + right_win)
        seg_r = signal[r1:end_r]
        q_end = min(len(signal) - 1, r + int(s_fallback_sec * fs))
        if seg_r.size > 2:
            inv_r = -seg_r
            pks_r, _ = find_peaks(inv_r, prominence=prom, width=1)
            if pks_r.size > 0:
                q_end = r1 + pks_r[0]

        if q_start >= q_end:
            half = (int((130 / 1000.0) * fs)) // 2
            q_start = max(0, r - half)
            q_end = min(len(signal) - 1, r + half)
        regions.append((q_start, q_end))
    return regions


def _detect_rsr(seg, fs):
    """Given a segment after R, detect rsR' morphology: at least two positive peaks within 120ms
    and the second peak amplitude >= 30% of the first."""
    pks, _ = find_peaks(seg, distance=fs * 0.02)
    if len(pks) >= 2:
        r1, r2 = seg[pks[0]], seg[pks[1]]
        if r2 > 0.3 * r1:
            return True, pks[1]
    return False, None


def _detect_deep_wide_s(seg):
    valleys, _ = find_peaks(-seg)
    if len(valleys):
        s_amp = -seg[valleys[0]]
        if s_amp > 0.05:  # >0.05 mV deep
            return True, valleys[0]
    return False, None


def _plot_rbbb_lead(t, sig, fs, title, ax, r_idx=None):
    plot_ecg_with_grid(t, sig, title, ax, fs, show_grid=True, style="color")
    if r_idx is not None and len(r_idx):
        ax.scatter(t[r_idx], sig[r_idx], c="dodgerblue", s=30, zorder=5, label='R-peaks')


def _categorize_qrs(duration_s: float) -> str:
    """Clinical bins for RBBB context.
    - narrow/uncertain: <0.06 s (or invalid)
    - normal: 0.06–0.11 s
    - borderline (iRBBB candidate): 0.11–0.12 s
    - wide: ≥0.12 s
    """
    if duration_s is None or duration_s < 0.06:
        return "narrow"
    if duration_s < 0.11:
        return "normal"
    if duration_s < 0.12:
        return "borderline"
    return "wide"


def generate_rbbb_visuals(record_name=None, dicom_filepath=None, db_dir=None, output_dir="."):
    # Leads needed
    need = ["V1", "V2", "V6", "I"]
    leads = {}
    try:
        if hasattr(dicom_filepath, 'WaveformSequence') and dicom_filepath.WaveformSequence:
            # 如果传递的是DICOM数据集对象
            from ..ecg_utils import load_dicom_signals_from_dataset
            raw = load_dicom_signals_from_dataset(dicom_filepath)
            # filter and rename as needed - handle both "V1" and "Lead V1" formats
            for k in list(raw.keys()):
                # Extract base lead name (remove "Lead " prefix if present)
                base_lead = k.replace("Lead ", "") if k.startswith("Lead ") else k
                if base_lead in need:
                    leads[base_lead] = raw[k]
                elif k == "I" or base_lead == "I":
                    leads["I"] = raw[k]
        elif dicom_filepath or (record_name and str(record_name).lower().endswith((".dcm", ".dicom"))):
            # 如果传递的是文件路径
            raw = load_dicom_signals(dicom_filepath or record_name)
            # filter and rename as needed - handle both "V1" and "Lead V1" formats
            for k in list(raw.keys()):
                # Extract base lead name (remove "Lead " prefix if present)
                base_lead = k.replace("Lead ", "") if k.startswith("Lead ") else k
                if base_lead in need:
                    leads[base_lead] = raw[k]
                elif k == "I" or base_lead == "I":
                    leads["I"] = raw[k]
        else:
            leads = load_wfdb_signals(record_name, leads=need, db_dir=db_dir, allow_missing=True)
    except Exception as e:
        print(f"Failed to load ECG for RBBB: {e}")
        leads = {}

    if not leads or ("V1" not in leads and "V2" not in leads):
        print("Missing V1/V2 leads, cannot fully assess for RBBB.")
        return None

    # Use I to replace missing V6
    if "V6" not in leads and "I" in leads:
        leads["V6 (subst. I)"] = leads.pop("I")

    qrs_pool = []
    rsr_count = 0
    deep_s_count = 0
    category_counts = {"narrow": 0, "normal": 0, "borderline": 0, "wide": 0}
    rsr_wide_or_borderline = 0
    deepS_wide_or_borderline = 0

    fig, axes = plt.subplots(len(leads), 1, figsize=(15, 3.8 * len(leads)), sharex=True)
    axes = np.array(axes).ravel()

    for ax, (lead, (sig, fs)) in zip(axes, leads.items()):
        t = np.linspace(0, len(sig) / fs, len(sig), endpoint=False)
        _, rp = nk.ecg_peaks(sig, sampling_rate=fs, method="neurokit")
        r_idx = rp["ECG_R_Peaks"]

        # Precisely estimate QRS start/end for each beat
        regions = _refined_qrs_regions(sig, fs, r_idx, is_late_terminal=False)
        durations = [(end - start) / fs for start, end in regions]
        qrs_pool.extend(durations)

        title = f"{lead} – RBBB features ({record_name})"
        _plot_rbbb_lead(t, sig, fs, title, ax, r_idx)

        # Highlight each QRS region, and color by clinical categories
        for i, ((start, end), dur) in enumerate(zip(regions, durations)):
            cat = _categorize_qrs(dur)
            if cat == 'wide':
                color = 'purple'
            elif cat == 'borderline':
                color = 'orange'
            elif cat == 'normal':
                color = 'green'
            else:
                color = 'lightgray'
            category_counts[cat] += 1
            ax.axvspan(t[start], t[end], alpha=0.28, color=color, ymin=0.32, ymax=0.72)

        # Morphology detection and annotation
        if ("V1" in lead) or ("V2" in lead):
            for k, r in enumerate(r_idx):
                seg = sig[r:r + int(0.12 * fs)]
                ok, r2_idx = _detect_rsr(seg, fs)
                if ok:
                    rsr_count += 1
                    if k < len(durations) and _categorize_qrs(durations[k]) in ("wide", "borderline"):
                        rsr_wide_or_borderline += 1
                    if r2_idx is not None:
                        x = t[min(len(t) - 1, r + r2_idx)]
                        y = sig[min(len(sig) - 1, r + r2_idx)]
                        ax.annotate("R'", (x, y), textcoords="offset points", xytext=(0, -12), ha='center', color='black', fontsize=8,
                                    bbox=dict(fc='yellow', alpha=0.3, pad=1))
                    break

        if ("V6" in lead) or ("I" in lead):
            for k, r in enumerate(r_idx):
                seg = sig[r:r + int(0.12 * fs)]
                ok, s_idx = _detect_deep_wide_s(seg)
                if ok:
                    deep_s_count += 1
                    if k < len(durations) and _categorize_qrs(durations[k]) in ("wide", "borderline"):
                        deepS_wide_or_borderline += 1
                    if s_idx is not None:
                        x = t[min(len(t) - 1, r + s_idx)]
                        y = sig[min(len(sig) - 1, r + s_idx)]
                        ax.annotate("S", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='black', fontsize=8,
                                    bbox=dict(fc='lightcyan', alpha=0.3, pad=1))
                    break

        # Legend (upper right corner of each subplot)
        import matplotlib.patches as mpatches
        handles, _ = ax.get_legend_handles_labels()
        handles.extend([
            mpatches.Patch(color='purple', alpha=0.28, label='Wide QRS (≥0.12s)'),
            mpatches.Patch(color='orange', alpha=0.28, label='Borderline (0.11–0.12s)'),
            mpatches.Patch(color='green', alpha=0.28, label='Normal (0.06–0.11s)'),
            mpatches.Patch(color='lightgray', alpha=0.28, label='Narrow/Uncertain (<0.06s)'),
        ])
        ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rbbb_leads_analysis.png"))
    plt.close(fig)

    qrs_pool = [d for d in qrs_pool if d is not None]
    qrs_mean = float(np.mean(qrs_pool)) if qrs_pool else float("nan")
    prop_wide = float(np.mean(np.array(qrs_pool) > 0.12)) if qrs_pool else float("nan")

    # Histogram (dynamic range, shaded areas)
    if qrs_pool:
        arr = np.array(qrs_pool)
        fig_hist, ax_hist = plt.subplots(figsize=(6.5, 3.6))
        max_v, min_v = float(np.max(arr)), float(np.min(arr))
        hmin = min(0.05, min_v - 0.01 if min_v > 0.05 else 0.04)
        hmax = max(0.25, max_v + 0.02)
        bins = np.arange(hmin, hmax, 0.01) if (hmax - hmin) > 0 else 'auto'
        ax_hist.hist(arr, bins=bins, ec='k', alpha=0.75, color='lightcoral')
        ax_hist.axvspan(0.06, 0.11, color='g', alpha=0.2, label='Normal (0.06–0.11s)')
        ax_hist.axvspan(0.11, 0.12, color='orange', alpha=0.2, label='Borderline (0.11–0.12s)')
        ax_hist.axvline(0.12, c='purple', ls='--', lw=2, label='0.12 s Threshold')
        ax_hist.axvline(qrs_mean, c='r', ls='--', lw=2, label=f'Mean {qrs_mean:.3f}s')
        ax_hist.set_title('QRS Duration Distribution')
        ax_hist.set_xlabel('Duration (s)'); ax_hist.set_ylabel('Count')
        ax_hist.legend(); ax_hist.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rbbb_qrs_histogram.png"))
        plt.close(fig_hist)

    rsr_detected = rsr_count > 0
    wide_s_detected = deep_s_count > 0
    meets_rbbb = qrs_pool and (qrs_mean >= 0.12) and (rsr_wide_or_borderline > 0) and (deepS_wide_or_borderline > 0)
    meets_irbbb = qrs_pool and (0.11 <= qrs_mean < 0.12) and (rsr_wide_or_borderline > 0) and (deepS_wide_or_borderline > 0)
    if meets_rbbb:
        conclusion = "Meets criteria for RBBB"
    elif meets_irbbb:
        conclusion = "Possible incomplete RBBB (borderline QRS with matching morphology)"
    else:
        conclusion = "Incomplete morphology or QRS duration; manual review recommended"

    report_text = "\n".join([
        f"▸ Record: {record_name}",
        f"▸ Average QRS width: {qrs_mean:.3f} s  (>0.12 s proportion: {prop_wide*100:.1f}% if available)",
        f"▸ QRS categories: wide={category_counts['wide']}, borderline={category_counts['borderline']}, normal={category_counts['normal']}, narrow/uncertain={category_counts['narrow']}",
        f"▸ rsR'/M pattern (V1/V2): {'Yes' if rsr_detected else 'No'}  (beats: {rsr_count}, wide/borderline beats: {rsr_wide_or_borderline})",
        f"▸ Wide deep S (V6/I): {'Yes' if wide_s_detected else 'No'}  (beats: {deep_s_count}, wide/borderline beats: {deepS_wide_or_borderline})",
        f"\nSimple Conclusion: {conclusion}"
    ])
    print(report_text)

    metrics = {
        "qrs_mean_s": qrs_mean if not np.isnan(qrs_mean) else None,
        "wide_qrs_proportion": prop_wide if not np.isnan(prop_wide) else None,
        "rsr_pattern_detected": rsr_detected,
        "rsr_beats_count": rsr_count,
        "rsr_beats_wide_or_borderline": rsr_wide_or_borderline,
        "wide_s_wave_detected": wide_s_detected,
        "deep_s_beats_count": deep_s_count,
        "deep_s_beats_wide_or_borderline": deepS_wide_or_borderline,
        "qrs_category_counts": category_counts,
    }
    summary = {
        'report_text': report_text,
        'is_abnormality_present': bool(qrs_pool and (qrs_mean >= 0.12) and (rsr_wide_or_borderline > 0) and (deepS_wide_or_borderline > 0)),
        'abnormality_count': int(np.sum(np.array(qrs_pool) > 0.12)) if qrs_pool else 0,
    }
    statistics = metrics
    return {
        'title': f'RBBB Screening ({record_name})',
        'report_text': report_text,
        'summary': summary,
        'statistics': statistics,
        'image_paths': {
            'rbbb_leads_analysis.png': os.path.join(output_dir, 'rbbb_leads_analysis.png'),
            'rbbb_qrs_histogram.png': os.path.join(output_dir, 'rbbb_qrs_histogram.png')
        },
    }