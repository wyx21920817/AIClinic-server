import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid


def generate_normal_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    """Visualization and simple metrics for Normal ECG (NR).

    Outputs:
    - normal_ecg_overview.png: ECG with R-peaks and a few RR spans
    - normal_rr_histogram.png: RR histogram with normal band shading (0.6–1.0 s)
    - normal_qrs_histogram.png (optional): QRS duration histogram if delineation available
    Returns a report dictionary consistent with other modules.
    """
    if ecg_signal is None:
        return None

    # 1) Clean and R-peaks
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
    r_idx = rpeaks.get("ECG_R_Peaks", np.array([], dtype=int))

    t = np.linspace(0, len(ecg_clean) / sampling_rate, len(ecg_clean), endpoint=False)

    # RR metrics
    rr = np.diff(r_idx) / sampling_rate if len(r_idx) > 1 else np.array([])
    avg_hr = 60.0 / np.mean(rr) if rr.size else 0.0
    rr_cv = float(np.std(rr) / np.mean(rr)) if rr.size else None

    # 2) Plot ECG overview
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(t, ecg_clean, f"normal ECG overview (heart rate: {avg_hr:.0f} bpm) - RR interval regularity ({lead_name})", ax, sampling_rate)
    if len(r_idx):
        ax.scatter(t[r_idx], ecg_clean[r_idx], c='dodgerblue', s=30, zorder=5, label='R-peaks')
        # Highlight a few RR spans
        for i in range(min(len(r_idx) - 1, 5)):
            ax.axvspan(t[r_idx[i]], t[r_idx[i+1]], color='lightgreen', alpha=0.3, ymin=0.45, ymax=0.55)
        # Add RR intervals to legend using matplotlib.patches.Patch
        import matplotlib.patches as mpatches
        rr_legend = mpatches.Patch(color='lightgreen', alpha=0.3, label='RR Intervals: continuous RR intervals (first 5)')
        # Get current legend handles and labels, then add our custom patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(rr_legend)
        ax.legend(handles=handles, loc='upper right')
    else:
        # If no R-peaks detected, still show basic legend
        ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "normal_ecg_overview.png")); plt.close(fig)

    # 3) RR histogram with normal band (0.6–1.0 s)
    fig_h, ax_h = plt.subplots(figsize=(7, 4))
    if rr.size:
        ax_h.hist(rr, bins='auto', ec='k', alpha=0.75, color='teal')
        mean_rr = float(np.mean(rr))
        ax_h.axvline(mean_rr, c='r', ls='--', lw=2, label=f"Mean RR {mean_rr:.3f}s (HR {60/mean_rr:.1f} bpm)")
    ax_h.axvspan(0.6, 1.0, color='green', alpha=0.15, label='normal heart rate range (60–100 bpm)')
    ax_h.set_title('RR Interval Distribution'); ax_h.set_xlabel('RR (s)'); ax_h.set_ylabel('Count'); ax_h.legend()
    ax_h.grid(axis='y', alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "normal_rr_histogram.png")); plt.close(fig_h)

    # 4) Try to estimate QRS durations for a normality check (optional)
    qrs_hist_path = None
    qrs_durations = []
    try:
        _, waves = nk.ecg_delineate(ecg_clean, rpeaks=r_idx, sampling_rate=sampling_rate, method="dwt")
        if waves and 'ECG_QRS_Onsets' in waves and 'ECG_QRS_Offsets' in waves:
            q_on = [int(q) for q in waves['ECG_QRS_Onsets'] if not np.isnan(q)]
            q_off = [int(q) for q in waves['ECG_QRS_Offsets'] if not np.isnan(q)]
            j = 0
            for on in q_on:
                while j < len(q_off) and q_off[j] <= on:
                    j += 1
                if j >= len(q_off):
                    break
                dur = (q_off[j] - on) / sampling_rate
                if 0.03 < dur < 0.30:
                    qrs_durations.append(dur)
                j += 1
    except Exception:
        pass

    if qrs_durations:
        fig_q, ax_q = plt.subplots(figsize=(7, 4))
        arr = np.array(qrs_durations)
        ax_q.hist(arr, bins='auto', ec='k', alpha=0.7, color='lightcoral')
        ax_q.axvspan(0.06, 0.11, color='green', alpha=0.2, label='Normal QRS (0.06–0.11 s)')
        ax_q.axvline(0.12, c='purple', ls='--', lw=2, label='Wide threshold (0.12 s)')
        ax_q.set_title('QRS Duration Distribution'); ax_q.set_xlabel('QRS (s)'); ax_q.set_ylabel('Count'); ax_q.legend()
        ax_q.grid(axis='y', alpha=0.6)
        plt.tight_layout(); qrs_hist_path = os.path.join(output_dir, "normal_qrs_histogram.png"); plt.savefig(qrs_hist_path); plt.close(fig_q)

    # 5) Enhanced Report Structure
    rr_mean = float(np.mean(rr)) if rr.size else None
    rr_std = float(np.std(rr)) if rr.size else None
    
    # Dynamic summary text based on actual measurements
    rhythm_assessment = "highly regular" if rr_cv and rr_cv < 0.08 else "showing some variation"
    basic_summary = f"Baseline rhythm analysis shows: 1) Heart rate is {'normal' if 60 <= avg_hr <= 100 else ('low' if avg_hr < 60 else 'elevated')} (average {avg_hr:.1f} bpm). 2) Rhythm is {rhythm_assessment} (RR coefficient of variation = {rr_cv:.6f}). This provides the baseline context for ECG interpretation." if rr_cv else f"Baseline rhythm analysis shows heart rate of {avg_hr:.1f} bpm. Rhythm assessment requires more data points for accurate evaluation."
    
    # Enhanced structured report sections
    detailed_report_sections = []
    
    # Section 1: Heart Rate Assessment
    hr_status = "Normal" if 60 <= avg_hr <= 100 else ("Low" if avg_hr < 60 else "High")
    hr_section = {
        "section_title": f"1. Heart Rate Assessment: {hr_status} (Average {avg_hr:.1f} bpm)",
        "content": f"The average heart rate is within the normal range. In the [RR Interval Distribution chart], the red dashed line representing the average RR interval ({rr_mean:.3f}s) falls squarely within the green shaded background, which indicates the 'normal heart rate range (60–100 bpm)'."
    }
    detailed_report_sections.append(hr_section)
    
    # Section 2: Rhythm Assessment
    rhythm_status = "Highly Regular" if rr_cv and rr_cv < 0.08 else "Irregular"
    rhythm_section = {
        "section_title": f"2. Rhythm Assessment: {rhythm_status}",
        "content": f"The heart rhythm is {'exceptionally consistent' if rr_cv and rr_cv < 0.08 else 'showing some variation'}. This is visually confirmed in the [ECG Overview chart]: the green boxes marking the 'continuous RR intervals' are of {'nearly identical width' if rr_cv and rr_cv < 0.08 else 'varying widths'}, demonstrating {'a very stable' if rr_cv and rr_cv < 0.08 else 'variable'} time between each heartbeat. This {'high regularity' if rr_cv and rr_cv < 0.08 else 'rhythm variation'} is also confirmed by the coefficient of variation (rr_cv = {rr_cv:.6f})." if rr_cv else "Rhythm assessment requires more data points for accurate evaluation."
    }
    detailed_report_sections.append(rhythm_section)
    
    # Add QRS section if available
    if qrs_durations:
        qrs_mean = np.mean(qrs_durations)
        qrs_normal_count = np.sum((np.array(qrs_durations) >= 0.06) & (np.array(qrs_durations) <= 0.11))
        qrs_status = "Normal" if qrs_mean < 0.12 else "Wide"
        qrs_section = {
            "section_title": f"3. QRS Duration Assessment: {qrs_status} (Average {qrs_mean:.3f}s)",
            "content": f"QRS complex duration analysis shows {'normal' if qrs_mean < 0.12 else 'prolonged'} conduction. {qrs_normal_count} out of {len(qrs_durations)} QRS complexes fall within the normal range (0.06-0.11s), as shown in the [QRS Duration Distribution chart] with green shading."
        }
        detailed_report_sections.append(qrs_section)
    
    # Summary structure
    summary = {
        'report_text': basic_summary,
        'is_abnormality_present': False,
        'abnormality_count': 0,
    }
    
    # Statistics structure  
    statistics = {
        "average_heart_rate_bpm": float(avg_hr),
        "rr_mean_s": rr_mean,
        "rr_std_s": rr_std,
        "rr_cv": rr_cv,
    }
    if qrs_durations:
        statistics.update({
            "qrs_mean_s": float(np.mean(qrs_durations)),
            "qrs_in_normal_band_count": int(np.sum((np.array(qrs_durations) >= 0.06) & (np.array(qrs_durations) <= 0.11)))
        })
    
    # Note: Validation will be handled by validate_nr() in validators.py
    # which will override any validation structure we set here

    image_paths = {
        "normal_ecg_overview.png": os.path.join(output_dir, "normal_ecg_overview.png"),
        "normal_rr_histogram.png": os.path.join(output_dir, "normal_rr_histogram.png")
    }
    if qrs_hist_path is not None:
        image_paths["normal_qrs_histogram.png"] = os.path.join(output_dir, "normal_qrs_histogram.png")

    return {
        "title": f"ECG Analysis Report: Normal Sinus Rhythm",
        "lead_name": lead_name,
        "summary": summary,
        "report_text": detailed_report_sections,
        "statistics": statistics,
        "image_paths": image_paths,
    }


