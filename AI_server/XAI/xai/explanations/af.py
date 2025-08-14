# explanation_af.py
import os, json, numpy as np, matplotlib.pyplot as plt, neurokit2 as nk
from xai.ecg_utils import plot_ecg_with_grid
from xai.clinical_reports import generate_report_afib

def generate_af_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    if ecg_signal is None: return None
    time_axis = np.linspace(0, len(ecg_signal) / sampling_rate, len(ecg_signal), endpoint=False)
    
    try:
        _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
        r_peak_indices = r_peaks['ECG_R_Peaks']
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}"); return None

    # Plot 1: ECG with R-peaks
    fig_ecg, ax_ecg = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(time_axis, ecg_signal, f'ECG {lead_name} - {ecg_title}', ax_ecg, sampling_rate)
    ax_ecg.scatter(time_axis[r_peak_indices], ecg_signal[r_peak_indices], color='dodgerblue', marker='o', s=30, label='R-peaks', zorder=5)
    ax_ecg.legend(); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "afib_ecg_with_rpeaks.png")); plt.close(fig_ecg)

    rr_intervals_sec = np.diff(r_peak_indices) / sampling_rate if len(r_peak_indices) > 1 else np.array([])
    
    # Plot 2: RR Interval Histogram
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    if rr_intervals_sec.size > 0:
        ax_hist.hist(rr_intervals_sec, bins='auto', edgecolor='black', alpha=0.7)
    ax_hist.set_title(f'RR Interval Histogram - {ecg_title}'); ax_hist.set_xlabel('RR Interval (s)'); ax_hist.set_ylabel('Frequency')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "afib_rr_histogram.png")); plt.close(fig_hist)
    
    # Plot 3: Inferred P-wave Regions (best-effort with grid)
    # Try delineation to obtain QRS onsets; fall back to fixed window before R
    qrs_onsets_from_delineation = []
    try:
        _, waves = nk.ecg_delineate(ecg_signal, rpeaks=r_peak_indices, sampling_rate=sampling_rate, method="dwt")
        if waves and 'ECG_QRS_Onsets' in waves:
            qrs_onsets_from_delineation = [int(q) for q in waves['ECG_QRS_Onsets'] if not np.isnan(q)]
    except Exception as e:
        print(f"Delineation failed for {ecg_title}: {e}.")

    fig_pwave, ax_pwave = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(time_axis, ecg_signal, f'ECG {lead_name} with Inferred P-wave Regions - {ecg_title}', ax_pwave, sampling_rate)
    ax_pwave.scatter(time_axis[r_peak_indices], ecg_signal[r_peak_indices], color='dodgerblue', marker='o', s=30, label='R-peaks', zorder=5)

    p_wave_typical_duration_sec = 0.10
    highlighted_p_region = False
    for r_peak_idx in r_peak_indices:
        potential_onsets = [q for q in qrs_onsets_from_delineation if q < r_peak_idx]
        if potential_onsets:
            qrs_onset = max(potential_onsets)
        else:
            qrs_onset = max(0, r_peak_idx - int(0.04 * sampling_rate))

        p_box_end_idx = qrs_onset
        p_box_start_idx = max(0, p_box_end_idx - int(p_wave_typical_duration_sec * sampling_rate))
        if p_box_start_idx < p_box_end_idx and p_box_end_idx < len(time_axis):
            label = 'Inferred P-wave Region' if not highlighted_p_region else ""
            ax_pwave.axvspan(time_axis[p_box_start_idx], time_axis[p_box_end_idx], color='yellow', alpha=0.3, label=label)
            highlighted_p_region = True

    if highlighted_p_region:
        ax_pwave.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "afib_pwave_regions.png")); plt.close(fig_pwave)

    report_text = generate_report_afib(ecg_title, lead_name)
    print(report_text)
    
    metrics = {"error": "Not enough R-peaks."} if rr_intervals_sec.size == 0 else {
        "rr_min_s": float(np.min(rr_intervals_sec)), "rr_max_s": float(np.max(rr_intervals_sec)),
        "rr_mean_s": float(np.mean(rr_intervals_sec)), "rr_std_s": float(np.std(rr_intervals_sec)),
        "rr_interval_count": len(rr_intervals_sec)
    }

    return {"title": ecg_title, "lead_name": lead_name, "report_text": report_text, "metrics": metrics,
            "image_paths": ["afib_ecg_with_rpeaks.png", "afib_rr_histogram.png", "afib_pwave_regions.png"]}