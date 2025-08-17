import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid
from ..clinical_reports import generate_report_1davb

def generate_avb_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    """
    Generate and save visualizations for first-degree AV block.
    This version strictly follows the validated Jupyter code logic to ensure absolute correctness.
    """
    if ecg_signal is None: return None

    # 1. Signal cleaning (consistent with Jupyter)
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")

    # 2. R-wave detection (consistent with Jupyter)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
    r_peak_indices = rpeaks['ECG_R_Peaks']

    # 3. Try multiple drawing methods in order (consistent with Jupyter)
    waves, used_method = None, None
    for method in ["prominence", "dwt", "cwt", "peak"]:
        try:
            print(f"--- Trying method: '{method}' ---") # Add debugging information
            _, w = nk.ecg_delineate(ecg_cleaned, rpeaks=r_peak_indices, sampling_rate=sampling_rate, method=method)
            waves_lower = {k.lower(): v for k, v in w.items()}
            
            if 'ecg_p_onsets' in waves_lower and 'ecg_r_onsets' in waves_lower and \
               np.nansum(~np.isnan(waves_lower['ecg_p_onsets'])) > 0 and \
               np.nansum(~np.isnan(waves_lower['ecg_r_onsets'])) > 0:
                waves, used_method = waves_lower, method
                print(f"✓ 勾画成功，使用方法: '{used_method}'")
                break
        except Exception as e:
            # Critical fix: print out the hidden error information!
            print(f"× 方法 '{method}' 执行时发生致命错误: {e}") 
            
    if waves is None:
        print("All drawing methods failed. Please check the error information above."); return None
            
    if waves is None: print("All drawing methods failed."); return None

    # 4. Precisely extract PR intervals (completely mimic Jupyter)
    p_onsets_raw = np.asarray(waves['ecg_p_onsets'], dtype=float)
    r_onsets_raw = np.asarray(waves['ecg_r_onsets'], dtype=float)

    p_onsets = p_onsets_raw[~np.isnan(p_onsets_raw)].astype(int)
    r_onsets = r_onsets_raw[~np.isnan(r_onsets_raw)].astype(int)

    pr_intervals, highlight_regions = [], []
    r_ptr = 0
    for p in p_onsets:
        while r_ptr < len(r_onsets) and r_onsets[r_ptr] <= p: r_ptr += 1
        if r_ptr >= len(r_onsets): break
        r = r_onsets[r_ptr]
        pr_sec = (r - p) / sampling_rate
        if 0.05 < pr_sec < 0.60:
            pr_intervals.append(pr_sec)
            highlight_regions.append((p, r))
    
    pr_intervals = np.array(pr_intervals)

    # 5. Plotting (consistent with Jupyter)
    time_axis = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(time_axis, ecg_cleaned, f'{ecg_title} ({lead_name}) - PR Intervals (Corrected)', ax, sampling_rate)
    ax.scatter(time_axis[r_peak_indices], ecg_cleaned[r_peak_indices], c="dodgerblue", s=40, zorder=5, label="R-peaks")

    for p, r in highlight_regions:
        color = "orange" if ((r - p) / sampling_rate) > 0.20 else "green"
        ax.axvspan(time_axis[p], time_axis[r], alpha=0.4, color=color, zorder=4)

    handles, _ = ax.get_legend_handles_labels()
    orange_patch = mpatches.Patch(color='orange', alpha=0.4, label='Abnormal PR (> 0.20s)')
    green_patch = mpatches.Patch(color='green', alpha=0.4, label='Normal PR (≤ 0.20s)')
    handles.extend([green_patch, orange_patch])
    ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avb1_pr_intervals_corrected.png"))
    plt.close(fig)


    # 6. Generate histogram and report
    if pr_intervals.size > 0:
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(pr_intervals, bins='auto', edgecolor='k', alpha=0.7)
        ax_hist.axvline(0.20, color="orange", ls="--", lw=2, label="0.20s threshold")
        ax_hist.set_title("PR interval distribution"); ax_hist.set_xlabel("PR interval (s)"); ax_hist.set_ylabel("count"); ax_hist.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "avb1_pr_histogram.png"))
        plt.close(fig_hist)
    
    report_text = generate_report_1davb(pr_intervals, used_method, ecg_title, lead_name)
    print(report_text)
    
    metrics = {"pr_mean_s": float(np.mean(pr_intervals)), "pr_min_s": float(np.min(pr_intervals)), "pr_max_s": float(np.max(pr_intervals)), "prolonged_pr_count": int(np.sum(pr_intervals > 0.20)), "total_pr_count": len(pr_intervals)}
    report_data = {"title": ecg_title, "lead_name": lead_name, "report_text": report_text, "metrics": metrics, "image_paths": ["avb1_pr_intervals_corrected.png", "avb1_pr_histogram.png"]}
    
    return report_data