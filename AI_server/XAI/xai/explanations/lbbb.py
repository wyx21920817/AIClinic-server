# explanation_lbbb.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks
from xai.ecg_utils import plot_ecg_with_grid
from xai.clinical_reports import generate_report_bbb

def generate_lbbb_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    """Left Bundle Branch Block analysis with refined QRS estimation (notebook-aligned)."""
    if ecg_signal is None:
        return None

    # 1) Clean signal
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
        print(f"Signal for {ecg_title} cleaned using 'neurokit' method.")
    except Exception as e:
        print(f"Warning: Cleaning failed for {ecg_title}: {e}. Using raw signal.")
        ecg_cleaned = ecg_signal

    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)

    # 2) R-peaks
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_idx = rpeaks['ECG_R_Peaks']
        print(f"R-peaks detected. Found {len(r_idx)} R-peaks.")
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}")
        return None

    # 3) Delineation
    waves, used_method = None, None
    for method in ["dwt", "cwt", "peak"]:
        try:
            _, w = nk.ecg_delineate(ecg_cleaned, rpeaks=r_idx, sampling_rate=sampling_rate, method=method)
            if w and 'ECG_QRS_Onsets' in w and 'ECG_QRS_Offsets' in w \
               and np.sum(~np.isnan(w['ECG_QRS_Onsets'])) > 0 and np.sum(~np.isnan(w['ECG_QRS_Offsets'])) > 0:
                waves, used_method = w, method
                print(f"Delineation successful via '{method}'.")
                break
            else:
                print(f"Delineation with '{method}' did not yield valid QRS onsets/offsets.")
        except Exception as e:
            print(f"Delineation with method '{method}' failed: {e}")

    qrs_durations_sec, highlight_regions = [], []
    estimation_type = "Delineated"

    if waves is not None:
        q_on = [int(q) for q in waves['ECG_QRS_Onsets'] if not np.isnan(q)]
        q_off = [int(q) for q in waves['ECG_QRS_Offsets'] if not np.isnan(q)]
        used_offsets = 0
        for on in q_on:
            # find next offset greater than onset
            while used_offsets < len(q_off) and q_off[used_offsets] <= on:
                used_offsets += 1
            if used_offsets >= len(q_off):
                break
            off = q_off[used_offsets]
            dur = (off - on) / sampling_rate
            if 0.04 < dur < 0.30:
                qrs_durations_sec.append(dur)
                highlight_regions.append((on, off))
                used_offsets += 1

    # 4) Refined estimation fallback (notebook logic)
    if not qrs_durations_sec:
        estimation_type = "Refined Estimation"
        print("Attempting refined R-peak based QRS boundary estimation.")

        prominence_factor = 0.05
        min_peak_width = 1
        q_search_sec = 0.10
        s_search_sec = 0.15
        q_fallback_sec = 0.05
        s_fallback_sec = 0.09  # LBBB tends to have longer terminal S

        signal_ptp = float(np.max(ecg_cleaned) - np.min(ecg_cleaned))
        prominence_threshold = (prominence_factor * signal_ptp) if signal_ptp > 0 else 0.01

        for r in r_idx:
            # left search for Q-like min
            left_win = int(q_search_sec * sampling_rate)
            start_left = max(0, r - left_win)
            seg_left = ecg_cleaned[start_left:r]
            q_start = max(0, r - int(q_fallback_sec * sampling_rate))
            if seg_left.size > min_peak_width + 1:
                inv_left = -seg_left
                peaks_l, _ = find_peaks(inv_left, prominence=prominence_threshold, width=min_peak_width)
                if peaks_l.size > 0:
                    q_start = start_left + peaks_l[-1]

            # right search for S/J-like min
            right_win = int(s_search_sec * sampling_rate)
            r_plus_1 = min(r + 1, len(ecg_cleaned) - 1)
            end_right = min(len(ecg_cleaned), r_plus_1 + right_win)
            seg_right = ecg_cleaned[r_plus_1:end_right]
            q_end = min(len(ecg_cleaned) - 1, r + int(s_fallback_sec * sampling_rate))
            if seg_right.size > min_peak_width + 1:
                inv_right = -seg_right
                peaks_r, _ = find_peaks(inv_right, prominence=prominence_threshold, width=min_peak_width)
                if peaks_r.size > 0:
                    q_end = r_plus_1 + peaks_r[0]

            if q_start >= q_end:
                # symmetric fallback 150 ms for LBBB
                half = (int((150 / 1000.0) * sampling_rate)) // 2
                q_start = max(0, r - half)
                q_end = min(len(ecg_cleaned) - 1, r + half)

            dur_samples = q_end - q_start
            if dur_samples > 0:
                qrs_durations_sec.append(dur_samples / sampling_rate)
                highlight_regions.append((q_start, q_end))

        if qrs_durations_sec:
            print("Refined Estimation QRS durations calculated.")
        else:
            print("Could not perform Refined Estimation for QRS durations.")
            estimation_type = "Failed Estimation"

    # 5) Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    plot_ecg_with_grid(t, ecg_cleaned, f'{ecg_title} ({lead_name}) - QRS Duration Analysis ({estimation_type})', axes[0], sampling_rate)
    axes[0].scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', marker='o', s=30, label='R-peaks', zorder=5)

    annotated = 0
    for on, off in highlight_regions:
        dur = (off - on) / sampling_rate
        color = 'purple' if dur > 0.12 else ('green' if dur >= 0.06 else 'lightgray')
        axes[0].axvspan(t[on], t[off], alpha=0.3, color=color, ymin=0.3, ymax=0.7)
        if annotated < 3 and dur > 0.12:
            seg_start, seg_end = max(0, on), min(len(ecg_cleaned) - 1, off)
            y_text = np.max(ecg_cleaned[seg_start:seg_end+1])
            axes[0].text((t[on] + t[off]) / 2, y_text + 0.1 * (np.max(ecg_cleaned) - np.min(ecg_cleaned)), f'QRS: {dur:.3f}s', ha='center', fontsize=9, color='k', bbox=dict(fc='w', alpha=0.7, pad=1))
            annotated += 1
    #  Legend: R-peaks + QRS classification
    import matplotlib.patches as mpatches
    handles, _ = axes[0].get_legend_handles_labels()
    handles.extend([
        mpatches.Patch(color='purple', alpha=0.3, label='Wide QRS (≥0.12s)'),
        mpatches.Patch(color='green', alpha=0.3, label='Normal QRS (0.06–0.11s)'),
        mpatches.Patch(color='lightgray', alpha=0.3, label='Other QRS'),
    ])
    axes[0].legend(handles=handles, loc='upper right', framealpha=0.9, facecolor='white')

    if qrs_durations_sec:
        qrs_arr = np.array(qrs_durations_sec)
        max_val = float(np.max(qrs_arr)); min_val = float(np.min(qrs_arr))
        hist_min = min(0.05, min_val - 0.01 if min_val > 0.05 else 0.04)
        hist_max = max(0.25, max_val + 0.02)
        bins = np.arange(hist_min, hist_max, 0.01)
        if bins.size == 0:
            bins = 'auto'
        axes[1].hist(qrs_arr, bins=bins, ec='k', alpha=0.7, color='lightcoral')
        mean_qrs = float(np.mean(qrs_arr))
        axes[1].axvline(mean_qrs, c='r', ls='--', lw=2, label=f'Mean QRS: {mean_qrs:.3f}s')
        axes[1].axvspan(0.06, 0.11, color='g', alpha=0.2, label='Normal QRS (0.06-0.11s)')
        axes[1].axvline(0.12, c='purple', ls=':', lw=2, label='0.12s Threshold (Wide)')
        axes[1].set_title('QRS Duration Distribution (Refined Estimation)' if estimation_type != 'Delineated' else 'QRS Duration Distribution (Delineated)')
        axes[1].set_xlabel('QRS Duration (s)'); axes[1].set_ylabel('Frequency'); axes[1].legend(); axes[1].grid(axis='y', alpha=0.75)
    else:
        axes[1].text(0.5, 0.5, 'No QRS durations calculated / to display.', ha='center')
        axes[1].set_title('QRS Duration Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lbbb_qrs_analysis.png'))
    plt.close(fig)

    # 6) Report
    report_text = generate_report_bbb(ecg_title, lead_name, estimation_type, qrs_durations_sec)
    print(report_text)
    metrics = {}
    if qrs_durations_sec:
        qrs_arr = np.array(qrs_durations_sec)
        metrics = {
            'qrs_mean_s': float(np.mean(qrs_arr)),
            'wide_qrs_count': int(np.sum(qrs_arr > 0.12)),
            'total_qrs_count': int(qrs_arr.size)
        }
    return {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'metrics': metrics,
        'image_paths': ['lbbb_qrs_analysis.png']
    }