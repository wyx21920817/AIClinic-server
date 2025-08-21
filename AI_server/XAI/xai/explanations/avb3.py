import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid


def generate_avb3_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    if ecg_signal is None:
        return None

    # Clean
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal

    # R-peaks and RR
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_idx = rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}")
        return None
    rr = np.diff(r_idx) / sampling_rate if len(r_idx) > 1 else np.array([])

    # Delineate to get P occurrences
    waves_lower = None
    for method in ["prominence", "dwt", "cwt", "peak"]:
        try:
            _, w = nk.ecg_delineate(ecg_cleaned, rpeaks=r_idx, sampling_rate=sampling_rate, method=method)
            waves_lower = {k.lower(): v for k, v in w.items()}
            if 'ecg_p_onsets' in waves_lower and 'ecg_r_onsets' in waves_lower:
                break
        except Exception:
            continue

    if not waves_lower or 'ecg_p_onsets' not in waves_lower:
        print("Could not delineate P onsets for AVB3 analysis.")
        return None

    p_onsets = np.asarray(waves_lower['ecg_p_onsets'], dtype=float)
    p_onsets = p_onsets[~np.isnan(p_onsets)].astype(int)

    # PP intervals
    pp = np.diff(p_onsets) / sampling_rate if len(p_onsets) > 1 else np.array([])

    # Conduction mapping: fraction of P followed by R within 80-320 ms
    conduction_window = (int(0.08 * sampling_rate), int(0.32 * sampling_rate))
    conducted_flags = []
    for p in p_onsets:
        after = r_idx[r_idx > p]
        if after.size == 0:
            conducted_flags.append(False)
            continue
        dt = after[0] - p
        conducted_flags.append(conduction_window[0] <= dt <= conduction_window[1])
    conducted_flags = np.array(conducted_flags, dtype=bool)
    conduction_ratio = float(np.mean(conducted_flags)) if conducted_flags.size else 0.0

    # PR intervals for variability
    pr_list = []
    r_onsets = np.asarray(waves_lower.get('ecg_r_onsets', []), dtype=float)
    r_onsets = r_onsets[~np.isnan(r_onsets)].astype(int)
    for p in p_onsets:
        r_after = r_onsets[r_onsets > p]
        if r_after.size:
            pr_s = (r_after[0] - p) / sampling_rate
            if 0.05 < pr_s < 0.60:
                pr_list.append(pr_s)
    pr_arr = np.array(pr_list)
    pr_cv = float(np.std(pr_arr) / np.mean(pr_arr)) if pr_arr.size and np.mean(pr_arr) > 0 else None

    # Rates
    rr_rate = 60.0 / float(np.mean(rr)) if rr.size and np.mean(rr) > 0 else None
    pp_rate = 60.0 / float(np.mean(pp)) if pp.size and np.mean(pp) > 0 else None

    # Heuristic decision: AV dissociation
    dissociated = (conduction_ratio < 0.2) and (rr_rate is not None and pp_rate is not None and abs(rr_rate - pp_rate) > 15)

    # Plot overview with events
    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - 3rd-degree AV Block", ax, sampling_rate)
    if len(r_idx):
        ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=30, label='R-peaks', zorder=5)
    if len(p_onsets):
        ax.scatter(t[p_onsets], ecg_cleaned[p_onsets], c='red', s=20, marker='^', label='P-onsets', zorder=5)
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'avb3_overview.png')); plt.close(fig)

    # PP & RR histograms
    img_paths = ['avb3_overview.png']
    if pp.size:
        fig_pp, ax_pp = plt.subplots(figsize=(6, 4))
        ax_pp.hist(pp, bins='auto', edgecolor='k', alpha=0.7, color='skyblue')
        ax_pp.set_title('PP interval distribution'); ax_pp.set_xlabel('PP (s)'); ax_pp.set_ylabel('count'); ax_pp.grid(axis='y', alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'avb3_pp_hist.png')); plt.close(fig_pp)
        img_paths.append('avb3_pp_hist.png')
    if rr.size:
        fig_rr, ax_rr = plt.subplots(figsize=(6, 4))
        ax_rr.hist(rr, bins='auto', edgecolor='k', alpha=0.7, color='lightcoral')
        ax_rr.set_title('RR interval distribution'); ax_rr.set_xlabel('RR (s)'); ax_rr.set_ylabel('count'); ax_rr.grid(axis='y', alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'avb3_rr_hist.png')); plt.close(fig_rr)
        img_paths.append('avb3_rr_hist.png')

    # PR raster (if available)
    if pr_arr.size:
        fig_pr, ax_pr = plt.subplots(figsize=(10, 3.5))
        ax_pr.plot(np.arange(len(pr_arr)), pr_arr, marker='o', linestyle='-', color='black')
        ax_pr.axhline(0.20, color='orange', ls='--', lw=2, label='0.20s')
        ax_pr.set_title('PR interval per conducted beat'); ax_pr.set_xlabel('Beat index'); ax_pr.set_ylabel('PR (s)'); ax_pr.legend()
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'avb3_pr_raster.png')); plt.close(fig_pr)
        img_paths.append('avb3_pr_raster.png')

    report_lines = [
        f"Conduction ratio (P→R within 80–320ms): {conduction_ratio:.2f}",
        f"Atrial rate (PP): {pp_rate:.1f} bpm" if pp_rate else "Atrial rate (PP): n/a",
        f"Ventricular rate (RR): {rr_rate:.1f} bpm" if rr_rate else "Ventricular rate (RR): n/a",
        f"PR variability (CV): {pr_cv:.2f}" if pr_cv is not None else "PR variability (CV): n/a",
        f"AV dissociation heuristic: {'Yes' if dissociated else 'No'}"
    ]
    report_text = "\n".join(report_lines)

    metrics = {
        'conduction_ratio': conduction_ratio,
        'pp_rate_bpm': float(pp_rate) if pp_rate is not None else None,
        'rr_rate_bpm': float(rr_rate) if rr_rate is not None else None,
        'pr_cv': float(pr_cv) if pr_cv is not None else None,
        'total_p_count': int(len(p_onsets)),
        'total_r_count': int(len(r_idx)),
        'av_dissociation': bool(dissociated),
    }

    return {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'metrics': metrics,
        'image_paths': img_paths
    }
