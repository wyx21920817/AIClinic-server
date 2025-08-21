import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid


def generate_avb2_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    if ecg_signal is None:
        return None

    # Clean signal
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal

    # Detect R peaks
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_idx = rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}")
        return None

    # Delineate to get P and R onsets
    waves_lower = None
    for method in ["prominence", "dwt", "cwt", "peak"]:
        try:
            _, w = nk.ecg_delineate(ecg_cleaned, rpeaks=r_idx, sampling_rate=sampling_rate, method=method)
            waves_lower = {k.lower(): v for k, v in w.items()}
            if 'ecg_p_onsets' in waves_lower and 'ecg_r_onsets' in waves_lower:
                break
        except Exception:
            continue

    if not waves_lower or 'ecg_p_onsets' not in waves_lower or 'ecg_r_onsets' not in waves_lower:
        print("Could not delineate P/R onsets for AVB2 analysis.")
        return None

    p_onsets_raw = np.asarray(waves_lower['ecg_p_onsets'], dtype=float)
    r_onsets_raw = np.asarray(waves_lower['ecg_r_onsets'], dtype=float)
    p_onsets = p_onsets_raw[~np.isnan(p_onsets_raw)].astype(int)
    r_onsets = r_onsets_raw[~np.isnan(r_onsets_raw)].astype(int)

    # Pair P to next R and detect drops
    pr_intervals, conducted_pairs, dropped_p_indices = [], [], []
    for p in p_onsets:
        later_r = r_onsets[r_onsets > p]
        if later_r.size == 0:
            break
        r = later_r[0]
        pr_s = (r - p) / sampling_rate
        if 0.05 < pr_s < 0.40:
            pr_intervals.append(pr_s)
            conducted_pairs.append((p, r))
        else:
            dropped_p_indices.append(p)

    # Heuristic: also mark P as dropped if no R occurs within 0.40s after P
    window = int(0.40 * sampling_rate)
    for p in p_onsets:
        if not any((r - p) > 0 and (r - p) <= window for r in r_onsets):
            dropped_p_indices.append(p)
    dropped_p_indices = sorted(set(int(x) for x in dropped_p_indices))

    pr_intervals = np.array(pr_intervals)
    total_p = int(len(p_onsets))
    total_r = int(len(r_onsets))
    conducted = int(len(conducted_pairs))
    dropped = int(len(dropped_p_indices))
    conduction_ratio = float(conducted / total_p) if total_p else 0.0

    # Detect Wenckebach (Mobitz I): increasing PR then a drop
    wenckebach_sequences = 0
    if len(pr_intervals) >= 3 and dropped > 0:
        # Compute diffs of PR; look for 2 or more consecutive positive increases
        diffs = np.diff(pr_intervals)
        streak = 0
        for d in diffs:
            if d > 0.01:
                streak += 1
                if streak >= 2:
                    wenckebach_sequences += 1
                    streak = 0
            else:
                streak = 0

    # Subtype guess
    subtype = None
    pr_std = float(np.std(pr_intervals)) if pr_intervals.size else None
    if dropped >= 1 and wenckebach_sequences >= 1:
        subtype = "Mobitz I (Wenckebach)"
    elif dropped >= 1 and pr_std is not None and pr_std < 0.03:
        subtype = "Mobitz II"
    elif dropped >= 1:
        subtype = "2nd-degree AVB (unspecified)"

    # Plot ECG with annotations
    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - 2nd-degree AV Block", ax, sampling_rate)
    if len(r_idx):
        ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=30, zorder=5, label='R-peaks')
    # Mark conducted PR spans
    for p, r in conducted_pairs:
        ax.axvspan(t[p], t[r], alpha=0.35, color='green')
    # Mark dropped P regions
    for p in dropped_p_indices:
        p_end = min(len(t)-1, p + int(0.08 * sampling_rate))
        ax.axvspan(t[p], t[p_end], alpha=0.35, color='orange')
    # Legend: conducted vs dropped
    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        mpatches.Patch(color='green', alpha=0.35, label='Conducted PR'),
        mpatches.Patch(color='orange', alpha=0.35, label='Dropped P-wave')
    ])
    ax.legend(handles=handles, loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'avb2_overview.png')); plt.close(fig)

    # PR trend plot
    if pr_intervals.size:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(np.arange(len(pr_intervals)), pr_intervals, marker='o')
        ax2.axhline(0.20, color='orange', ls='--', lw=2, label='0.20s')
        ax2.set_title('PR Interval Trend'); ax2.set_xlabel('Beat index'); ax2.set_ylabel('PR (s)')
        ax2.legend(); plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'avb2_pr_trend.png')); plt.close(fig2)

        # PR histogram (align with 1dAVB)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(pr_intervals, bins='auto', edgecolor='k', alpha=0.7)
        ax3.axvline(0.20, color='orange', ls='--', lw=2, label='0.20s threshold')
        ax3.set_title('PR interval distribution'); ax3.set_xlabel('PR interval (s)'); ax3.set_ylabel('count'); ax3.legend()
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'avb2_pr_hist.png')); plt.close(fig3)

    report_lines = [
        f"Conduction ratio (R/P): {conduction_ratio:.2f} ({conducted}/{total_p})",
        f"Dropped P count: {dropped}",
        f"Wenckebach sequences: {wenckebach_sequences}",
        f"Subtype: {subtype or 'undetermined'}"
    ]
    report_text = "\n".join(report_lines)

    metrics = {
        'pr_mean_s': float(np.mean(pr_intervals)) if pr_intervals.size else None,
        'pr_std_s': pr_std,
        'total_p_count': total_p,
        'total_r_count': total_r,
        'conducted_count': conducted,
        'dropped_p_count': dropped,
        'conduction_ratio': conduction_ratio,
        'wenckebach_seq_count': int(wenckebach_sequences),
        'subtype': subtype,
    }

    image_paths = ['avb2_overview.png']
    if pr_intervals.size:
        image_paths.append('avb2_pr_trend.png')
        image_paths.append('avb2_pr_hist.png')

    return {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'metrics': metrics,
        'image_paths': image_paths
    }
