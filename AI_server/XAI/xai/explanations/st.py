
# explanation_st.py
import os, json, numpy as np, matplotlib.pyplot as plt, neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid
from ..clinical_reports import generate_report_sinus_tachy

def generate_st_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    if ecg_signal is None: return None
    # 1) Clean + R-peaks
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    r_idx = rpeaks['ECG_R_Peaks']

    rr = np.diff(r_idx) / sampling_rate if len(r_idx) > 1 else np.array([])
    avg_hr = 60.0 / np.mean(rr) if rr.size > 0 else 0
    tachy_flags = (rr < 0.6) if rr.size else np.array([])
    tachy_ratio = float(np.mean(tachy_flags)) if rr.size else 0.0

    # 2) Plot: top-ECG, middle-HR timeline, bottom-RR histogram
    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={'height_ratios': [3, 1.2, 1]})
    plot_ecg_with_grid(t, ecg_cleaned, f'{ecg_title} ({lead_name}) - Avg HR ≈ {avg_hr:.0f} bpm', axes[0], sampling_rate)
    if len(r_idx):
        axes[0].scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', marker='o', s=30, label='R-peaks', zorder=5)
        # Highlight first 3-5 RR intervals and values
        for i in range(min(len(r_idx) - 1, 5)):
            start, end = r_idx[i], r_idx[i+1]
            axes[0].axvspan(t[start], t[end], color='lightgreen', alpha=0.3, ymin=0.45, ymax=0.55)
            if i < 3 and rr.size:
                axes[0].text((t[start] + t[end]) / 2, np.median(ecg_cleaned), f'RR {rr[i]:.2f}s', ha='center', fontsize=9, color='k', bbox=dict(fc='w', alpha=0.7, pad=1))
    # Legend (only when there are annotated objects)
    _h, _l = axes[0].get_legend_handles_labels()
    if _l:
        axes[0].legend(loc='upper right')
    # Upper right status badge
    status = "Tachycardia" if (avg_hr > 100 or tachy_ratio > 0.5) else "No tachy"
    status_color = 'crimson' if status == "Tachycardia" else 'black'
    axes[0].text(0.99, 0.03, f"{status}\nHR {avg_hr:.0f} bpm\nTachy beats {tachy_ratio*100:.0f}%", transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=10, color=status_color,
                 bbox=dict(fc='white', alpha=0.9, ec='none', pad=6))

    # Middle: HR timeline (by RR midpoint time)
    if rr.size:
        rr_mid_t = (t[r_idx[1:]] + t[r_idx[:-1]]) / 2.0
        hr_series = 60.0 / rr
        axes[1].plot(rr_mid_t, hr_series, '-o', color='teal', ms=4, lw=1.5, label='Instant HR')
        axes[1].axhline(100, color='crimson', ls='--', lw=2, label='100 bpm threshold')
        # Color tachycardia zone
        ymax = max(120.0, float(np.max(hr_series)) * 1.1)
        axes[1].axhspan(100, ymax, color='lightpink', alpha=0.25)
        axes[1].set_ylim(50, ymax)
        axes[1].set_ylabel('HR (bpm)'); axes[1].legend(loc='upper right')
        axes[1].grid(axis='y', alpha=0.3)

    # Bottom: RR histogram
    if rr.size > 0:
        axes[2].hist(rr, bins='auto', ec='k', alpha=0.7, color='teal')
        mean_rr = np.mean(rr)
        axes[2].axvline(mean_rr, c='r', ls='--', lw=2, label=f'Mean RR {mean_rr:.3f}s (HR {60/mean_rr:.1f})')
        axes[2].axvspan(0, 0.6, color='lightpink', alpha=0.3, label='Tachy zone (RR < 0.6s / HR > 100)')
        axes[2].axvspan(0.6, 1.0, color='green', alpha=0.1, label='Normal band (0.6–1.0 s)')
        axes[2].set_title('RR Interval Distribution'); axes[2].legend(loc='upper right')
        axes[2].grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "sinus_tachycardia_analysis.png"), dpi=160); plt.close(fig)

    report_text = generate_report_sinus_tachy(ecg_title, lead_name, avg_hr)
    print(report_text)
    return {
        "title": ecg_title,
        "lead_name": lead_name,
        "report_text": report_text,
        "metrics": {
            "average_heart_rate_bpm": float(avg_hr),
            "mean_rr_interval_s": float(np.mean(rr)) if rr.size > 0 else None
        },
        "image_paths": ["sinus_tachycardia_analysis.png"],
    }