import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid


def generate_sb_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    """Sinus Bradycardia (SB) visualization and metrics.

    Criteria (simple): sustained HR < 60 bpm (RR > 1.0 s) for a significant portion of beats.
    Visuals:
      - Top: ECG with R-peaks + a few RR spans and values + status badge
      - Middle: Instantaneous HR timeline with 60 bpm threshold; brady area shaded below
      - Bottom: RR histogram with normal band (0.6–1.0 s) and brady band (>1.0 s)
    """
    if ecg_signal is None:
        return None

    # Clean + R-peaks
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
    r_idx = rpeaks.get("ECG_R_Peaks", np.array([], dtype=int))

    rr = np.diff(r_idx) / sampling_rate if len(r_idx) > 1 else np.array([])
    avg_hr = 60.0 / np.mean(rr) if rr.size else 0.0
    brady_flags = (rr > 1.0) if rr.size else np.array([])
    brady_ratio = float(np.mean(brady_flags)) if rr.size else 0.0

    t = np.linspace(0, len(ecg_clean) / sampling_rate, len(ecg_clean), endpoint=False)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={'height_ratios': [3, 1.2, 1]})

    # Top: ECG overview
    plot_ecg_with_grid(t, ecg_clean, f"{ecg_title} ({lead_name}) - Avg HR ≈ {avg_hr:.0f} bpm", axes[0], sampling_rate)
    if len(r_idx):
        axes[0].scatter(t[r_idx], ecg_clean[r_idx], c='dodgerblue', s=30, label='R-peaks', zorder=5)
        for i in range(min(len(r_idx) - 1, 5)):
            s, e = r_idx[i], r_idx[i+1]
            axes[0].axvspan(t[s], t[e], color='lightgreen', alpha=0.3, ymin=0.45, ymax=0.55)
            if i < 3 and rr.size:
                axes[0].text((t[s] + t[e]) / 2, np.median(ecg_clean), f'RR {rr[i]:.2f}s',
                              ha='center', fontsize=9, color='k', bbox=dict(fc='w', alpha=0.7, pad=1))
    _h, _l = axes[0].get_legend_handles_labels()
    if _l:
        axes[0].legend(loc='upper right')
    # Status badge
    status = "Bradycardia" if (avg_hr < 60 or brady_ratio > 0.5) else "No brady"
    color = 'crimson' if status == 'Bradycardia' else 'black'
    axes[0].text(0.99, 0.03, f"{status}\nHR {avg_hr:.0f} bpm\nBrady beats {brady_ratio*100:.0f}%",
                 transform=axes[0].transAxes, ha='right', va='bottom', fontsize=10, color=color,
                 bbox=dict(fc='white', alpha=0.9, ec='none', pad=6))

    # Middle: HR timeline
    if rr.size:
        rr_mid_t = (t[r_idx[1:]] + t[r_idx[:-1]]) / 2.0
        hr_series = 60.0 / rr
        axes[1].plot(rr_mid_t, hr_series, '-o', color='teal', ms=4, lw=1.5, label='Instant HR')
        axes[1].axhline(60, color='crimson', ls='--', lw=2, label='60 bpm threshold')
        ymin = min(40.0, float(np.min(hr_series)) * 0.9)
        axes[1].axhspan(0, 60, color='lightblue', alpha=0.25)
        axes[1].set_ylim(ymin, max(80.0, float(np.max(hr_series)) * 1.1))
        axes[1].set_ylabel('HR (bpm)'); axes[1].legend(loc='upper right')
        axes[1].grid(axis='y', alpha=0.3)

    # Bottom: RR histogram
    if rr.size:
        axes[2].hist(rr, bins='auto', ec='k', alpha=0.7, color='teal')
        mean_rr = float(np.mean(rr))
        axes[2].axvline(mean_rr, c='r', ls='--', lw=2, label=f'Mean RR {mean_rr:.3f}s (HR {60/mean_rr:.1f})')
        axes[2].axvspan(0.6, 1.0, color='green', alpha=0.1, label='Normal band (0.6–1.0 s)')
        axes[2].axvspan(1.0, max(2.0, float(np.max(rr)) * 1.05), color='lightblue', alpha=0.25, label='Brady zone (>1.0 s)')
        axes[2].set_title('RR Interval Distribution'); axes[2].legend(loc='upper right')
        axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "sinus_bradycardia_analysis.png"), dpi=160); plt.close(fig)

    metrics = {
        "average_heart_rate_bpm": float(avg_hr),
        "mean_rr_interval_s": float(np.mean(rr)) if rr.size else None,
        "brady_ratio": brady_ratio,
    }
    report = (
        f"\nObs for {ecg_title} ({lead_name}):\n"
        f"- HR < 60 bpm or RR > 1.0 s considered bradycardia.\n"
        f"- Avg HR: {avg_hr:.1f} bpm; brady beats: {brady_ratio*100:.0f}%.\n"
        f"- HR timeline shows threshold at 60 bpm; RR histogram shades >1.0 s."
    )

    return {
        "title": ecg_title,
        "lead_name": lead_name,
        "report_text": report,
        "metrics": metrics,
        "image_paths": ["sinus_bradycardia_analysis.png"],
    }


