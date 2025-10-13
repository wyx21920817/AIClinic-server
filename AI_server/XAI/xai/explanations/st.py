
# explanation_st.py
import os, numpy as np, matplotlib.pyplot as plt, neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid
# Removed dependency on generate_report_sinus_tachy - using structured reports instead

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
    
    # Calculate instantaneous heart rate and peak HR for enhanced validation
    instantaneous_hr = 60.0 / rr if rr.size > 0 else np.array([])
    max_instantaneous_hr = float(np.max(instantaneous_hr)) if len(instantaneous_hr) > 0 else 0

    # 2) Plot: top-ECG, middle-HR timeline, bottom-RR histogram
    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={'height_ratios': [3, 1.2, 1]})
    plot_ecg_with_grid(t, ecg_cleaned, f'{ecg_title} ({lead_name}) - Avg HR ≈ {avg_hr:.0f} bpm', axes[0], sampling_rate)
    if len(r_idx):
        axes[0].scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', marker='o', s=30, label='R-peaks', zorder=5)
        # Highlight first 3-5 RR intervals with diagnostic color coding
        for i in range(min(len(r_idx) - 1, 5)):
            start, end = r_idx[i], r_idx[i+1]
            rr_val = rr[i] if i < len(rr) else 0
            
            # Diagnostic color coding based on RR interval
            if rr_val < 0.6:  # Tachycardia (< 0.6s = HR > 100 bpm)
                color = 'lightcoral'  # Red for tachycardia
                text_color = 'darkred'
                text_weight = 'bold'
            elif rr_val > 1.0:  # Bradycardia (> 1.0s = HR < 60 bpm)
                color = 'lightblue'  # Blue for bradycardia
                text_color = 'darkblue'
                text_weight = 'bold'
            else:  # Normal (0.6-1.0s = HR 60-100 bpm)
                color = 'lightgreen'  # Green for normal
                text_color = 'darkgreen'
                text_weight = 'normal'
            
            axes[0].axvspan(t[start], t[end], color=color, alpha=0.4, ymin=0.45, ymax=0.55)
            if i < 3 and rr.size:
                # Enhanced text annotation with better contrast and highlighting
                axes[0].text((t[start] + t[end]) / 2, np.median(ecg_cleaned), f'RR {rr[i]:.2f}s', 
                           ha='center', fontsize=10, color=text_color, weight=text_weight,
                           bbox=dict(fc='white', alpha=0.9, ec=text_color, lw=1, pad=2))
    # Add diagnostic color coding legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.4, label='Normal RR (0.6-1.0s) - 60-100 bpm'),
        Patch(facecolor='lightblue', alpha=0.4, label='Bradycardia RR (>1.0s) - <60 bpm'),
        Patch(facecolor='lightcoral', alpha=0.4, label='Tachycardia RR (<0.6s) - >100 bpm')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)
    # Status badge - unified clear statement
    if tachy_ratio > 0.0:
        status = f"Tachycardia: Detected ({tachy_ratio*100:.0f}% of beats)"
        status_color = 'crimson'
    else:
        status = f"Tachycardia: Not Detected (0% of beats)"
        status_color = 'black'
    
    axes[0].text(0.99, 0.03, f"{status}\nHR {avg_hr:.0f} bpm", transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=10, color=status_color,
                 bbox=dict(fc='white', alpha=0.9, ec='none', pad=6))

    # Middle: HR timeline (by RR midpoint time)
    if rr.size:
        rr_mid_t = (t[r_idx[1:]] + t[r_idx[:-1]]) / 2.0
        hr_series = 60.0 / rr
        axes[1].plot(rr_mid_t, hr_series, '-o', color='teal', ms=4, lw=1.5, label='Instant HR')
        axes[1].axhline(100, color='crimson', ls='--', lw=2, label='100 bpm threshold')
        # Color tachycardia zone - matching ECG overview colors
        ymax = max(120.0, float(np.max(hr_series)) * 1.1)
        axes[1].axhspan(100, ymax, color='lightcoral', alpha=0.25)
        axes[1].set_ylim(50, ymax)
        axes[1].set_ylabel('HR (bpm)'); axes[1].legend(loc='upper right')
        axes[1].grid(axis='y', alpha=0.3)

    # Bottom: RR histogram
    if rr.size > 0:
        axes[2].hist(rr, bins='auto', ec='k', alpha=0.7, color='teal')
        mean_rr = np.mean(rr)
        axes[2].axvline(mean_rr, c='r', ls='--', lw=2, label=f'Mean RR {mean_rr:.3f}s (HR {60/mean_rr:.1f})')
        axes[2].axvspan(0, 0.6, color='lightcoral', alpha=0.3, label='Tachy zone (RR < 0.6s / HR > 100)')
        axes[2].axvspan(0.6, 1.0, color='lightgreen', alpha=0.3, label='Normal band (0.6–1.0 s)')
        axes[2].set_title('RR Interval Distribution'); axes[2].legend(loc='upper right')
        axes[2].grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "sinus_tachycardia_analysis.png"), dpi=160); plt.close(fig)

    # (Report text centralized in clinical_reports.ReportGenerator)
    
    # Prepare metrics for unified report generator
    metrics = {
        'average_heart_rate_bpm': float(avg_hr),
        'mean_rr_interval_s': float(np.mean(rr)) if rr.size > 0 else None,
        'tachy_ratio': tachy_ratio,
        'max_instantaneous_hr': max_instantaneous_hr,
    }
    
    # Use unified report generator
    from ..clinical_reports import ReportGenerator
    report_generator = ReportGenerator()
    
    # Generate complete report using unified generator
    report_data = report_generator.generate_st_report(metrics, {})
    
    # Add lead_name and image_paths to the report
    report_data['lead_name'] = lead_name
    report_data['image_paths'] = {
        'sinus_tachycardia_analysis.png': os.path.join(output_dir, "sinus_tachycardia_analysis.png")
    }
    
    return report_data