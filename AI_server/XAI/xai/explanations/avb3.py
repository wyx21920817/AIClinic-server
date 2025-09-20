import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ..ecg_utils import plot_ecg_with_grid
from scipy.signal import butter, filtfilt, find_peaks


def _bandpass(signal, fs, low, high, order=4):
    nyq = 0.5 * fs
    lowc = max(0.001, low / nyq)
    highc = min(0.99, high / nyq)
    b, a = butter(order, [lowc, highc], btype='band')
    return filtfilt(b, a, signal)


def _estimate_p_onsets_fallback(ecg_cleaned, r_peaks, fs):
    """Fallback P-wave estimation by QRS/T blanking + bandpass + peak detection.
    Returns numpy array of candidate P onsets (indices)."""
    n = len(ecg_cleaned)
    if n == 0:
        return np.array([], dtype=int)
    mask = np.ones(n, dtype=bool)
    # Blank QRS and early T around each R
    qrs_pre = int(0.12 * fs)
    t_post = int(0.28 * fs)
    for r in np.asarray(r_peaks, dtype=int):
        start = max(0, r - qrs_pre)
        end = min(n, r + t_post)
        mask[start:end] = False
    blanked = np.copy(ecg_cleaned)
    blanked[~mask] = 0.0
    try:
        atrial = _bandpass(blanked, fs, 0.5, 12.0, order=4)
    except Exception:
        atrial = blanked
    # Detect modest peaks separated sufficiently
    min_dist = max(1, int(0.20 * fs))
    prom = 0.05 * np.std(atrial) if np.std(atrial) > 0 else 0.0
    peaks, _ = find_peaks(np.abs(atrial), distance=min_dist, prominence=prom)
    return peaks.astype(int)


def generate_avb3_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    """
    Generate 3rd-degree AV Block analysis based on core clinical criteria:
    1. Complete AV dissociation (P waves and QRS complexes are independent)
    2. Atrial rate > Ventricular rate
    3. Regular atrial rhythm (P waves march through)
    4. Regular ventricular escape rhythm
    5. No consistent PR relationship
    """
    if ecg_signal is None:
        return None

    # Clean signal
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal

    # R-peak detection
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_peaks = rpeaks['ECG_R_Peaks']
        if len(r_peaks) < 3:
            print(f"Insufficient R peaks detected ({len(r_peaks)}) for 3dAVB analysis")
            return None
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}")
        return None

    # P-wave detection using multiple methods
    p_peaks = None
    for method in ["prominence", "dwt", "cwt"]:
        try:
            _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks=r_peaks, 
                                       sampling_rate=sampling_rate, method=method)
            if 'ECG_P_Onsets' in waves:
                p_onsets = waves['ECG_P_Onsets']
                p_onsets = p_onsets[~np.isnan(p_onsets)].astype(int)
                if len(p_onsets) >= 3:
                    p_peaks = p_onsets
                    break
        except Exception:
            continue

    # Fallback if insufficient P detections
    if p_peaks is None or len(p_peaks) < 3:
        fallback = _estimate_p_onsets_fallback(ecg_cleaned, r_peaks, sampling_rate)
        if len(fallback) >= 3:
            p_peaks = fallback
        else:
            # proceed with minimal P info to avoid hard failure
            p_peaks = fallback  # could be empty

    # Calculate basic intervals
    rr_intervals = np.diff(r_peaks) / sampling_rate
    pp_intervals = np.diff(p_peaks) / sampling_rate if len(p_peaks) > 1 else np.array([])
    
    # Calculate rates
    rr_rate = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else None
    pp_rate = 60.0 / np.mean(pp_intervals) if len(pp_intervals) > 0 else None

    # Core 3dAVB Analysis
    analysis_results = analyze_3davb_criteria(
        ecg_cleaned, r_peaks, p_peaks, rr_intervals, pp_intervals, 
        rr_rate, pp_rate, sampling_rate
    )

    # Generate visualizations
    image_paths = create_3davb_visualizations(
        ecg_cleaned, r_peaks, p_peaks, rr_intervals, pp_intervals,
        analysis_results, ecg_title, lead_name, output_dir, sampling_rate
    )

    # Generate clinical report
    report_text = generate_clinical_report(analysis_results, rr_rate, pp_rate)

    # Compile metrics
    metrics = compile_metrics(analysis_results, r_peaks, p_peaks, rr_rate, pp_rate)

    # Add warnings if P detection is weak
    if len(p_peaks) < 3:
        analysis_results['criteria_failed'].append('insufficient_p_detections')
        metrics['notes'] = 'insufficient_p_detections'

    # Build summary/statistics
    summary = {
        'report_text': report_text,
        'is_abnormality_present': bool(analysis_results.get('diagnosis','').startswith('Complete') or analysis_results.get('diagnosis','').endswith('AV Block')),
        'abnormality_count': 1 if analysis_results.get('diagnosis','') else 0,
    }
    statistics = {
        'conduction_ratio': metrics.get('conduction_ratio'),
        'pp_rate_bpm': metrics.get('pp_rate_bpm'),
        'rr_rate_bpm': metrics.get('rr_rate_bpm'),
        'pr_cv': metrics.get('pr_cv'),
        'phase_lock_corr': metrics.get('phase_lock_corr'),
        'pp_cv': analysis_results.get('pp_cv'),
        'rr_cv': analysis_results.get('rr_cv'),
        'confidence': metrics.get('confidence'),
        'dissociation_score': metrics.get('dissociation_score'),
        'complete_av_block': metrics.get('complete_av_block'),
    }

    return {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'summary': summary,
        'statistics': statistics,
        'image_paths': image_paths
    }


def analyze_3davb_criteria(ecg_signal, r_peaks, p_peaks, rr_intervals, pp_intervals, 
                          rr_rate, pp_rate, sampling_rate):
    """Analyze core 3dAVB criteria with clinical precision."""
    
    results = {
        'av_dissociation_score': 0.0,
        'criteria_met': [],
        'criteria_failed': [],
        'diagnosis': 'Normal',
        'confidence': 0.0,
        'conduction_ratio': None,
        'pr_cv': None,
        'pp_rate_bpm': float(pp_rate) if pp_rate else None,
        'rr_rate_bpm': float(rr_rate) if rr_rate else None,
        'pp_cv': None,
        'rr_cv': None,
    }

    # Pre-compute rhythm regularity
    pp_cv_val = (np.std(pp_intervals) / np.mean(pp_intervals)) if (len(pp_intervals) > 1 and np.mean(pp_intervals) > 0) else None
    rr_cv_val = (np.std(rr_intervals) / np.mean(rr_intervals)) if (len(rr_intervals) > 1 and np.mean(rr_intervals) > 0) else None
    results['pp_cv'] = float(pp_cv_val) if pp_cv_val is not None else None
    results['rr_cv'] = float(rr_cv_val) if rr_cv_val is not None else None

    # Criterion 1: Rate relationship (Atrial rate > Ventricular rate)
    if pp_rate is not None and rr_rate is not None:
        if pp_rate > rr_rate:
            rate_ratio = pp_rate / rr_rate
            if rate_ratio >= 1.5:
                results['av_dissociation_score'] += 0.25
                results['criteria_met'].append(f"Atrial rate significantly faster (ratio: {rate_ratio:.1f})")
            elif rate_ratio >= 1.2:
                results['av_dissociation_score'] += 0.15
                results['criteria_met'].append(f"Atrial rate faster (ratio: {rate_ratio:.1f})")
        else:
            results['criteria_failed'].append(f"Atrial rate not faster than ventricular rate")

    # Criterion 2: Atrial rhythm regularity (P waves march through)
    if pp_cv_val is not None:
        if pp_cv_val < 0.15:  # Regular atrial rhythm
            results['av_dissociation_score'] += 0.2
            results['criteria_met'].append(f"Regular atrial rhythm (PP CV: {pp_cv_val:.3f})")
        else:
            results['criteria_failed'].append(f"Irregular atrial rhythm (PP CV: {pp_cv_val:.3f})")

    # Criterion 3: Ventricular rhythm regularity (Escape rhythm)
    if rr_cv_val is not None:
        if rr_cv_val < 0.20:  # Regular ventricular rhythm
            results['av_dissociation_score'] += 0.2
            results['criteria_met'].append(f"Regular ventricular rhythm (RR CV: {rr_cv_val:.3f})")
        else:
            results['criteria_failed'].append(f"Irregular ventricular rhythm (RR CV: {rr_cv_val:.3f})")

    # Criterion 4: Ventricular bradycardia
    if rr_rate is not None:
        if rr_rate < 45:  # Severe bradycardia
            results['av_dissociation_score'] += 0.15
            results['criteria_met'].append(f"Severe bradycardia ({rr_rate:.1f} bpm)")
        elif rr_rate < 60:  # Mild bradycardia
            results['av_dissociation_score'] += 0.1
            results['criteria_met'].append(f"Bradycardia ({rr_rate:.1f} bpm)")

    # Criterion 5: AV dissociation (P-QRS independence)
    dissociation_evidence = assess_av_dissociation(ecg_signal, r_peaks, p_peaks, sampling_rate)
    results['av_dissociation_score'] += dissociation_evidence['score']
    results['conduction_ratio'] = dissociation_evidence.get('conduction_ratio')
    results['pr_cv'] = dissociation_evidence.get('pr_cv')
    results['phase_lock_corr'] = dissociation_evidence.get('phase_lock_corr')
    if dissociation_evidence['score'] > 0:
        results['criteria_met'].extend(dissociation_evidence['evidence'])
    else:
        results['criteria_failed'].extend(dissociation_evidence['failures'])

    # Hard-rule override for complete AV block (more permissive)
    if (results['conduction_ratio'] is not None and results['conduction_ratio'] <= 0.20 and
        pp_rate is not None and rr_rate is not None and pp_rate > rr_rate and
        (results['phase_lock_corr'] is None or results['phase_lock_corr'] < 0.30) and
        (pp_cv_val is None or pp_cv_val < 0.25) and (rr_cv_val is None or rr_cv_val < 0.30)):
        results['diagnosis'] = "Complete AV Block (3rd-degree)"
        results['confidence'] = max(results['av_dissociation_score'], 0.80)
        results['criteria_met'].append("rule_override_complete_av_block_v2")
        return results

    # Final diagnosis by score thresholds
    if results['av_dissociation_score'] >= 0.7:
        results['diagnosis'] = "Complete AV Block (3rd-degree)"
        results['confidence'] = min(1.0, results['av_dissociation_score'])
    elif results['av_dissociation_score'] >= 0.5:
        # Favor 3dAVB if conduction is very low and phase corr low
        if (results['conduction_ratio'] is not None and results['conduction_ratio'] <= 0.20 and
            (results['phase_lock_corr'] is None or results['phase_lock_corr'] < 0.30)):
            results['diagnosis'] = "Complete AV Block (3rd-degree)"
        else:
            results['diagnosis'] = "High-grade AV Block"
        results['confidence'] = results['av_dissociation_score']
    elif results['av_dissociation_score'] >= 0.3:
        results['diagnosis'] = "2nd-degree AV Block"
        results['confidence'] = results['av_dissociation_score']
    else:
        results['diagnosis'] = "Normal AV Conduction"
        results['confidence'] = 1.0 - results['av_dissociation_score']

    return results


def assess_av_dissociation(ecg_signal, r_peaks, p_peaks, sampling_rate):
    """Assess AV dissociation by analyzing P-QRS relationships."""
    
    result = {
        'score': 0.0,
        'evidence': [],
        'failures': [],
        'conduction_ratio': None,
        'pr_cv': None,
        'phase_lock_corr': None,
    }

    # Stricter P→R conduction window (80–220 ms)
    min_pr = int(0.08 * sampling_rate)  # 80ms
    max_pr = int(0.22 * sampling_rate)  # 220ms
    
    conducted_count = 0
    pr_intervals = []
    pr_series = []
    p_series = []
    r_series = []
    conducted_flags_raw = []
    pr_intervals_raw = []
    
    for p_idx in p_peaks:
        following_r = r_peaks[r_peaks > p_idx]
        if len(following_r) > 0:
            pr_interval = following_r[0] - p_idx
            pr_series.append(pr_interval)
            p_series.append(p_idx)
            r_series.append(following_r[0])
            is_conducted = (min_pr <= pr_interval <= max_pr)
            conducted_flags_raw.append(is_conducted)
            if is_conducted:
                pr_intervals_raw.append(pr_interval / sampling_rate)
            else:
                pr_intervals_raw.append(None)

    # Filter: only keep conduction if it appears in runs >=2 with low jitter (<30 ms)
    conducted_flags = [False] * len(conducted_flags_raw)
    filtered_pr_intervals = []
    i = 0
    while i < len(conducted_flags_raw):
        if conducted_flags_raw[i]:
            j = i
            tmp_vals = []
            while j < len(conducted_flags_raw) and conducted_flags_raw[j]:
                if pr_intervals_raw[j] is not None:
                    tmp_vals.append(pr_intervals_raw[j])
                j += 1
            run_len = j - i
            if run_len >= 2 and len(tmp_vals) >= 2:
                # jitter threshold 30 ms
                if np.std(tmp_vals) <= 0.03:
                    for k in range(i, j):
                        conducted_flags[k] = True
                    filtered_pr_intervals.extend(tmp_vals)
            i = j
        else:
            i += 1

    conducted_count = int(np.sum(conducted_flags))

    # Calculate conduction ratio
    conduction_ratio = conducted_count / len(p_peaks) if len(p_peaks) > 0 else 0
    result['conduction_ratio'] = float(conduction_ratio)

    # Phase-locking correlation between P and R sequences (low corr suggests independence)
    if len(p_series) >= 3 and len(r_series) >= 3:
        try:
            p_t = np.array(p_series) / float(sampling_rate)
            r_t = np.array(r_series) / float(sampling_rate)
            k = min(len(p_t), len(r_t))
            p_t = p_t[:k]; r_t = r_t[:k]
            corr = np.corrcoef(p_t, r_t)[0, 1]
            result['phase_lock_corr'] = float(corr)
            if corr < 0.2:
                result['score'] += 0.2
                result['evidence'].append(f"low_phase_lock_corr {corr:.2f}")
            elif corr < 0.4:
                result['score'] += 0.1
                result['evidence'].append(f"moderate_phase_lock_corr {corr:.2f}")
        except Exception:
            pass

    # Conduction category
    if conduction_ratio < 0.1:
        result['score'] += 0.3
        result['evidence'].append(f"Complete AV dissociation (conduction: {conduction_ratio:.1%})")
    elif conduction_ratio < 0.2:
        result['score'] += 0.2
        result['evidence'].append(f"High-grade AV block (conduction: {conduction_ratio:.1%})")
    elif conduction_ratio < 0.5:
        result['score'] += 0.1
        result['evidence'].append(f"Partial AV block (conduction: {conduction_ratio:.1%})")
    else:
        result['failures'].append(f"Normal AV conduction (conduction: {conduction_ratio:.1%})")

    # PR variability based on filtered intervals
    if len(filtered_pr_intervals) > 1:
        pr_cv = float(np.std(filtered_pr_intervals) / np.mean(filtered_pr_intervals)) if np.mean(filtered_pr_intervals) > 0 else None
        result['pr_cv'] = pr_cv
        if pr_cv and pr_cv > 0.3:
            result['score'] += 0.1
            result['evidence'].append(f"High PR variability suggests dissociation (CV: {pr_cv:.3f})")
    else:
        result['pr_cv'] = None

    return result


def create_3davb_visualizations(ecg_signal, r_peaks, p_peaks, rr_intervals, pp_intervals,
                               analysis_results, ecg_title, lead_name, output_dir, sampling_rate):
    """Create comprehensive 3dAVB visualizations."""
    
    image_paths = []
    t = np.linspace(0, len(ecg_signal) / sampling_rate, len(ecg_signal), endpoint=False)

    # 1. Main ECG overview with AV dissociation analysis
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_ecg_with_grid(t, ecg_signal, 
                      f"{ecg_title} ({lead_name}) - 3rd-degree AV Block Analysis", 
                      ax, sampling_rate)
    
    # Plot R peaks
    if len(r_peaks) > 0:
        ax.scatter(t[r_peaks], ecg_signal[r_peaks], c='blue', s=40, 
                  marker='o', label='R-peaks', zorder=5)
    
    # Plot P peaks
    if len(p_peaks) > 0:
        ax.scatter(t[p_peaks], ecg_signal[p_peaks], c='red', s=30, 
                  marker='^', label='P-waves', zorder=5)
    
    # Show P wave marching through (independence)
    if len(p_peaks) > 1:
        for i in range(len(p_peaks) - 1):
            p1, p2 = p_peaks[i], p_peaks[i + 1]
            if p1 < len(t) and p2 < len(t):
                ax.plot([t[p1], t[p2]], [ecg_signal[p1], ecg_signal[p2]], 
                       'r--', alpha=0.6, linewidth=1.5)
        ax.plot([], [], 'r--', alpha=0.6, linewidth=1.5, label='P wave independence')

    # Add diagnosis annotation
    # Removed per user request: no top-left label box for diagnosis/confidence/score
    # diagnosis_color = 'red' if 'Complete' in analysis_results['diagnosis'] else 'orange'
    # ax.text(0.02, 0.98, 
    #        f"Diagnosis: {analysis_results['diagnosis']}\n"
    #        f"Confidence: {analysis_results['confidence']:.2f}\n"
    #        f"Score: {analysis_results['av_dissociation_score']:.2f}/1.0",
    #        transform=ax.transAxes, fontsize=11, verticalalignment='top',
    #        bbox=dict(boxstyle='round', facecolor=diagnosis_color, alpha=0.8), 
    #        color='white', weight='bold')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avb3_overview.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    image_paths.append('avb3_overview.png')

    # 2. Rhythm analysis - PP and RR intervals
    if len(pp_intervals) > 0 and len(rr_intervals) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # PP intervals
        beat_nums_p = np.arange(1, len(pp_intervals) + 1)
        ax1.plot(beat_nums_p, pp_intervals, 'ro-', linewidth=2, markersize=6)
        ax1.set_title('Atrial Rhythm (PP Intervals)', fontsize=14, weight='bold')
        ax1.set_xlabel('Beat Number')
        ax1.set_ylabel('PP Interval (s)')
        ax1.grid(True, alpha=0.3)
        pp_mean = np.mean(pp_intervals)
        pp_std = np.std(pp_intervals)
        ax1.axhline(pp_mean, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {pp_mean:.3f}s ± {pp_std:.3f}s')
        ax1.legend()
        
        # RR intervals  
        beat_nums_r = np.arange(1, len(rr_intervals) + 1)
        ax2.plot(beat_nums_r, rr_intervals, 'bo-', linewidth=2, markersize=6)
        ax2.set_title('Ventricular Rhythm (RR Intervals)', fontsize=14, weight='bold')
        ax2.set_xlabel('Beat Number')
        ax2.set_ylabel('RR Interval (s)')
        ax2.grid(True, alpha=0.3)
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        ax2.axhline(rr_mean, color='blue', linestyle='--', alpha=0.7,
                   label=f'Mean: {rr_mean:.3f}s ± {rr_std:.3f}s')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avb3_rhythm_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        image_paths.append('avb3_rhythm_analysis.png')

    # 3. Rate comparison histogram
    if len(pp_intervals) > 0 and len(rr_intervals) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # PP histogram
        ax1.hist(pp_intervals, bins=min(10, len(pp_intervals)), alpha=0.7, 
                color='red', edgecolor='black')
        ax1.set_title('Atrial Intervals Distribution', fontsize=12, weight='bold')
        ax1.set_xlabel('PP Interval (s)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(pp_intervals), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(pp_intervals):.3f}s')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RR histogram
        ax2.hist(rr_intervals, bins=min(10, len(rr_intervals)), alpha=0.7, 
                color='blue', edgecolor='black')
        ax2.set_title('Ventricular Intervals Distribution', fontsize=12, weight='bold')
        ax2.set_xlabel('RR Interval (s)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(rr_intervals), color='darkblue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rr_intervals):.3f}s')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avb3_interval_distributions.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        image_paths.append('avb3_interval_distributions.png')

    return image_paths


def generate_clinical_report(analysis_results, rr_rate, pp_rate):
    """Generate comprehensive clinical report for 3dAVB."""
    
    report_lines = [
        "=== 3rd-Degree AV Block Clinical Analysis ===",
        "",
        f"Final Diagnosis: {analysis_results['diagnosis']}",
        f"Diagnostic Confidence: {analysis_results['confidence']:.2f}",
        f"AV Dissociation Score: {analysis_results['av_dissociation_score']:.2f}/1.0",
        "",
        "=== Heart Rates ===",
        f"Atrial Rate: {pp_rate:.1f} bpm" if pp_rate else "Atrial Rate: Not measurable",
        f"Ventricular Rate: {rr_rate:.1f} bpm" if rr_rate else "Ventricular Rate: Not measurable",
    ]
    
    if pp_rate and rr_rate:
        report_lines.append(f"Atrial/Ventricular Ratio: {pp_rate/rr_rate:.2f}")
    
    report_lines.extend([
        "",
        f"=== Clinical Criteria Met ({len(analysis_results['criteria_met'])}) ===",
    ])
    
    for criterion in analysis_results['criteria_met']:
        report_lines.append(f"✅ {criterion}")
    
    if analysis_results['criteria_failed']:
        report_lines.extend([
            "",
            f"=== Clinical Criteria Not Met ({len(analysis_results['criteria_failed'])}) ===",
        ])
        for criterion in analysis_results['criteria_failed']:
            report_lines.append(f"❌ {criterion}")
    
    report_lines.extend([
        "",
        "=== Clinical Interpretation ===",
    ])
    
    if analysis_results['confidence'] >= 0.7:
        report_lines.append("High confidence 3rd-degree AV block diagnosis.")
        report_lines.append("Complete AV dissociation with independent atrial and ventricular rhythms.")
    elif analysis_results['confidence'] >= 0.5:
        report_lines.append("Possible high-grade AV block.")
        report_lines.append("Consider further evaluation and monitoring.")
    else:
        report_lines.append("Low probability of 3rd-degree AV block.")
        report_lines.append("Consider other diagnoses.")
    
    return "\n".join(report_lines)


def compile_metrics(analysis_results, r_peaks, p_peaks, rr_rate, pp_rate):
    """Compile comprehensive metrics for 3dAVB analysis."""
    
    return {
        'confidence': float(analysis_results['confidence']),
        'dissociation_score': float(analysis_results['av_dissociation_score']),
        'diagnosis_confidence': float(analysis_results['confidence']),
        'conduction_ratio': float(analysis_results.get('conduction_ratio')) if analysis_results.get('conduction_ratio') is not None else None,
        'pp_rate_bpm': float(pp_rate) if pp_rate else None,
        'rr_rate_bpm': float(rr_rate) if rr_rate else None,
        'pr_cv': float(analysis_results.get('pr_cv')) if analysis_results.get('pr_cv') is not None else None,
        'phase_lock_corr': float(analysis_results.get('phase_lock_corr')) if analysis_results.get('phase_lock_corr') is not None else None,
        'total_p_count': int(len(p_peaks)),
        'total_r_count': int(len(r_peaks)),
        'complete_av_block': bool(analysis_results.get('diagnosis', '').startswith('Complete AV Block')),
        'severe_bradycardia': bool(rr_rate is not None and rr_rate < 45),
        'atrial_faster_than_ventricular': bool(pp_rate and rr_rate and pp_rate > rr_rate),
    }