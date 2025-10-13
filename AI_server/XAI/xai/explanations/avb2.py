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

    # Detect R peaks with improved parameters
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_idx = rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}")
        return None

    # Enhanced delineation with multiple methods and quality assessment
    waves_lower = None
    delineation_quality = 0
    for method in ["prominence", "dwt", "cwt", "peak"]:
        try:
            _, w = nk.ecg_delineate(ecg_cleaned, rpeaks=r_idx, sampling_rate=sampling_rate, method=method)
            waves_lower = {k.lower(): v for k, v in w.items()}
            if 'ecg_p_onsets' in waves_lower and 'ecg_r_onsets' in waves_lower:
                # Assess quality by counting valid detections
                p_onsets_raw = np.asarray(waves_lower['ecg_p_onsets'], dtype=float)
                r_onsets_raw = np.asarray(waves_lower['ecg_r_onsets'], dtype=float)
                valid_p = np.sum(~np.isnan(p_onsets_raw))
                valid_r = np.sum(~np.isnan(r_onsets_raw))
                quality = (valid_p + valid_r) / (len(p_onsets_raw) + len(r_onsets_raw)) if (len(p_onsets_raw) + len(r_onsets_raw)) > 0 else 0
                if quality > delineation_quality:
                    delineation_quality = quality
                if quality > 0.7:  # Good enough quality
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

    # Improved P-R pairing with adaptive conduction window
    def find_best_pr_pair(p_onset, r_onsets, sampling_rate):
        """Find the best R-wave for a given P-wave based on physiological constraints."""
        # Ensure r_onsets is a numpy array
        r_onsets = np.asarray(r_onsets)
        
        # Physiological PR range: 0.12-0.20s normal, up to 0.40s in AVB
        min_pr = int(0.08 * sampling_rate)   # 80ms minimum
        max_pr = int(0.45 * sampling_rate)   # 450ms maximum (allowing for severe AVB)
        
        candidates = r_onsets[(r_onsets > p_onset) & 
                             (r_onsets - p_onset >= min_pr) & 
                             (r_onsets - p_onset <= max_pr)]
        
        # Ensure candidates is a numpy array
        candidates = np.asarray(candidates)
        
        if candidates.size == 0:
            return None, None
        
        # Prefer the closest R within normal PR range (0.12-0.20s)
        normal_range = (int(0.12 * sampling_rate), int(0.20 * sampling_rate))
        normal_candidates = candidates[(candidates - p_onset >= normal_range[0]) & 
                                     (candidates - p_onset <= normal_range[1])]
        
        # Ensure normal_candidates is a numpy array
        normal_candidates = np.asarray(normal_candidates)
        
        if normal_candidates.size > 0:
            best_r = normal_candidates[np.argmin(normal_candidates - p_onset)]
        else:
            # If no normal PR, take the closest within extended range
            best_r = candidates[np.argmin(candidates - p_onset)]
        
        pr_s = (best_r - p_onset) / sampling_rate
        return best_r, pr_s

    # Enhanced P-R pairing with morphological validation
    pr_intervals, conducted_pairs, dropped_p_indices = [], [], []
    conduction_quality_scores = []
    
    for p in p_onsets:
        best_r, pr_s = find_best_pr_pair(p, r_onsets, sampling_rate)
        
        if best_r is not None and pr_s is not None:
            # Additional morphological validation
            p_amplitude = ecg_cleaned[p]
            r_amplitude = ecg_cleaned[best_r]
            
            # Check if P and R have reasonable amplitudes
            if abs(p_amplitude) > 0.1 * np.std(ecg_cleaned) and abs(r_amplitude) > 0.3 * np.std(ecg_cleaned):
                pr_intervals.append(pr_s)
                conducted_pairs.append((p, best_r))
                
                # Calculate conduction quality score
                quality_score = 1.0
                if 0.12 <= pr_s <= 0.20:  # Normal PR range
                    quality_score = 1.0
                elif 0.08 <= pr_s <= 0.45:  # Extended range
                    quality_score = 0.7
                else:
                    quality_score = 0.3
                conduction_quality_scores.append(quality_score)
            else:
                dropped_p_indices.append(p)
        else:
            dropped_p_indices.append(p)
    
    # Convert to numpy arrays for consistency
    pr_intervals = np.array(pr_intervals)
    conducted_pairs = np.array(conducted_pairs)
    dropped_p_indices = np.array(dropped_p_indices)
    conduction_quality_scores = np.array(conduction_quality_scores)
    
    # Handle edge cases where no P waves or R waves are detected
    if len(p_onsets) == 0:
        print(f"No P waves detected for {ecg_title}")
        return None
    
    if len(r_onsets) == 0:
        print(f"No R waves detected for {ecg_title}")
        return None
    
    # Check if we have any valid PR intervals
    if len(pr_intervals) == 0:
        print(f"No valid PR intervals found for {ecg_title}")
        # Still continue with analysis, but set default values
        pr_intervals = np.array([])
        conducted_pairs = np.array([])
        dropped_p_indices = np.array([p for p in p_onsets])  # Mark all P waves as dropped
        conduction_quality_scores = np.array([])
        
        # Set default subtype for complete block
        subtype = "2nd-degree AVB (complete block)"
        subtype_confidence = 0.8
        wenckebach_sequences = 0
        wenckebach_confidence = 0.0
        overall_conduction_quality = 0.0

    # Enhanced Wenckebach detection with adaptive thresholds
    wenckebach_sequences = 0
    wenckebach_confidence = 0.0
    
    if len(pr_intervals) >= 3 and len(dropped_p_indices) > 0:
        # Adaptive threshold based on signal quality
        base_threshold = 0.008  # 8ms base threshold
        quality_factor = 1.0 - (delineation_quality * 0.3)  # Reduce threshold for better quality
        adaptive_threshold = base_threshold * quality_factor
        
        # Look for progressive PR prolongation
        diffs = np.diff(pr_intervals)
        progressive_count = 0
        total_progressive = 0
        
        for i, d in enumerate(diffs):
            if d > adaptive_threshold:
                progressive_count += 1
                total_progressive += d
            else:
                if progressive_count >= 2:  # At least 2 consecutive increases
                    wenckebach_sequences += 1
                    # Calculate confidence based on consistency
                    avg_increase = total_progressive / progressive_count
                    wenckebach_confidence = min(1.0, avg_increase / 0.02)  # Normalize to 20ms
                progressive_count = 0
                total_progressive = 0
        
        # Check for final sequence
        if progressive_count >= 2:
            wenckebach_sequences += 1
            avg_increase = total_progressive / progressive_count
            wenckebach_confidence = min(1.0, avg_increase / 0.02)

    # Improved subtype classification with confidence scoring
    subtype = None
    subtype_confidence = 0.0
    pr_std = float(np.std(pr_intervals)) if pr_intervals.size > 0 else None
    
    if len(dropped_p_indices) >= 1:
        if wenckebach_sequences >= 1 and wenckebach_confidence > 0.6:
            subtype = "Mobitz I (Wenckebach)"
            subtype_confidence = wenckebach_confidence
        elif pr_std is not None and pr_std < 0.025:  # Tighter threshold for Mobitz II
            subtype = "Mobitz II"
            subtype_confidence = 1.0 - (pr_std / 0.025)  # Higher confidence for lower variability
        elif len(dropped_p_indices) >= 2:
            subtype = "2nd-degree AVB (high-grade)"
            subtype_confidence = 0.8
        else:
            subtype = "2nd-degree AVB (unspecified)"
            subtype_confidence = 0.6

    # Calculate overall conduction quality
    overall_conduction_quality = np.mean(conduction_quality_scores) if len(conduction_quality_scores) > 0 else 0.0

    # Plot ECG with annotations
    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - 2nd-degree AV Block", ax, sampling_rate)
    if len(r_idx):
        ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=30, zorder=5, label='R-peaks')
    
    # Mark conducted PR spans (only if we have conducted pairs)
    if len(conducted_pairs) > 0:
        for p, r in conducted_pairs:
            if p < len(t) and r < len(t):
                ax.axvspan(t[p], t[r], alpha=0.35, color='green')
    
    # Mark dropped P regions (only if we have dropped indices)
    if len(dropped_p_indices) > 0:
        for p in dropped_p_indices:
            if p < len(t):
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
    if pr_intervals.size > 0:
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

    # Defer report text generation to ReportGenerator; only metrics and images remain
    dropped_count = len(dropped_p_indices)

    # Prepare metrics for unified report generator
    metrics = {
        'total_p_count': int(len(p_onsets)),
        'total_r_count': int(len(r_onsets)),
        'conducted_count': int(len(conducted_pairs)),
        'dropped_beats': int(dropped_count),
        'dropped_p_count': int(len(dropped_p_indices)),
        'pr_mean_s': float(np.mean(pr_intervals)) if pr_intervals.size > 0 else None,
        'pr_std_s': pr_std,
        'wenckebach_seq_count': int(wenckebach_sequences),
        'subtype': subtype,
        'subtype_confidence': subtype_confidence,
        'conduction_ratio': overall_conduction_quality,
        'total_beats': int(len(r_onsets)),
        'pr_variation': pr_std if pr_intervals.size > 0 else 0,
    }
    
    # Use unified report generator
    from ..clinical_reports import ReportGenerator
    report_generator = ReportGenerator()
    
    # Generate complete report using unified generator
    report_data = report_generator.generate_avb2_report(metrics, {})
    
    # Add lead_name and image_paths to the report
    report_data['lead_name'] = lead_name
    
    # Prepare image paths
    image_paths = {
        'avb2_overview.png': os.path.join(output_dir, 'avb2_overview.png')
    }
    if pr_intervals.size > 0:
        image_paths['avb2_pr_trend.png'] = os.path.join(output_dir, 'avb2_pr_trend.png')
        image_paths['avb2_pr_hist.png'] = os.path.join(output_dir, 'avb2_pr_hist.png')
    
    report_data['image_paths'] = image_paths
    
    return report_data
