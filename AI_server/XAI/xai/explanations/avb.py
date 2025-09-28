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
    This version uses improved PR interval calculation logic for clinical accuracy.
    """
    if ecg_signal is None: 
        print("Error: ECG signal is None")
        return None

    # 1. Signal cleaning (consistent with Jupyter)
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception as e:
        print(f"Error in ECG cleaning: {e}")
        return None

    # 2. R-wave detection (consistent with Jupyter)
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_peak_indices = rpeaks['ECG_R_Peaks']
        if len(r_peak_indices) < 3:
            print(f"Error: Insufficient R-peaks detected ({len(r_peak_indices)} < 3)")
            return None
    except Exception as e:
        print(f"Error in R-peak detection: {e}")
        return None

    # 3. Try multiple delineation methods in order
    waves, used_method = None, None
    for method in ["prominence", "dwt", "cwt", "peak"]:
        try:
            print(f"--- Trying method: '{method}' ---")
            _, w = nk.ecg_delineate(ecg_cleaned, rpeaks=r_peak_indices, sampling_rate=sampling_rate, method=method)
            waves_lower = {k.lower(): v for k, v in w.items()}
            
            # 验证P波和R波检测的质量
            if 'ecg_p_onsets' in waves_lower and 'ecg_r_onsets' in waves_lower:
                p_onsets_raw = np.asarray(waves_lower['ecg_p_onsets'], dtype=float)
                r_onsets_raw = np.asarray(waves_lower['ecg_r_onsets'], dtype=float)
                
                # 检查是否有足够的有效检测点
                valid_p = np.sum(~np.isnan(p_onsets_raw))
                valid_r = np.sum(~np.isnan(r_onsets_raw))
                
                if valid_p > 0 and valid_r > 0 and valid_p >= len(r_peak_indices) * 0.5:
                    waves, used_method = waves_lower, method
                    print(f"✓ Delineation successful using method: '{used_method}'")
                    print(f"  Valid P-onsets: {valid_p}, Valid R-onsets: {valid_r}")
                    break
                else:
                    print(f"  Method '{method}' insufficient: P={valid_p}, R={valid_r}")
        except Exception as e:
            print(f"× Method '{method}' failed with error: {e}")
            
    if waves is None:
        print("All delineation methods failed. Cannot proceed with PR interval analysis.")
        return None

    # 4. Improved PR interval calculation with clinical validation
    p_onsets_raw = np.asarray(waves['ecg_p_onsets'], dtype=float)
    r_onsets_raw = np.asarray(waves['ecg_r_onsets'], dtype=float)

    # 过滤掉NaN值并转换为整数索引
    p_onsets = p_onsets_raw[~np.isnan(p_onsets_raw)].astype(int)
    r_onsets = r_onsets_raw[~np.isnan(r_onsets_raw)].astype(int)
    
    if len(p_onsets) == 0 or len(r_onsets) == 0:
        print("Error: No valid P-onsets or R-onsets detected")
        return None

    # 改进的PR interval配对算法
    pr_intervals, highlight_regions, pr_quality_scores = [], [], []
    
    for p_idx in p_onsets:
        # 寻找在P波之后的最近R波
        valid_r_indices = [r for r in r_onsets if r > p_idx]
        
        if valid_r_indices:
            # 选择最近的R波
            r_idx = min(valid_r_indices)
            
            # 计算PR interval
            pr_sec = (r_idx - p_idx) / sampling_rate
            
            # 临床验证：PR interval应该在生理范围内
            if 0.05 < pr_sec < 0.60:  # 50ms to 600ms
                pr_intervals.append(pr_sec)
                highlight_regions.append((p_idx, r_idx))
                
                # 计算质量分数（基于信号质量）
                quality_score = _calculate_pr_quality_score(ecg_cleaned, p_idx, r_idx, sampling_rate)
                pr_quality_scores.append(quality_score)
                
                print(f"PR interval: {pr_sec:.3f}s (P:{p_idx}, R:{r_idx}, Quality:{quality_score:.2f})")
            else:
                print(f"PR interval {pr_sec:.3f}s outside physiological range, skipping")
    
    if len(pr_intervals) == 0:
        print("Error: No valid PR intervals calculated")
        return None
    
    pr_intervals = np.array(pr_intervals)
    pr_quality_scores = np.array(pr_quality_scores)
    
    # 5. Enhanced plotting with quality indicators
    time_axis = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_ecg_with_grid(time_axis, ecg_cleaned, f'{ecg_title} ({lead_name}) - PR Intervals Analysis', ax, sampling_rate)
    
    # Plot R-peaks
    ax.scatter(time_axis[r_peak_indices], ecg_cleaned[r_peak_indices], c="dodgerblue", s=40, zorder=5, label="R-peaks")
    
    # Plot P-onsets
    p_times = time_axis[p_onsets]
    p_values = ecg_cleaned[p_onsets]
    ax.scatter(p_times, p_values, c="red", s=30, zorder=5, label="P-onsets", marker='^')

    # Highlight PR intervals with quality-based colors
    for i, (p, r) in enumerate(highlight_regions):
        pr_sec = (r - p) / sampling_rate
        quality = pr_quality_scores[i] if i < len(pr_quality_scores) else 0.5
        
        # 颜色编码：绿色=正常PR，橙色=延长PR，红色=低质量
        if quality < 0.3:
            color = "red"  # 低质量
        elif pr_sec > 0.20:
            color = "orange"  # 延长PR
        else:
            color = "green"  # 正常PR
            
        ax.axvspan(time_axis[p], time_axis[r], alpha=0.4, color=color, zorder=4)

    # Enhanced legend
    handles, _ = ax.get_legend_handles_labels()
    green_patch = mpatches.Patch(color='green', alpha=0.4, label='Normal PR (≤ 0.20s)')
    orange_patch = mpatches.Patch(color='orange', alpha=0.4, label='Prolonged PR (> 0.20s)')
    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Low Quality PR')
    handles.extend([green_patch, orange_patch, red_patch])
    ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avb1_pr_intervals_enhanced.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 6. Enhanced histogram with quality distribution
    if pr_intervals.size > 0:
        fig_hist, (ax_hist, ax_quality) = plt.subplots(2, 1, figsize=(8, 8))
        
        # PR interval distribution
        ax_hist.hist(pr_intervals, bins='auto', edgecolor='k', alpha=0.7, color='skyblue')
        ax_hist.axvline(0.20, color="orange", ls="--", lw=2, label="0.20s threshold")
        ax_hist.axvline(0.12, color="green", ls="--", lw=2, label="0.12s normal")
        ax_hist.set_title("PR Interval Distribution")
        ax_hist.set_xlabel("PR Interval (s)")
        ax_hist.set_ylabel("Count")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Quality score distribution
        if len(pr_quality_scores) > 0:
            ax_quality.hist(pr_quality_scores, bins='auto', edgecolor='k', alpha=0.7, color='lightcoral')
            ax_quality.axvline(0.5, color="orange", ls="--", lw=2, label="0.5 quality threshold")
            ax_quality.set_title("PR Interval Quality Distribution")
            ax_quality.set_xlabel("Quality Score")
            ax_quality.set_ylabel("Count")
            ax_quality.legend()
            ax_quality.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "avb1_enhanced_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_hist)
    
    # 7. Enhanced report generation
    report_text = generate_report_1davb(pr_intervals, used_method, ecg_title, lead_name)
    print(report_text)
    
    # Enhanced metrics with quality information
    metrics = {
        "pr_mean_s": float(np.mean(pr_intervals)),
        "pr_min_s": float(np.min(pr_intervals)),
        "pr_max_s": float(np.max(pr_intervals)),
        "pr_std_s": float(np.std(pr_intervals)),
        "prolonged_pr_count": int(np.sum(pr_intervals > 0.20)),
        "total_pr_count": len(pr_intervals),  # This is the key field for validation
        "pr_quality_mean": float(np.mean(pr_quality_scores)) if len(pr_quality_scores) > 0 else 0.0,
        "pr_quality_min": float(np.min(pr_quality_scores)) if len(pr_quality_scores) > 0 else 0.0,
        "delineation_method": used_method,
        "signal_duration_s": float(len(ecg_cleaned) / sampling_rate)
    }
    
    # Build concise summary/statistics with improved 1dAVB detection criteria
    prolonged = int(np.sum(pr_intervals > 0.20))
    total = int(len(pr_intervals))
    
    # Improved 1dAVB detection criteria:
    # 1. Mean PR > 0.20s OR significant portion (≥50%) of intervals > 0.20s
    # 2. More lenient criteria for borderline cases (mean PR ≥ 0.20s)
    mean_pr = metrics["pr_mean_s"]
    prolonged_ratio = (prolonged / total) if total > 0 else 0.0
    
    # Enhanced detection logic for 1dAVB
    present = bool(
        (mean_pr >= 0.20 and prolonged_ratio >= 0.5) or  # Borderline mean with significant prolonged intervals
        (mean_pr > 0.20 and prolonged_ratio >= 0.3) or   # Clear mean prolongation with some prolonged intervals
        (prolonged_ratio >= 0.8)                         # Majority of intervals prolonged
    )
    
    summary = {
        "report_text": report_text,
        "is_abnormality_present": present,
        "abnormality_count": int(prolonged if present else 0),
    }
    statistics = {
        "total_beats": total,
        "pr_mean_s": metrics["pr_mean_s"],
        "pr_min_s": metrics["pr_min_s"],
        "pr_max_s": metrics["pr_max_s"],
        "prolonged_pr_count": prolonged,
        "total_pr_count": total,  # Add this field for validator compatibility
    }
    
    report_data = {
        "title": ecg_title, 
        "lead_name": lead_name, 
        "report_text": report_text, 
        "summary": summary,
        "statistics": statistics,
        "image_paths": ["avb1_pr_intervals_enhanced.png", "avb1_enhanced_analysis.png"],
        # Include metrics directly for validation compatibility
        "pr_mean_s": metrics["pr_mean_s"],
        "total_pr_count": metrics["total_pr_count"],
        "prolonged_pr_count": metrics["prolonged_pr_count"],
    }
    
    return report_data

def _calculate_pr_quality_score(ecg_signal, p_idx, r_idx, sampling_rate):
    """
    Calculate quality score for PR interval based on signal characteristics.
    
    Returns:
        float: Quality score between 0 and 1 (1 = best quality)
    """
    try:
        # 检查信号幅度
        p_amplitude = abs(ecg_signal[p_idx])
        r_amplitude = abs(ecg_signal[r_idx])
        
        # 检查信号稳定性（周围区域的标准差）
        window = int(0.02 * sampling_rate)  # 20ms window
        p_start = max(0, p_idx - window)
        p_end = min(len(ecg_signal), p_idx + window)
        r_start = max(0, r_idx - window)
        r_end = min(len(ecg_signal), r_idx + window)
        
        p_stability = 1.0 / (1.0 + np.std(ecg_signal[p_start:p_end]))
        r_stability = 1.0 / (1.0 + np.std(ecg_signal[r_start:r_end]))
        
        # 综合质量分数
        amplitude_score = min(p_amplitude, r_amplitude) / max(p_amplitude, r_amplitude)
        stability_score = (p_stability + r_stability) / 2.0
        
        quality_score = (amplitude_score * 0.4 + stability_score * 0.6)
        return min(1.0, max(0.0, quality_score))
        
    except Exception as e:
        print(f"Error calculating quality score: {e}")
        return 0.5  # 默认中等质量