import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.patches import Patch
from ..ecg_utils import plot_ecg_with_grid


def _bandpass(signal, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)


def _build_qrs_template(ecg_cleaned: np.ndarray, r_idx: np.ndarray, fs: float, median_rr: float, qrs_widths: np.ndarray):
    win = int(0.12 * fs)
    templates = []
    rr = np.diff(r_idx) / fs if r_idx.size > 1 else np.array([])
    for k in range(1, len(r_idx) - 1):
        # use only narrow likely-sinus beats for template (relaxed width threshold)
        if k < qrs_widths.size and (not np.isnan(qrs_widths[k])) and qrs_widths[k] < 0.12:
            ok_prev = (k-1 < len(rr)) and (abs(rr[k-1] - (median_rr or rr[k-1])) <= (0.1 * (median_rr or rr[k-1])))
            ok_next = (k < len(rr)) and (abs(rr[k] - (median_rr or rr[k])) <= (0.1 * (median_rr or rr[k])))
            if ok_prev and ok_next:
                s = max(0, r_idx[k] - win)
                e = min(len(ecg_cleaned), r_idx[k] + win)
                seg = ecg_cleaned[s:e]
                if seg.size == 2 * win:
                    v = seg - np.mean(seg)
                    n = np.linalg.norm(v) + 1e-8
                    templates.append(v / n)
        if len(templates) >= 5:
            break
    return np.mean(templates, axis=0) if templates else None


def _estimate_qrs_onset_offsets(ecg: np.ndarray, r_idx: np.ndarray, fs: float):
    """Simplified QRS onset/offset detection using amplitude thresholds.
    - Onset: first point before R where signal drops below 10% of R-peak amplitude
    - Offset: first point after R where signal drops below 10% of R-peak amplitude
    - Use fixed windows to avoid edge cases
    """
    n = len(ecg)
    onset_idx = np.full(len(r_idx), -1, dtype=int)
    offset_idx = np.full(len(r_idx), -1, dtype=int)
    width_s = np.full(len(r_idx), np.nan, dtype=float)

    # Fixed search windows
    left_win = int(0.12 * fs)  # 120ms before R
    right_win = int(0.20 * fs)  # 200ms after R

    for i, r in enumerate(r_idx):
        s = max(0, r - left_win)
        e = min(n, r + right_win)
        
        if e <= s:
            continue
            
        # Get signal segment around R-peak
        seg = ecg[s:e]
        r_rel = r - s  # R-peak position in segment
        
        if r_rel >= len(seg):
            continue
            
        # R-peak amplitude
        r_amp = abs(ecg[r])
        if r_amp < 1e-6:
            continue
            
        # Threshold: 10% of R-peak amplitude
        threshold = 0.1 * r_amp
        
        # Onset: scan backward from R-peak
        li = r
        for kidx in range(r_rel, 0, -1):
            if abs(seg[kidx]) < threshold:
                li = s + kidx
                break
        
        # Offset: scan forward from R-peak
        ri = r
        for kidx in range(r_rel, len(seg)):
            if abs(seg[kidx]) < threshold:
                ri = s + kidx
                break
        
        onset_idx[i] = li
        offset_idx[i] = ri
        
        # Calculate width
        if ri > li:
            width_s[i] = (ri - li) / fs
        else:
            # Fallback: use fixed width based on signal characteristics
            width_s[i] = 0.08  # Default 80ms for narrow QRS
    
    return onset_idx, offset_idx, width_s


def _detect_pvc(ecg_cleaned: np.ndarray, fs: float, aux_v1: np.ndarray | None = None):
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="neurokit")
        r_idx = np.asarray(rpeaks['ECG_R_Peaks'], dtype=int)
    except Exception:
        r_idx = np.array([], dtype=int)

    if r_idx.size < 3:
        return r_idx, np.array([], dtype=int), {}, []

    rr = np.diff(r_idx) / fs if r_idx.size > 1 else np.array([])
    median_rr = float(np.median(rr)) if rr.size else None
    baseline_rr = median_rr

    # QRS onset/offset based widths (tied to each R)
    on_idx, off_idx, qrs_widths = _estimate_qrs_onset_offsets(ecg_cleaned, r_idx, fs)
    qrs_threshold_s = 0.12

    # Build baseline QRS template for morphology abnormality (PVC expect low similarity)
    qrs_template = _build_qrs_template(ecg_cleaned, r_idx, fs, median_rr, qrs_widths)

    # Precompute atrial signal and global P delineation to assess P absence
    blanked = ecg_cleaned.copy()
    t_blank = int(0.24 * fs)
    for r in r_idx:
        s = r
        e = min(len(blanked), r + t_blank)
        blanked[s:e] = 0.0
    atrial = _bandpass(blanked, fs, 3.0, 12.0)
    # Optional V1 atrial band for P-wave assistance, and raw V1 for morphology support
    atrial_v1 = None
    v1 = None
    if aux_v1 is not None and isinstance(aux_v1, np.ndarray):
        v1 = aux_v1[:len(ecg_cleaned)] if aux_v1.size >= len(ecg_cleaned) else None
        if v1 is not None:
            blanked_v1 = v1.copy()
            for r in r_idx:
                s = r
                e = min(len(blanked_v1), r + t_blank)
                blanked_v1[s:e] = 0.0
            atrial_v1 = _bandpass(blanked_v1, fs, 3.0, 12.0)
    try:
        _, waves = nk.ecg_delineate(ecg_cleaned, sampling_rate=fs, method="cwt", show=False)
        p_peaks_global = np.asarray(waves.get('ECG_P_Peaks', []), dtype=int)
    except Exception:
        p_peaks_global = np.array([], dtype=int)

    pvc_indices = []
    compensatory_flags = []
    candidates = []
    # Weights for overall PVC score
    w_prem, w_width, w_morph_abn, w_comp, w_p_abs = 0.25, 0.25, 0.25, 0.15, 0.10
    win = int(0.12 * fs)
    # Iterate beats except ends for RR context
    for beat_idx in range(1, len(r_idx) - 1):
        i_prev = beat_idx - 1  # RR before beat
        i_next = beat_idx      # RR after beat
        # Prematurity: current RR shorter than baseline and ratio to previous RR
        is_premature = False
        if baseline_rr and i_prev < len(rr):
            ratio_to_baseline = rr[i_prev] / baseline_rr
            ratio_to_prev = rr[i_prev] / max(rr[i_prev-1], 1e-6) if i_prev-1 >= 0 else ratio_to_baseline
            # increase sensitivity slightly per feedback
            is_premature = (ratio_to_baseline < 0.95) or (ratio_to_prev < 0.85)
        # Width assessment
        qrs_w = qrs_widths[beat_idx] if beat_idx < qrs_widths.size else np.nan
        is_wide_core = bool((not np.isnan(qrs_w)) and (qrs_w >= qrs_threshold_s))
        is_wide_border = bool((not np.isnan(qrs_w)) and (0.11 <= qrs_w < qrs_threshold_s))
        is_wide = is_wide_core
        compensatory = False
        if (i_prev < len(rr)) and (i_next < len(rr)) and median_rr:
            # Compensatory pause must follow a short RR and be followed by a long RR around baseline
            rr_prev = rr[i_prev]
            rr_next = rr[i_next]
            was_short = rr_prev <= (0.90 * median_rr)
            is_long = rr_next >= (1.10 * median_rr)
            compensatory = bool(was_short and is_long and ((rr_prev + rr_next) >= (1.80 * median_rr)))
        # Do not hard prefilter here; we may discover dominant negative complexes between R-peaks.

        # Scores
        # prematurity
        prem_score = 0.0
        if baseline_rr and i_prev < len(rr):
            shorten = (baseline_rr - rr[i_prev]) / max(baseline_rr, 1e-6)
            prem_score = float(np.clip(shorten / 0.25, 0.0, 1.0)) if is_premature else 0.0
            # ensure consistency with boolean flag
            if is_premature and prem_score == 0.0:
                prem_score = 0.3
        # width score
        qrs_w = qrs_widths[beat_idx] if beat_idx < qrs_widths.size else np.nan
        if np.isnan(qrs_w):
            width_score = 0.0
        else:
            # below 110ms -> 0, 110-120ms -> 0.3..0.6, 120-160ms -> 0.6..1.0
            if qrs_w < 0.11:
                width_score = 0.0
            elif 0.11 <= qrs_w < 0.12:
                width_score = float(0.3 + (qrs_w - 0.11) * (0.3 / 0.01))
            elif 0.12 <= qrs_w < 0.16:
                width_score = float(0.6 + (qrs_w - 0.12) * (0.4 / 0.04))
            elif qrs_w >= 0.16:
                width_score = 1.0
        # morphology abnormality (1 - similarity)
        morph_abn = 0.5
        if qrs_template is not None and 0 < r_idx[beat_idx] - win and r_idx[beat_idx] + win <= len(ecg_cleaned):
            seg = ecg_cleaned[r_idx[beat_idx]-win:r_idx[beat_idx]+win]
            sim = _xcorr_max_norm(qrs_template, seg, max_lag=int(0.02*fs))
            morph_abn = float(np.clip(1.0 - sim, 0.0, 1.0))
        # Dominant negative S-wave within QRS and across inter-beat interval
        s_qrs = int(on_idx[beat_idx]) if beat_idx < len(on_idx) and on_idx[beat_idx] >= 0 else int(max(0, r_idx[beat_idx] - int(0.08*fs)))
        e_qrs = int(off_idx[beat_idx]) if beat_idx < len(off_idx) and off_idx[beat_idx] >= 0 else int(min(len(ecg_cleaned), r_idx[beat_idx] + int(0.12*fs)))
        qrs_seg = ecg_cleaned[s_qrs:e_qrs] if e_qrs > s_qrs else np.array([])

        ib_start = r_idx[beat_idx]
        ib_end = r_idx[beat_idx + 1] if beat_idx + 1 < len(r_idx) else e_qrs
        inter_seg = ecg_cleaned[ib_start:ib_end] if ib_end > ib_start else np.array([])

        s_min_idx = int(r_idx[beat_idx])
        s_dominant = False
        forced_wide_from_negative = False
        neg_width_s = None
        neg_dom_ratio = 0.0
        if inter_seg.size > 3:
            inter_min_rel = int(np.argmin(inter_seg))
            inter_max_rel = int(np.argmax(inter_seg))
            inter_min_idx = ib_start + inter_min_rel
            inter_max_idx = ib_start + inter_max_rel
            inter_min_amp = float(inter_seg[inter_min_rel])
            inter_max_amp = float(inter_seg[inter_max_rel])
            neg_dom_ratio = abs(inter_min_amp) / max(1e-6, abs(inter_max_amp))
            if (inter_min_idx - ib_start) >= int(0.10 * fs) and (neg_dom_ratio >= 1.10):
                s_min_idx = inter_min_idx
                s_dominant = True
                thr = 0.4 * abs(inter_min_amp)
                li = inter_min_rel
                while li > 0 and abs(inter_seg[li]) >= thr:
                    li -= 1
                ri = inter_min_rel
                while ri < inter_seg.size - 1 and abs(inter_seg[ri]) >= thr:
                    ri += 1
                neg_width_s = (ri - li) / fs if ri > li else 0.0
                if neg_width_s is not None and neg_width_s >= 0.10:
                    forced_wide_from_negative = True
        if (not s_dominant) and qrs_seg.size > 3:
            local_min_rel = int(np.argmin(qrs_seg))
            local_max_rel = int(np.argmax(qrs_seg))
            min_amp = float(qrs_seg[local_min_rel])
            max_amp = float(qrs_seg[local_max_rel])
            s_min_idx = s_qrs + local_min_rel
            s_dominant = (abs(min_amp) >= 1.15 * max(1e-6, abs(max_amp)))

        if forced_wide_from_negative and (np.isnan(qrs_w) or qrs_w < 0.12):
            qrs_w = max(qrs_w if not np.isnan(qrs_w) else 0.0, neg_width_s or 0.0)
            is_wide_core = True
            is_wide = True
            morph_abn = max(morph_abn, 0.95)
        # V1 negative-dominant support (raw V1 between current and next R)
        v1_neg_dom_ratio = 0.0
        v1_neg_width_s = None
        v1_support = False
        v1_event_idx = None
        if v1 is not None and (beat_idx + 1) < len(r_idx):
            ibs = r_idx[beat_idx]
            ibe = r_idx[beat_idx + 1]
            seg_v1 = v1[ibs:ibe]
            if seg_v1.size > 3:
                v1_min_rel = int(np.argmin(seg_v1))
                v1_max_rel = int(np.argmax(seg_v1))
                v1_min_amp = float(seg_v1[v1_min_rel])
                v1_max_amp = float(seg_v1[v1_max_rel])
                v1_neg_dom_ratio = abs(v1_min_amp) / max(1e-6, abs(v1_max_amp))
                if (v1_min_rel >= int(0.06 * fs)) and (v1_neg_dom_ratio >= 1.20):
                    thr_v1 = 0.4 * abs(v1_min_amp)
                    li_v1 = v1_min_rel
                    while li_v1 > 0 and abs(seg_v1[li_v1]) >= thr_v1:
                        li_v1 -= 1
                    ri_v1 = v1_min_rel
                    while ri_v1 < seg_v1.size - 1 and abs(seg_v1[ri_v1]) >= thr_v1:
                        ri_v1 += 1
                    v1_neg_width_s = (ri_v1 - li_v1) / fs if ri_v1 > li_v1 else 0.0
                    v1_support = (v1_neg_width_s is not None and v1_neg_width_s >= 0.06)
                    v1_event_idx = ibs + v1_min_rel
        # compensatory score
        comp_score = 1.0 if compensatory else 0.0
        # P absence score in 120–220 ms before R (more stringent P detection)
        lookback = int(0.22 * fs)
        p_start = max(0, r_idx[beat_idx] - lookback)
        p_end = r_idx[beat_idx]
        segp = atrial[p_start:p_end]
        p_present = False
        p_absence_score = 1.0
        if segp.size > 0:
            # More stringent P-wave detection: higher prominence threshold
            peaks, props = find_peaks(segp, distance=int(0.12 * fs), prominence=0.15 * np.std(segp))
            if peaks.size > 0:
                p_present = True
                p_absence_score = 0.0
        # V1-assisted P wave detection
        if atrial_v1 is not None:
            segp_v1 = atrial_v1[p_start:p_end]
            if segp_v1.size > 0:
                peaks_v1, props_v1 = find_peaks(segp_v1, distance=int(0.12 * fs), prominence=0.15 * np.std(segp_v1))
                if peaks_v1.size > 0:
                    p_present = True
                    p_absence_score = 0.0
        glob_hits = p_peaks_global[(p_peaks_global >= p_start) & (p_peaks_global <= p_end)]
        if glob_hits.size > 0:
            p_present = True
            p_absence_score = 0.0

        overall = (
            w_prem * prem_score +
            w_width * width_score +
            w_morph_abn * morph_abn +
            w_comp * comp_score +
            w_p_abs * p_absence_score
        )

        cand_time = r_idx[beat_idx] / fs
        cand = {
            'beat_index': int(beat_idx),
            'time_s': float(cand_time),
            'is_wide_qrs': bool(is_wide),
            'qrs_width_s': float(qrs_w) if not np.isnan(qrs_w) else None,
            'is_premature': bool(is_premature),
            'has_compensatory_pause': bool(compensatory),
            'preceding_p_wave_found': bool(p_present),
            'qrs_onset_index': int(on_idx[beat_idx]) if beat_idx < len(on_idx) else None,
            'qrs_offset_index': int(off_idx[beat_idx]) if beat_idx < len(off_idx) else None,
            'scores': {
                'prematurity': prem_score,
                'qrs_width': width_score,
                'qrs_morphology_abnormality': morph_abn,
                'compensatory_pause': comp_score,
                'p_absence': p_absence_score,
                'overall_pvc': overall,
            },
            'weights': {
                'prematurity': w_prem,
                'qrs_width': w_width,
                'qrs_morphology_abnormality': w_morph_abn,
                'compensatory_pause': w_comp,
                'p_absence': w_p_abs,
            }
        }

        # Classification rules (enhanced)
        soft_nonprem = (is_wide and (morph_abn >= 0.90) and (p_absence_score >= 0.9) and not is_premature)
        width_min_for_morph = (not np.isnan(qrs_w)) and (qrs_w >= 0.12)
        # Only allow morphology path if also negative-dominant or P absent
        morph_gate = ((morph_abn >= 0.95) and (s_dominant or forced_wide_from_negative or (p_absence_score >= 0.9)))
        is_wide_or_morph = (
            is_wide_core or
            (is_wide_border and (morph_abn >= 0.90) and (s_dominant or forced_wide_from_negative or v1_support or (p_absence_score >= 0.9))) or
            (width_min_for_morph and (morph_gate or v1_support))
        )
        s_support = (forced_wide_from_negative or v1_support or (neg_dom_ratio >= 1.25)) and (is_wide_border or is_wide_core or morph_abn >= 0.95)
        very_wide = (not np.isnan(qrs_w)) and (qrs_w >= 0.16) and (morph_abn >= 0.95) and (p_absence_score >= 0.9)
        # Negative-dominant strong rule for giant negative waves (no prematurity required)
        negative_pvc_strong = (
            # II: strong negative dominance with measurable width
            (neg_dom_ratio >= 1.60 and ((neg_width_s or 0.0) >= 0.06) and (morph_abn >= 0.90)) or
            # II: early dominant negative even if width under-estimated
            (is_premature and (neg_dom_ratio >= 1.90) and (morph_abn >= 0.96)) or
            # V1: strong negative dominance with width
            (v1_support and (morph_abn >= 0.90)) or
            # V1: early dominant negative even if宽度偏小
            (is_premature and (v1_neg_dom_ratio >= 1.90) and (morph_abn >= 0.96))
        )
        classified = ((is_wide_or_morph and (is_premature or compensatory)) or very_wide or negative_pvc_strong or soft_nonprem)
        cand['classified_as_pvc'] = bool(classified)
        # Always store event index/time for consistent plotting
        cand['event_index'] = int(s_min_idx)
        cand['event_time_s'] = float(s_min_idx / fs)

        if classified:
            pvc_indices.append(beat_idx)
            compensatory_flags.append(True)
        else:
            cand['failure_reason'] = 'Overall confidence below rule threshold' if overall < 0.5 else 'Rule conditions not fully met'
        # suspicious flag for shading
        suspicious = (not cand['classified_as_pvc']) and (
            # only show when there is clear prematurity with supporting feature
            (is_premature and ((is_wide_border and morph_abn >= 0.90) or (morph_abn >= 0.95))) or
            # or a dominant negative complex (possible missed PVC)
            (forced_wide_from_negative and morph_abn >= 0.90)
        )
        cand['suspicious'] = bool(suspicious)
        cand['neg_valley_width_s'] = float(neg_width_s) if neg_width_s is not None else None
        cand['neg_dom_ratio'] = float(neg_dom_ratio)
        cand['v1_support'] = bool(v1_support)
        cand['v1_neg_valley_width_s'] = float(v1_neg_width_s) if v1_neg_width_s is not None else None
        cand['v1_neg_dom_ratio'] = float(v1_neg_dom_ratio)
        if v1_event_idx is not None:
            cand['v1_event_index'] = int(v1_event_idx)
            cand['v1_event_time_s'] = float(v1_event_idx / fs)
        candidates.append(cand)

    pvc_indices = np.asarray(pvc_indices, dtype=int)
    metrics = {
        'median_rr_s': median_rr,
        'baseline_rr_s': baseline_rr,
        'total_beats': int(r_idx.size),
        'total_pvc': int(pvc_indices.size),
        'compensatory_fraction': float(np.mean(compensatory_flags)) if compensatory_flags else None,
        'qrs_widths_s': qrs_widths.tolist(),
        'wide_qrs_fraction': float(np.mean(qrs_widths >= qrs_threshold_s)) if np.isfinite(qrs_widths).any() else None,
        'rr_mean_s': float(np.mean(rr)) if rr.size else None,
        'rr_cv': float(np.std(rr)/np.mean(rr)) if rr.size and np.mean(rr) > 0 else None,
        'candidates': candidates,
    }
    # include onset/offset in metrics for debugging
    metrics['qrs_onsets'] = on_idx.tolist()
    metrics['qrs_offsets'] = off_idx.tolist()
    return r_idx, pvc_indices, metrics, candidates, on_idx, off_idx


def _xcorr_max_norm(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Return max normalized cross-correlation between a and b within +/- max_lag samples.
    Inputs must be 1D arrays. Returns value in [0,1]."""
    if a.size < 3 or b.size < 3:
        return 0.0
    
    # Ensure both arrays have the same length
    min_len = min(a.size, b.size)
    a = a[:min_len]
    b = b[:min_len]
    
    # Normalize arrays
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    a = a / na
    b = b / nb
    
    # Test both polarities
    sims = []
    for bb in (b, -b):
        # Compute correlation for different lags
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                x = a[lag:]
                y = bb[:len(a) - lag]
            else:
                x = a[:len(a) + lag]
                y = bb[-lag:]
            
            if len(x) < 3 or len(y) < 3:
                continue
                
            # Use minimum length to avoid size mismatch
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            
            # Calculate normalized correlation
            corr = np.dot(x, y) / len(x)
            sims.append(float(corr))
    
    if not sims:
        return 0.0
    return float(np.clip(max(sims), 0.0, 1.0))


def generate_pvc_visuals(ecg_signal: np.ndarray, sampling_rate: float, ecg_title: str, lead_name: str, output_dir: str, aux_v1: np.ndarray | None = None, meta: dict | None = None):
    if ecg_signal is None or sampling_rate is None:
        return None

    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal

    r_idx, pvc_indices, det_metrics, candidates, on_idx, off_idx = _detect_pvc(ecg_cleaned, sampling_rate, aux_v1=aux_v1)

    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)

    # Draw II and optional V1 stacked for richer context
    ax_v1 = None
    if aux_v1 is not None and isinstance(aux_v1, np.ndarray):
        fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
        ax = axes[0]
        plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - PVC Overview (Lead II)", ax, sampling_rate)
        if r_idx.size:
            ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=18, label='R-peaks', zorder=5)
        ymin, ymax = ax.get_ylim()
        # ensure time ticks show on top subplot as well
        ax.tick_params(axis='x', which='both', labelbottom=True)
        ax.set_xlabel('Time (s)')
        # V1 panel
        ax_v1 = axes[1]
        v1_draw = aux_v1[:len(ecg_cleaned)] if aux_v1.size >= len(ecg_cleaned) else aux_v1
        plot_ecg_with_grid(t, v1_draw, "Aux Lead V1", ax_v1, sampling_rate)
    else:
        fig, ax = plt.subplots(figsize=(15, 4))
        plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - PVC Overview", ax, sampling_rate)
        if r_idx.size:
            ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=18, label='R-peaks', zorder=5)
        ymin, ymax = ax.get_ylim()
    # Intelligent annotations: highlight wide/premature candidates
    suspicious_times = []
    pvc_present = False
    show_suspicious = os.getenv('XAI_SHOW_SUSPICIOUS', '0') not in ('0', 'false', 'False', None)

    # First collect red PVC windows to prevent orange overlap
    red_windows = []
    show_v1_morph = os.getenv('XAI_SHOW_V1_MORPH', '1') not in ('0', 'false', 'False', None)
    for cand in candidates:
        bi = cand['beat_index']
        if bi - 1 < 0:
            continue
        if not cand.get('classified_as_pvc'):
            continue
        evt_idx = cand.get('event_index')
        if evt_idx is None:
            x0, x1 = t[r_idx[bi-1]], t[r_idx[bi]]
        else:
            evt_t = t[int(evt_idx)]
            halfw = 0.10
            x0, x1 = max(t[0], evt_t - halfw), min(t[-1], evt_t + halfw)
        red_windows.append((x0, x1))
        ax.axvspan(x0, x1, color='crimson', alpha=0.22)
        pvc_present = True
        # mirror highlight on V1 as well for confirmed PVCs unless morphology overlay is enabled
        if ax_v1 is not None and not show_v1_morph:
            ax_v1.axvspan(x0, x1, color='crimson', alpha=0.12)

    # V1 morphology overlay with color coding and text labels
    if ax_v1 is not None and show_v1_morph:
        v1_handles, v1_labels = ax_v1.get_legend_handles_labels()
        for cand in candidates:
            if not cand.get('classified_as_pvc'):
                continue
            bi = cand['beat_index']
            if bi - 1 < 0:
                continue
            evt_idx = cand.get('event_index')
            if evt_idx is None:
                evt_t = t[r_idx[bi]]
                halfw = 0.10
                x0, x1 = max(t[0], evt_t - halfw), min(t[-1], evt_t + halfw)
            else:
                evt_t = t[int(evt_idx)]
                halfw = 0.10
                x0, x1 = max(t[0], evt_t - halfw), min(t[-1], evt_t + halfw)

            # Determine V1 morphology from dominance between positive and negative peaks
            ibs = r_idx[bi]
            ibe = r_idx[bi+1] if (bi + 1) < len(r_idx) else min(len(ecg_cleaned)-1, int(evt_idx) + int(0.2*sampling_rate))
            seg_v1 = None
            if aux_v1 is not None and isinstance(aux_v1, np.ndarray):
                v1_draw = aux_v1[:len(ecg_cleaned)] if aux_v1.size >= len(ecg_cleaned) else aux_v1
                seg_v1 = v1_draw[ibs:ibe]
            label = None
            color = None
            if seg_v1 is not None and seg_v1.size > 3:
                vmax = float(np.max(seg_v1))
                vmin = float(np.min(seg_v1))
                pos_dom = abs(vmax) / max(1e-6, abs(vmin))
                neg_dom = abs(vmin) / max(1e-6, abs(vmax))
                if pos_dom >= 1.15:
                    # RBBB-like in V1 → LV origin
                    label = 'RBBB pattern (LV origin)'
                    color = 'red'
                elif neg_dom >= 1.15:
                    # LBBB-like in V1 → RV origin
                    label = 'LBBB pattern (RV origin)'
                    color = 'purple'
            # Fallback neutral
            if label is None:
                label = 'Indeterminate morphology'
                color = 'gray'
            ax_v1.axvspan(x0, x1, color=color, alpha=0.18)
        # Legend for V1 morphology
        if 'RBBB (LV)' not in v1_labels:
            v1_handles.append(Patch(facecolor='red', edgecolor='red', alpha=0.18, label='RBBB pattern (LV origin)'))
        if 'LBBB (RV)' not in v1_labels:
            v1_handles.append(Patch(facecolor='purple', edgecolor='purple', alpha=0.18, label='LBBB pattern (RV origin)'))
        ax_v1.legend(handles=v1_handles, loc='upper right')

    # Then draw suspicious windows, skipping any that overlap with red
    if show_suspicious:
        for cand in candidates:
            if not cand.get('suspicious'):
                continue
            bi = cand['beat_index']
            if bi - 1 < 0:
                continue
            evt_idx = cand.get('event_index')
            if evt_idx is None:
                # use a narrow window around R when no event index
                evt_t = t[r_idx[bi]]
                halfw = 0.06
                x0, x1 = max(t[0], evt_t - halfw), min(t[-1], evt_t + halfw)
            else:
                evt_t = t[int(evt_idx)]
                halfw = 0.06
                x0, x1 = max(t[0], evt_t - halfw), min(t[-1], evt_t + halfw)

            # overlap check
            overlaps = False
            for rx0, rx1 in red_windows:
                if (x0 < rx1) and (x1 > rx0):
                    overlaps = True
                    break
            if not overlaps:
                ax.axvspan(x0, x1, color='orange', alpha=0.18)
                suspicious_times.append(cand.get('time_s'))
    # Build unified legend entries (avoid overlap with ad-hoc text)
    handles, labels = ax.get_legend_handles_labels()
    if show_suspicious and suspicious_times:
        suspicious_patch = Patch(facecolor='orange', alpha=0.18, edgecolor='orange', label='Suspicious (not classified)')
        handles.append(suspicious_patch)
    if pvc_present:
        pvc_patch = Patch(facecolor='crimson', alpha=0.22, edgecolor='crimson', label='PVC')
        handles.append(pvc_patch)
    ax.legend(handles=handles, loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'pvc_overview.png')); plt.close(fig)

    # RR tachogram + histogram with PVC-coupled intervals excluded for clearer baseline view
    if r_idx.size > 1:
        rr_all = np.diff(r_idx) / sampling_rate
        beats = np.arange(rr_all.size)

        mask = np.ones_like(rr_all, dtype=bool)
        if pvc_indices.size:
            for bi in pvc_indices:
                pre_i = bi - 1
                post_i = bi
                if 0 <= pre_i < rr_all.size:
                    mask[pre_i] = False
                if 0 <= post_i < rr_all.size:
                    mask[post_i] = False

        rr_baseline = rr_all[mask]

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.6))
        ax1.plot(beats, rr_all, marker='o', lw=1)
        excl_beats = beats[~mask]
        if excl_beats.size:
            ax1.scatter(excl_beats, rr_all[~mask], c='tomato', s=16)
        ax1.set_title('RR tachogram')
        ax1.set_xlabel('Beat')
        ax1.set_ylabel('RR (s)')
        ax1.grid(alpha=0.3)

        rr_excluded = rr_all[~mask]
        bins = np.arange(max(0.3, rr_all.min() - 0.05), rr_all.max() + 0.1, 0.05) if rr_all.size else 10

        # Baseline (PVC-excluded) distribution
        if rr_baseline.size:
            ax2.hist(rr_baseline, bins=bins, color='#4C78A8', edgecolor='white', label='Baseline RR (PVC-excluded)')
            median_base = float(np.median(rr_baseline))
            ax2.axvline(median_base, color='crimson', linestyle='--', linewidth=1.4, alpha=0.85, label='Median baseline RR')
        # PVC-coupled distribution for reference
        if rr_excluded.size:
            ax2.hist(rr_excluded, bins=bins, color='tomato', edgecolor='white', alpha=0.55, label='PVC-coupled RR')

        # Fallback if everything empty
        if not rr_baseline.size and not rr_excluded.size and rr_all.size:
            ax2.hist(rr_all, bins=bins, color='#4C78A8', edgecolor='white', label='RR')
            median_base = float(np.median(rr_all))
            ax2.axvline(median_base, color='crimson', linestyle='--', linewidth=1.4, alpha=0.85, label='Median RR')

        ax2.set_title('RR histogram')
        ax2.set_xlabel('RR (s)')
        ax2.set_ylabel('Count')
        ax2.grid(alpha=0.2)
        # Show legend if any labeled element exists
        if len(ax2.get_legend_handles_labels()[0]):
            ax2.legend(loc='upper right')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'pvc_rr.png')); plt.close(fig2)

    # # QRS debug plot
    # figd, axd = plt.subplots(figsize=(15, 3))
    # axd.plot(t, ecg_cleaned, 'k-')
    # for j in range(len(r_idx)):
    #     if j < len(on_idx) and on_idx[j] >= 0:
    #         axd.axvline(t[on_idx[j]], color='green', alpha=0.4)
    #     if j < len(off_idx) and off_idx[j] >= 0:
    #         axd.axvline(t[off_idx[j]], color='red', alpha=0.4)
    # axd.set_title('QRS onset (green) / offset (red) debug')
    # axd.grid(True, alpha=0.3)
    # plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'pvc_qrs_debug.png')); plt.close(figd)

    # Concise report text only
    median_rr = det_metrics.get('median_rr_s')
    bpm = 60.0 / median_rr if median_rr else None
    baseline_txt = (
        f"median RR {median_rr:.2f}s ({bpm:.0f} bpm)" if bpm else
        (f"median RR {median_rr:.2f}s" if median_rr else "median RR n/a")
    )
    pvc_cnt = det_metrics.get('total_pvc', 0)
    if pvc_cnt > 0:
        keep = [c for c in candidates if c.get('classified_as_pvc')]
        times = ', '.join(f"{c['time_s']:.2f}s" for c in keep) if keep else ''
        # Optional morphology note from V1 for the first PVC
        morph_note = ''
        if show_v1_morph and aux_v1 is not None and keep:
            first = keep[0]
            bi = first['beat_index']
            ibs = r_idx[bi]
            ibe = r_idx[bi+1] if (bi + 1) < len(r_idx) else min(len(ecg_cleaned)-1, int(first.get('event_index', r_idx[bi]) + int(0.2*sampling_rate)))
            v1_draw = aux_v1[:len(ecg_cleaned)] if aux_v1.size >= len(ecg_cleaned) else aux_v1
            seg_v1 = v1_draw[ibs:ibe]
            if seg_v1.size > 3:
                vmax = float(np.max(seg_v1)); vmin = float(np.min(seg_v1))
                pos_dom = abs(vmax) / max(1e-6, abs(vmin))
                neg_dom = abs(vmin) / max(1e-6, abs(vmax))
                if pos_dom >= 1.15:
                    morph_note = ' V1 morphology: RBBB pattern (LV origin).'
                elif neg_dom >= 1.15:
                    morph_note = ' V1 morphology: LBBB pattern (RV origin).'
        report_text = f"Report: Sinus baseline with {baseline_txt}. PVC count: {pvc_cnt}. {('At ' + times) if times else ''}{morph_note}"
    else:
        report_text = f"Report: Sinus baseline with {baseline_txt}. PVC not detected."

    metrics = {
        **det_metrics,
        'r_count': int(r_idx.size),
        'pvc_indices': r_idx[pvc_indices].tolist() if pvc_indices.size else [],
    }

    # Concise summary/statistics for default output
    confirmed = [c for c in candidates if c.get('classified_as_pvc')]
    abnormality_times = [round(c.get('time_s', 0.0), 2) for c in confirmed]
    abnormality_indices = [int(c.get('beat_index')) for c in confirmed]
    heart_rate_bpm = 60.0 / det_metrics['median_rr_s'] if det_metrics.get('median_rr_s') else None

    summary = {
        'report_text': report_text,
        'is_abnormality_present': bool(det_metrics.get('total_pvc', 0) > 0),
        'abnormality_count': int(det_metrics.get('total_pvc', 0)),
        'abnormality_indices': abnormality_indices,
        'abnormality_times_s': abnormality_times,
    }
    statistics = {
        'total_beats': int(r_idx.size),
        'median_rr_s': det_metrics.get('median_rr_s'),
        'baseline_rr_s': det_metrics.get('baseline_rr_s'),
        'heart_rate_bpm': round(heart_rate_bpm, 1) if heart_rate_bpm else None,
        'rr_cv': det_metrics.get('rr_cv'),
        'wide_qrs_fraction': det_metrics.get('wide_qrs_fraction'),
    }

    concise_metrics = {
        'total_pvc': det_metrics.get('total_pvc', 0),
        'total_beats': int(r_idx.size),
        'median_rr_s': det_metrics.get('median_rr_s'),
        'baseline_rr_s': det_metrics.get('baseline_rr_s'),
        'rr_cv': det_metrics.get('rr_cv'),
        'wide_qrs_fraction': det_metrics.get('wide_qrs_fraction'),
    }

    # Recompute RR for image path conditional
    rr = np.diff(r_idx) / sampling_rate if r_idx.size > 1 else np.array([])
    out = {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'summary': summary,
        'statistics': statistics,
        'image_paths': ['pvc_overview.png', 'pvc_rr.png'] if rr.size else ['pvc_overview.png'],
    }

    # Optional debug block via env
    dbg = os.getenv('XAI_DEBUG_VERBOSE', '0')
    if dbg not in ('0', 'false', 'False', None):
        out['debug_details'] = {
            'candidates_analysis': candidates,
            'raw_beat_metrics': {
                'qrs_widths_s': det_metrics.get('qrs_widths_s'),
                'qrs_onsets': det_metrics.get('qrs_onsets'),
                'qrs_offsets': det_metrics.get('qrs_offsets'),
                'r_idx': r_idx.tolist(),
            },
            'debug_image_paths': {
                # pvc_qrs_debug.png currently disabled in concise output
            },
            'available_leads': (meta or {}).get('available_leads') if isinstance(meta, dict) else None,
        }

    return out




