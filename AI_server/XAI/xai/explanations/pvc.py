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
        # use only narrow likely-sinus beats for template
        if k < qrs_widths.size and (not np.isnan(qrs_widths[k])) and qrs_widths[k] < 0.11:
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
    """Robust QRS onset/offset using derivative envelope and persistence criteria.
    - Build |d/dt| envelope, smooth with moving average
    - Onset: last time before R where envelope < thr_low for >=20 ms
    - Offset: first time after R where envelope < thr_low and stays low for >=30 ms
    - Offset window extended to 300 ms to capture broad PVC tails
    Returns onset_idx, offset_idx, width_s
    """
    n = len(ecg)
    onset_idx = np.full(len(r_idx), -1, dtype=int)
    offset_idx = np.full(len(r_idx), -1, dtype=int)
    width_s = np.full(len(r_idx), np.nan, dtype=float)

    # derivative envelope
    diff = np.abs(np.diff(ecg, prepend=ecg[0]))
    # smooth
    k = max(3, int(0.01 * fs))
    kernel = np.ones(k) / k
    env = np.convolve(diff, kernel, mode='same')

    left_win = int(0.18 * fs)
    right_win = int(0.30 * fs)
    min_on_persist = int(0.02 * fs)
    min_off_persist = int(0.03 * fs)

    for i, r in enumerate(r_idx):
        s = max(0, r - left_win)
        e = min(n, r + right_win)
        seg_env = env[s:e]
        if seg_env.size < 5:
            continue
        # adaptive thresholds from local env percentiles
        p50 = np.percentile(seg_env, 50)
        p70 = np.percentile(seg_env, 70)
        thr_low = max(1e-8, 0.5 * p50)
        thr_high = max(thr_low * 1.5, p70)

        # Onset: scan backward from R to s
        li = r
        low_count = 0
        for kidx in range(r, s, -1):
            if env[kidx] < thr_low:
                low_count += 1
                if low_count >= min_on_persist:
                    li = kidx
                    break
            else:
                low_count = 0
        # Offset: scan forward from R to e requiring sustained low after a decay region
        ri = r
        low_count = 0
        seen_peak = False
        peak_env = env[r]
        for kidx in range(r, e):
            if env[kidx] > peak_env:
                peak_env = env[kidx]
            if not seen_peak and env[kidx] < thr_high:
                seen_peak = True
            if seen_peak and env[kidx] < thr_low:
                low_count += 1
                if low_count >= min_off_persist:
                    ri = kidx
                    break
            else:
                low_count = 0
        onset_idx[i] = li
        offset_idx[i] = ri
        if ri > li:
            width_s[i] = (ri - li) / fs
    return onset_idx, offset_idx, width_s


def _detect_pvc(ecg_cleaned: np.ndarray, fs: float):
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
            is_premature = (ratio_to_baseline < 0.90) or (ratio_to_prev < 0.85)
        is_wide = bool((beat_idx < qrs_widths.size) and (not np.isnan(qrs_widths[beat_idx])) and (qrs_widths[beat_idx] >= qrs_threshold_s))
        compensatory = False
        if (i_prev < len(rr)) and (i_next < len(rr)) and median_rr:
            compensatory = (rr[i_prev] + rr[i_next]) >= (1.9 * median_rr)
        # Prefilter: only consider suspicious beats
        consider = is_wide or is_premature
        if not consider:
            continue

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
        # compensatory score
        comp_score = 1.0 if compensatory else 0.0
        # P absence score in 120–220 ms before R
        lookback = int(0.22 * fs)
        p_start = max(0, r_idx[beat_idx] - lookback)
        p_end = r_idx[beat_idx]
        segp = atrial[p_start:p_end]
        p_present = False
        p_absence_score = 1.0
        if segp.size > 0:
            peaks, props = find_peaks(segp, distance=int(0.08 * fs), prominence=0.08 * np.std(segp))
            if peaks.size > 0:
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

        # Classification: strong PVC when wide, premature, compensatory, and morphology abnormal
        # Rare non-premature PVC soft rule: wide and very abnormal morphology, P absent, and following RR close to median (interpolated coupling)
        soft_nonprem = (is_wide and (morph_abn >= 0.8) and (p_absence_score >= 0.9) and not is_premature)
        classified = (is_wide and is_premature and compensatory and (morph_abn >= 0.6)) or soft_nonprem
        cand['classified_as_pvc'] = bool(classified)
        if classified:
            pvc_indices.append(beat_idx)
            compensatory_flags.append(True)
        else:
            cand['failure_reason'] = 'Overall confidence below rule threshold' if overall < 0.5 else 'Rule conditions not fully met'
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
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    a = a / na
    b = b / nb
    # test both polarities
    sims = []
    for bb in (b, -b):
        # compute sliding correlation by shifting bb
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                x = a[lag:]
                y = bb[:a.size - lag]
            else:
                x = a[:a.size + lag]
                y = bb[-lag:]
            if x.size < 3:
                continue
            sims.append(float(np.dot(x, y) / max(1, x.size)))
    if not sims:
        return 0.0
    return float(np.clip(max(sims), 0.0, 1.0))


def generate_pvc_visuals(ecg_signal: np.ndarray, sampling_rate: float, ecg_title: str, lead_name: str, output_dir: str):
    if ecg_signal is None or sampling_rate is None:
        return None

    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal

    r_idx, pvc_indices, det_metrics, candidates, on_idx, off_idx = _detect_pvc(ecg_cleaned, sampling_rate)

    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)

    fig, ax = plt.subplots(figsize=(15, 4))
    plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - PVC Overview", ax, sampling_rate)
    if r_idx.size:
        ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=18, label='R-peaks', zorder=5)
    ymin, ymax = ax.get_ylim()
    # Intelligent annotations: highlight wide/premature candidates
    suspicious_times = []
    pvc_present = False
    for cand in candidates:
        bi = cand['beat_index']
        if bi - 1 >= 0:
            x0, x1 = t[r_idx[bi-1]], t[r_idx[bi]]
            if cand.get('classified_as_pvc'):
                ax.axvspan(x0, x1, color='crimson', alpha=0.22)
                pvc_present = True
            else:
                ax.axvspan(x0, x1, color='orange', alpha=0.18)
                suspicious_times.append(cand.get('time_s'))
    # Build unified legend entries (avoid overlap with ad-hoc text)
    handles, labels = ax.get_legend_handles_labels()
    if suspicious_times:
        suspicious_patch = Patch(facecolor='orange', alpha=0.18, edgecolor='orange', label='Suspicious (not classified)')
        handles.append(suspicious_patch)
    if pvc_present:
        pvc_patch = Patch(facecolor='crimson', alpha=0.22, edgecolor='crimson', label='PVC')
        handles.append(pvc_patch)
    ax.legend(handles=handles, loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'pvc_overview.png')); plt.close(fig)

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
        report_text = f"Report: Sinus baseline with {baseline_txt}. PVC count: {pvc_cnt}. {('At ' + times) if times else ''}"
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
        'image_paths': ['pvc_overview.png', 'pvc_rr.png'] if rr.size else ['pvc_overview.png']
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
            },
            'debug_image_paths': {
                # pvc_qrs_debug.png currently disabled in concise output
            }
        }

    return out




