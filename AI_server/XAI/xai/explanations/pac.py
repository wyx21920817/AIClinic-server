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


def _detect_pac(ecg_cleaned: np.ndarray, fs: float):
    # R-peaks
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="neurokit")
        r_idx = np.asarray(rpeaks['ECG_R_Peaks'], dtype=int)
    except Exception:
        r_idx = np.array([], dtype=int)

    if r_idx.size < 3:
        return r_idx, np.array([], dtype=int), {}, []

    rr = np.diff(r_idx) / fs
    median_rr = float(np.median(rr)) if rr.size else None

    # P-wave enhancement: asymmetric blanking to suppress T-wave (post-R only) + 3-12 Hz bandpass
    blanked = ecg_cleaned.copy()
    t_blank = int(0.24 * fs)  # suppress T-wave tails up to ~240ms after R
    for r in r_idx:
        s = r
        e = min(len(blanked), r + t_blank)
        blanked[s:e] = 0.0
    atrial = _bandpass(blanked, fs, 3.0, 12.0)

    # Precompute delineation-based P peaks as auxiliary evidence
    p_peaks_global = np.array([], dtype=int)
    try:
        _, waves = nk.ecg_delineate(ecg_cleaned, sampling_rate=fs, method="cwt", show=False)
        p_peaks_global = np.asarray(waves.get('ECG_P_Peaks', []), dtype=int)
    except Exception:
        p_peaks_global = np.array([], dtype=int)

    # QRS widths for morphology (PAC typically narrow QRS)
    qrs_widths = _estimate_qrs_widths(ecg_cleaned, r_idx, fs)

    # Build baseline QRS template using early non-premature beats
    win = int(0.12 * fs)
    templates = []
    for k in range(1, len(r_idx) - 1):
        # choose beats that are close to median RR on both sides
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
    qrs_template = np.mean(templates, axis=0) if templates else None

    # Detect premature beats: RR(i) markedly shorter than median
    pac_indices = []
    noncomp_flags = []
    p_visibility = []
    candidates = []
    # weights
    w_prem, w_p, w_morph = 0.4, 0.4, 0.2
    for i in range(len(rr)):
        if median_rr and rr[i] < 0.8 * median_rr:  # premature
            beat_idx = i + 1  # the premature beat R index in r_idx
            # Non-compensatory pause heuristic: rr[i] + rr[i+1] < 2 * median_rr (if i+1 exists)
            noncomp = False
            if i + 1 < len(rr) and median_rr:
                noncomp = (rr[i] + rr[i+1]) < (1.9 * median_rr)

            # P-wave evidence before premature R within 120-220 ms window
            lookback = int(0.26 * fs)
            p_start = max(0, r_idx[beat_idx] - lookback)
            p_end = r_idx[beat_idx]
            seg = atrial[p_start:p_end]
            p_present = False
            p_score = 0.0
            if seg.size > 0:
                peaks, props = find_peaks(seg, distance=int(0.08 * fs), prominence=0.08 * np.std(seg))
                prom_score = 0.0
                if peaks.size > 0:
                    prom = props.get('prominences', np.array([0.0]))
                    base = np.std(seg) + 1e-8
                    prom_score = float(np.clip(np.max(prom) / (2.5 * base), 0.0, 1.0))
                # Global delineation support
                glob_hits = p_peaks_global[(p_peaks_global >= p_start) & (p_peaks_global <= p_end)]
                glob_score = 1.0 if glob_hits.size > 0 else 0.0
                # Absolute amplitude fallback (for buried-in-T cases after T suppression)
                abs_amp = float(np.max(np.abs(seg))) if seg.size else 0.0
                amp_score = float(np.clip(abs_amp / (3.0 * (np.std(atrial) + 1e-8)), 0.0, 1.0))
                # Combine conservatively
                p_score = 0.5 * prom_score + 0.3 * glob_score + 0.2 * amp_score
                p_present = p_score >= 0.3

            # Prematurity score based on RR shortening magnitude
            shorten = (median_rr - rr[i]) / median_rr if median_rr else 0.0
            prem_score = float(np.clip(shorten / 0.3, 0.0, 1.0))

            # QRS morphology: width and similarity to baseline template
            qrs_w = qrs_widths[beat_idx] if beat_idx < qrs_widths.size else np.nan
            if np.isnan(qrs_w):
                width_score = 0.5
            else:
                if qrs_w <= 0.08:
                    width_score = 1.0
                elif qrs_w >= 0.14:
                    width_score = 0.0
                else:
                    width_score = float(np.clip(1.0 - (qrs_w - 0.08) / (0.14 - 0.08), 0.0, 1.0))
            # similarity
            sim_score = 0.5
            if qrs_template is not None and 0 < r_idx[beat_idx] - win and r_idx[beat_idx] + win <= len(ecg_cleaned):
                seg = ecg_cleaned[r_idx[beat_idx]-win:r_idx[beat_idx]+win]
                v = seg - np.mean(seg)
                n = np.linalg.norm(v) + 1e-8
                v = v / n
                sim = float(np.clip(np.dot(v, qrs_template), 0.0, 1.0))
                sim_score = sim
            morph_score = 0.5 * width_score + 0.5 * sim_score

            # overall score (with hard veto on P-wave evidence)
            overall = w_prem * prem_score + w_p * p_score + w_morph * morph_score
            hard_veto = p_score < 0.2
            if hard_veto:
                overall = 0.0

            cand_time = r_idx[beat_idx] / fs
            cand = {
                'beat_index': int(beat_idx),
                'time_s': float(cand_time),
                'rr_short_s': float(rr[i]),
                'rr_short_pct': float(shorten * 100.0),
                'noncompensatory': bool(noncomp),
                'p_present': bool(p_present),
                'scores': {
                    'prematurity': prem_score,
                    'p_wave': p_score,
                    'qrs_width': width_score,
                    'qrs_similarity': sim_score,
                    'qrs_morphology': morph_score,
                    'overall_pac': overall,
                },
                'weights': {'prematurity': w_prem, 'p_wave': w_p, 'qrs_morphology': w_morph},
                'hard_veto': hard_veto,
            }

            # Soft accept: borderline P (>=0.2) with strong prematurity, morphology, and non-compensatory pause
            soft_accept = (p_score >= 0.2 and prem_score >= 0.9 and morph_score >= 0.9 and noncomp)

            if p_present or soft_accept:
                pac_indices.append(beat_idx)
                noncomp_flags.append(noncomp)
                p_visibility.append(True)
                cand['classified_as_pac'] = True
                if soft_accept and not p_present:
                    cand['soft_accept_reason'] = 'borderline P-wave with strong prematurity, morphology, and non-compensatory pause'
            else:
                cand['classified_as_pac'] = False
                cand['failure_reason'] = 'ectopic P-wave not identified with sufficient confidence'
            candidates.append(cand)

    pac_indices = np.asarray(pac_indices, dtype=int)
    metrics = {
        'median_rr_s': median_rr,
        'total_beats': int(r_idx.size),
        'total_pac': int(pac_indices.size),
        'noncompensatory_fraction': float(np.mean(noncomp_flags)) if noncomp_flags else None,
        'p_visibility_fraction': float(np.mean(p_visibility)) if p_visibility else None,
        'rr_mean_s': float(np.mean(rr)) if rr.size else None,
        'rr_cv': float(np.std(rr)/np.mean(rr)) if rr.size and np.mean(rr) > 0 else None,
        'candidates': candidates,
        'qrs_widths_s': qrs_widths.tolist(),
        'scoring_weights': {'prematurity': w_prem, 'p_wave': w_p, 'qrs_morphology': w_morph},
    }
    return r_idx, pac_indices, metrics, candidates


def _estimate_qrs_widths(ecg_cleaned: np.ndarray, r_idx: np.ndarray, fs: float):
    qrs_widths = []
    for r in r_idx:
        w = int(0.2 * fs)
        s = max(0, r - w)
        e = min(len(ecg_cleaned), r + w)
        seg = ecg_cleaned[s:e]
        if seg.size < 3:
            qrs_widths.append(np.nan)
            continue
        vmax = np.max(np.abs(seg))
        if vmax <= 0:
            qrs_widths.append(np.nan)
            continue
        half = 0.5 * vmax
        above = np.where(np.abs(seg) >= half)[0]
        width = (above[-1] - above[0]) / fs if above.size >= 2 else np.nan
        qrs_widths.append(width)
    return np.asarray(qrs_widths)


def generate_pac_visuals(ecg_signal: np.ndarray, sampling_rate: float, ecg_title: str, lead_name: str, output_dir: str):
    if ecg_signal is None or sampling_rate is None:
        return None

    # Clean
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal

    r_idx, pac_indices, det_metrics, candidates = _detect_pac(ecg_cleaned, sampling_rate)

    # Time base
    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)

    # Overview plot with highlights
    fig, ax = plt.subplots(figsize=(15, 4))
    plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - PAC Overview", ax, sampling_rate)
    if r_idx.size:
        ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=18, label='R-peaks', zorder=5)
    ymin, ymax = ax.get_ylim()
    suspicious_times = []
    pac_present = False
    # highlight confirmed PACs and suspicious (not classified)
    for cand in candidates:
        bi = cand['beat_index']
        if bi - 1 >= 0:
            x0, x1 = t[r_idx[bi-1]], t[r_idx[bi]]
            if cand.get('classified_as_pac'):
                ax.axvspan(x0, x1, color='crimson', alpha=0.22)
                pac_present = True
            else:
                ax.axvspan(x0, x1, color='orange', alpha=0.18)
                suspicious_times.append(cand.get('time_s'))
    # build legend entries unified at top-right
    handles, labels = ax.get_legend_handles_labels()
    if pac_present:
        handles.append(Patch(facecolor='crimson', alpha=0.22, edgecolor='crimson', label='PAC'))
    if suspicious_times:
        handles.append(Patch(facecolor='orange', alpha=0.18, edgecolor='orange', label='Suspicious (not classified)'))
    ax.legend(handles=handles, loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'pac_overview.png')); plt.close(fig)

    # RR tachogram and histogram
    rr = np.diff(r_idx) / sampling_rate if r_idx.size > 1 else np.array([])
    if rr.size:
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 3.6))
        ax2[0].plot(rr, color='#1f77b4', marker='o', linestyle='-', linewidth=2, markersize=4)
        ax2[0].set_title('RR tachogram'); ax2[0].set_xlabel('Beat'); ax2[0].set_ylabel('RR (s)')
        ax2[0].grid(True, alpha=0.3)
        # Mark candidate premature RR with red circles and annotate value
        for cand in candidates:
            bi = cand['beat_index']
            j = bi - 1
            if 0 <= j < rr.size:
                ax2[0].plot(j, rr[j], 'o', ms=6, color='orangered')
                ax2[0].annotate(f"{rr[j]:.2f}s", (j, rr[j]), textcoords="offset points", xytext=(0, -12), ha='center', color='red', fontsize=8)
        ax2[1].hist(rr, bins=30, color='#1f77b4', alpha=0.6, edgecolor='#1f77b4')
        ax2[1].set_title('RR histogram'); ax2[1].set_xlabel('RR (s)'); ax2[1].set_ylabel('Count')
        ax2[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'pac_rr.png')); plt.close(fig2)

    # Dynamic report text
    median_rr = det_metrics.get('median_rr_s')
    bpm = 60.0 / median_rr if median_rr else None
    baseline_txt = (
        f"median RR {median_rr:.2f}s ({bpm:.0f} bpm)" if bpm else
        (f"median RR {median_rr:.2f}s" if median_rr else "median RR n/a")
    )
    pac_cnt = det_metrics.get('total_pac', 0)
    if candidates:
        # choose most premature candidate for narrative
        top = sorted(candidates, key=lambda c: c.get('scores', {}).get('prematurity', 0), reverse=True)[0]
        if top.get('classified_as_pac'):
            report_text = (
                f"Report: Sinus baseline with {baseline_txt}. A premature beat at {top['time_s']:.2f}s "
                f"(RR {top['rr_short_s']:.2f}s, -{top['rr_short_pct']:.0f}%) was classified as PAC. "
                f"Final PAC count: {pac_cnt}."
            )
        else:
            report_text = (
                f"Report: Sinus baseline with {baseline_txt}. A significant premature beat at {top['time_s']:.2f}s "
                f"(RR {top['rr_short_s']:.2f}s, -{top['rr_short_pct']:.0f}%) was not classified as PAC due to low "
                f"ectopic P-wave confidence. Final PAC count: {pac_cnt}."
            )
    else:
        report_text = (
            f"Report: Sinus baseline with {baseline_txt}. No premature beats detected consistent with PAC criteria."
        )

    metrics = {
        **det_metrics,
        'r_count': int(r_idx.size),
        'pac_indices': r_idx[pac_indices].tolist() if pac_indices.size else [],
    }

    # Build concise summary and statistics
    confirmed = [c for c in candidates if c.get('classified_as_pac')]
    abnormality_times = [round(c.get('time_s', 0.0), 2) for c in confirmed]
    abnormality_indices = [int(c.get('beat_index')) for c in confirmed]
    heart_rate_bpm = 60.0 / det_metrics['median_rr_s'] if det_metrics.get('median_rr_s') else None

    summary = {
        'report_text': report_text,
        'is_abnormality_present': bool(det_metrics.get('total_pac', 0) > 0),
        'abnormality_count': int(det_metrics.get('total_pac', 0)),
        'abnormality_indices': abnormality_indices,
        'abnormality_times_s': abnormality_times,
    }
    statistics = {
        'total_beats': int(r_idx.size),
        'median_rr_s': det_metrics.get('median_rr_s'),
        'heart_rate_bpm': round(heart_rate_bpm, 1) if heart_rate_bpm else None,
        'rr_cv': det_metrics.get('rr_cv'),
    }

    # Default: concise metrics (prune heavy internals)
    concise_metrics = {
        'total_pac': det_metrics.get('total_pac', 0),
        'total_beats': int(r_idx.size),
        'median_rr_s': det_metrics.get('median_rr_s'),
        'rr_cv': det_metrics.get('rr_cv'),
    }

    out = {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'summary': summary,
        'statistics': statistics,
        'image_paths': ['pac_overview.png', 'pac_rr.png'] if rr.size else ['pac_overview.png']
    }

    # Optional: attach debug details when enabled via env var
    dbg = os.getenv('XAI_DEBUG_VERBOSE', '0')
    if dbg not in ('0', 'false', 'False', None):
        out['debug_details'] = {
            'candidates_analysis': candidates,
            'raw_beat_metrics': {
                # PAC doesn t use onsets/offsets; keep candidates and r indices
                'r_indices': r_idx.tolist(),
            }
        }

    return out




