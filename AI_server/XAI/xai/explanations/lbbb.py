# explanation_lbbb.py
import json
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks
from ..ecg_utils import plot_ecg_with_grid, load_wfdb_signals, load_dicom_signals
from ..clinical_reports import generate_report_bbb

def generate_lbbb_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    """Left Bundle Branch Block analysis with refined QRS estimation (notebook-aligned)."""
    if ecg_signal is None:
        return None

    # 1) Clean signal
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
        print(f"Signal for {ecg_title} cleaned using 'neurokit' method.")
    except Exception as e:
        print(f"Warning: Cleaning failed for {ecg_title}: {e}. Using raw signal.")
        ecg_cleaned = ecg_signal

    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)

    # 2) R-peaks
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_idx = rpeaks['ECG_R_Peaks']
        print(f"R-peaks detected. Found {len(r_idx)} R-peaks.")
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}")
        return None

    # 3) Delineation
    waves, used_method = None, None
    for method in ["dwt", "cwt", "peak"]:
        try:
            _, w = nk.ecg_delineate(ecg_cleaned, rpeaks=r_idx, sampling_rate=sampling_rate, method=method)
            if w and 'ECG_QRS_Onsets' in w and 'ECG_QRS_Offsets' in w \
               and np.sum(~np.isnan(w['ECG_QRS_Onsets'])) > 0 and np.sum(~np.isnan(w['ECG_QRS_Offsets'])) > 0:
                waves, used_method = w, method
                print(f"Delineation successful via '{method}'.")
                break
            else:
                print(f"Delineation with '{method}' did not yield valid QRS onsets/offsets.")
        except Exception as e:
            print(f"Delineation with method '{method}' failed: {e}")

    qrs_durations_sec, highlight_regions = [], []
    estimation_type = "Delineated"

    if waves is not None:
        q_on = [int(q) for q in waves['ECG_QRS_Onsets'] if not np.isnan(q)]
        q_off = [int(q) for q in waves['ECG_QRS_Offsets'] if not np.isnan(q)]
        used_offsets = 0
        for on in q_on:
            # find next offset greater than onset
            while used_offsets < len(q_off) and q_off[used_offsets] <= on:
                used_offsets += 1
            if used_offsets >= len(q_off):
                break
            off = q_off[used_offsets]
            dur = (off - on) / sampling_rate
            if 0.04 < dur < 0.30:
                qrs_durations_sec.append(dur)
                highlight_regions.append((on, off))
                used_offsets += 1

    # 4) Refined estimation fallback (notebook logic)
    if not qrs_durations_sec:
        estimation_type = "Refined Estimation"
        print("Attempting refined R-peak based QRS boundary estimation.")

        prominence_factor = 0.05
        min_peak_width = 1
        q_search_sec = 0.10
        s_search_sec = 0.15
        q_fallback_sec = 0.05
        s_fallback_sec = 0.09  # LBBB tends to have longer terminal S

        signal_ptp = float(np.max(ecg_cleaned) - np.min(ecg_cleaned))
        prominence_threshold = (prominence_factor * signal_ptp) if signal_ptp > 0 else 0.01

        for r in r_idx:
            # left search for Q-like min
            left_win = int(q_search_sec * sampling_rate)
            start_left = max(0, r - left_win)
            seg_left = ecg_cleaned[start_left:r]
            q_start = max(0, r - int(q_fallback_sec * sampling_rate))
            if seg_left.size > min_peak_width + 1:
                inv_left = -seg_left
                peaks_l, _ = find_peaks(inv_left, prominence=prominence_threshold, width=min_peak_width)
                if peaks_l.size > 0:
                    q_start = start_left + peaks_l[-1]

            # right search for S/J-like min
            right_win = int(s_search_sec * sampling_rate)
            r_plus_1 = min(r + 1, len(ecg_cleaned) - 1)
            end_right = min(len(ecg_cleaned), r_plus_1 + right_win)
            seg_right = ecg_cleaned[r_plus_1:end_right]
            q_end = min(len(ecg_cleaned) - 1, r + int(s_fallback_sec * sampling_rate))
            if seg_right.size > min_peak_width + 1:
                inv_right = -seg_right
                peaks_r, _ = find_peaks(inv_right, prominence=prominence_threshold, width=min_peak_width)
                if peaks_r.size > 0:
                    q_end = r_plus_1 + peaks_r[0]

            if q_start >= q_end:
                # symmetric fallback 150 ms for LBBB
                half = (int((150 / 1000.0) * sampling_rate)) // 2
                q_start = max(0, r - half)
                q_end = min(len(ecg_cleaned) - 1, r + half)

            dur_samples = q_end - q_start
            if dur_samples > 0:
                qrs_durations_sec.append(dur_samples / sampling_rate)
                highlight_regions.append((q_start, q_end))

        if qrs_durations_sec:
            print("Refined Estimation QRS durations calculated.")
        else:
            print("Could not perform Refined Estimation for QRS durations.")
            estimation_type = "Failed Estimation"

    # 5) Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    plot_ecg_with_grid(t, ecg_cleaned, f'{ecg_title} ({lead_name}) - QRS Duration Analysis ({estimation_type})', axes[0], sampling_rate)
    axes[0].scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', marker='o', s=30, label='R-peaks', zorder=5)

    annotated = 0
    for on, off in highlight_regions:
        dur = (off - on) / sampling_rate
        color = 'purple' if dur > 0.12 else ('green' if dur >= 0.06 else 'lightgray')
        axes[0].axvspan(t[on], t[off], alpha=0.3, color=color, ymin=0.3, ymax=0.7)
        if annotated < 3 and dur > 0.12:
            seg_start, seg_end = max(0, on), min(len(ecg_cleaned) - 1, off)
            y_text = np.max(ecg_cleaned[seg_start:seg_end+1])
            axes[0].text((t[on] + t[off]) / 2, y_text + 0.1 * (np.max(ecg_cleaned) - np.min(ecg_cleaned)), f'QRS: {dur:.3f}s', ha='center', fontsize=9, color='k', bbox=dict(fc='w', alpha=0.7, pad=1))
            annotated += 1
    # Legend: R-peaks + QRS classification
    import matplotlib.patches as mpatches
    handles, _ = axes[0].get_legend_handles_labels()
    handles.extend([
        mpatches.Patch(color='purple', alpha=0.3, label='Wide QRS (≥0.12s)'),
        mpatches.Patch(color='green', alpha=0.3, label='Normal QRS (0.06–0.11s)'),
        mpatches.Patch(color='lightgray', alpha=0.3, label='Other QRS'),
    ])
    axes[0].legend(handles=handles, loc='upper right', framealpha=0.9, facecolor='white')

    if qrs_durations_sec:
        qrs_arr = np.array(qrs_durations_sec)
        max_val = float(np.max(qrs_arr)); min_val = float(np.min(qrs_arr))
        hist_min = min(0.05, min_val - 0.01 if min_val > 0.05 else 0.04)
        hist_max = max(0.25, max_val + 0.02)
        bins = np.arange(hist_min, hist_max, 0.01)
        if bins.size == 0:
            bins = 'auto'
        axes[1].hist(qrs_arr, bins=bins, ec='k', alpha=0.7, color='lightcoral')
        mean_qrs = float(np.mean(qrs_arr))
        axes[1].axvline(mean_qrs, c='r', ls='--', lw=2, label=f'Mean QRS: {mean_qrs:.3f}s')
        axes[1].axvspan(0.06, 0.11, color='g', alpha=0.2, label='Normal QRS (0.06-0.11s)')
        axes[1].axvline(0.12, c='purple', ls=':', lw=2, label='0.12s Threshold (Wide)')
        axes[1].set_title('QRS Duration Distribution (Refined Estimation)' if estimation_type != 'Delineated' else 'QRS Duration Distribution (Delineated)')
        axes[1].set_xlabel('QRS Duration (s)'); axes[1].set_ylabel('Frequency'); axes[1].legend(); axes[1].grid(axis='y', alpha=0.75)
    else:
        axes[1].text(0.5, 0.5, 'No QRS durations calculated / to display.', ha='center')
        axes[1].set_title('QRS Duration Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lbbb_qrs_analysis.png'))
    plt.close(fig)

    # 6) Report
    report_text = generate_report_bbb(ecg_title, lead_name, estimation_type, qrs_durations_sec)
    print(report_text)
    metrics = {}
    if qrs_durations_sec:
        qrs_arr = np.array(qrs_durations_sec)
        metrics = {
            'qrs_mean_s': float(np.mean(qrs_arr)),
            'wide_qrs_count': int(np.sum(qrs_arr > 0.12)),
            'total_qrs_count': int(qrs_arr.size)
        }
    return {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'metrics': metrics,
        'image_paths': {'lbbb_qrs_analysis.png': os.path.join(output_dir, 'lbbb_qrs_analysis.png')}
    }


# ------------------------
# Multi-lead LBBB analysis
# ------------------------

def _refined_qrs_regions(signal, fs, r_idx):
    """Same refined estimation approach, tuned for LBBB (slightly longer terminal S)."""
    q_search_sec = 0.10
    s_search_sec = 0.15
    q_fallback_sec = 0.05
    s_fallback_sec = 0.09

    ptp = float(np.max(signal) - np.min(signal))
    prom = (0.05 * ptp) if ptp > 0 else 0.01

    regions = []
    for r in r_idx:
        left_win = int(q_search_sec * fs)
        start_l = max(0, r - left_win)
        seg_l = signal[start_l:r]
        q_start = max(0, r - int(q_fallback_sec * fs))
        if seg_l.size > 2:
            inv_l = -seg_l
            pks_l, _ = find_peaks(inv_l, prominence=prom, width=1)
            if pks_l.size > 0:
                q_start = start_l + pks_l[-1]

        right_win = int(s_search_sec * fs)
        r1 = min(r + 1, len(signal) - 1)
        end_r = min(len(signal), r1 + right_win)
        seg_r = signal[r1:end_r]
        q_end = min(len(signal) - 1, r + int(s_fallback_sec * fs))
        if seg_r.size > 2:
            inv_r = -seg_r
            pks_r, _ = find_peaks(inv_r, prominence=prom, width=1)
            if pks_r.size > 0:
                q_end = r1 + pks_r[0]

        if q_start >= q_end:
            half = (int((150 / 1000.0) * fs)) // 2
            q_start = max(0, r - half)
            q_end = min(len(signal) - 1, r + half)
        regions.append((q_start, q_end))
    return regions


def _detect_deep_s(seg):
    valleys, _ = find_peaks(-seg)
    if len(valleys):
        s_amp = -seg[valleys[0]]
        if s_amp > 0.05:
            return True, valleys[0]
    return False, None


def _detect_broad_or_notched_r(seg, fs):
    """Simple heuristic for broad/possibly notched R in lateral leads.
    - Use find_peaks with width to estimate peak width within 120ms window.
    - Consider broad if any peak width > 40 ms or there are two nearby peaks.
    """
    pks, props = find_peaks(seg, distance=max(1, int(0.02 * fs)), width=1)
    if len(pks) == 0:
        return False, None
    # Broad if any width > 0.04 s
    widths = props.get('widths', np.array([]))
    if widths.size and np.any(widths / fs > 0.04):
        return True, int(pks[np.argmax(widths)])
    # Or double-peaked within 80 ms
    for i in range(len(pks) - 1):
        if (pks[i+1] - pks[i]) / fs < 0.08 and seg[pks[i+1]] > 0.2 * seg[pks[i]]:
            return True, int(pks[i])
    return False, int(pks[0])


def _categorize_qrs(duration_s: float) -> str:
    if duration_s is None or duration_s < 0.06:
        return "narrow"
    if duration_s < 0.11:
        return "normal"
    if duration_s < 0.12:
        return "borderline"
    return "wide"


def generate_lbbb_multilead(record_name=None, dicom_filepath=None, db_dir=None, output_dir="."):
    """Multi-lead LBBB screening and visualization (WFDB or DICOM).
    Leads: prioritize V1, V6, I, aVL. Substitute V5 for missing V6, and I for missing aVL.
    """
    need = ["V1", "V2", "V5", "V6", "I", "aVL", "II", "aVF"]
    raw = {}
    try:
        if dicom_filepath or (record_name and str(record_name).lower().endswith((".dcm", ".dicom"))):
            raw = load_dicom_signals(dicom_filepath or record_name)
        else:
            raw = load_wfdb_signals(record_name, leads=need, db_dir=db_dir, allow_missing=True)
    except Exception as e:
        print(f"Failed to load multi-lead ECG for LBBB: {e}")
        raw = {}
    if not raw:
        print("Insufficient leads for LBBB multi-lead assessment.")
        return None

    # Build exactly-up-to-4 unique leads: V1, (V6 or V5), I, aVL (if present);
    # if still <4, fill with V2, then II/aVF as final fallbacks.
    from collections import OrderedDict
    def find_token_key(keys, token):
        tok = token.upper()
        for k in keys:
            tokens = re.findall(r"[A-Za-z0-9]+", k.upper())
            if tok in tokens:
                return k
        return None

    leads = OrderedDict()
    # V1 primary
    k_v1 = find_token_key(raw.keys(), "V1")
    if k_v1:
        leads[k_v1] = raw[k_v1]
    # Lateral chest V6 or substitute V5
    k_v6 = find_token_key(raw.keys(), "V6")
    if k_v6:
        leads[k_v6] = raw[k_v6]
    else:
        k_v5 = find_token_key(raw.keys(), "V5")
        if k_v5:
            leads[f"V6 (subst. {k_v5})"] = raw[k_v5]
    # Limb: I and aVL
    k_I = find_token_key(raw.keys(), "I")
    if k_I:
        leads[k_I] = raw[k_I]
    k_avl = find_token_key(raw.keys(), "aVL")
    if k_avl:
        leads[k_avl] = raw[k_avl]
    # Fill to 4
    if len(leads) < 4:
        k_v2 = find_token_key(raw.keys(), "V2")
        if k_v2 and k_v2 not in leads:
            leads[k_v2] = raw[k_v2]
    for cand in ("II", "aVF"):
        if len(leads) >= 4:
            break
        k_c = find_token_key(raw.keys(), cand)
        if k_c and k_c not in leads:
            leads[k_c] = raw[k_c]

    if not leads:
        print("Insufficient leads for LBBB multi-lead assessment.")
        return None

    qrs_pool = []
    deep_s_count = 0  # expected in V1
    broad_r_count = 0  # expected in lateral leads
    rs_counts = {"wide": 0, "borderline": 0, "normal": 0, "narrow": 0}

    fig, axes = plt.subplots(len(leads), 1, figsize=(15, 3.8 * len(leads)), sharex=True)
    axes = np.array(axes).ravel()

    perbeat_rows = []
    for ax, (lead, (sig, fs)) in zip(axes, leads.items()):
        t = np.linspace(0, len(sig) / fs, len(sig), endpoint=False)
        _, rp = nk.ecg_peaks(sig, sampling_rate=fs, method="neurokit")
        r_idx = rp["ECG_R_Peaks"]

        regions = _refined_qrs_regions(sig, fs, r_idx)
        durations = [(end - start) / fs for start, end in regions]
        qrs_pool.extend(durations)
        for i, (start, end) in enumerate(regions):
            perbeat_rows.append([lead, i, start, end, (end-start)/fs, t[start], t[end], fs])

        title = f"{lead} – LBBB features ({record_name})"
        plot_ecg_with_grid(t, sig, title, ax, fs, show_grid=True, style="color")
        if len(r_idx):
            ax.scatter(t[r_idx], sig[r_idx], c="dodgerblue", s=30, zorder=5, label='R-peaks')

        # Highlight QRS regions
        for (start, end), dur in zip(regions, durations):
            cat = _categorize_qrs(dur)
            color = 'purple' if cat == 'wide' else ('orange' if cat == 'borderline' else ('green' if cat == 'normal' else 'lightgray'))
            rs_counts[cat] += 1
            ax.axvspan(t[start], t[end], alpha=0.28, color=color, ymin=0.32, ymax=0.72)

        # Morphology
        if ("V1" in lead):
            for r in r_idx:
                seg = sig[r:r + int(0.14 * fs)]
                ok, s_idx = _detect_deep_s(seg)
                if ok:
                    deep_s_count += 1
                    if s_idx is not None:
                        x = t[min(len(t) - 1, r + s_idx)]
                        y = sig[min(len(sig) - 1, r + s_idx)]
                        ax.annotate("S", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='black', fontsize=8,
                                    bbox=dict(fc='lightcyan', alpha=0.3, pad=1))
                    break
        if ("V6" in lead) or ("aVL" in lead) or ("I" in lead):
            for r in r_idx:
                seg = sig[r:r + int(0.14 * fs)]
                ok, rpk = _detect_broad_or_notched_r(seg, fs)
                if ok:
                    broad_r_count += 1
                    if rpk is not None:
                        x = t[min(len(t) - 1, r + rpk)]
                        y = sig[min(len(sig) - 1, r + rpk)]
                        ax.annotate("R", (x, y), textcoords="offset points", xytext=(0, -12), ha='center', color='black', fontsize=8,
                                    bbox=dict(fc='yellow', alpha=0.3, pad=1))
                    break

        import matplotlib.patches as mpatches
        handles, _ = ax.get_legend_handles_labels()
        handles.extend([
            mpatches.Patch(color='purple', alpha=0.28, label='Wide QRS (≥0.12s)'),
            mpatches.Patch(color='orange', alpha=0.28, label='Borderline (0.11–0.12s)'),
            mpatches.Patch(color='green', alpha=0.28, label='Normal (0.06–0.11s)'),
            mpatches.Patch(color='lightgray', alpha=0.28, label='Narrow/Uncertain (<0.06s)'),
        ])
        ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lbbb_leads_analysis.png"))
    plt.close(fig)

    qrs_pool = [d for d in qrs_pool if d is not None]
    qrs_mean = float(np.mean(qrs_pool)) if qrs_pool else float("nan")
    prop_wide = float(np.mean(np.array(qrs_pool) > 0.12)) if qrs_pool else float("nan")

    # Histogram
    if qrs_pool:
        arr = np.array(qrs_pool)
        fig_hist, ax_hist = plt.subplots(figsize=(6.5, 3.6))
        max_v, min_v = float(np.max(arr)), float(np.min(arr))
        hmin = min(0.05, min_v - 0.01 if min_v > 0.05 else 0.04)
        hmax = max(0.25, max_v + 0.02)
        bins = np.arange(hmin, hmax, 0.01) if (hmax - hmin) > 0 else 'auto'
        ax_hist.hist(arr, bins=bins, ec='k', alpha=0.75, color='lightcoral')
        ax_hist.axvspan(0.06, 0.11, color='g', alpha=0.2, label='Normal (0.06–0.11s)')
        ax_hist.axvspan(0.11, 0.12, color='orange', alpha=0.2, label='Borderline (0.11–0.12s)')
        ax_hist.axvline(0.12, c='purple', ls='--', lw=2, label='0.12 s Threshold')
        ax_hist.axvline(qrs_mean, c='r', ls='--', lw=2, label=f'Mean {qrs_mean:.3f}s')
        ax_hist.set_title('QRS Duration Distribution')
        ax_hist.set_xlabel('Duration (s)'); ax_hist.set_ylabel('Count')
        ax_hist.legend(); ax_hist.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lbbb_qrs_histogram.png"))
        plt.close(fig_hist)
        # # 写出逐搏 CSV 供人工核对
        # try:
        #     import csv
        #     csv_path = os.path.join(output_dir, 'lbbb_qrs_durations.csv')
        #     with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        #         w = csv.writer(fcsv)
        #         w.writerow(["lead", "beat_index", "qrs_start_idx", "qrs_end_idx", "duration_s", "start_time_s", "end_time_s", "fs"])
        #         for row in perbeat_rows:
        #             lead, bi, sidx, eidx, dur, ts, te, fs_ = row
        #             w.writerow([lead, bi, sidx, eidx, f"{dur:.6f}", f"{ts:.6f}", f"{te:.6f}", fs_])
        # except Exception:
        #     pass

    meets_lbbb = qrs_pool and (qrs_mean >= 0.12) and (broad_r_count > 0) and (deep_s_count > 0)
    meets_ilbbb = qrs_pool and (0.11 <= qrs_mean < 0.12) and (broad_r_count > 0) and (deep_s_count > 0)
    if meets_lbbb:
        conclusion = "Meets criteria for LBBB"
    elif meets_ilbbb:
        conclusion = "Possible incomplete LBBB (borderline QRS with matching morphology)"
    else:
        conclusion = "Incomplete morphology or QRS duration; manual review recommended"

    report_text = "\n".join([
        f"▸ Record: {record_name}",
        f"▸ Average QRS width: {qrs_mean:.3f} s  (>0.12 s proportion: {prop_wide*100:.1f}% if available)",
        f"▸ QRS categories: wide={rs_counts['wide']}, borderline={rs_counts['borderline']}, normal={rs_counts['normal']}, narrow/uncertain={rs_counts['narrow']}",
        f"▸ Deep S in V1: {'Yes' if deep_s_count>0 else 'No'}",
        f"▸ Broad/Notched R in lateral (V6/I/aVL): {'Yes' if broad_r_count>0 else 'No'}",
        f"\nSimple Conclusion: {conclusion}"
    ])

    metrics = {
        "qrs_mean_s": qrs_mean if not np.isnan(qrs_mean) else None,
        "wide_qrs_proportion": prop_wide if not np.isnan(prop_wide) else None,
        "deep_s_v1": deep_s_count > 0,
        "broad_r_lateral": broad_r_count > 0,
        "qrs_category_counts": rs_counts,
    }

    return {
        "title": f"LBBB Screening ({record_name})",
        "report_text": report_text,
        "metrics": metrics,
        "image_paths": {
            "lbbb_leads_analysis.png": os.path.join(output_dir, "lbbb_leads_analysis.png"),
            "lbbb_qrs_histogram.png": os.path.join(output_dir, "lbbb_qrs_histogram.png")
        },
        "criteria_flags": {
            "mean_qrs_ge_0_12": bool(qrs_pool and (qrs_mean >= 0.12)),
            "deep_s_v1": deep_s_count > 0,
            "broad_r_lateral": broad_r_count > 0,
        }
    }