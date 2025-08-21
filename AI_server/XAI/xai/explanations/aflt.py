import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
from ..ecg_utils import plot_ecg_with_grid


def _bandpass(signal, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)


def _autocorr(x):
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode='full')
    return ac[ac.size // 2:]


def _analyze_single_lead(ecg_signal, sampling_rate):
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_idx = rpeaks['ECG_R_Peaks']
    except Exception:
        r_idx = np.array([], dtype=int)
    # blanking
    blanked = ecg_cleaned.copy()
    blank_win = int(0.08 * sampling_rate)
    for r in r_idx:
        s = max(0, r - blank_win)
        e = min(len(blanked), r + blank_win)
        blanked[s:e] = 0.0
    atrial = _bandpass(blanked, sampling_rate, 3.0, 15.0)
    n = len(atrial)
    if n == 0:
        return None
    freqs = np.fft.rfftfreq(n, d=1.0/sampling_rate)
    spec = np.abs(np.fft.rfft(atrial))
    mask = (freqs >= 3.5) & (freqs <= 6.0)
    dom_freq_hz = None
    power_ratio = None
    if np.any(mask):
        idx = np.argmax(spec[mask])
        dom_freq_hz = float(freqs[mask][idx])
        band_power = float(np.sum(spec[mask] ** 2))
        total_power = float(np.sum(spec ** 2)) if np.sum(spec ** 2) > 0 else None
        power_ratio = (band_power / total_power) if total_power else None
    dom_rate_bpm = float(dom_freq_hz * 60.0) if dom_freq_hz else None
    ac = _autocorr(atrial)
    lag_min = int(sampling_rate / 6.0)
    lag_max = int(sampling_rate / 3.5)
    ac_seg = ac[lag_min:lag_max] if lag_max > lag_min else ac
    ac_dom_lag = None
    if ac_seg.size:
        pk_idx, _ = find_peaks(ac_seg)
        if pk_idx.size:
            ac_dom_lag = int(pk_idx[np.argmax(ac_seg[pk_idx])]) + lag_min
    ac_rate_bpm = float(60.0 * sampling_rate / ac_dom_lag) if ac_dom_lag and ac_dom_lag > 0 else None
    # F peaks
    atrial_hp = _bandpass(blanked, sampling_rate, 5.0, 15.0)
    f_peaks, _ = find_peaks(atrial_hp, distance=max(1, int(0.06 * sampling_rate)))
    f_rate_bpm = float(len(f_peaks) / (len(atrial) / sampling_rate) * 60.0) if len(f_peaks) else None
    rr = np.diff(r_idx) / sampling_rate if len(r_idx) > 1 else np.array([])
    rr_rate_bpm = float(60.0 / np.mean(rr)) if rr.size and np.mean(rr) > 0 else None
    return {
        'ecg_cleaned': ecg_cleaned,
        'r_idx': r_idx,
        'atrial_hp': atrial_hp,
        'f_peaks': f_peaks,
        'dom_freq_hz': dom_freq_hz,
        'dom_rate_bpm': dom_rate_bpm,
        'band_power_ratio': float(power_ratio) if power_ratio is not None else None,
        'ac_rate_bpm': ac_rate_bpm,
        'f_rate_bpm': f_rate_bpm,
        'rr_rate_bpm': rr_rate_bpm,
    }


def generate_aflt_multilead(leads_data, ecg_title, output_dir):
    """Multi-lead AFLT with clinical preference: visualize II and V1 only.
    - II is primary; if II missing, fall back to III, then aVF as "inferior substitute".
    - V1 is the right atrial lead; include if present.
    Other inferior leads are analyzed only for assistance (not plotted) when II is absent.
    """
    def pick(keys, token):
        tok = token.upper()
        for k in keys:
            base = k.replace("Lead ", "").strip().upper()
            if base == tok:
                return k
        return None
    keys = list(leads_data.keys())
    k_II  = pick(keys, 'II')
    k_III = pick(keys, 'III')
    k_aVF = pick(keys, 'aVF') or pick(keys, 'AVF')
    k_V1  = pick(keys, 'V1')

    # Choose inferior primary (II -> III -> aVF)
    inf_key = k_II or k_III or k_aVF
    if not inf_key and not k_V1:
        print("No preferred leads (II/III/aVF/V1) found for AFLT multi-lead.")
        return None

    # Analyze only the two displayed leads: inferior primary and V1 (if present)
    plot_pairs = []
    if inf_key:
        lab_inf = 'II' if inf_key == k_II else ('III' if inf_key == k_III else 'aVF')
        plot_pairs.append((lab_inf, inf_key))
    if k_V1:
        plot_pairs.append(('V1', k_V1))

    per_lead = {}
    for lab, key in plot_pairs:
        sig, fs = leads_data[key]
        res = _analyze_single_lead(sig, fs)
        if res:
            res['fs'] = fs
            res['label'] = lab
            res['lead_key'] = key
            per_lead[lab] = res

    # Inferior consistency: II/III/aVF present and close freq
    # We only plot one inferior lead; consistency check is skipped unless both II and a substitute were analyzed
    inferior_labels = [l for l in ("II", "III", "aVF") if l in per_lead]
    inferior_consistent = False
    inferior_mean_rate = None

    v1_evidence = None
    if 'V1' in per_lead:
        v1_pr = per_lead['V1'].get('band_power_ratio')
        v1_rate = per_lead['V1'].get('dom_rate_bpm')
        v1_evidence = None if v1_pr is None or v1_rate is None else float(min(1.0, max(0.0, (v1_pr - 0.2) / 0.5)))

    # AV ratio estimation from best inferior (prefer II)
    best_for_ratio = None
    for cand in ("II", "III", "aVF"):
        if cand in per_lead:
            best_for_ratio = cand; break
    estimated_av_ratio = None
    if best_for_ratio:
        f_bpm = per_lead[best_for_ratio].get('f_rate_bpm') or per_lead[best_for_ratio].get('dom_rate_bpm')
        rr_bpm = per_lead[best_for_ratio].get('rr_rate_bpm')
        if f_bpm and rr_bpm and rr_bpm > 0:
            estimated_av_ratio = int(round(f_bpm / rr_bpm))

    # Plot multi-lead atrial emphasized signals
    h = max(1, len(per_lead))
    fig, axes = plt.subplots(h, 1, figsize=(15, 3.2 * h), sharex=True)
    axes = np.array(axes).ravel()
    # Build summary text for figure
    summary_parts = []
    if inferior_labels:
        summary_parts.append(f"Inferior consistent: {'Yes' if inferior_consistent else 'No'}")
        if inferior_mean_rate:
            summary_parts.append(f"Inferior mean ≈ {inferior_mean_rate:.0f} bpm")
    if v1_evidence is not None:
        summary_parts.append(f"V1 evidence {v1_evidence:.2f}")
    if estimated_av_ratio:
        summary_parts.append(f"AV ratio ≈ {estimated_av_ratio}:1")

    for ax, (lab, res) in zip(axes, per_lead.items()):
        sig = res['atrial_hp']; fs = res['fs']
        t = np.linspace(0, len(sig) / fs, len(sig), endpoint=False)
        # Title with rates
        title = f"{lab} – Atrial-emphasized (5–15 Hz)"
        if res.get('dom_rate_bpm'):
            title += f" | FFT {res['dom_rate_bpm']:.0f} bpm"
        if res.get('ac_rate_bpm'):
            title += f" | AC {res['ac_rate_bpm']:.0f} bpm"
        # Use ECG grid (major 0.2s, minor 0.04s) to match other modules
        plot_ecg_with_grid(t, sig, title, ax, fs, show_grid=True, style="color")
        fp = res['f_peaks']
        if fp is not None and len(fp):
            ax.scatter(t[fp], sig[fp], c='red', s=10)
        # overlay R-peak timestamps as faint vertical lines for AV ratio intuition
        if res.get('r_idx') is not None and len(res['r_idx']):
            ymin, ymax = ax.get_ylim()
            ax.vlines(t[res['r_idx']], ymin, ymax, color='dodgerblue', alpha=0.15, lw=0.6)
            # Highlight RR windows by number of F-waves (2:1 green, 3:1 orange, 4:1 purple)
            try:
                r_idx = res['r_idx']
                if fp is not None and len(fp) and len(r_idx) >= 2:
                    for r1, r2 in zip(r_idx[:-1], r_idx[1:]):
                        # bounds check
                        if r2 <= r1 or r2 >= len(t):
                            continue
                        cnt = int(np.sum((fp > r1) & (fp < r2)))
                        if cnt == 2:
                            color = 'green'
                        elif cnt == 3:
                            color = 'orange'
                        elif cnt >= 4:
                            color = 'purple'
                        else:
                            color = 'lightgray'
                        ax.axvspan(t[r1], t[r2], alpha=0.12, color=color)
                        # optional label for inferior lead only
                        # place small text label near top
                        if lab in ("II", "III", "aVF") and cnt in (2,3,4):
                            ax.text((t[r1]+t[r2])/2, ymax - 0.05*(ymax-ymin), f"{cnt}:1", ha='center', va='top', fontsize=8, color=color)
            except Exception:
                pass
        # grid already applied by plot_ecg_with_grid
    axes[-1].set_xlabel('Time (s)')
    # add legend proxies and summary
    try:
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
        proxies = [
            Line2D([0], [0], color='purple', lw=1.5, label='Atrial-emphasized'),
            Line2D([0], [0], color='dodgerblue', lw=2, alpha=0.3, label='R-peaks (vlines)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='F-wave peaks'),
            mpatches.Patch(color='green',  alpha=0.25, label='2:1 window'),
            mpatches.Patch(color='orange', alpha=0.25, label='3:1 window'),
            mpatches.Patch(color='purple', alpha=0.25, label='≥4:1 window'),
        ]
        axes[0].legend(handles=proxies, loc='upper right')
    except Exception:
        pass
    if summary_parts:
        fig.suptitle(' | '.join(summary_parts), fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(output_dir, 'aflt_multilead_overview.png')); plt.close(fig)

    # Combined spectrum up to 15 Hz for available leads
    try:
        fig_sp, ax_sp = plt.subplots(figsize=(10, 4.5))
        colors = {
            'II': 'tab:blue', 'III': 'tab:green', 'aVF': 'tab:orange', 'V1': 'tab:red'
        }
        for lab, res in per_lead.items():
            sig = res['atrial_hp']; fs = res['fs']
            n = len(sig)
            if n <= 0: continue
            freqs = np.fft.rfftfreq(n, d=1.0/fs)
            spec = np.abs(np.fft.rfft(sig))
            # normalize for comparison
            spec = spec / (np.max(spec) if np.max(spec) > 0 else 1.0)
            mask = freqs <= 15.0
            ax_sp.plot(freqs[mask], spec[mask], label=lab, color=colors.get(lab, None), alpha=0.9)
            # mark dominant freq if available
            if res.get('dom_rate_bpm'):
                df_hz = res['dom_rate_bpm'] / 60.0
                if df_hz <= 15.0:
                    ax_sp.axvline(df_hz, color=colors.get(lab, 'gray'), ls='--', lw=1)
        ax_sp.axvspan(3.5, 6.0, color='orange', alpha=0.15, label='Flutter band (3.5–6.0 Hz)')
        ax_sp.set_xlim(0, 15); ax_sp.set_xlabel('Frequency (Hz)'); ax_sp.set_ylabel('Normalized magnitude')
        ax_sp.set_title('Multi-lead spectrum (atrial-emphasized)')
        ax_sp.grid(True, alpha=0.3); ax_sp.legend(loc='upper right')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'aflt_multilead_spectrum.png')); plt.close(fig_sp)
    except Exception:
        pass

    # Combined autocorrelation up to 1.0 s
    try:
        fig_ac, ax_ac = plt.subplots(figsize=(10, 4.5))
        colors = {
            'II': 'tab:blue', 'III': 'tab:green', 'aVF': 'tab:orange', 'V1': 'tab:red'
        }
        for lab, res in per_lead.items():
            sig = res['atrial_hp']; fs = res['fs']
            ac = _autocorr(sig)
            lags = np.arange(len(ac)) / fs
            mask = lags <= 1.0
            ax_ac.plot(lags[mask], ac[mask] / (np.max(ac) if np.max(ac) > 0 else 1.0), color=colors.get(lab, None), label=lab)
            if res.get('ac_rate_bpm'):
                lag_s = 60.0 / res['ac_rate_bpm'] if res['ac_rate_bpm'] else None
                if lag_s and lag_s <= 1.0:
                    ax_ac.axvline(lag_s, color=colors.get(lab, 'gray'), ls='--', lw=1)
        ax_ac.set_xlim(0, 1.0); ax_ac.set_xlabel('Lag (s)'); ax_ac.set_ylabel('Normalized AC')
        ax_ac.set_title('Multi-lead autocorrelation (atrial-emphasized)')
        ax_ac.grid(True, alpha=0.3); ax_ac.legend(loc='upper right')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'aflt_multilead_autocorr.png')); plt.close(fig_ac)
    except Exception:
        pass

    # Build metrics
    metrics = {
        'inferior_consistent': bool(inferior_consistent),
        'inferior_mean_rate_bpm': float(inferior_mean_rate) if inferior_mean_rate else None,
        'v1_evidence_score': float(v1_evidence) if v1_evidence is not None else None,
        'estimated_av_ratio': int(estimated_av_ratio) if estimated_av_ratio else None,
        'per_lead': {
            lab: {
                'dominant_rate_bpm': per_lead[lab].get('dom_rate_bpm'),
                'band_power_ratio': per_lead[lab].get('band_power_ratio'),
                'autocorr_rate_bpm': per_lead[lab].get('ac_rate_bpm'),
                'fwave_rate_bpm': per_lead[lab].get('f_rate_bpm'),
                'rr_rate_bpm': per_lead[lab].get('rr_rate_bpm'),
            } for lab in per_lead
        }
    }

    report_lines = [
        f"Inferior leads consistent: {'Yes' if inferior_consistent else 'No'}",
        f"Inferior mean atrial rate: {inferior_mean_rate:.0f} bpm" if inferior_mean_rate else "Inferior mean atrial rate: n/a",
        f"V1 evidence score: {v1_evidence:.2f}" if v1_evidence is not None else "V1 evidence score: n/a",
        f"Estimated AV conduction ratio: {estimated_av_ratio}:1" if estimated_av_ratio else "Estimated AV conduction ratio: n/a",
    ]
    report_text = "\n".join(report_lines)

    return {
        'title': ecg_title,
        'lead_name': 'II/III/aVF/V1',
        'report_text': report_text,
        'metrics': metrics,
        'image_paths': ['aflt_multilead_overview.png', 'aflt_multilead_spectrum.png', 'aflt_multilead_autocorr.png']
    }


def generate_aflt_visuals(ecg_signal, sampling_rate, ecg_title, lead_name, output_dir):
    if ecg_signal is None:
        return None

    # Clean
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    except Exception:
        ecg_cleaned = ecg_signal

    # R-peaks for blanking and AV ratio
    try:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        r_idx = rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"R-peak detection failed for {ecg_title}: {e}")
        r_idx = np.array([], dtype=int)

    # QRS blanking (±80 ms)
    blanked = ecg_cleaned.copy()
    blank_win = int(0.08 * sampling_rate)
    for r in r_idx:
        s = max(0, r - blank_win)
        e = min(len(blanked), r + blank_win)
        blanked[s:e] = 0.0

    # Emphasize atrial activity (3–15 Hz)
    atrial = _bandpass(blanked, sampling_rate, 3.0, 15.0)

    # Frequency analysis (FFT)
    n = len(atrial)
    if n == 0:
        return None
    freqs = np.fft.rfftfreq(n, d=1.0/sampling_rate)
    spec = np.abs(np.fft.rfft(atrial))

    # Dominant frequency in 3.5–6.0 Hz (210–360 bpm)
    mask = (freqs >= 3.5) & (freqs <= 6.0)
    dom_freq_hz = None
    power_ratio = None
    if np.any(mask):
        idx = np.argmax(spec[mask])
        dom_freq_hz = float(freqs[mask][idx])
        band_power = float(np.sum(spec[mask] ** 2))
        total_power = float(np.sum(spec ** 2)) if np.sum(spec ** 2) > 0 else None
        power_ratio = (band_power / total_power) if total_power else None
    dom_rate_bpm = float(dom_freq_hz * 60.0) if dom_freq_hz else None

    # Autocorrelation-based periodicity check
    ac = _autocorr(atrial)
    lag_min = int(sampling_rate / 6.0)
    lag_max = int(sampling_rate / 3.5)
    ac_seg = ac[lag_min:lag_max] if lag_max > lag_min else ac
    ac_dom_lag = None
    if ac_seg.size:
        pk_idx, _ = find_peaks(ac_seg)
        if pk_idx.size:
            ac_dom_lag = int(pk_idx[np.argmax(ac_seg[pk_idx])]) + lag_min
    ac_rate_bpm = float(60.0 * sampling_rate / ac_dom_lag) if ac_dom_lag and ac_dom_lag > 0 else None

    # Estimate AV conduction ratio from F-wave vs R-peaks
    atrial_hp = _bandpass(blanked, sampling_rate, 5.0, 15.0)
    f_peaks, _ = find_peaks(atrial_hp, distance=max(1, int(0.06 * sampling_rate)))
    f_rate_bpm = float(len(f_peaks) / (len(atrial) / sampling_rate) * 60.0) if len(f_peaks) else None
    rr = np.diff(r_idx) / sampling_rate if len(r_idx) > 1 else np.array([])
    rr_rate_bpm = float(60.0 / np.mean(rr)) if rr.size and np.mean(rr) > 0 else None
    av_ratio = None
    if f_rate_bpm and rr_rate_bpm and rr_rate_bpm > 0:
        av_ratio = float(round(f_rate_bpm / rr_rate_bpm))

    # Plots (same as before)
    t = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned), endpoint=False)

    fig, ax = plt.subplots(figsize=(15, 4))
    plot_ecg_with_grid(t, ecg_cleaned, f"{ecg_title} ({lead_name}) - Atrial Flutter Overview", ax, sampling_rate)
    if len(r_idx):
        ax.scatter(t[r_idx], ecg_cleaned[r_idx], c='dodgerblue', s=20, label='R-peaks', zorder=5)
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'aflt_overview.png')); plt.close(fig)

    # Removed aflt_atrial_signal.png per request

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(freqs, spec, color='black')
    ax3.axvspan(3.5, 6.0, color='orange', alpha=0.2, label='Flutter band (3.5–6.0 Hz)')
    if dom_freq_hz:
        ax3.axvline(dom_freq_hz, color='red', ls='--', lw=2, label=f'Dom {dom_freq_hz:.2f} Hz ({dom_rate_bpm:.0f} bpm)')
    ax3.set_xlim(0, 15); ax3.set_xlabel('Frequency (Hz)'); ax3.set_ylabel('Magnitude'); ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'aflt_spectrum.png')); plt.close(fig3)

    lags = np.arange(len(ac)) / sampling_rate
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(lags, ac, color='gray')
    if ac_dom_lag:
        ax4.axvline(ac_dom_lag / sampling_rate, color='red', ls='--', lw=2, label='AC peak')
    ax4.set_xlim(0, 1.0); ax4.set_title('Autocorrelation of atrial-emphasized signal'); ax4.set_xlabel('Lag (s)'); ax4.set_ylabel('AC'); ax4.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'aflt_autocorr.png')); plt.close(fig4)

    report_lines = [
        f"Dominant atrial frequency (FFT): {dom_freq_hz:.2f} Hz" if dom_freq_hz else "Dominant atrial frequency (FFT): n/a",
        f"Dominant atrial rate (FFT): {dom_rate_bpm:.0f} bpm" if dom_rate_bpm else "Dominant atrial rate (FFT): n/a",
        f"Autocorr atrial rate: {ac_rate_bpm:.0f} bpm" if ac_rate_bpm else "Autocorr atrial rate: n/a",
        f"Power ratio (3.5–6.0 Hz / total): {power_ratio:.2f}" if power_ratio is not None else "Power ratio: n/a",
        f"Estimated AV conduction ratio: {int(av_ratio)}:1" if av_ratio else "Estimated AV conduction ratio: n/a",
    ]
    report_text = "\n".join(report_lines)

    metrics = {
        'dominant_freq_hz': dom_freq_hz,
        'dominant_rate_bpm': dom_rate_bpm,
        'band_power_ratio': float(power_ratio) if power_ratio is not None else None,
        'autocorr_rate_bpm': ac_rate_bpm,
        'fwave_rate_bpm': f_rate_bpm,
        'rr_rate_bpm': rr_rate_bpm,
        'estimated_av_ratio': int(av_ratio) if av_ratio else None,
    }

    return {
        'title': ecg_title,
        'lead_name': lead_name,
        'report_text': report_text,
        'metrics': metrics,
        'image_paths': ['aflt_overview.png', 'aflt_spectrum.png', 'aflt_autocorr.png']
    }
