"""Rule-based validators for each abnormality.

Each function returns a dict:
  {
    "status": "pass"|"warn"|"fail",
    "reasons": [list of short strings],
    "metrics_used": {subset of metrics}
  }

These validators rely on metrics returned by visualization modules. If a
metric is missing, the validator will degrade to a warn/fail with a reason.
"""

def _mk(status, reasons, metrics_used):
    return {"status": status, "reasons": reasons, "metrics_used": metrics_used}


def validate_af(metrics: dict) -> dict:
    rr_mean = metrics.get("rr_mean_s")
    rr_std = metrics.get("rr_std_s")
    n = metrics.get("rr_interval_count")
    reasons = []
    if rr_mean and rr_std and n and n >= 6:
        cv = rr_std / rr_mean if rr_mean else 0.0
        if cv >= 0.12:
            return _mk("pass", [f"rr_cv {cv:.2f}≥0.12", f"n={n}"], {"rr_cv": round(cv, 3), "n": n})
        else:
            return _mk("warn", [f"rr_cv {cv:.2f}<0.12 (weak irregularity)", f"n={n}"], {"rr_cv": round(cv, 3), "n": n})
    return _mk("fail", ["insufficient RR metrics for AF validation"], {"rr_mean_s": rr_mean, "rr_std_s": rr_std, "n": n})


def validate_st(metrics: dict) -> dict:
    hr = metrics.get("average_heart_rate_bpm") or 0
    tr = metrics.get("tachy_ratio")
    reasons = []
    if tr is None:
        # Fallback on HR only
        if hr > 100:
            return _mk("warn", [f"avg_hr {hr:.1f}>100 but tachy_ratio missing"], {"avg_hr": hr})
        return _mk("fail", [f"avg_hr {hr:.1f}≤100 and tachy_ratio missing"], {"avg_hr": hr})
    if hr > 100 and tr >= 0.6:
        return _mk("pass", [f"avg_hr {hr:.1f}>100", f"tachy_ratio {tr:.2f}≥0.60"], {"avg_hr": hr, "tachy_ratio": tr})
    if hr > 100 or tr >= 0.6:
        return _mk("warn", [f"avg_hr {hr:.1f}", f"tachy_ratio {tr:.2f}"], {"avg_hr": hr, "tachy_ratio": tr})
    return _mk("fail", [f"avg_hr {hr:.1f}≤100 and tachy_ratio {tr:.2f}<0.60"], {"avg_hr": hr, "tachy_ratio": tr})


def validate_sb(metrics: dict) -> dict:
    hr = metrics.get("average_heart_rate_bpm") or 0
    br = metrics.get("brady_ratio")
    if br is None:
        if hr < 60:
            return _mk("warn", [f"avg_hr {hr:.1f}<60 but brady_ratio missing"], {"avg_hr": hr})
        return _mk("fail", [f"avg_hr {hr:.1f}≥60 and brady_ratio missing"], {"avg_hr": hr})
    if hr < 60 or br >= 0.5:
        return _mk("pass", [f"avg_hr {hr:.1f}<60 or brady_ratio {br:.2f}≥0.50"], {"avg_hr": hr, "brady_ratio": br})
    return _mk("fail", [f"avg_hr {hr:.1f}≥60 and brady_ratio {br:.2f}<0.50"], {"avg_hr": hr, "brady_ratio": br})


def validate_avb1(metrics: dict) -> dict:
    mean_pr = metrics.get("pr_mean_s")
    total = metrics.get("total_pr_count") or 0
    prolonged = metrics.get("prolonged_pr_count") or 0
    if mean_pr is None or total == 0:
        return _mk("fail", ["no PR metrics"], {"pr_mean_s": mean_pr, "total": total})
    share = prolonged / total if total else 0.0
    if mean_pr > 0.20 and share >= 0.8:
        return _mk("pass", [f"mean_pr {mean_pr:.3f}>0.20", f"share {share:.2f}≥0.80"], {"mean_pr": mean_pr, "share": round(share, 2)})
    if mean_pr > 0.20 or share >= 0.8:
        return _mk("warn", [f"mean_pr {mean_pr:.3f}", f"share {share:.2f}"], {"mean_pr": mean_pr, "share": round(share, 2)})
    return _mk("fail", [f"mean_pr {mean_pr:.3f}≤0.20 and share {share:.2f}<0.80"], {"mean_pr": mean_pr, "share": round(share, 2)})


def validate_lbbb(metrics: dict) -> dict:
    mean_qrs = metrics.get("qrs_mean_s")
    total = metrics.get("total_qrs_count") or 0
    wide = metrics.get("wide_qrs_count") or 0
    if mean_qrs is None:
        return _mk("fail", ["no QRS metrics"], {})
    share = wide / total if total else 0.0
    if (mean_qrs >= 0.12) or (share >= 0.5 and total >= 5):
        return _mk("pass", [f"mean_qrs {mean_qrs:.3f}≥0.12 or wide_share {share:.2f}≥0.50"], {"mean_qrs": mean_qrs, "wide_share": round(share, 2)})
    if mean_qrs >= 0.11 or share >= 0.3:
        return _mk("warn", [f"mean_qrs {mean_qrs:.3f}", f"wide_share {share:.2f}"], {"mean_qrs": mean_qrs, "wide_share": round(share, 2)})
    return _mk("fail", [f"mean_qrs {mean_qrs:.3f}<0.11 and wide_share {share:.2f}<0.30"], {"mean_qrs": mean_qrs, "wide_share": round(share, 2)})


def validate_rbbb(metrics: dict) -> dict:
    mean_qrs = metrics.get("qrs_mean_s")
    rsr = metrics.get("rsr_pattern_detected")
    deep_s = metrics.get("wide_s_wave_detected")
    if mean_qrs is None or rsr is None or deep_s is None:
        return _mk("fail", ["incomplete metrics for RBBB"], {})
    if (mean_qrs >= 0.12) and rsr and deep_s:
        return _mk("pass", ["mean_qrs ≥0.12", "rsR' in V1/V2", "deep S in V6/I"], {"mean_qrs": mean_qrs, "rsr": rsr, "deep_s": deep_s})
    if (0.11 <= mean_qrs < 0.12) and rsr and deep_s:
        return _mk("warn", ["borderline QRS with morphology (iRBBB)"], {"mean_qrs": mean_qrs, "rsr": rsr, "deep_s": deep_s})
    return _mk("fail", ["QRS duration or morphology not sufficient"], {"mean_qrs": mean_qrs, "rsr": rsr, "deep_s": deep_s})


def validate_nr(metrics: dict) -> dict:
    hr = metrics.get("average_heart_rate_bpm")
    rr_cv = metrics.get("rr_cv")
    ok_hr = hr is not None and 60 <= hr <= 100
    ok_cv = rr_cv is not None and rr_cv < 0.08
    if ok_hr and ok_cv:
        return _mk("pass", ["HR 60–100", f"rr_cv {rr_cv:.2f}<0.08"], {"avg_hr": hr, "rr_cv": rr_cv})
    if ok_hr or ok_cv:
        return _mk("warn", [f"avg_hr {hr}", f"rr_cv {rr_cv}"], {"avg_hr": hr, "rr_cv": rr_cv})
    return _mk("fail", ["neither HR nor rr_cv in normal range"], {"avg_hr": hr, "rr_cv": rr_cv})


def validate_avb2(metrics: dict) -> dict:
    conducted = metrics.get('conducted_count') or 0
    total_p = metrics.get('total_p_count') or 0
    dropped = metrics.get('dropped_p_count') or 0
    wenck = metrics.get('wenckebach_seq_count') or 0
    pr_std = metrics.get('pr_std_s')
    if total_p == 0:
        return _mk('fail', ['no P detections'], {})
    ratio = conducted / total_p if total_p else 0.0
    # Pass if there are dropped beats and either wenckebach present (Mobitz I) or low PR variability (Mobitz II)
    if dropped >= 1 and (wenck >= 1 or (pr_std is not None and pr_std < 0.03)):
        return _mk('pass', [f'dropped={dropped}', f'wenck={wenck}', f'pr_std={pr_std}'], {'dropped': dropped, 'wenck': wenck, 'pr_std': pr_std, 'ratio': round(ratio, 2)})
    if dropped >= 1:
        return _mk('warn', [f'dropped={dropped} without clear subtype'], {'dropped': dropped, 'ratio': round(ratio, 2)})
    return _mk('fail', ['no dropped beats detected'], {'ratio': round(ratio, 2)})


def validate_avb3(metrics: dict) -> dict:
    ratio = metrics.get('conduction_ratio')
    pp_rate = metrics.get('pp_rate_bpm')
    rr_rate = metrics.get('rr_rate_bpm')
    pr_cv = metrics.get('pr_cv')
    if ratio is None or pp_rate is None or rr_rate is None:
        return _mk('fail', ['insufficient metrics for AV dissociation'], {'ratio': ratio, 'pp_rate': pp_rate, 'rr_rate': rr_rate})
    rate_gap = abs(pp_rate - rr_rate)
    if ratio < 0.2 and rate_gap > 15 and (pr_cv is None or pr_cv > 0.5):
        return _mk('pass', [f'conduction_ratio {ratio:.2f}<0.20', f'rate_gap {rate_gap:.1f}>15 bpm', f'pr_cv {pr_cv}'], {'ratio': round(ratio, 2), 'rate_gap_bpm': round(rate_gap, 1), 'pr_cv': pr_cv})
    if ratio < 0.3 and rate_gap > 10:
        return _mk('warn', [f'conduction_ratio {ratio:.2f}', f'rate_gap {rate_gap:.1f}'], {'ratio': round(ratio, 2), 'rate_gap_bpm': round(rate_gap, 1)})
    return _mk('fail', ['criteria for complete AV block not met'], {'ratio': round(ratio, 2), 'rate_gap_bpm': round(rate_gap, 1)})


def validate_aflt(metrics: dict) -> dict:
    dom = metrics.get('dominant_rate_bpm')
    pr = metrics.get('band_power_ratio')
    if dom is None or pr is None:
        return _mk('fail', ['insufficient spectrum metrics'], {'dominant_rate_bpm': dom, 'band_power_ratio': pr})
    if 220 <= dom <= 360 and pr >= 0.40:
        return _mk('pass', [f'dominant_rate {dom:.0f} bpm in flutter band', f'power_ratio {pr:.2f}≥0.40'], {'dominant_rate_bpm': dom, 'band_power_ratio': pr})
    if 200 <= dom <= 380 and pr >= 0.25:
        return _mk('warn', [f'dominant_rate {dom:.0f} bpm', f'power_ratio {pr:.2f}'], {'dominant_rate_bpm': dom, 'band_power_ratio': pr})
    return _mk('fail', [f'no clear flutter signature: rate {dom}, ratio {pr}'], {'dominant_rate_bpm': dom, 'band_power_ratio': pr})



