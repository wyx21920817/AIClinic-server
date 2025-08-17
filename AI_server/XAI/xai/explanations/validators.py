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



