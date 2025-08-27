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
    subtype_confidence = metrics.get('subtype_confidence', 0.0)
    conduction_quality = metrics.get('conduction_ratio', 0.0)
    
    if total_p == 0:
        return _mk('fail', ['no P detections'], {})
    
    ratio = conducted / total_p if total_p else 0.0
    
    # Enhanced validation with confidence scoring
    reasons = []
    metrics_used = {'dropped': dropped, 'ratio': round(ratio, 2)}
    
    # Check for dropped beats (primary criterion)
    if dropped >= 1:
        reasons.append(f'dropped={dropped}')
        
        # Check for clear subtype identification
        if wenck >= 1 and subtype_confidence > 0.6:
            reasons.append(f'wenck={wenck} (Mobitz I)')
            metrics_used['wenck'] = wenck
            metrics_used['subtype_confidence'] = round(subtype_confidence, 2)
            return _mk('pass', reasons, metrics_used)
        
        elif pr_std is not None and pr_std < 0.025 and subtype_confidence > 0.7:
            reasons.append(f'pr_std={pr_std:.3f} (Mobitz II)')
            metrics_used['pr_std'] = pr_std
            metrics_used['subtype_confidence'] = round(subtype_confidence, 2)
            return _mk('pass', reasons, metrics_used)
        
        elif dropped >= 2:
            reasons.append('high-grade block')
            return _mk('pass', reasons, metrics_used)
        
        else:
            reasons.append('unspecified subtype')
            return _mk('warn', reasons, metrics_used)
    
    return _mk('fail', ['no dropped beats detected'], {'ratio': round(ratio, 2)})


def validate_avb3(metrics: dict) -> dict:
    ratio = metrics.get('conduction_ratio')
    pp_rate = metrics.get('pp_rate_bpm')
    rr_rate = metrics.get('rr_rate_bpm')
    pr_cv = metrics.get('pr_cv')
    dissociation_score = metrics.get('dissociation_score')
    diagnosis_confidence = metrics.get('diagnosis_confidence')
    subtype = metrics.get('subtype', '')
    
    if ratio is None or pp_rate is None or rr_rate is None:
        return _mk('fail', ['insufficient metrics for AV dissociation'], {'ratio': ratio, 'pp_rate': pp_rate, 'rr_rate': rr_rate})
    
    # Enhanced validation using new scoring system
    reasons = []
    metrics_used = {'ratio': round(ratio, 2)}
    
    # Use dissociation score if available
    if dissociation_score is not None:
        if dissociation_score >= 0.8:
            reasons.append(f'high_dissociation_score_{dissociation_score:.2f}')
            return _mk('pass', reasons, {'dissociation_score': round(dissociation_score, 2), **metrics_used})
        elif dissociation_score >= 0.6:
            reasons.append(f'dissociation_score_{dissociation_score:.2f}')
            return _mk('pass', reasons, {'dissociation_score': round(dissociation_score, 2), **metrics_used})
        elif dissociation_score >= 0.4:
            reasons.append(f'moderate_dissociation_{dissociation_score:.2f}')
            return _mk('warn', reasons, {'dissociation_score': round(dissociation_score, 2), **metrics_used})
    
    # Fallback to traditional criteria
    rate_gap = abs(pp_rate - rr_rate)
    
    if ratio < 0.2 and rate_gap > 15 and (pr_cv is None or pr_cv > 0.5):
        reasons.extend([
            f'conduction_ratio {ratio:.2f}<0.20',
            f'rate_gap {rate_gap:.1f}>15 bpm',
            f'pr_cv {pr_cv}'
        ])
        return _mk('pass', reasons, {'ratio': round(ratio, 2), 'rate_gap_bpm': round(rate_gap, 1), 'pr_cv': pr_cv})
    
    if ratio < 0.3 and rate_gap > 10:
        reasons.extend([f'conduction_ratio {ratio:.2f}', f'rate_gap {rate_gap:.1f}'])
        return _mk('warn', reasons, {'ratio': round(ratio, 2), 'rate_gap_bpm': round(rate_gap, 1)})
    
    return _mk('fail', ['criteria for complete AV block not met'], {'ratio': round(ratio, 2), 'rate_gap_bpm': round(rate_gap, 1)})


def validate_aflt(metrics: dict) -> dict:
    # Handle both single-lead and multi-lead AFLT metrics
    dom = metrics.get('dominant_rate_bpm')
    pr = metrics.get('band_power_ratio')
    
    # If top-level metrics are missing, try to extract from per_lead
    if dom is None or pr is None:
        per_lead = metrics.get('per_lead', {})
        if per_lead:
            # Find the best lead for validation (prefer II, then III, then aVF)
            best_lead = None
            for lead_candidate in ["II", "III", "aVF"]:
                if lead_candidate in per_lead:
                    best_lead = lead_candidate
                    break
            
            if best_lead:
                lead_data = per_lead[best_lead]
                dom = dom or lead_data.get('dominant_rate_bpm')
                pr = pr or lead_data.get('band_power_ratio')
    
    if dom is None or pr is None:
        return _mk('fail', ['insufficient spectrum metrics'], {'dominant_rate_bpm': dom, 'band_power_ratio': pr})
    
    # Additional validation checks
    reasons = []
    metrics_used = {'dominant_rate_bpm': dom, 'band_power_ratio': pr}
    
    # Check if atrial rate is in physiological range for flutter
    if 220 <= dom <= 360:
        reasons.append(f'dominant_rate {dom:.0f} bpm in flutter band')
        rate_score = 1.0
    elif 200 <= dom <= 380:
        reasons.append(f'dominant_rate {dom:.0f} bpm (borderline)')
        rate_score = 0.7
    else:
        reasons.append(f'dominant_rate {dom:.0f} bpm outside flutter range')
        rate_score = 0.0
    
    # Check power ratio
    if pr >= 0.40:
        reasons.append(f'power_ratio {pr:.2f}≥0.40 (strong)')
        power_score = 1.0
    elif pr >= 0.25:
        reasons.append(f'power_ratio {pr:.2f} (moderate)')
        power_score = 0.6
    else:
        reasons.append(f'power_ratio {pr:.2f} (weak)')
        power_score = 0.0
    
    # Additional checks for multi-lead AFLT
    if 'per_lead' in metrics:
        per_lead = metrics['per_lead']
        v1_evidence = metrics.get('v1_evidence_score')
        av_ratio = metrics.get('estimated_av_ratio')
        
        if v1_evidence is not None:
            reasons.append(f'V1 evidence {v1_evidence:.2f}')
            metrics_used['v1_evidence_score'] = v1_evidence
        
        if av_ratio is not None:
            reasons.append(f'AV ratio {av_ratio}:1')
            metrics_used['estimated_av_ratio'] = av_ratio
    
    # Calculate overall score
    overall_score = (rate_score + power_score) / 2
    
    if overall_score >= 0.8:
        return _mk('pass', reasons, metrics_used)
    elif overall_score >= 0.5:
        return _mk('warn', reasons, metrics_used)
    else:
        return _mk('fail', reasons, metrics_used)



