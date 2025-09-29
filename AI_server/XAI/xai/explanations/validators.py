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


def _extract_from_report(report_or_metrics, keys):
    """Helper: accept either legacy metrics dict or new report dict.
    Returns a dict of requested keys if present in any known place.
    """
    out = {k: None for k in keys}
    obj = report_or_metrics or {}
    if isinstance(obj, dict) and ('summary' in obj or 'statistics' in obj):
        stats = obj.get('statistics', {}) or {}
        summ = obj.get('summary', {}) or {}
        for k in keys:
            if k in stats:
                out[k] = stats.get(k)
            elif k in summ:
                out[k] = summ.get(k)
    else:
        for k in keys:
            out[k] = obj.get(k) if isinstance(obj, dict) else None
    return out


def validate_af(report_or_metrics) -> dict:
    fields = _extract_from_report(report_or_metrics, ["rr_mean_s", "rr_std_s", "rr_interval_count"])
    rr_mean = fields.get("rr_mean_s")
    rr_std = fields.get("rr_std_s")
    n = fields.get("rr_interval_count")
    reasons = []
    if rr_mean and rr_std and n and n >= 6:
        cv = rr_std / rr_mean if rr_mean else 0.0
        if cv >= 0.12:
            return _mk("pass", [f"RR interval variability indicates irregular rhythm (CV = {cv:.3f} ≥ 0.12)", f"Sufficient RR intervals analyzed (n={n})"], {"rr_cv": round(cv, 3), "n": n})
        else:
            return _mk("warn", [f"RR interval variability shows mild irregularity (CV = {cv:.3f} < 0.12)", f"RR intervals analyzed (n={n})"], {"rr_cv": round(cv, 3), "n": n})
    return _mk("fail", ["Insufficient RR interval data for reliable atrial fibrillation assessment"], {"rr_mean_s": rr_mean, "rr_std_s": rr_std, "n": n})


def validate_st(report_or_metrics) -> dict:
    """
    Enhanced Sinus Tachycardia validation with clinically appropriate criteria.
    
    Clinical criteria for sinus tachycardia:
    - Sustained: Average HR > 100 bpm OR high proportion of tachycardic beats (≥60%)
    - Significant episodes: Any instantaneous HR > 120 bpm OR moderate proportion (≥15%)
    - Normal: No significant tachycardic episodes
    """
    fields = _extract_from_report(report_or_metrics, ["average_heart_rate_bpm", "tachy_ratio", "max_instantaneous_hr"])
    hr = fields.get("average_heart_rate_bpm") or 0
    tr = fields.get("tachy_ratio")
    max_hr = fields.get("max_instantaneous_hr")  # Peak instantaneous heart rate
    
    if tr is None:
        if hr > 100:
            return _mk("warn", [f"Heart rate indicates tachycardia (HR = {hr:.1f} bpm > 100) but tachycardia ratio data missing"], {"avg_hr": hr})
        return _mk("fail", [f"Heart rate is normal (HR = {hr:.1f} bpm ≤ 100) and tachycardia ratio data missing"], {"avg_hr": hr})
    
    # Enhanced validation criteria
    reasons = []
    
    # Rule 1: Sustained sinus tachycardia (original criteria)
    if hr > 100 and tr >= 0.6:
        reasons.append(f"Sustained sinus tachycardia: average HR {hr:.1f} bpm > 100 and high tachycardic beat ratio {tr:.2f} ≥ 0.60")
        return _mk("pass", reasons, {"avg_hr": hr, "tachy_ratio": tr, "max_hr": max_hr})
    
    # Rule 2: Peak heart rate criterion (NEW)
    if max_hr and max_hr > 120:
        reasons.append(f"Significant tachycardic episodes detected: peak instantaneous HR {max_hr:.1f} bpm > 120")
        if tr and tr >= 0.15:  # Additional evidence
            reasons.append(f"Moderate proportion of tachycardic beats ({tr:.2f} ≥ 0.15) supports diagnosis")
        return _mk("pass", reasons, {"avg_hr": hr, "tachy_ratio": tr, "max_hr": max_hr})
    
    # Rule 3: Moderate tachycardic beat ratio (LOWERED THRESHOLD)
    if tr >= 0.15:  # Lowered from 0.6 to 0.15
        reasons.append(f"Moderate proportion of tachycardic beats ({tr:.2f} ≥ 0.15) indicates significant episodes")
        return _mk("pass", reasons, {"avg_hr": hr, "tachy_ratio": tr, "max_hr": max_hr})
    
    # Rule 4: Borderline cases
    if hr > 100 or (tr and tr >= 0.1):  # Some evidence but not meeting full criteria
        reasons.append(f"Borderline tachycardia: average HR {hr:.1f} bpm, tachycardic ratio {tr:.2f}")
        return _mk("warn", reasons, {"avg_hr": hr, "tachy_ratio": tr, "max_hr": max_hr})
    
    # Rule 5: Normal rhythm
    reasons.append(f"Normal sinus rhythm: average HR {hr:.1f} bpm ≤ 100 and low tachycardic beat ratio ({tr:.2f} < 0.15)")
    return _mk("fail", reasons, {"avg_hr": hr, "tachy_ratio": tr, "max_hr": max_hr})


def validate_sb(report_or_metrics) -> dict:
    fields = _extract_from_report(report_or_metrics, ["average_heart_rate_bpm", "brady_ratio"])
    hr = fields.get("average_heart_rate_bpm") or 0
    br = fields.get("brady_ratio")
    if br is None:
        if hr < 60:
            return _mk("warn", [f"Heart rate indicates bradycardia (HR = {hr:.1f} bpm < 60) but bradycardia ratio data missing"], {"avg_hr": hr})
        return _mk("fail", [f"Heart rate is normal (HR = {hr:.1f} bpm ≥ 60) and bradycardia ratio data missing"], {"avg_hr": hr})
    if hr < 60 or br >= 0.5:
        return _mk("pass", [f"Heart rate indicates sinus bradycardia (HR = {hr:.1f} bpm < 60) or high proportion of bradycardic beats (ratio = {br:.2f} ≥ 0.50)"], {"avg_hr": hr, "brady_ratio": br})
    return _mk("fail", [f"Heart rate is normal (HR = {hr:.1f} bpm ≥ 60) and low bradycardic beat ratio ({br:.2f} < 0.50)"], {"avg_hr": hr, "brady_ratio": br})


def validate_avb1(report_or_metrics) -> dict:
    """
    Enhanced 1st Degree AV Block validation with clinically appropriate criteria.
    
    Clinical definition: PR interval consistently > 0.20 seconds.
    - Pass: Mean PR ≥ 0.20s with significant evidence (≥50% prolonged intervals)
    - Warn: Borderline cases with some evidence
    - Fail: No evidence of prolonged PR intervals
    """
    fields = _extract_from_report(report_or_metrics, ["pr_mean_s", "total_pr_count", "prolonged_pr_count"])
    mean_pr = fields.get("pr_mean_s")
    total = fields.get("total_pr_count") or 0
    prolonged = fields.get("prolonged_pr_count") or 0
    
    if mean_pr is None or total == 0:
        return _mk("fail", ["PR interval metrics are not available for first-degree AV block assessment"], {"pr_mean_s": mean_pr, "total": total})
    
    share = prolonged / total if total else 0.0
    
    # Enhanced validation criteria for 1dAVB
    # Pass conditions (clinical 1dAVB present):
    if mean_pr >= 0.20 and share >= 0.5:
        return _mk("pass", [f"PR interval indicates first-degree AV block (mean = {mean_pr:.3f}s ≥ 0.20s)", f"High proportion of prolonged intervals ({share:.1%} ≥ 50%)"], 
                  {"mean_pr": mean_pr, "share": round(share, 2)})
    
    if mean_pr > 0.20 and share >= 0.3:
        return _mk("pass", [f"PR interval shows clear prolongation (mean = {mean_pr:.3f}s > 0.20s)", f"Moderate proportion of prolonged intervals ({share:.1%} ≥ 30%)"], 
                  {"mean_pr": mean_pr, "share": round(share, 2)})
    
    if share >= 0.8:  # Majority of intervals prolonged regardless of mean
        return _mk("pass", [f"Majority of PR intervals are prolonged ({share:.1%} ≥ 80%)", f"Mean PR interval is {mean_pr:.3f}s"], 
                  {"mean_pr": mean_pr, "share": round(share, 2)})
    
    # Special case: Mean exactly at threshold (0.20) with any prolonged intervals
    if abs(mean_pr - 0.20) < 0.001 and share > 0.0:
        return _mk("pass", [f"Mean PR interval is at the diagnostic threshold ({mean_pr:.3f}s = 0.20s)", f"Some prolonged intervals detected ({share:.1%} > 0%)"], 
                  {"mean_pr": mean_pr, "share": round(share, 2)})
    
    # Warn conditions (borderline/possible 1dAVB):
    if mean_pr >= 0.19 and share >= 0.3:
        return _mk("warn", [f"Borderline PR interval duration (mean = {mean_pr:.3f}s approaching 0.20s threshold)", f"Moderate proportion of prolonged intervals ({share:.1%} ≥ 30%)"], 
                  {"mean_pr": mean_pr, "share": round(share, 2)})
    
    if mean_pr > 0.20 or share >= 0.5:
        return _mk("warn", [f"Partial evidence for first-degree AV block (mean PR = {mean_pr:.3f}s, prolonged proportion = {share:.1%})"], 
                  {"mean_pr": mean_pr, "share": round(share, 2)})
    
    # Fail condition (no significant evidence of 1dAVB):
    return _mk("fail", [f"The mean PR interval is within the normal range ({mean_pr:.3f}s < 0.20s) and the proportion of prolonged intervals is too low to be clinically significant ({share:.1%} < 50%)"], 
              {"mean_pr": mean_pr, "share": round(share, 2)})


def validate_lbbb(report_or_metrics) -> dict:
    fields = _extract_from_report(report_or_metrics, ["qrs_mean_s", "total_qrs_count", "wide_qrs_count"])
    mean_qrs = fields.get("qrs_mean_s")
    total = fields.get("total_qrs_count") or 0
    wide = fields.get("wide_qrs_count") or 0
    if mean_qrs is None:
        return _mk("fail", ["QRS duration metrics are not available for LBBB assessment"], {})
    share = wide / total if total else 0.0
    if (mean_qrs >= 0.12) or (share >= 0.5 and total >= 5):
        return _mk("pass", [f"QRS duration indicates left bundle branch block (mean = {mean_qrs:.3f}s ≥ 0.12s) or high proportion of wide complexes ({share:.1%} ≥ 50%)"], {"mean_qrs": mean_qrs, "wide_share": round(share, 2)})
    if mean_qrs >= 0.11 or share >= 0.3:
        return _mk("warn", [f"Borderline QRS duration suggesting incomplete LBBB (mean = {mean_qrs:.3f}s, wide complex proportion = {share:.1%})"], {"mean_qrs": mean_qrs, "wide_share": round(share, 2)})
    return _mk("fail", [f"QRS duration is normal (mean = {mean_qrs:.3f}s < 0.11s) and low proportion of wide complexes ({share:.1%} < 30%)"], {"mean_qrs": mean_qrs, "wide_share": round(share, 2)})


def validate_rbbb(report_or_metrics) -> dict:
    fields = _extract_from_report(report_or_metrics, ["qrs_mean_s", "rsr_pattern_detected", "wide_s_wave_detected"])
    mean_qrs = fields.get("qrs_mean_s")
    rsr = fields.get("rsr_pattern_detected")
    deep_s = fields.get("wide_s_wave_detected")
    if mean_qrs is None or rsr is None or deep_s is None:
        return _mk("fail", ["Incomplete morphological or duration metrics for RBBB assessment"], {})
    if (mean_qrs >= 0.12) and rsr and deep_s:
        return _mk("pass", [f"QRS duration indicates right bundle branch block (mean = {mean_qrs:.3f}s ≥ 0.12s)", "rsR' pattern detected in V1/V2", "Deep S wave detected in V6/I"], {"mean_qrs": mean_qrs, "rsr": rsr, "deep_s": deep_s})
    if (0.11 <= mean_qrs < 0.12) and rsr and deep_s:
        return _mk("warn", [f"Borderline QRS duration with RBBB morphology suggesting incomplete RBBB (mean = {mean_qrs:.3f}s)"], {"mean_qrs": mean_qrs, "rsr": rsr, "deep_s": deep_s})
    return _mk("fail", [f"QRS duration ({mean_qrs:.3f}s if mean_qrs else 'unavailable') or morphological criteria insufficient for RBBB diagnosis"], {"mean_qrs": mean_qrs, "rsr": rsr, "deep_s": deep_s})


def validate_nr(report_or_metrics) -> dict:
    """Baseline rhythm analysis validation - provides context rather than pass/fail diagnosis."""
    fields = _extract_from_report(report_or_metrics, ["average_heart_rate_bpm", "rr_cv"])
    hr = fields.get("average_heart_rate_bpm")
    rr_cv = fields.get("rr_cv")
    
    if hr is None:
        return _mk("fail", ["Baseline heart rate data is not available for rhythm assessment"], {"avg_hr": hr})
    
    reasons = []
    
    # Heart rate assessment with baseline context
    if 60 <= hr <= 100:
        reasons.append(f"Baseline heart rate is normal ({hr:.1f} bpm within 60-100 bpm range)")
    elif hr < 60:
        reasons.append(f"Baseline heart rate indicates bradycardia ({hr:.1f} bpm < 60 bpm)")
    else:
        reasons.append(f"Baseline heart rate indicates tachycardia ({hr:.1f} bpm > 100 bpm)")
    
    # Rhythm regularity assessment with clinical interpretation
    if rr_cv is not None:
        if rr_cv < 0.08:
            reasons.append(f"Baseline rhythm is highly regular (coefficient of variation = {rr_cv:.6f} < 0.08)")
        else:
            reasons.append(f"Baseline rhythm shows variation (coefficient of variation = {rr_cv:.6f} ≥ 0.08, suggesting irregular rhythm)")
    else:
        reasons.append("Baseline rhythm regularity assessment requires more RR interval data points")
    
    # Always pass for baseline analysis - this provides context, not diagnosis
    return _mk("pass", reasons, {"avg_hr": hr, "rr_cv": rr_cv})


def validate_avb2(report_or_metrics) -> dict:
    fields = _extract_from_report(report_or_metrics, ['conducted_count','total_p_count','dropped_p_count','wenckebach_seq_count','pr_std_s','subtype_confidence','conduction_ratio'])
    conducted = fields.get('conducted_count') or 0
    total_p = fields.get('total_p_count') or 0
    dropped = fields.get('dropped_p_count') or 0
    wenck = fields.get('wenckebach_seq_count') or 0
    pr_std = fields.get('pr_std_s')
    subtype_confidence = fields.get('subtype_confidence', 0.0)
    conduction_quality = fields.get('conduction_ratio', 0.0)
    
    if total_p == 0:
        return _mk('fail', ['P-wave detection is insufficient for second-degree AV block assessment'], {})
    
    ratio = conducted / total_p if total_p else 0.0
    
    # Enhanced validation with detailed clinical reasoning
    reasons = []
    metrics_used = {'dropped': dropped, 'ratio': round(ratio, 2)}
    
    if dropped >= 1:
        reasons.append(f'A clinically significant number of dropped beats were detected (count: {dropped})')
        if wenck >= 1 and (subtype_confidence or 0) > 0.6:
            reasons.append(f'Wenckebach sequences identified (count: {wenck}), indicating Mobitz Type I pattern')
            metrics_used['wenck'] = wenck
            metrics_used['subtype_confidence'] = round(subtype_confidence or 0, 2)
            return _mk('pass', reasons, metrics_used)
        elif pr_std is not None and pr_std < 0.025 and (subtype_confidence or 0) > 0.7:
            reasons.append(f'PR interval variability is minimal (std: {pr_std:.3f}s), consistent with Mobitz Type II pattern')
            metrics_used['pr_std'] = pr_std
            metrics_used['subtype_confidence'] = round(subtype_confidence or 0, 2)
            return _mk('pass', reasons, metrics_used)
        elif dropped >= 2:
            reasons.append('The pattern is consistent with high-grade AV block (multiple dropped beats)')
            return _mk('pass', reasons, metrics_used)
        else:
            reasons.append('Second-degree AV block detected but subtype classification requires further evaluation')
            return _mk('warn', reasons, metrics_used)
    
    return _mk('fail', [f'No dropped beats detected - conduction ratio is normal ({ratio:.1%})'], {'ratio': round(ratio, 2)})


def validate_avb3(report_or_metrics) -> dict:
    fields = _extract_from_report(report_or_metrics, ['conduction_ratio','pp_rate_bpm','rr_rate_bpm','pr_cv','dissociation_score','diagnosis_confidence'])
    ratio = fields.get('conduction_ratio')
    pp_rate = fields.get('pp_rate_bpm')
    rr_rate = fields.get('rr_rate_bpm')
    pr_cv = fields.get('pr_cv')
    dissociation_score = fields.get('dissociation_score')
    diagnosis_confidence = fields.get('diagnosis_confidence')
    
    if ratio is None or pp_rate is None or rr_rate is None:
        return _mk('fail', ['Insufficient cardiac rhythm metrics available for third-degree AV block assessment'], {'ratio': ratio, 'pp_rate': pp_rate, 'rr_rate': rr_rate})
    
    reasons = []
    metrics_used = {'ratio': round(ratio, 2)}
    
    if dissociation_score is not None:
        if dissociation_score >= 0.8:
            reasons.append(f'Strong evidence of AV dissociation detected (dissociation score: {dissociation_score:.2f} ≥ 0.8)')
            return _mk('pass', reasons, {'dissociation_score': round(dissociation_score, 2), **metrics_used})
        elif dissociation_score >= 0.6:
            reasons.append(f'Moderate evidence of AV dissociation found (dissociation score: {dissociation_score:.2f} ≥ 0.6)')
            return _mk('pass', reasons, {'dissociation_score': round(dissociation_score, 2), **metrics_used})
        elif dissociation_score >= 0.4:
            reasons.append(f'Some evidence of AV dissociation present (dissociation score: {dissociation_score:.2f} ≥ 0.4)')
            return _mk('warn', reasons, {'dissociation_score': round(dissociation_score, 2), **metrics_used})
    
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


def validate_aflt(report_or_metrics) -> dict:
    """Validate atrial flutter based on frequency analysis and AV ratio."""
    fields = _extract_from_report(report_or_metrics, [
        "dominant_rate_bpm", "band_power_ratio", "estimated_av_ratio", 
        "fwave_rate_bpm", "autocorr_rate_bpm"
    ])
    
    dom_rate = fields.get("dominant_rate_bpm")
    power_ratio = fields.get("band_power_ratio")
    av_ratio = fields.get("estimated_av_ratio")
    f_rate = fields.get("fwave_rate_bpm")
    ac_rate = fields.get("autocorr_rate_bpm")
    
    if dom_rate is None and f_rate is None and ac_rate is None:
        return _mk("fail", ["Frequency analysis data unavailable - atrial flutter assessment cannot be completed"], {
            "dominant_rate_bpm": dom_rate,
            "fwave_rate_bpm": f_rate,
            "autocorr_rate_bpm": ac_rate
        })
    
    # Use best available rate estimate
    flutter_rate = dom_rate or f_rate or ac_rate
    
    # Atrial flutter criteria: 240-350 bpm flutter rate, power concentration, AV block
    criteria_met = 0
    reasons = []
    metrics = {}
    
    # Clinical criteria assessment with detailed reasoning
    if flutter_rate and 230 <= flutter_rate <= 350:
        criteria_met += 1
        reasons.append(f"Atrial rate of {flutter_rate:.0f} bpm is within the flutter range (230-350 bpm)")
        metrics["flutter_rate_bpm"] = flutter_rate
    elif flutter_rate:
        reasons.append(f"Atrial rate of {flutter_rate:.0f} bpm is outside the typical flutter range (230-350 bpm)")
        metrics["flutter_rate_bpm"] = flutter_rate
        
    if power_ratio and power_ratio > 0.6:
        criteria_met += 1
        reasons.append(f"Atrial signal power is well concentrated (power ratio {power_ratio:.2f} > 0.6)")
        metrics["power_ratio"] = power_ratio
    elif power_ratio:
        reasons.append(f"Atrial signal power is somewhat dispersed (power ratio {power_ratio:.2f} ≤ 0.6)")
        metrics["power_ratio"] = power_ratio
        
    if av_ratio and av_ratio >= 2:
        criteria_met += 1
        reasons.append(f"AV block is present (conduction ratio {av_ratio}:1 ≥ 2:1)")
        metrics["av_ratio"] = av_ratio
    elif av_ratio:
        reasons.append(f"AV conduction is near normal (conduction ratio {av_ratio}:1 < 2:1)")
        metrics["av_ratio"] = av_ratio
    
    if criteria_met >= 3:
        return _mk("pass", reasons + ["All three criteria for atrial flutter are satisfied"], metrics)
    elif criteria_met >= 2:
        return _mk("warn", reasons + [f"Conclusion based on {criteria_met} out of 3 criteria being met - suggestive of atrial flutter"], metrics)
    else:
        return _mk("fail", reasons + [f"Only {criteria_met} out of 3 criteria met - insufficient evidence for atrial flutter diagnosis"], metrics)


def validate_pac(report_or_metrics) -> dict:
    """Validation for PAC with candidate-aware reasons.

    Logic:
    - Pass if total_pac >= 1
    - Else fail, but include granular reasons from top candidates (hard veto / low P-wave).
    """
    fields = _extract_from_report(report_or_metrics, ['total_pac', 'noncompensatory_fraction', 'p_visibility_fraction', 'candidates', 'abnormality_count'])
    total_from_summary = fields.get('abnormality_count')
    total_pac = fields.get('total_pac')
    if total_from_summary is not None:
        total_pac = total_from_summary
    total_pac = total_pac or 0
    noncomp = fields.get('noncompensatory_fraction')
    pvis = fields.get('p_visibility_fraction')
    candidates = fields.get('candidates') or []

    reasons = []

    if total_pac >= 1:
        return {
            'status': 'pass',
            'score': 1.0,
            'reasons': [f'Premature atrial complexes detected (count = {total_pac})'],
            'metrics_used': {'total_pac': total_pac}
        }

    # Build granular failure reasons
    reasons.append('Final PAC count is 0.')
    if candidates:
        # sort by overall_pac desc
        sorted_cands = sorted(
            candidates,
            key=lambda c: c.get('scores', {}).get('overall_pac', 0.0),
            reverse=True
        )
        top = sorted_cands[0]
        p_score = top.get('scores', {}).get('p_wave', 0.0)
        overall = top.get('scores', {}).get('overall_pac', 0.0)
        veto = top.get('hard_veto', False)
        ts = top.get('time_s')
        rr_s = top.get('rr_short_s')
        if veto or p_score < 0.2:
            reasons.append(
                f"Candidate beat at {ts:.2f}s was rejected due to: ectopic P-wave not identified with sufficient confidence (score: {p_score:.2f})."
            )
        else:
            reasons.append(
                f"Candidate beat at {ts:.2f}s did not meet combined score (overall={overall:.2f})."
            )
        # Optional morphology info
        sim = top.get('scores', {}).get('qrs_similarity')
        if sim is not None:
            reasons.append(f"QRS similarity to baseline: {sim:.2f}")
        if rr_s is not None:
            reasons.append(f"Premature RR: {rr_s:.2f}s")

    # Aggregate score is 0 when none classified
    return {
        'status': 'fail',
        'score': 0.0,
        'reasons': reasons,
        'metrics_used': {
            'total_pac': total_pac,
            'noncompensatory_fraction': noncomp,
            'p_visibility_fraction': pvis,
        }
    }


def validate_pvc(report_or_metrics) -> dict:
    """Validation for PVC with candidate-aware reasons.

    Pass if total_pvc >= 1; otherwise include granular candidate summaries.
    """
    # Prefer summary.abnormality_count when present
    fields = _extract_from_report(report_or_metrics, ['total_pvc', 'compensatory_fraction', 'wide_qrs_fraction', 'candidates', 'abnormality_count'])
    total_from_summary = fields.get('abnormality_count')
    total_pvc = fields.get('total_pvc')
    if total_from_summary is not None:
        total_pvc = total_from_summary
    total_pvc = total_pvc or 0
    comp = fields.get('compensatory_fraction')
    wide = fields.get('wide_qrs_fraction')
    candidates = fields.get('candidates') or []

    if total_pvc >= 1:
        return {
            'status': 'pass',
            'score': 1.0,
            'reasons': [f'Premature ventricular complexes detected (count = {total_pvc})'],
            'metrics_used': {'total_pvc': total_pvc}
        }

    reasons = ['Final PVC count is 0.']
    if candidates:
        rejected = [c for c in candidates if not c.get('classified_as_pvc')]
        reasons.append(f"{len(rejected)} wide/premature candidates were rejected due to insufficient overall evidence.")
        # Include top one detail
        top = sorted(candidates, key=lambda c: c.get('scores', {}).get('overall_pvc', 0.0), reverse=True)[0]
        ts = top.get('time_s')
        sc = top.get('scores', {})
        reasons.append(
            f"Candidate at {ts:.2f}s details: Prematurity={sc.get('prematurity',0):.2f}, "
            f"QRS_Width={sc.get('qrs_width',0):.2f}, QRS_Morph_Abn={sc.get('qrs_morphology_abnormality',0):.2f}, "
            f"CompPause={sc.get('compensatory_pause',0):.2f}, P_absence={sc.get('p_absence',0):.2f}, "
            f"Overall={sc.get('overall_pvc',0):.2f} (Threshold≈0.5)"
        )

    return {
        'status': 'fail',
        'score': 0.0,
        'reasons': reasons,
        'metrics_used': {
            'total_pvc': total_pvc,
            'compensatory_fraction': comp,
            'wide_qrs_fraction': wide,
        }
    }
