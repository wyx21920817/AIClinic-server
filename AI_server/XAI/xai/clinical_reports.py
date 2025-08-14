# clinical_reports.py

import numpy as np

def generate_report_1davb(pr_intervals, used_delineation_method, ecg_title, lead_name):
    """Generates a text report for 1st Degree AV Block analysis."""
    report = [f"\nClinical Observation for {ecg_title} ({lead_name}):"]
    
    if pr_intervals is None or pr_intervals.size == 0:
        delineation_status = used_delineation_method if used_delineation_method else "Failed"
        report.append(f"- Automatic delineation (method: {delineation_status}) could not reliably compute PR intervals.")
    else:
        mean_pr = np.mean(pr_intervals)
        count_total = len(pr_intervals)
        count_prolonged = np.sum(pr_intervals > 0.20)
        percentage_prolonged = (count_prolonged / count_total) * 100 if count_total > 0 else 0
        
        report.append("- PR interval is prolonged if > 0.20 seconds.")
        report.append(f"- Calculated PR intervals: Mean={mean_pr:.3f}s, Min={np.min(pr_intervals):.3f}s, Max={np.max(pr_intervals):.3f}s.")
        report.append(f"- {count_prolonged} out of {count_total} ({percentage_prolonged:.1f}%) calculated PR intervals are > 0.20s.")

    return "\n".join(report)

def generate_report_afib(ecg_title, lead_name):
    """Generates a text report for Atrial Fibrillation analysis."""
    return (
        f"\nObservation for {ecg_title} ({lead_name}):\n"
        "- P-waves typically absent/fibrillatory.\n"
        "- Yellow boxes: Inferred P-wave regions.\n"
        "- RR histogram shows 'irregularly irregular' rhythm."
    )

def generate_report_sinus_tachy(ecg_title, lead_name, avg_hr):
    """Generates a text report for Sinus Tachycardia analysis."""
    return (
        f"\nObs for {ecg_title} ({lead_name}):\n"
        f"- HR > 100 bpm. Avg: {avg_hr:.1f} bpm. RR intervals < 0.6s.\n"
        "- For Sinus rhythms, P waves normal & precede QRS. Rhythm regular."
    )

def generate_report_bbb(abnormality_name, lead_name, estimation_type, qrs_durations_sec):
    """Generates a text report for Bundle Branch Block analysis."""
    report = [f"\nClinical Observation for {abnormality_name} ({lead_name}): QRS duration is considered prolonged if > 0.12 seconds (typically)."]
    if qrs_durations_sec:
        qrs_durations_sec = np.array(qrs_durations_sec)
        num_wide = np.sum(qrs_durations_sec > 0.12)
        report.append(f"- QRS durations ({estimation_type}): Mean={np.mean(qrs_durations_sec):.3f}s, Min={np.min(qrs_durations_sec):.3f}s, Max={np.max(qrs_durations_sec):.3f}s.")
        report.append(f"- {num_wide} out of {len(qrs_durations_sec)} QRS durations ({estimation_type}) are > 0.12s.")
    else:
        report.append("- No QRS durations could be reliably calculated or estimated.")
    return "\n".join(report)