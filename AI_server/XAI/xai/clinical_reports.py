# clinical_reports.py

import numpy as np
import os
from typing import Dict, List, Any, Optional

class ReportGenerator:
    """
    Unified report generator for all ECG abnormality analysis modules.
    Provides consistent report formatting and content generation across all modules.
    """
    
    def __init__(self):
        self.templates = {
            'heart_rate_assessment': self._heart_rate_template,
            'rhythm_analysis': self._rhythm_analysis_template,
            'morphology_assessment': self._morphology_template,
            'conduction_assessment': self._conduction_template,
        }
    
    def generate_st_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Sinus Tachycardia (ST) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        avg_hr = metrics.get('average_heart_rate_bpm', 0)
        tachy_ratio = metrics.get('tachy_ratio', 0)
        max_hr = metrics.get('max_instantaneous_hr', 0)
        
        # Enhanced classification logic (consistent with validator)
        is_sustained_tachycardia = bool(avg_hr > 100 and tachy_ratio >= 0.6)
        is_significant_tachycardia = bool(max_hr > 120 or tachy_ratio >= 0.15)
        is_tachycardia_present = is_sustained_tachycardia or is_significant_tachycardia
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Heart Rate Assessment
        hr_section = self._generate_st_hr_section(avg_hr, tachy_ratio, max_hr, is_sustained_tachycardia, is_significant_tachycardia)
        sections.append(hr_section)
        
        # Section 2: Rhythm Analysis
        rhythm_section = self._generate_st_rhythm_section(tachy_ratio, is_sustained_tachycardia, is_significant_tachycardia, is_tachycardia_present)
        sections.append(rhythm_section)
        
        # Generate summary
        summary = self._generate_st_summary(avg_hr, tachy_ratio, max_hr, is_sustained_tachycardia, is_significant_tachycardia, is_tachycardia_present)
        
        # Generate title
        title = self._generate_st_title(is_sustained_tachycardia, is_significant_tachycardia, tachy_ratio)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_st_hr_section(self, avg_hr: float, tachy_ratio: float, max_hr: float, 
                               is_sustained: bool, is_significant: bool) -> Dict[str, str]:
        """Generate heart rate assessment section for ST analysis."""
        if is_sustained:
            hr_status = "Sustained Sinus Tachycardia"
            hr_description = "sustained tachycardia"
            hr_explanation = f"Sustained elevated heart rate with average {avg_hr:.0f} bpm and {tachy_ratio*100:.0f}% tachycardic beats."
        elif is_significant:
            if max_hr > 120:
                hr_status = "Sinus Rhythm with Tachycardic Episodes"
                hr_description = "sinus rhythm with significant tachycardic episodes"
                hr_explanation = f"Peak instantaneous heart rate reached {max_hr:.0f} bpm with {tachy_ratio*100:.0f}% of beats showing tachycardic intervals."
            else:
                hr_status = "Sinus Rhythm with Tachycardic Episodes"
                hr_description = "sinus rhythm with moderate tachycardic episodes"
                hr_explanation = f"Moderate proportion ({tachy_ratio*100:.0f}%) of beats showed tachycardic intervals, indicating significant episodes."
        else:
            # Borderline or normal cases
            if avg_hr > 100 or tachy_ratio >= 0.1:
                hr_status = "Borderline Tachycardia"
                hr_description = "borderline tachycardia with some elevated rates"
                hr_explanation = f"Some evidence of elevated heart rate with average {avg_hr:.0f} bpm and {tachy_ratio*100:.0f}% tachycardic beats, but not meeting full diagnostic criteria."
            else:
                hr_status = "Normal Heart Rate"
                hr_description = "normal heart rate"
                hr_explanation = "Heart rate remains within normal physiological range throughout the recording."
        
        return {
            "section_title": f"1. Heart Rate Assessment: {hr_status} ({avg_hr:.0f} bpm)",
            "content": f"Heart rate analysis shows {hr_description} with an average of {avg_hr:.0f} beats per minute. The [Heart Rate Timeline] displays instantaneous heart rate over time, with the 100 bpm threshold clearly marked. {hr_explanation}"
        }
    
    def _generate_st_rhythm_section(self, tachy_ratio: float, is_sustained: bool, 
                                   is_significant: bool, is_present: bool) -> Dict[str, str]:
        """Generate rhythm analysis section for ST analysis."""
        if is_sustained:
            rhythm_status = "Sustained Fast Sinus Rhythm"
            rhythm_description = "consistent but significantly accelerated"
            rhythm_explanation = f"sustained tachycardic rates"
        elif is_significant:
            rhythm_status = "Sinus Rhythm with Tachycardic Episodes"
            rhythm_description = "sinus rhythm with intermittent acceleration"
            rhythm_explanation = f"significant tachycardic episodes"
        else:
            # Borderline or normal cases
            if tachy_ratio >= 0.1:
                rhythm_status = "Borderline Accelerated Rhythm"
                rhythm_description = "sinus rhythm with occasional acceleration"
                rhythm_explanation = f"some intervals falling into the tachycardic zone"
            else:
                rhythm_status = "Normal Sinus Rhythm"
                rhythm_description = "normal"
                rhythm_explanation = "normal rates"
        
        return {
            "section_title": f"2. Rhythm Analysis: {rhythm_status}",
            "content": f"RR interval analysis demonstrates {rhythm_description}. The [RR Interval Distribution] shows the pattern of beat-to-beat intervals, with {tachy_ratio*100:.0f}% of intervals falling into the tachycardic zone. The [ECG Overview] displays regular P-QRS-T complexes {'at an elevated rate' if is_present else 'at normal rate'}."
        }
    
    def _generate_st_summary(self, avg_hr: float, tachy_ratio: float, max_hr: float,
                           is_sustained: bool, is_significant: bool, is_present: bool) -> Dict[str, Any]:
        """Generate summary for ST analysis."""
        if is_sustained:
            rhythm_type = "sustained sinus tachycardia"
            clinical_note = f"Sustained sinus tachycardia confirmed with average heart rate {avg_hr:.0f} bpm and {tachy_ratio*100:.0f}% tachycardic beats."
        elif is_significant:
            if max_hr > 120:
                rhythm_type = "sinus rhythm with tachycardic episodes"
                clinical_note = f"Sinus rhythm with significant tachycardic episodes detected. Peak instantaneous heart rate reached {max_hr:.0f} bpm with {tachy_ratio*100:.0f}% of beats showing tachycardic intervals."
            else:
                rhythm_type = "sinus rhythm with tachycardic episodes"
                clinical_note = f"Sinus rhythm with moderate tachycardic episodes detected. {tachy_ratio*100:.0f}% of beats showed tachycardic intervals, indicating significant episodes despite normal average heart rate."
        else:
            # Borderline or normal cases
            if avg_hr > 100 or tachy_ratio >= 0.1:
                rhythm_type = "borderline tachycardia"
                clinical_note = f"Borderline tachycardia with some evidence of elevated heart rate (average {avg_hr:.0f} bpm, {tachy_ratio*100:.0f}% tachycardic beats) but not meeting full diagnostic criteria."
            else:
                rhythm_type = "normal sinus rhythm"
                clinical_note = "Heart rate remains within normal physiological limits."
        
        return {
            'report_text': f"Heart rate analysis shows {rhythm_type} with average rate of {avg_hr:.0f} bpm. {clinical_note}",
            'is_abnormality_present': is_present,
            'abnormality_count': 1 if is_present else 0,
        }
    
    def _generate_st_title(self, is_sustained: bool, is_significant: bool, tachy_ratio: float) -> str:
        """Generate title for ST analysis."""
        if is_sustained:
            return "Sustained Sinus Tachycardia"
        elif is_significant:
            return "Sinus Rhythm with Tachycardic Episodes"
        elif tachy_ratio >= 0.1:
            return "Borderline Tachycardia"
        else:
            return "Normal Sinus Rhythm"
    
    def generate_sb_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Sinus Bradycardia (SB) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        avg_hr = metrics.get('average_heart_rate_bpm', 0)
        brady_ratio = metrics.get('brady_ratio', 0)
        
        # Enhanced classification logic (consistent with validator)
        is_bradycardia_present = bool(avg_hr < 60 or brady_ratio >= 0.5)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Heart Rate Assessment
        hr_section = self._generate_sb_hr_section(avg_hr, brady_ratio, is_bradycardia_present)
        sections.append(hr_section)
        
        # Section 2: Rhythm Analysis
        rhythm_section = self._generate_sb_rhythm_section(brady_ratio, is_bradycardia_present)
        sections.append(rhythm_section)
        
        # Generate summary
        summary = self._generate_sb_summary(avg_hr, brady_ratio, is_bradycardia_present)
        
        # Generate title
        title = self._generate_sb_title(is_bradycardia_present, brady_ratio)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_sb_hr_section(self, avg_hr: float, brady_ratio: float, is_present: bool) -> Dict[str, str]:
        """Generate heart rate assessment section for SB analysis."""
        if is_present:
            hr_status = "Sinus Bradycardia"
            hr_description = "bradycardia"
            hr_explanation = f"Sustained slow heart rate with average {avg_hr:.0f} bpm and {brady_ratio*100:.0f}% bradycardic beats."
        else:
            if brady_ratio > 0.0:
                hr_status = "Normal Average Rate"
                hr_description = "normal average rate with occasional slow beats"
                hr_explanation = f"While the average heart rate is normal, {brady_ratio*100:.0f}% of beats showed bradycardic intervals, though not meeting sustained bradycardia criteria."
            else:
                hr_status = "Normal Heart Rate"
                hr_description = "normal heart rate"
                hr_explanation = "Heart rate remains within normal physiological range throughout the recording."
        
        return {
            "section_title": f"1. Heart Rate Assessment: {hr_status} ({avg_hr:.0f} bpm)",
            "content": f"Heart rate analysis shows {hr_description} with an average of {avg_hr:.0f} beats per minute. The [Heart Rate Timeline] displays instantaneous heart rate over time, with the 60 bpm threshold clearly marked. {hr_explanation}"
        }
    
    def _generate_sb_rhythm_section(self, brady_ratio: float, is_present: bool) -> Dict[str, str]:
        """Generate rhythm analysis section for SB analysis."""
        if is_present:
            rhythm_status = "Slow Sinus Rhythm"
            rhythm_description = "consistent but slowed"
            rhythm_explanation = f"bradycardic rates (>1.0s)"
        else:
            if brady_ratio > 0.0:
                rhythm_status = "Sinus Rhythm with Rate Variation"
                rhythm_description = "sinus rhythm with occasional slowing"
                rhythm_explanation = f"some intervals falling into the bradycardic zone"
            else:
                rhythm_status = "Normal Sinus Rhythm"
                rhythm_description = "normal"
                rhythm_explanation = "normal rates"
        
        return {
            "section_title": f"2. Rhythm Analysis: {rhythm_status}",
            "content": f"RR interval analysis demonstrates {rhythm_description}. The [RR Interval Distribution] shows the pattern of beat-to-beat intervals, with {brady_ratio*100:.0f}% of intervals falling into the bradycardic zone. The brady zone (>1.0s) is highlighted in the histogram to show the proportion of slow beats."
        }
    
    def _generate_sb_summary(self, avg_hr: float, brady_ratio: float, is_present: bool) -> Dict[str, Any]:
        """Generate summary for SB analysis."""
        if is_present:
            rhythm_type = "sinus bradycardia"
            additional_info = f"Sustained slow heart rate with {brady_ratio*100:.0f}% of beats showing bradycardic intervals."
        else:
            if brady_ratio > 0.0:
                rhythm_type = "sinus rhythm with rate variation"
                additional_info = f"Heart rate shows variation with {brady_ratio*100:.0f}% of beats falling into the bradycardic zone, though not meeting sustained bradycardia criteria."
            else:
                rhythm_type = "normal sinus rhythm"
                additional_info = "Heart rate remains within normal physiological limits."
        
        return {
            'report_text': f"Heart rate analysis shows {rhythm_type} with average rate of {avg_hr:.0f} bpm. {additional_info}",
            'is_abnormality_present': is_present,
            'abnormality_count': 1 if is_present else 0,
        }
    
    def _generate_sb_title(self, is_present: bool, brady_ratio: float) -> str:
        """Generate title for SB analysis."""
        if is_present:
            return "Sinus Bradycardia"
        elif brady_ratio > 0.0:
            return "Sinus Rhythm with Rate Variation"
        else:
            return "Normal Sinus Rhythm"
    
    def generate_norm_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Normal (NORM) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        avg_hr = metrics.get('average_heart_rate_bpm', 0)
        rr_cv = metrics.get('rr_cv', 0)
        qrs_mean = metrics.get('qrs_mean_s', None)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Heart Rate Assessment
        hr_section = self._generate_norm_hr_section(avg_hr)
        sections.append(hr_section)
        
        # Section 2: Rhythm Assessment
        rhythm_section = self._generate_norm_rhythm_section(rr_cv)
        sections.append(rhythm_section)
        
        # Section 3: QRS Duration Assessment (if available)
        if qrs_mean is not None:
            qrs_section = self._generate_norm_qrs_section(qrs_mean, metrics)
            sections.append(qrs_section)
        
        # Generate summary
        summary = self._generate_norm_summary(avg_hr, rr_cv, qrs_mean)
        
        # Generate title
        title = self._generate_norm_title(rr_cv, qrs_mean)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_norm_hr_section(self, avg_hr: float) -> Dict[str, str]:
        """Generate heart rate assessment section for NORM analysis."""
        return {
            "section_title": f"1. Heart Rate Assessment: Normal (Average {avg_hr:.1f} bpm)",
            "content": f"The average heart rate is within the normal range. In the [RR Interval Distribution chart], the red dashed line representing your average RR interval ({60/avg_hr:.3f}s) falls squarely within the green shaded background, which indicates the 'normal heart rate range (60–100 bpm)'."
        }
    
    def _generate_norm_rhythm_section(self, rr_cv: float) -> Dict[str, str]:
        """Generate rhythm assessment section for NORM analysis."""
        if rr_cv < 0.08:
            rhythm_status = "Highly Regular"
            rhythm_description = "exceptionally consistent"
            rhythm_explanation = f"very stable time between each heartbeat. This high regularity is also confirmed by the extremely low coefficient of variation (rr_cv = {rr_cv:.4f})"
        elif rr_cv < 0.12:
            rhythm_status = "Regular"
            rhythm_description = "consistent"
            rhythm_explanation = f"stable time between heartbeats with moderate regularity (rr_cv = {rr_cv:.4f})"
        else:
            rhythm_status = "Irregular"
            rhythm_description = "some variation"
            rhythm_explanation = f"some variability in beat-to-beat intervals (rr_cv = {rr_cv:.4f})"
        
        return {
            "section_title": f"2. Rhythm Assessment: {rhythm_status}",
            "content": f"Your heart rhythm is {rhythm_description}. This is visually confirmed in the [ECG Overview chart]: the green boxes marking the 'continuous RR intervals' show {rhythm_explanation}."
        }
    
    def _generate_norm_qrs_section(self, qrs_mean: float, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate QRS duration assessment section for NORM analysis."""
        qrs_status = "Normal" if qrs_mean < 0.12 else "Wide"
        qrs_normal_count = metrics.get('qrs_in_normal_band_count', 0)
        
        return {
            "section_title": f"3. QRS Duration Assessment: {qrs_status} (Average {qrs_mean:.3f}s)",
            "content": f"QRS complex duration analysis shows {'normal' if qrs_mean < 0.12 else 'prolonged'} conduction. {qrs_normal_count} QRS complexes fall within the normal range (0.06-0.11s), as shown in the [QRS Duration Distribution chart] with green shading."
        }
    
    def _generate_norm_summary(self, avg_hr: float, rr_cv: float, qrs_mean: Optional[float]) -> Dict[str, Any]:
        """Generate summary for NORM analysis."""
        if rr_cv < 0.08:
            rhythm_desc = "highly regular"
        elif rr_cv < 0.12:
            rhythm_desc = "regular"
        else:
            rhythm_desc = "somewhat irregular"
        
        if qrs_mean is not None:
            qrs_desc = f"QRS duration is {'normal' if qrs_mean < 0.12 else 'prolonged'} ({qrs_mean:.3f}s)."
        else:
            qrs_desc = ""
        
        summary_text = f"Baseline rhythm analysis shows heart rate of {avg_hr:.1f} bpm. Rhythm assessment shows {rhythm_desc} pattern with coefficient of variation {rr_cv:.4f}. {qrs_desc} All measurements fall within normal physiological ranges."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': False,
            'abnormality_count': 0,
        }
    
    def _generate_norm_title(self, rr_cv: float, qrs_mean: Optional[float]) -> str:
        """Generate title for NORM analysis."""
        if rr_cv < 0.08:
            return "Normal Sinus Rhythm"
        elif rr_cv < 0.12:
            return "Normal Sinus Rhythm"
        else:
            return "Irregular Sinus Rhythm"
    
    def generate_af_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Atrial Fibrillation (AF) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        rr_cv = metrics.get('rr_cv', 0)
        rr_mean = metrics.get('rr_mean_s', 0)
        rr_std = metrics.get('rr_std_s', 0)
        rr_count = metrics.get('rr_interval_count', 0)
        
        # Determine if AF is present based on CV threshold
        is_present = bool(rr_cv >= 0.12)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Rhythm Assessment
        rhythm_section = self._generate_af_rhythm_section(is_present, rr_cv, rr_mean, rr_std)
        sections.append(rhythm_section)
        
        # Section 2: P-wave Assessment
        p_section = self._generate_af_pwave_section(is_present)
        sections.append(p_section)
        
        # Generate summary
        summary = self._generate_af_summary(is_present, rr_cv)
        
        # Generate title
        title = self._generate_af_title(is_present)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_af_rhythm_section(self, is_present: bool, rr_cv: float, rr_mean: float, rr_std: float) -> Dict[str, str]:
        """Generate rhythm assessment section for AF analysis."""
        if is_present:
            rhythm_status = "Irregular"
            rhythm_description = "significant irregularity"
            rhythm_explanation = f"which exceeds the 0.12 threshold indicating atrial fibrillation"
        else:
            rhythm_status = "Regular"
            rhythm_description = "regularity"
            rhythm_explanation = f"which is within normal limits"
        
        return {
            "section_title": f"1. Rhythm Assessment: {rhythm_status}",
            "content": f"Heart rhythm analysis shows {rhythm_description}. The RR interval coefficient of variation is {rr_cv:.3f}, {rhythm_explanation}. This is visualized in the [RR Interval Histogram] showing the distribution of beat-to-beat intervals."
        }
    
    def _generate_af_pwave_section(self, is_present: bool) -> Dict[str, str]:
        """Generate P-wave assessment section for AF analysis."""
        if is_present:
            p_status = "Absent/Fibrillatory"
            p_description = "absence of clear P-waves, replaced by fibrillatory waves characteristic of atrial fibrillation"
            p_explanation = "instead showing irregular baseline oscillations"
        else:
            p_status = "Present"
            p_description = "presence of clear P-waves preceding each QRS complex"
            p_explanation = "indicating normal atrial activation"
        
        return {
            "section_title": f"2. P-wave Assessment: {p_status}",
            "content": f"Analysis of P-wave regions shows {p_description}. The [ECG Overview] and [P-wave Regions] charts highlight areas where normal P-waves would be expected, {p_explanation}."
        }
    
    def _generate_af_summary(self, is_present: bool, rr_cv: float) -> Dict[str, Any]:
        """Generate summary for AF analysis."""
        if is_present:
            rhythm_desc = "irregular rhythm pattern consistent with atrial fibrillation"
            detail_desc = "P-waves are absent or fibrillatory, and RR intervals show characteristic irregularly irregular pattern"
        else:
            rhythm_desc = "regular rhythm without evidence of atrial fibrillation"
            detail_desc = "P-waves are present and RR intervals show regular pattern"
        
        summary_text = f"Atrial fibrillation analysis shows {rhythm_desc}. {detail_desc}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': 1 if is_present else 0,
        }
    
    def _generate_af_title(self, is_present: bool) -> str:
        """Generate title for AF analysis."""
        if is_present:
            return "Atrial Fibrillation"
        else:
            return "Regular Rhythm"
    
    def generate_pvc_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Premature Ventricular Complex (PVC) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        pvc_count = metrics.get('total_pvc', 0)
        total_beats = metrics.get('total_beats', 0)
        wide_qrs_fraction = metrics.get('wide_qrs_fraction', 0)
        heart_rate_bpm = metrics.get('heart_rate_bpm', 0)
        
        # Determine if PVC is present
        is_present = bool(pvc_count > 0)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Ectopic Beat Assessment
        beat_section = self._generate_pvc_beat_section(is_present, pvc_count, total_beats)
        sections.append(beat_section)
        
        # Section 2: Morphological Analysis
        morphology_section = self._generate_pvc_morphology_section(is_present, wide_qrs_fraction, pvc_count, total_beats)
        sections.append(morphology_section)
        
        # Section 3: Clinical Significance (if PVC present)
        if is_present:
            clinical_section = self._generate_pvc_clinical_section(pvc_count, heart_rate_bpm)
            sections.append(clinical_section)
        
        # Generate summary
        summary = self._generate_pvc_summary(is_present, pvc_count)
        
        # Generate title
        title = self._generate_pvc_title(is_present)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_pvc_beat_section(self, is_present: bool, pvc_count: int, total_beats: int) -> Dict[str, str]:
        """Generate ectopic beat assessment section for PVC analysis."""
        if is_present:
            beat_status = f"Detected ({pvc_count})"
            beat_description = f"detected {pvc_count} PVC events"
            beat_explanation = f"representing {pvc_count/total_beats*100:.1f}% of total beats"
        else:
            beat_status = "Not Detected"
            beat_description = "did not detect any PVC events"
            beat_explanation = "all beats appear to originate from normal sinus rhythm"
        
        return {
            "section_title": f"1. Ectopic Beat Assessment: {beat_status}",
            "content": f"Premature ventricular complex analysis {beat_description}. This {beat_explanation}, as visualized in the [ECG Overview] chart where PVCs are highlighted with red markers."
        }
    
    def _generate_pvc_morphology_section(self, is_present: bool, wide_qrs_fraction: float, pvc_count: int = 0, total_beats: int = 0) -> Dict[str, str]:
        """Generate morphological analysis section for PVC analysis."""
        if is_present:
            morph_status = "Wide QRS Morphology"
            morph_description = "ectopic ventricular beats show characteristic wide QRS morphology"
            # Use PVC count for consistency with beat assessment section
            morph_explanation = f"with {pvc_count/total_beats*100:.1f}% of beats exhibiting wide QRS complexes" if total_beats > 0 else f"with {wide_qrs_fraction*100:.1f}% of beats exhibiting wide QRS complexes"
        else:
            morph_status = "Normal QRS Morphology"
            morph_description = "all beats appear to originate from normal sinus rhythm without ventricular ectopy"
            morph_explanation = "QRS complexes maintain normal duration and morphology"
        
        return {
            "section_title": f"2. Morphological Analysis: {morph_status}",
            "content": f"QRS complex analysis shows {morph_description}. This is {morph_explanation}, as demonstrated in the [QRS Width Analysis] chart."
        }
    
    def _generate_pvc_clinical_section(self, pvc_count: int, heart_rate_bpm: float) -> Dict[str, str]:
        """Generate clinical significance section for PVC analysis."""
        if pvc_count == 1:
            clinical_significance = "isolated PVC"
            clinical_desc = "represents an isolated ectopic beat"
        elif pvc_count <= 5:
            clinical_significance = "occasional PVCs"
            clinical_desc = "represents occasional ventricular ectopy"
        else:
            clinical_significance = "frequent PVCs"
            clinical_desc = "represents frequent ventricular ectopy that may warrant clinical attention"
        
        return {
            "section_title": f"3. Clinical Significance: {clinical_significance}",
            "content": f"The detection of {pvc_count} PVC(s) {clinical_desc}. The underlying heart rate is {heart_rate_bpm:.1f} bpm. {'Frequent PVCs may require further evaluation' if pvc_count > 5 else 'Occasional PVCs are often benign but should be monitored'}."
        }
    
    def _generate_pvc_summary(self, is_present: bool, pvc_count: int) -> Dict[str, Any]:
        """Generate summary for PVC analysis."""
        if is_present:
            detection_text = f"detected {pvc_count} PVC events"
            morphology_text = "Ectopic ventricular beats show characteristic wide QRS morphology and compensatory pauses"
        else:
            detection_text = "did not detect any PVC events"
            morphology_text = "All beats appear to originate from normal sinus rhythm without ventricular ectopy"
        
        summary_text = f"Premature ventricular complex analysis {detection_text}. {morphology_text}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': int(pvc_count),
        }
    
    def _generate_pvc_title(self, is_present: bool) -> str:
        """Generate title for PVC analysis."""
        if is_present:
            return "Premature Ventricular Complexes"
        else:
            return "No PVC Detected"
    
    def generate_pac_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Premature Atrial Complex (PAC) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        pac_count = metrics.get('total_pac', 0)
        total_beats = metrics.get('total_beats', 0)
        p_visibility_fraction = metrics.get('p_visibility_fraction', 0)
        noncompensatory_fraction = metrics.get('noncompensatory_fraction', 0)
        heart_rate_bpm = metrics.get('heart_rate_bpm', 0)
        
        # Determine if PAC is present
        is_present = bool(pac_count > 0)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Ectopic Beat Assessment
        beat_section = self._generate_pac_beat_section(is_present, pac_count, total_beats)
        sections.append(beat_section)
        
        # Section 2: P-wave and Conduction Analysis
        p_section = self._generate_pac_pwave_section(is_present, p_visibility_fraction, noncompensatory_fraction)
        sections.append(p_section)
        
        # Section 3: Clinical Significance (if PAC present)
        if is_present:
            clinical_section = self._generate_pac_clinical_section(pac_count, heart_rate_bpm)
            sections.append(clinical_section)
        
        # Generate summary
        summary = self._generate_pac_summary(is_present, pac_count)
        
        # Generate title
        title = self._generate_pac_title(is_present)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_pac_beat_section(self, is_present: bool, pac_count: int, total_beats: int) -> Dict[str, str]:
        """Generate ectopic beat assessment section for PAC analysis."""
        if is_present:
            beat_status = f"Detected ({pac_count})"
            beat_description = f"detected {pac_count} PAC events"
            beat_explanation = f"representing {pac_count/total_beats*100:.1f}% of total beats"
        else:
            beat_status = "Not Detected"
            beat_description = "did not detect any PAC events"
            beat_explanation = "all beats appear to originate from normal sinus node"
        
        return {
            "section_title": f"1. Ectopic Beat Assessment: {beat_status}",
            "content": f"Premature atrial complex analysis {beat_description}. This {beat_explanation}, as visualized in the [ECG Overview] chart where PACs are highlighted with their characteristic early P-waves and narrow QRS complexes."
        }
    
    def _generate_pac_pwave_section(self, is_present: bool, p_visibility_fraction: float, noncompensatory_fraction: float) -> Dict[str, str]:
        """Generate P-wave and conduction analysis section for PAC analysis."""
        if is_present:
            p_status = "Ectopic P-waves Present"
            p_description = f"clear ectopic P-waves in {p_visibility_fraction*100:.1f}% of detected PACs"
            p_explanation = f"Non-compensatory pauses were observed in {noncompensatory_fraction*100:.1f}% of PACs, consistent with atrial origin"
        else:
            p_status = "Normal P-wave Morphology"
            p_description = "normal P-wave morphology throughout the recording"
            p_explanation = "no ectopic atrial activity detected"
        
        return {
            "section_title": f"2. P-wave and Conduction Analysis: {p_status}",
            "content": f"P-wave analysis shows {p_description}. {p_explanation}. The [RR Interval Analysis] demonstrates the timing characteristics typical of supraventricular ectopy."
        }
    
    def _generate_pac_clinical_section(self, pac_count: int, heart_rate_bpm: float) -> Dict[str, str]:
        """Generate clinical significance section for PAC analysis."""
        if pac_count == 1:
            clinical_significance = "isolated PAC"
            clinical_desc = "represents an isolated atrial ectopic beat"
        elif pac_count <= 5:
            clinical_significance = "occasional PACs"
            clinical_desc = "represents occasional atrial ectopy"
        else:
            clinical_significance = "frequent PACs"
            clinical_desc = "represents frequent atrial ectopy that may warrant clinical attention"
        
        return {
            "section_title": f"3. Clinical Significance: {clinical_significance}",
            "content": f"The detection of {pac_count} PAC(s) {clinical_desc}. The underlying heart rate is {heart_rate_bpm:.1f} bpm. {'Frequent PACs may be associated with atrial arrhythmias and require monitoring' if pac_count > 5 else 'Occasional PACs are common and usually benign'}."
        }
    
    def _generate_pac_summary(self, is_present: bool, pac_count: int) -> Dict[str, Any]:
        """Generate summary for PAC analysis."""
        if is_present:
            detection_text = f"detected {pac_count} PAC events"
            morphology_text = "Ectopic atrial beats show characteristic early P-waves, narrow QRS complexes, and non-compensatory pauses"
        else:
            detection_text = "did not detect any PAC events"
            morphology_text = "All beats appear to originate from normal sinus node without atrial ectopy"
        
        summary_text = f"Premature atrial complex analysis {detection_text}. {morphology_text}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': int(pac_count),
        }
    
    def _generate_pac_title(self, is_present: bool) -> str:
        """Generate title for PAC analysis."""
        if is_present:
            return "Premature Atrial Complexes"
        else:
            return "No PAC Detected"
    
    def generate_avb_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for First-Degree AV Block (AVB) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        mean_pr = metrics.get('pr_mean_s', 0)
        min_pr = metrics.get('pr_min_s', 0)
        max_pr = metrics.get('pr_max_s', 0)
        prolonged_count = metrics.get('prolonged_pr_count', 0)
        total_count = metrics.get('total_pr_count', 0)
        
        # Determine if AVB is present (PR > 0.20s)
        is_present = bool(mean_pr > 0.20)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: PR Interval Assessment
        pr_section = self._generate_avb_pr_section(is_present, mean_pr, min_pr, max_pr)
        sections.append(pr_section)
        
        # Section 2: Conduction Analysis
        conduction_section = self._generate_avb_conduction_section(is_present, prolonged_count, total_count)
        sections.append(conduction_section)
        
        # Generate summary
        summary = self._generate_avb_summary(is_present, mean_pr, prolonged_count, total_count)
        
        # Generate title
        title = self._generate_avb_title(is_present)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_avb_pr_section(self, is_present: bool, mean_pr: float, min_pr: float, max_pr: float) -> Dict[str, str]:
        """Generate PR interval assessment section for AVB analysis."""
        if is_present:
            pr_status = "Prolonged"
            pr_description = "prolonged PR intervals"
            pr_explanation = f"indicating delayed AV conduction with mean duration of {mean_pr:.3f}s (normal < 0.20s)"
        else:
            pr_status = "Normal"
            pr_description = "normal PR intervals"
            pr_explanation = f"indicating normal AV conduction with mean duration of {mean_pr:.3f}s"
        
        return {
            "section_title": f"1. PR Interval Assessment: {pr_status} (Mean: {mean_pr:.3f}s)",
            "content": f"PR interval analysis shows {pr_description}, {pr_explanation}. The PR interval range is {min_pr:.3f}s to {max_pr:.3f}s, as visualized in the [PR Interval Distribution] chart."
        }
    
    def _generate_avb_conduction_section(self, is_present: bool, prolonged_count: int, total_count: int) -> Dict[str, str]:
        """Generate conduction analysis section for AVB analysis."""
        if is_present:
            conduction_status = "First-Degree AV Block"
            conduction_description = f"significant delay in AV conduction"
            conduction_explanation = f"{prolonged_count} out of {total_count} intervals exceed the diagnostic threshold (>0.20s), confirming first-degree AV block"
        else:
            conduction_status = "Normal AV Conduction"
            conduction_description = "normal AV conduction"
            conduction_explanation = f"all {total_count} intervals are within normal limits, indicating normal AV conduction"
        
        return {
            "section_title": f"2. Conduction Analysis: {conduction_status}",
            "content": f"AV conduction analysis shows {conduction_description}. {conduction_explanation}. This is demonstrated in the [Enhanced AVB Analysis] chart showing the relationship between P-waves and QRS complexes."
        }
    
    def _generate_avb_summary(self, is_present: bool, mean_pr: float, prolonged_count: int, total_count: int) -> Dict[str, Any]:
        """Generate summary for AVB analysis."""
        if is_present:
            pr_desc = "prolonged PR intervals"
            confirmation = f"{prolonged_count} out of {total_count} intervals exceed the diagnostic threshold, confirming first-degree AV block"
        else:
            pr_desc = "normal PR intervals"
            confirmation = f"indicating normal AV conduction"
        
        summary_text = f"First-degree AV block analysis shows {pr_desc} with mean duration of {mean_pr:.3f} seconds. {confirmation}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': int(prolonged_count if is_present else 0),
        }
    
    def _generate_avb_title(self, is_present: bool) -> str:
        """Generate title for AVB analysis."""
        if is_present:
            return "First-Degree AV Block"
        else:
            return "Normal AV Conduction"
    
    def generate_lbbb_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Left Bundle Branch Block (LBBB) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        qrs_mean = metrics.get('qrs_mean_s', 0)
        wide_qrs_count = metrics.get('wide_qrs_count', 0)
        total_qrs_count = metrics.get('total_qrs_count', 0)
        
        # Determine if LBBB is present (QRS > 0.12s)
        is_present = bool(qrs_mean and qrs_mean >= 0.12)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: QRS Duration Assessment
        qrs_section = self._generate_lbbb_qrs_section(is_present, qrs_mean)
        sections.append(qrs_section)
        
        # Section 2: Morphological Assessment
        morph_section = self._generate_lbbb_morphology_section(is_present, wide_qrs_count, total_qrs_count)
        sections.append(morph_section)
        
        # Generate summary
        summary = self._generate_lbbb_summary(is_present, qrs_mean, wide_qrs_count, total_qrs_count)
        
        # Generate title
        title = self._generate_lbbb_title(is_present)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_lbbb_qrs_section(self, is_present: bool, qrs_mean: float) -> Dict[str, str]:
        """Generate QRS duration assessment section for LBBB analysis."""
        if is_present:
            qrs_status = "Prolonged"
            qrs_description = "prolonged QRS duration"
            qrs_explanation = f"indicating delayed left ventricular activation with mean duration of {qrs_mean:.3f}s (>0.12s threshold)"
        else:
            qrs_status = "Normal"
            qrs_description = "normal QRS duration"
            qrs_explanation = f"indicating normal ventricular conduction with mean duration of {qrs_mean:.3f}s"
        
        return {
            "section_title": f"1. QRS Duration Assessment: {qrs_status} (Mean: {qrs_mean:.3f}s)",
            "content": f"QRS complex duration analysis shows {qrs_description}, {qrs_explanation}. This is visualized in the [QRS Duration Distribution] chart with the diagnostic threshold clearly marked."
        }
    
    def _generate_lbbb_morphology_section(self, is_present: bool, wide_qrs_count: int, total_qrs_count: int) -> Dict[str, str]:
        """Generate morphological assessment section for LBBB analysis."""
        if is_present:
            morph_status = "LBBB Pattern"
            morph_description = "characteristic LBBB morphology"
            morph_explanation = f"{wide_qrs_count} out of {total_qrs_count} complexes show wide QRS morphology consistent with left bundle branch block"
        else:
            morph_status = "Normal Morphology"
            morph_description = "normal QRS morphology"
            morph_explanation = f"all {total_qrs_count} complexes maintain normal QRS morphology"
        
        return {
            "section_title": f"2. Morphological Assessment: {morph_status}",
            "content": f"Morphological analysis shows {morph_description}. {morph_explanation}. The [QRS Analysis] chart demonstrates the characteristic features of left bundle branch block when present."
        }
    
    def _generate_lbbb_summary(self, is_present: bool, qrs_mean: float, wide_qrs_count: int, total_qrs_count: int) -> Dict[str, Any]:
        """Generate summary for LBBB analysis."""
        if is_present:
            pattern_desc = "LBBB pattern"
            morphology_desc = f"{wide_qrs_count} out of {total_qrs_count} complexes show wide QRS morphology"
        else:
            pattern_desc = "normal conduction"
            morphology_desc = f"QRS complexes maintain normal duration and morphology"
        
        summary_text = f"Left bundle branch block analysis shows {pattern_desc} with mean QRS duration of {qrs_mean:.3f} seconds. {morphology_desc}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': int(wide_qrs_count) if is_present else 0,
        }
    
    def _generate_lbbb_title(self, is_present: bool) -> str:
        """Generate title for LBBB analysis."""
        if is_present:
            return "Left Bundle Branch Block"
        else:
            return "Normal Conduction"
    
    def generate_avb2_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Second-Degree AV Block (AVB2) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        dropped_count = metrics.get('dropped_beats', 0)
        subtype = metrics.get('subtype', '')
        pr_variation = metrics.get('pr_variation', 0)
        total_beats = metrics.get('total_beats', 0)
        
        # Determine if AVB2 is present
        is_present = bool(dropped_count >= 1)
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Conduction Failure Assessment
        failure_section = self._generate_avb2_failure_section(is_present, dropped_count)
        sections.append(failure_section)
        
        # Section 2: Subtype Classification
        subtype_section = self._generate_avb2_subtype_section(is_present, subtype, pr_variation)
        sections.append(subtype_section)
        
        # Generate summary
        summary = self._generate_avb2_summary(is_present, dropped_count, subtype)
        
        # Generate title
        title = self._generate_avb2_title(is_present, subtype)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_avb2_failure_section(self, is_present: bool, dropped_count: int) -> Dict[str, str]:
        """Generate conduction failure assessment section for AVB2 analysis."""
        if is_present:
            failure_status = f"Detected ({dropped_count})"
            failure_description = f"significant AV conduction failure"
            failure_explanation = f"{dropped_count} dropped beat{'s' if dropped_count != 1 else ''} detected, confirming 2nd-degree AV block"
        else:
            failure_status = "Not Detected"
            failure_description = "no significant AV conduction abnormalities"
            failure_explanation = "no dropped beats detected, indicating normal AV conduction"
        
        return {
            "section_title": f"1. Conduction Failure Assessment: {failure_status}",
            "content": f"AV conduction analysis shows {failure_description}. This is evidenced by {failure_explanation}. The [ECG Overview] chart highlights areas where P-waves are not followed by QRS complexes."
        }
    
    def _generate_avb2_subtype_section(self, is_present: bool, subtype: str, pr_variation: float) -> Dict[str, str]:
        """Generate subtype classification section for AVB2 analysis."""
        if is_present:
            if subtype == 'mobitz_ii':
                subtype_status = "Mobitz Type II"
                subtype_description = "constant PR intervals before dropped beats"
                subtype_explanation = "indicating infranodal conduction block"
            elif subtype == 'mobitz_i':
                subtype_status = "Mobitz Type I (Wenckebach)"
                subtype_description = "progressive PR interval lengthening"
                subtype_explanation = "indicating nodal conduction block"
            else:
                subtype_status = "Unspecified Type"
                subtype_description = "AV conduction block"
                subtype_explanation = "requiring further characterization"
        else:
            subtype_status = "Normal AV Conduction"
            subtype_description = "normal AV conduction pattern"
            subtype_explanation = "no evidence of conduction block"
        
        return {
            "section_title": f"2. Subtype Classification: {subtype_status}",
            "content": f"AV block classification shows {subtype_description}. This pattern is {subtype_explanation}. The [Enhanced AVB Analysis] chart demonstrates the relationship between P-waves and QRS complexes."
        }
    
    def _generate_avb2_summary(self, is_present: bool, dropped_count: int, subtype: str) -> Dict[str, Any]:
        """Generate summary for AVB2 analysis."""
        if is_present:
            detection_text = f"confirmed the presence of 2nd-degree AV block"
            subtype_text = f" ({subtype.replace('_', ' ').title()})" if subtype else ""
            detail_text = f"with {dropped_count} dropped beat{'s' if dropped_count != 1 else ''} detected"
        else:
            detection_text = "did not detect significant AV conduction abnormalities"
            subtype_text = ""
            detail_text = "no dropped beats detected"
        
        summary_text = f"Analysis {detection_text}{subtype_text}, {detail_text}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': int(dropped_count) if is_present else 0,
        }
    
    def _generate_avb2_title(self, is_present: bool, subtype: str) -> str:
        """Generate title for AVB2 analysis."""
        if is_present:
            if subtype == 'mobitz_ii':
                return "Second-Degree AV Block (Mobitz Type II)"
            elif subtype == 'mobitz_i':
                return "Second-Degree AV Block (Mobitz Type I)"
            else:
                return "Second-Degree AV Block"
        else:
            return "Normal AV Conduction"
    
    def generate_avb3_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Third-Degree AV Block (AVB3) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        diagnosis = metrics.get('diagnosis', '')
        confidence = metrics.get('confidence', 0)
        atrial_rate = metrics.get('atrial_rate', 0)
        ventricular_rate = metrics.get('ventricular_rate', 0)
        
        # Determine if AVB3 is present
        is_present = bool(diagnosis.startswith('Complete') or diagnosis.endswith('AV Block'))
        
        # Generate structured report sections
        sections = []
        
        # Section 1: AV Dissociation Assessment
        dissociation_section = self._generate_avb3_dissociation_section(is_present, diagnosis)
        sections.append(dissociation_section)
        
        # Section 2: Rhythm Analysis
        rhythm_section = self._generate_avb3_rhythm_section(is_present, atrial_rate, ventricular_rate)
        sections.append(rhythm_section)
        
        # Generate summary
        summary = self._generate_avb3_summary(is_present, diagnosis, confidence)
        
        # Generate title
        title = self._generate_avb3_title(is_present, diagnosis)
        
        # Centralized compact statistics for AVB3 (validators-compatible keys included)
        stats_compact = {
            'diagnosis': diagnosis,
            'confidence': float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            # validators expect pp_rate_bpm/rr_rate_bpm
            'pp_rate_bpm': metrics.get('pp_rate_bpm') or metrics.get('atrial_rate'),
            'rr_rate_bpm': metrics.get('rr_rate_bpm') or metrics.get('ventricular_rate'),
            # keep original for completeness
            'atrial_rate': metrics.get('atrial_rate'),
            'ventricular_rate': metrics.get('ventricular_rate'),
            'conduction_ratio': metrics.get('conduction_ratio'),
            'dissociation_score': metrics.get('dissociation_score'),
            'complete_av_block': metrics.get('complete_av_block'),
            # validators expect pr_cv and diagnosis_confidence if available
            'pr_cv': metrics.get('pr_cv'),
            'diagnosis_confidence': metrics.get('diagnosis_confidence') or confidence,
        }

        # Optional: finalize summary with standardized diagnosis tag
        if isinstance(summary, dict):
            summary['final_diagnosis'] = 'Complete AV Block (3rd-degree)' if is_present else 'Not 3rd-degree AV Block'

        # Optional centralized image path mapping if provided in metrics
        image_paths = None
        if isinstance(metrics.get('generated_images'), list) and metrics.get('output_dir'):
            base_dir = metrics.get('output_dir')
            image_paths = {name: os.path.join(base_dir, name) for name in metrics['generated_images']}

        result = {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': stats_compact,
            'validation': validation_result
        }

        # Attach lead_name and image_paths when available
        if metrics.get('lead_name'):
            result['lead_name'] = metrics.get('lead_name')
        if image_paths:
            result['image_paths'] = image_paths

        return result
    
    def _generate_avb3_dissociation_section(self, is_present: bool, diagnosis: str) -> Dict[str, str]:
        """Generate AV dissociation assessment section for AVB3 analysis."""
        if is_present:
            dissociation_status = "Complete AV Dissociation"
            dissociation_description = "complete atrioventricular dissociation"
            dissociation_explanation = "atrial and ventricular rhythms are completely independent"
        else:
            dissociation_status = "No AV Dissociation"
            dissociation_description = "no evidence of complete AV dissociation"
            dissociation_explanation = "atrial and ventricular rhythms maintain some degree of coordination"
        
        return {
            "section_title": f"1. AV Dissociation Assessment: {dissociation_status}",
            "content": f"Third-degree AV block analysis shows {dissociation_description}. This indicates that {dissociation_explanation}. The [ECG Overview] chart demonstrates the independent nature of atrial and ventricular activity."
        }
    
    def _generate_avb3_rhythm_section(self, is_present: bool, atrial_rate: float, ventricular_rate: float) -> Dict[str, str]:
        """Generate rhythm analysis section for AVB3 analysis."""
        if is_present:
            rhythm_status = "Independent Rhythms"
            rhythm_description = f"independent atrial ({atrial_rate:.1f} bpm) and ventricular ({ventricular_rate:.1f} bpm) rhythms"
            rhythm_explanation = "ventricular rhythm is typically slower and completely dissociated from atrial activity"
        else:
            rhythm_status = "Coordinated Rhythms"
            rhythm_description = "coordinated atrial and ventricular activity"
            rhythm_explanation = "some degree of AV conduction is maintained"
        
        return {
            "section_title": f"2. Rhythm Analysis: {rhythm_status}",
            "content": f"Rhythm analysis demonstrates {rhythm_description}. This shows that {rhythm_explanation}. The [Enhanced AVB Analysis] chart illustrates the relationship between P-waves and QRS complexes."
        }
    
    def _generate_avb3_summary(self, is_present: bool, diagnosis: str, confidence: float) -> Dict[str, Any]:
        """Generate summary for AVB3 analysis."""
        if is_present:
            diagnosis_text = f"{diagnosis} with confidence level of {confidence:.2f}"
            detail_text = "Complete AV dissociation confirmed with independent atrial and ventricular rhythms"
        else:
            diagnosis_text = f"{diagnosis}"
            detail_text = "Insufficient evidence for complete AV block diagnosis"
        
        summary_text = f"Third-degree AV block analysis shows {diagnosis_text}. {detail_text}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': 1 if is_present else 0,
        }
    
    def _generate_avb3_title(self, is_present: bool, diagnosis: str) -> str:
        """Generate title for AVB3 analysis."""
        if is_present:
            return "Third-Degree AV Block"
        else:
            return "Normal AV Conduction"
    
    def generate_aflt_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Atrial Flutter (AFLT) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        dominant_rate = metrics.get('dominant_rate', 0)
        av_ratio = metrics.get('av_ratio', 0)
        atrial_rate = metrics.get('atrial_rate', 0)
        power_ratio = metrics.get('power_ratio', 0)
        
        # Determine if AFLT is present (rate 230-350 bpm, power ratio > 0.6, AV block)
        is_present = bool(
            (230 <= dominant_rate <= 350) and 
            (power_ratio > 0.6) and 
            (av_ratio >= 2)
        )
        
        # Generate structured report sections
        sections = []
        
        # Section 1: Atrial Rate Assessment
        rate_section = self._generate_aflt_rate_section(is_present, dominant_rate, atrial_rate)
        sections.append(rate_section)
        
        # Section 2: AV Conduction Analysis
        conduction_section = self._generate_aflt_conduction_section(is_present, av_ratio, power_ratio)
        sections.append(conduction_section)
        
        # Generate summary
        summary = self._generate_aflt_summary(is_present, dominant_rate, av_ratio)
        
        # Generate title
        title = self._generate_aflt_title(is_present)
        
        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': metrics,
            'validation': validation_result
        }
    
    def _generate_aflt_rate_section(self, is_present: bool, dominant_rate: float, atrial_rate: float) -> Dict[str, str]:
        """Generate atrial rate assessment section for AFLT analysis."""
        if is_present:
            rate_status = "Flutter Rate"
            rate_description = f"typical flutter pattern with dominant rate of {dominant_rate:.0f} bpm"
            rate_explanation = f"atrial rate of {atrial_rate:.0f} bpm falls within the flutter range (230-350 bpm)"
        else:
            rate_status = "Normal Atrial Rate"
            rate_description = "normal atrial rhythm"
            rate_explanation = f"atrial rate of {atrial_rate:.0f} bpm is outside the flutter range"
        
        return {
            "section_title": f"1. Atrial Rate Assessment: {rate_status}",
            "content": f"Atrial flutter analysis shows {rate_description}. This indicates that {rate_explanation}. The [Atrial Rate Analysis] chart demonstrates the frequency characteristics of atrial activity."
        }
    
    def _generate_aflt_conduction_section(self, is_present: bool, av_ratio: float, power_ratio: float) -> Dict[str, str]:
        """Generate AV conduction analysis section for AFLT analysis."""
        if is_present:
            conduction_status = "AV Block Present"
            conduction_description = f"AV conduction ratio of {av_ratio:.0f}:1 with power ratio {power_ratio:.2f}"
            conduction_explanation = "indicating typical flutter with AV block and regular atrial activity"
        else:
            conduction_status = "Normal AV Conduction"
            conduction_description = "normal AV conduction pattern"
            conduction_explanation = "no evidence of flutter-related conduction block"
        
        return {
            "section_title": f"2. AV Conduction Analysis: {conduction_status}",
            "content": f"AV conduction analysis shows {conduction_description}. This pattern is {conduction_explanation}. The [Multi-lead Analysis] chart illustrates the relationship between atrial and ventricular activity."
        }
    
    def _generate_aflt_summary(self, is_present: bool, dominant_rate: float, av_ratio: float) -> Dict[str, Any]:
        """Generate summary for AFLT analysis."""
        if is_present:
            pattern_desc = "typical flutter pattern"
            criteria_desc = "Clinical criteria for atrial flutter are met"
        else:
            pattern_desc = "normal atrial rhythm"
            criteria_desc = "No evidence of atrial flutter detected"
        
        summary_text = f"Atrial flutter analysis shows {pattern_desc} with dominant rate of {dominant_rate:.0f} bpm and AV conduction ratio of {av_ratio:.0f}:1. {criteria_desc}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': 1 if is_present else 0,
        }
    
    def _generate_aflt_title(self, is_present: bool) -> str:
        """Generate title for AFLT analysis."""
        if is_present:
            return "Atrial Flutter"
        else:
            return "Normal Atrial Rhythm"
    
    def generate_rbbb_report(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report for Right Bundle Branch Block (RBBB) analysis.
        
        Args:
            metrics: Dictionary containing analysis metrics
            validation_result: Dictionary containing validation results
            
        Returns:
            Dictionary containing complete report structure
        """
        qrs_mean = metrics.get('qrs_mean_s', 0)
        wide_qrs_count = metrics.get('wide_qrs_count', 0)
        total_qrs_count = metrics.get('total_qrs_count', 0)
        rsr_detected = metrics.get('rsr_detected', False)
        wide_s_detected = metrics.get('wide_s_detected', False)
        
        # Determine if RBBB is present (QRS > 0.12s + morphological criteria)
        is_present = bool(
            qrs_mean >= 0.12 and 
            rsr_detected and 
            wide_s_detected
        )
        
        # Generate structured report sections
        sections = []
        
        # Section 1: QRS Duration Assessment
        qrs_section = self._generate_rbbb_qrs_section(is_present, qrs_mean)
        sections.append(qrs_section)
        
        # Section 2: Morphological Assessment
        morph_section = self._generate_rbbb_morphology_section(is_present, rsr_detected, wide_s_detected)
        sections.append(morph_section)
        
        # Generate summary
        summary = self._generate_rbbb_summary(is_present, qrs_mean, rsr_detected, wide_s_detected)
        
        # Generate title
        title = self._generate_rbbb_title(is_present)
        
        # Enrich summary with optional centralized fields
        final_rbbb = bool(is_present)
        rsr_detected_opt = metrics.get('rsr_detected')
        wide_s_detected_opt = metrics.get('wide_s_detected')
        if isinstance(summary, dict):
            # final diagnosis
            if final_rbbb:
                summary['final_diagnosis'] = 'Complete RBBB'
            else:
                summary['final_diagnosis'] = 'RBBB not confirmed'
            # morphology findings (if available)
            if rsr_detected_opt is not None or wide_s_detected_opt is not None:
                summary['morphology_findings'] = {
                    'v1_v2_rsr_pattern': bool(rsr_detected_opt) if rsr_detected_opt is not None else False,
                    'lateral_wide_s_wave': bool(wide_s_detected_opt) if wide_s_detected_opt is not None else False,
                }

        # Centralized compact statistics structure (includes validator-compatible keys)
        qrs_mean_val = metrics.get('qrs_mean_s')
        wide_prop_val = metrics.get('wide_qrs_proportion')
        total_beats_val = metrics.get('total_qrs_count') if metrics.get('total_qrs_count') is not None else metrics.get('total_beats')
        stats_compact = {
            'total_beats': int(total_beats_val) if isinstance(total_beats_val, (int, float)) else 0,
            'qrs_mean_s': float(qrs_mean_val) if isinstance(qrs_mean_val, (int, float)) else None,
            'wide_qrs_proportion': float(wide_prop_val) if isinstance(wide_prop_val, (int, float)) else None,
            'morphology_counts': {
                'rsr_beat_count': int(metrics.get('rsr_beats_count', 0) or 0),
                'wide_s_wave_beat_count': int(metrics.get('deep_s_beats_count', 0) or 0),
            },
            'qrs_category_counts': metrics.get('qrs_category_counts', {}),
            # compatibility fields expected by validators.validate_rbbb
            'rsr_pattern_detected': bool(metrics.get('rsr_detected')) if metrics.get('rsr_detected') is not None else None,
            'wide_s_wave_detected': bool(metrics.get('wide_s_detected')) if metrics.get('wide_s_detected') is not None else None,
        }

        return {
            'title': f'ECG Analysis Report: {title}',
            'summary': summary,
            'report_text': sections,
            'statistics': stats_compact,
            'validation': validation_result
        }
    
    def _generate_rbbb_qrs_section(self, is_present: bool, qrs_mean: float) -> Dict[str, str]:
        """Generate QRS duration assessment section for RBBB analysis."""
        if is_present:
            qrs_status = "Prolonged"
            qrs_description = "prolonged QRS duration"
            qrs_explanation = f"indicating delayed right ventricular activation with mean duration of {qrs_mean:.3f}s (>0.12s threshold)"
        else:
            qrs_status = "Normal"
            qrs_description = "normal QRS duration"
            qrs_explanation = f"indicating normal ventricular conduction with mean duration of {qrs_mean:.3f}s"
        
        return {
            "section_title": f"1. QRS Duration Assessment: {qrs_status} (Mean: {qrs_mean:.3f}s)",
            "content": f"QRS complex duration analysis shows {qrs_description}, {qrs_explanation}. This is visualized in the [QRS Duration Distribution] chart with the diagnostic threshold clearly marked."
        }
    
    def _generate_rbbb_morphology_section(self, is_present: bool, rsr_detected: bool, wide_s_detected: bool) -> Dict[str, str]:
        """Generate morphological assessment section for RBBB analysis."""
        if is_present:
            morph_status = "RBBB Pattern"
            morph_description = "characteristic RBBB morphology"
            morph_explanation = "RSR' pattern in V1 and wide S waves in lateral leads, consistent with right bundle branch block"
        else:
            morph_status = "Normal/Incomplete Pattern"
            morph_description = "normal or incomplete RBBB pattern"
            morph_explanation = "morphological criteria for RBBB are not fully met"
        
        return {
            "section_title": f"2. Morphological Assessment: {morph_status}",
            "content": f"Morphological analysis shows {morph_description}. This indicates that {morph_explanation}. The [QRS Analysis] chart demonstrates the characteristic features of right bundle branch block when present."
        }
    
    def _generate_rbbb_summary(self, is_present: bool, qrs_mean: float, rsr_detected: bool, wide_s_detected: bool) -> Dict[str, Any]:
        """Generate summary for RBBB analysis."""
        if is_present:
            pattern_desc = "RBBB pattern"
            criteria_desc = "Met"
        else:
            pattern_desc = "normal or incomplete pattern"
            criteria_desc = "Not fully met"
        
        summary_text = f"Right bundle branch block analysis shows {pattern_desc} with mean QRS duration of {qrs_mean:.3f} seconds. Morphological criteria: {criteria_desc}."
        
        return {
            'report_text': summary_text,
            'is_abnormality_present': is_present,
            'abnormality_count': 1 if is_present else 0,
        }
    
    def _generate_rbbb_title(self, is_present: bool) -> str:
        """Generate title for RBBB analysis."""
        if is_present:
            return "Right Bundle Branch Block"
        else:
            return "Normal Conduction"
    
    # Template methods for future expansion
    def _heart_rate_template(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Template for heart rate assessment sections."""
        pass
    
    def _rhythm_analysis_template(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Template for rhythm analysis sections."""
        pass
    
    def _morphology_template(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Template for morphology assessment sections."""
        pass
    
    def _conduction_template(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Template for conduction assessment sections."""
        pass


# Legacy functions for backward compatibility
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