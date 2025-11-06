#!/usr/bin/env python3
"""
Comprehensive XAI Testing Script
Tests all XAI modules with sample data to ensure functionality.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the XAI directory (parent of tests) to Python path so we can import xai module
XAI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if XAI_ROOT not in sys.path:
    sys.path.insert(0, XAI_ROOT)

def test_all_xai_modules():
    """Test all XAI modules with synthetic data."""
    
    print("XAI Comprehensive Testing")
    print("=" * 50)
    
    # Test results
    results = {}
    
    # Create test output directory
    test_output_dir = "test_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Test data
    sampling_rate = 500
    duration = 10
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Create synthetic ECG signal
    heart_rate = 75
    rr_interval = 60.0 / heart_rate
    r_peaks = np.arange(0, duration, rr_interval)
    
    ecg_signal = np.zeros_like(t)
    for r_peak in r_peaks:
        if r_peak < duration:
            peak_idx = int(r_peak * sampling_rate)
            if peak_idx < len(ecg_signal):
                # Simple R-peak shape
                for i in range(-5, 6):
                    idx = peak_idx + i
                    if 0 <= idx < len(ecg_signal):
                        ecg_signal[idx] += np.exp(-(i/3)**2) * 0.5
    
    # Add noise
    ecg_signal += np.random.normal(0, 0.05, len(ecg_signal))
    
    # Test modules
    modules_to_test = [
        ("ST", "xai.explanations.st", "generate_st_visuals"),
        ("SB", "xai.explanations.sb", "generate_sb_visuals"),
        ("NORM", "xai.explanations.normal", "generate_normal_visuals"),
        ("AF", "xai.explanations.af", "generate_af_visuals"),
        ("PVC", "xai.explanations.pvc", "generate_pvc_visuals"),
        ("PAC", "xai.explanations.pac", "generate_pac_visuals"),
        ("AVB", "xai.explanations.avb", "generate_avb_visuals"),
        ("AVB2", "xai.explanations.avb2", "generate_avb2_visuals"),
        ("AVB3", "xai.explanations.avb3", "generate_avb3_visuals"),
        ("AFLT", "xai.explanations.aflt", "generate_aflt_visuals"),
    ]
    
    for module_name, module_path, function_name in modules_to_test:
        try:
            print(f"\nTesting {module_name} module...")
            
            # Import module
            module = __import__(module_path, fromlist=[function_name])
            func = getattr(module, function_name)
            
            # Test function
            if module_name in ["RBBB", "LBBB"]:
                # Multi-lead modules need special handling
                print(f"  Skipping {module_name} - requires multi-lead data")
                results[module_name] = "SKIPPED"
            else:
                # Single-lead modules
                report_data = func(
                    ecg_signal=ecg_signal,
                    sampling_rate=sampling_rate,
                    ecg_title=f"Test {module_name}",
                    lead_name="Lead II",
                    output_dir=test_output_dir
                )
                
                if report_data:
                    print(f"  [OK] {module_name} analysis completed")
                    print(f"    Title: {report_data.get('title', 'N/A')}")
                    print(f"    Lead: {report_data.get('lead_name', 'N/A')}")
                    print(f"    Summary: {report_data.get('summary', {}).get('report_text', 'N/A')[:100]}...")
                    print(f"    Abnormality present: {report_data.get('summary', {}).get('is_abnormality_present', 'N/A')}")
                    print(f"    Statistics keys: {list(report_data.get('statistics', {}).keys())}")
                    print(f"    Report sections: {len(report_data.get('report_text', []))}")
                    results[module_name] = "PASS"
                else:
                    print(f"  [FAIL] {module_name} returned None")
                    results[module_name] = "FAIL"
                    
        except Exception as e:
            print(f"  [ERROR] {module_name} failed: {e}")
            results[module_name] = f"ERROR: {str(e)}"
    
    # Test ReportGenerator
    print(f"\nTesting ReportGenerator...")
    try:
        from xai.clinical_reports import ReportGenerator
        
        report_generator = ReportGenerator()
        
        # Test ST report generation
        metrics = {
            'average_heart_rate_bpm': 85,
            'tachy_ratio': 0.3,
            'max_instantaneous_hr': 120
        }
        validation_result = {
            'status': 'pass',
            'reasons': ['Test validation'],
            'metrics_used': {'tachy_ratio': 0.3}
        }
        
        st_report = report_generator.generate_st_report(metrics, validation_result)
        
        if st_report:
            print(f"  [OK] ReportGenerator ST report generated")
            print(f"    Title: {st_report.get('title', 'N/A')}")
            print(f"    Summary: {st_report.get('summary', {}).get('report_text', 'N/A')[:100]}...")
            print(f"    Report sections: {len(st_report.get('report_text', []))}")
            results["ReportGenerator"] = "PASS"
        else:
            print(f"  [FAIL] ReportGenerator returned None")
            results["ReportGenerator"] = "FAIL"
            
    except Exception as e:
        print(f"  [ERROR] ReportGenerator failed: {e}")
        results["ReportGenerator"] = f"ERROR: {str(e)}"
    
    # Summary
    print(f"\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for module, result in results.items():
        if result == "PASS":
            print(f"{module}: [OK] PASS")
            passed += 1
        elif result == "SKIPPED":
            print(f"{module}: [SKIP] SKIPPED")
            skipped += 1
        else:
            print(f"{module}: [FAIL] {result}")
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\nAll tests passed! XAI modules are working correctly.")
    else:
        print(f"\n{failed} tests failed. Please check the errors above.")
    
    return results

if __name__ == "__main__":
    test_all_xai_modules()

