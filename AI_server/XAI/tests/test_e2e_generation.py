import json
import os

from main import main as run_main


def _assert_files_exist(base_dir, relative_paths):
    for rel in relative_paths:
        p = os.path.join(base_dir, rel)
        assert os.path.exists(p), f"Missing expected output: {rel}"


def test_e2e_dicom_tasks(tmp_path):
    # Use available sample DICOMs in XAI root
    out_dir = tmp_path / "e2e_outputs"
    cfg = {
        "output_dir": str(out_dir),
        "tasks": [
            {"abnormality": "AF", "filepath": "00001_lr.dcm", "lead_index": "II"},
            {"abnormality": "NR", "filepath": "00001_lr.dcm", "lead_index": "II"},
            {"abnormality": "1dAVB", "filepath": "00234_lr.dcm", "lead_index": "Lead II"},
        ],
    }

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Run pipeline
    run_main(str(cfg_path))

    # AF outputs
    af_dir = out_dir / "00001_lr_AF"
    _assert_files_exist(af_dir, [
        "analysis_report.json",
        "afib_ecg_with_rpeaks.png",
        "afib_rr_histogram.png",
        "afib_pwave_regions.png",
    ])

    # NR outputs (allow optional QRS histogram)
    nr_dir = out_dir / "00001_lr_NR"
    assert os.path.exists(nr_dir / "analysis_report.json")
    assert os.path.exists(nr_dir / "normal_ecg_overview.png")
    assert os.path.exists(nr_dir / "normal_rr_histogram.png")

    # 1dAVB outputs
    avb_dir = out_dir / "00234_lr_1dAVB"
    _assert_files_exist(avb_dir, [
        "analysis_report.json",
        "avb1_pr_intervals_corrected.png",
        "avb1_pr_histogram.png",
    ])

    # Results index at root
    assert os.path.exists(out_dir / "results_index.json")


