# main.py

import json
import os
import time
from xai.ecg_utils import load_ecg_data
from xai.result_io import write_results_index
from xai.explanations.validators import (
    validate_af, validate_st, validate_sb, validate_avb1, validate_lbbb, validate_rbbb, validate_nr
)
from xai.explanations.normal import generate_normal_visuals
from xai.explanations.sb import generate_sb_visuals
from xai.explanations.af import generate_af_visuals
from xai.explanations.st import generate_st_visuals
from xai.explanations.lbbb import generate_lbbb_visuals
from xai.explanations.rbbb import generate_rbbb_visuals
from xai.explanations.avb import generate_avb_visuals

VISUALIZATION_DISPATCHER = {
    "AF": generate_af_visuals,
    "ST": generate_st_visuals,
    "LBBB": generate_lbbb_visuals,
    "RBBB": generate_rbbb_visuals,
    "1dAVB": generate_avb_visuals,
    "NR": generate_normal_visuals,
    "SB": generate_sb_visuals,
}

# main.py 

def run_xai_tasks(
    tasks,
    output_dir=None,
    default_dicom_dir=None,
    default_db_dir=None,
    default_lead_index=None,
    write_index=True,
    return_reports=True,
    verbose=True,
):
    """
    Run ECG visualizations directly with in-memory tasks and return structured results.

    Parameters
    - tasks: list[dict]. Preferred keys per task:
        - 'abnormality': AF/ST/LBBB/RBBB/1dAVB/NR/SB
        - 'instance_id': unified identifier
            * DICOM: resolved via task['dicom_dir'] or default_dicom_dir → <dicom_dir>/<instance_id>.dcm
            * WFDB: resolved via task['db_dir'] or default_db_dir as record name base
        - Optional compatibility keys: 'filepath' (DICOM path), 'record_name' (WFDB record)
        - Optional: 'lead_index', 'dicom_dir', 'db_dir'
    - output_dir: output base directory; default XAI/data/ecg_analysis_results
    - default_dicom_dir: fallback DICOM directory when task does not specify
    - default_db_dir: fallback WFDB base directory when task does not specify
    - default_lead_index: fallback lead index when task does not specify
    - write_index: write a unified results index JSON
    - return_reports: include full report JSON in return payload
    - verbose: print progress logs

    Returns
    - dict with keys: 'output_dir', 'index_path' (optional), 'results' (list of per-task results)
    """
    default_out = os.path.join(os.path.dirname(__file__), "data", "ecg_analysis_results")
    output_dir = output_dir or default_out

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for i, task in enumerate(tasks):
        # Merge defaults
        merged_task = dict(task or {})
        if default_dicom_dir and not merged_task.get("dicom_dir"):
            merged_task["dicom_dir"] = default_dicom_dir
        if default_db_dir and not merged_task.get("db_dir"):
            merged_task["db_dir"] = default_db_dir
        if default_lead_index is not None and merged_task.get("lead_index") is None:
            merged_task["lead_index"] = default_lead_index

        abnormality = merged_task.get("abnormality")
        # Unified display identifier: prioritize instance_id, then filepath/record_name
        record_identifier = (
            merged_task.get("instance_id")
            or merged_task.get("filepath")
            or merged_task.get("record_name", "Unknown_Record")
        )

        if record_identifier == "Unknown_Record":
            if verbose:
                print(f"Skipping task {i+1}: Missing 'filepath' or 'record_name'.")
            results.append({
                "status": "skipped",
                "reason": "missing_identifier",
                "abnormality": abnormality,
                "task": task,
            })
            continue

        if verbose:
            print(f"\n--- Processing Task {i+1}: {abnormality} for {record_identifier} ---")

        start_time = time.perf_counter()
        base_filename = os.path.splitext(os.path.basename(str(record_identifier)))[0]
        task_output_dir = os.path.join(output_dir, f"{base_filename}_{abnormality}")
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)

        visualization_func = VISUALIZATION_DISPATCHER.get(abnormality)
        if not visualization_func:
            if verbose:
                print(f"Warning: No function for '{abnormality}'.")
            results.append({
                "status": "skipped",
                "reason": "unsupported_abnormality",
                "abnormality": abnormality,
                "record_identifier": record_identifier,
                "task_output_dir": task_output_dir,
            })
            continue

        report_data = None
        error = None

        try:
            if abnormality == "RBBB":
                report_data = visualization_func(
                    record_name=merged_task.get("instance_id") or merged_task.get("record_name"),
                    db_dir=merged_task.get("db_dir"),
                    output_dir=task_output_dir,
                )
            else:
                ecg_signal, fs, lead_name = load_ecg_data(merged_task)
                if ecg_signal is not None:
                    title_map = {
                        "LBBB": "Left Bundle Branch Block", "1dAVB": "1st Degree AV Block",
                        "AF": "Atrial Fibrillation", "ST": "Sinus Tachycardia"
                    }
                    title = f"{title_map.get(abnormality, abnormality)} ({record_identifier})"
                    report_data = visualization_func(ecg_signal, fs, title, lead_name, task_output_dir)
                else:
                    if verbose:
                        print(f"Failed to load data for {record_identifier}. Skipping.")
            end_time = time.perf_counter()
            duration = end_time - start_time
            if verbose:
                print(f"Task completed in {duration:.2f} seconds.")

            if report_data:
                try:
                    val = None
                    if abnormality == "AF":
                        val = validate_af(report_data.get("metrics", {}))
                    elif abnormality == "ST":
                        val = validate_st(report_data.get("metrics", {}))
                    elif abnormality == "SB":
                        val = validate_sb(report_data.get("metrics", {}))
                    elif abnormality == "1dAVB":
                        val = validate_avb1(report_data.get("metrics", {}))
                    elif abnormality == "LBBB":
                        val = validate_lbbb(report_data.get("metrics", {}))
                    elif abnormality == "RBBB":
                        val = validate_rbbb(report_data.get("metrics", {}))
                    elif abnormality == "NR":
                        val = validate_nr(report_data.get("metrics", {}))
                    if val:
                        report_data["validation"] = val
                except Exception as _e:
                    report_data["validation"] = {"status": "warn", "reasons": ["validator error"], "metrics_used": {}}
                report_data['processing_time_seconds'] = round(duration, 2)
                report_path = os.path.join(task_output_dir, "analysis_report.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=4, ensure_ascii=False)
                if verbose:
                    print(f"Analysis report saved to: {report_path}")

                results.append({
                    "status": "ok",
                    "abnormality": abnormality,
                    "record_identifier": record_identifier,
                    "task_output_dir": task_output_dir,
                    "report_path": report_path,
                    "report": report_data if return_reports else None,
                })
            else:
                results.append({
                    "status": "failed",
                    "reason": "load_or_generate_failed",
                    "abnormality": abnormality,
                    "record_identifier": record_identifier,
                    "task_output_dir": task_output_dir,
                })
        except Exception as e:
            error = str(e)
            if verbose:
                print(f"Error processing task {i+1}: {error}")
            results.append({
                "status": "error",
                "error": error,
                "abnormality": abnormality,
                "record_identifier": record_identifier,
                "task_output_dir": task_output_dir,
            })

    index_path = None
    if write_index:
        try:
            index_path = write_results_index(output_dir)
            if verbose:
                print(f"Results index written: {index_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to write results index: {e}")

    return {"output_dir": output_dir, "index_path": index_path, "results": results}


def main(config_path):
    """
    Backward-compatible entrypoint: read JSON config, then run tasks.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    default_out = os.path.join(os.path.dirname(__file__), "data", "ecg_analysis_results")
    output_dir = config.get("output_dir", default_out)
    tasks = config.get("tasks", [])

    run_xai_tasks(tasks, output_dir=output_dir, write_index=True, return_reports=False, verbose=True)
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.json>")
    else:
        main(sys.argv[1])