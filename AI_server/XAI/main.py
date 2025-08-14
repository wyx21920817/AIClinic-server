# main.py

import json
import os
import time
from .xai.ecg_utils import load_ecg_data
from .xai.explanations.validators import (
    validate_af, validate_st, validate_sb, validate_avb1, validate_lbbb, validate_rbbb, validate_nr
)
from .xai.explanations.normal import generate_normal_visuals
from .xai.explanations.sb import generate_sb_visuals
from .xai.explanations.af import generate_af_visuals
from .xai.explanations.st import generate_st_visuals
from .xai.explanations.lbbb import generate_lbbb_visuals
from .xai.explanations.rbbb import generate_rbbb_visuals
from .xai.explanations.avb import generate_avb_visuals

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

def run_xai_tasks_advanced(
    tasks,
    output_dir=None,
    default_dicom_dir=None,
    default_db_dir=None,
    default_lead_index=None,
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
    - return_reports: include full report JSON in return payload
    - verbose: print progress logs

    Returns
    - dict with key: 'results' (list of per-task results)
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
            # Unified loading: allow modules to accept either single-lead arrays or multi-lead dicts
            # For RBBB we may call specialized function with optional dicom path
            if abnormality == "RBBB":
                report_data = visualization_func(
                    record_name=merged_task.get("instance_id") or merged_task.get("record_name"),
                    dicom_filepath=(
                        os.path.join(merged_task.get("dicom_dir"), f"{merged_task.get('instance_id')}.dcm")
                        if merged_task.get("dicom_dir") and merged_task.get("instance_id") else None
                    ),
                    db_dir=merged_task.get("db_dir"),
                    output_dir=task_output_dir,
                )
            else:
                data = load_ecg_data(merged_task)
                if isinstance(data, tuple) and len(data) == 3:
                    ecg_signal, fs, lead_name = data
                    if ecg_signal is not None:
                        title_map = {
                            "LBBB": "Left Bundle Branch Block", "1dAVB": "1st Degree AV Block",
                            "AF": "Atrial Fibrillation", "ST": "Sinus Tachycardia"
                        }
                        title = f"{title_map.get(abnormality, abnormality)} ({record_identifier})"
                        report_data = visualization_func(ecg_signal, fs, title, lead_name, task_output_dir)
                elif isinstance(data, dict):
                    # For future multi-lead modules (e.g., LBBB multi-lead), dispatch accordingly if needed
                    # Currently single-lead functions do not take dict; skip here.
                    pass
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

    return {"results": results}


def run_xai_tasks(
    abnormalities,
    instance_id: str,
    return_reports: bool = True,
    verbose: bool = True,
):
    """
    Simplified API: run XAI by specifying one or multiple abnormality names and an instance_id.

    Parameters:
    - abnormalities: str | list[str] — e.g., 'AF' or ['AF','1dAVB']
    - instance_id: str — unified identifier
    - return_reports: include full report JSON in return payload

    Behavior:
    - Non-RBBB tasks assume DICOM under AI_server/dicoms
    - RBBB tasks assume WFDB under XAI/data/ecg_files

    Returns:
    - { "results": [...] }
    """
    base_dir = os.path.dirname(__file__)
    default_dicom_dir = os.path.normpath(os.path.join(base_dir, "..", "dicoms"))
    default_db_dir = os.path.normpath(os.path.join(base_dir, "data", "ecg_files"))

    abnormalities_list = [abnormalities] if isinstance(abnormalities, str) else list(abnormalities or [])

    tasks = []
    for abn in abnormalities_list:
        if abn == "RBBB":
            tasks.append({
                "abnormality": abn,
                "instance_id": instance_id,
                "db_dir": default_db_dir,
            })
        else:
            tasks.append({
                "abnormality": abn,
                "instance_id": instance_id,
                "lead_index": 1,
                "dicom_dir": default_dicom_dir,
            })

    out = run_xai_tasks_advanced(
        tasks=tasks,
        default_dicom_dir=default_dicom_dir,
        default_db_dir=default_db_dir,
        default_lead_index=1,
        return_reports=return_reports,
        verbose=verbose,
    )
    return {"results": out.get("results", [])}


def main(config_path):
    """
    Backward-compatible entrypoint: read JSON config, then run tasks.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    default_out = os.path.join(os.path.dirname(__file__), "data", "ecg_analysis_results")
    output_dir = config.get("output_dir", default_out)
    tasks = config.get("tasks", [])

    # Backward-compatible JSON entrypoint uses advanced API
    out = run_xai_tasks_advanced(tasks, output_dir=output_dir, return_reports=False, verbose=True)
    # Print a minimal summary to stdout for CLI usage
    print(json.dumps({"results": out.get("results", [])}, ensure_ascii=False))
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.json>")
    else:
        main(sys.argv[1])