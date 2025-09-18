# main.py

import json
import os
import time
import re

from .xai.ecg_utils import load_dicom_signals_from_dataset
from .xai.explanations.validators import (
    validate_af, validate_st, validate_sb, validate_avb1, validate_lbbb, validate_rbbb, validate_nr,
    validate_avb2, validate_aflt, validate_pac, validate_pvc
)
from .xai.explanations.normal import generate_normal_visuals
from .xai.explanations.sb import generate_sb_visuals
from .xai.explanations.af import generate_af_visuals
from .xai.explanations.st import generate_st_visuals
from .xai.explanations.lbbb import generate_lbbb_visuals
from .xai.explanations.rbbb import generate_rbbb_visuals
from .xai.explanations.avb import generate_avb_visuals
from .xai.explanations.avb2 import generate_avb2_visuals
from .xai.explanations.avb3 import generate_avb3_visuals
from .xai.explanations.aflt import generate_aflt_multilead, generate_aflt_visuals
from .xai.explanations.pac import generate_pac_visuals
from .xai.explanations.pvc import generate_pvc_visuals

from dicom_utils import get_instance_dicom

from dotenv import load_dotenv
load_dotenv()
# api url for images
def get_api_download_url(filename, instance_id):
    """generate api url for images"""
    base_url = os.getenv("BASE_URL", "")
    return f"{base_url}/static/images/{instance_id}/{filename}"


# Helper: canonicalize lead names like 'Lead II (Einthoven)' → 'II', 'aVF' preserved, 'V 1' → 'V1'
def _canonical_lead_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    s = name.upper()
    # remove parentheses content e.g., (Einthoven)
    s = re.sub(r"\(.*?\)", "", s)
    for token in ["LEAD", "DERIVATION", "CHANNEL", "ECG", " ", "-", "_"]:
        s = s.replace(token, "")
    if "AVR" in s: return "aVR"
    if "AVL" in s: return "aVL"
    if "AVF" in s: return "aVF"
    for k in ["V1","V2","V3","V4","V5","V6"]:
        if k in s: return k
    if "III" in s: return "III"
    if "II" in s: return "II"
    if s.endswith("I") or s == "I":
        return "I"
    return name

# Helper: pick preferred lead given a dict {lead_name: (sig, fs)} using canonical mapping
def _pick_preferred_lead(leads_data: dict, preferred_order):
    canon_map = {}
    for k in leads_data.keys():
        canon_map[_canonical_lead_name(k)] = k
    for want in preferred_order:
        if want in canon_map:
            return canon_map[want]
    for want in preferred_order:
        for k in leads_data.keys():
            if want.lower() in str(k).lower():
                return k
    return list(leads_data.keys())[0]


VISUALIZATION_DISPATCHER = {
    "AFIB": generate_af_visuals,
    "STACH": generate_st_visuals,
    "SBRAD": generate_sb_visuals,
    "1dAVB": generate_avb_visuals,
    "2dAVB": generate_avb2_visuals,
    "3dAVB": generate_avb3_visuals,
    "AFLT": generate_aflt_visuals,  # will be overridden to multilead below
    "LBBB": generate_lbbb_visuals,
    "RBBB": generate_rbbb_visuals,
    "NORM": generate_normal_visuals,
    "SBRAD": generate_sb_visuals,
    "PAC": generate_pac_visuals,
    "PVC": generate_pvc_visuals,
}


def run_xai_tasks(
    abnormalities,
    instance_id: str,
    token: str = None,  
    return_reports: bool = True,
    verbose: bool = True,
    web_accessible: bool = None,
):
    """
    Simplified API: run XAI by specifying one or multiple abnormality names and an instance_id.

    Parameters:
    - abnormalities: str | list[str] — e.g., 'AFIB' or ['AFIB','1dAVB']
    - instance_id: str — unified identifier
    - token: str — authentication token for DICOM server access
    - return_reports: include full report JSON in return payload
    - verbose: print progress logs
    - web_accessible: if True, generate API download URLs for images

    Behavior:
    - All tasks now fetch DICOM data directly from server using token
    - No more local DICOM file dependency
    - Images are saved to web-accessible directory and API URLs are generated

    Returns:
    - { "results": [...] } with API download URLs for image access
    """
    # verify token
    if not token:
        raise ValueError("token is required for DICOM access")
    
    # simplify web processing logic
    if web_accessible is None:
        web_accessible = True  # default enable API URL generation
    
    if verbose:
        if web_accessible:
            print("API URL generation enabled for images")
        else:
            print("API URL generation disabled")
    
    base_dir = os.path.dirname(__file__)
    default_dicom_dir = os.path.normpath(os.path.join(base_dir, "..", "dicoms"))
    default_db_dir = os.path.normpath(os.path.join(base_dir, "data", "ecg_files"))

    abnormalities_list = [abnormalities] if isinstance(abnormalities, str) else list(abnormalities or [])

    results = []

    for i, abn in enumerate(abnormalities_list):
        # backward-compatibility: normalize old code 'SB' to 'SBRAD'
        if abn == "SB":
            abn = "SBRAD"
        if verbose:
            print(f"\n--- Processing Task {i+1}: {abn} for {instance_id} ---")

        start_time = time.perf_counter()
        
        # create web accessible output directory
        web_output_dir = os.path.join(os.path.dirname(__file__), "static", "images", instance_id)
        if not os.path.exists(web_output_dir):
            os.makedirs(web_output_dir)

        visualization_func = VISUALIZATION_DISPATCHER.get(abn)
        if not visualization_func:
            if verbose:
                print(f"Warning: No function for '{abn}'.")
            results.append({
                "status": "skipped",
                "reason": "unsupported_abnormality",
                "abnormality": abn,
                "instance_id": instance_id,
            })
            continue

        report_data = None
        error = None

        try:
            # first get DICOM data from server (all abnormalities need)
            try:
                if verbose:
                    print(f"Fetching DICOM data from server for {instance_id}")
                
                # use token to get DICOM data from server
                dicom_dataset = get_instance_dicom(instance_id, token)
                if verbose:
                    print(f"Successfully retrieved DICOM data for {instance_id}")
                
            except Exception as e:
                if verbose:
                    print(f"Failed to fetch DICOM data for {instance_id}: {e}")
                results.append({
                    "status": "failed",
                    "reason": "dicom_fetch_failed",
                    "abnormality": abn,
                    "instance_id": instance_id,
                    "error": str(e)
                })
                continue

            # handle different types of abnormalities
            if abn == "RBBB" or abn == "LBBB":
                # LBBB and RBBB need four leads for analysis: V1, V2, V6, I
                try:
                    # build task configuration
                    task_config = {
                        "abnormality": abn,
                        "instance_id": instance_id,
                        "lead_index": ["V1", "V2", "V6", "I"],  # four leads
                        "db_dir": default_db_dir,
                        "dicom_dataset": dicom_dataset,  # pass DICOM dataset
                    }
                    
                    if abn == "RBBB":
                        # RBBB processing - pass dicom_dataset as dicom_filepath parameter
                        report_data = visualization_func(
                            record_name=instance_id,
                            dicom_filepath=dicom_dataset,  # fix: pass DICOM dataset
                            db_dir=default_db_dir,
                            output_dir=web_output_dir,
                        )
                    else:  # LBBB
                        # LBBB use multi-lead analysis
                        try:
                            from .xai.explanations.lbbb import generate_lbbb_multilead
                        except ImportError:
                            try:
                                from xai.explanations.lbbb import generate_lbbb_multilead
                            except ImportError:
                                if verbose:
                                    print(f"Failed to import LBBB multi-lead function for {instance_id}. Skipping.")
                                results.append({
                                    "status": "failed",
                                    "reason": "import_failed",
                                    "abnormality": abn,
                                    "instance_id": instance_id,
                                })
                                continue
                        
                        # call multi-lead LBBB analysis function - fix: pass DICOM dataset as dicom_filepath
                        report_data = generate_lbbb_multilead(
                            record_name=instance_id,
                            dicom_filepath=dicom_dataset,  # fix: pass DICOM dataset
                            db_dir=default_db_dir,
                            output_dir=web_output_dir
                        )
                        
                        if not report_data:
                            if verbose:
                                print(f"Failed to generate LBBB analysis for {instance_id}. Skipping.")
                            results.append({
                                "status": "failed",
                                "reason": "generate_failed",
                                "abnormality": abn,
                                "instance_id": instance_id,
                            })
                            continue
                            
                except Exception as e:
                    if verbose:
                        print(f"Failed to process {abn} for {instance_id}: {e}")
                    results.append({
                        "status": "failed",
                        "reason": "processing_failed",
                        "abnormality": abn,
                        "instance_id": instance_id,
                        "error": str(e)
                    })
                    continue
            else:
                # other abnormalities: load ECG data directly from DICOM dataset
                try:
                    # build task configuration
                    task_config = {
                        "abnormality": abn,
                        "instance_id": instance_id,
                        "lead_index": 1,
                        "dicom_dataset": dicom_dataset,  # pass DICOM dataset
                    }
                    
                    # use new function to load ECG data directly from DICOM dataset
                    leads_data = load_dicom_signals_from_dataset(dicom_dataset)
                    
                    if leads_data and len(leads_data) > 0:
                        # choose lead based on abnormality or run multilead
                        if abn == "3dAVB" or abn == "2dAVB":
                            preferred = ["II", "III", "aVF", "V1", "I"]
                            lead_name = _pick_preferred_lead(leads_data, preferred)
                            ecg_signal, fs = leads_data[lead_name]
                            title_map = {
                                "LBBB": "Left Bundle Branch Block", 
                                "1dAVB": "1st Degree AV Block",
                                "2dAVB": "2nd Degree AV Block",
                                "3dAVB": "3rd Degree AV Block",
                                "AFLT": "Atrial Flutter",
                                "AFIB": "Atrial Fibrillation", 
                                "STACH": "Sinus Tachycardia",
                                "SBRAD": "Sinus Bradycardia"
                            }
                            title = f"{title_map.get(abn, abn)} ({instance_id})"
                            report_data = visualization_func(ecg_signal, fs, title, lead_name, 
                                                          web_output_dir)
                        elif abn == "AFLT":
                            # Multi-lead AFLT analysis
                            title_map = {
                                "AFLT": "Atrial Flutter"
                            }
                            title = f"{title_map.get(abn, abn)} ({instance_id})"
                            report_data = generate_aflt_multilead(leads_data, title, web_output_dir)
                        elif abn == "PAC":
                            preferred = ["II", "III", "aVF", "V1", "I"]
                            lead_name = _pick_preferred_lead(leads_data, preferred)
                            ecg_signal, fs = leads_data[lead_name]
                            title = f"Premature Atrial Complex ({instance_id})"
                            report_data = visualization_func(ecg_signal, fs, title, lead_name, web_output_dir)
                        elif abn == "PVC":
                            preferred = ["II", "III", "aVF", "V1", "I"]
                            lead_name = _pick_preferred_lead(leads_data, preferred)
                            ecg_signal, fs = leads_data[lead_name]
                            title = f"Premature Ventricular Complex ({instance_id})"
                            report_data = visualization_func(ecg_signal, fs, title, lead_name, web_output_dir)
                        else:
                            # get first available lead data (prefer II for rhythm if available)
                            preferred = ["II", "III", "aVF", "V1", "I"]
                            lead_name = _pick_preferred_lead(leads_data, preferred)
                            ecg_signal, fs = leads_data[lead_name]
                            title_map = {
                                "LBBB": "Left Bundle Branch Block", 
                                "1dAVB": "1st Degree AV Block",
                                "2dAVB": "2nd Degree AV Block",
                                "3dAVB": "3rd Degree AV Block",
                                "AFLT": "Atrial Flutter",
                                "AFIB": "Atrial Fibrillation", 
                                "STACH": "Sinus Tachycardia",
                                "SBRAD": "Sinus Bradycardia"
                            }
                            title = f"{title_map.get(abn, abn)} ({instance_id})"
                            report_data = visualization_func(ecg_signal, fs, title, lead_name, 
                                                          web_output_dir)
                        
                        if not report_data:
                            if verbose:
                                print(f"Failed to generate {abn} analysis for {instance_id}. Skipping.")
                            results.append({
                                "status": "failed",
                                "reason": "generate_failed",
                                "abnormality": abn,
                                "instance_id": instance_id,
                            })
                            continue
                    else:
                        if verbose:
                            print(f"Failed to extract ECG signal for {instance_id}. Skipping.")
                        results.append({
                            "status": "failed",
                            "reason": "signal_extraction_failed",
                            "abnormality": abn,
                            "instance_id": instance_id,
                        })
                        continue
                        
                except Exception as e:
                    if verbose:
                        print(f"Failed to process {abn} for {instance_id}: {e}")
                    results.append({
                        "status": "failed",
                        "reason": "processing_failed",
                        "abnormality": abn,
                        "instance_id": instance_id,
                        "error": str(e)
                    })
                    continue

            end_time = time.perf_counter()
            duration = end_time - start_time
            if verbose:
                print(f"Task completed in {duration:.2f} seconds.")

            if report_data:
                try:
                    val = None
                    if abn == "AFIB":
                        val = validate_af(report_data.get("metrics", {}))
                    elif abn == "STACH":
                        val = validate_st(report_data.get("metrics", {}))
                    elif abn == "SBRAD":
                        val = validate_sb(report_data.get("metrics", {}))
                    elif abn == "1dAVB":
                        val = validate_avb1(report_data.get("metrics", {}))
                    elif abn == "3dAVB":
                        from .xai.explanations.validators import validate_avb3
                        val = validate_avb3(report_data.get("metrics", {}))
                    elif abn == "LBBB":
                        val = validate_lbbb(report_data.get("metrics", {}))
                    elif abn == "RBBB":
                        val = validate_rbbb(report_data.get("metrics", {}))
                    elif abn == "NORM":
                        val = validate_nr(report_data.get("metrics", {}))
                    elif abn == "PAC":
                        val = validate_pac(report_data)
                    elif abn == "PVC":
                        val = validate_pvc(report_data)
                    if val:
                        report_data["validation"] = val
                except Exception as _e:
                    report_data["validation"] = {"status": "warn", "reasons": ["validator error"], "metrics_used": {}}
                
                # Additional validators
                if abn == "2dAVB":
                    try:
                        report_data["validation"] = validate_avb2(report_data.get("metrics", {}))
                    except Exception:
                        pass
                if abn == "AFLT":
                    try:
                        report_data["validation"] = validate_aflt(report_data.get("metrics", {}))
                    except Exception:
                        pass

                report_data['processing_time_seconds'] = round(duration, 2)
                
                # generate API format image URL
                if web_accessible and report_data.get("image_paths"):
                    # convert image_paths to API format URL
                    api_image_paths = {}
                    
                    # process different format of image_paths
                    if isinstance(report_data["image_paths"], dict):
                        # if image_paths is dictionary format
                        for img_name, img_path in report_data["image_paths"].items():
                            if os.path.exists(img_path):
                                # generate API download URL
                                api_url = get_api_download_url(img_name, instance_id)
                                api_image_paths[img_name] = api_url
                                if verbose:
                                    print(f"Generated API URL for {img_name}: {api_url}")
                    elif isinstance(report_data["image_paths"], list):
                        # if image_paths is list format
                        for img_filename in report_data["image_paths"]:
                            # build full file path
                            img_path = os.path.join(web_output_dir, img_filename)
                            if os.path.exists(img_path):
                                # generate API download URL
                                api_url = get_api_download_url(img_filename, instance_id)
                                api_image_paths[img_filename] = api_url
                                if verbose:
                                    print(f"Generated API URL for {img_filename}: {api_url}")
                    
                    # update image_paths in report_data
                    report_data["image_paths"] = api_image_paths
                    
                    if verbose:
                        print(f"API image URLs generated: {list(api_image_paths.keys())}")

                results.append({
                    "status": "ok",
                    "abnormality": abn,
                    "instance_id": instance_id,
                    "report": report_data if return_reports else None,
                })
            else:
                results.append({
                    "status": "failed",
                    "reason": "generate_failed",
                    "abnormality": abn,
                    "instance_id": instance_id,
                })
        except Exception as e:
            error = str(e)
            if verbose:
                print(f"Error processing task {i+1}: {error}")
            results.append({
                "status": "error",
                "error": error,
                "abnormality": abn,
                "instance_id": instance_id,
            })

    return {"results": results}


def main(config_path):
    """
    Backward-compatible entrypoint: read JSON config, then run tasks.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    default_out = os.path.join(os.path.dirname(__file__), "data", "ecg_analysis_results")
    output_dir = config.get("output_dir", default_out)
    tasks = config.get("tasks", [])
    web_accessible = config.get("web_accessible", None)  # None means use default from config

    # backward-compatible JSON entrypoint uses simplified API
    out = run_xai_tasks(tasks, output_dir=output_dir, return_reports=False, verbose=True, web_accessible=web_accessible)
    # print a minimal summary to stdout for CLI usage
    print(json.dumps({"results": out.get("results", [])}, ensure_ascii=False))
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.json>")
    else:
        main(sys.argv[1])