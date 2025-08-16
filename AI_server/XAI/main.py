# main.py

import json
import os
import time
from dotenv import load_dotenv


load_dotenv()


# api url for images
def get_api_download_url(filename, instance_id):
    """generate api url for images"""
    base_url = os.getenv("BASE_URL", "")
    return f"{base_url}/static/images/{instance_id}/{filename}"

# try to import XAI modules
try:
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
except ImportError:
    try:
        from xai.ecg_utils import load_ecg_data
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
    except ImportError:
        # if all imports fail, create placeholder functions
        print("Warning: XAI modules not available, creating placeholder functions")
        
        def load_ecg_data(config_task):
            return None, None, None
        
        def validate_af(metrics):
            return {"status": "placeholder", "reasons": ["XAI modules not available"]}
        
        def validate_st(metrics):
            return {"status": "placeholder", "reasons": ["XAI modules not available"]}
        
        def validate_sb(metrics):
            return {"status": "placeholder", "reasons": ["XAI modules not available"]}
        
        def validate_avb1(metrics):
            return {"status": "placeholder", "reasons": ["XAI modules not available"]}
        
        def validate_lbbb(metrics):
            return {"status": "placeholder", "reasons": ["XAI modules not available"]}
        
        def validate_rbbb(metrics):
            return {"status": "placeholder", "reasons": ["XAI modules not available"]}
        
        def validate_nr(metrics):
            return {"status": "placeholder", "reasons": ["XAI modules not available"]}
        
        def generate_normal_visuals(ecg_signal, fs, title, lead_name, output_dir):
            return {"image_paths": {"placeholder.png": os.path.join(output_dir, "placeholder.png")}}
        
        def generate_sb_visuals(ecg_signal, fs, title, lead_name, output_dir):
            return {"image_paths": {"placeholder.png": os.path.join(output_dir, "placeholder.png")}}
        
        def generate_af_visuals(ecg_signal, fs, title, lead_name, output_dir):
            return {"image_paths": {"placeholder.png": os.path.join(output_dir, "placeholder.png")}}
        
        def generate_st_visuals(ecg_signal, fs, title, lead_name, output_dir):
            return {"image_paths": {"placeholder.png": os.path.join(output_dir, "placeholder.png")}}
        
        def generate_lbbb_visuals(ecg_signal, fs, title, lead_name, output_dir):
            return {"image_paths": {"placeholder.png": os.path.join(output_dir, "placeholder.png")}}
        
        def generate_rbbb_visuals(record_name, dicom_filepath, db_dir, output_dir):
            return {"image_paths": {"placeholder.png": os.path.join(output_dir, "placeholder.png")}}
        
        def generate_avb_visuals(ecg_signal, fs, title, lead_name, output_dir):
            return {"image_paths": {"placeholder.png": os.path.join(output_dir, "placeholder.png")}}

VISUALIZATION_DISPATCHER = {
    "AFIB": generate_af_visuals,
    "STACH": generate_st_visuals,
    "LBBB": generate_lbbb_visuals,
    "RBBB": generate_rbbb_visuals,
    "1dAVB": generate_avb_visuals,
    "NORM": generate_normal_visuals,
    "SB": generate_sb_visuals,
}



def run_xai_tasks(
    abnormalities,
    instance_id: str,
    return_reports: bool = True,
    verbose: bool = True,
    web_accessible: bool = None,
):
    """
    Simplified API: run XAI by specifying one or multiple abnormality names and an instance_id.

    Parameters:
    - abnormalities: str | list[str] — e.g., 'AFIB' or ['AFIB','1dAVB']
    - instance_id: str — unified identifier
    - return_reports: include full report JSON in return payload
    - verbose: print progress logs
    - web_accessible: if True, generate API download URLs for images

    Behavior:
    - Non-RBBB tasks assume DICOM under AI_server/dicoms
    - RBBB tasks assume WFDB under XAI/data/ecg_files
    - Images are saved to local output directory and API URLs are generated

    Returns:
    - { "results": [...] } with API download URLs for image access
    """
    # 简化web处理逻辑
    if web_accessible is None:
        web_accessible = True  # 默认启用API URL生成
    
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
        if verbose:
            print(f"\n--- Processing Task {i+1}: {abn} for {instance_id} ---")

        start_time = time.perf_counter()
        
        # 创建web可访问的输出目录
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
            # handle different types of abnormalities
            if abn == "RBBB" or abn == "LBBB":
                # LBBB and RBBB need four leads for analysis: V1, V2, V6, I
                # support both DICOM and WFDB data sources
                dicom_filepath = None
                if os.path.exists(default_dicom_dir):
                    # try DICOM file
                    dicom_filepath = os.path.join(default_dicom_dir, f"{instance_id}.dcm")
                    if not os.path.exists(dicom_filepath):
                        # if DICOM does not exist, try WFDB
                        dicom_filepath = None
                
                task_config = {
                    "abnormality": abn,
                    "instance_id": instance_id,
                    "lead_index": ["V1", "V2", "V6", "I"],  # 四个导联
                    "db_dir": default_db_dir,
                    "dicom_dir": default_dicom_dir,
                }
                
                if abn == "RBBB":
                    # RBBB can directly process file paths
                    # note: RBBB function internally handles lead name matching
                    # but need to ensure DICOM file path is correctly passed
                    if dicom_filepath and os.path.exists(dicom_filepath):
                        report_data = visualization_func(
                            record_name=instance_id,
                            dicom_filepath=dicom_filepath,
                            db_dir=default_db_dir,
                            output_dir=web_output_dir,
                        )
                    else:
                        # if DICOM file does not exist, try WFDB
                        report_data = visualization_func(
                            record_name=instance_id,
                            dicom_filepath=None,
                            db_dir=default_db_dir,
                            output_dir=web_output_dir,
                        )
                else:  # LBBB
                    # LBBB uses multi-lead analysis, directly call multi-lead function
                    # import multi-lead function
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
                    
                    # call multi-lead LBBB analysis function
                    report_data = generate_lbbb_multilead(
                        record_name=instance_id,
                        dicom_filepath=dicom_filepath,
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
            else:
                # build task configuration
                task_config = {
                    "abnormality": abn,
                    "instance_id": instance_id,
                    "lead_index": 1,
                    "dicom_dir": default_dicom_dir,
                }
                
                data = load_ecg_data(task_config)
                if isinstance(data, tuple) and len(data) == 3:
                    ecg_signal, fs, lead_name = data
                    if ecg_signal is not None:
                        title_map = {
                            "LBBB": "Left Bundle Branch Block", 
                            "1dAVB": "1st Degree AV Block",
                            "AFIB": "Atrial Fibrillation", 
                            "STACH": "Sinus Tachycardia"
                        }
                        title = f"{title_map.get(abn, abn)} ({instance_id})"
                        report_data = visualization_func(ecg_signal, fs, title, lead_name, 
                                                      web_output_dir)
                else:
                    if verbose:
                        print(f"Failed to load data for {instance_id}. Skipping.")
                    results.append({
                        "status": "failed",
                        "reason": "load_failed",
                        "abnormality": abn,
                        "instance_id": instance_id,
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
                    elif abn == "SB":
                        val = validate_sb(report_data.get("metrics", {}))
                    elif abn == "1dAVB":
                        val = validate_avb1(report_data.get("metrics", {}))
                    elif abn == "LBBB":
                        val = validate_lbbb(report_data.get("metrics", {}))
                    elif abn == "RBBB":
                        val = validate_rbbb(report_data.get("metrics", {}))
                    elif abn == "NORM":
                        val = validate_nr(report_data.get("metrics", {}))
                    if val:
                        report_data["validation"] = val
                except Exception as _e:
                    report_data["validation"] = {"status": "warn", "reasons": ["validator error"], "metrics_used": {}}
                
                report_data['processing_time_seconds'] = round(duration, 2)
                
                # 生成API格式的图片URL
                if web_accessible and report_data.get("image_paths"):
                    # 将image_paths转换为API格式的URL
                    api_image_paths = {}
                    for img_name, img_path in report_data["image_paths"].items():
                        if os.path.exists(img_path):
                            # 生成API下载URL
                            api_url = get_api_download_url(img_name, instance_id)
                            api_image_paths[img_name] = api_url
                            if verbose:
                                print(f"Generated API URL for {img_name}: {api_url}")
                    
                    # 更新report_data中的image_paths
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

    # Backward-compatible JSON entrypoint uses simplified API
    out = run_xai_tasks(tasks, output_dir=output_dir, return_reports=False, verbose=True, web_accessible=web_accessible)
    # Print a minimal summary to stdout for CLI usage
    print(json.dumps({"results": out.get("results", [])}, ensure_ascii=False))
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.json>")
    else:
        main(sys.argv[1])