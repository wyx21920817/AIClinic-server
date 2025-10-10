# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import tempfile
import os
import json
import time
from dicom_utils import get_instance_dicom, store_instance_dicom
from preprocess import extract_waveform
from model_utils import load_model, predict_ecg, predict_hubert_head_from_ds
from pydicom import dcmread
from ecg_ptbxl_benchmarking_master.extract_and_predict_ecg import process_and_predict 
from fastapi.param_functions import Form
from typing import List
from pydicom.filewriter import dcmwrite
from XAI.main import run_xai_tasks
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
# mount static files
app.mount("/static", StaticFiles(directory="XAI/static"), name="static")

# Add middleware to ensure connections are properly released
@app.middleware("http")
async def close_http_connections(request, call_next):
    response = await call_next(request)
    # Force garbage collection to help release connections
    import gc
    gc.collect()
    return response

# load models when initialize
from model_utils import load_models
load_models()

import logging
import concurrent.futures
import asyncio
from functools import partial

logging.basicConfig(level=logging.DEBUG)

# Configure a custom thread pool executor with optimized settings
# This will be used instead of the default Starlette threadpool
XAI_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,  # Limit concurrent XAI tasks to prevent overloading
    thread_name_prefix="xai_worker"
)

async def run_in_custom_threadpool(func):
    """Run a function in our custom thread pool instead of the default Starlette threadpool.
    
    This prevents the default threadpool from being blocked by long-running tasks.
    
    Args:
        func: A function to run in the threadpool (can be a partial function with args)
        
    Returns:
        The result of the function
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(XAI_THREAD_POOL, func)

class PredictRequest(BaseModel):
    token: str

@app.post("/predict/{instance_id}")
async def predict(instance_id: str, request: PredictRequest):
    token = request.token
    logging.debug(f"Start prediction for instance_id: {instance_id} with token provided.")

    ds = None
    temp_dicom_file_path = None

    try:
        # 1. Retrieve DICOM
        ds = get_instance_dicom(instance_id, token)
        logging.debug(f"DICOM data retrieve success for original model.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            dcmwrite(tmp_file.name, ds)
            temp_dicom_file_path = tmp_file.name
        logging.debug(f"DICOM dataset saved to temporary file: {temp_dicom_file_path}")

        # 2. Transform to adapt input of models
        logging.debug("Start waveform extraction...")
        ecg, sampling_rate, lead_names = extract_waveform(ds)
        logging.debug(f"ECG data extracted for original model with shape: {ecg.shape}")
        logging.debug(f"ECG data shape for original model prediction: {ecg.shape}.")

        preds = {}
        new_model_predictions = {}
        hubert_head_predictions = {}
        combined_predictions = {}

        # 3. Predict (Original Model)
        preds = predict_ecg(ecg)
        logging.debug(f"Original model predict outcome: {preds}")

        # 4. Predict (ResNet Fine-tuned Model)
        new_model_predictions = process_and_predict(temp_dicom_file_path)
        logging.debug(f"New model predict outcome: {new_model_predictions}")

        # 5. Predict (HuBERT + MLP Head Model)
        hubert_head_predictions = predict_hubert_head_from_ds(ds)
        allowed_hubert = ["CRBBB", "CLBBB", "PVC", "AFLT"]
        hubert_head_predictions = {k: hubert_head_predictions[k] for k in allowed_hubert if k in hubert_head_predictions}
        logging.debug(f"HuBERT-Head model predict outcome: {hubert_head_predictions}")

        combined_predictions = {
            "original_dnn_model": preds,
            "resnet_fine_tuned_model": new_model_predictions,
            "hubert_mlp_head_model": hubert_head_predictions
        }
        logging.debug(f"Combined predictions: {combined_predictions}")

        # --- MODIFIED LOGIC for storing ---
        comment = "; ".join(f"{k}:{v:.3f}" for k, v in preds.items())
        comment += " | ResNet Top5: " + "; ".join(f"{k}:{v:.3f}" for k, v in new_model_predictions.items())
        # Append HuBERT Head Top5
        comment += " | HuBERT-Head: " + "; ".join(
        f"{k}:{hubert_head_predictions[k]:.3f}"
        for k in ["CRBBB","CLBBB","PVC","AFLT"] if k in hubert_head_predictions
        )
        logging.debug(f"Prepared comment string for PatientComments: {comment}")

        new_instance_id = store_instance_dicom(instance_id, comment, token, ds)

        return {"modified_instance_id": new_instance_id, "predictions": combined_predictions}

    except requests.exceptions.RequestException as e:
        logging.error(f"Orthanc retrieve, file saving, or waveform extraction failed for instance {instance_id}: {e}")
        if e.response:
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to fetch DICOM file: {e.response.text}")
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    except Exception as e:
        logging.error(f"Prediction failed for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        if temp_dicom_file_path and os.path.exists(temp_dicom_file_path):
            try:
                os.remove(temp_dicom_file_path)
                logging.debug(f"Temporary DICOM file {temp_dicom_file_path} deleted.")
            except Exception as e:
                logging.error(f"Failed to delete temporary file {temp_dicom_file_path}: {e}")


class DiagnoseRequest(BaseModel):
    temp_token: str

@app.post("/diagnose/{instance_id}")
async def diagnose_ecg(instance_id: str, req: DiagnoseRequest):
    request_start_time = time.time()
    logging.info(f"Receive a diagnostic request,Instance ID: {instance_id}")

    try:
        token = req.temp_token
        if not token:
            raise HTTPException(status_code=401, detail="Failed to obtain authentication token")

        logging.debug(f"Successfully obtained token for instance {instance_id}")
       

        result = await predict(instance_id, PredictRequest(token=token))

        output = {
            "instance_id": instance_id,
            "status": "success",
            "predictions": result["predictions"],
            "modified_instance_id": result["modified_instance_id"],
            "message": f"Successfully diagnosed ECG instance: {instance_id}"
        }

        request_end_time = time.time()
        logging.info(f"[TIMING] Total request processing time for instance {instance_id}: {request_end_time - request_start_time:.2f} seconds")
        return output

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Diagnostic failure,Instance ID: {instance_id}, error: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")
    finally:
        import gc
        gc.collect()


@app.post("/interprete/{instance_id}")
async def interprete(
    instance_id: str,
    disease: List[str] = Form(...),
    temp_token: str = Form(...),
):
    logging.info(f"[interprete] instance_id={instance_id}, disease={disease}")

    try:
        token = temp_token
        if not token:
            raise HTTPException(status_code=401, detail="Failed to obtain authentication token")
        abnormalities = disease

        # Use our custom thread pool instead of the default Starlette threadpool
        # This prevents blocking the default threadpool which is used for other operations
        xai_task = partial(run_xai_tasks, abnormalities, instance_id, token, True, True)
        xai_out = await run_in_custom_threadpool(xai_task)

        return {
            "instance_id": instance_id,
            "abnormalities": abnormalities,
            "status": "success",
            "results": xai_out.get("results", []),
            "message": "XAI interpretation completed.",
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"[interprete] Requests error: {e}")
        if getattr(e, "response", None) is not None:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Failed to fetch DICOM file: {e.response.text}")
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[interprete] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Interpretation failed: {e}")
