# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import the modified store_instance_dicom function (now returns str | None)
import requests
import tempfile
import os
import json
from starlette.concurrency import run_in_threadpool
from dicom_utils import get_instance_dicom, store_instance_dicom, ORTHANC_URL, get_headers
from preprocess import extract_waveform # add_predictions_to_dicom is no longer used for /modify
from model_utils import load_model, predict_ecg
from pydicom import dcmread
from ecg_ptbxl_benchmarking_master.extract_and_predict_ecg import process_and_predict 
from fastapi.param_functions import Form
from typing import List
from XAI.main import run_xai_tasks
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
# mount static files
app.mount("/static", StaticFiles(directory="XAI/static"), name="static")
# load models when intialize
from model_utils import load_models
load_models()

import logging

# set log record
logging.basicConfig(level=logging.DEBUG)

class PredictRequest(BaseModel):
    token: str

@app.post("/predict/{instance_id}")
async def predict(instance_id: str, request: PredictRequest):
    token = request.token
    logging.debug(f"Start prediction for instance_id: {instance_id} with token provided.")

    ds = None
    temp_dicom_file_path = None

    try:
        # 1. Retrieve DICOM (Still need to retrieve to get waveform for original model)
        ds = await run_in_threadpool(get_instance_dicom, instance_id, token)
        logging.debug(f"DICOM data retrieve success for original model.")

        logging.debug(f"Attempting to fetch DICOM binary data for instance {instance_id} for new model.")
        headers = get_headers(token)
        dicom_binary_url = f"{ORTHANC_URL}/instances/{instance_id}/file"
        logging.debug(f"Requesting URL: {dicom_binary_url}")
        logging.debug(f"Request headers: {headers}")

        response = requests.get(dicom_binary_url, headers=headers, timeout=60)
        response.raise_for_status()
        logging.debug(f"Response size={len(response.content)} bytes, status={response.status_code}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            logging.debug("Starting to write DICOM to temp file...")
            tmp_file.write(response.content)
            temp_dicom_file_path = tmp_file.name
        logging.debug(f"DICOM binary data saved to temporary file: {temp_dicom_file_path}")

        # 2. Transform to adapt input of models (This is for the original model path)
        logging.debug("Start waveform extraction...")
        ecg, sampling_rate, lead_names = await run_in_threadpool(extract_waveform, ds)
        logging.debug(f"ECG data extracted for original model with shape: {ecg.shape}")
        logging.debug(f"ECG data shape for original model prediction: {ecg.shape}.")

        preds = {}
        new_model_predictions = {}
        combined_predictions = {}

        # 3. Predict (Original Model)
        preds = await run_in_threadpool(predict_ecg, ecg)
        logging.debug(f"Original model predict outcome: {preds}")
        new_model_predictions = await run_in_threadpool(process_and_predict, temp_dicom_file_path)
        logging.debug(f"New model predict outcome: {new_model_predictions}")
        combined_predictions = {
            "original_dnn_model": preds,
            "resnet_fine_tuned_model": new_model_predictions
        }
        logging.debug(f"Combined predictions: {combined_predictions}")

        # --- MODIFIED LOGIC for storing ---
        # We generate the comment string
        comment = "; ".join(f"{k}:{v:.3f}" for k, v in preds.items())
        comment += " | ResNet Top5: " + "; ".join(f"{k}:{v:.3f}" for k, v in new_model_predictions.items())
        logging.debug(f"Prepared comment string for PatientComments: {comment}")

        # The in-memory modification using add_predictions_to_dicom is not used for /modify approach
        # add_predictions_to_dicom is no longer imported or used.

        # --- MODIFIED CALL to store_instance_dicom ---
        new_instance_id = await run_in_threadpool(store_instance_dicom, instance_id, comment, token)
        # -------------------------------------

        # --- MODIFIED RETURN ---
        # We return the potentially None new_instance_id.
        # The key name remains "modified_instance_id".
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
    """
    ECG diagnostic API for external use
    Simply enter the instance_id to automatically obtain a token and return the diagnostic results.
    Also saves each diagnosis result to a local JSON file.
    """
    logging.info(f"Receive a diagnostic request,Instance ID: {instance_id}")

    try:
        token = req.temp_token
        if not token:
            raise HTTPException(status_code=401, detail="Failed to obtain authentication token")

        logging.debug(f"Successfully obtained token for instance {instance_id}")

        # request = PredictRequest(token=token)
        result = await predict(instance_id, PredictRequest(token=token))

        output = {
            "instance_id": instance_id,
            "status": "success",
            "predictions": result["predictions"],
            "modified_instance_id": result["modified_instance_id"],
            "message": f"Successfully diagnosed ECG instance: {instance_id}"
        }

        return output

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Diagnostic failure,Instance ID: {instance_id}, error: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")

@app.post("/interprete/{instance_id}")
async def interprete(
    instance_id: str,
    disease: List[str] = Form(...),
    temp_token: str = Form(...),
):
    """
    XAI interprete:
    - call run_xai_tasks(abnormalities=disease, instance_id=instance_id)
    """
    logging.info(f"[interprete] instance_id={instance_id}, disease={disease}")

    try:
        token = temp_token
        if not token:
            raise HTTPException(status_code=401, detail="Failed to obtain authentication token")
        abnormalities = disease
        xai_out = await run_in_threadpool(
            run_xai_tasks,
            abnormalities,        # abnormalities
            instance_id,          # instance_id
            token,
            True,                 # return_reports
            True,                 # verbose
        )

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