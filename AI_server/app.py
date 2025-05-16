# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import the modified store_instance_dicom function (now returns str | None)
from dicom_utils import get_instance_dicom, store_instance_dicom
from preprocess import extract_waveform # add_predictions_to_dicom is no longer used for /modify
from model_utils import load_model, predict_ecg

app = FastAPI()

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
    try:
        # 1. Retrieve DICOM (Still need to retrieve to get waveform)
        ds = get_instance_dicom(instance_id, token)
        logging.debug(f"DICOM data retrieve success.")
    except Exception as e:
        logging.error(f"Orthanc retrieve failed for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Orthanc retrieve failed: {e}")

    try:
        # 2. Transform to adapt input of models
        ecg, sampling_rate, lead_names = extract_waveform(ds)
        logging.debug(f"ECG data extracted with shape: {ecg.shape}")

        # Remove the transpose as per previous fix
        # ecg = ecg.T

        logging.debug(f"ECG data shape for prediction: {ecg.shape}.")

    except Exception as e:
        logging.error(f"Transform DICOM failed for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Transform DICOM failed: {e}")

    preds = {}
    try:
        # 3. Predict
        preds = predict_ecg(ecg)
        logging.debug(f"predict outcome: {preds}")
    except Exception as e:
        logging.error(f"predict failed for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=f"predict failed: {e}")

    # --- MODIFIED LOGIC for storing ---
    # We generate the comment string
    comment = "; ".join(f"{k}:{v:.3f}" for k, v in preds.items())
    logging.debug(f"Prepared comment string for PatientComments: {comment}")

    # The in-memory modification using add_predictions_to_dicom is not used for /modify approach
    # add_predictions_to_dicom is no longer imported or used.

    # --- MODIFIED CALL to store_instance_dicom ---
    # This function might return None if the new ID cannot be obtained from the response.
    # Removed the try/except block as store_instance_dicom now handles its own exceptions
    # and returns None on failure.
    new_instance_id = store_instance_dicom(instance_id, comment, token)
    # -------------------------------------


    # --- MODIFIED RETURN ---
    # We return the potentially None new_instance_id.
    # The key name remains "modified_instance_id".
    return {"modified_instance_id": new_instance_id, "predictions": preds}