from fastapi import FastAPI, HTTPException
from dicom_utils import get_instance_dicom, store_instance_dicom
from preprocess import extract_waveform, add_predictions_to_dicom
from model_utils import load_model, predict_ecg

app = FastAPI()

# load models when intialize
from model_utils import load_models
load_models()

import logging

# set log record
logging.basicConfig(level=logging.DEBUG)

@app.post("/predict/{instance_id}")
async def predict(instance_id: str):
    logging.debug(f"Start prediction for instance_id: {instance_id}")
    try:
        # 1. Retrieve DICOM
        ds = get_instance_dicom(instance_id)
        logging.debug(f"DICOM data retrieve success: {ds}")
    except Exception as e:
        logging.error(f"Orthanc retrieve failed: {e}")
        raise HTTPException(status_code=500, detail=f"Orthanc retrieve failed: {e}")

    try:
        # 2. Transfrom to adapt input of models
        ecg, sampling_rate, lead_names = extract_waveform(ds)
        logging.debug(f"ECG data transform success: {ecg.shape}")
    except Exception as e:
        logging.error(f"Transform DICOM failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transform DICOM failed: {e}")
    
    try:
        # 3. Predict
        preds = predict_ecg(ecg)
        logging.debug(f"predict outcome: {preds}")
    except Exception as e:
        logging.error(f"predict failed: {e}")
        raise HTTPException(status_code=500, detail=f"predict failed: {e}")

    try:
        # 4. Write outcomes to DICOM
        ds = add_predictions_to_dicom(ds, preds)
        logging.debug(f"write back to DICOM success")
    except Exception as e:
        logging.error(f"failed to write outcome back to DICOM: {e}")
        raise HTTPException(status_code=500, detail=f"failed to write outcome back to DICOM: {e}")

    try:
        # 5. Upload back to Orthanc
        comment = "; ".join(f"{k}:{v:.3f}" for k, v in preds.items())
        store_instance_dicom(instance_id, comment)
        logging.debug(f"Upload DICOM to Orthanc successfully")
    except Exception as e:
        logging.error(f"DICOM upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"DICOM upload failed: {e}")

    return {"instance_id": instance_id, "predictions": preds}

