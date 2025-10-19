# ECG AI Server (FastAPI + DICOM + Orthanc)

Production‑ready FastAPI service for ECG analysis on DICOM waveforms. It pulls ECG waveforms from an Orthanc server, runs multiple ML models (Keras DNN, ResNet fine‑tuned, HuBERT+MLP head), writes predictions back to a new DICOM instance, and optionally runs XAI interpretation workflows.

* DICOM in, predictions out; Orthanc used for retrieval and storage
* Multi‑model inference with consistent preprocessing
* New DICOM instance is created with results in PatientComments
* XAI routines executed in a dedicated thread pool; static XAI assets served at `/static`
* Built‑in timeouts, per‑request sessions, and GC for stability

## Contents

* app: `app.py`
* DICOM I/O: `dicom_utils.py`
* Preprocessing: `preprocess.py`
* Models and inference: `model_utils.py`
* XAI (external): `XAI/` (expects `XAI/main.py` and `XAI/static/`)
* Additional model: `ecg_ptbxl_benchmarking_master/` (expects `extract_and_predict_ecg.py`)

## Architecture

* Retrieve DICOM waveform from Orthanc via Bearer token
* Extract and calibrate 12‑lead ECG, resample, crop/pad
* Run three predictors:

  * Original Keras DNN (`models/model.hdf5`)
  * ResNet fine‑tuned (via `process_and_predict(path_to_dcm)`)
  * HuBERT backbone (HuggingFace) + local MLP head (`models/clf_head.pt`)
* Compose a comment string, write to PatientComments (0010,4000), update UIDs, upload as a new Orthanc instance
* Optionally run XAI for selected abnormalities; results available via `/static`

## Prerequisites

* Python 3.9+ recommended
* Orthanc server reachable via HTTP(S) with a Bearer token workflow
* Model artifacts placed locally:

  * `models/model.hdf5` (Keras)
  * `models/clf_head.pt` (PyTorch)
* Internet access on first run to fetch HuBERT backbone from Hugging Face (unless cached/offline)
* The external modules are available/importable:

  * `ecg_ptbxl_benchmarking_master/extract_and_predict_ecg.py`
  * `XAI/main.py` and `XAI/static/`

Install Dependencies

* Using requirements.txt

  * The provided requirements.txt pins the project’s dependencies for reproducible setup. Install everything in a fresh virtual environment.
  * Note: If you need GPU, edit the torch/tensorflow lines in requirements.txt to match your CUDA/cuDNN stack or install the appropriate wheels after the base install.

```
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Set the Orthanc base URL via `.env` or environment variable:

```
ORTHANC_URL=https://orthanc.example.org
```

Optional (to resolve external modules without installing as packages):

```
export PYTHONPATH="$PYTHONPATH:$(pwd)/ecg_ptbxl_benchmarking_master"
```

HuBERT backbone:

* Model id: `Edoardo-BS/hubert-ecg-base`
* Downloads on first use; cached by `transformers`

## Running

Start the API:

```
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

* Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Static XAI assets: [http://localhost:8000/static/](http://localhost:8000/static/)

The app loads models on startup, mounts `XAI/static`, and uses a custom thread pool for XAI tasks to avoid blocking the default Starlette pool.

## API

All endpoints require an Orthanc Bearer token in the request body (not as an HTTP header to this service). The service forwards it to Orthanc.

### POST /predict/{instance\_id}

Run all models and write results to a new DICOM instance in Orthanc.

Request (JSON):

```json
{
  "token": "YOUR_ORTHANC_BEARER_TOKEN"
}
```

Response (JSON):

```json
{
  "modified_instance_id": "new-orthanc-instance-id",
  "predictions": {
    "original_dnn_model": {"1dAVB": 0.12, "RBBB": 0.34, "LBBB": 0.01, "SBRAD": 0.07, "AFIB": 0.45, "STACH": 0.02},
    "resnet_fine_tuned_model": {"LABEL_A": 0.91, "LABEL_B": 0.04, "...": 0.01},
    "hubert_mlp_head_model": {"CRBBB": 0.11, "CLBBB": 0.05, "PVC": 0.22, "AFLT": 0.09}
  }
}
```

Notes:

* The service composes `PatientComments` with Original DNN, ResNet Top‑5, and HuBERT‑Head scores.
* It modifies SOPInstanceUID and SeriesInstanceUID and sets ContentDate/Time before upload, ensuring a new instance is created. The original instance is not changed.

Example:

```
curl -X POST "http://localhost:8000/predict/INSTANCE_ID" \
  -H "Content-Type: application/json" \
  -d '{"token":"BEARER_TOKEN"}'
```

### POST /diagnose/{instance\_id}

Thin wrapper around `/predict` with a different body key, returns a slightly different envelope.

Request (JSON):

```json
{
  "temp_token": "YOUR_ORTHANC_BEARER_TOKEN"
}
```

Response (JSON):

```json
{
  "instance_id": "INSTANCE_ID",
  "status": "success",
  "predictions": { ...same as /predict... },
  "modified_instance_id": "new-orthanc-instance-id",
  "message": "Successfully diagnosed ECG instance: INSTANCE_ID"
}
```

### POST /interprete/{instance\_id}

Run XAI interpretation for selected abnormalities. Uses a custom thread pool. Form‑encoded body.

Request (form-data):

* `disease`: List\[str] (one or more form fields with the same key)
* `temp_token`: str

Example:

```
curl -X POST "http://localhost:8000/interprete/INSTANCE_ID" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "disease=AFIB" \
  --data-urlencode "disease=RBBB" \
  --data-urlencode "temp_token=BEARER_TOKEN"
```

Response (JSON):

```json
{
  "instance_id": "INSTANCE_ID",
  "abnormalities": ["AFIB", "RBBB"],
  "status": "success",
  "results": [ ... task-specific XAI outputs ... ],
  "message": "XAI interpretation completed."
}
```

## Models and Labels

* Original DNN (Keras, expects 12x4096 at 400 Hz after preprocessing):

  * Outputs: `1dAVB, RBBB, LBBB, SBRAD, AFIB, STACH`
* ResNet fine‑tuned (external, returns Top‑k dictionary)
* HuBERT + MLP head (PyTorch):

  * Backbone: `Edoardo-BS/hubert-ecg-base`
  * Head: `models/clf_head.pt`
  * Full label set: `["1AVB","CRBBB","CLBBB","AFIB","STACH","2AVB","3AVB","PAC","PVC","AFLT"]`
  * API returns a subset: `CRBBB, CLBBB, PVC, AFLT`

## DICOM/Orthanc Behavior

* Retrieval: `GET {ORTHANC_URL}/instances/{instance_id}/file` (per‑request session, timeouts 5s connect / 15s read)
* Upload: `POST {ORTHANC_URL}/instances` with `Content-Type: application/dicom`
* PatientComments content:

  * Original DNN scores
  * `ResNet Top5: ...`
  * `HuBERT-Head: ...`
* New instance creation is guaranteed by updating UIDs and date/time

## Preprocessing Overview

* Extract 12‑lead waveform from `WaveformSequence`
* Calibrate to mV via `ChannelSensitivity` when available (fallback to 5 uV/LSB)
* Resample to 400 Hz (DNN) or 100 Hz + flatten 5 s segment (HuBERT)
* Crop/pad for fixed length inputs

## Performance and Reliability

* Dedicated XAI thread pool (`max_workers=4`) to avoid starving Starlette’s default pool
* Explicit per‑request `requests.Session()` and `.close()` to avoid connection pool exhaustion
* Conservative HTTP timeouts for Orthanc I/O
* Opportunistic `gc.collect()` after requests and in middleware
* CPU/GPU selection for PyTorch is automatic (`cuda` if available)

## Troubleshooting

* 401 Unauthorized: Provide a valid `token`/`temp_token`; the service forwards it as `Authorization: Bearer ...` to Orthanc.
* 404 from Orthanc: Verify `ORTHANC_URL` and `instance_id`.
* 500 Prediction failed:

  * Missing `models/model.hdf5` or `models/clf_head.pt`
  * HuBERT backbone unavailable offline without cache
  * Import path for `ecg_ptbxl_benchmarking_master` missing; add it to `PYTHONPATH`
  * DICOM has no `WaveformSequence` or unexpected structure
* TensorFlow/PyTorch conflicts: Prefer CPU builds, or isolate with virtualenv/conda. Reduce TF logs with `TF_CPP_MIN_LOG_LEVEL=2`.

## Security and Privacy

* This service does not store tokens; it forwards them to Orthanc per request.
* Logs avoid printing tokens but include IDs and timing; review logging level in production.
* Handle PHI according to your organizational policies; ensure TLS to Orthanc and the API gateway.

## Acknowledgements

* HuBERT ECG backbone: `Edoardo-BS/hubert-ecg-base` (Hugging Face)
* Waveform processing uses NumPy/SciPy/WFDB and pydicom
* Orthanc used for DICOM storage and retrieval