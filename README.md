# AIClinic-server
FastAPI-based service for end-to-end ECG analysis on DICOM waveforms with Orthanc integration, multi-model inference, and explainable AI (XAI) visualizations. This README ties together the three parts of the project: the API server, the XAI module, and the PTB-XL ResNet integration.

* Pull DICOM from Orthanc using a per-request Bearer token
* Extract and normalize ECG waveforms (12-lead), run 3 model families
* Write results into PatientComments and upload as a new DICOM instance
* Run XAI pipelines and serve artifacts under /static

Paths in this repository:

* API Server: app.py, dicom\_utils.py, preprocess.py, model\_utils.py
* XAI module: XAI/ (details in XAI/README.md)
* PTB-XL ResNet integration: ecg\_ptbxl\_benchmarking\_master/ (expects extract\_and\_predict\_ecg.py)

**Architecture**

* DICOM I/O (Orthanc): Retrieve instance → modify locally → upload as a new instance
* Preprocess: unit calibration (uV/LSB → mV), resample, crop/pad, flatten when needed
* Models:

  * Keras DNN (models/model.hdf5): 12x4096 at 400 Hz
  * ResNet fine-tuned (ecg\_ptbxl\_benchmarking\_master/process\_and\_predict)
  * HuBERT (Edoardo-BS/hubert-ecg-base) + MLP head (models/clf\_head.pt), 5 s @100 Hz → 6000 sample flat
* XAI: task pipeline for AF/AFLT/STACH/SB/LBBB/RBBB/AVB/PAC/PVC/NR with unified JSON reports and images

**Install Dependencies**

* Using requirements.txt

  * The provided requirements.txt pins this project’s dependencies for reproducible setup. Use a fresh virtual environment.
  * GPU users: adjust or reinstall torch/tensorflow to match your CUDA/CuDNN; the file may default to CPU wheels.

Linux/macOS:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Configuration**

* Create a .env at project root:

```
ORTHANC_URL=https://orthanc.example.org
```

* Place models locally:

  * models/model.hdf5 (Keras DNN)
  * models/clf\_head.pt (PyTorch MLP head)
* Ensure ResNet helper is importable:

  * ecg\_ptbxl\_benchmarking\_master/extract\_and\_predict\_ecg.py
  * If not installed as a package: set PYTHONPATH to include the folder
* Optional XAI debug:

  * XAI\_DEBUG\_VERBOSE=1 to enable detailed outputs
* Hugging Face caching/offline:

  * First run downloads hubert-ecg-base; set TRANSFORMERS\_CACHE if needed

**Run**

* Start the API (serves /docs and /static):

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

* The app:

  * Loads models on startup
  * Mounts XAI/static at /static
  * Uses a dedicated thread pool for XAI (prevents blocking Starlette’s default pool)
  * Uses per-request HTTP sessions and explicit closes with timeouts

**API**

* Token handling: pass your Orthanc Bearer token in the request body; the service forwards it in Authorization to Orthanc.

* POST /predict/{instance\_id}

  * Body: {"token":"..."}
  * Runs Keras DNN, ResNet FT, and HuBERT+MLP; composes PatientComments; uploads a new DICOM instance
  * Response: {"modified\_instance\_id":"...", "predictions":{...}}

* POST /diagnose/{instance\_id}

  * Body: {"temp\_token":"..."}
  * Thin wrapper over /predict with a different envelope

* POST /interprete/{instance\_id}

  * Form: disease=\[...] (List\[str]), temp\_token=...
  * Runs XAI tasks in the custom thread pool; returns task results
  * XAI assets are accessible under /static

Example:

```
curl -X POST "http://localhost:8000/predict/INSTANCE_ID" \
  -H "Content-Type: application/json" \
  -d '{"token":"BEARER_TOKEN"}'
```

**Data Flow and DICOM Behavior**

* Retrieve: GET {ORTHANC\_URL}/instances/{id}/file with timeouts (5s connect/15s read)
* Modify: set PatientComments; update SOPInstanceUID and SeriesInstanceUID; set ContentDate/Time
* Upload: POST {ORTHANC\_URL}/instances (Content-Type: application/dicom) to create a new instance
* Original instance remains unchanged

**Models and Labels**

* Keras DNN (predict\_ecg)

  * Outputs: 1dAVB, RBBB, LBBB, SBRAD, AFIB, STACH
  * Input after preprocess: (4096, 12) at 400 Hz
* ResNet fine-tuned (process\_and\_predict)

  * Returns a Top-K-like dict; used in comments as “ResNet Top5”
* HuBERT + MLP head

  * Backbone: Edoardo-BS/hubert-ecg-base; pooled embedding fed to MLP head
  * Labels: \["1AVB","CRBBB","CLBBB","AFIB","STACH","2AVB","3AVB","PAC","PVC","AFLT"]
  * API returns subset in the response and writes 4 selected labels in PatientComments

**Preprocessing**

* DICOM WaveformSequence\[0]: int16 → float32, apply ChannelSensitivity when present; fallback 5 uV/LSB
* Resample:

  * DNN: to 400 Hz; crop/pad to 4096
  * HuBERT: FIR bandpass, resample to 100 Hz, scale to unit, 5 s center crop/pad; flatten to 6000

**Project Layout**

```
app.py
dicom_utils.py
preprocess.py
model_utils.py
models/
  model.hdf5
  clf_head.pt
XAI/
  README.md
  main.py
  static/...
ecg_ptbxl_benchmarking_master/
  extract_and_predict_ecg.py
requirements.txt
```

**XAI Module (Summary)**

* Generates clinical visual explanations and JSON reports per abnormality
* Unified output schema: summary + statistics + validation + image\_paths
* See XAI/README.md for tasks, configuration, debug mode, and examples
* Run standalone: python XAI/main.py config.json

**Performance and Ops**

* Custom XAI thread pool (max\_workers=4) to avoid thread starvation
* Per-request requests.Session with explicit close; GC hints in middleware
* PyTorch device selection: CUDA when available; else CPU
* Start with --workers 1 unless you size memory to load multiple model copies per worker

**Security and Privacy**

* Tokens are not persisted; forwarded to Orthanc per request
* Use HTTPS for Orthanc and the API endpoint
* Logs include IDs and timings; avoid sensitive data at higher log levels in production

**Troubleshooting**

* 401/403: token invalid/expired; ensure the token has Orthanc access scope
* 404: instance not found at Orthanc\_URL; verify instance\_id and base URL
* 500 Prediction failed:

  * Missing models/model.hdf5 or models/clf\_head.pt
  * ecg\_ptbxl\_benchmarking\_master not importable (set PYTHONPATH)
  * HuBERT backbone offline with no cache
  * DICOM has no WaveformSequence or unexpected structure
* TensorFlow/PyTorch conflicts: prefer CPU builds or isolate via venv/conda; reduce TF logs via TF\_CPP\_MIN\_LOG\_LEVEL=2

**Licensing and Use**

* Academic/learning project; not for clinical use without proper validation
* Acknowledge Hugging Face model Edoardo-BS/hubert-ecg-base, NumPy/SciPy/WFDB/pydicom, and Orthanc