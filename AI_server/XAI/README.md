  # Overview

  XAI (explainable AI) module provides comprehensive visual explanations and clinical interpretations for ECG abnormality analysis. It bridges the gap between AI model predictions and clinical understanding by generating detailed visualizations, statistical metrics, and structured reports that help explain why certain abnormalities were detected.

  ## Features

  ## Core Capabilities
  - **Multi-abnormality Analysis**: Supports 12 different ECG abnormalities with specialized analysis modules
  - **Clinical Visualizations**: Generates publication-quality plots showing key diagnostic features
  - **Structured Reports**: Produces standardized JSON reports with summary, statistics, and validation
  - **Automated Metrics**: Calculates clinically relevant parameters (heart rate, intervals, morphology)
  - **Validation Logic**: Built-in validators to ensure diagnostic accuracy and consistency

  ## Supported Abnormalities
  - **Arrhythmias**: AF (Atrial Fibrillation), AFLT (Atrial Flutter), PAC (Premature Atrial Contraction), PVC (Premature Ventricular Contraction)
  - **Conduction Blocks**: 1dAVB (First-degree AV Block), 2dAVB (Second-degree AV Block), 3dAVB (Third-degree AV Block), LBBB (Left Bundle Branch Block), RBBB (Right Bundle Branch Block)
  - **Rate Abnormalities**: STACH (Sinus Tachycardia), SBRAD (Sinus Bradycardia)
  - **Normal**: NORM (Normal ECG) baseline analysis

  ## Visualization Features
  - Rhythm analysis with RR interval histograms and trend plots
  - Morphology analysis showing QRS complexes, P-waves, and interval measurements
  - Spectral analysis for atrial flutter detection
  - Multi-lead comparisons for bundle branch blocks
  - Color-coded diagnostic indicators (tachycardia/bradycardia/normal ranges)

   ## XAI ECG Visualization and Explanation

  Clinically oriented ECG visualizations with structured JSON reports. Supports WFDB and DICOM ECG.
  All report content is centralized via `clinical_reports.ReportGenerator` for consistent outputs across modules.


  ## Report Structure
  Each analysis generates:
  - **Summary**: Clinical interpretation text and abnormality presence indicators
  - **Statistics**: Quantitative metrics (heart rates, intervals, ratios, frequencies)
  - **Validation**: Automated diagnostic validation with pass/warn/fail status
  - **Images**: Visual explanations saved as PNG files with web-accessible URLs

  ## Highlights
  - Abnormalities: AF, AFLT, STACH, SB, LBBB, RBBB, 1dAVB, 2dAVB, 3dAVB, PAC, PVC, NORM
  - Data sources: WFDB records, DICOM ECG waveform (via Orthanc)
  - Outputs:
    - Images per task + `analysis_report.json`
    - Root `results_index.json`
    - Unified `summary`/`report_text`/`statistics`/`validation` via `ReportGenerator`

  ## Layout
  ```
  XAI/
    main.py                  # XAI task runner (called by app.py)
    ecg_utils.py             # ECG data loading utilities
    clinical_reports.py      # ReportGenerator (centralized reports)
    test_comprehensive.py    # Unit tests for XAI model
    xai/explanations/*.py    # Module visuals + metrics only
    static/                  # Generated images (served at /static)
  ```

  ## Usage

  **XAI module is integrated into the main API server. To use XAI functionality:**

  1. Start the API server (see `AIClinic-server/README.md` for setup):
     ```bash
     uvicorn app:app --host 0.0.0.0 --port 8000
     ```

  2. Use the `/interprete/{instance_id}` API endpoint to run XAI analysis:
     ```bash
     curl -X POST "http://localhost:8000/interprete/INSTANCE_ID" \
       -H "Content-Type: application/x-www-form-urlencoded" \
       --data-urlencode "disease=AFIB" \
       --data-urlencode "disease=RBBB" \
       --data-urlencode "temp_token=BEARER_TOKEN"
     ```

  3. XAI results are available:
     - JSON response with structured reports
     - Images accessible at `http://localhost:8000/static/images/{instance_id}/`

  For detailed API documentation, see `AIClinic-server/README.md`.

  ## Output schema (summary)
  - `title`, `lead_name`
  - `summary`: `report_text`, `is_abnormality_present`, `abnormality_count`, optional indices/times
  - `report_text`: sections with `section_title` + `content`
  - `statistics`: key metrics (e.g., heart rate, morphology, timing)
  - `validation`: `status` (pass/warn/fail), human-readable `reasons`, `metrics_used`
  - `image_paths`: filename → path mapping

  ## Notes
  - Debug mode (optional): set `XAI_DEBUG_VERBOSE=1` environment variable to include candidate analysis/raw metrics
  - RBBB requires V1/V2 leads (V6/I recommended as fallback)

  ## Troubleshooting
  - API errors: Check `AIClinic-server/README.md` for API troubleshooting
  - RBBB missing leads: V1/V2 required for proper analysis
  - DICOM without waveform: Ensure DICOM file contains WaveformSequence
  - Images not accessible: Verify `/static` mount point and instance_id path structure
