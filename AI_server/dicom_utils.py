# dicom_utils_fixed.py

import requests
from pydicom import dcmread, Dataset, dcmwrite
from io import BytesIO
import json
import logging
import os
import time
import psutil
import gc
from typing import Union, Optional
from dotenv import load_dotenv

load_dotenv()
ORTHANC_URL = os.getenv("ORTHANC_URL", "")
ORTHANC_SERVER_URL = ORTHANC_URL

logging.basicConfig(level=logging.DEBUG)

def get_headers(token: str):
    """Return headers with authentication token"""
    return {"Authorization": f"Bearer {token}"}


def get_instance_dicom(instance_id: str, token: str) -> Dataset:
    """
    Retrieve DICOM and return pydicom.Dataset, using token for authentication.
    Args:
        instance_id: Orthanc instance ID.
        token: The temporary access token.
    Returns:
        pydicom.Dataset object.
    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    start_time = time.time()
    logging.debug(f"[REQUEST] Attempting to retrieve DICOM instance: {instance_id} from {ORTHANC_URL} using token at {start_time:.2f}")
    url = f"{ORTHANC_URL}/instances/{instance_id}/file"
    
    # Create a new session for each request to avoid connection pool resource exhaustion
    session = requests.Session()
    logging.debug(f"[SESSION] Created new session for get_instance_dicom {instance_id} with id {id(session)}")

    headers = {
        "Authorization": f"Bearer {token}"
    }
    r = None
    try:
        r = session.get(url, headers=headers, timeout=(5, 15))  # Connection timeout 5s, read timeout 15s
        r.raise_for_status()
        end_time = time.time()
        logging.debug(f"[TIMING] Retrieved DICOM instance: {instance_id} in {end_time - start_time:.2f} seconds")
        result = dcmread(BytesIO(r.content))
        
        # Explicitly close session and response
        try:
            r.close()
            session.close()
            logging.debug(f"[SESSION] Session for get_instance_dicom {instance_id} (id: {id(session)}) closed successfully")
        except Exception as e:
            logging.warning(f"[ERROR] Error closing session for get_instance_dicom {instance_id}: {e}")
            
        return result
    except requests.exceptions.RequestException as e:
        logging.error(f"[ERROR] Failed to retrieve DICOM instance {instance_id}: {e}")
        if hasattr(r, 'text'):
            logging.error(f"[ERROR] Orthanc response text: {r.text}")
        # Ensure session is closed even in case of exception
        try:
            if r is not None and hasattr(r, 'close'):
                r.close()
            session.close()
            logging.debug(f"[SESSION] Session for get_instance_dicom {instance_id} closed after error")
        except Exception as close_error:
            logging.warning(f"[ERROR] Error closing session after error for {instance_id}: {close_error}")
        raise


def store_instance_dicom(instance_id: str, comment: str, token: str, ds: Dataset = None) -> Optional[str]:
    """
    Modify DICOM dataset and upload as a new instance, returning the new instance ID.
    The original DICOM instance remains unchanged, this function creates a new instance.
    If the ds parameter is provided, it uses that dataset directly; otherwise, it will first download the DICOM data.
    
    Args:
        instance_id: Original Orthanc instance ID
        comment: Comment to add to the PatientComments tag
        token: Authentication token
        ds: Optional, already loaded DICOM dataset
    
    Returns:
        Optional[str]: Newly created instance ID, or None if failed
    """
    # Record current process memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    logging.debug(f"[MEMORY] Before store_instance_dicom for {instance_id}: {mem_before:.2f} MB")
    
    # Record start time
    start_time = time.time()
    logging.debug(f"[TIMING] Starting local modify+upload strategy for instance {instance_id} at {start_time:.2f}")
    
    try:
        # If no DICOM dataset is provided, download it
        if ds is None:
            ds = get_instance_dicom(instance_id, token)
            logging.debug(f"Downloaded DICOM data for instance {instance_id}")
        
        # Modify DICOM dataset locally
        ds.PatientComments = comment
        logging.debug(f"Modified PatientComments field with: {comment[:50]}...")
        
        # Modify DICOM unique identifiers to ensure a new instance is created
        import uuid
        # Modify SOPInstanceUID, which is the instance-level unique identifier
        if hasattr(ds, 'SOPInstanceUID'):
            original_uid = ds.SOPInstanceUID
            ds.SOPInstanceUID = f"1.2.826.0.1.3680043.{uuid.uuid4().int % 10000000000}"
            logging.debug(f"Changed SOPInstanceUID from {original_uid} to {ds.SOPInstanceUID}")
        
        # Modify SeriesInstanceUID, which is the series-level unique identifier
        if hasattr(ds, 'SeriesInstanceUID'):
            original_series_uid = ds.SeriesInstanceUID
            ds.SeriesInstanceUID = f"1.2.826.0.1.3680043.{uuid.uuid4().int % 10000000000}"
            logging.debug(f"Changed SeriesInstanceUID from {original_series_uid} to {ds.SeriesInstanceUID}")
        
        # Add timestamp to ensure the instance is unique
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ds.ContentDate = datetime.now().strftime("%Y%m%d")
        ds.ContentTime = datetime.now().strftime("%H%M%S")
        logging.debug(f"Added timestamp {timestamp} to DICOM content")
        
        # Write the modified data to memory
        buffer = BytesIO()
        dcmwrite(buffer, ds)
        buffer.seek(0)
        modified_data = buffer.getvalue()
        
        # Upload the modified DICOM, creating a new instance instead of overwriting the original
        # Using the /instances endpoint will automatically generate a new instance ID
        url = f"{ORTHANC_SERVER_URL}/instances"
        headers = get_headers(token)
        headers["Content-Type"] = "application/dicom"
        
        logging.debug(f"[NEW_INSTANCE] Creating new DICOM instance from modified data of {instance_id}")
        
        # Create a new session
        session = requests.Session()
        
        try:
            logging.debug(f"[LOCAL] Uploading modified DICOM to {url}")
            response = session.post(url, headers=headers, data=modified_data, timeout=(5, 15))
            response.raise_for_status()
            
            # Parse response to get the new instance ID
            result = response.json()
            new_instance_id = result.get("ID")
            
            if new_instance_id:
                logging.debug(f"[NEW_INSTANCE] Successfully created new DICOM instance {new_instance_id} from original instance {instance_id}")
                logging.debug(f"[NEW_INSTANCE] Original instance {instance_id} remains unchanged")
            else:
                logging.error(f"[LOCAL] Upload successful but no instance ID in response: {result}")
                
            return new_instance_id
        finally:
            session.close()
            
    except Exception as e:
        logging.error(f"[LOCAL] Modify and upload failed: {e}")
        return None
    finally:
        # Record memory usage at the end
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        end_time = time.time()
        logging.debug(f"[MEMORY] After store_instance_dicom for {instance_id}: {mem_after:.2f} MB (Diff: {mem_after - mem_before:.2f} MB)")
        logging.debug(f"[TIMING] Completed local modify+upload strategy for instance {instance_id} in {end_time - start_time:.2f} seconds")
        
        # Force garbage collection
        gc.collect()
