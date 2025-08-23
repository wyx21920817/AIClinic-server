# dicom_utils.py

import requests
from pydicom import dcmread, Dataset
from io import BytesIO
import json
import logging
import os

ORTHANC_URL  = os.getenv("ORTHANC_URL", "http://54.79.181.49:8042")
# ORTHANC_AUTH = ("orthanc", "orthanc") # REMOVED

logging.basicConfig(level=logging.DEBUG)

# Assuming get_headers function exists elsewhere or define it here if needed
# If you don't have get_headers, replace get_headers(token) with {"Authorization": f"Bearer {token}"}
def get_headers(token: str):
     return {"Authorization": f"Bearer {token}"}

# You might need ORTHANC_SERVER_URL defined if your get_headers or other parts use it
ORTHANC_SERVER_URL = ORTHANC_URL


# get_instance_dicom function remains the same
def get_instance_dicom(instance_id: str, token: str) -> Dataset:
    """
    拉取 DICOM 并返回 pydicom.Dataset, 使用 token 进行认证.
    Args:
        instance_id: Orthanc instance ID.
        token: The temporary access token.
    Returns:
        pydicom.Dataset object.
    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    logging.debug(f"Attempting to retrieve DICOM instance: {instance_id} from {ORTHANC_URL} using token.")
    url = f"{ORTHANC_URL}/instances/{instance_id}/file"

    headers = {
        "Authorization": f"Bearer {token}"
    }
    r = None
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        logging.debug(f"Successfully retrieved DICOM instance: {instance_id}")
        return dcmread(BytesIO(r.content))
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve DICOM instance {instance_id}: {e}")
        if hasattr(r, 'text'):
            logging.error(f"Orthanc response text: {r.text}")
        raise


# --- MODIFIED store_instance_dicom function using the modify+upload strategy ---
# This implements the two-step process: modify via /modify, then upload resulting binary via /instances.
def store_instance_dicom(instance_id: str, comment: str, token: str) -> str | None:
    """
    Modify instance via /instances/{id}/modify to get modified binary data,
    then upload the resulting binary data via /instances to get the new instance ID.
    """
    logging.debug(f"Attempting modify+upload strategy for instance {instance_id}.")
    modify_url = f"{ORTHANC_SERVER_URL}/instances/{instance_id}/modify"

    # Headers for the modify request (Content-Type: application/json)
    modify_headers = get_headers(token)
    modify_headers["Content-Type"] = "application/json"

    # Payload for the modify request
    modify_payload = {
        "Replace": {
            "PatientComments": comment # Modify PatientComments tag (0010,4000)
            # If using private tags like classmate's code, add them here with tag numbers as keys
            # e.g., "00091010": "Some notes",
            # "00091011": "Some tags"
            # Remember to add PrivateCreator if using private tags
            # "PrivateCreator": "YourPrivateCreator" # In modify_payload root if needed
        },
        "Transcode": "1.2.840.10008.1.2.1", # Lossless transfer syntax
        "KeepSource": True # Keep the original instance
    }

    modify_response = None
    new_instance_id = None # Initialize new_instance_id as None

    try:
        # Step 1: Call /modify
        logging.debug(f"Calling /modify endpoint for instance {instance_id}.")
        modify_response = requests.post(
            modify_url,
            json=modify_payload,
            headers=modify_headers
        )
        modify_response.raise_for_status() # Check for 4xx/5xx errors. If raised, caught by outer except.

        # If modify is successful (status 2xx)
        logging.debug(f"/modify request for {instance_id} completed with status {modify_response.status_code}. Response body length: {len(modify_response.content)}")
        # The response content is the modified DICOM binary data

        # Step 2: Upload the received binary data to /instances
        upload_url = f"{ORTHANC_SERVER_URL}/instances"
        # Headers for the upload request (Content-Type: application/dicom)
        upload_headers = get_headers(token) # Get headers again for the new request
        upload_headers["Content-Type"] = "application/dicom" # Set Content-Type for binary upload

        logging.debug(f"Attempting to upload modified instance data to {upload_url}.")
        upload_response = requests.post(
            upload_url,
            headers=upload_headers,
            data=modify_response.content # Send the binary content from the modify response
        )
        upload_response.raise_for_status() # Check for 4xx/5xx errors on upload. If raised, caught by outer except.

        # If upload is successful (status 2xx)
        logging.debug(f"Upload request completed with status {upload_response.status_code}.")

        # Step 3: Parse the JSON response from the upload to get the new instance ID
        try:
            upload_result = upload_response.json()
            new_instance_id = upload_result.get("ID") # Get the ID from the upload response JSON

            if not new_instance_id:
                logging.error(f"Upload successful (status {upload_response.status_code}) but JSON response did not contain expected 'ID': {upload_result}")
                # new_instance_id remains None
            else:
                logging.debug(f"Successfully obtained new instance ID '{new_instance_id}' from upload response.")

        except json.JSONDecodeError:
            logging.error(f"Upload successful (status {upload_response.status_code}) but response was not JSON. Cannot get new instance ID from upload response.")
            # new_instance_id remains None

    except requests.exceptions.RequestException as e:
        # Catch network errors, timeouts, and HTTP errors (4xx/5xx) during either modify or upload.
        logging.error(f"RequestException during modify+upload for instance {instance_id}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Orthanc response status code: {e.response.status_code}")
            # Safely attempt to log response text if available
            try:
                logging.error(f"Orthanc response text: {e.response.text}")
            except: # Catch potential error if response.text is not accessible (e.g. binary)
                logging.error(f"Orthanc response content (first 200 bytes): {e.response.content[:200]}...")
        new_instance_id = None # Ensure new_instance_id is None on failure

    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected error occurred during modify+upload process: {e}")
        new_instance_id = None

    # Return the new_instance_id (which will be the actual ID or None)
    return new_instance_id