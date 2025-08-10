# check_dicom_comments.py

import requests
import json
import logging
from pydicom import dcmread, Dataset
from io import BytesIO

# Orthanc 服务器的 URL
ORTHANC_URL = "http://54.79.181.49:8042"

# 认证服务的 URL，用于获取临时 token
AUTH_SERVICE_URL = "http://54.79.181.49:8000/api/auth/temp-token"

# 要检查的 DICOM 实例 ID (使用 test.py 中成功处理的那个 ID)
INSTANCE_ID_TO_CHECK = "79f1c5f9-356696cd-4391a2cb-8d3713c1-a756d2f3"

# Patient Comments 标签的 Group 和 Element
PATIENT_COMMENTS_TAG = "0010-4000"

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# --- Function to get temporary token (remains the same) ---
def get_temp_token() -> str | None:
    """
    调用认证服务 API 获取临时访问 token.
    Returns:
        str: 获取到的临时 token, 如果失败则返回 None.
    """
    print(f"Attempting to obtain temporary token from {AUTH_SERVICE_URL}")
    try:
        response = requests.get(AUTH_SERVICE_URL)
        response.raise_for_status()

        token_data = response.json()
        temp_token = token_data.get("token")

        if temp_token:
            print(f"Successfully obtained temporary token (expires in: {token_data.get('expires_in')} seconds).")
            return temp_token
        else:
            print(f"Error: Token response did not contain 'token' field. Response: {token_data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to obtain temporary token: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Auth service response text: {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during token acquisition: {e}")
        return None

# --- Function to check Patient Comments using token (remains the same) ---
def check_patient_comments(instance_id: str, token: str):
    """
    使用 token 调用 Orthanc API 获取指定实例的 Patient Comments 内容.
    """
    url = f"{ORTHANC_URL}/instances/{instance_id}/content/{PATIENT_COMMENTS_TAG}"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    print(f"\nAttempting to retrieve PatientComments for instance: {instance_id}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        comments_content = response.text

        print(f"Successfully retrieved PatientComments for instance {instance_id}:")
        print(f"--- Start PatientComments ---")
        print(comments_content)
        print(f"--- End PatientComments ---")

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve PatientComments for instance {instance_id}: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
             print(f"Orthanc response status code: {e.response.status_code}")
             print(f"Orthanc response text: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred during comments retrieval: {e}")


# --- Function to get Orthanc Patient ID using DICOM Patient ID (Iterative search) ---
def find_orthanc_patient_id_by_dicom_id_iterative(dicom_patient_id: str, token: str) -> str | None:
    """
    Iterates through all patients in Orthanc to find the one matching the given DICOM Patient ID.
    This is a fallback if POST /patients?query is not available (due to 405 error).
    """
    print(f"\nAttempting to find Orthanc patient ID for DICOM Patient ID '{dicom_patient_id}' by iterating through all patients.")
    list_patients_url = f"{ORTHANC_URL}/patients"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.get(list_patients_url, headers=headers)
        response.raise_for_status()
        all_patient_orthanc_ids = response.json()

        print(f"Retrieved a list of {len(all_patient_orthanc_ids)} patient(s). Checking each one...")

        for patient_orthanc_id in all_patient_orthanc_ids:
            get_patient_details_url = f"{ORTHANC_URL}/patients/{patient_orthanc_id}"
            try:
                detail_response = requests.get(get_patient_details_url, headers=headers)
                detail_response.raise_for_status()
                patient_details = detail_response.json()

                main_dicom_tags = patient_details.get("MainDicomTags", {})
                found_dicom_patient_id = main_dicom_tags.get("PatientID")

                if found_dicom_patient_id == dicom_patient_id:
                    print(f"Match found! Orthanc Patient ID is: {patient_orthanc_id}")
                    return patient_orthanc_id

            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to get details for patient {patient_orthanc_id}: {e}. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: An unexpected error occurred while checking patient {patient_orthanc_id}: {e}. Skipping.")
                continue

        print(f"No Orthanc patient found matching DICOM Patient ID '{dicom_patient_id}' after checking all patients.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to list all patients from Orthanc: {e}")
        if hasattr(response, 'text'):
             print(f"Orthanc response status code: {response.status_code}")
             print(f"Orthanc response text: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during iterative patient search: {e}")
        return None


# --- Function to list instance IDs by Orthanc Patient ID (remains the same) ---
def list_instances_by_patient_orthanc_id(patient_orthanc_id: str, token: str):
    """
    Lists all instance IDs associated with a given Orthanc patient ID.
    """
    print(f"\nAttempting to list instances for patient: {patient_orthanc_id}")
    url = f"{ORTHANC_URL}/patients/{patient_orthanc_id}/instances"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        instance_ids = response.json()

        print(f"Successfully retrieved {len(instance_ids)} instance(s) for patient {patient_orthanc_id}:")
        instance_ids.sort()
        for instance_id in instance_ids:
            print(f"- {instance_id}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to list instances for patient {patient_orthanc_id}: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
             print(f"Orthanc response status code: {response.status_code}")
             print(f"Orthanc response text: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred listing instances by patient ID: {e}")


# --- ADDED: get_instance_dicom function needed in check_dicom_comments.py ---
def get_instance_dicom(instance_id: str, token: str) -> Dataset:
    """
    拉取 DICOM 并返回 pydicom.Dataset, 使用 token 进行认证.
    (Copied from dicom_utils.py)
    """
    logging.debug(f"Attempting to retrieve DICOM instance: {instance_id} from {ORTHANC_URL} using token.")
    url = f"{ORTHANC_URL}/instances/{instance_id}/file"

    headers = {
        "Authorization": f"Bearer {token}"
    }

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


if __name__ == "__main__":
    temp_token = get_temp_token()

    if temp_token:
        check_patient_comments(INSTANCE_ID_TO_CHECK, temp_token)

        ds_original = None
        dicom_patient_id = None
        patient_orthanc_id = None

        try:
            print(f"\nGetting DICOM Patient ID and checking for PatientComments tag from instance {INSTANCE_ID_TO_CHECK}...")
            ds_original = get_instance_dicom(INSTANCE_ID_TO_CHECK, temp_token)

            # --- ADDED: Check for PatientComments tag in original instance ---
            if 'PatientComments' in ds_original:
                print(f"DEBUG: Original instance {INSTANCE_ID_TO_CHECK} HAS the PatientComments tag.")
                # Optional: print its value if present
                # print(f"DEBUG: Original PatientComments value: '{ds_original.PatientComments}'")
            else:
                print(f"DEBUG: Original instance {INSTANCE_ID_TO_CHECK} DOES NOT HAVE the PatientComments tag.")
            # -----------------------------------------------------------------

            if 'PatientID' in ds_original:
                 dicom_patient_id = ds_original.PatientID
                 print(f"Original DICOM Patient ID found: '{dicom_patient_id}'")

                 patient_orthanc_id = find_orthanc_patient_id_by_dicom_id_iterative(dicom_patient_id, temp_token)

            else:
                 print(f"Error: 'PatientID' tag (0010,0020) not found in DICOM instance {INSTANCE_ID_TO_CHECK}.")

        except Exception as e:
            print(f"\nAn error occurred while trying to get DICOM Patient ID, check tags, or during iterative search: {e}")


        if patient_orthanc_id:
            list_instances_by_patient_orthanc_id(patient_orthanc_id, temp_token)
        else:
             print(f"\nCould not list instances under patient ID as Patient ID lookup failed for DICOM Patient ID: '{dicom_patient_id}'" if dicom_patient_id is not None else "\nCould not list instances under patient ID as DICOM Patient ID was not found or lookup failed.")

    else:
        print("\nSkipping checks due to token acquisition failure.")