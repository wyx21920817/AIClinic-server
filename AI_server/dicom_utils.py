import requests
from pydicom import dcmread, Dataset
from io import BytesIO
import json
ORTHANC_URL  = "http://54.79.224.136:8042"
ORTHANC_AUTH = ("orthanc", "orthanc")

def get_instance_dicom(instance_id: str) -> Dataset:
    """拉取 DICOM 并返回 pydicom.Dataset."""
    r = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/file",
                     auth=ORTHANC_AUTH)
    r.raise_for_status()
    return dcmread(BytesIO(r.content))



def store_instance_dicom(instance_id: str, new_patient_comments: str) -> None:
    url = f"{ORTHANC_URL}/instances/{instance_id}/modify"
    payload = {
        "Replace": {
            "OtherPatientIDs": new_patient_comments  # 修正标签
        },
        "Transcode": "1.2.840.10008.1.2.1"
    }
    
    try:
        response = requests.post(
            url,
            auth=ORTHANC_AUTH,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        # 获取新实例的 ID
        result = response.json()
        new_instance_id = result["ID"]
        print(f"新实例 ID: {new_instance_id}")
        
        # 验证修改后的标签
        check_modified_instance(new_instance_id)
        
    except Exception as e:
        print(f"请求失败: {e}")
        print(f"响应内容: {response.text}")

def check_modified_instance(instance_id: str):
    url = f"{ORTHANC_URL}/instances/{instance_id}/content/0010-4000"
    r = requests.get(url, auth=ORTHANC_AUTH)
    print(f"PatientComments 值: {r.text}")

