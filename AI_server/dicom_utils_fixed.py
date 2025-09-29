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
    拉取 DICOM 并返回 pydicom.Dataset, 使用 token 进行认证.
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
    
    # 为每个请求创建新的会话，避免连接池资源耗尽问题
    session = requests.Session()
    logging.debug(f"[SESSION] Created new session for get_instance_dicom {instance_id} with id {id(session)}")

    headers = {
        "Authorization": f"Bearer {token}"
    }
    r = None
    try:
        r = session.get(url, headers=headers, timeout=(5, 15))  # 连接超时5秒，读取超时15秒
        r.raise_for_status()
        end_time = time.time()
        logging.debug(f"[TIMING] Retrieved DICOM instance: {instance_id} in {end_time - start_time:.2f} seconds")
        result = dcmread(BytesIO(r.content))
        
        # 显式关闭会话和响应
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
        # 确保即使在异常情况下也关闭会话
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
    修改DICOM数据集并上传，返回新的实例ID。
    如果提供了ds参数，则直接使用该数据集；否则，会先下载DICOM数据。
    
    Args:
        instance_id: Orthanc实例ID
        comment: 要添加到PatientComments标签的注释
        token: 认证令牌
        ds: 可选，已经加载的DICOM数据集
    
    Returns:
        Optional[str]: 新的实例ID，如果失败则为None
    """
    # 记录当前进程的内存使用情况
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    logging.debug(f"[MEMORY] Before store_instance_dicom for {instance_id}: {mem_before:.2f} MB")
    
    # 记录开始时间
    start_time = time.time()
    logging.debug(f"[TIMING] Starting local modify+upload strategy for instance {instance_id} at {start_time:.2f}")
    
    try:
        # 如果没有提供 DICOM 数据集，则下载
        if ds is None:
            ds = get_instance_dicom(instance_id, token)
            logging.debug(f"Downloaded DICOM data for instance {instance_id}")
        
        # 在本地修改DICOM数据集
        ds.PatientComments = comment
        logging.debug(f"Modified PatientComments field with: {comment[:50]}...")
        
        # 将修改后的数据写入内存
        buffer = BytesIO()
        dcmwrite(buffer, ds)
        buffer.seek(0)
        modified_data = buffer.getvalue()
        
        # 上传修改后的DICOM
        url = f"{ORTHANC_SERVER_URL}/instances"
        headers = get_headers(token)
        headers["Content-Type"] = "application/dicom"
        
        # 创建新的会话
        session = requests.Session()
        
        try:
            logging.debug(f"[LOCAL] Uploading modified DICOM to {url}")
            response = session.post(url, headers=headers, data=modified_data, timeout=(5, 15))
            response.raise_for_status()
            
            # 解析响应获取新的实例ID
            result = response.json()
            new_instance_id = result.get("ID")
            
            if new_instance_id:
                logging.debug(f"[LOCAL] Upload successful, new instance ID: {new_instance_id}")
            else:
                logging.error(f"[LOCAL] Upload successful but no instance ID in response: {result}")
                
            return new_instance_id
        finally:
            session.close()
            
    except Exception as e:
        logging.error(f"[LOCAL] Modify and upload failed: {e}")
        return None
    finally:
        # 记录结束时的内存使用情况
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        end_time = time.time()
        logging.debug(f"[MEMORY] After store_instance_dicom for {instance_id}: {mem_after:.2f} MB (Diff: {mem_after - mem_before:.2f} MB)")
        logging.debug(f"[TIMING] Completed local modify+upload strategy for instance {instance_id} in {end_time - start_time:.2f} seconds")
        
        # 强制垃圾回收
        gc.collect()
