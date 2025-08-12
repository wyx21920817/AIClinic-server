# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import the modified store_instance_dicom function (now returns str | None)
# 引入必要的模块：requests 用于HTTP请求，tempfile 用于创建临时文件，os 用于文件操作
import requests
import tempfile
import os
# 引入 run_in_threadpool 以在异步环境中执行同步操作
from starlette.concurrency import run_in_threadpool
# 确保从 dicom_utils 导入 ORTHANC_URL 和 get_headers，因为它们是获取二进制数据所必需的
from dicom_utils import get_instance_dicom, store_instance_dicom, ORTHANC_URL, get_headers
from preprocess import extract_waveform # add_predictions_to_dicom is no longer used for /modify
from model_utils import load_model, predict_ecg
from pydicom import dcmread # 确保导入 dcmread，可能在其他地方被使用
from ecg_ptbxl_benchmarking_master.extract_and_predict_ecg import process_and_predict 
app = FastAPI()

# load models when intialize
from model_utils import load_models
load_models()

import logging

# set log record
logging.basicConfig(level=logging.DEBUG)

# 认证服务URL - 新增
AUTH_SERVICE_URL = "http://54.79.181.49:8001/api/auth/temp-token"

class PredictRequest(BaseModel):
    token: str

async def get_temp_token() -> str | None:
    """
    自动获取临时认证token
    """
    logging.debug(f"Attempting to obtain temporary token from {AUTH_SERVICE_URL}")
    try:
        response = await run_in_threadpool(requests.get, AUTH_SERVICE_URL)
        response.raise_for_status()
        token_data = response.json()
        temp_token = token_data.get("token")
        
        if temp_token:
            logging.debug(f"Successfully obtained temporary token (expires in: {token_data.get('expires_in')} seconds).")
            return temp_token
        else:
            logging.error(f"Token response did not contain 'token' field. Response: {token_data}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to obtain temporary token: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during token acquisition: {e}")
        return None

@app.post("/predict/{instance_id}")
async def predict(instance_id: str, request: PredictRequest):
    token = request.token
    logging.debug(f"Start prediction for instance_id: {instance_id} with token provided.")

    ds = None
    temp_dicom_file_path = None # 初始化临时文件路径变量

    try:
        # 1. Retrieve DICOM (Still need to retrieve to get waveform for original model)
        # 将同步的 get_instance_dicom 调用包装在 run_in_threadpool 中
        ds = await run_in_threadpool(get_instance_dicom, instance_id, token)
        logging.debug(f"DICOM data retrieve success for original model.")

        # --- 新增的代码开始 ---
        # 2. 使用 token 获取 DICOM 文件的原始二进制数据并写入临时文件
        logging.debug(f"Attempting to fetch DICOM binary data for instance {instance_id} for new model.")
        headers = get_headers(token) # 获取包含 Bearer token 的请求头
        dicom_binary_url = f"{ORTHANC_URL}/instances/{instance_id}/file" # 构建获取二进制文件的 URL

        # 将同步的 requests.get 调用包装在 run_in_threadpool 中
        response = await run_in_threadpool(requests.get, dicom_binary_url, headers=headers)
        response.raise_for_status() # 检查 HTTP 错误状态码，如果不是 2xx 则抛出异常

        # 创建一个临时文件，将 DICOM 二进制数据写入其中
        # delete=False 确保文件在关闭后不会立即删除，我们需要它来传递给 process_and_predict
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            tmp_file.write(response.content)
            temp_dicom_file_path = tmp_file.name # 保存临时文件的完整路径
        logging.debug(f"DICOM binary data saved to temporary file: {temp_dicom_file_path}")
        # --- 新增的代码结束 ---

        # 2. Transform to adapt input of models (This is for the original model path)
        # 将同步的 extract_waveform 调用包装在 run_in_threadpool 中
        ecg, sampling_rate, lead_names = await run_in_threadpool(extract_waveform, ds)
        logging.debug(f"ECG data extracted for original model with shape: {ecg.shape}")

        # Remove the transpose as per previous fix
        # # ecg = ecg.T # 这一行在之前的修改中已移除或注释掉

        logging.debug(f"ECG data shape for original model prediction: {ecg.shape}.")

        preds = {} # 原模型预测结果
        new_model_predictions = {} # 新模型预测结果
        combined_predictions = {} # 合并后的预测结果

        # 3. Predict (Original Model)
        # 将同步的 predict_ecg 调用包装在 run_in_threadpool 中
        preds = await run_in_threadpool(predict_ecg, ecg)
        logging.debug(f"Original model predict outcome: {preds}")

        # 在这里，如果 `process_and_predict` 函数可用，你可以使用 `temp_dicom_file_path` 调用它
        new_model_predictions = await run_in_threadpool(process_and_predict, temp_dicom_file_path)
        logging.debug(f"New model predict outcome: {new_model_predictions}")
        
        # 将 preds 和 new_model_predictions 合并到 combined_preds 中
        combined_predictions = {
            "original_dnn_model": preds,
            "resnet_fine_tuned_model": new_model_predictions
        }
        logging.debug(f"Combined predictions: {combined_predictions}")

        # --- MODIFIED LOGIC for storing ---
        # We generate the comment string
        comment = "; ".join(f"{k}:{v:.3f}" for k, v in preds.items())
        comment += " | ResNet Top5: " + "; ".join(f"{k}:{v:.3f}" for k, v in new_model_predictions.items())
        logging.debug(f"Prepared comment string for PatientComments: {comment}")

        # The in-memory modification using add_predictions_to_dicom is not used for /modify approach
        # add_predictions_to_dicom is no longer imported or used.

        # --- MODIFIED CALL to store_instance_dicom ---
        # 将同步的 store_instance_dicom 调用包装在 run_in_threadpool 中
        new_instance_id = await run_in_threadpool(store_instance_dicom, instance_id, comment, token)
        # -------------------------------------

        # --- MODIFIED RETURN ---
        # We return the potentially None new_instance_id.
        # The key name remains "modified_instance_id".
        # 最终的返回语句
        return {"modified_instance_id": new_instance_id, "predictions": combined_predictions}

    except requests.exceptions.RequestException as e:
        logging.error(f"Orthanc retrieve, file saving, or waveform extraction failed for instance {instance_id}: {e}")
        # 如果在获取 DICOM 或保存临时文件时发生错误，抛出 HTTPException
        if e.response:
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to fetch DICOM file: {e.response.text}")
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    except Exception as e:
        logging.error(f"Prediction failed for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally: # 现在这个 finally 块与外层的 try 块在同一缩进级别
        # 确保删除临时文件，无论是否发生异常
        if temp_dicom_file_path and os.path.exists(temp_dicom_file_path):
            try:
                os.remove(temp_dicom_file_path)
                logging.debug(f"Temporary DICOM file {temp_dicom_file_path} deleted.")
            except Exception as e:
                logging.error(f"Failed to delete temporary file {temp_dicom_file_path}: {e}")

@app.post("/diagnose/{instance_id}")
async def diagnose_ecg(instance_id: str):
    """
    ECG diagnostic API for external use
    Simply enter the instance_id to automatically obtain a token and return the diagnostic results.
    """
    logging.info(f"Receive a diagnostic request,Instance ID: {instance_id}")
    
    try:
        # Automatically obtain tokens (users do not need to worry about authentication details)
        token = await get_temp_token()
        if not token:
            raise HTTPException(status_code=401, detail="Failed to obtain authentication token")
        
        logging.debug(f"Successfully obtained token for instance {instance_id}")
        
        # Call the existing predict logic
        request = PredictRequest(token=token)
        result = await predict(instance_id, request)
        
        # Returns a simplified response format
        return {
            "instance_id": instance_id,
            "status": "success",
            "predictions": result["predictions"],
            "modified_instance_id": result["modified_instance_id"],
            "message": f"Successfully diagnosed ECG instance: {instance_id}"
        }
        
    except HTTPException:
        # Rethrowing HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Diagnostic failure,Instance ID: {instance_id}, error: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")