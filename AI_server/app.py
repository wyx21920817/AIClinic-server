# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import tempfile
import os
from io import BytesIO # 可能需要，如果 extract_waveform 需要 Dataset 对象
from pydicom import dcmread
# 从 dicom_utils 导入两个获取 DICOM 的函数
from dicom_utils import get_instance_dicom, get_instance_dicom_binary, store_instance_dicom

# 导入原有模型处理函数
from preprocess import extract_waveform
from model_utils import load_model, predict_ecg

# 导入新的 ECG 预测函数
from ecg_ptbxl_benchmarking_master.extract_and_predict_ecg import process_and_predict

app = FastAPI()

# 加载原有模型
from model_utils import load_models
load_models() 

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    token: str

# 辅助函数：格式化预测结果为 PatientComments 字符串
def format_predictions_for_dicom(predictions_old_model: dict, predictions_new_model: dict) -> str:
    """
    Formats two dictionaries of predictions into a semicolon-separated string
    suitable for DICOM PatientComments.
    """
    old_preds_str = "; ".join([f"Old_{label}:{prob:.3f}" for label, prob in predictions_old_model.items()])
    new_preds_str = "; ".join([f"New_{label}:{prob:.3f}" for label, prob in predictions_new_model.items()])
    
    return f"Old_Model_Results: {old_preds_str}; New_Model_Results: {new_preds_str}"


@app.post("/predict/{instance_id}")
async def predict(instance_id: str, request: PredictRequest):
    token = request.token
    logger.debug(f"Start prediction for instance_id: {instance_id} with token provided.")

    # 声明变量，用于存储两种类型的 DICOM 数据
    dicom_pydicom_dataset = None  # 用于原有模型的 pydicom.Dataset 对象
    dicom_binary_data = None      # 用于新模型的原始二进制数据
    temp_dicom_path = None        # 用于新模型的临时文件路径

    try:
        # --- 获取 DICOM 原始二进制数据 (供新模型使用) ---
        # 假设 get_instance_dicom_binary 是 async 函数
        dicom_binary_data = await get_instance_dicom_binary(instance_id, token) 
        if not dicom_binary_data:
            logger.error(f"Orthanc retrieve failed: DICOM instance {instance_id} not found or empty.")
            raise HTTPException(status_code=404, detail=f"Orthanc retrieve failed: DICOM instance {instance_id} not found or could not be retrieved.")
        logger.debug(f"DICOM binary data retrieved successfully for instance {instance_id}.")

        # --- 将二进制数据转换为 pydicom.Dataset (供原有模型使用) ---
        # 如果 get_instance_dicom 已经返回 pydicom.Dataset，这里就不需要再转换
        # 但如果上面 get_instance_dicom_binary 是唯一获取二进制数据的方式，
        # 那么需要从 binary_data 转换回 Dataset 给旧模型用
        # 注意：如果 get_instance_dicom 已经返回 Dataset，那这里就不用再做一遍了
        # 我这里假设你已经调整了dicom_utils使其get_instance_dicom_binary返回bytes
        # 那么get_instance_dicom就需要自行调用BytesIO(r.content)了，或者直接让get_instance_dicom_binary返回bytes，然后这里转换
        # 这里为了简化，我直接用 BytesIO(dicom_binary_data) 来创建一个 Dataset 对象给旧模型
        dicom_pydicom_dataset = dcmread(BytesIO(dicom_binary_data)) 

    except Exception as e:
        logger.error(f"Orthanc retrieval or initial DICOM parsing failed for instance {instance_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Orthanc retrieval or DICOM parsing failed: {e}")

    # --- 原有模型预测逻辑 ---
    preds_old_model = {}
    try:
        # 2a. 转换以适应原有模型输入 (使用 pydicom.Dataset)
        ecg, sampling_rate, lead_names = extract_waveform(dicom_pydicom_dataset) 
        logger.debug(f"Original ECG data extracted for old model with shape: {ecg.shape}")

        # 3a. 使用原有模型进行预测
        preds_old_model = predict_ecg(ecg) 
        logger.debug(f"Original model prediction outcome: {preds_old_model}")
        
    except Exception as e:
        logger.error(f"Original model prediction failed for instance {instance_id}: {e}", exc_info=True)
        preds_old_model = {"Error_Old_Model": f"Prediction failed: {e}"} # 记录错误，不中断流程

    # --- 新的 ECG 模型预测逻辑 ---
    preds_new_model = {}
    try:
        # 2b. 保存二进制数据到临时文件 (供新模型使用)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            tmp_file.write(dicom_binary_data) # 使用获取到的原始二进制数据
            temp_dicom_path = tmp_file.name # 获取临时文件的路径
        logger.debug(f"DICOM binary data saved to temporary file for new model: {temp_dicom_path}")

        # 3b. 调用新模型预测函数
        # process_and_predict 期望文件路径作为输入
        preds_new_model = process_and_predict(temp_dicom_path) 
        
        if not preds_new_model: 
            logger.error(f"New ECG model did not return any predictions for instance {instance_id}.")
            raise RuntimeError("New ECG model returned no predictions.")
        logger.debug(f"New ECG model prediction outcome: {preds_new_model}")

    except Exception as e:
        logger.error(f"New model prediction failed for instance {instance_id}: {e}", exc_info=True)
        preds_new_model = {"Error_New_Model": f"Prediction failed: {e}"} # 记录错误，不中断流程
    finally:
        # 确保临时文件被清理
        if temp_dicom_path and os.path.exists(temp_dicom_path):
            os.remove(temp_dicom_path)
            logger.debug(f"Cleaned up temporary DICOM file: {temp_dicom_path}")

    # --- 整合并存储结果 ---
    # 将两个模型的预测结果合并并格式化为 PatientComments 字符串
    comment = format_predictions_for_dicom(preds_old_model, preds_new_model)
    logger.debug(f"Prepared combined comment string for PatientComments: {comment}")

    # 将更新后的 DICOM 存回 Orthanc
    # 假设 store_instance_dicom 是 async 函数
    new_instance_id = await store_instance_dicom(instance_id, comment, token)
    
    if not new_instance_id:
        logger.error(f"Failed to store updated DICOM back to Orthanc for instance {instance_id}.")
        raise HTTPException(status_code=500, detail="Failed to store updated DICOM back to Orthanc.")

    logger.info(f"Successfully processed instance {instance_id}. New DICOM instance ID: {new_instance_id}")
    
    # 返回原始实例ID、新实例ID以及两个模型的预测结果字典
    return {
        "original_instance_id": instance_id, 
        "new_instance_id": new_instance_id, 
        "predictions_old_model": preds_old_model, 
        "predictions_new_model": preds_new_model  
    }