import os
import json

def build_new_feature_input_from_prediction(
    prediction_result: dict, 
    lead_index: int = 1, 
    db_dir: str = "."
) -> dict:
    """
    根据AI服务返回的诊断结果字典，生成新功能需要的输入字典。
    
    Args:
        prediction_result: 预测API返回的完整字典
        lead_index: 新功能需要的导联索引
        db_dir: 新功能需要的数据目录
    Returns:
        dict，包含 abnormality, record_name, lead_index, db_dir
    """
    # 1. 获取病症概率字典
    dnn_probs = prediction_result["predictions"]["original_dnn_model"]

    # 2. 病症映射（API返回字段→新功能需要的缩写）
    label_map = {
        'AF': 'AF',
        'ST': 'ST',
        'LBBB': 'LBBB',
        'RBBB': 'RBBB',
        '1dAVb': '1dAVB',  # 注意V大写
        'SB': 'SB'
    }

    # 3. abnormality: 选最大概率的标签，阈值可调
    threshold = 0.5
    abnormality = "NR"  # 默认正常
    max_label, max_prob = None, 0.0
    for key, val in dnn_probs.items():
        if key in label_map and val > threshold and val > max_prob:
            max_label = label_map[key]
            max_prob = val
    if max_label:
        abnormality = max_label

    # 4. record_name（优先用你本地业务的dicom实际文件名，否则用instance_id）
    # 假设只能用instance_id:
    record_name = prediction_result.get("instance_id", "unknown")

    # 5. 组装
    return {
        "abnormality": abnormality,
        "record_name": record_name,
        "lead_index": lead_index,
        "db_dir": db_dir
    }

# 目录配置
diagnose_dir = "diagnose_logs"
dicom_dir = "dicoms"
db_dir = dicom_dir    # 通常db_dir和dicom_dir一致

# 遍历所有诊断json
for fname in os.listdir(diagnose_dir):
    if not fname.endswith('.json'):
        continue
    # 获取 instance_id，去掉扩展名
    instance_id = fname.rsplit('.', 1)[0]
    json_path = os.path.join(diagnose_dir, fname)
    dicom_path = os.path.join(dicom_dir, f"{instance_id}.dcm")

    # 检查对应 DICOM 文件是否存在
    if not os.path.exists(dicom_path):
        print(f"⚠️ 缺少DICOM文件: {dicom_path}, 跳过")
        continue

    # 读取诊断json
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    # 自动生成新功能输入参数
    params = build_new_feature_input_from_prediction(
        result, 
        lead_index=1,           # 按需可调整
        db_dir=db_dir           # 或传其他路径
    )
    # 加上DICOM路径
    params["dicom_path"] = dicom_path

    # 传递给新功能
    # new_feature_function(**params)
    print(f"已生成新功能参数: {params}")
