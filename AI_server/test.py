import requests

# Orthanc 服务器的 URL 和认证信息
ORTHANC_URL = "http://54.79.224.136:8042"
ORTHANC_AUTH = ("orthanc", "orthanc")

# FastAPI 服务器的 URL
API_URL = "http://localhost:8000/predict/"

# 要测试的 instance_id
instance_id = "11afdfd9-fa429bbd-995265c4-6d965eb0-4820f0dd"  # 从上面图片中提取的 SOPInstanceUID

def test_dicom_prediction(instance_id: str):
    """
    测试通过 FastAPI API 对指定的 DICOM 实例进行预测
    """

    # 发送请求到 FastAPI 进行预测
    response = requests.post(f"{API_URL}{instance_id}")

    if response.status_code == 200:
        # 获取预测结果
        prediction = response.json()
        print("Predict outcomes:", prediction)
    else:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")

if __name__ == "__main__":
    test_dicom_prediction(instance_id)
