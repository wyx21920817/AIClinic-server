import requests
import json
import os # 用于可能的临时文件操作，虽然在这个片段中还未删除

# --- 配置 Orthanc 服务器信息 ---
ORTHANC_URL = "http://52.63.231.27:8042"

# --- 认证服务的 URL，用于获取临时 token ---
# 根据之前的分析，认证服务可能运行在 52.63.231.27:8000
AUTH_SERVICE_URL = "http://52.63.231.27:8000/api/auth/temp-token"

# --- 要测试的 instance_id ---
instance_id = "79f1c5f9-356696cd-4391a2cb-8d3713c1-a756d2f3"

# --- 获取临时 token 的函数 (从 test.py 复制过来) ---
def get_temp_token() -> str | None:
    """
    调用认证服务 API 获取临时访问 token.
    Returns:
        str: 获取到的临时 token, 如果失败则返回 None.
    """
    print(f"Attempting to obtain temporary token from {AUTH_SERVICE_URL}")
    try:
        response = requests.get(AUTH_SERVICE_URL)
        response.raise_for_status() # 如果响应状态码不是 2xx，则抛出异常

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

# --- 主要逻辑 ---
if __name__ == "__main__":
    temp_token = get_temp_token() # 首先获取 token

    if temp_token:
        # 构建请求头，将 token 放入 Authorization 字段
        headers = {
            "Authorization": f"Bearer {temp_token}"
        }

        # 构建 DICOM 文件的下载 URL
        dicom_download_url = f"{ORTHANC_URL}/instances/{instance_id}/file"

        print(f"Attempting to download DICOM file from: {dicom_download_url} with token.")
        try:
            # 发送 GET 请求，使用构建的 headers
            response = requests.get(dicom_download_url, headers=headers)
            response.raise_for_status() # 检查 HTTP 错误状态码

            output_filename = "downloaded_ecg.dcm" # 建议使用更具描述性的文件名
            with open(output_filename, "wb") as f:
                f.write(response.content)
            print(f"DICOM 文件 '{output_filename}' 下载成功。")

            # 此时，'output_filename' 就是 process_and_predict 函数需要的输入文件路径
            # 例如：
            # from your_module.model_processing import process_and_predict
            # predictions = process_and_predict(output_filename)
            # print("Predictions:", predictions)

            # !!! 重要：在实际的 FastAPI 应用中，你需要在此处（或后续步骤中）删除这个临时文件
            # import os
            # os.remove(output_filename)

        except requests.exceptions.RequestException as e:
            print(f"下载 DICOM 文件失败: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Orthanc 服务器响应: {e.response.text}")
        except Exception as e:
            print(f"下载过程中发生意外错误: {e}")
    else:
        print("未能获取到临时 token，无法下载 DICOM 文件。")