import requests
import json # Added for handling JSON response

# Orthanc 服务器的 URL 和认证信息 (不再直接使用)
# ORTHANC_URL = "http://52.63.231.27:8042"
# ORTHANC_AUTH = ("orthanc", "orthanc")

# FastAPI AI Server 的 URL
API_URL = "http://localhost:8000/predict/"

# 认证服务的 URL，用于获取临时 token

# 根据截图，认证服务可能运行在 54.79.224.136:8000
# AUTH_SERVICE_URL = "http://54.79.224.136:8000/api/auth/temp-token"
AUTH_SERVICE_URL = "http://52.63.231.27:8001/api/auth/temp-token"



# 要测试的 instance_id
instance_id = "6ceda576-0f1c6f1c-df730f7c-1528e82b-1f58d06b"


# --- ADDED: Function to get temporary token ---
def get_temp_token() -> str | None:
    """
    调用认证服务 API 获取临时访问 token.
    Returns:
        str: 获取到的临时 token, 如果失败则返回 None.
    """
    print(f"Attempting to obtain temporary token from {AUTH_SERVICE_URL}")
    try:
        # Send a GET request to the authentication service
        response = requests.get(AUTH_SERVICE_URL)
        response.raise_for_status() # If the response status code is not 2xx, an exception is thrown

        # Parsing JSON Response
        token_data = response.json()

        # Extract the 'token' field from the response
        temp_token = token_data.get("token")

        if temp_token:
            print(f"Successfully obtained temporary token (expires in: {token_data.get('expires_in')} seconds).")
            return temp_token
        else:
            print(f"Error: Token response did not contain 'token' field. Response: {token_data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to obtain temporary token: {e}")
        # 如果请求失败，尝试打印响应内容以获取更多信息（如果请求成功发送但服务器返回错误）
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Auth service response text: {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during token acquisition: {e}")
        return None
# --- End of ADDED Function ---


def test_dicom_prediction(instance_id: str, token: str | None):
    """
    测试通过 FastAPI API 对指定的 DICOM 实例进行预测，传递 token.
    """
    if not token:
        print("Error: No token provided or token acquisition failed. Cannot test.")
        return # 如果没有 token，直接返回，不发送请求

    # 准备请求体，包含 token
    payload = {
        "token": token
    }

    print(f"Sending prediction request for instance_id: {instance_id}")
    # 发送 POST 请求到 FastAPI，使用 json=payload 参数发送请求体
    response = requests.post(f"{API_URL}{instance_id}", json=payload)

    if response.status_code == 200:
        # 获取预测结果
        prediction = response.json()
        print("Predict outcomes:", prediction)
        # 检查并打印新实例的 ID
        if "new_instance_id" in prediction:
            print(f"Predictions are stored in the new Orthanc instance with ID: {prediction['new_instance_id']}")
    else:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")


if __name__ == "__main__":
    # --- MODIFIED: Automatically get token before running test ---
    temp_token = get_temp_token() # 调用新函数获取 token

    # 如果成功获取到 token，再运行测试函数
    if temp_token:
        test_dicom_prediction(instance_id, temp_token)
    else:
        print("Skipping prediction test due to token acquisition failure.")
    # --- End of MODIFIED ---