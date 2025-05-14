import requests

# Orthanc 服务器的 URL 和认证信息
ORTHANC_URL = "http://54.79.224.136:8042"
ORTHANC_AUTH = ("orthanc", "orthanc")

# 要测试的 instance_id
instance_id = "11afdfd9-fa429bbd-995265c4-6d965eb0-4820f0dd"

response = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/file", auth=ORTHANC_AUTH)

if response.status_code == 200:
    with open("test.dcm", "wb") as f:
        f.write(response.content)
    print("文件下载成功")
else:
    print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
