#!/bin/bash

echo "[INFO] 激活虚拟环境..."
# 检查虚拟环境是否存在
if [ -d "venv" ]; then
  source venv/bin/activate
else
  echo "[WARNING] 虚拟环境不存在，创建新的虚拟环境..."
  python3 -m venv venv
  source venv/bin/activate
fi

echo "[INFO] 设置环境变量..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 安装必要的系统依赖
if command -v sudo &> /dev/null; then
  echo "[INFO] 安装系统依赖..."
  sudo apt-get update
  sudo apt-get install -y lsof
else
  echo "[WARNING] 无sudo权限，跳过系统依赖安装"
fi

echo "[INFO] 安装Python依赖..."
pip install -r requirements.txt
pip install uvicorn
pip install python-multipart

# 降级protobuf解决兼容性问题
echo "[INFO] 降级protobuf到兼容版本..."
pip install protobuf==3.20.0 --force-reinstall

echo "[INFO] 启动FastAPI服务..."
PORT=8000

# 检查端口是否被占用
PID=$(lsof -t -i tcp:$PORT)

if [ -n "$PID" ]; then
  echo "[WARNING] 端口 $PORT 正被进程 $PID 占用。正在终止该进程..."
  kill -9 $PID
  echo "[INFO] 已终止占用端口 $PORT 的进程 $PID。"
fi

# 创建Logs目录如果不存在
mkdir -p logs

echo "[INFO] 在端口 $PORT 上启动FastAPI服务..."
nohup uvicorn app:app --host 0.0.0.0 --port $PORT --reload --log-level info > logs/server.log 2>&1 &

echo "[INFO] 服务已在后台启动。日志保存在 logs/server.log"
