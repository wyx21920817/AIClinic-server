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
PIDS=$(lsof -t -i tcp:$PORT)

if [ -n "$PIDS" ]; then
  echo "[WARNING] 端口 $PORT 正被以下进程占用: $PIDS"
  echo "[INFO] 正在终止所有占用端口 $PORT 的进程..."
  
  # 逐个终止所有进程
  for PID in $PIDS; do
    echo "[INFO] 终止进程 $PID..."
    kill -9 $PID
    sleep 1
  done
  
  # 等待一下确保端口释放
  sleep 2
  
  # 再次检查端口是否已释放
  NEW_PIDS=$(lsof -t -i tcp:$PORT)
  if [ -n "$NEW_PIDS" ]; then
    echo "[ERROR] 无法释放端口 $PORT，仍被进程 $NEW_PIDS 占用"
    echo "[WARNING] 将继续尝试使用端口 $PORT，可能会启动失败"
  else
    echo "[INFO] 已成功释放端口 $PORT"
  fi
fi

# 创建Logs目录如果不存在
mkdir -p logs

echo "[INFO] 在端口 $PORT 上启动FastAPI服务..."
nohup uvicorn app:app --host 0.0.0.0 --port $PORT --reload --log-level info > logs/server.log 2>&1 &

  echo "[INFO] 服务已在后台启动。日志保存在 logs/server.log"
