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
    echo "[INFO] 已终止占用端口 $PORT 的进程 $PID。"
  done
  
  # 验证所有进程是否已终止
  sleep 1
  REMAINING_PIDS=$(lsof -t -i tcp:$PORT)
  if [ -n "$REMAINING_PIDS" ]; then
    echo "[WARNING] 以下进程仍然占用端口 $PORT: $REMAINING_PIDS"
    echo "[INFO] 尝试强制终止所有剩余进程..."
    for PID in $REMAINING_PIDS; do
      echo "[INFO] 强制终止进程 $PID..."
      kill -9 $PID 2>/dev/null || echo "[ERROR] 无法终止进程 $PID"
    done
  fi
  
  echo "[INFO] 已成功释放端口 $PORT"
fi

# 创建Logs目录如果不存在
mkdir -p logs

echo "[INFO] 在端口 $PORT 上启动FastAPI服务..."
nohup uvicorn app:app --host 0.0.0.0 --port $PORT --log-level info > logs/server.log 2>&1 &

  echo "[INFO] 服务已在后台启动。日志保存在 logs/server.log"
