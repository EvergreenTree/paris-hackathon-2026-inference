#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/prototype/logs"
RUN_DIR="${ROOT_DIR}/prototype/run"
export PATH="${HOME}/.local/bin:${VENV_DIR}/bin:${PATH}"

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  uv venv "${VENV_DIR}" --python 3.12
fi

if [ ! -x "${VENV_DIR}/bin/vllm" ]; then
  uv pip install --python "${VENV_DIR}/bin/python" "vllm==${VLLM_VERSION}" ninja
fi

if ! command -v ninja >/dev/null 2>&1; then
  uv pip install --python "${VENV_DIR}/bin/python" ninja
fi

if [ -f "${RUN_DIR}/server.pid" ]; then
  existing_pid="$(cat "${RUN_DIR}/server.pid")"
  if kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Server already running with PID ${existing_pid}"
    exit 0
  fi
  rm -f "${RUN_DIR}/server.pid"
fi

nohup "${VENV_DIR}/bin/vllm" serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > "${LOG_DIR}/vllm.log" 2>&1 &

server_pid=$!
echo "${server_pid}" > "${RUN_DIR}/server.pid"
echo "Started vLLM with PID ${server_pid}"
echo "Logs: ${LOG_DIR}/vllm.log"

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    python3 "${ROOT_DIR}/eval/check_server.py" --base-url "http://127.0.0.1:${PORT}"
    exit 0
  fi

  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM exited unexpectedly. Recent log output:"
    tail -n 100 "${LOG_DIR}/vllm.log" || true
    exit 1
  fi

  sleep 5
done

echo "Timed out waiting for vLLM to become healthy. Recent log output:"
tail -n 100 "${LOG_DIR}/vllm.log" || true
exit 1
