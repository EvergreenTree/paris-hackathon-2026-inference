#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/prototype/logs"
RUN_DIR="${ROOT_DIR}/prototype/run"
PID_FILE="${RUN_DIR}/min_server.pid"
LOG_FILE="${LOG_DIR}/min_server.log"

export PATH="${HOME}/.local/bin:${VENV_DIR}/bin:${PATH}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
}

ensure_venv() {
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv "${VENV_DIR}" --python 3.12
  fi
}

has_imports() {
  "${VENV_DIR}/bin/python" - "$@" <<'PY'
import importlib.util
import sys

mods = sys.argv[1:]
missing = [mod for mod in mods if importlib.util.find_spec(mod) is None]
if missing:
    print("Missing modules:", ", ".join(missing))
    raise SystemExit(1)
PY
}

ensure_runtime() {
  uv pip install --python "${VENV_DIR}/bin/python" -e "${ROOT_DIR}"

  if ! has_imports fastapi uvicorn transformers torch; then
    uv pip install --python "${VENV_DIR}/bin/python" \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      "fastapi>=0.115" \
      "uvicorn>=0.30" \
      "transformers>=4.45" \
      "torch>=2.10.0"
  fi
}

cleanup_stale_pid() {
  if [ -f "${PID_FILE}" ]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Prototype server already running with PID ${pid}"
      exit 0
    fi
    rm -f "${PID_FILE}"
  fi
}

ensure_uv
ensure_venv
ensure_runtime
cleanup_stale_pid

nohup env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LOG_LEVEL="${LOG_LEVEL}" \
  "${VENV_DIR}/bin/python" "${ROOT_DIR}/prototype/min_server.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  > "${LOG_FILE}" 2>&1 &

server_pid=$!
echo "${server_pid}" > "${PID_FILE}"
echo "Started prototype server with PID ${server_pid}"
echo "Logs: ${LOG_FILE}"

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    "${VENV_DIR}/bin/python" -m eval.check_server --base-url "http://127.0.0.1:${PORT}"
    exit 0
  fi

  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Prototype server exited unexpectedly. Recent log output:"
    tail -n 100 "${LOG_FILE}" || true
    exit 1
  fi

  sleep 5
done

echo "Timed out waiting for the prototype server to become healthy. Recent log output:"
tail -n 100 "${LOG_FILE}" || true
exit 1
