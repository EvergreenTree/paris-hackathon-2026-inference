#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_FILE="${LOG_FILE:-server.log}"
PID_FILE="${PID_FILE:-server.pid}"
HEALTHCHECK_TIMEOUT_SEC="${HEALTHCHECK_TIMEOUT_SEC:-120}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found at ${PYTHON_BIN}" >&2
  echo "Create the venv first, e.g.: uv venv --python 3.12 && uv pip install -e '.[server]'" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "Server already running with PID $(cat "${PID_FILE}")"
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "Health check OK at ${HEALTH_URL}"
  else
    echo "Warning: existing PID found but health check failed at ${HEALTH_URL}" >&2
  fi
  exit 0
fi

if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
  echo "A healthy server is already responding at ${HEALTH_URL}"
  exit 0
fi

"${PYTHON_BIN}" -m server.app --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" >"${PID_FILE}"
echo "Started server PID ${SERVER_PID} on ${HOST}:${PORT}"

for ((i = 1; i <= HEALTHCHECK_TIMEOUT_SEC; i++)); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Server process exited early. Check ${LOG_FILE}" >&2
    exit 1
  fi
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "Server is ready at ${HEALTH_URL}"
    exit 0
  fi
  sleep 1
done

echo "Server failed health check within ${HEALTHCHECK_TIMEOUT_SEC}s. Check ${LOG_FILE}" >&2
kill "${SERVER_PID}" >/dev/null 2>&1 || true
exit 1

