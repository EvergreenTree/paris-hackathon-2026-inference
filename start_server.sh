#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_FILE="${LOG_FILE:-server.log}"
PID_FILE="${PID_FILE:-server.pid}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "Server already running with PID $(cat "${PID_FILE}")"
  exit 0
fi

.venv/bin/run-server --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" >"${PID_FILE}"
echo "Started server PID ${SERVER_PID} on ${HOST}:${PORT}"

