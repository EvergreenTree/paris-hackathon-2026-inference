#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_FILE="${LOG_FILE:-server.log}"
PID_FILE="${PID_FILE:-server.pid}"
HEALTHCHECK_TIMEOUT_SEC="${HEALTHCHECK_TIMEOUT_SEC:-600}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKER_BASE_PORT="${HACKATHON_WORKER_BASE_PORT:-8100}"
WORKER_PID_FILE="${WORKER_PID_FILE:-server.workers.pid}"
WORKER_LOG_PREFIX="${WORKER_LOG_PREFIX:-server_worker}"

# Hardcoded scheduler/model profile for current hackathon sprint.
# Values can still be overridden from shell if needed.
: "${HACKATHON_BACKEND:=hf}"
: "${HACKATHON_MODEL_ID:=Qwen/Qwen3.5-35B-A3B}"
: "${HACKATHON_DEVICE:=cuda}"
: "${HACKATHON_DTYPE:=bfloat16}"
: "${HACKATHON_DATA_PARALLEL_REPLICAS:=8}"
: "${HACKATHON_WORKER_COUNT:=8}"
: "${HACKATHON_MAX_WORKER_COUNT:=8}"
: "${HACKATHON_AUTOSCALE_WORKERS:=1}"
: "${HACKATHON_AUTOSCALE_CHECK_MS:=200}"
: "${HACKATHON_AUTOSCALE_SCALE_UP_Q:=6}"
: "${HACKATHON_AUTOSCALE_SCALE_DOWN_Q:=2}"
: "${HACKATHON_BATCH_MAX_SIZE:=8}"
: "${HACKATHON_BATCH_WAIT_MS:=5.0}"
: "${HACKATHON_ADAPTIVE_BATCH_WAIT:=1}"
: "${HACKATHON_PRIORITY_ENABLE:=1}"
: "${HACKATHON_PRIORITY_MAX_TOKENS:=256}"
: "${HACKATHON_PRIORITY_BURST:=4}"
: "${HACKATHON_SHAPE_BUCKETING_ENABLE:=0}"
: "${HACKATHON_SHAPE_BUCKET_CHARS:=512}"
: "${HACKATHON_MAX_PENDING_REQUESTS:=4096}"
: "${HACKATHON_OVERLOAD_WAIT_MS:=15}"
: "${HACKATHON_PER_WORKER_COUNT:=1}"
: "${HACKATHON_PER_WORKER_MAX_COUNT:=1}"
: "${HACKATHON_ROUTER_POLICY:=least_pending}"
: "${HACKATHON_ROUTER_MAX_RETRIES:=2}"
: "${HACKATHON_TORCH_COMPILE:=1}"
: "${HACKATHON_COMPILE_MODE:=max-autotune-no-cudagraphs}"
: "${HACKATHON_MAX_NEW_TOKENS_CAP:=2048}"
: "${HACKATHON_FAST_GENERATE:=1}"
: "${HACKATHON_ATTN_IMPL:=auto}"

export HACKATHON_BACKEND HACKATHON_MODEL_ID HACKATHON_DEVICE HACKATHON_DTYPE
export HACKATHON_DATA_PARALLEL_REPLICAS
export HACKATHON_WORKER_COUNT HACKATHON_MAX_WORKER_COUNT HACKATHON_AUTOSCALE_WORKERS
export HACKATHON_AUTOSCALE_CHECK_MS HACKATHON_AUTOSCALE_SCALE_UP_Q HACKATHON_AUTOSCALE_SCALE_DOWN_Q
export HACKATHON_BATCH_MAX_SIZE HACKATHON_BATCH_WAIT_MS HACKATHON_ADAPTIVE_BATCH_WAIT
export HACKATHON_PRIORITY_ENABLE HACKATHON_PRIORITY_MAX_TOKENS HACKATHON_PRIORITY_BURST
export HACKATHON_SHAPE_BUCKETING_ENABLE HACKATHON_SHAPE_BUCKET_CHARS
export HACKATHON_MAX_PENDING_REQUESTS HACKATHON_OVERLOAD_WAIT_MS
export HACKATHON_ROUTER_POLICY HACKATHON_ROUTER_MAX_RETRIES
export HACKATHON_TORCH_COMPILE HACKATHON_COMPILE_MODE HACKATHON_MAX_NEW_TOKENS_CAP
export HACKATHON_FAST_GENERATE HACKATHON_ATTN_IMPL

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found at ${PYTHON_BIN}" >&2
  echo "Create the venv first, e.g.: uv venv --python 3.12 && uv pip install -e '.[server]'" >&2
  exit 1
fi

cleanup_pids() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] || continue
      kill "${pid}" >/dev/null 2>&1 || true
    done < "${file}"
    rm -f "${file}"
  fi
}

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

cleanup_pids "${WORKER_PID_FILE}"
rm -f "${PID_FILE}"

replicas="${HACKATHON_DATA_PARALLEL_REPLICAS}"
if [[ "${replicas}" -lt 1 ]]; then
  replicas=1
fi

echo "Starting process-router profile: backend=${HACKATHON_BACKEND} replicas=${replicas} router_port=${PORT} worker_base_port=${WORKER_BASE_PORT}"
touch "${WORKER_PID_FILE}"
worker_urls=()
for ((i = 0; i < replicas; i++)); do
  wport=$((WORKER_BASE_PORT + i))
  wlog="${WORKER_LOG_PREFIX}_${i}.log"
  worker_urls+=("http://127.0.0.1:${wport}")
  HACKATHON_DEVICE="cuda:${i}" \
  HACKATHON_DATA_PARALLEL_REPLICAS="1" \
  HACKATHON_WORKER_COUNT="${HACKATHON_PER_WORKER_COUNT}" \
  HACKATHON_MAX_WORKER_COUNT="${HACKATHON_PER_WORKER_MAX_COUNT}" \
  "${PYTHON_BIN}" -m server.app --host 127.0.0.1 --port "${wport}" >"${wlog}" 2>&1 &
  echo "$!" >> "${WORKER_PID_FILE}"
done

for ((i = 0; i < replicas; i++)); do
  wport=$((WORKER_BASE_PORT + i))
  wurl="http://127.0.0.1:${wport}/health"
  ready=0
  for ((j = 1; j <= HEALTHCHECK_TIMEOUT_SEC; j++)); do
    if curl -fsS "${wurl}" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "Worker ${i} failed health check at ${wurl}" >&2
    cleanup_pids "${WORKER_PID_FILE}"
    exit 1
  fi
done

worker_urls_csv="$(IFS=,; echo "${worker_urls[*]}")"
HACKATHON_ROUTER_WORKER_URLS="${worker_urls_csv}" \
"${PYTHON_BIN}" -m server.router --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "Started router PID ${SERVER_PID} on ${HOST}:${PORT}"

for ((i = 1; i <= HEALTHCHECK_TIMEOUT_SEC; i++)); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Server process exited early. Check ${LOG_FILE}" >&2
    cleanup_pids "${WORKER_PID_FILE}"
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
cleanup_pids "${WORKER_PID_FILE}"
exit 1

