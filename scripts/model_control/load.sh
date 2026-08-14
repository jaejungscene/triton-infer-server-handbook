#!/usr/bin/env bash
# Load a model in EXPLICIT control mode and verify that it becomes ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/model_control/common.sh
source "${SCRIPT_DIR}/common.sh"
model_control_init "load" "${1:-}" "${2:-http://localhost:8000}" "${3:-120}"

model_control_temp_file body_file

echo "[load] Requesting model load: ${MODEL_NAME}"
if ! http_code=$(model_control_curl -sS \
    --proto "=http,https" \
    --connect-timeout 3 \
    --max-time "${TIMEOUT_SECONDS}" \
    -o "${body_file}" \
    -w "%{http_code}" \
    -X POST \
    "${BASE_URL}/v2/repository/models/${MODEL_NAME}/load"); then
    echo "[load] Model-control request failed" >&2
    exit 1
fi

if [[ "${http_code}" != "200" ]]; then
    echo "[load] FAIL (HTTP ${http_code}): $(<"${body_file}")" >&2
    exit 1
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
until model_control_curl -fsS --proto "=http,https" \
    --connect-timeout 3 --max-time 5 \
    "${BASE_URL}/v2/models/${MODEL_NAME}/ready" > /dev/null; do
    if (( SECONDS >= deadline )); then
        echo "[load] Timed out waiting for ${MODEL_NAME} to become ready" >&2
        exit 1
    fi
    sleep 1
done

echo "[load] OK: ${MODEL_NAME} is ready"
