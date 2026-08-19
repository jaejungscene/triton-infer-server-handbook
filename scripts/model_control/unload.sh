#!/usr/bin/env bash
# Unload a model in EXPLICIT control mode and verify that it is no longer ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/model_control/common.sh
source "${SCRIPT_DIR}/common.sh"
model_control_init "unload" "${1:-}" "${2:-http://localhost:8000}" "${3:-120}"

model_control_temp_file body_file
model_control_temp_file index_body_file

echo "[unload] Requesting model unload: ${MODEL_NAME}"
if ! http_code=$(model_control_curl -sS \
    --proto "=http,https" \
    --connect-timeout 3 \
    --max-time "${TIMEOUT_SECONDS}" \
    -o "${body_file}" \
    -w "%{http_code}" \
    -X POST \
    "${BASE_URL}/v2/repository/models/${MODEL_NAME}/unload"); then
    echo "[unload] Model-control request failed" >&2
    exit 1
fi

if [[ "${http_code}" != "200" ]]; then
    echo "[unload] FAIL (HTTP ${http_code}): $(<"${body_file}")" >&2
    exit 1
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
while true; do
    if ! index_code=$(model_control_curl -sS --proto "=http,https" \
        --connect-timeout 3 --max-time 5 \
        -o "${index_body_file}" \
        -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"ready":true}' \
        "${BASE_URL}/v2/repository/index"); then
        echo "[unload] Repository Index request failed" >&2
        exit 1
    fi
    if [[ "${index_code}" != "200" ]]; then
        echo "[unload] Repository Index failed (HTTP ${index_code}): $(<"${index_body_file}")" >&2
        exit 1
    fi

    if ! model_state=$(python3 - "${MODEL_NAME}" "${index_body_file}" <<'PYTHON'
import json
import sys

model_name = sys.argv[1]
with open(sys.argv[2], encoding="utf-8") as response_file:
    payload = json.load(response_file)
if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
    raise SystemExit("Repository Index response must be a JSON array of objects")
print("ready" if any(item.get("name") == model_name for item in payload) else "absent")
PYTHON
    ); then
        echo "[unload] Invalid Repository Index response" >&2
        exit 1
    fi
    if [[ "${model_state}" == "absent" ]]; then
        break
    fi
    if (( SECONDS >= deadline )); then
        echo "[unload] Timed out waiting for ${MODEL_NAME} to unload" >&2
        exit 1
    fi
    sleep 1
done

echo "[unload] OK: ${MODEL_NAME} is not ready"
