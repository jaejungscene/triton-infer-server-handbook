#!/usr/bin/env bash
# Unload a model in EXPLICIT control mode and verify that it is no longer ready.

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model_name> [base_url] [timeout_seconds]}"
BASE_URL="${2:-http://localhost:8000}"
TIMEOUT_SECONDS="${3:-120}"
BASE_URL="${BASE_URL%/}"

if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[unload] Invalid model name: ${MODEL_NAME}" >&2
    exit 2
fi
if [[ ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[unload] timeout_seconds must be a positive integer" >&2
    exit 2
fi

body_file=$(mktemp)
trap 'rm -f "${body_file}"' EXIT

echo "[unload] Requesting model unload: ${MODEL_NAME}"
if ! http_code=$(curl -sS \
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
while curl -fsS --connect-timeout 3 --max-time 5 \
    "${BASE_URL}/v2/models/${MODEL_NAME}/ready" > /dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        echo "[unload] Timed out waiting for ${MODEL_NAME} to unload" >&2
        exit 1
    fi
    sleep 1
done

echo "[unload] OK: ${MODEL_NAME} is not ready"
